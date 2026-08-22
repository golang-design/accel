// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package front turns type-checked Go into the typed IR, or into positioned
// rejections.
//
// # Why go/types and not an AST walk
//
// Over a bare AST walk, type checking buys four things the predecessor got
// wrong without it. Intrinsics resolve by object identity, so a user function
// named Dot no longer lowers to the GPU builtin. float32(x) and Sqrt(x) are the
// same AST shape and go/types tells them apart, where the predecessor put
// float32 in its builtins map next to sqrt. Untyped constants arrive with their
// resolved type, which dissolves the GLSL integer-literal divergence at the
// root. And scopes, shadowing, and method sets come for free, where the
// predecessor reimplemented identifier resolution as a bare-identifier check.
//
// What it does not buy is a guarantee that the program is a legal GPU program.
// That is this package's job, and it is a separate analysis.
//
// # Every rejection is ours
//
// A front end whose coverage depends on what the parser happens to refuse has
// an unstated dependency on the toolchain's release. Go 1.27 is the live
// example: through 1.26 the parser rejected generic methods and discarded their
// type parameters, and now it keeps them, so a walk that inherited that refusal
// would silently lower a generic method's body as though the type parameters
// were not there. Every construct outside the closed node set is rejected here,
// with a position. See specs/004-kernel-authoring.md.
package front

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"sort"
	"strconv"
	"strings"

	"golang.design/x/accel/internal/kernelc/ir"
	"golang.org/x/tools/go/packages"
)

// KernelDirective marks an entry function.
const KernelDirective = "//accel:kernel"

// HelperDirective marks a function callable from a kernel. Helpers are spec
// 013's; the directive is recognized here so that using one reports that it
// arrives later rather than that it is unknown.
const HelperDirective = "//accel:helper"

// Diagnostic is one rejection, positioned.
//
// Positioned is not decoration. A diagnostic that names the right problem at
// the wrong line sends a reader to the wrong place, which for a compiler is
// most of the cost, so the corpus asserts the position as well as the message.
type Diagnostic struct {
	Pos token.Position
	Msg string
}

func (d Diagnostic) Error() string { return d.Pos.String() + ": " + d.Msg }

// Diagnostics is a sorted set of rejections.
type Diagnostics []Diagnostic

func (ds Diagnostics) Error() string {
	msgs := make([]string, len(ds))
	for i, d := range ds {
		msgs[i] = d.Error()
	}
	return strings.Join(msgs, "\n")
}

// LoadMode is what the front end needs from go/packages.
const LoadMode = packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
	packages.NeedSyntax | packages.NeedTypes | packages.NeedTypesInfo |
	packages.NeedDeps | packages.NeedImports

// Load type-checks packages for [Check].
//
// go/packages is a build-tool dependency and nothing at runtime imports it: type
// checking needs the package loaded with its import graph and the go tool
// present, which a deployed binary does not have. That is the whole reason
// compilation is a generator rather than something that happens at startup.
func Load(dir string, patterns ...string) ([]*packages.Package, error) {
	cfg := &packages.Config{Mode: LoadMode, Dir: dir}
	pkgs, err := packages.Load(cfg, patterns...)
	if err != nil {
		return nil, fmt.Errorf("accel: loading %v: %w", patterns, err)
	}
	var bad []string
	for _, p := range pkgs {
		for _, e := range p.Errors {
			bad = append(bad, e.Error())
		}
	}
	if len(bad) > 0 {
		return nil, fmt.Errorf("accel: %s did not type-check:\n%s", strings.Join(patterns, " "), strings.Join(bad, "\n"))
	}
	return pkgs, nil
}

// Check finds every kernel in a package and builds its IR.
//
// It returns the kernels it could build and every rejection it found. Both, not
// one or the other: reporting only the first rejection makes fixing a kernel a
// sequence of round trips, and reporting none of the kernels when one is wrong
// makes an unrelated typo hide a working corpus.
func Check(pkg *packages.Package) ([]*ir.Func, Diagnostics) {
	c := &checker{pkg: pkg, fset: pkg.Fset, info: pkg.TypesInfo}

	for _, file := range pkg.Syntax {
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok {
				continue
			}
			kind, extent, ok := directiveOf(fn)
			if !ok {
				continue
			}
			if kind == HelperDirective {
				c.errorf(fn.Pos(), "%s is out of scope for now: helper functions arrive with "+
					"spec 013 (specs/013-kernel-subset.md)", HelperDirective)
				continue
			}
			if k := c.kernel(fn, extent); k != nil {
				c.funcs = append(c.funcs, k)
			}
		}
	}

	sort.Slice(c.diags, func(i, j int) bool {
		if c.diags[i].Pos.Filename != c.diags[j].Pos.Filename {
			return c.diags[i].Pos.Filename < c.diags[j].Pos.Filename
		}
		if c.diags[i].Pos.Line != c.diags[j].Pos.Line {
			return c.diags[i].Pos.Line < c.diags[j].Pos.Line
		}
		return c.diags[i].Pos.Column < c.diags[j].Pos.Column
	})
	return c.funcs, c.diags
}

// checker holds one package's state.
type checker struct {
	pkg  *packages.Package
	fset *token.FileSet
	info *types.Info

	funcs []*ir.Func
	diags Diagnostics

	// current is the kernel being built, for access inference and intrinsic
	// recording.
	current *ir.Func
	locals  map[types.Object]*ir.Local
	nextID  int
}

func (c *checker) errorf(p token.Pos, format string, args ...any) {
	c.diags = append(c.diags, Diagnostic{Pos: c.fset.Position(p), Msg: fmt.Sprintf(format, args...)})
}

// directiveOf reports which accel directive a declaration carries.
func directiveOf(fn *ast.FuncDecl) (kind string, extent [3]uint32, ok bool) {
	if fn.Doc == nil {
		return "", extent, false
	}
	for _, cm := range fn.Doc.List {
		text := strings.TrimSpace(cm.Text)
		switch {
		case strings.HasPrefix(text, KernelDirective):
			e, err := parseWorkgroup(strings.TrimPrefix(text, KernelDirective))
			if err != nil {
				// Reported by the caller against the declaration, since the
				// directive's own position is inside a comment nobody edits by
				// column.
				return KernelDirective, [3]uint32{}, true
			}
			return KernelDirective, e, true
		case strings.HasPrefix(text, HelperDirective):
			return HelperDirective, extent, true
		}
	}
	return "", extent, false
}

// parseWorkgroup reads `workgroup=X[,Y[,Z]]`, defaulting the omitted axes to 1.
func parseWorkgroup(rest string) ([3]uint32, error) {
	out := [3]uint32{0, 1, 1}
	rest = strings.TrimSpace(rest)
	if !strings.HasPrefix(rest, "workgroup=") {
		return out, fmt.Errorf("expected workgroup=")
	}
	parts := strings.Split(strings.TrimPrefix(rest, "workgroup="), ",")
	if len(parts) > 3 {
		return out, fmt.Errorf("at most three extents")
	}
	for i, p := range parts {
		n, err := strconv.ParseUint(strings.TrimSpace(p), 10, 32)
		if err != nil || n == 0 {
			return out, fmt.Errorf("extent %q is not a positive number", p)
		}
		out[i] = uint32(n)
	}
	return out, nil
}

// kernel validates one entry function and builds its IR, or reports why not.
func (c *checker) kernel(fn *ast.FuncDecl, extent [3]uint32) *ir.Func {
	name := fn.Name.Name

	if extent == [3]uint32{} {
		c.errorf(fn.Pos(), "kernel %s: %s needs a workgroup extent, spelled "+
			"workgroup=64 or workgroup=16,8", name, KernelDirective)
		return nil
	}
	if fn.Recv != nil {
		c.errorf(fn.Pos(), "kernel %s is a method: a kernel is a package-level function", name)
		return nil
	}
	// Go 1.27 accepts type parameters on methods and the parser no longer
	// discards them, so this rejection is ours to make rather than one to
	// inherit. Generic kernels are out of scope for v0, not unrepresentable.
	if fn.Type.TypeParams != nil && len(fn.Type.TypeParams.List) > 0 {
		c.errorf(fn.Type.TypeParams.Pos(), "kernel %s is generic: generic kernels are out of "+
			"scope for v0 (specs/004-kernel-authoring.md)", name)
		return nil
	}
	if fn.Type.Results != nil && len(fn.Type.Results.List) > 0 {
		c.errorf(fn.Type.Results.Pos(), "kernel %s returns a value: a kernel writes through its "+
			"bindings and returns nothing", name)
		return nil
	}
	if fn.Body == nil {
		c.errorf(fn.Pos(), "kernel %s has no body", name)
		return nil
	}

	k := &ir.Func{Name: name, Kernel: true, Workgroup: extent, Thread: -1}
	k.P = fn.Pos()

	if !c.signature(fn, k) {
		return nil
	}

	c.current = k
	c.locals = map[types.Object]*ir.Local{}
	c.nextID = 0
	body := c.block(fn.Body)
	c.current = nil
	if body == nil {
		return nil
	}
	k.Body = body
	inferAccess(k)

	// A binding nothing touches is a binding the caller has to supply for no
	// reason, and it is nearly always a typo in the body rather than a
	// deliberate signature.
	for _, b := range k.Bindings {
		if !b.Read && !b.Write {
			c.errorf(k.Params[b.Index].Pos(), "kernel %s: binding %q is never read or written",
				name, b.Name)
		}
	}
	return k
}

// signature maps parameters onto bindings. The signature is the binding layout,
// so everything a caller must supply is decided here and nothing is declared
// twice.
func (c *checker) signature(fn *ast.FuncDecl, k *ir.Func) bool {
	ok := true
	index := 0
	for _, field := range fn.Type.Params.List {
		names := field.Names
		if len(names) == 0 {
			c.errorf(field.Pos(), "kernel %s: every parameter needs a name, because the name "+
				"is the binding's name", k.Name)
			return false
		}
		for _, id := range names {
			obj := c.info.Defs[id]
			if obj == nil {
				return false
			}
			p, good := c.param(k, index, id, obj)
			if !good {
				ok = false
			}
			if p != nil {
				k.Params = append(k.Params, p)
			}
			index++
		}
	}

	if k.Thread < 0 {
		c.errorf(fn.Type.Params.Pos(), "kernel %s: the first parameter must be accel.Thread, "+
			"which carries the invocation's ids", k.Name)
		ok = false
	} else if k.Thread != 0 {
		c.errorf(k.Params[k.Thread].Pos(), "kernel %s: accel.Thread must be the first parameter",
			k.Name)
		ok = false
	}
	if len(k.Bindings) == 0 && ok {
		c.errorf(fn.Type.Params.Pos(), "kernel %s has no bindings: a kernel with nothing to read "+
			"or write cannot observe anything", k.Name)
		ok = false
	}
	return ok
}

// param classifies one parameter.
func (c *checker) param(k *ir.Func, index int, id *ast.Ident, obj types.Object) (*ir.Param, bool) {
	t := obj.Type()

	if isThread(t) {
		if k.Thread >= 0 {
			c.errorf(id.Pos(), "kernel %s takes accel.Thread twice", k.Name)
			return nil, false
		}
		k.Thread = index
		return ir.NewParam(id.Pos(), &ir.Type{Kind: ir.Struct, Name: "Thread"}, index, id.Name, obj), true
	}

	if id.Name == "_" {
		c.errorf(id.Pos(), "kernel %s: a binding cannot be named _, because the name is how "+
			"an error and a caller's binding list refer to it", k.Name)
		return nil, false
	}

	switch u := types.Unalias(t).Underlying().(type) {
	case *types.Slice:
		elem, err := elementKind(u.Elem())
		if err != nil {
			c.errorf(id.Pos(), "kernel %s: binding %q is []%s, and %s", k.Name, id.Name, u.Elem(), err)
			return nil, false
		}
		it := &ir.Type{Kind: ir.Slice, Elem: &ir.Type{Kind: elem}}
		k.Bindings = append(k.Bindings, &ir.Binding{Name: id.Name, Index: index, Type: it})
		return ir.NewParam(id.Pos(), it, index, id.Name, obj), true

	case *types.Pointer:
		if _, isArray := u.Elem().Underlying().(*types.Array); isArray {
			c.errorf(id.Pos(), "kernel %s: parameter %q is workgroup-shared memory, which is out "+
				"of scope for now: cooperative kernels arrive at M4 "+
				"(specs/009-sequencing.md)", k.Name, id.Name)
			return nil, false
		}
		c.errorf(id.Pos(), "kernel %s: parameter %q is a pointer, and the only pointer the subset "+
			"admits is a pointer to a fixed-size array, as workgroup-shared memory", k.Name, id.Name)
		return nil, false

	case *types.Struct:
		c.errorf(id.Pos(), "kernel %s: parameter %q is a by-value struct, which is a uniform and "+
			"is out of scope for now: uniform structs arrive with spec 014 "+
			"(specs/014-kernel-uniforms.md)", k.Name, id.Name)
		return nil, false
	}

	c.errorf(id.Pos(), "kernel %s: parameter %q has type %s, which is not a resource: a binding is "+
		"a slice, and the first parameter is accel.Thread", k.Name, id.Name, t)
	return nil, false
}

// isThread reports whether a type is accel.Thread, by identity rather than by
// name.
func isThread(t types.Type) bool { return isKernelType(t, "Thread") }

// isID3 reports whether a type is accel.ID3.
func isID3(t types.Type) bool { return isKernelType(t, "ID3") }

// isKernelType resolves an authored name to the type it aliases.
//
// The Unalias is load-bearing and was not in the first draft. accel.Thread is an
// alias of internal/kernel.Thread, and Go 1.27 removed the gotypesalias GODEBUG
// so go/types now *always* produces Alias nodes. Asserting straight to
// *types.Named therefore fails, and a parameter written accel.Thread falls
// through to the struct case and is rejected as a uniform. That is the shape of
// bug this whole design is built to avoid, caught here only because the corpus
// kernel is compiled from its real source rather than from a restatement of it.
func isKernelType(t types.Type, name string) bool {
	n, ok := types.Unalias(t).(*types.Named)
	if !ok || n.Obj() == nil || n.Obj().Pkg() == nil {
		return false
	}
	return n.Obj().Pkg().Path() == kernelPkgPath && n.Obj().Name() == name
}

// kernelPkgPath is where the aliased authored types resolve to.
const kernelPkgPath = "golang.design/x/accel/internal/kernel"

// scalarKind maps a Go scalar onto an IR kind, for a value.
func scalarKind(t types.Type) (ir.Kind, error) {
	if b, ok := types.Unalias(t).Underlying().(*types.Basic); ok && b.Kind() == types.Bool {
		return ir.Bool, nil
	}
	return elementKind(t)
}

// elementKind maps a Go scalar onto an IR kind, for a binding's element.
//
// It is the narrower of the two because **bool is not a storage type**. A
// condition is a bool and a buffer of them is not: Go's bool is one byte, every
// target's is four or is a lane mask, and spec 004's parameter table admits
// float32, int32, uint32, int8, uint8, and the two narrow floats and nothing
// else. Letting a bool binding through would produce a buffer whose element
// width differs between the CPU oracle and every GPU, which is a wrong-answer
// bug rather than a compile error.
func elementKind(t types.Type) (ir.Kind, error) {
	b, ok := types.Unalias(t).Underlying().(*types.Basic)
	if !ok {
		return ir.Invalid, fmt.Errorf("its element is not a scalar the subset admits")
	}
	switch b.Kind() {
	case types.Float32:
		return ir.F32, nil
	case types.Int32:
		return ir.I32, nil
	case types.Uint32:
		return ir.U32, nil
	case types.Int8:
		return ir.I8, nil
	case types.Uint8:
		return ir.U8, nil
	}
	return ir.Invalid, fmt.Errorf("its element type %s is not one of float32, int32, uint32, "+
		"int8, or uint8", b)
}
