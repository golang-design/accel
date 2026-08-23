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
	"go/printer"
	"go/token"
	"go/types"
	"sort"
	"strconv"
	"strings"

	"golang.design/x/accel/internal/kernelc/intrin"
	"golang.design/x/accel/internal/kernelc/ir"
	"golang.design/x/accel/internal/kernelc/std140"
	"golang.org/x/tools/go/packages"
)

// KernelDirective marks an entry function.
const KernelDirective = "//accel:kernel"

// RequiresDirective asserts which capabilities a kernel needs.
//
// An **assertion**, not a source. What a kernel requires is inferred from its
// body, because a declaration can be forgotten and the failure is silent: a
// kernel using a feature the device lacks would produce wrong results rather
// than an error. This directive is checked against the inferred set and a
// mismatch in either direction fails generation.
//
// Declaring more than the body needs is as much a bug as declaring less. It
// makes the kernel unavailable on devices that could run it, and nobody
// notices, because the symptom is a device being skipped. See
// specs/020-cooperative-atomics.md section 3.
const RequiresDirective = "//accel:requires"

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
	c := &checker{
		pkg: pkg, fset: pkg.Fset, info: pkg.TypesInfo,
		helpers: map[types.Object]*ir.Func{},
		calls:   map[*ir.Func][]*ir.Func{},
		layouts: map[string]*std140.Layout{},
	}

	var decls []declaration

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
			decls = append(decls, declaration{fn: fn, kind: kind, extent: extent})
		}
	}

	// Helpers are collected before any kernel is built, because a kernel may
	// call a helper declared after it. Go has no forward declarations and this
	// walk is source-ordered, so a single pass would reject a perfectly ordinary
	// file for the order its author chose.
	for _, d := range decls {
		if d.kind == HelperDirective {
			if h := c.helper(d.fn); h != nil {
				c.helpers[c.info.Defs[d.fn.Name]] = h
			}
		}
	}
	// Signatures before bodies, and both before any kernel. A helper that calls
	// another helper needs the callee's parameters to check the call against, and
	// building bodies in declaration order would leave the first helper checking
	// a call to a signature that does not exist yet. Three passes rather than
	// one is what makes the order a file is written in stop mattering.
	for _, d := range decls {
		if d.kind == HelperDirective {
			if h := c.helpers[c.info.Defs[d.fn.Name]]; h != nil {
				c.helperSignature(h, d.fn)
			}
		}
	}
	for _, d := range decls {
		if d.kind == HelperDirective {
			if h := c.helpers[c.info.Defs[d.fn.Name]]; h != nil {
				c.helperBody(h, d.fn)
			}
		}
	}
	for _, d := range decls {
		if d.kind == KernelDirective {
			if k := c.kernel(d.fn, d.extent); k != nil {
				c.checkRequires(k, d.fn)
				c.funcs = append(c.funcs, k)
			}
		}
	}
	c.checkRecursion()

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

// declaration is one directive-carrying function, collected before anything is
// built so that declaration order in the file does not decide what compiles.
type declaration struct {
	fn     *ast.FuncDecl
	kind   string
	extent [3]uint32
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

	// loops is the current loop nesting depth, so break and continue can be
	// rejected where they have nothing to bind to.
	loops int

	// helpers maps a declared helper to its IR, and calls records the call graph
	// the recursion check runs over.
	helpers map[types.Object]*ir.Func
	calls   map[*ir.Func][]*ir.Func

	// layouts is every uniform struct's std140 placement, by type name, so the
	// generator can emit one codec per type however many kernels take it.
	layouts map[string]*std140.Layout

	// order is every function in declaration order.
	//
	// The recursion check walks this rather than ranging over calls, because Go
	// randomizes map iteration and a cycle reached from a different starting
	// point is reported as a different cycle. CI found it: the same source said
	// "a -> b -> a" on one machine and "b -> a -> b" on another. A compiler
	// whose diagnostics depend on the run cannot have a golden test and cannot
	// have a reproducible bug report.
	order []*ir.Func
}

// normalize prints a declaration back from its AST.
//
// From the AST rather than from the file, because a package under test may not
// be on disk, and because normalizing is the behaviour worth having: a gofmt
// run must not force every kernel to be regenerated, while an edit that changes
// what the code means must.
func (c *checker) normalize(fn *ast.FuncDecl) string {
	var b strings.Builder
	cfg := printer.Config{Mode: printer.RawFormat, Tabwidth: 8}
	if err := cfg.Fprint(&b, c.fset, fn); err != nil {
		// Printing an AST that type-checked cannot fail; if it does, the digest
		// falls back to something that is at least stable per position rather
		// than silently becoming the empty string, which would make every kernel
		// in the package share a digest.
		return fmt.Sprintf("unprintable:%v", c.fset.Position(fn.Pos()))
	}
	return b.String()
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

// checkRequires compares an //accel:requires assertion against what the body
// implies.
//
// A mismatch fails in **either** direction. Declaring less than the body needs
// is the obvious bug; declaring more is equally one, and quieter: it makes the
// kernel unavailable on devices that could run it, and the symptom is a device
// being skipped rather than an error anybody sees.
func (c *checker) checkRequires(k *ir.Func, fn *ast.FuncDecl) {
	names, ok := requiresOf(fn)
	if !ok {
		return
	}
	var declared intrin.Capability
	for _, n := range names {
		cap, known := intrin.CapByName(n)
		if !known {
			c.errorf(fn.Pos(), "kernel %s: //accel:requires names %q, which is not a "+
				"capability; the set is %s", k.Name, n, strings.Join(intrin.CapNames(), ", "))
			return
		}
		declared |= cap
	}

	inferred := intrin.Capability(k.Caps)
	if declared == inferred {
		return
	}
	if missing := inferred &^ declared; missing != 0 {
		c.errorf(fn.Pos(), "kernel %s: its body requires %s, which //accel:requires does "+
			"not declare: a device without it would run this kernel and produce wrong "+
			"results rather than an error", k.Name, intrin.DescribeCaps(missing))
	}
	if extra := declared &^ inferred; extra != 0 {
		c.errorf(fn.Pos(), "kernel %s: //accel:requires declares %s, which its body does "+
			"not use: over-declaring makes the kernel unavailable on devices that could "+
			"run it, and the symptom is a device being skipped rather than an error",
			k.Name, intrin.DescribeCaps(extra))
	}
}

// requiresOf reads the //accel:requires assertion, if the declaration carries
// one, as a set of capability names.
func requiresOf(fn *ast.FuncDecl) ([]string, bool) {
	if fn.Doc == nil {
		return nil, false
	}
	for _, cm := range fn.Doc.List {
		text := strings.TrimSpace(cm.Text)
		if !strings.HasPrefix(text, RequiresDirective) {
			continue
		}
		rest := strings.TrimSpace(strings.TrimPrefix(text, RequiresDirective))
		var out []string
		for _, name := range strings.Split(rest, ",") {
			if n := strings.TrimSpace(name); n != "" {
				out = append(out, n)
			}
		}
		return out, true
	}
	return nil, false
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

	k := &ir.Func{Name: name, Stage: ir.StageCompute, Workgroup: extent, Thread: -1}
	k.P = fn.Pos()
	k.Source = c.normalize(fn)
	c.order = append(c.order, k)

	if !c.signature(fn, k) {
		return nil
	}

	c.current = k
	c.locals = map[types.Object]*ir.Local{}
	c.nextID = 0
	c.loops = 0
	body := c.block(fn.Body)
	c.current = nil
	if body == nil {
		return nil
	}
	k.Body = body
	inferAccess(k)

	// A resource nothing touches is one the caller has to supply for no reason,
	// and it is nearly always a typo in the body rather than a deliberate
	// signature.
	for _, b := range k.Bindings {
		if !b.Read && !b.Write {
			c.errorf(k.Params[b.Index].Pos(), "kernel %s: binding %q is never read or written",
				name, b.Name)
		}
	}
	for _, u := range k.Uniforms {
		if !u.Reads {
			c.errorf(k.Params[u.Index].Pos(), "kernel %s: uniform %q is never read", name, u.Name)
		}
	}
	return k
}

// helper declares a //accel:helper function, without building its body.
//
// Declaring first and building second is what lets a helper call another helper
// declared later in the file. Go has no forward declarations, and rejecting a
// file for the order its author chose would be a rule about this compiler
// rather than about the subset.
func (c *checker) helper(fn *ast.FuncDecl) *ir.Func {
	name := fn.Name.Name
	if fn.Recv != nil {
		c.errorf(fn.Pos(), "helper %s is a method: a helper is a package-level function", name)
		return nil
	}
	if fn.Type.TypeParams != nil && len(fn.Type.TypeParams.List) > 0 {
		c.errorf(fn.Type.TypeParams.Pos(), "helper %s is generic: generic helpers are out of "+
			"scope for v0 (specs/004-kernel-authoring.md)", name)
		return nil
	}
	if fn.Body == nil {
		c.errorf(fn.Pos(), "helper %s has no body", name)
		return nil
	}
	// One result or none. Multiple results are sequencing rather than a wall:
	// no target spells a tuple return, and the workaround is an out parameter,
	// which the subset has no pointer to express yet.
	if fn.Type.Results != nil && len(fn.Type.Results.List) > 1 {
		c.errorf(fn.Type.Results.Pos(), "helper %s returns %d values: multiple helper results "+
			"are out of scope for v0 (specs/004-kernel-authoring.md)", name, len(fn.Type.Results.List))
		return nil
	}

	h := &ir.Func{Name: name, Thread: -1}
	h.P = fn.Pos()
	h.Source = c.normalize(fn)
	c.order = append(c.order, h)

	if fn.Type.Results != nil && len(fn.Type.Results.List) == 1 {
		rt, err := c.irType(c.info.TypeOf(fn.Type.Results.List[0].Type))
		if err != nil {
			c.errorf(fn.Type.Results.Pos(), "helper %s returns %s", name, err)
			return nil
		}
		h.Result = rt
	}
	return h
}

// helperSignature fills in a declared helper's parameters.
func (c *checker) helperSignature(h *ir.Func, fn *ast.FuncDecl) {
	index := 0
	for _, field := range fn.Type.Params.List {
		if len(field.Names) == 0 {
			c.errorf(field.Pos(), "helper %s: every parameter needs a name", h.Name)
			return
		}
		for _, id := range field.Names {
			obj := c.info.Defs[id]
			if obj == nil {
				return
			}
			p, ok := c.helperParam(h, index, id, obj)
			if !ok {
				return
			}
			h.Params = append(h.Params, p)
			index++
		}
	}
	h.SignatureBuilt = true
}

// helperBody builds a declared helper's body, with its signature already known.
func (c *checker) helperBody(h *ir.Func, fn *ast.FuncDecl) {
	if !h.SignatureBuilt {
		return
	}
	c.current = h
	c.locals = map[types.Object]*ir.Local{}
	c.nextID = 0
	c.loops = 0
	defer func() { c.current = nil }()

	for _, p := range h.Params {
		_ = p // parameters resolve through h.Params in ident
	}

	body := c.block(fn.Body)
	if body == nil {
		return
	}
	h.Body = body
	inferAccess(h)
}

// helperParam classifies one helper parameter.
//
// A helper takes scalars and resource slices, and it may take the Thread. It
// takes a slice by the same rule a kernel does, and the access it makes is a
// property of the call site rather than of the helper, which is why the
// inferred mode is recorded per binding and merged into the caller's.
func (c *checker) helperParam(h *ir.Func, index int, id *ast.Ident, obj types.Object) (*ir.Param, bool) {
	t := obj.Type()

	if isThread(t) {
		if h.Thread >= 0 {
			c.errorf(id.Pos(), "helper %s takes accel.Thread twice", h.Name)
			return nil, false
		}
		h.Thread = index
		return ir.NewParam(id.Pos(), &ir.Type{Kind: ir.Struct, Name: "Thread"}, index, id.Name, obj), true
	}

	it, err := c.irType(t)
	if err != nil {
		c.errorf(id.Pos(), "helper %s: parameter %q is %s", h.Name, id.Name, err)
		return nil, false
	}
	if it.Kind == ir.Slice {
		h.Bindings = append(h.Bindings, &ir.Binding{Name: id.Name, Index: index, Type: it})
	}
	return ir.NewParam(id.Pos(), it, index, id.Name, obj), true
}

// checkRecursion rejects a cycle in the call graph.
//
// No target can express recursion: there is no call stack to grow, so this is a
// wall rather than a schedule. It is checked over the IR's call graph rather
// than the AST, because a helper reached only through another helper is still
// in the cycle and an AST walk per function would not see it.
func (c *checker) checkRecursion() {
	const (
		unvisited = 0
		onStack   = 1
		done      = 2
	)
	state := map[*ir.Func]int{}
	var path []*ir.Func

	var walk func(f *ir.Func) bool
	walk = func(f *ir.Func) bool {
		switch state[f] {
		case done:
			return false
		case onStack:
			// Name the cycle rather than the function: "h calls itself" and
			// "a calls b calls a" are different bugs and the second is the one a
			// reader cannot see from one declaration.
			names := []string{f.Name}
			for i := len(path) - 1; i >= 0; i-- {
				names = append(names, path[i].Name)
				if path[i] == f {
					break
				}
			}
			reverse(names)
			c.errorf(f.Pos(), "%s is recursive (%s): no target has a call stack to grow, so "+
				"recursion is not expressible rather than merely unscheduled",
				f.Name, strings.Join(names, " -> "))
			return true
		}
		state[f] = onStack
		path = append(path, f)
		for _, callee := range c.calls[f] {
			if walk(callee) {
				state[f] = done
				path = path[:len(path)-1]
				return true
			}
		}
		path = path[:len(path)-1]
		state[f] = done
		return false
	}

	for _, f := range c.order {
		walk(f)
	}
}

func reverse[T any](s []T) {
	for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
		s[i], s[j] = s[j], s[i]
	}
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
		c.errorf(fn.Type.Params.Pos(), "kernel %s has no slice bindings: a uniform is read-only, "+
			"so a kernel with no slice has nowhere to put a result", k.Name)
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
		if arr, isArray := u.Elem().Underlying().(*types.Array); isArray {
			return c.sharedParam(k, index, id, obj, arr)
		}
		c.errorf(id.Pos(), "kernel %s: parameter %q is a pointer, and the only pointer the subset "+
			"admits is a pointer to a fixed-size array, as workgroup-shared memory", k.Name, id.Name)
		return nil, false

	case *types.Struct:
		return c.uniformParam(k, index, id, obj)
	}

	c.errorf(id.Pos(), "kernel %s: parameter %q has type %s, which is not a resource: a binding is "+
		"a slice, and the first parameter is accel.Thread", k.Name, id.Name, t)
	return nil, false
}

// sharedParam places a pointer-to-array parameter as workgroup-shared memory.
//
// A pointer to a fixed-size array rather than a slice, because the size is
// fixed at pipeline creation on every backend: it appears in the GLSL layout
// qualifier and in Metal's threadgroup attribute, so it cannot be a runtime
// length. Go's array type is where that size already lives.
//
// A pointer rather than a value because a workgroup shares one copy: passing
// the array by value would give every invocation its own, which compiles and
// computes something else.
func (c *checker) sharedParam(k *ir.Func, index int, id *ast.Ident, obj types.Object, arr *types.Array) (*ir.Param, bool) {
	elem, err := elementKind(arr.Elem())
	if err != nil {
		c.errorf(id.Pos(), "kernel %s: shared memory %q is *[%d]%s, and %s",
			k.Name, id.Name, arr.Len(), arr.Elem(), err)
		return nil, false
	}
	n := int(arr.Len())
	if n <= 0 {
		c.errorf(id.Pos(), "kernel %s: shared memory %q has length %d", k.Name, id.Name, n)
		return nil, false
	}
	// Shared memory makes the kernel cooperative even without a barrier: it is
	// storage a workgroup shares, so the invocations have to run together.
	k.Cooperative = true

	it := &ir.Type{Kind: ir.Array, Elem: &ir.Type{Kind: elem}, Len: n}
	k.Shared = append(k.Shared, &ir.SharedMem{Name: id.Name, Index: index, Type: it})
	return ir.NewParam(id.Pos(), it, index, id.Name, obj), true
}

// uniformParam places a by-value struct parameter.
//
// By value is what "uniform" means in Go: immutable for the dispatch. The
// layout is computed here rather than at emission because the block's size is
// checked against the device's limit and a rejected layout has to name the
// field, which needs the type.
func (c *checker) uniformParam(k *ir.Func, index int, id *ast.Ident, obj types.Object) (*ir.Param, bool) {
	name := typeName(obj.Type())
	if name == "" {
		c.errorf(id.Pos(), "kernel %s: parameter %q is an unnamed struct: a uniform's codec is "+
			"generated for a named type, because a caller has to be able to write the value down",
			k.Name, id.Name)
		return nil, false
	}

	layout, err := std140.Of(name, obj.Type())
	if err != nil {
		c.errorf(id.Pos(), "kernel %s: %s", k.Name, err)
		return nil, false
	}

	k.Uniforms = append(k.Uniforms, &ir.Uniform{
		Name: id.Name, Index: index, TypeName: name, Size: layout.Size,
		Fields: uniformFields(layout),
	})
	c.layouts[name] = layout

	return ir.NewParam(id.Pos(), &ir.Type{Kind: ir.Struct, Name: name}, index, id.Name, obj), true
}

// uniformFields flattens a layout into what the emitter needs.
func uniformFields(l *std140.Layout) []ir.UniformField {
	out := make([]ir.UniformField, 0, len(l.Fields))
	for _, f := range l.Fields {
		uf := ir.UniformField{
			Name: f.Name, Offset: f.Offset, Scalar: f.Scalar.String(), Len: f.Len,
		}
		switch f.Kind {
		case std140.KScalar:
			uf.Kind = "scalar"
		case std140.KVector:
			uf.Kind = "vector"
		case std140.KArray:
			uf.Kind, uf.Stride = "array", 16
		case std140.KMatrix:
			uf.Kind, uf.Stride = "matrix", 16
		case std140.KStruct:
			uf.Kind = "struct"
		}
		out = append(out, uf)
	}
	return out
}

// typeName is a named type's name, or empty for an unnamed one.
func typeName(t types.Type) string {
	n, ok := types.Unalias(t).(*types.Named)
	if !ok || n.Obj() == nil {
		return ""
	}
	return n.Obj().Name()
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
	// The narrow storage types are named structs rather than basics, which is
	// what stops arithmetic on them from compiling. They are still scalars as
	// far as a binding is concerned.
	if isKernelType(t, "Float16") {
		return ir.F16, nil
	}
	if isKernelType(t, "BFloat16") {
		return ir.BF16, nil
	}

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
		"int8, uint8, accel.Float16, or accel.BFloat16", b)
}
