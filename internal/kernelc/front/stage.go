// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package front

import (
	"go/ast"
	"go/token"
	"go/types"
	"strings"

	"golang.design/x/accel/internal/kernelc/ir"
	"golang.design/x/accel/internal/mslabi"
)

// The graphics stage signatures of specs/032-stage-abi.md.
//
// A stage is the same authored Go as a compute kernel — the same subset, the
// same go/types resolution, the same IR. What is new is the *shape* of the
// signature, which is fixed-function hardware and therefore not something a Go
// signature describes by accident. That is why this file is signature checking
// and nothing else: the body goes through the same builder a kernel's does.

// isVertexReceiver and isFragmentReceiver resolve the two stage receivers.
func isVertexReceiver(t types.Type) bool   { return isKernelType(t, "Vertex") }
func isFragmentReceiver(t types.Type) bool { return isKernelType(t, "Fragment") }

// stage validates one graphics stage and builds its IR, or reports why not.
func (c *checker) stage(fn *ast.FuncDecl, directive string) *ir.Func {
	name := fn.Name.Name
	vertex := directive == VertexDirective
	what := "fragment stage"
	if vertex {
		what = "vertex stage"
	}

	if fn.Recv != nil {
		c.errorf(fn.Pos(), "%s %s is a method: a stage is a package-level function", what, name)
		return nil
	}
	if fn.Type.TypeParams != nil && len(fn.Type.TypeParams.List) > 0 {
		c.errorf(fn.Type.TypeParams.Pos(), "%s %s is generic: generic stages are out of scope "+
			"for v0", what, name)
		return nil
	}
	if fn.Body == nil {
		c.errorf(fn.Pos(), "%s %s has no body", what, name)
		return nil
	}

	k := &ir.Func{Name: name, Thread: -1}
	if vertex {
		k.Stage = ir.StageVertex
	} else {
		k.Stage = ir.StageFragment
	}
	k.P = fn.Pos()
	k.Source = c.normalize(fn)
	c.order = append(c.order, k)

	if !c.stageSignature(fn, k, what) {
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

	// A graphics stage has no workgroup, so it cannot cooperate: a barrier,
	// shared memory or a subgroup operation in one is a compiler error rather
	// than a convention. The body builder already recorded what it reached.
	if k.Cooperative || len(k.Shared) > 0 {
		c.errorf(fn.Pos(), "%s %s reaches a barrier, shared memory or a subgroup "+
			"operation: those need a workgroup, and a graphics stage has none",
			what, k.Name)
		return nil
	}
	return k
}

// stageSignature checks the parameters and results, and records what the stage
// declared.
//
// Position is load-bearing here in a way it is not for a compute kernel. After
// specs/032-stage-abi.md section 2.3 made the vector types aliases for Go
// arrays, a fragment stage's varyings struct and its uniform structs are both
// *types.Struct — indistinguishable by type. So parameter 1 of a fragment stage
// is the varyings struct *by position*, whatever its type, and only parameters
// 2 and beyond are classified.
func (c *checker) stageSignature(fn *ast.FuncDecl, k *ir.Func, what string) bool {
	params := flatParams(fn)
	if len(params) == 0 {
		c.errorf(fn.Pos(), "%s %s takes no parameters: the first is accel.%s",
			what, k.Name, receiverName(k.Stage))
		return false
	}

	// Parameter 0: the receiver.
	first := c.info.TypeOf(params[0].typ)
	okRecv := isVertexReceiver(first)
	if k.Stage == ir.StageFragment {
		okRecv = isFragmentReceiver(first)
	}
	if !okRecv {
		c.errorf(params[0].id.Pos(), "%s %s takes %s first, and the first parameter of a "+
			"%s is accel.%s", what, k.Name, first, what, receiverName(k.Stage))
		return false
	}

	// The receiver is recorded the way a compute kernel records accel.Thread:
	// as parameter zero, so the body can resolve v.VertexIndex() against it.
	k.Thread = 0
	k.Params = append(k.Params, ir.NewParam(params[0].id.Pos(),
		&ir.Type{Kind: ir.Struct, Name: receiverName(k.Stage)}, 0, params[0].id.Name,
		c.info.Defs[params[0].id]))

	next := 1
	if k.Stage == ir.StageFragment {
		// Parameter 1: the varyings struct, by position.
		if len(params) < 2 {
			c.errorf(fn.Pos(), "fragment stage %s has no varyings parameter: it is the "+
				"second parameter, and it is identified by position because a varyings "+
				"struct and a uniform struct are both structs", k.Name)
			return false
		}
		vt := c.info.TypeOf(params[1].typ)
		if _, ok := types.Unalias(vt).Underlying().(*types.Struct); !ok {
			c.errorf(params[1].id.Pos(), "fragment stage %s takes %s as its varyings "+
				"parameter, and varyings are a struct", k.Name, vt)
			return false
		}
		k.Varyings = c.namedStructType(vt)
		c.checkVaryingSlots(params[1].id.Pos(), k.Name, k.Varyings)
		k.Params = append(k.Params, ir.NewParam(params[1].id.Pos(), k.Varyings, 1,
			params[1].id.Name, c.info.Defs[params[1].id]))
		next = 2
	}

	// The remaining parameters classify by type, as a compute kernel's do, plus
	// one row: a by-value array is a vertex attribute.
	for i := next; i < len(params); i++ {
		p := params[i]
		t := types.Unalias(c.info.TypeOf(p.typ))
		if arr, isArray := t.Underlying().(*types.Array); isArray {
			if k.Stage != ir.StageVertex {
				c.errorf(p.id.Pos(), "fragment stage %s takes %s by value, and a by-value "+
					"array is a vertex attribute; a fragment stage reads interpolated "+
					"varyings instead", k.Name, t)
				return false
			}
			elem, err := elementKind(arr.Elem())
			if err != nil {
				c.errorf(p.id.Pos(), "vertex stage %s: attribute %q is [%d]%s, and %s",
					k.Name, p.id.Name, arr.Len(), arr.Elem(), err)
				return false
			}
			at := &ir.Type{Kind: ir.Array, Len: int(arr.Len()), Elem: &ir.Type{Kind: elem}}
			k.Attributes = append(k.Attributes, &ir.Attribute{
				Name: p.id.Name, Index: len(k.Attributes), Type: at,
			})
			// Also a parameter, so the body resolves it. An attribute is the
			// one thing a stage reads that is neither a binding nor a local.
			k.Params = append(k.Params, ir.NewParam(p.id.Pos(), at, i, p.id.Name,
				c.info.Defs[p.id]))
			continue
		}
		prm, ok := c.param(k, i, p.id, c.info.Defs[p.id])
		if !ok {
			return false
		}
		k.Params = append(k.Params, prm)
	}

	return c.stageResults(fn, k, what)
}

// stageResults checks what a stage returns.
func (c *checker) stageResults(fn *ast.FuncDecl, k *ir.Func, what string) bool {
	res := fn.Type.Results
	if res == nil || len(res.List) == 0 {
		c.errorf(fn.Pos(), "%s %s returns nothing: %s", what, k.Name, returnShape(k.Stage))
		return false
	}
	flat := flatResults(res)

	if k.Stage == ir.StageVertex {
		if len(flat) != 2 {
			c.errorf(res.Pos(), "vertex stage %s returns %d values: %s",
				k.Name, len(flat), returnShape(k.Stage))
			return false
		}
		// The position is not a varying: it is consumed by primitive assembly
		// and never reaches the fragment stage as itself, which is why it is a
		// separate result rather than a field a caller could reorder or drop.
		if pt := types.Unalias(c.info.TypeOf(flat[0])); !isVec4(pt) {
			c.errorf(flat[0].Pos(), "vertex stage %s returns %s as its position, and a clip "+
				"position is accel.Clip", k.Name, pt)
			return false
		}
		vt := c.info.TypeOf(flat[1])
		if _, ok := types.Unalias(vt).Underlying().(*types.Struct); !ok {
			c.errorf(flat[1].Pos(), "vertex stage %s returns %s as its varyings, and "+
				"varyings are a struct", k.Name, vt)
			return false
		}
		k.Varyings = c.namedStructType(vt)
		c.checkVaryingSlots(flat[1].Pos(), k.Name, k.Varyings)
		return true
	}

	// A fragment stage returns one struct, whose fields map in declaration
	// order onto the pipeline's colour attachments.
	if len(flat) != 1 {
		c.errorf(res.Pos(), "fragment stage %s returns %d values: %s",
			k.Name, len(flat), returnShape(k.Stage))
		return false
	}
	rt := types.Unalias(c.info.TypeOf(flat[0]))
	st, ok := rt.Underlying().(*types.Struct)
	if !ok {
		c.errorf(flat[0].Pos(), "fragment stage %s returns %s, and a stage returns a struct "+
			"whose fields are its colour attachments — one field per attachment, even "+
			"when there is one", k.Name, rt)
		return false
	}
	if st.NumFields() == 0 {
		c.errorf(flat[0].Pos(), "fragment stage %s returns a struct with no fields, so it "+
			"writes no attachment", k.Name)
		return false
	}
	for i := range st.NumFields() {
		f := st.Field(i)
		ft := types.Unalias(f.Type())
		if !isVec4(ft) {
			c.errorf(flat[0].Pos(), "fragment stage %s: attachment %d (%s) is %s, and a "+
				"colour attachment is accel.Vec4", k.Name, i, f.Name(), ft)
			return false
		}
		k.Outputs = append(k.Outputs, &ir.Target{
			Name: f.Name(), Index: i,
			Type: &ir.Type{Kind: ir.Array, Len: 4, Elem: &ir.Type{Kind: ir.F32}},
		})
	}
	return true
}

// isVec4 reports whether a type is the four-component float vector accel.Vec4
// and accel.Clip both alias.
func isVec4(t types.Type) bool {
	arr, ok := types.Unalias(t).Underlying().(*types.Array)
	if !ok || arr.Len() != 4 {
		return false
	}
	b, ok := arr.Elem().Underlying().(*types.Basic)
	return ok && b.Kind() == types.Float32
}

// varyingSlots is how many four-component interpolation slots a varyings struct
// occupies, specs/032-stage-abi.md section 3.2's formula.
//
// Not varyingFloats. That counts the floats the flat form carries, which is 4
// for a Vec4; this counts slots, which is 1. Reusing the other would refuse
// every stage with four float vectors in it.
func varyingSlots(t *ir.Type) int {
	n := 0
	for _, f := range t.Fields {
		c := 1
		if f.Type != nil && f.Type.Kind == ir.Array {
			c = f.Type.Len
		}
		n += (c + 3) / 4
	}
	return n
}

// checkVaryingSlots refuses a varyings struct that exceeds the interpolation
// budget, at generation and with the source position, which section 9 requires
// of every stage error.
//
// The limit is the target profile's rather than the device's. Section 3.2 says
// both, and section 9 and decision 6 settle it: an error that waits for
// pipeline creation arrives without the position that makes it actionable, and
// the generator has no device to ask.
func (c *checker) checkVaryingSlots(pos token.Pos, name string, t *ir.Type) {
	n := varyingSlots(t)
	if n <= mslabi.StageVaryingSlotLimit {
		return
	}
	fields := make([]string, 0, len(t.Fields))
	for _, f := range t.Fields {
		fields = append(fields, f.Name)
	}
	c.errorf(pos, "stage %s varyings %s occupy %d interpolation slots and the limit is %d: "+
		"the count is the sum of ceil(components/4) over the fields %s, and it does not pack "+
		"(specs/032-stage-abi.md section 3.2)",
		name, t.Name, n, mslabi.StageVaryingSlotLimit, strings.Join(fields, ", "))
}

// namedStructType records a varyings struct in the IR, keeping the name so a
// mismatch between the two stages can be reported by name.
func (c *checker) namedStructType(t types.Type) *ir.Type {
	name := t.String()
	if n, ok := types.Unalias(t).(*types.Named); ok {
		name = n.Obj().Name()
	}
	out := &ir.Type{Kind: ir.Struct, Name: name}
	st, ok := types.Unalias(t).Underlying().(*types.Struct)
	if !ok {
		return out
	}
	for i := range st.NumFields() {
		f := st.Field(i)
		k, err := elementKind(f.Type())
		ft := &ir.Type{Kind: ir.F32}
		if err == nil {
			ft = &ir.Type{Kind: k}
		} else if arr, isArr := types.Unalias(f.Type()).Underlying().(*types.Array); isArr {
			if ek, err := elementKind(arr.Elem()); err == nil {
				ft = &ir.Type{Kind: ir.Array, Len: int(arr.Len()), Elem: &ir.Type{Kind: ek}}
			}
		}
		out.Fields = append(out.Fields, ir.Field{Name: f.Name(), Type: ft})
	}
	return out
}

func receiverName(s ir.Stage) string {
	if s == ir.StageVertex {
		return "Vertex"
	}
	return "Fragment"
}

func returnShape(s ir.Stage) string {
	if s == ir.StageVertex {
		return "a vertex stage returns a clip position and a varyings struct"
	}
	return "a fragment stage returns one struct whose fields are its colour attachments"
}

// param is a named field or result, with the identifier a diagnostic points at.
type namedParam struct {
	id  *ast.Ident
	typ ast.Expr
}

// flatParams expands a parameter list so that `a, b int` is two entries.
func flatParams(fn *ast.FuncDecl) []namedParam {
	var out []namedParam
	if fn.Type.Params == nil {
		return out
	}
	for _, f := range fn.Type.Params.List {
		for _, n := range f.Names {
			out = append(out, namedParam{id: n, typ: f.Type})
		}
	}
	return out
}

// flatResults expands a result list into one expression per value.
func flatResults(res *ast.FieldList) []ast.Expr {
	var out []ast.Expr
	for _, f := range res.List {
		n := len(f.Names)
		if n == 0 {
			n = 1
		}
		for range n {
			out = append(out, f.Type)
		}
	}
	return out
}
