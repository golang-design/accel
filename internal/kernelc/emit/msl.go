// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package emit

import (
	"bytes"
	"fmt"
	"go/constant"
	"go/token"
	"math"
	"strings"

	"golang.design/x/accel/internal/kernelc/ir"
)

// # The Metal Shading Language target
//
// specs/021-metal-bringup.md section 5 fixes the subset this emits: a
// straight-line kernel over buffers, one std140 uniform block, thread ids,
// arithmetic, indexing, if/else and for. Threadgroup memory, barriers, atomics,
// subgroups, helper calls and kmath intrinsics belong to
// specs/022-msl-target.md, and each is refused here by name and position rather
// than emitted approximately. A wrong lowering that compiles is the failure this
// whole project is arranged to avoid, so the boundary of the subset is a
// diagnostic and not a comment.
//
// # Buffer indices are the contract
//
// driver.Dispatch already says why position is load-bearing: "a reordering here
// silently swaps two buffers." So the assignment is fixed, total, and derivable
// by the host without extra metadata:
//
//	binding k       -> [[buffer(k)]]
//	slice lengths   -> [[buffer(len(bindings))]]
//	uniform i       -> [[buffer(len(bindings) + 1 + i)]]
//
// The lengths slot is reserved whether or not the body calls len, because a
// layout that depends on the body is a layout the host has to be told about,
// and one unused argument slot is cheaper than a second source of truth.

// MSLLengthsIndex reports the buffer index the generated slice lengths occupy
// for a kernel with n bindings.
func MSLLengthsIndex(n int) int { return n }

// MSLUniformIndex reports the buffer index uniform i occupies for a kernel with
// n bindings.
func MSLUniformIndex(n, i int) int { return n + 1 + i }

// MSL lowers one kernel to Metal Shading Language.
func MSL(k *ir.Func) (string, error) {
	m := &msl{fn: k, binding: map[string]int{}}
	// The slot is the position in Bindings, not Binding.Index, which is the
	// parameter index and counts the Thread and any uniforms. The generated Go
	// lowering already indexes by position -- KernelSlice(a, 0), (a, 1) -- and
	// driver.Dispatch binds by position too, so this is the one numbering the
	// whole system agrees on.
	for i, b := range k.Bindings {
		m.binding[b.Name] = i
	}
	m.emit()
	if m.err != nil {
		return "", m.err
	}
	return m.buf.String(), nil
}

type msl struct {
	buf     bytes.Buffer
	fn      *ir.Func
	binding map[string]int
	err     error
}

func (m *msl) fail(format string, args ...any) {
	if m.err == nil {
		m.err = fmt.Errorf("accel: "+format, args...)
	}
}

// refuse reports a construct outside this child's subset.
//
// It names the target, which specs/004-kernel-authoring.md requires: "a
// target-specific rejection names the target." A kernel legal on the CPU and
// not yet on Metal is a different fact from a kernel that is illegal.
func (m *msl) refuse(what string, pos token.Pos) {
	m.fail("%s is not in the MSL subset of specs/021-metal-bringup.md and belongs to "+
		"specs/022-msl-target.md (kernel %s, position %v)", what, m.fn.Name, pos)
}

func (m *msl) printf(format string, args ...any) {
	if m.err == nil {
		fmt.Fprintf(&m.buf, format, args...)
	}
}

func (m *msl) emit() {
	k := m.fn
	if !k.Kernel {
		m.fail("MSL is emitted for kernels, and %s is a helper", k.Name)
		return
	}
	if k.Cooperative {
		m.refuse("a cooperative kernel (shared memory, a barrier, or a subgroup operation)", k.Pos())
		return
	}
	if len(k.Shared) > 0 {
		m.refuse("workgroup-shared memory", k.Pos())
		return
	}
	if len(k.Helpers) > 0 {
		m.refuse("a helper function call", k.Pos())
		return
	}

	m.printf("#include <metal_stdlib>\n")
	m.printf("using namespace metal;\n\n")

	for _, u := range k.Uniforms {
		m.uniformStruct(u)
	}

	m.printf("kernel void %s(", k.Name)
	nb := len(k.Bindings)
	for i, b := range k.Bindings {
		qual := "device"
		if !b.Write {
			qual = "const device"
		}
		m.printf("\n    %s %s *%s [[buffer(%d)]],", qual, m.dtype(b.Type.Elem), b.Name, i)
	}
	m.printf("\n    constant uint *_lens [[buffer(%d)]],", MSLLengthsIndex(nb))
	for i, u := range k.Uniforms {
		m.printf("\n    constant %s &%s [[buffer(%d)]],", u.TypeName, u.Name, MSLUniformIndex(nb, i))
	}
	// The three ids are always declared. MSL does not object to an unused
	// parameter, and declaring them unconditionally keeps the signature a
	// function of the binding layout alone.
	m.printf("\n    uint3 _gid [[thread_position_in_grid]],")
	m.printf("\n    uint3 _lid [[thread_position_in_threadgroup]],")
	m.printf("\n    uint3 _wid [[threadgroup_position_in_grid]]) {\n")
	m.block(k.Body, 1)
	m.printf("}\n")
}

// uniformStruct emits a std140 block with its padding spelled out.
//
// The padding is explicit because MSL's own struct layout is not std140:
// specs/004-kernel-authoring.md settles that "the GPU layout owns the padding",
// and the generated codec on the host already writes at these offsets. Emitting
// a struct whose members merely happen to land there on this compiler would be
// a layout nobody checked.
func (m *msl) uniformStruct(u *ir.Uniform) {
	m.printf("struct %s {\n", u.TypeName)
	at := 0
	for i, f := range u.Fields {
		if f.Kind != "scalar" {
			m.refuse("a "+f.Kind+" member of uniform block "+u.TypeName, m.fn.Pos())
			return
		}
		if f.Offset < at {
			m.fail("uniform block %s field %s is at offset %d, behind the previous field's end at %d",
				u.TypeName, f.Name, f.Offset, at)
			return
		}
		if pad := f.Offset - at; pad > 0 {
			m.printf("    char _pad%d[%d];\n", i, pad)
		}
		m.printf("    %s %s;\n", m.scalar(f.Scalar), f.Name)
		at = f.Offset + 4
	}
	if pad := u.Size - at; pad > 0 {
		m.printf("    char _tail[%d];\n", pad)
	}
	m.printf("};\n\n")
}

func (m *msl) scalar(goSpelling string) string {
	switch goSpelling {
	case "uint32":
		return "uint"
	case "int32":
		return "int"
	case "float32":
		return "float"
	}
	m.fail("uniform member type %q has no MSL spelling in this subset", goSpelling)
	return "void"
}

func (m *msl) dtype(t *ir.Type) string {
	if t == nil {
		m.fail("a binding with no element type")
		return "void"
	}
	switch t.Kind {
	case ir.Bool:
		return "bool"
	case ir.I32:
		return "int"
	case ir.U32:
		return "uint"
	case ir.F32:
		return "float"
	case ir.F16:
		return "half"
	}
	m.fail("dtype %v has no MSL spelling in the subset of specs/021-metal-bringup.md", t.Kind)
	return "void"
}

func (m *msl) block(b *ir.Block, depth int) {
	for _, s := range b.List {
		m.stmt(s, depth)
	}
}

func mslIndent(depth int) string { return strings.Repeat("    ", depth) }

func (m *msl) stmt(s ir.Stmt, depth int) {
	switch s := s.(type) {
	case *ir.Block:
		m.printf("%s{\n", mslIndent(depth))
		m.block(s, depth+1)
		m.printf("%s}\n", mslIndent(depth))

	case *ir.Declare:
		m.printf("%s%s %s = ", mslIndent(depth), m.dtype(s.Local.Type()), s.Local.Name)
		m.value(s.Init)
		m.printf(";\n")

	case *ir.Assign:
		m.printf("%s", mslIndent(depth))
		m.value(s.LHS)
		m.printf(" = ")
		m.value(s.RHS)
		m.printf(";\n")

	case *ir.ExprStmt:
		m.printf("%s", mslIndent(depth))
		m.value(s.X)
		m.printf(";\n")

	case *ir.If:
		m.printf("%sif (", mslIndent(depth))
		m.value(s.Cond)
		m.printf(") {\n")
		m.block(s.Then, depth+1)
		if s.Else == nil {
			m.printf("%s}\n", mslIndent(depth))
			return
		}
		switch els := s.Else.(type) {
		case *ir.Block:
			m.printf("%s} else {\n", mslIndent(depth))
			m.block(els, depth+1)
			m.printf("%s}\n", mslIndent(depth))
		default:
			m.printf("%s} else\n", mslIndent(depth))
			m.stmt(els, depth)
		}

	case *ir.For:
		m.forStmt(s, depth)

	case *ir.Break:
		m.printf("%sbreak;\n", mslIndent(depth))

	case *ir.Continue:
		m.printf("%scontinue;\n", mslIndent(depth))

	case *ir.Return:
		if s.Value != nil {
			m.fail("a kernel returns nothing, and %s returns a value", m.fn.Name)
			return
		}
		m.printf("%sreturn;\n", mslIndent(depth))

	default:
		m.fail("no MSL lowering for statement %T", s)
	}
}

// forStmt emits a C for loop.
//
// The declaration stays inside the clause, unlike the Go lowering which hoists
// it: C's for-init takes a declaration directly, so the reason the Go target
// wraps the loop in a block does not apply.
func (m *msl) forStmt(s *ir.For, depth int) {
	m.printf("%sfor (", mslIndent(depth))
	switch init := s.Init.(type) {
	case nil:
	case *ir.Declare:
		m.printf("%s %s = ", m.dtype(init.Local.Type()), init.Local.Name)
		m.value(init.Init)
	case *ir.Assign:
		m.value(init.LHS)
		m.printf(" = ")
		m.value(init.RHS)
	default:
		m.fail("no MSL lowering for %T in a for clause", s.Init)
	}
	m.printf("; ")
	if s.Cond != nil {
		m.value(s.Cond)
	}
	m.printf("; ")
	switch post := s.Post.(type) {
	case nil:
	case *ir.Assign:
		m.value(post.LHS)
		m.printf(" = ")
		m.value(post.RHS)
	default:
		m.fail("no MSL lowering for %T in a for clause", s.Post)
	}
	m.printf(") {\n")
	m.block(s.Body, depth+1)
	m.printf("%s}\n", mslIndent(depth))
}

func (m *msl) value(v ir.Value) {
	switch v := v.(type) {
	case *ir.Const:
		m.constant(v)

	case *ir.Param:
		m.printf("%s", v.Name)

	case *ir.Local:
		m.printf("%s", v.Name)

	case *ir.FieldSel:
		m.value(v.X)
		// An id component is a vector lane in MSL and a struct field in Go. A
		// uniform member keeps its own name.
		if isID(v.X) {
			m.printf(".%s", strings.ToLower(v.Name))
			return
		}
		m.printf(".%s", v.Name)

	case *ir.IndexExpr:
		m.value(v.X)
		m.printf("[")
		m.value(v.Index)
		m.printf("]")

	case *ir.Unary:
		// Go spells bitwise complement ^x and C spells it ~x. Emitting the Go
		// token would compile in C as exclusive-or with a missing operand, which
		// is a syntax error here and would be a silent sign flip if it ever
		// parsed.
		op := v.Op.String()
		if v.Op == token.XOR {
			op = "~"
		}
		m.printf("%s", op)
		m.value(v.X)

	case *ir.Binary:
		m.binary(v)

	case *ir.Convert:
		m.printf("%s(", m.dtype(v.Type()))
		m.value(v.X)
		m.printf(")")

	case *ir.Len:
		p, ok := v.X.(*ir.Param)
		if !ok {
			m.fail("len of %T is not a binding, which is the only thing with a length on the GPU", v.X)
			return
		}
		idx, ok := m.binding[p.Name]
		if !ok {
			m.fail("len(%s) refers to something that is not a binding", p.Name)
			return
		}
		// int, matching the Go lowering's int32(len(x)): the comparison a
		// kernel writes is against a signed index.
		m.printf("int(_lens[%d])", idx)

	case *ir.IntrinsicCall:
		m.intrinsic(v)

	case *ir.Call:
		m.refuse("a helper function call", v.Pos())

	default:
		m.fail("no MSL lowering for value %T", v)
	}
}

// isID reports whether a value is a thread-id vector, whose components are
// lanes rather than named fields.
func isID(v ir.Value) bool {
	c, ok := v.(*ir.IntrinsicCall)
	if !ok {
		return false
	}
	switch c.Op {
	case ir.OpGlobalID, ir.OpLocalID, ir.OpGroupID:
		return true
	}
	return false
}

func (m *msl) binary(v *ir.Binary) {
	// Go's &^ has no C operator. Spelling it as an explicit complement keeps
	// the meaning rather than approximating it.
	if v.Op == token.AND_NOT {
		m.printf("(")
		m.value(v.X)
		m.printf(" & ~")
		m.value(v.Y)
		m.printf(")")
		return
	}
	m.printf("(")
	m.value(v.X)
	m.printf(" %s ", v.Op)
	m.value(v.Y)
	m.printf(")")
}

// constant emits a literal in its resolved type.
//
// The f32 case carries the same argument as the Go lowering's: a value spelled
// as a decimal has to round-trip through another compiler's parser, and one at
// the boundary between two f32s depends on that parser rounding the way this one
// did. as_type<float> names the bits, so it cannot.
func (m *msl) constant(c *ir.Const) {
	if c.Val == nil {
		m.fail("a constant with no value at %v", c.Pos())
		return
	}
	switch c.Type().Kind {
	case ir.Bool:
		m.printf("%s", c.Val.String())
	case ir.F32:
		f, exact := constant.Float32Val(c.Val)
		bits := math.Float32bits(f)
		if exact && f == float32(int32(f)) && math.Abs(float64(f)) < 1<<23 {
			m.printf("float(%d)", int32(f))
			return
		}
		m.printf("as_type<float>(0x%08Xu) /* %v */", bits, f)
	case ir.U32:
		m.printf("uint(%s)", c.Val.String())
	case ir.I32:
		m.printf("int(%s)", c.Val.String())
	default:
		m.fail("constant of kind %v has no MSL spelling in this subset", c.Type().Kind)
	}
}

func (m *msl) intrinsic(v *ir.IntrinsicCall) {
	switch v.Op {
	case ir.OpGlobalID:
		m.printf("_gid")
	case ir.OpLocalID:
		m.printf("_lid")
	case ir.OpGroupID:
		m.printf("_wid")
	default:
		m.refuse(v.Op.String(), v.Pos())
	}
}
