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

// MSLContractOff is the pragma every emitted kernel carries.
//
// It is a pragma rather than a compile option because MTLCompileOptions has no
// control over contraction: MTLMathMode.safe disables reassociation and
// denormal flushing and leaves a multiply-add free to fuse, which was measured
// on an M2 rather than assumed. Metal's default is -ffp-contract=fast, so
// without this a*b+c becomes fma(a,b,c) and differs from the CPU backend in the
// last bit -- and specs/006-backends.md makes the CPU backend the oracle, which
// turns that difference into a failure rather than a tolerance to widen.
//
// specs/008-numerics.md section 6 requires contraction to be controlled rather
// than observed. This is where it is controlled.
const MSLContractOff = "#pragma METAL fp contract(off)"

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
	m.atomic = atomicBindings(k)
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

	// atomic is every binding the body touches with an atomic, by name.
	//
	// It changes the parameter's *type*, not only the operation: MSL requires
	// device atomic_uint * rather than device uint *, and a plain read of one
	// has to go through atomic_load_explicit. That is why this is computed
	// before the signature is printed rather than discovered while printing the
	// body.
	atomic map[string]bool

	err error
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

	m.printf("#include <metal_stdlib>\n")
	m.printf("using namespace metal;\n")
	m.printf("%s\n\n", MSLContractOff)

	for _, u := range k.Uniforms {
		m.uniformStruct(u)
	}
	// Helpers before their callers, which C requires and Go does not. The IR
	// already orders them, so this is a print rather than a sort.
	for _, h := range k.Helpers {
		m.helper(h)
	}

	m.printf("kernel void %s(", k.Name)
	nb := len(k.Bindings)
	for i, b := range k.Bindings {
		qual := "device"
		if !b.Write && !m.atomic[b.Name] {
			qual = "const device"
		}
		m.printf("\n    %s %s *%s [[buffer(%d)]],", qual, m.bindingType(b), b.Name, i)
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

// helper emits one free function.
//
// static, because MSL compiles one kernel per library here and a helper with
// external linkage would collide if two kernels in a package declared the same
// one. The Go lowering has the same rule for a different reason, which
// specs/004-kernel-authoring.md records: "emitted ahead of its callers, static
// in MSL, plain in GLSL."
func (m *msl) helper(h *ir.Func) {
	if h.Thread >= 0 {
		// A helper taking a Thread would need the id parameters threaded
		// through every call site. It is legal in the authored subset and no
		// corpus kernel does it, so it is refused by name rather than
		// half-supported.
		m.refuse("a helper taking a Thread", h.Pos())
		return
	}
	ret := "void"
	if h.Result != nil {
		ret = m.dtype(h.Result)
	}
	m.printf("static %s %s(", ret, h.Name)
	for i, p := range h.Params {
		if i > 0 {
			m.printf(", ")
		}
		t := p.Type()
		if t != nil && t.Kind == ir.Slice {
			// A slice parameter is a device pointer, and its length does not
			// travel with it. specs/004-kernel-authoring.md's helper storage
			// rules are what keep this sound; a helper calling len on one is
			// refused below, where the binding index it would need does not
			// exist.
			//
			// The const qualifier is not decoration: a read-only *binding* is
			// emitted as `const device T *`, and C will not pass one to a
			// parameter that drops the qualifier. So a helper that never writes
			// through a parameter must say so, and one that does must not.
			qual := "const device"
			if writesThrough(h.Body, p) {
				qual = "device"
			}
			m.printf("%s %s *%s", qual, m.dtype(t.Elem), p.Name)
			continue
		}
		m.printf("%s %s", m.dtype(t), p.Name)
	}
	m.printf(") {\n")
	m.block(h.Body, 1)
	m.printf("}\n\n")
}

// writesThrough reports whether a body assigns through a pointer parameter.
//
// It answers the question C asks at every call site and Go never asks: whether
// this parameter needs to be mutable. The IR infers read and write for a
// kernel's *bindings* and not for a helper's parameters, because no other
// target needed the distinction -- the Go lowering passes a slice either way
// and GLSL forbids the parameter entirely.
//
// Conservative in the safe direction: anything it cannot see through is treated
// as a write, since emitting const on a parameter that is written is a compile
// error at the assignment, which is a worse place to find out.
func writesThrough(body ir.Stmt, p *ir.Param) bool {
	found := false
	var walkValue func(ir.Value)
	var walk func(ir.Stmt)

	isParam := func(v ir.Value) bool {
		q, ok := v.(*ir.Param)
		return ok && q == p
	}
	walkValue = func(v ir.Value) {
		switch v := v.(type) {
		case *ir.Call:
			// A helper handing the pointer to another helper may write through
			// it there. Following the callee would be the precise answer and
			// this is the safe one, which is the right trade for a case no
			// corpus kernel exercises.
			for _, a := range v.Args {
				if isParam(a) {
					found = true
				}
			}
		case *ir.IntrinsicCall:
			for _, a := range v.Args {
				if isParam(a) {
					found = true
				}
			}
			if v.Recv != nil && isParam(v.Recv) {
				found = true
			}
		}
	}
	walk = func(s ir.Stmt) {
		switch s := s.(type) {
		case *ir.Block:
			for _, x := range s.List {
				walk(x)
			}
		case *ir.Declare:
			walkValue(s.Init)
		case *ir.Assign:
			if idx, ok := s.LHS.(*ir.IndexExpr); ok && isParam(idx.X) {
				found = true
			}
			walkValue(s.RHS)
		case *ir.ExprStmt:
			walkValue(s.X)
		case *ir.If:
			walkValue(s.Cond)
			walk(s.Then)
			if s.Else != nil {
				walk(s.Else)
			}
		case *ir.For:
			if s.Init != nil {
				walk(s.Init)
			}
			if s.Post != nil {
				walk(s.Post)
			}
			walk(s.Body)
		case *ir.Return:
			if s.Value != nil {
				walkValue(s.Value)
			}
		}
	}
	walk(body)
	return found
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

// bindingType is a binding's element type as the signature declares it.
//
// A binding touched atomically is atomic_uint or atomic_int rather than uint or
// int. Metal makes that a property of the pointer rather than of the call, so a
// kernel cannot pass a plain pointer to an atomic operation, and this is the
// only place the two facts meet.
func (m *msl) bindingType(b *ir.Binding) string {
	if !m.atomic[b.Name] {
		return m.dtype(b.Type.Elem)
	}
	switch b.Type.Elem.Kind {
	case ir.U32:
		return "atomic_uint"
	case ir.I32:
		return "atomic_int"
	case ir.F32:
		// atomic<float> exists from Metal 3 and only for add. Refused by name
		// until the capability table can answer for it, rather than emitted and
		// found at run time on a device that lacks it.
		m.refuse("an f32 atomic (atomic<float> is a Metal version capability)", m.fn.Pos())
		return "void"
	}
	m.fail("binding %s is touched atomically and is %v, which has no atomic type in MSL",
		b.Name, b.Type.Elem.Kind)
	return "void"
}

// atomicBindings finds every binding an atomic operates on.
//
// The same walk specs/012-kernel-pipeline.md's access inference makes, repeated
// here because it answers a different question: that pass records that the
// binding is read and written, and this one records that its *type* changes.
func atomicBindings(k *ir.Func) map[string]bool {
	out := map[string]bool{}
	var walkValue func(ir.Value)
	var walk func(ir.Stmt)
	walkValue = func(v ir.Value) {
		switch v := v.(type) {
		case *ir.IntrinsicCall:
			if v.Op.IsAtomic() && len(v.Args) > 0 {
				if p, ok := v.Args[0].(*ir.Param); ok {
					out[p.Name] = true
				}
			}
			for _, a := range v.Args {
				walkValue(a)
			}
		case *ir.Binary:
			walkValue(v.X)
			walkValue(v.Y)
		case *ir.Unary:
			walkValue(v.X)
		case *ir.Convert:
			walkValue(v.X)
		case *ir.IndexExpr:
			walkValue(v.Index)
		case *ir.Call:
			for _, a := range v.Args {
				walkValue(a)
			}
		}
	}
	walk = func(s ir.Stmt) {
		switch s := s.(type) {
		case *ir.Block:
			for _, x := range s.List {
				walk(x)
			}
		case *ir.Declare:
			walkValue(s.Init)
		case *ir.Assign:
			walkValue(s.RHS)
		case *ir.ExprStmt:
			walkValue(s.X)
		case *ir.If:
			walkValue(s.Cond)
			walk(s.Then)
			if s.Else != nil {
				walk(s.Else)
			}
		case *ir.For:
			if s.Init != nil {
				walk(s.Init)
			}
			if s.Post != nil {
				walk(s.Post)
			}
			walk(s.Body)
		case *ir.Return:
			if s.Value != nil {
				walkValue(s.Value)
			}
		}
	}
	walk(k.Body)
	return out
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
		if idx, ok := s.LHS.(*ir.IndexExpr); ok {
			if p, isParam := idx.X.(*ir.Param); isParam && m.atomic[p.Name] {
				m.printf("%satomic_store_explicit(&%s[", mslIndent(depth), p.Name)
				m.value(idx.Index)
				m.printf("], ")
				m.value(s.RHS)
				m.printf(", memory_order_relaxed);\n")
				return
			}
		}
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
		if s.Value == nil {
			m.printf("%sreturn;\n", mslIndent(depth))
			return
		}
		m.printf("%sreturn ", mslIndent(depth))
		m.value(s.Value)
		m.printf(";\n")

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
		// A plain read of a binding whose type is atomic has to go through
		// atomic_load_explicit: MSL will not let an atomic_uint decay to a
		// uint, which is the whole reason the qualifier is on the pointer.
		if p, ok := v.X.(*ir.Param); ok && m.atomic[p.Name] {
			m.printf("atomic_load_explicit(&%s[", p.Name)
			m.value(v.Index)
			m.printf("], memory_order_relaxed)")
			break
		}
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
		m.printf("%s(", v.Callee.Name)
		for i, a := range v.Args {
			if i > 0 {
				m.printf(", ")
			}
			m.value(a)
		}
		m.printf(")")

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
		return
	case ir.OpLocalID:
		m.printf("_lid")
		return
	case ir.OpGroupID:
		m.printf("_wid")
		return

	// The narrow-float conversions. MSL has half natively, so these are casts
	// rather than the bit-packing sequence a target without the format needs --
	// which is why specs/012-kernel-pipeline.md makes them intrinsics instead of
	// IR conversions in the first place.
	case ir.OpF16ToF32:
		m.printf("float(")
		m.value(v.Recv)
		m.printf(")")
		return
	case ir.OpF32ToF16:
		if len(v.Args) != 1 {
			m.fail("ToFloat16 takes one argument at %v", v.Pos())
			return
		}
		m.printf("half(")
		m.value(v.Args[0])
		m.printf(")")
		return
	case ir.OpBF16ToF32, ir.OpF32ToBF16:
		// bfloat exists in Metal 3.1 on some families and not others, so it is
		// a capability rather than a spelling. Refused by name until the
		// capability table can answer for it.
		m.refuse(v.Op.String()+" (bfloat is a Metal family capability, not a type)", v.Pos())
		return
	}

	if v.Op.IsAtomic() {
		m.atomicCall(v)
		return
	}

	if fn, ok := mslIntrinsic[v.Op]; ok {
		m.printf("%s(", fn)
		for i, a := range v.Args {
			if i > 0 {
				m.printf(", ")
			}
			m.value(a)
		}
		m.printf(")")
		return
	}
	m.refuse(v.Op.String(), v.Pos())
}

// atomicCall emits one read-modify-write.
//
// Every atomic here is relaxed, which is what specs/002-compute-model.md
// section 4 specifies: an atomic in this model guarantees atomicity of the
// operation and orders nothing else. Ordering between dispatches is the graph's
// barrier, and ordering within a workgroup is Thread.Barrier. Emitting a
// stronger order would make Metal pay for a guarantee the model does not offer
// and the CPU oracle does not provide.
func (m *msl) atomicCall(v *ir.IntrinsicCall) {
	fn, ok := mslAtomic[v.Op]
	if !ok {
		m.refuse(v.Op.String(), v.Pos())
		return
	}
	if len(v.Args) < 2 {
		m.fail("atomic %v takes a buffer and an index at %v", v.Op, v.Pos())
		return
	}
	m.printf("%s(&", fn)
	m.value(v.Args[0])
	m.printf("[")
	m.value(v.Args[1])
	m.printf("]")
	for _, a := range v.Args[2:] {
		m.printf(", ")
		m.value(a)
	}
	m.printf(", memory_order_relaxed)")
}

// mslAtomic is each atomic's Metal spelling.
//
// Sub is add of the negation on the unsigned side in some languages and a
// distinct function here, so each is named rather than derived. Compare-exchange
// takes an expected *pointer* in MSL and a value in this model, which is why it
// is absent: the shapes differ and a wrong translation would silently never
// swap.
var mslAtomic = map[ir.Opcode]string{
	ir.OpAtomicAddU32:      "atomic_fetch_add_explicit",
	ir.OpAtomicAddI32:      "atomic_fetch_add_explicit",
	ir.OpAtomicSubU32:      "atomic_fetch_sub_explicit",
	ir.OpAtomicSubI32:      "atomic_fetch_sub_explicit",
	ir.OpAtomicMinU32:      "atomic_fetch_min_explicit",
	ir.OpAtomicMinI32:      "atomic_fetch_min_explicit",
	ir.OpAtomicMaxU32:      "atomic_fetch_max_explicit",
	ir.OpAtomicMaxI32:      "atomic_fetch_max_explicit",
	ir.OpAtomicAndU32:      "atomic_fetch_and_explicit",
	ir.OpAtomicOrU32:       "atomic_fetch_or_explicit",
	ir.OpAtomicXorU32:      "atomic_fetch_xor_explicit",
	ir.OpAtomicExchangeU32: "atomic_exchange_explicit",
	ir.OpAtomicExchangeI32: "atomic_exchange_explicit",
}

// mslIntrinsic is each bounded scalar operation's Metal spelling.
//
// These are metal_stdlib's own functions rather than a reimplementation,
// which is the same choice the Go lowering makes by calling accel/kmath: the
// point of specs/008-numerics.md section 6 is that each has a normative domain
// and error ceiling, and the probes are what check a backend meets it. An
// operation whose bound this device misses is answered by changing the lowering
// or narrowing the domain, never by widening the bound.
//
// abs is fabs because MSL's abs is the integer one, and the C rule that picks
// between them silently returns an int for a float argument.
var mslIntrinsic = map[ir.Opcode]string{
	ir.OpSqrt:  "sqrt",
	ir.OpRSqrt: "rsqrt",
	ir.OpExp:   "exp",
	ir.OpLog:   "log",
	ir.OpSin:   "sin",
	ir.OpCos:   "cos",
	ir.OpTanh:  "tanh",
	ir.OpAbs:   "fabs",
	ir.OpMin:   "min",
	ir.OpMax:   "max",
}
