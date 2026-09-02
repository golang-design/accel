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
	"golang.design/x/accel/internal/mslabi"
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

// The MSL argument numbering lives in internal/mslabi, not here.
//
// The Metal backend binds against it and needs nothing else from this package,
// and importing this one reaches go/types through ir — which put a type checker
// in every darwin binary linking accel, for one constant and two additions. See
// that package's doc and specs/012-kernel-pipeline.md.
const MSLContractOff = mslabi.ContractOff

// MSLLengthsIndex reports the buffer index the generated slice lengths occupy
// for a kernel with n bindings.
func MSLLengthsIndex(n int) int { return mslabi.LengthsIndex(n) }

// MSLUniformIndex reports the buffer index uniform i occupies for a kernel with
// n bindings.
func MSLUniformIndex(n, i int) int { return mslabi.UniformIndex(n, i) }

// MSLFaultIndex reports the buffer index the fault word occupies for a kernel
// with n bindings and u uniforms.
func MSLFaultIndex(n, u int) int { return mslabi.FaultIndex(n, u) }

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
	m.param = map[string]*ir.Param{}
	for _, p := range k.Params {
		m.param[p.Name] = p
	}
	m.need = map[string]bool{}
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

	// stageOut and the three that follow are set while a graphics stage's body
	// is being emitted, so a return can assemble this target's output struct.
	// Empty for a compute kernel, which returns nothing.
	stageOut      string
	stageKind     ir.Stage
	stageVaryings *ir.Type
	stageOutputs  []*ir.Target

	// atomic is every binding the body touches with an atomic, by name.
	//
	// It changes the parameter's *type*, not only the operation: MSL requires
	// device atomic_uint * rather than device uint *, and a plain read of one
	// has to go through atomic_load_explicit. That is why this is computed
	// before the signature is printed rather than discovered while printing the
	// body.
	atomic map[*ir.Param]bool

	// param is the kernel's own parameters by name, which is what bindingType
	// needs to ask atomic about a binding. Keyed by name here and by identity
	// in atomic, so a helper parameter that happens to share a binding's
	// name is its own parameter and not an atomic one.
	param map[string]*ir.Param

	// need records the built-in helpers the body reached, so they are emitted
	// once and only when used. MSL has no statement expression, so an operation
	// that needs more than one statement becomes a function.
	need map[string]bool

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
	if m.err == nil {
		m.err = &Refusal{Kernel: m.fn.Name, What: what, Pos: pos}
	}
}

// Refusal is the MSL emitter's report of a construct outside its subset.
//
// A type rather than a formatted string so that the generator, which has the
// file set, can print the position as file:line:col; the emitter has only the
// token.Pos.
type Refusal struct {
	Kernel string
	What   string
	Pos    token.Pos
}

func (r *Refusal) Error() string {
	return fmt.Sprintf("accel: %s is not in the MSL subset of specs/021-metal-bringup.md and "+
		"belongs to specs/022-msl-target.md (kernel %s, position %v)", r.What, r.Kernel, r.Pos)
}

// Message is the refusal for the generated record, with its position already
// resolved by the caller to the file:line:col the other generated positions
// use.
func (r *Refusal) Message(pos string) string {
	return fmt.Sprintf("%s is not in the MSL subset (specs/022-msl-target.md), at %s",
		r.What, pos)
}

func (m *msl) printf(format string, args ...any) {
	if m.err == nil {
		fmt.Fprintf(&m.buf, format, args...)
	}
}

func (m *msl) emit() {
	k := m.fn
	if !k.Stage.Entry() {
		m.fail("MSL is emitted for entry points, and %s is a helper", k.Name)
		return
	}
	// A cooperative kernel is emitted as the author wrote it. MSL has real
	// barriers and real threadgroup memory, so the resumable state machine
	// specs/018-cooperative-lowering.md generates for the CPU has no reason to
	// exist here -- it is a way to run a barrier on a target that has none.
	//
	// Both forms come from one IR, which is what makes the differential
	// meaningful: the CPU advances a program counter and Metal executes a
	// barrier, and they must still agree on every element.

	// The body is emitted into a scratch buffer first, because which built-in
	// helpers it needs is only known once it has been lowered -- the same
	// reason Generate emits the Go body before its import list.
	head := m.buf
	m.buf = bytes.Buffer{}
	if k.Stage == ir.StageVertex || k.Stage == ir.StageFragment {
		m.stageEmit(k)
	} else {
		m.body(k)
	}
	body := m.buf
	m.buf = head

	m.printf("#include <metal_stdlib>\n")
	m.printf("using namespace metal;\n")
	m.printf("%s\n\n", MSLContractOff)
	m.prelude()
	m.buf.Write(body.Bytes())
}

// prelude emits the built-in helpers the body reached.
func (m *msl) prelude() {
	for _, h := range mslPrelude {
		if m.need[h.name] {
			m.printf("%s\n", h.text)
		}
	}
}

// mslPrelude is every built-in helper, in a fixed order so output is stable.
//
// # Why compare-exchange is a function
//
// MSL offers only the *weak* form, which may fail spuriously: it can report no
// swap while the value did equal the expected one. This model's operation
// returns the previous value, so a caller compares it against what they
// expected -- and a spurious failure would return the expected value while
// nothing was swapped, which reads as success. The loop distinguishes the two:
// a genuine mismatch leaves a different value in e and returns it, and a
// spurious failure leaves the expected value and retries.
//
// The Go lowering has no equivalent because a mutex-free CPU emulation cannot
// fail spuriously in the first place.
var mslPrelude = []struct{ name, text string }{
	// Saturating float-to-integer, specs/051-float-to-int.md §2.
	//
	// The branch order mirrors kmath.ToI32 and kmath.ToU32 exactly, because the
	// intrinsic's class is Exact: the two backends must agree bit for bit, and
	// the cheapest way to make that true is one shape written twice rather than
	// two clever ones. MSL's int(x) is as undefined as Go's for a value the
	// destination cannot hold, so clamping is not an optimisation of it.
	//
	// x != x is the NaN test in both, and it is spelled that way rather than
	// with isnan() so the contraction-off pragma cannot change it.
	{"to_i32", `static int _accel_to_i32(float x) {
    if (x != x) { return 0; }
    if (x <= -2147483648.0f) { return (-2147483647 - 1); }
    if (x >= 2147483648.0f) { return 2147483647; }
    return int(x);
}
`},
	{"to_u32", `static uint _accel_to_u32(float x) {
    if (x != x) { return 0u; }
    if (x <= 0.0f) { return 0u; }
    if (x >= 4294967296.0f) { return 4294967295u; }
    return uint(x);
}
`},

	// NaN-propagating minimum and maximum, kmath.Min and kmath.Max's contract.
	//
	// The NaN test is x != x for the reason to_i32's is, and the result is the
	// quiet NaN kmath returns, spelled by its bits so the differential can
	// compare them exactly. Once neither operand is NaN, fmin and fmax are the
	// ordinary minimum and maximum.
	{"fmin", `static float _accel_fmin(float a, float b) {
    if (a != a || b != b) { return as_type<float>(0x7FC00000u); }
    return fmin(a, b);
}
`},
	{"fmax", `static float _accel_fmax(float a, float b) {
    if (a != a || b != b) { return as_type<float>(0x7FC00000u); }
    return fmax(a, b);
}
`},
	// The subgroup minimum and maximum over f32, with the same contract:
	// a NaN in any active lane makes the reduction NaN. simd_min and simd_max
	// are fmin and fmax across the lanes, so they would return the smallest
	// non-NaN and agree with the CPU scheduler on every input but the one
	// the contract is about. simd_any is the cross-lane "is any NaN".
	{"simd_fmin", `static float _accel_simd_fmin(float x) {
    if (simd_any(x != x)) { return as_type<float>(0x7FC00000u); }
    return simd_min(x);
}
`},
	{"simd_fmax", `static float _accel_simd_fmax(float x) {
    if (simd_any(x != x)) { return as_type<float>(0x7FC00000u); }
    return simd_max(x);
}
`},
	// Integer division and shifts, specs/008-numerics.md section 3.
	//
	// Go defines a shift by a count at or above the width (zero, or the sign
	// for a signed right shift) and MSL does not, so the helpers spell Go's
	// result and the differential is exact on every count. Division by zero
	// is an execution error in Go and undefined in MSL: the kernel records
	// mslabi.FaultDivByZero in its fault word and yields zero, and the backend
	// reports the word after the submission. MinInt32 / -1 wraps in Go and
	// traps or is undefined in C; it wraps here, since Go's result is defined
	// and exact. A stage has no fault word, so its division yields zero and
	// records nothing (specs/022-msl-target.md section 5).
	{"shl_u32", `static uint _accel_shl_u32(uint x, uint n) { return n >= 32u ? 0u : (x << n); }
`},
	{"shr_u32", `static uint _accel_shr_u32(uint x, uint n) { return n >= 32u ? 0u : (x >> n); }
`},
	{"shl_i32", `static int _accel_shl_i32(int x, uint n) { return n >= 32u ? 0 : int(uint(x) << n); }
`},
	{"shr_i32", `static int _accel_shr_i32(int x, uint n) {
    if (n >= 32u) { return x < 0 ? -1 : 0; }
    return x < 0 ? ~((~x) >> n) : (x >> n);
}
`},
	{"div_i32", `static int _accel_div_i32(int a, int b, device atomic_uint *fault) {
    if (b == 0) { atomic_store_explicit(fault, 1u, memory_order_relaxed); return 0; }
    if (a == INT_MIN && b == -1) { return INT_MIN; }
    return a / b;
}
`},
	{"rem_i32", `static int _accel_rem_i32(int a, int b, device atomic_uint *fault) {
    if (b == 0) { atomic_store_explicit(fault, 1u, memory_order_relaxed); return 0; }
    if (b == -1) { return 0; }
    return a % b;
}
`},
	{"div_u32", `static uint _accel_div_u32(uint a, uint b, device atomic_uint *fault) {
    if (b == 0u) { atomic_store_explicit(fault, 1u, memory_order_relaxed); return 0u; }
    return a / b;
}
`},
	{"rem_u32", `static uint _accel_rem_u32(uint a, uint b, device atomic_uint *fault) {
    if (b == 0u) { atomic_store_explicit(fault, 1u, memory_order_relaxed); return 0u; }
    return a % b;
}
`},
	{"sdiv_i32", `static int _accel_sdiv_i32(int a, int b) {
    if (b == 0) { return 0; }
    if (a == INT_MIN && b == -1) { return INT_MIN; }
    return a / b;
}
`},
	{"srem_i32", `static int _accel_srem_i32(int a, int b) { return (b == 0 || b == -1) ? 0 : (a % b); }
`},
	{"sdiv_u32", `static uint _accel_sdiv_u32(uint a, uint b) { return b == 0u ? 0u : (a / b); }
`},
	{"srem_u32", `static uint _accel_srem_u32(uint a, uint b) { return b == 0u ? 0u : (a % b); }
`},
	{"cas_u32", `static uint _accel_cas_u32(device atomic_uint *p, uint expected, uint desired) {
    uint e = expected;
    while (!atomic_compare_exchange_weak_explicit(p, &e, desired,
                                                  memory_order_relaxed, memory_order_relaxed)) {
        if (e != expected) { return e; }
        e = expected;
    }
    return expected;
}
`},
	{"cas_i32", `static int _accel_cas_i32(device atomic_int *p, int expected, int desired) {
    int e = expected;
    while (!atomic_compare_exchange_weak_explicit(p, &e, desired,
                                                  memory_order_relaxed, memory_order_relaxed)) {
        if (e != expected) { return e; }
        e = expected;
    }
    return expected;
}
`},

	// # Why the fetch is a function and why the guard is spelled this way
	//
	// Metal leaves texture2d::read out of bounds undefined, and
	// specs/032-stage-abi.md section 5 requires zero, so the test is emitted
	// here whatever the coordinate. A function rather than a conditional
	// expression at the call site, because the coordinate arguments would
	// otherwise be evaluated three times each and the subset admits an argument
	// with a side effect.
	//
	// The two halves of the test are not interchangeable. get_width returns
	// uint, so `x < t.get_width()` with an int x is compared *unsigned* by C's
	// usual arithmetic conversions: x = -1 becomes 4294967295, which passes
	// every plausible width and produces exactly the out-of-range read the rule
	// exists to prevent. The sign test comes first, and the magnitude test
	// converts explicitly only after it is known non-negative.
	{"fetch2d", `static float4 _accel_fetch2d(texture2d<float> t, int x, int y) {
    if (x < 0 || y < 0) { return float4(0.0); }
    if (uint(x) >= t.get_width() || uint(y) >= t.get_height()) { return float4(0.0); }
    return t.read(uint2(uint(x), uint(y)));
}
`},
}

// body emits the uniform blocks, the helpers, and the kernel itself.
func (m *msl) body(k *ir.Func) {
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
		if !b.Write && !m.atomic[m.param[b.Name]] {
			qual = "const device"
		}
		m.printf("\n    %s %s *%s [[buffer(%d)]],", qual, m.bindingType(b), mslIdent(b.Name), i)
	}
	m.printf("\n    constant uint *_lens [[buffer(%d)]],", MSLLengthsIndex(nb))
	for i, u := range k.Uniforms {
		m.printf("\n    constant %s &%s [[buffer(%d)]],", u.TypeName, mslIdent(u.Name), MSLUniformIndex(nb, i))
	}
	// The three ids and the grid size are always declared. MSL does not object
	// to an unused parameter, and declaring them unconditionally keeps the
	// signature a function of the binding layout alone.
	//
	// The workgroup extent is *not* here: it is a compile-time constant from
	// the accel:kernel directive, so specs/052-dispatch-shape.md §2 lowers it
	// to a literal at each use rather than to a fifth attribute.
	// The fault word, specs/008-numerics.md section 3: reserved whether or
	// not the body divides, for the reason the lengths slot is.
	m.printf("\n    device atomic_uint *_fault [[buffer(%d)]],", MSLFaultIndex(nb, len(k.Uniforms)))
	m.printf("\n    uint3 _gid [[thread_position_in_grid]],")
	m.printf("\n    uint3 _lid [[thread_position_in_threadgroup]],")
	m.printf("\n    uint3 _wid [[threadgroup_position_in_grid]],")
	m.printf("\n    uint3 _ngroups [[threadgroups_per_grid]],")
	// The subgroup attributes are always declared for the same reason the ids
	// are: the signature stays a function of the binding layout alone, and MSL
	// does not object to a parameter nothing reads.
	m.printf("\n    uint _sgsize [[threads_per_simdgroup]],")
	m.printf("\n    uint _sglane [[thread_index_in_simdgroup]],")
	m.printf("\n    uint _sgid [[simdgroup_index_in_threadgroup]]) {\n")
	// Threadgroup storage is declared in the body rather than passed as a
	// [[threadgroup(k)]] parameter. Its extent is fixed at pipeline creation
	// either way, and declaring it here means the host binds nothing and cannot
	// bind the wrong size: specs/002-compute-model.md makes the extent the
	// compiler's, and the authored form is a pointer to a fixed-size array for
	// exactly this reason.
	for _, sh := range k.Shared {
		if sh.Type == nil || sh.Type.Kind != ir.Array {
			m.fail("shared %s is %v, and shared storage is a fixed-size array", sh.Name, sh.Type)
			return
		}
		m.printf("    threadgroup %s %s[%d];\n", m.dtype(sh.Type.Elem), mslIdent(sh.Name), sh.Type.Len)
	}
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
	m.printf("static %s %s(", ret, mslIdent(h.Name))
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
			m.printf("%s %s *%s", qual, m.dtype(t.Elem), mslIdent(p.Name))
			continue
		}
		m.printf("%s %s", m.dtype(t), mslIdent(p.Name))
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
		decl, size, ok := m.uniformField(u, f)
		if !ok {
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
		m.printf("    %s\n", decl)
		at = f.Offset + size
	}
	if pad := u.Size - at; pad > 0 {
		m.printf("    char _tail[%d];\n", pad)
	}
	m.printf("};\n\n")
}

// uniformField declares one member, and reports how many bytes it occupies.
//
// # Why nothing here uses an MSL vector or matrix type
//
// float3 is sixteen bytes in MSL and twelve in std140, and the difference is not
// padding nobody can see: std140 packs the *next* scalar into the remainder, so
// a block with Origin at 16 puts Steps at 28. Declaring float3 would place Steps
// at 32 and every value after it would be read from the wrong offset -- a kernel
// that compiles, runs, and computes something else.
//
// So a vector is its components and a matrix is its columns, spelled as plain C
// arrays whose element stride is the one the generated codec wrote at. The Go
// source indexes them the same way, so nothing in the body changes.
func (m *msl) uniformField(u *ir.Uniform, f ir.UniformField) (decl string, size int, ok bool) {
	elem := m.scalar(f.Scalar)
	switch f.Kind {
	case "scalar":
		return fmt.Sprintf("%s %s;", elem, f.Name), 4, true

	case "vector":
		// Components are contiguous in std140, which is what lets the next
		// member share the tail.
		if f.Len < 2 || f.Len > 4 {
			m.fail("uniform block %s field %s is a vector of %d", u.TypeName, f.Name, f.Len)
			return "", 0, false
		}
		return fmt.Sprintf("%s %s[%d];", elem, f.Name, f.Len), f.Len * 4, true

	case "matrix":
		// Stride is the byte distance between columns, which std140 rounds up
		// to sixteen. The extra components are the padding, declared rather
		// than skipped so the array indexes the way the source does.
		if f.Stride%4 != 0 || f.Stride == 0 || f.Len == 0 {
			m.fail("uniform block %s field %s is a matrix with %d columns of stride %d",
				u.TypeName, f.Name, f.Len, f.Stride)
			return "", 0, false
		}
		return fmt.Sprintf("%s %s[%d][%d];", elem, f.Name, f.Len, f.Stride/4),
			f.Len * f.Stride, true
	}
	// An array member's std140 stride is sixteen whatever its element type, so
	// it cannot be one C array of the element type that a caller indexes with
	// one index. It is spelled the way the matrix case above is -- an outer
	// array of the element count, each row the stride in elements -- and the
	// body appends the inner index, which m.value does when it recognises an
	// index into a uniform member.
	//
	// This was refused, on the grounds that no corpus kernel needed it. Pack
	// did, and Pack is what tensor.Contiguous lowers to, so the refusal made a
	// public operator CPU-only without anything above the emitter saying so --
	// a consumer found it when a graph that compiled on the CPU failed on Metal
	// after the weights were already uploaded (accel issue 19).
	if f.Kind == "array" {
		if f.Stride%4 != 0 || f.Stride == 0 || f.Len == 0 {
			m.fail("uniform block %s field %s is an array of %d with stride %d",
				u.TypeName, f.Name, f.Len, f.Stride)
			return "", 0, false
		}
		return fmt.Sprintf("%s %s[%d][%d];", elem, f.Name, f.Len, f.Stride/4),
			f.Len * f.Stride, true
	}
	m.refuse("a "+f.Kind+" member of uniform block "+u.TypeName, m.fn.Pos())
	return "", 0, false
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
	if !m.atomic[m.param[b.Name]] {
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
func atomicBindings(k *ir.Func) map[*ir.Param]bool {
	out := map[*ir.Param]bool{}
	var walkValue func(ir.Value)
	var walk func(ir.Stmt)
	walkValue = func(v ir.Value) {
		switch v := v.(type) {
		case *ir.IntrinsicCall:
			if v.Op.IsAtomic() && len(v.Args) > 0 {
				if p, ok := v.Args[0].(*ir.Param); ok {
					out[p] = true
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

// composite spells a struct or vector literal.
//
// MSL constructs a vector by call syntax and a struct by braces, which is the
// one place the two spellings differ and the reason this is not a single
// printf. Go writes both with braces.
func (m *msl) composite(c *ir.Composite) {
	t := c.Type()
	if t == nil {
		m.fail("a composite with no type")
		return
	}
	open, close := "{", "}"
	if t.Kind == ir.Array {
		open, close = m.dtype(t)+"(", ")"
	} else if t.Kind == ir.Struct {
		open = t.Name + "{"
	} else {
		m.fail("a composite of %v, and MSL constructs a vector or a struct", t)
		return
	}
	m.printf("%s", open)
	for i, e := range c.Elems {
		if i > 0 {
			m.printf(", ")
		}
		m.value(e)
	}
	m.printf("%s", close)
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
	case ir.BF16:
		// ushort, not bfloat: the type is a Metal family capability and the
		// storage is not. specs/021-metal-bringup.md section 5 admits bf16 as
		// sixteen opaque bits, which is all a binding that is only loaded and
		// widened ever needs -- ir.go already forbids arithmetic on it.
		return "ushort"
	case ir.I8:
		// char, not int8_t: MSL's char is signed 8-bit, and the narrow integer
		// types exist here for the reason specs/001-device-resources.md gives
		// -- a quantized plane is bytes, and float(x) on one is an ordinary
		// conversion rather than an intrinsic.
		return "char"
	case ir.U8:
		return "uchar"
	case ir.ID3Kind:
		// uint3, which is what the id attributes already are: a kernel that
		// binds one to a local -- `ws := t.WorkgroupSize()` -- needs the type
		// spelled, and the lanes it then selects are the same lanes isID
		// selects on the intrinsic call itself.
		return "uint3"
	case ir.Array:
		// A short array of one scalar is a vector, which is what a graphics
		// stage exchanges: accel.Vec4 is [4]float32 and MSL spells it float4.
		// Only here and not in the compute path's bindings, where an array
		// element would be a slice of arrays and specs/021-metal-bringup.md
		// leaves that out of the subset.
		if t.Elem != nil && t.Len >= 2 && t.Len <= 4 {
			return fmt.Sprintf("%s%d", m.dtype(t.Elem), t.Len)
		}
	}
	// A refusal rather than a failure: a type the subset has no spelling for
	// (the ballot mask, specs/058-ballot.md) is a statement about how far the
	// target has got, and it is recorded as NoMSL rather than dropped.
	m.refuse(fmt.Sprintf("the %v type", t.Kind), m.fn.Pos())
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
		m.printf("%s%s %s = ", mslIndent(depth), m.dtype(s.Local.Type()), mslIdent(s.Local.Name))
		m.value(s.Init)
		m.printf(";\n")

	case *ir.Assign:
		if idx, ok := s.LHS.(*ir.IndexExpr); ok {
			if p, isParam := idx.X.(*ir.Param); isParam && m.atomic[p] {
				m.printf("%satomic_store_explicit(&%s[", mslIndent(depth), mslIdent(p.Name))
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
		if m.stageOut != "" {
			m.stageReturn(s, depth)
			return
		}
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
		m.printf("%s", mslIdent(v.Name))

	case *ir.Local:
		m.printf("%s", mslIdent(v.Name))

	case *ir.Composite:
		m.composite(v)

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
		if p, ok := v.X.(*ir.Param); ok && m.atomic[p] {
			m.printf("atomic_load_explicit(&%s[", mslIdent(p.Name))
			m.value(v.Index)
			m.printf("], memory_order_relaxed)")
			break
		}
		m.value(v.X)
		m.printf("[")
		m.value(v.Index)
		m.printf("]")
		// A std140 array member is declared as an outer array of padded rows,
		// so one Go index is two here and the second is always zero: the
		// element sits at the start of its sixteen-byte slot. See uniformField.
		if m.paddedArrayMember(v.X) {
			m.printf("[0]")
		}

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

// isID reports whether a value is an ID3, whose components are vector lanes in
// MSL and named struct fields in Go.
//
// # Why it asks the type and not the expression
//
// It used to match the *intrinsic call*, which is right for `t.GroupID().X`
// and wrong for `g := t.GroupID(); g.X` -- the second is a FieldSel on a local,
// so the lane went out as `.X` and Metal refused it as an illegal vector
// component. No kernel in the corpus bound an id to a local until
// specs/052-dispatch-shape.md added one, so a latent refusal sat here behind a
// spelling nothing used.
//
// The type is what the property belongs to: an ID3 is uint3 in MSL wherever it
// came from.
func isID(v ir.Value) bool {
	t := v.Type()
	return t != nil && t.Kind == ir.ID3Kind
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
	// Signed add, subtract and multiply wrap, which specs/008-numerics.md
	// section 3 makes exact and which Go's int32 does. MSL's int does not: an
	// overflowing signed operation is undefined, and the compiler folds what
	// it may assume never happens. Unsigned arithmetic is defined to wrap, so
	// the operation goes through uint and the bits come back as int -- the
	// same bits Go produces, on every input.
	if v.Type() != nil && v.Type().Kind == ir.I32 && wrapsOnOverflow(v.Op) {
		m.printf("int(uint(")
		m.value(v.X)
		m.printf(") %s uint(", v.Op)
		m.value(v.Y)
		m.printf("))")
		return
	}
	// Integer shifts and division go through helpers that spell Go's result
	// where MSL's is undefined, and record a fault where Go's is a panic.
	if v.Type() != nil && (v.Type().Kind == ir.I32 || v.Type().Kind == ir.U32) {
		suffix := "u32"
		if v.Type().Kind == ir.I32 {
			suffix = "i32"
		}
		switch v.Op {
		case token.SHL, token.SHR:
			name := "shl_" + suffix
			if v.Op == token.SHR {
				name = "shr_" + suffix
			}
			m.need[name] = true
			m.printf("_accel_%s(", name)
			m.value(v.X)
			m.printf(", uint(")
			m.value(v.Y)
			m.printf("))")
			return
		case token.QUO, token.REM:
			name := "div_" + suffix
			if v.Op == token.REM {
				name = "rem_" + suffix
			}
			if m.fn.Stage != ir.StageCompute {
				name = "s" + name
			}
			m.need[name] = true
			m.printf("_accel_%s(", name)
			m.value(v.X)
			m.printf(", ")
			m.value(v.Y)
			if m.fn.Stage == ir.StageCompute {
				m.printf(", _fault")
			}
			m.printf(")")
			return
		}
	}
	m.printf("(")
	m.value(v.X)
	m.printf(" %s ", v.Op)
	m.value(v.Y)
	m.printf(")")
}

// wrapsOnOverflow reports whether an integer operator is one specs/008 defines
// as wrapping. Shifts and division are not, and go through the helpers above.
func wrapsOnOverflow(op token.Token) bool {
	return op == token.ADD || op == token.SUB || op == token.MUL
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
	// The graphics built-ins, each a parameter this target declares on every
	// stage signature. Declared unconditionally for the reason the compute ids
	// are: the signature stays a function of the declared inputs alone.
	case ir.OpVertexIndex:
		m.printf("_vid")
		return
	case ir.OpInstanceIndex:
		m.printf("_iid")
		return
	case ir.OpFragCoord:
		// The interpolated window position arrives in the varyings struct's
		// own [[position]] field. MSL rejects a signature declaring the
		// attribute twice, so there is no separate parameter to name.
		m.printf("_in._pos")
		return
	case ir.OpFrontFacing:
		m.printf("_front")
		return
	case ir.OpDiscard:
		// discard_fragment() does not return: MSL runs on and drops the
		// fragment afterwards, which is why a discarding stage still returns
		// its attachment struct here and on the CPU alike.
		m.printf("discard_fragment()")
		return

	case ir.OpTexelFetch:
		if len(v.Args) != 3 {
			m.fail("Fetch takes a texture and two coordinates at %v", v.Pos())
			return
		}
		m.need["fetch2d"] = true
		m.printf("_accel_fetch2d(")
		for i, a := range v.Args {
			if i > 0 {
				m.printf(", ")
			}
			m.value(a)
		}
		m.printf(")")
		return

	case ir.OpGlobalID:
		m.printf("_gid")
		return
	case ir.OpLocalID:
		m.printf("_lid")
		return
	case ir.OpGroupID:
		m.printf("_wid")
		return

	// The flat indices, x-fastest, which is internal/kernel's linearization
	// and what the oracle computes. LocalIndex and GlobalIndex fold the
	// workgroup extent in as literals for OpWorkgroupSize's reason below;
	// GroupIndex reads the grid's group count.
	case ir.OpLocalIndex:
		w := m.fn.Workgroup
		m.printf("(_lid.x + %du * (_lid.y + %du * _lid.z))", w[0], w[1])
		return
	case ir.OpGroupIndex:
		m.printf("(_wid.x + _ngroups.x * (_wid.y + _ngroups.y * _wid.z))")
		return
	case ir.OpGlobalIndex:
		w := m.fn.Workgroup
		m.printf("((_wid.x + _ngroups.x * (_wid.y + _ngroups.y * _wid.z)) * %du + "+
			"(_lid.x + %du * (_lid.y + %du * _lid.z)))", w[0]*w[1]*w[2], w[0], w[1])
		return

	// The dispatch shape, specs/052-dispatch-shape.md §2.
	//
	// WorkgroupSize is a **literal**, not a read: it is the accel:kernel
	// directive's value, which the front end put in ir.Func.Workgroup and which
	// the host also uses to size the threadgroup. That is what §2 means by
	// keeping a loop bounded by it compile-time uniform -- MSL sees a constant,
	// so a barrier inside such a loop is reached by every invocation for a
	// reason the compiler can see.
	case ir.OpWorkgroupSize:
		w := m.fn.Workgroup
		m.printf("uint3(%d, %d, %d)", w[0], w[1], w[2])
		return
	case ir.OpNumGroups:
		m.printf("_ngroups")
		return
	// Derived rather than a fourth attribute, so the two cannot disagree.
	case ir.OpGlobalSize:
		w := m.fn.Workgroup
		m.printf("(uint3(%d, %d, %d) * _ngroups)", w[0], w[1], w[2])
		return

	case ir.OpToI32:
		m.need["to_i32"] = true
		m.printf("_accel_to_i32(")
		m.value(v.Args[0])
		m.printf(")")
		return

	case ir.OpToU32:
		m.need["to_u32"] = true
		m.printf("_accel_to_u32(")
		m.value(v.Args[0])
		m.printf(")")
		return

	// A float minimum or maximum propagates a NaN, which is kmath's contract
	// and is not MSL's: min and max on float are fmin and fmax, which return
	// the *other* operand when one is NaN. The oracle runs kmath, so a bare
	// min here agreed with it on every input but the one the contract is
	// about. The integer forms have no NaN and keep the built-in.
	case ir.OpMin, ir.OpMax:
		if len(v.Args) == 2 && v.Args[0].Type() != nil && v.Args[0].Type().Kind == ir.F32 {
			name := "fmin"
			if v.Op == ir.OpMax {
				name = "fmax"
			}
			m.need[name] = true
			m.printf("_accel_%s(", name)
			m.value(v.Args[0])
			m.printf(", ")
			m.value(v.Args[1])
			m.printf(")")
			return
		}

	case ir.OpBarrier:
		// Shared *and* storage, which is what Thread.Barrier means:
		// specs/002-compute-model.md §2.5's table gives this exact lowering.
		//
		// Corrected 2026-08-27. This emitted mem_threadgroup alone, and the
		// comment that stood here asserted the narrower rule as if it were the
		// spec's -- so the code and its own justification were wrong together,
		// which is why reading either against the other would not have caught
		// it. §2.5 is unambiguous: t.Barrier() makes shared and storage writes
		// visible across the workgroup.
		//
		// Nothing in the corpus depended on the storage half, so this was a
		// latent divergence rather than a live wrong answer -- and it stayed
		// latent only until someone wrote the kernel §2.5 says is legal.
		// specs/050-barrier-scopes.md owns the masked variants that let a
		// caller ask for the cheaper one on purpose.
		m.printf("threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)")
		return

	// The masked variants, specs/050-barrier-scopes.md §2. §2.5's lowering
	// table gives each target's text, so these are transcribed from it rather
	// than derived -- which is what makes the assertion about them an
	// assertion about the *emitted text*: a kernel whose data fits in one
	// threadgroup gets the right answer at any of the three scopes, so a
	// result cannot tell them apart.
	case ir.OpBarrierShared:
		m.printf("threadgroup_barrier(mem_flags::mem_threadgroup)")
		return
	case ir.OpBarrierStorage:
		m.printf("threadgroup_barrier(mem_flags::mem_device)")
		return

	// Subgroup scope. 002 §2.5's table gives simdgroup_barrier, and the mask is
	// the wide one because §2.5 says this call makes shared *and* storage
	// visible -- what narrows here is the set of lanes, not the memory.
	case ir.OpSubgroupBarrier:
		m.printf("simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)")
		return

	// Ballot, specs/058-ballot.md §3. Refused rather than approximated:
	// simd_ballot returns a simd_vote, which is not an integer and has no
	// conversion to one in the subset (specs/022-msl-target.md §5). This is
	// the first kernel-visible capability the first backend does not have, so
	// the refusal names what is missing rather than reading as a compiler gap.
	case ir.OpBallot, ir.OpMaskCount, ir.OpMaskBit, ir.OpMaskLowestSet,
		ir.OpMaskCountLower, ir.OpMaskAny:
		m.fail("%v needs a subgroup ballot, and MSL's simd_ballot returns a simd_vote "+
			"rather than an integer this subset can hold (specs/022-msl-target.md "+
			"section 5). The capability is subgroup_ballot and this device does not "+
			"have it; kernel %s, position %v", v.Op, m.fn.Name, v.Pos())
		return

	case ir.OpSubgroupSize:
		m.printf("_sgsize")
		return
	case ir.OpSubgroupInvocationID:
		m.printf("_sglane")
		return
	case ir.OpSubgroupID:
		m.printf("_sgid")
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
	case ir.OpBF16ToF32:
		// The widening needs no bfloat type. bf16 *is* f32's top half -- the
		// same eight-bit exponent, seven mantissa bits, and sixteen zero bits
		// below -- so widening is a shift and a bitcast over the storage, which
		// every Metal family has. That includes the infinities and NaNs, whose
		// payload bits move with everything else.
		m.printf("as_type<float>(uint(")
		m.value(v.Recv)
		m.printf(") << 16)")
		return
	case ir.OpF32ToBF16:
		// The narrowing is a different question and stays refused. It has to
		// round, and the rounding this project admits is to nearest-even
		// (specs/008-numerics.md section 4) with a tie rule over a mantissa
		// Metal has no type for: bfloat exists in Metal 3.1 on some families
		// and not others, so it is a capability rather than a spelling.
		m.refuse(v.Op.String()+" (bfloat is a Metal family capability, not a type)", v.Pos())
		return
	}

	if v.Op.IsAtomic() {
		m.atomicCall(v)
		return
	}

	if fn, ok := mslSubgroupNullary[v.Op]; ok {
		m.printf("%s()", fn)
		return
	}

	if fn, ok := mslSubgroup[v.Op]; ok {
		if len(v.Args) != 1 {
			m.fail("subgroup operation %v takes one value at %v", v.Op, v.Pos())
			return
		}
		// The f32 minimum and maximum go through the NaN-propagating helpers
		// for the reason OpMin and OpMax do above: the built-in drops a NaN.
		switch v.Op {
		case ir.OpSubgroupMinF32:
			fn, m.need["simd_fmin"] = "_accel_simd_fmin", true
		case ir.OpSubgroupMaxF32:
			fn, m.need["simd_fmax"] = "_accel_simd_fmax", true
		}
		m.printf("%s(", fn)
		m.value(v.Args[0])
		m.printf(")")
		return
	}

	if fn, ok := mslLaneRead[v.Op]; ok {
		if len(v.Args) != 2 {
			m.fail("subgroup operation %v takes a value and a lane at %v", v.Op, v.Pos())
			return
		}
		m.printf("%s(", fn)
		m.value(v.Args[0])
		m.printf(", ushort(")
		m.value(v.Args[1])
		m.printf("))")
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
	// Compare-exchange takes its expected value by pointer in MSL and by value
	// here, and the weak form it offers can fail spuriously, so it is a helper
	// rather than a call. See mslPrelude.
	switch v.Op {
	case ir.OpAtomicCompareExchangeU32, ir.OpAtomicCompareExchangeI32:
		name := "cas_u32"
		if v.Op == ir.OpAtomicCompareExchangeI32 {
			name = "cas_i32"
		}
		if len(v.Args) != 4 {
			m.fail("compare-exchange takes a buffer, an index, an expected and a desired "+
				"value at %v", v.Pos())
			return
		}
		m.need[name] = true
		m.printf("_accel_%s(&", name)
		m.value(v.Args[0])
		m.printf("[")
		m.value(v.Args[1])
		m.printf("], ")
		m.value(v.Args[2])
		m.printf(", ")
		m.value(v.Args[3])
		m.printf(")")
		return
	}

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

// mslSubgroupNullary is each subgroup operation that takes no value.
//
// A map with one entry rather than a case in the switch above, so that the gate
// walking the rendezvous opcodes can see it: a spelling reachable only from a
// switch is one that test cannot tell apart from a missing one.
var mslSubgroupNullary = map[ir.Opcode]string{
	ir.OpElect: "simd_is_first",
}

// mslSubgroup is each single-argument subgroup operation's Metal spelling.
//
// Ballot is absent because simd_ballot returns a simd_vote rather than an
// integer, and the conversion is family-dependent. That is a property of the
// return type rather than of inactive lanes, which is why the shuffles below
// are here and it is not.
var mslSubgroup = map[ir.Opcode]string{
	ir.OpSubgroupAddF32: "simd_sum",
	// The f32 minimum and maximum are listed so the table says the operation
	// lowers; the emitter substitutes the NaN-propagating helpers at the call.
	ir.OpSubgroupMinF32: "simd_min",
	ir.OpSubgroupMaxF32: "simd_max",

	// The integer minima and maxima, specs/059-subgroup-reductions.md §5.
	// simd_min and simd_max are overloaded on the argument type in MSL, so
	// these share a spelling with the f32 pair and differ only in what the
	// emitter passes them -- which is why the table has four rows naming two
	// functions rather than four functions.
	ir.OpSubgroupMinI32: "simd_min",
	ir.OpSubgroupMaxI32: "simd_max",
	ir.OpSubgroupMinU32: "simd_min",
	ir.OpSubgroupMaxU32: "simd_max",

	// The bitwise family, specs/059-subgroup-reductions.md §6's second slice.
	// Overloaded on the argument type as the minima are.
	ir.OpSubgroupAndI32: "simd_and",
	ir.OpSubgroupOrI32:  "simd_or",
	ir.OpSubgroupXorI32: "simd_xor",
	ir.OpSubgroupAndU32: "simd_and",
	ir.OpSubgroupOrU32:  "simd_or",
	ir.OpSubgroupXorU32: "simd_xor",

	// The products, specs/059-subgroup-reductions.md §6's third slice.
	// simd_product is overloaded on the argument type as simd_min is.
	ir.OpSubgroupMulF32:    "simd_product",
	ir.OpSubgroupMulI32:    "simd_product",
	ir.OpSubgroupMulU32:    "simd_product",
	ir.OpBroadcastFirstF32: "simd_broadcast_first",
	ir.OpSubgroupAny:       "simd_any",
	ir.OpSubgroupAll:       "simd_all",

	// The scans. Metal accumulates in whatever order the hardware scans in,
	// which is not the oracle's ascending lane order, so two backends agree
	// exactly only where every ordering of the sum is exact. That is a property
	// of a kernel's inputs rather than of this spelling, and it is why the
	// differential's case says so.
	ir.OpSubgroupInclusiveAddF32: "simd_prefix_inclusive_sum",
	ir.OpSubgroupExclusiveAddF32: "simd_prefix_exclusive_sum",
}

// mslLaneRead is each lane-addressed read's Metal spelling.
//
// Metal takes the lane operand as a ushort, so the emitter casts: a u32
// expression passed to a ushort parameter is a narrowing conversion, and MSL's
// warning about one is the sort of thing that is turned into an error by a
// build setting outside this repository's control.
//
// Metal leaves the same reads undefined that specs/002-compute-model.md
// section 5.2 does -- an inactive lane, or an index past the SIMD width -- so
// this is a spelling and not an emulation.
var mslLaneRead = map[ir.Opcode]string{
	ir.OpBroadcastF32:   "simd_broadcast",
	ir.OpShuffleF32:     "simd_shuffle",
	ir.OpShuffleXorF32:  "simd_shuffle_xor",
	ir.OpShuffleUpF32:   "simd_shuffle_up",
	ir.OpShuffleDownF32: "simd_shuffle_down",
}

// mslIntrinsic is each bounded scalar operation's Metal spelling.
//
// These are metal_stdlib's own functions rather than a reimplementation, which
// is the same choice the Go lowering makes by calling accel/kmath: the point of
// specs/008-numerics.md section 6 is that each has a normative domain and error
// ceiling, and the probes are what check a backend meets it. An operation whose
// bound this device misses is answered by changing the lowering or narrowing the
// domain, never by widening the bound.
//
// # Why precise::
//
// This is that rule applied, not a preference. The default namespace missed
// three ceilings on an M2, measured against a higher-precision reference:
//
//	exp   18 ULP against a ceiling of 4
//	sin   1.9e-3 absolute against a ceiling of 2^-20, near |x| = 2^16
//	cos   1.8e-3 absolute, the same
//
// The sin and cos misses are argument reduction giving up on large arguments,
// which is exactly where v0 RoPE positions live, so the fast versions would
// have been wrong precisely where this corpus needs them. specs/008-numerics.md
// section 6 already required the precise operation for sqrt and said why --
// "must not substitute a reciprocal-square-root sequence" -- and the
// measurement extends the same reasoning to the rest.
//
// rsqrt stays in the default namespace: it meets its 4 ULP ceiling there, and
// precise:: has no rsqrt to move it to.
//
// abs is fabs because MSL's abs is the integer one, and the C rule that picks
// between them silently returns an int for a float argument.
//
// min and max here are the integer forms only. A float minimum goes through
// the NaN-propagating helper the intrinsic switch selects first, because
// MSL's float min is fmin and fmin drops a NaN operand.
var mslIntrinsic = map[ir.Opcode]string{
	ir.OpSqrt:  "precise::sqrt",
	ir.OpRSqrt: "rsqrt",
	ir.OpExp:   "precise::exp",
	ir.OpLog:   "precise::log",
	ir.OpSin:   "precise::sin",
	ir.OpCos:   "precise::cos",
	ir.OpTanh:  "precise::tanh",
	ir.OpAbs:   "fabs",
	ir.OpMin:   "min",
	ir.OpMax:   "max",
}

// paddedArrayMember reports whether v is a uniform block's array member, which
// is declared with a padding dimension the Go source does not have.
//
// Only an array: a matrix member is already two-dimensional in the Go source,
// so its two indices are the caller's own and nothing is appended.
func (m *msl) paddedArrayMember(v ir.Value) bool {
	f, ok := v.(*ir.FieldSel)
	if !ok || !m.arraySpelled(v) {
		return false
	}
	for _, u := range m.fn.Uniforms {
		for _, fl := range u.Fields {
			if fl.Name == f.Name {
				return fl.Kind == "array"
			}
		}
	}
	return false
}

// mslIdent spells a Go identifier as an MSL one.
//
// Go and C++ do not share a reserved set: `half`, `float`, `new`, `this`,
// `device`, `kernel` and the rest are ordinary Go names and MSL keywords or
// types, and a name beginning with an underscore is the emitter's own space
// (`_gid`, `_lens`). A colliding name gets a trailing underscore, which no Go
// identifier the front end passes can already end in without also colliding,
// and every other name is printed as it is, so the generated text stays
// readable and the golden stays stable.
func mslIdent(name string) string {
	if strings.HasPrefix(name, "_") || mslReserved[name] {
		return name + "_"
	}
	return name
}

// mslReserved is every word an MSL identifier cannot be, or that names a type
// or a built-in function the emitted text uses: C++ keywords, MSL address
// space and function qualifiers, the scalar and vector types, and the math
// functions the intrinsic table lowers to. Go's own keywords are absent
// because a Go identifier cannot be one.
var mslReserved = map[string]bool{}

func init() {
	for _, w := range strings.Fields(`
		alignas alignof and and_eq asm auto bitand bitor bool catch char char16_t
		char32_t class compl concept consteval constexpr constinit const_cast
		co_await co_return co_yield decltype delete do double dynamic_cast enum
		explicit export extern false float friend inline int long mutable
		namespace new noexcept not not_eq nullptr operator or or_eq private
		protected public register reinterpret_cast requires short signed sizeof
		static static_assert static_cast template this thread_local throw true
		try typedef typeid typename union unsigned using virtual void volatile
		wchar_t while xor xor_eq
		device constant threadgroup thread kernel vertex fragment
		half uint ushort uchar ulong size_t ptrdiff_t
		float2 float3 float4 half2 half3 half4 int2 int3 int4 uint2 uint3 uint4
		short2 short3 short4 ushort2 ushort3 ushort4 bool2 bool3 bool4
		float2x2 float3x3 float4x4 float2x4 float4x2 float3x4 float4x3
		texture2d sampler atomic_uint atomic_int atomic_float
		abs min max clamp mix step smoothstep sign floor ceil round trunc fract
		sqrt rsqrt exp exp2 log log2 pow sin cos tan asin acos atan atan2 sinh
		cosh tanh fma fmin fmax fmod isnan isinf isfinite select any all dot
		cross length normalize distance
		simd_sum simd_min simd_max simd_and simd_or simd_xor simd_product
		simd_any simd_all simd_ballot simd_broadcast simd_shuffle
		threadgroup_barrier simdgroup_barrier as_type
	`) {
		mslReserved[w] = true
	}
}
