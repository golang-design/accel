// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package emit_test

import (
	"go/constant"
	"go/token"
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/emit"
	"golang.design/x/accel/internal/kernelc/ir"
)

func TestMSLShape(t *testing.T) {
	src, err := emit.MSL(corpusKernel(t, "Add"))
	if err != nil {
		t.Fatalf("MSL: %v", err)
	}
	t.Log("\n" + src)
	for _, want := range []string{
		"kernel void Add(",
		"const device float *a [[buffer(0)]]",
		"device float *out [[buffer(2)]]",
		"constant uint *_lens [[buffer(3)]]",
		"[[thread_position_in_grid]]",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("emitted MSL is missing %q", want)
		}
	}
}

// The MSL target refuses what it cannot lower, by name and with a position.
//
// Every corpus kernel now lowers, so these paths are unreachable from the
// corpus and would otherwise never run -- which is the state a refusal is in
// just before somebody discovers it emits nothing at all. The kernels are the
// corpus's, mutated: loadCorpus rebuilds them per call, so a mutation here
// cannot reach another test.
func TestMSLRefusals(t *testing.T) {
	cases := []struct {
		name string
		mut  func(*ir.Func)
		want string
	}{{
		name: "bfloat",
		mut: func(k *ir.Func) {
			// The *narrowing* is refused: it has to round, and bfloat is a
			// Metal family capability rather than a spelling, so the rounding
			// cannot be spelled either. The widening lowers -- it is a shift
			// over the storage and needs no type -- which is why this case
			// names ToBFloat16 and not BFloat16.F32.
			k.Body = &ir.Block{List: []ir.Stmt{ir.NewExprStmt(k.Pos(),
				ir.NewIntrinsic(k.Pos(), &ir.Type{Kind: ir.BF16}, ir.OpF32ToBF16, nil,
					[]ir.Value{ir.NewConst(k.Pos(), &ir.Type{Kind: ir.F32},
						constant.MakeFloat64(1))}))}}
		},
		want: "bfloat",
	}, {
		name: "a subgroup ballot",
		mut: func(k *ir.Func) {
			k.Body = &ir.Block{List: []ir.Stmt{ir.NewExprStmt(k.Pos(),
				ir.NewIntrinsic(k.Pos(), &ir.Type{Kind: ir.U32}, ir.OpBallot, nil,
					[]ir.Value{ir.NewConst(k.Pos(), &ir.Type{Kind: ir.Bool},
						constant.MakeBool(true))}))}}
		},
		want: "Ballot",
	}, {
		name: "a helper taking a Thread",
		mut: func(k *ir.Func) {
			k.Helpers = []*ir.Func{{Name: "h", Thread: 0}}
		},
		want: "a helper taking a Thread",
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			k := corpusKernel(t, "Add")
			c.mut(k)
			_, err := emit.MSL(k)
			if err == nil {
				t.Fatalf("%s was lowered rather than refused", c.name)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("the refusal should mention %q, got %v", c.want, err)
			}
			// specs/004-kernel-authoring.md: a target-specific rejection names
			// its target, because a kernel legal on the CPU and not on Metal is
			// a different fact from an illegal one.
			if !strings.Contains(err.Error(), "MSL") {
				t.Errorf("the refusal should name the target: %v", err)
			}
		})
	}
}

// A helper is emitted static, ahead of its callers, and its read-only pointer
// parameters carry const.
//
// The const qualifier is the part worth asserting. A read-only binding is
// emitted as `const device T *`, and C refuses to pass one to a parameter that
// drops the qualifier -- so a helper whose parameter is wrongly mutable does not
// produce a subtly wrong kernel, it produces one that will not compile, and
// this is the cheaper place to find out.
func TestMSLHelperConstness(t *testing.T) {
	src, err := emit.MSL(corpusKernel(t, "SegmentSum"))
	if err != nil {
		t.Fatalf("MSL: %v", err)
	}
	if !strings.Contains(src, "static float accumulate(const device float *in") {
		t.Errorf("accumulate only reads its slice, so its parameter should be const:\n%s", src)
	}
	if strings.Index(src, "static ") > strings.Index(src, "kernel void") {
		t.Error("helpers must precede their callers, which C requires and Go does not")
	}
}

// Go's bitwise operators that C spells differently are translated, not passed
// through.
//
// ^x is exclusive-or in C, so emitting it verbatim is a syntax error at best
// and a wrong operation at worst; &^ has no C operator at all.
func TestMSLBitwiseOperatorsAreTranslated(t *testing.T) {
	k := corpusKernel(t, "Add")
	u32 := &ir.Type{Kind: ir.U32}
	x := ir.NewLocal(k.Pos(), u32, 0, "x", nil)
	k.Body = &ir.Block{List: []ir.Stmt{
		ir.NewDeclare(k.Pos(), x, ir.NewConst(k.Pos(), u32, constant.MakeInt64(3))),
		ir.NewAssign(k.Pos(), x, ir.NewUnary(k.Pos(), u32, token.XOR, x)),
		ir.NewAssign(k.Pos(), x, ir.NewBinary(k.Pos(), u32, token.AND_NOT, x, x)),
	}}
	src, err := emit.MSL(k)
	if err != nil {
		t.Fatalf("MSL: %v", err)
	}
	if !strings.Contains(src, "~x") {
		t.Errorf("Go's ^x is C's ~x:\n%s", src)
	}
	if !strings.Contains(src, "& ~") {
		t.Errorf("Go's &^ has no C operator and must be spelled out:\n%s", src)
	}
	if strings.Contains(src, "^x") {
		t.Errorf("a bare ^ reached the output, where C reads it as exclusive-or:\n%s", src)
	}
}

// A barrier lowers to the scope specs/002-compute-model.md §2.5 specifies.
//
// Asserted on the emitted text rather than on a result, and that is the whole
// point of the test. `t.Barrier()` is normative over shared *and* storage
// memory, and a lowering that names only threadgroup scope still produces the
// right answer for every kernel whose data fits in one workgroup — which is
// every kernel in this corpus. So a behavioural test cannot fail reliably here:
// Apple hardware may make the write visible anyway, and "may" is exactly what
// undefined behaviour looks like from the inside.
//
// §2.5's lowering table gives the target text for each backend, which makes
// this checkable rather than a matter of reading intent. This is the MSL row.
func TestABarrierLowersToTheScopeTheSpecStates(t *testing.T) {
	k := corpusKernel(t, "ReduceSum")
	src, err := emit.MSL(k)
	if err != nil {
		t.Fatalf("MSL: %v", err)
	}
	if !strings.Contains(src, "threadgroup_barrier(") {
		t.Fatalf("no barrier reached the output, so this test checks nothing:\n%s", src)
	}
	// Both classes. Order and spelling follow mem_flags, not the spec's prose.
	for _, want := range []string{"mem_flags::mem_threadgroup", "mem_flags::mem_device"} {
		if !strings.Contains(src, want) {
			t.Errorf("§2.5 makes Barrier order shared and storage, and the lowering "+
				"omits %s — a kernel publishing through a buffer across a barrier "+
				"is then undefined on this backend:\n%s", want, src)
		}
	}
}

// The f32 atomic lowers to atomic<float>, and a device without the capability
// refuses it at pipeline creation rather than here.
//
// Until 2026-09-02 the emitter refused it by name because the backend's
// capability table never asked the device; the table asks now, so the source
// is emitted for the devices that have it.
func TestTheF32AtomicLowersToAtomicFloat(t *testing.T) {
	k := corpusKernel(t, "Add")
	k.Body = &ir.Block{List: []ir.Stmt{ir.NewExprStmt(k.Pos(),
		ir.NewIntrinsic(k.Pos(), &ir.Type{Kind: ir.F32}, ir.OpAtomicAddF32,
			nil, []ir.Value{k.Params[1], ir.NewConst(k.Pos(), &ir.Type{Kind: ir.U32},
				constant.MakeInt64(0)), ir.NewConst(k.Pos(), &ir.Type{Kind: ir.F32}, constant.MakeFloat64(1))}))}}
	src, err := emit.MSL(k)
	if err != nil {
		t.Fatalf("the f32 atomic was refused: %v", err)
	}
	for _, want := range []string{"atomic<float>", "atomic_fetch_add_explicit"} {
		if !strings.Contains(src, want) {
			t.Errorf("the MSL does not contain %q:\n%s", want, src)
		}
	}
}
