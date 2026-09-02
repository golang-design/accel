// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package emit_test

import (
	"go/constant"
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/emit"
	"golang.design/x/accel/internal/kernelc/ir"
)

// A Go identifier that is an MSL keyword or type is renamed in the MSL.
//
// `half` and `float` are ordinary Go names and MSL types, and `new` is an
// ordinary Go name and a C++ keyword. The emitter printed every name verbatim,
// so `float half = ...` generated, passed the golden, and failed in the
// device compiler after the caller had been told the kernel was fine. A name
// beginning with an underscore is the emitter's own space and is renamed for
// the same reason.
func TestIdentifiersThatAreMSLWordsAreRenamed(t *testing.T) {
	f32 := &ir.Type{Kind: ir.F32}
	for _, name := range []string{"half", "float", "new", "this", "device", "_gid", "min"} {
		t.Run(name, func(t *testing.T) {
			l := ir.NewLocal(0, f32, 0, name, nil)
			k := kernelWith(ir.NewDeclare(0, l, ir.NewConst(0, f32, constant.MakeFloat64(1))))
			src, err := emit.MSL(k)
			if err != nil {
				t.Fatalf("emit: %v", err)
			}
			if !strings.Contains(src, "float "+name+"_ = ") {
				t.Fatalf("the local %q was not renamed:\n%s", name, src)
			}
			// And not renamed twice, or renamed where it is a type.
			if strings.Contains(src, name+"__") {
				t.Fatalf("the local %q was renamed twice:\n%s", name, src)
			}
		})
	}
}

// An ordinary name is printed as it is, so the generated text stays readable.
func TestOrdinaryIdentifiersAreNotRenamed(t *testing.T) {
	f32 := &ir.Type{Kind: ir.F32}
	l := ir.NewLocal(0, f32, 0, "acc", nil)
	k := kernelWith(ir.NewDeclare(0, l, ir.NewConst(0, f32, constant.MakeFloat64(1))))
	src, err := emit.MSL(k)
	if err != nil {
		t.Fatalf("emit: %v", err)
	}
	if !strings.Contains(src, "float acc = ") || strings.Contains(src, "acc_") {
		t.Fatalf("an ordinary local was renamed:\n%s", src)
	}
}

// A helper parameter that shares a name with an atomic binding is its own
// parameter, and is not emitted as an atomic.
//
// Atomic classification was keyed by name across the kernel and its helpers,
// so a helper `func f(counts []uint32)` next to a kernel whose `counts`
// binding is touched atomically had its parameter typed as a pointer to
// atomic_uint and its loads emitted through atomic_load_explicit, which does
// not compile against the plain pointer the call site passes.
func TestAHelperParameterNamedLikeAnAtomicBindingIsPlain(t *testing.T) {
	u32 := &ir.Type{Kind: ir.U32}
	slice := &ir.Type{Kind: ir.Slice, Elem: u32}
	thread := ir.NewParam(0, &ir.Type{Kind: ir.Struct, Name: "Thread"}, 0, "t", nil)
	counts := ir.NewParam(0, slice, 1, "counts", nil)
	zero := ir.NewConst(0, u32, constant.MakeUint64(0))

	// The helper reads counts[0] through a parameter of the same name.
	hp := ir.NewParam(0, slice, 0, "counts", nil)
	helper := &ir.Func{
		Name: "first", Thread: -1, Params: []*ir.Param{hp}, Result: u32,
		Body: &ir.Block{List: []ir.Stmt{
			ir.NewReturn(0, ir.NewIndex(0, u32, hp, zero, hp.Index)),
		}},
	}
	// The kernel adds to counts[0] atomically and calls the helper.
	k := &ir.Func{
		Name: "K", Stage: ir.StageCompute, Thread: 0, Workgroup: [3]uint32{64, 1, 1},
		Params:   []*ir.Param{thread, counts},
		Bindings: []*ir.Binding{{Name: "counts", Index: 1, Type: slice, Read: true, Write: true}},
		Helpers:  []*ir.Func{helper},
		Body: &ir.Block{List: []ir.Stmt{
			ir.NewExprStmt(0, ir.NewIntrinsic(0, u32, ir.OpAtomicAddU32, thread,
				[]ir.Value{counts, zero, ir.NewConst(0, u32, constant.MakeUint64(1))})),
			ir.NewExprStmt(0, ir.NewCall(0, u32, helper, []ir.Value{counts})),
		}},
	}
	src, err := emit.MSL(k)
	if err != nil {
		t.Fatalf("emit: %v", err)
	}
	if !strings.Contains(src, "atomic_uint *counts [[buffer(0)]]") {
		t.Fatalf("the kernel's binding is not atomic:\n%s", src)
	}
	if !strings.Contains(src, "static uint first(const device uint *counts)") ||
		strings.Contains(src, "atomic_load_explicit(&counts[uint(0)]") {
		t.Fatalf("the helper's parameter was classified by its name:\n%s", src)
	}
}
