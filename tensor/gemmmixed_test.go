// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// f32 activations multiply f16 weights, with no Cast and on the mixed kernel.
//
// # What this closes
//
// A transformer's two operands are never the same width and cannot be: the
// activations are f32 because every other operator in this package is, and the
// weights are f16 because a four billion parameter model is 16 GB in f32. So a
// rule that the two agree put a Cast in front of every projection -- four per
// layer, 144 dispatches per forward pass at 36 layers, each a full pass over
// the activations existing only to satisfy the check (accel issue 14).
//
// # Why the kernel name is asserted and not only the numbers
//
// Because relaxing the refusal without keying the *selection* on the operand
// pair sends an f16 weight to the f32 GEMM's f32 binding: a buffer of the wrong
// element size read as the right one, which is not a diagnostic at any layer
// below this, and produces a plausible matrix. The numbers alone would very
// nearly pass -- the first half of each weight row is read as the low mantissa
// bits of the wrong f32 -- so the assertion that catches it is the name.
func TestF32ActivationsMultiplyF16Weights(t *testing.T) {
	const m, k, n = 5, 12, 7
	rt := newRuntime(t)
	d := rt.Device()

	xs := make([]float32, m*k)
	ws := make([]float32, k*n)
	for i := range xs {
		xs[i] = float32(math.Sin(float64(i)*0.23)) * 2
	}
	for i := range ws {
		ws[i] = float32(math.Cos(float64(i)*0.31)) * 2
	}

	b := rt.NewBuilder("mixedgemm")
	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F32, Shape: tensor.Shape{m, k},
	})
	w := tensor.Weight(b, tensor.ValueDesc{
		Name: "w", DType: accel.F16, Shape: tensor.Shape{k, n},
	})
	tensor.Output(b, "out", tensor.MatMul(b, x, w))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "mixedgemm"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	sel := plan.Selections()
	if len(sel) != 1 || sel[0].Kernel != "MatMulTiledF32F16" {
		t.Fatalf("selections are %+v, want the mixed tiled GEMM; a selection keyed on "+
			"the activation's width alone would bind an f16 weight to an f32 binding "+
			"and report the f32 kernel here", sel)
	}
	if !strings.Contains(sel[0].Reason, "f32 activations against f16 weights") {
		t.Errorf("the selection does not say which operand is which: %q", sel[0].Reason)
	}
	for _, s := range sel {
		if strings.Contains(s.Op, "Cast") {
			t.Errorf("the plan contains %s; the mixed kernel exists so that it does not", s.Op)
		}
	}

	out := f32Buffer(t, d, "out", make([]float32, m*n))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f32Buffer(t, d, "x", xs), "w": f16Buffer(t, d, "w", ws), "out": out,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, m*n)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}

	// The reference is computed from the *f16-rounded* weights, not the values
	// written. The kernel reads what the buffer holds, so the weight's rounding
	// is the test's input rather than the kernel's error, and referencing the
	// originals would force a term this comparison does not need to carry.
	//
	// What is left is the f32 accumulation over K terms, which
	// specs/008-numerics.md section 7 bounds at Ku/(1-Ku) of the sum of
	// magnitudes, u = 2^-24.
	const u = 1.0 / (1 << 24)
	for row := range m {
		for col := range n {
			var exact, magnitude float64
			for kk := range k {
				a := float64(xs[row*k+kk])
				wv := float64(accel.ToFloat16(ws[kk*n+col]).F32())
				exact += a * wv
				magnitude += math.Abs(a * wv)
			}
			budget := (k * u) / (1 - k*u) * magnitude
			if diff := math.Abs(float64(got[row*n+col]) - exact); diff > budget {
				t.Fatalf("(%d,%d) is %v against an exact %v: off by %v, budget %v",
					row, col, got[row*n+col], exact, diff, budget)
			}
		}
	}
}

// A weight wider than its activations is still refused.
//
// This is the direction the same-dtype rule was always right about, and the one
// the relaxation must not take with it. An activation is produced by the graph
// and a weight is loaded from a file: the graph's width is the one that may be
// wider, because the weight's width is a memory decision and widening it is
// that decision made in the expensive direction. `MatMul(b, x, w)` says which
// operand is which by position, so the rule needs no annotation to state.
func TestAWeightWiderThanItsActivationsIsRefused(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("wideweight")
	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F16, Shape: tensor.Shape{4, 8},
	})
	w := tensor.Weight(b, tensor.ValueDesc{
		Name: "w", DType: accel.F32, Shape: tensor.Shape{8, 4},
	})
	tensor.MatMul(b, x, w)
	if err := b.Err(); err == nil {
		t.Fatal("an f32 weight against f16 activations was accepted")
	} else if !strings.Contains(err.Error(), "wider than the activation") {
		t.Errorf("the refusal should name the direction, got %v", err)
	}
}

// And a width with no GEMM at all names what is registered.
func TestAMatMulOverAnUnregisteredWidthIsRefused(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("i8gemm")
	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F32, Shape: tensor.Shape{4, 8},
	})
	w := tensor.Weight(b, tensor.ValueDesc{
		Name: "w", DType: accel.I8, Shape: tensor.Shape{8, 4},
	})
	tensor.MatMul(b, x, w)
	if err := b.Err(); err == nil {
		t.Fatal("an i8 weight was accepted by MatMul")
	} else if !strings.Contains(err.Error(), "f32 activations times f16 weights") {
		t.Errorf("the refusal should list the registered pairs, got %v", err)
	}
}

// The mixed pair takes the tile at M=1, and Selections says what that costs.
//
// The matrix-vector kernel reads f16 on both operands, so a mixed decode has no
// specialized kernel. That is a real gap and it is reported rather than hidden:
// a caller reading Selections sees that seven of eight tile rows are idle,
// which is the number they would need to decide whether it matters.
func TestAMixedMatMulAtOneRowReportsTheIdleTileRows(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("mixeddecode")
	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F32, Shape: tensor.Shape{1, 8},
	})
	w := tensor.Weight(b, tensor.ValueDesc{
		Name: "w", DType: accel.F16, Shape: tensor.Shape{8, 4},
	})
	tensor.Output(b, "out", tensor.MatMul(b, x, w))
	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "mixeddecode"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	sel := plan.Selections()
	if len(sel) != 1 || sel[0].Kernel != "MatMulTiledF32F16" {
		t.Fatalf("selections are %+v, want the mixed tiled GEMM", sel)
	}
	var named bool
	for _, r := range sel[0].Rejected {
		if strings.Contains(r, "rows are idle") {
			named = true
		}
	}
	if !named {
		t.Errorf("the selection does not report the idle tile rows, so the cost of "+
			"having no mixed matrix-vector kernel is invisible: %+v", sel[0])
	}
}
