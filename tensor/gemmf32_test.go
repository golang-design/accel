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

// An f32 MatMul computes what the f16 one does, and a graph built from f32
// activations contains no Cast.
//
// # What the casts cost
//
// A consumer reported it: every other operator in this package is f32, so a
// transformer casts before each projection and back after — seven per layer,
// each a full pass over the activations existing only to satisfy a dtype
// check. On a 36-layer model that is 252 extra dispatches per forward pass, on
// a workload that is memory-bound.
//
// The f16 form stays and is not a fallback: weights ship narrow, and reading
// them narrow is worth what it saves. What changed is that an f32 graph no
// longer has to become an f16 one and back in order to multiply.
func TestAnF32MatMulNeedsNoCast(t *testing.T) {
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

	b := rt.NewBuilder("f32gemm")
	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F32, Shape: tensor.Shape{m, k},
	})
	w := tensor.Weight(b, tensor.ValueDesc{
		Name: "w", DType: accel.F32, Shape: tensor.Shape{k, n},
	})
	tensor.Output(b, "out", tensor.MatMul(b, x, w))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "f32gemm"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	// No conversion anywhere in the plan, which is the point: the graph has as
	// many nodes as it has operations.
	for _, s := range plan.Selections() {
		if strings.Contains(s.Op, "Cast") {
			t.Errorf("the plan contains %s; an f32 graph should need no conversion to "+
				"multiply", s.Op)
		}
	}

	out := f32Buffer(t, d, "out", make([]float32, m*n))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f32Buffer(t, d, "x", xs), "w": f32Buffer(t, d, "w", ws), "out": out,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, m*n)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}

	// Against the product computed here in f64. The operands are exact in f32,
	// so the only error is the accumulation's: γ(k) over k terms, which for
	// k=12 is far inside this.
	for r := range m {
		for c := range n {
			var want float64
			for i := range k {
				want += float64(xs[r*k+i]) * float64(ws[i*n+c])
			}
			if diff := math.Abs(float64(got[r*n+c]) - want); diff > 1e-4 {
				t.Fatalf("(%d,%d) is %v, want about %v", r, c, got[r*n+c], want)
			}
		}
	}
}

// Operands of different widths are refused, because one kernel reads both.
func TestAMixedWidthMatMulIsRefused(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("mixed")
	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F32, Shape: tensor.Shape{4, 8},
	})
	w := tensor.Weight(b, tensor.ValueDesc{
		Name: "w", DType: accel.F16, Shape: tensor.Shape{8, 4},
	})
	tensor.MatMul(b, x, w)
	if err := b.Err(); err == nil {
		t.Fatal("operands of different widths were accepted")
	} else if !strings.Contains(err.Error(), "share a dtype") {
		t.Errorf("the refusal should say they share a dtype, got %v", err)
	}
}

// Linear stays f16 and says what to use instead.
//
// The composed form — MatMul then Add — takes f32 and is the reference the
// fused kernel is checked against, so it is the right answer rather than a
// second fused variant existing for one dtype.
func TestAnF32LinearPointsAtTheComposedForm(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("linear32")
	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F32, Shape: tensor.Shape{4, 8},
	})
	w := tensor.Weight(b, tensor.ValueDesc{
		Name: "w", DType: accel.F32, Shape: tensor.Shape{8, 4},
	})
	bias := tensor.Weight(b, tensor.ValueDesc{
		Name: "b", DType: accel.F32, Shape: tensor.Shape{4},
	})
	tensor.Linear(b, x, w, bias)
	if err := b.Err(); err == nil {
		t.Fatal("an f32 Linear was accepted")
	} else if !strings.Contains(err.Error(), "MatMul then Add") {
		t.Errorf("the refusal should name the composed form, got %v", err)
	}
}
