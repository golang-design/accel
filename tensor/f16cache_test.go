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

// An f16 KV cache produces what an f32 one does, within the bound its storage
// implies.
//
// # Why this is worth halving
//
// The KV cache is the largest allocation a serving process has after the
// weights, and the only one that scales with both concurrency and context. A
// consumer measured 9.66 GB for a single 32k sequence on a 36-layer model —
// larger than the int8 weights of the model it serves.
//
// # Why narrow storage is defensible here and narrow accumulation is not
//
// specs/002-compute-model.md's default is right in general: a narrow type is
// storage that converts on load, and accumulating a long dot product in f16
// loses badly. That argument is about the *accumulator*. K and V are operands;
// the score accumulates in f32 whatever they are stored as, which is the trade
// `MatMul` already makes when it reads f16.
//
// # The bound, derived rather than tuned
//
// Each cache entry carries f16's relative error, u₁₆ = 2⁻¹¹. The score is a dot
// product of length d over one narrowed operand, so its relative error is about
// u₁₆ plus the f32 accumulation's γ(d), and the softmax and the V-weighted sum
// each pass that through. Taking three such terms and rounding up gives 2⁻⁸,
// which is what is asserted — never a figure a run produced.
func TestAnF16CacheAgreesWithAnF32One(t *testing.T) {
	const qHeads, kvHeads, headDim, capacity = 4, 2, 8, 6
	rt := newRuntime(t)
	d := rt.Device()

	scale := float32(1 / math.Sqrt(headDim))
	qs := make([]float32, qHeads*headDim)
	for i := range qs {
		qs[i] = float32(math.Sin(float64(i) * 0.3))
	}
	ks := make([]float32, capacity*kvHeads*headDim)
	vs := make([]float32, capacity*kvHeads*headDim)
	for i := range ks {
		ks[i] = float32(math.Cos(float64(i) * 0.17))
		vs[i] = float32(math.Sin(float64(i)*0.11)) * 2
	}

	run := func(t *testing.T, dt accel.DType) []float32 {
		t.Helper()
		b := rt.NewBuilder("cache")
		tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
		q := tensor.Input(b, tensor.ValueDesc{
			Name: "q", DType: accel.F32, Shape: tensor.Shape{qHeads, headDim},
		})
		kc := tensor.NewState(b, tensor.StateDesc{
			Name: "kcache", DType: dt, Shape: tensor.Shape{capacity, kvHeads, headDim},
		})
		vc := tensor.NewState(b, tensor.StateDesc{
			Name: "vcache", DType: dt, Shape: tensor.Shape{capacity, kvHeads, headDim},
		})
		lengths := tensor.Input(b, tensor.ValueDesc{
			Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
		})
		tensor.Output(b, "out", tensor.Attention(b, q, kc, vc, tensor.AttentionOptions{
			Lengths: lengths, ScaleName: "scale",
		}))

		plan, err := b.Compile(rt, tensor.CompileOptions{Label: "cache"})
		if err != nil {
			t.Fatalf("compile %v: %v", dt, err)
		}
		defer plan.Close()

		buf := func(name string, vals []float32) accel.BufferView {
			if dt == accel.F16 {
				return f16Buffer(t, d, name, vals)
			}
			return f32Buffer(t, d, name, vals)
		}
		out := f32Buffer(t, d, "out", make([]float32, qHeads*headDim))
		f := plan.Submit(d.Queue(), tensor.Bindings{
			Buffers: map[string]accel.BufferView{
				"q":      f32Buffer(t, d, "q", qs),
				"kcache": buf("kcache", ks), "vcache": buf("vcache", vs),
				"out": out,
				"len": u32Buffer(t, d, "len", []uint32{capacity}),
			},
			Scalars: map[string]tensor.ScalarValue{"scale": tensor.F32(scale)},
		})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit %v: %v", dt, err)
		}
		got := make([]float32, qHeads*headDim)
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}
		return got
	}

	wide, narrow := run(t, accel.F32), run(t, accel.F16)

	// Three terms of f16's relative error, rounded up.
	const bound = 3.0 / 2048

	var moved bool
	for i := range wide {
		if wide[i] != narrow[i] {
			moved = true
		}
		if d := math.Abs(float64(wide[i] - narrow[i])); d > bound {
			t.Errorf("element %d is %v with an f32 cache and %v with an f16 one, %v apart "+
				"against a derived bound of %v", i, wide[i], narrow[i], d, bound)
		}
	}
	// The two must not be bit-identical, or the narrow path is not narrow: an
	// f16 cache that round-tripped exactly would mean the values were being
	// stored wide somewhere.
	if !moved {
		t.Error("the f16 and f32 caches gave identical results, so the narrow cache is " +
			"not actually storing f16")
	}
}

// A prefill over a narrow cache is refused, naming what is missing.
//
// Only the decode kernel reads f16. Running a prefill against the f32 kernel
// would read every second entry of the cache, which produces a well-shaped
// tensor of the wrong numbers — so this is a refusal rather than a fallback.
func TestAPrefillOverAnF16CacheIsRefused(t *testing.T) {
	const qHeads, kvHeads, headDim, capacity, qSeq = 4, 2, 8, 6, 3
	rt := newRuntime(t)
	b := rt.NewBuilder("prefill16")
	tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
	tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarU32})

	q := tensor.Input(b, tensor.ValueDesc{
		Name: "q", DType: accel.F32, Shape: tensor.Shape{qSeq, qHeads, headDim},
	})
	kc := tensor.NewState(b, tensor.StateDesc{
		Name: "k", DType: accel.F16, Shape: tensor.Shape{capacity, kvHeads, headDim},
	})
	vc := tensor.NewState(b, tensor.StateDesc{
		Name: "v", DType: accel.F16, Shape: tensor.Shape{capacity, kvHeads, headDim},
	})
	tensor.Attention(b, q, kc, vc, tensor.AttentionOptions{
		Lengths: tensor.Input(b, tensor.ValueDesc{
			Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
		}),
		ScaleName: "scale", BaseName: "base",
	})

	err := b.Err()
	if err == nil {
		t.Fatal("a prefill over an f16 cache was accepted")
	}
	if !strings.Contains(err.Error(), "only the decode kernel reads a narrow cache") {
		t.Errorf("the refusal should name what is missing, got %v", err)
	}
}

// The two caches must agree on dtype, because one kernel reads both.
func TestTheTwoCachesMustShareADType(t *testing.T) {
	const kvHeads, headDim, capacity, qHeads = 2, 8, 6, 4
	rt := newRuntime(t)
	b := rt.NewBuilder("mixed")
	tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
	q := tensor.Input(b, tensor.ValueDesc{
		Name: "q", DType: accel.F32, Shape: tensor.Shape{qHeads, headDim},
	})
	kc := tensor.NewState(b, tensor.StateDesc{
		Name: "k", DType: accel.F16, Shape: tensor.Shape{capacity, kvHeads, headDim},
	})
	vc := tensor.NewState(b, tensor.StateDesc{
		Name: "v", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
	})
	tensor.Attention(b, q, kc, vc, tensor.AttentionOptions{
		Lengths: tensor.Input(b, tensor.ValueDesc{
			Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
		}),
		ScaleName: "scale",
	})
	if err := b.Err(); err == nil {
		t.Fatal("a mixed-dtype cache pair was accepted")
	} else if !strings.Contains(err.Error(), "same dtype") {
		t.Errorf("the refusal should say both caches share a dtype, got %v", err)
	}
}
