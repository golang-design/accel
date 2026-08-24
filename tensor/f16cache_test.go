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

// An f16 KV cache is written from inside the graph, then read.
//
// This is accel issue 13, and the reporter's diagnosis of why nothing here
// caught it is the reason this test is shaped the way it is:
// TestAnF16CacheAgreesWithAnF32One populates the cache **from the host**, which
// is a fine way to test the read path and exactly what hides the write path. A
// test can put bytes in a buffer; a model has to compute them on the device and
// write them from inside the graph.
//
// So this one writes only through kernels: scatter the prompt's key and value
// rows into an f16 cache, prefill over what was written, scatter one more
// position, and decode over the longer cache. The host supplies no cache bytes
// at all.
func TestAnF16CacheIsFilledAndReadInsideOneGraph(t *testing.T) {
	const qHeads, kvHeads, headDim, capacity = 4, 2, 8, 16
	const prompt, width = 5, kvHeads * headDim

	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("f16cycle")

	tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
	tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarU32})
	kc := tensor.NewState(b, tensor.StateDesc{
		Name: "kc", DType: accel.F16, Shape: tensor.Shape{capacity, kvHeads, headDim},
	})
	vc := tensor.NewState(b, tensor.StateDesc{
		Name: "vc", DType: accel.F16, Shape: tensor.Shape{capacity, kvHeads, headDim},
	})
	pk := tensor.Input(b, tensor.ValueDesc{
		Name: "pk", DType: accel.F16, Shape: tensor.Shape{prompt, width},
	})
	pv := tensor.Input(b, tensor.ValueDesc{
		Name: "pv", DType: accel.F16, Shape: tensor.Shape{prompt, width},
	})
	slots := tensor.Input(b, tensor.ValueDesc{
		Name: "slots", DType: accel.U32, Shape: tensor.Shape{prompt},
	})
	pq := tensor.Input(b, tensor.ValueDesc{
		Name: "pq", DType: accel.F32, Shape: tensor.Shape{prompt, qHeads, headDim},
	})
	lens := tensor.Input(b, tensor.ValueDesc{
		Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
	})

	k1 := tensor.ScatterRows(b, kc, pk, slots)
	v1 := tensor.ScatterRows(b, vc, pv, slots)
	tensor.Output(b, "prefill", tensor.Attention(b, pq, k1, v1,
		tensor.AttentionOptions{Lengths: lens, ScaleName: "scale", BaseName: "base"}))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "f16cycle"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	// Both halves must be the narrow kernels, or this passes for the wrong
	// reason.
	var sawScatter, sawPrefill int
	for _, s := range plan.Selections() {
		switch s.Kernel {
		case "ScatterRowsF16":
			sawScatter++
		case "AttentionPrefillF16":
			sawPrefill++
		}
	}
	if sawScatter != 2 || sawPrefill != 1 {
		t.Fatalf("selections are %+v, want two f16 scatters and one f16 prefill",
			plan.Selections())
	}

	ks := make([]float32, prompt*width)
	vs := make([]float32, prompt*width)
	qs := make([]float32, prompt*qHeads*headDim)
	for i := range ks {
		ks[i] = float32(math.Cos(float64(i) * 0.23))
		vs[i] = float32(math.Sin(float64(i) * 0.17))
	}
	for i := range qs {
		qs[i] = float32(math.Sin(float64(i) * 0.41))
	}
	ids := make([]uint32, prompt)
	for i := range ids {
		ids[i] = uint32(i)
	}
	scale := float32(1 / math.Sqrt(headDim))

	// The cache starts as zeros from the host and is never written by it.
	out := f32Buffer(t, d, "prefill", make([]float32, prompt*qHeads*headDim))
	f := plan.Submit(d.Queue(), tensor.Bindings{
		Buffers: map[string]accel.BufferView{
			"kc": f16Buffer(t, d, "kc", make([]float32, capacity*width)),
			"vc": f16Buffer(t, d, "vc", make([]float32, capacity*width)),
			"pk": f16Buffer(t, d, "pk", ks), "pv": f16Buffer(t, d, "pv", vs),
			"slots":   u32Buffer(t, d, "slots", ids),
			"pq":      f32Buffer(t, d, "pq", qs),
			"len":     u32Buffer(t, d, "len", []uint32{prompt}),
			"prefill": out,
		},
		Scalars: map[string]tensor.ScalarValue{
			"scale": tensor.F32(scale), "base": tensor.U32(0),
		},
	})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, prompt*qHeads*headDim)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}

	// The reference reads what the cache holds after narrowing, so this is not
	// charged for the f16 storage the caller chose.
	keys := make([][]float32, prompt)
	vals := make([][]float32, prompt)
	for j := range prompt {
		kr := make([]float32, width)
		vr := make([]float32, width)
		for i := range width {
			kr[i] = accel.ToFloat16(ks[j*width+i]).F32()
			vr[i] = accel.ToFloat16(vs[j*width+i]).F32()
		}
		keys[j], vals[j] = kr, vr
	}
	var nonzero bool
	for s := range prompt {
		want := attentionReference(qs[s*qHeads*headDim:(s+1)*qHeads*headDim],
			keys[:s+1], vals[:s+1], qHeads, kvHeads, headDim, float64(scale))
		for i := range qHeads * headDim {
			g := float64(got[s*qHeads*headDim+i])
			if g != 0 {
				nonzero = true
			}
			if math.Abs(g-want[i]) > 1e-4*(1+math.Abs(want[i])) {
				t.Fatalf("query %d element %d is %v, want about %v", s, i, g, want[i])
			}
		}
	}
	// A cache that was never written would attend over zeros and agree with a
	// reference built the same way, so the answer being non-zero is what makes
	// the comparison mean anything.
	if !nonzero {
		t.Fatal("every output is zero, so nothing was written to the cache")
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

// Paging and a narrow cache compose, which is the other half of accel issue 13:
// the two allocations the memory argument is about were each available and
// could not be chosen together.
func TestAPagedDecodeReadsAnF16Cache(t *testing.T) {
	const qHeads, kvHeads, headDim, block = 4, 2, 8, 4
	const poolBlocks, pageCount, kvLen = 10, 3, 9
	const width = kvHeads * headDim

	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("pagedf16")

	tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
	q := tensor.Input(b, tensor.ValueDesc{
		Name: "q", DType: accel.F32, Shape: tensor.Shape{qHeads, headDim},
	})
	pages := tensor.Input(b, tensor.ValueDesc{
		Name: "pages", DType: accel.U32, Shape: tensor.Shape{pageCount},
	})
	lens := tensor.Input(b, tensor.ValueDesc{
		Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
	})
	kc := tensor.NewState(b, tensor.StateDesc{
		Name: "kc", DType: accel.F16,
		Shape: tensor.Shape{poolBlocks * block, kvHeads, headDim},
	})
	vc := tensor.NewState(b, tensor.StateDesc{
		Name: "vc", DType: accel.F16,
		Shape: tensor.Shape{poolBlocks * block, kvHeads, headDim},
	})
	tensor.Output(b, "out", tensor.Attention(b, q, kc, vc, tensor.AttentionOptions{
		Lengths: lens, Pages: pages, Block: block, ScaleName: "scale",
	}))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "pagedf16"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()
	sel := plan.Selections()
	if len(sel) != 1 || sel[0].Kernel != "AttentionDecodePagedF16" {
		t.Fatalf("selections are %+v, want the paged f16 decode kernel", sel)
	}

	ks := make([]float32, poolBlocks*block*width)
	vs := make([]float32, len(ks))
	for i := range ks {
		ks[i] = float32(math.Cos(float64(i) * 0.11))
		vs[i] = float32(math.Sin(float64(i) * 0.07))
	}
	qs := make([]float32, qHeads*headDim)
	for i := range qs {
		qs[i] = float32(math.Sin(float64(i) * 0.53))
	}
	table := []uint32{7, 1, 4} // scattered and out of order
	scale := float32(1 / math.Sqrt(headDim))

	out := f32Buffer(t, d, "out", make([]float32, qHeads*headDim))
	f := plan.Submit(d.Queue(), tensor.Bindings{
		Buffers: map[string]accel.BufferView{
			"q": f32Buffer(t, d, "q", qs), "kc": f16Buffer(t, d, "kc", ks),
			"vc": f16Buffer(t, d, "vc", vs), "pages": u32Buffer(t, d, "pages", table),
			"len": u32Buffer(t, d, "len", []uint32{kvLen}), "out": out,
		},
		Scalars: map[string]tensor.ScalarValue{"scale": tensor.F32(scale)},
	})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, qHeads*headDim)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}

	keys := make([][]float32, kvLen)
	vals := make([][]float32, kvLen)
	for j := range kvLen {
		phys := int(table[j/block])*block + j%block
		kr := make([]float32, width)
		vr := make([]float32, width)
		for i := range width {
			kr[i] = accel.ToFloat16(ks[phys*width+i]).F32()
			vr[i] = accel.ToFloat16(vs[phys*width+i]).F32()
		}
		keys[j], vals[j] = kr, vr
	}
	want := attentionReference(qs, keys, vals, qHeads, kvHeads, headDim, float64(scale))
	for i := range got {
		if math.Abs(float64(got[i])-want[i]) > 1e-4*(1+math.Abs(want[i])) {
			t.Fatalf("element %d is %v, want about %v", i, got[i], want[i])
		}
	}
}
