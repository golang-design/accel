// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"
	"golang.design/x/accel/tensor"
)

func i8Buffer(t *testing.T, d *accel.Device, label string, vals []int8) accel.BufferView {
	t.Helper()
	b, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.I8, Count: len(vals), Label: label,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
	})
	if err != nil {
		t.Fatalf("buffer %s: %v", label, err)
	}
	t.Cleanup(func() { _ = b.Close() })
	if err := d.Queue().WriteBuffer(b, 0, vals); err != nil {
		t.Fatalf("write %s: %v", label, err)
	}
	v, err := b.View(0, len(vals))
	if err != nil {
		t.Fatalf("view %s: %v", label, err)
	}
	return v
}

// A plan mixing quantized and unquantized matrices compiles and runs, and the
// quantized half lands within specs/027-quantization.md's budget.
//
// specs/027-quantization.md's fifth done criterion. Mixed on purpose: a model
// quantizes its large projections and leaves the small ones alone, so the
// interesting case is the two in one graph rather than either by itself.
func TestAPlanMixesQuantizedAndUnquantized(t *testing.T) {
	const m, k, n = 4, 64, 16

	acts := make([]float32, m*k)
	big := make([]float32, k*n)   // quantized
	small := make([]float32, k*n) // not
	for i := range acts {
		acts[i] = float32(math.Sin(float64(i)*0.19)) * 2
	}
	for i := range big {
		big[i] = float32(math.Cos(float64(i)*0.23)) * 1.5
		small[i] = float32(math.Sin(float64(i)*0.41)) * 0.5
	}
	bq, bs := quant.Int8Quantize(big)

	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("mixed")

	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F16, Shape: tensor.Shape{m, k},
	})
	wq := tensor.Weight(b, tensor.ValueDesc{
		Name: "wq", DType: accel.I8, Shape: tensor.Shape{k, n},
	})
	ws := tensor.Weight(b, tensor.ValueDesc{
		Name: "ws", DType: accel.F16, Shape: tensor.Shape{len(bs)},
	})
	wf := tensor.Weight(b, tensor.ValueDesc{
		Name: "wf", DType: accel.F16, Shape: tensor.Shape{k, n},
	})

	q := tensor.QuantMatMul(b, x, tensor.Quantized{Quants: wq, Scales: ws})
	u := tensor.MatMul(b, x, wf)
	tensor.Output(b, "q", q)
	tensor.Output(b, "u", u)
	// And the two summed, so the graph really is one plan rather than two
	// unrelated halves sharing a submission.
	tensor.Output(b, "sum", tensor.Add(b, q, u))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "mixed"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	var sawQuant, sawPlain bool
	for _, s := range plan.Selections() {
		switch s.Kernel {
		case "QuantMatMul":
			sawQuant = true
		case "MatMulTiled":
			sawPlain = true
		}
	}
	if !sawQuant || !sawPlain {
		t.Fatalf("the plan should report both kernels; selections: %+v", plan.Selections())
	}

	qOut := f32Buffer(t, d, "q", make([]float32, m*n))
	uOut := f32Buffer(t, d, "u", make([]float32, m*n))
	sOut := f32Buffer(t, d, "sum", make([]float32, m*n))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x":  f16Buffer(t, d, "x", acts),
		"wq": i8Buffer(t, d, "wq", bq),
		"ws": f16BitsBuffer(t, d, "ws", bs),
		"wf": f16Buffer(t, d, "wf", small),
		"q":  qOut, "u": uOut, "sum": sOut,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	gotQ := make([]float32, m*n)
	gotU := make([]float32, m*n)
	gotS := make([]float32, m*n)
	for _, r := range []struct {
		b   accel.BufferView
		out []float32
	}{{qOut, gotQ}, {uOut, gotU}, {sOut, gotS}} {
		if err := d.Queue().ReadBuffer(r.b.Buffer, 0, r.out); err != nil {
			t.Fatalf("readback: %v", err)
		}
	}

	for row := range m {
		for col := range n {
			i := row*n + col
			// The quantized half against the exact product over the ORIGINAL
			// weights, within 027's budget: quantization plus the f32
			// accumulation on top. Against the originals because that is the
			// question a caller has -- what did quantizing cost me.
			var exact, magnitude, qErr float64
			for kk := range k {
				a := float64(accel.ToFloat16(acts[row*k+kk]).F32())
				w := float64(big[kk*n+col])
				exact += a * w
				magnitude += math.Abs(a * w)
				s := float64(bs[(kk*n+col)/quant.Int8Block].F32())
				qErr += math.Abs(a) * s / 2
			}
			const u = 1.0 / (1 << 24)
			accErr := (float64(k-1) * u) / (1 - float64(k-1)*u) * magnitude
			if d := math.Abs(float64(gotQ[i]) - exact); d > qErr+accErr {
				t.Fatalf("quantized (%d,%d) is %v against an exact %v: off by %v, budget "+
					"%v + %v", row, col, gotQ[i], exact, d, qErr, accErr)
			}

			// And the sum really is the two, which is what makes this one plan
			// rather than two that happen to share a submission.
			if want := gotQ[i] + gotU[i]; gotS[i] != want {
				t.Fatalf("sum (%d,%d) is %v and its operands are %v and %v",
					row, col, gotS[i], gotQ[i], gotU[i])
			}
		}
	}
}

// f16BitsBuffer uploads Float16 values as the bit patterns an f16 buffer holds.
func f16BitsBuffer(t *testing.T, d *accel.Device, label string, vals []accel.Float16) accel.BufferView {
	t.Helper()
	bits := make([]uint16, len(vals))
	for i, v := range vals {
		bits[i] = v.Bits()
	}
	b, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.F16, Count: len(vals), Label: label,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
	})
	if err != nil {
		t.Fatalf("buffer %s: %v", label, err)
	}
	t.Cleanup(func() { _ = b.Close() })
	if err := d.Queue().WriteBuffer(b, 0, bits); err != nil {
		t.Fatalf("write %s: %v", label, err)
	}
	v, err := b.View(0, len(vals))
	if err != nil {
		t.Fatalf("view %s: %v", label, err)
	}
	return v
}

// The quantized operators refuse a pair that does not describe one matrix.
//
// Bundling quants and scales in one value is what stops a caller binding one
// matrix's quants against another's, which would compile, run, and produce
// noise. These are the checks for a pair built by hand.
func TestQuantizedRefusals(t *testing.T) {
	rt := newRuntime(t)
	i8 := func(b *tensor.Builder, n string, dims ...int) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: n, DType: accel.I8, Shape: dims})
	}
	f16 := func(b *tensor.Builder, n string, dims ...int) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: n, DType: accel.F16, Shape: dims})
	}
	u32 := func(b *tensor.Builder, n string, dims ...int) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: n, DType: accel.U32, Shape: dims})
	}

	for _, tc := range []struct {
		name  string
		build func(b *tensor.Builder)
		want  string
	}{{
		name: "a scale count that does not match the quants",
		build: func(b *tensor.Builder) {
			tensor.QuantMatMul(b, f16(b, "x", 2, 64), tensor.Quantized{
				Quants: i8(b, "q", 64, 4), Scales: f16(b, "s", 3),
			})
		},
		want: "blocks of 32",
	}, {
		name: "quants that are not i8",
		build: func(b *tensor.Builder) {
			tensor.QuantMatMul(b, f16(b, "x", 2, 64), tensor.Quantized{
				Quants: f16(b, "q", 64, 4), Scales: f16(b, "s", 8),
			})
		},
		want: "quants are f16",
	}, {
		name: "scales that are not f16",
		build: func(b *tensor.Builder) {
			s32 := tensor.Input(b, tensor.ValueDesc{
				Name: "s", DType: accel.F32, Shape: tensor.Shape{8},
			})
			tensor.QuantMatMul(b, f16(b, "x", 2, 64), tensor.Quantized{
				Quants: i8(b, "q", 64, 4), Scales: s32,
			})
		},
		want: "scales are f32",
	}, {
		name: "a missing plane",
		build: func(b *tensor.Builder) {
			tensor.QuantMatMul(b, f16(b, "x", 2, 64), tensor.Quantized{
				Quants: i8(b, "q", 64, 4),
			})
		},
		want: "both a quant plane and a scale plane",
	}, {
		name: "f32 activations",
		build: func(b *tensor.Builder) {
			x := tensor.Input(b, tensor.ValueDesc{
				Name: "x", DType: accel.F32, Shape: tensor.Shape{2, 64},
			})
			tensor.QuantMatMul(b, x, tensor.Quantized{
				Quants: i8(b, "q", 64, 4), Scales: f16(b, "s", 8),
			})
		},
		want: "the registered kernel reads",
	}, {
		name: "contracted axes that disagree",
		build: func(b *tensor.Builder) {
			tensor.QuantMatMul(b, f16(b, "x", 2, 32), tensor.Quantized{
				Quants: i8(b, "q", 64, 4), Scales: f16(b, "s", 8),
			})
		},
		want: "contracted axes",
	}, {
		name: "a quantized table that is not a matrix",
		build: func(b *tensor.Builder) {
			tensor.QuantGatherRows(b, tensor.Quantized{
				Quants: i8(b, "q", 64), Scales: f16(b, "s", 2),
			}, u32(b, "i", 2))
		},
		want: "[vocab, width]",
	}, {
		name: "ids that are not u32",
		build: func(b *tensor.Builder) {
			tensor.QuantGatherRows(b, tensor.Quantized{
				Quants: i8(b, "q", 8, 8), Scales: f16(b, "s", 2),
			}, f16(b, "i", 2))
		},
		want: "ids are f16",
	}} {
		t.Run(tc.name, func(t *testing.T) {
			b := rt.NewBuilder(tc.name)
			tc.build(b)
			err := b.Err()
			if err == nil {
				t.Fatalf("%s was accepted", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("the refusal should say %q, got %v", tc.want, err)
			}
		})
	}

	// A poisoned plane flows through without a second diagnostic.
	b := rt.NewBuilder("poison")
	bad := tensor.QuantMatMul(b, f16(b, "x", 2, 32), tensor.Quantized{
		Quants: i8(b, "q", 64, 4), Scales: f16(b, "s", 8),
	})
	before := b.Err().Error()
	tensor.QuantMatMul(b, bad, tensor.Quantized{Quants: bad, Scales: bad})
	tensor.QuantGatherRows(b, tensor.Quantized{Quants: bad, Scales: bad}, bad)
	if b.Err().Error() != before {
		t.Errorf("a quantized operator on a poisoned tensor recorded a diagnostic:\n%v",
			b.Err())
	}
}

// At M=1 the quantized path selects a matrix-vector kernel.
//
// accel issue 11: MatMul specialized M=1 and QuantMatMul had one kernel at
// every M, so a decode step -- M=1, every step after the first -- ran with no
// specialization on the path `auto` picks whenever f16 does not fit. The
// configuration a model lands in because it is large was the one with the
// fewest kernels.
//
// The budget is the one TestAPlanMixesQuantizedAndUnquantized derives, with the
// accumulation term over the reduction's depth rather than its length: this
// kernel folds K across the lanes and tree reduces, so the error grows with
// log2 of the fold rather than with K.
func TestQuantMatMulSelectsMatVecAtOneRow(t *testing.T) {
	const k, n = 64, 16

	acts := make([]float32, k)
	weights := make([]float32, k*n)
	for i := range acts {
		acts[i] = float32(math.Sin(float64(i)*0.31)) * 2
	}
	for i := range weights {
		weights[i] = float32(math.Cos(float64(i)*0.17)) * 1.5
	}
	wq, ws := quant.Int8Quantize(weights)

	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("qmatvec")

	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F16, Shape: tensor.Shape{1, k},
	})
	qw := tensor.Weight(b, tensor.ValueDesc{
		Name: "wq", DType: accel.I8, Shape: tensor.Shape{k, n},
	})
	sw := tensor.Weight(b, tensor.ValueDesc{
		Name: "ws", DType: accel.F16, Shape: tensor.Shape{len(ws)},
	})
	tensor.Output(b, "out", tensor.QuantMatMul(b, x, tensor.Quantized{Quants: qw, Scales: sw}))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "qmatvec"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	sel := plan.Selections()
	if len(sel) != 1 || sel[0].Kernel != "QuantMatVec" {
		t.Fatalf("selections are %+v, want the quantized matrix-vector kernel: a "+
			"selection nobody can see is a selection nobody can evaluate", sel)
	}
	// And it says what it turned down, so a reader can tell the choice was made
	// rather than defaulted.
	var namedGEMM bool
	for _, r := range sel[0].Rejected {
		if strings.Contains(r, "per-element quantized GEMM") {
			namedGEMM = true
		}
	}
	if !namedGEMM {
		t.Errorf("the selection does not say it rejected the per-element GEMM: %+v", sel[0])
	}

	out := f32Buffer(t, d, "out", make([]float32, n))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f16Buffer(t, d, "x", acts), "wq": i8Buffer(t, d, "wq", wq),
		"ws": f16BitsBuffer(t, d, "ws", ws), "out": out,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, n)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}

	for col := range n {
		var exact, magnitude, qErr float64
		for kk := range k {
			a := float64(accel.ToFloat16(acts[kk]).F32())
			w := float64(weights[kk*n+col])
			exact += a * w
			magnitude += math.Abs(a * w)
			s := float64(ws[(kk*n+col)/quant.Int8Block].F32())
			qErr += math.Abs(a) * s / 2
		}
		// Each lane folds ceil(K/128) products sequentially, then a tree of
		// depth log2(128) combines them.
		const u = 1.0 / (1 << 24)
		depth := math.Ceil(float64(k)/128) + 7
		accErr := (depth * u) / (1 - depth*u) * magnitude
		if diff := math.Abs(float64(got[col]) - exact); diff > qErr+accErr {
			t.Fatalf("column %d is %v against an exact %v: off by %v, budget %v + %v",
				col, got[col], exact, diff, qErr, accErr)
		}
	}
}
