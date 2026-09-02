// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// rowLogits is three rows of a small vocabulary with distinct orders.
func rowLogits(vocab int) []float32 {
	out := make([]float32, 3*vocab)
	for r := 0; r < 3; r++ {
		for i := 0; i < vocab; i++ {
			out[r*vocab+i] = float32(math.Sin(float64(i*(r+1))*0.7))*3 + float32(i%5)*0.1
		}
	}
	return out
}

// A per-row policy samples each row as the shared policy would sample it
// alone, and a greedy row beside a stochastic one takes its argmax.
//
// specs/064-per-row-sampling.md. Each row of a batch of three carries its own
// temperature, k and p, and its token is compared with the token Sample
// produces for that row on its own under a SamplingOptions holding the same
// values and the same draw; row 1 is greedy and is compared with Argmax.
func TestARowPolicySamplesEachRowAsItsOwnPolicyWould(t *testing.T) {
	const vocab = 64
	rt := newRuntime(t)
	d := rt.Device()
	logits := rowLogits(vocab)
	draws := []float32{0.37, 0.5, 0.81}
	factors := []float32{1 / 0.8, 0, 1 / 1.5}
	ks := []uint32{4, 0, 0}
	ps := []float32{0, 0, 0.6}

	b := rt.NewBuilder("rows")
	x := tensor.Input(b, tensor.ValueDesc{Name: "x", DType: accel.F32, Shape: tensor.Shape{3, vocab}})
	dr := tensor.Input(b, tensor.ValueDesc{Name: "draws", DType: accel.F32, Shape: tensor.Shape{3}})
	f := tensor.Input(b, tensor.ValueDesc{Name: "factor", DType: accel.F32, Shape: tensor.Shape{3}})
	k := tensor.Input(b, tensor.ValueDesc{Name: "k", DType: accel.U32, Shape: tensor.Shape{3}})
	p := tensor.Input(b, tensor.ValueDesc{Name: "p", DType: accel.F32, Shape: tensor.Shape{3}})
	tensor.Output(b, "tok", tensor.SampleRows(b, x, dr, tensor.RowSampling{Factor: f, TopK: k, TopP: p}, "s"))
	plan, err := b.Compile(rt, tensor.CompileOptions{})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()
	tok := u32Buffer(t, d, "tok", make([]uint32, 3))
	err = plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f32Buffer(t, d, "x", logits), "draws": f32Buffer(t, d, "draws", draws),
		"factor": f32Buffer(t, d, "factor", factors), "k": u32Buffer(t, d, "k", ks),
		"p": f32Buffer(t, d, "p", ps), "tok": tok,
	}}).Wait()
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]uint32, 3)
	if err := d.Queue().ReadBuffer(tok.Buffer, 0, got); err != nil {
		t.Fatal(err)
	}

	// Each row alone under the shared-policy operator.
	for r := 0; r < 3; r++ {
		b := rt.NewBuilder("alone")
		x := tensor.Input(b, tensor.ValueDesc{Name: "x", DType: accel.F32, Shape: tensor.Shape{vocab}})
		buffers := map[string]accel.BufferView{"x": f32Buffer(t, d, "x", logits[r*vocab:(r+1)*vocab])}
		if factors[r] == 0 {
			tensor.Output(b, "tok", tensor.Argmax(b, x))
		} else {
			dr := tensor.Input(b, tensor.ValueDesc{Name: "draws", DType: accel.F32, Shape: tensor.Shape{1}})
			buffers["draws"] = f32Buffer(t, d, "draws", draws[r:r+1])
			tensor.Scalar(b, tensor.ScalarDesc{Name: "s.invT", Kind: tensor.ScalarF32})
			tensor.Output(b, "tok", tensor.Sample(b, x, dr, nil, nil, tensor.SamplingOptions{
				Temperature: 1 / factors[r], TopK: int(ks[r]), TopP: ps[r],
			}, "s"))
		}
		plan, err := b.Compile(rt, tensor.CompileOptions{})
		if err != nil {
			t.Fatalf("row %d compile: %v", r, err)
		}
		want := u32Buffer(t, d, "want", make([]uint32, 1))
		buffers["tok"] = want
		scalars := map[string]tensor.ScalarValue{}
		if factors[r] != 0 {
			scalars["s.invT"] = tensor.F32(factors[r])
		}
		if err := plan.Submit(d.Queue(), tensor.Bindings{Buffers: buffers, Scalars: scalars}).Wait(); err != nil {
			t.Fatalf("row %d submit: %v", r, err)
		}
		w := make([]uint32, 1)
		if err := d.Queue().ReadBuffer(want.Buffer, 0, w); err != nil {
			t.Fatal(err)
		}
		plan.Close()
		if got[r] != w[0] {
			t.Errorf("row %d: the per-row policy chose %d and the row alone chose %d", r, got[r], w[0])
		}
	}
}

// A bias moves a row's argmax to the biased id, and only that row's.
func TestARowBiasMovesOnlyItsRow(t *testing.T) {
	const vocab = 32
	rt := newRuntime(t)
	d := rt.Device()
	logits := rowLogits(vocab)
	b := rt.NewBuilder("bias")
	x := tensor.Input(b, tensor.ValueDesc{Name: "x", DType: accel.F32, Shape: tensor.Shape{3, vocab}})
	dr := tensor.Input(b, tensor.ValueDesc{Name: "draws", DType: accel.F32, Shape: tensor.Shape{3}})
	f := tensor.Input(b, tensor.ValueDesc{Name: "factor", DType: accel.F32, Shape: tensor.Shape{3}})
	ids := tensor.Input(b, tensor.ValueDesc{Name: "ids", DType: accel.U32, Shape: tensor.Shape{3, 2}})
	vals := tensor.Input(b, tensor.ValueDesc{Name: "vals", DType: accel.F32, Shape: tensor.Shape{3, 2}})
	tensor.Output(b, "tok", tensor.SampleRows(b, x, dr, tensor.RowSampling{Factor: f, Bias: &tensor.RowBias{IDs: ids, Values: vals}}, "s"))
	plan, err := b.Compile(rt, tensor.CompileOptions{})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()
	run := func(biasIDs []uint32, biasVals []float32) []uint32 {
		tok := u32Buffer(t, d, "tok", make([]uint32, 3))
		err := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
			"x": f32Buffer(t, d, "x", logits), "draws": f32Buffer(t, d, "draws", []float32{0.5, 0.5, 0.5}),
			"factor": f32Buffer(t, d, "factor", []float32{0, 0, 0}),
			"ids":    u32Buffer(t, d, "ids", biasIDs), "vals": f32Buffer(t, d, "vals", biasVals), "tok": tok,
		}}).Wait()
		if err != nil {
			t.Fatalf("submit: %v", err)
		}
		got := make([]uint32, 3)
		if err := d.Queue().ReadBuffer(tok.Buffer, 0, got); err != nil {
			t.Fatal(err)
		}
		return got
	}
	none := run([]uint32{vocab, vocab, vocab, vocab, vocab, vocab}, make([]float32, 6))
	biased := run([]uint32{vocab, vocab, 7, vocab, vocab, vocab}, []float32{0, 0, 100, 0, 0, 0})
	if biased[1] != 7 {
		t.Errorf("row 1 chose %d with a bias of 100 on id 7", biased[1])
	}
	if biased[0] != none[0] || biased[2] != none[2] {
		t.Errorf("rows 0 and 2 chose %v with row 1 biased and %v without", biased, none)
	}
}

// Per-row penalties reproduce the shared penalty on each row alone.
func TestRowPenaltiesMatchTheSharedPenaltyPerRow(t *testing.T) {
	const vocab, cap = 32, 8
	rt := newRuntime(t)
	d := rt.Device()
	logits := rowLogits(vocab)
	history := []uint32{3, 3, 9, 0, 0, 0, 0, 0, 5, 1, 1, 1, 0, 0, 0, 0, 20, 21, 22, 23, 24, 25, 26, 27}
	filled := []uint32{3, 4, 8}
	rep := []float32{1.5, 1, 2}
	pres := []float32{0, 0.5, 0}
	freq := []float32{0.1, 0, 0.3}

	b := rt.NewBuilder("pen")
	x := tensor.Input(b, tensor.ValueDesc{Name: "x", DType: accel.F32, Shape: tensor.Shape{3, vocab}})
	dr := tensor.Input(b, tensor.ValueDesc{Name: "draws", DType: accel.F32, Shape: tensor.Shape{3}})
	f := tensor.Input(b, tensor.ValueDesc{Name: "factor", DType: accel.F32, Shape: tensor.Shape{3}})
	hs := tensor.NewState(b, tensor.StateDesc{Name: "history", DType: accel.U32, Shape: tensor.Shape{3, cap}})
	cs := tensor.NewState(b, tensor.StateDesc{Name: "counts", DType: accel.U32, Shape: tensor.Shape{3, vocab}})
	in := func(name string, dt accel.DType) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: name, DType: dt, Shape: tensor.Shape{3}})
	}
	tensor.Output(b, "tok", tensor.SampleRows(b, x, dr, tensor.RowSampling{
		Factor: f, Penalties: &tensor.RowPenalties{
			History: hs, Counts: cs, Filled: in("filled", accel.U32),
			Repetition: in("rep", accel.F32), Presence: in("pres", accel.F32), Frequency: in("freq", accel.F32),
		},
	}, "s"))
	plan, err := b.Compile(rt, tensor.CompileOptions{})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()
	tok := u32Buffer(t, d, "tok", make([]uint32, 3))
	err = plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f32Buffer(t, d, "x", logits), "draws": f32Buffer(t, d, "draws", []float32{0, 0, 0}),
		"factor":  f32Buffer(t, d, "factor", []float32{0, 0, 0}),
		"history": u32Buffer(t, d, "history", history), "counts": u32Buffer(t, d, "counts", make([]uint32, 3*vocab)),
		"filled": u32Buffer(t, d, "filled", filled), "rep": f32Buffer(t, d, "rep", rep),
		"pres": f32Buffer(t, d, "pres", pres), "freq": f32Buffer(t, d, "freq", freq), "tok": tok,
	}}).Wait()
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]uint32, 3)
	if err := d.Queue().ReadBuffer(tok.Buffer, 0, got); err != nil {
		t.Fatal(err)
	}
	// The host's reading of the same penalty, then the argmax.
	for r := 0; r < 3; r++ {
		counts := make([]int, vocab)
		for i := 0; i < int(filled[r]); i++ {
			counts[history[r*cap+i]]++
		}
		best, bestV := 0, float32(math.Inf(-1))
		for i := 0; i < vocab; i++ {
			l := logits[r*vocab+i]
			if c := counts[i]; c > 0 {
				if rep[r] != 0 && rep[r] != 1 {
					if l > 0 {
						l /= rep[r]
					} else {
						l *= rep[r]
					}
				}
				l = l - pres[r] - freq[r]*float32(c)
			}
			if l > bestV {
				best, bestV = i, l
			}
		}
		if int(got[r]) != best {
			t.Errorf("row %d chose %d and the host's penalised argmax is %d", r, got[r], best)
		}
	}
}
