// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"
	"golang.design/x/accel/tensor"
)

// f32s widens a scale plane so the f16 buffer helper can take it.
//
// quant returns accel.Float16 because that is the storage; the helper takes
// f32 and rounds, which is the same value here because these came from f16 in
// the first place.
func f32s(v []accel.Float16) []float32 {
	out := make([]float32, len(v))
	for i := range v {
		out[i] = v[i].F32()
	}
	return out
}

// A 4-bit matvec through the public surface lands within its derived budget.
//
// The end-to-end shape a caller writes: quantize on the host, bind three
// planes, read one vector back. Compared against the *unquantized* product,
// because specs/048-int4.md §3's bound is stated about that distance rather
// than about agreement between two spellings of the same approximation.
func TestAnInt4MatVecIsWithinItsBudget(t *testing.T) {
	const K, N = 256, 3
	rt := newRuntime(t)
	d := rt.Device()

	rng := rand.New(rand.NewPCG(41, 43))
	w := make([]float32, K*N)
	for i := range w {
		w[i] = rng.Float32()*2 - 1
	}
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(math.Sin(float64(i) * 0.29))
	}
	codes, scales, zeros := quant.Int4Quantize(w)

	b := rt.NewBuilder("int4")
	av := tensor.Input(b, tensor.ValueDesc{
		Name: "a", DType: accel.F32, Shape: tensor.Shape{K},
	})
	cw := tensor.Weight(b, tensor.ValueDesc{
		Name: "codes", DType: accel.U32, Shape: tensor.Shape{len(codes)},
	})
	sw := tensor.Weight(b, tensor.ValueDesc{
		Name: "scales", DType: accel.F16, Shape: tensor.Shape{len(scales)},
	})
	zw := tensor.Weight(b, tensor.ValueDesc{
		Name: "zeros", DType: accel.F16, Shape: tensor.Shape{len(zeros)},
	})
	tensor.Output(b, "out", tensor.Int4MatVec(b, av, tensor.Int4{
		Codes: cw, Scales: sw, Zeros: zw, Weights: K * N,
	}))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "int4"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	out := f32Buffer(t, d, "out", make([]float32, N))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"a":      f32Buffer(t, d, "a", a),
		"codes":  u32Buffer(t, d, "codes", codes),
		"scales": f16Buffer(t, d, "scales", f32s(scales)),
		"zeros":  f16Buffer(t, d, "zeros", f32s(zeros)),
		"out":    out,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, N)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}

	for n := range N {
		var exact float64
		ranges := make([]float32, K)
		for k := range K {
			idx := k*N + n
			exact += float64(a[k]) * float64(w[idx])
			g := idx / quant.Int4Group
			lo := g * quant.Int4Group
			hi := min(lo+quant.Int4Group, len(w))
			lowest, highest := w[lo], w[lo]
			for _, v := range w[lo:hi] {
				lowest, highest = min(lowest, v), max(highest, v)
			}
			ranges[k] = highest - lowest
		}
		budget := quant.Int4ErrorBound(a, ranges)
		var mag float64
		for k := range K {
			mag += math.Abs(float64(a[k]) * float64(w[k*N+n]))
		}
		budget += mag * 1e-6

		if e := math.Abs(float64(got[n]) - exact); e > budget {
			t.Fatalf("column %d is %v against an exact %v, off by %v where the derived "+
				"budget is %v", n, got[n], exact, e, budget)
		}
	}
}

// Every plane a 4-bit matrix needs is refused when it is missing or wrong.
//
// The bundle exists so a caller cannot bind one matrix's codes against
// another's scales, and these are the cases where a caller built the triple by
// hand instead of taking what quant.Int4Quantize returned.
func TestAnInt4MatrixRefusesAMismatchedTriple(t *testing.T) {
	const K, N = 128, 2
	weights := K * N
	words := (weights + 7) / 8
	groups := (weights + quant.Int4Group - 1) / quant.Int4Group

	for _, c := range []struct {
		name string
		mut  func(*tensor.Int4)
		want string
	}{
		{"no zero plane", func(q *tensor.Int4) { q.Zeros = nil }, "not optional"},
		{"a weight count nobody declared", func(q *tensor.Int4) { q.Weights = 0 }, "cannot be derived"},
		{"a weight count that disagrees with the words",
			func(q *tensor.Int4) { q.Weights = weights + 64 }, "words"},
	} {
		t.Run(c.name, func(t *testing.T) {
			rt := newRuntime(t)
			b := rt.NewBuilder("refusal")
			a := tensor.Input(b, tensor.ValueDesc{
				Name: "a", DType: accel.F32, Shape: tensor.Shape{K},
			})
			q := tensor.Int4{
				Codes: tensor.Weight(b, tensor.ValueDesc{
					Name: "codes", DType: accel.U32, Shape: tensor.Shape{words},
				}),
				Scales: tensor.Weight(b, tensor.ValueDesc{
					Name: "scales", DType: accel.F16, Shape: tensor.Shape{groups},
				}),
				Zeros: tensor.Weight(b, tensor.ValueDesc{
					Name: "zeros", DType: accel.F16, Shape: tensor.Shape{groups},
				}),
				Weights: weights,
			}
			c.mut(&q)
			tensor.Output(b, "out", tensor.Int4MatVec(b, a, q))
			_, err := b.Compile(rt, tensor.CompileOptions{Label: "refusal"})
			if err == nil {
				t.Fatal("accepted")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Fatalf("refused with %q, which does not mention %q", err, c.want)
			}
		})
	}
}
