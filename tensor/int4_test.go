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

// Every plane's dtype and count is refused by name.
//
// These are the branches a caller reaches by building the triple themselves
// rather than taking what quant.Int4Quantize returned, and each names which
// plane is wrong -- a refusal that said only "the weight matrix is invalid"
// would leave three places to look.
func TestAnInt4MatrixRefusesEachWrongPlane(t *testing.T) {
	const K, N = 128, 2
	weights := K * N
	words := (weights + 7) / 8
	groups := (weights + quant.Int4Group - 1) / quant.Int4Group

	type planes struct{ codes, scales, zeros tensor.ValueDesc }
	good := planes{
		codes:  tensor.ValueDesc{Name: "codes", DType: accel.U32, Shape: tensor.Shape{words}},
		scales: tensor.ValueDesc{Name: "scales", DType: accel.F16, Shape: tensor.Shape{groups}},
		zeros:  tensor.ValueDesc{Name: "zeros", DType: accel.F16, Shape: tensor.Shape{groups}},
	}
	for _, c := range []struct {
		name string
		mut  func(*planes)
		want string
	}{
		{"codes of the wrong dtype", func(p *planes) { p.codes.DType = accel.F32 }, "must be u32"},
		{"scales of the wrong dtype", func(p *planes) { p.scales.DType = accel.F32 }, "scales are"},
		{"zeros of the wrong dtype", func(p *planes) { p.zeros.DType = accel.F32 }, "zeros are"},
		{"too few scales", func(p *planes) { p.scales.Shape = tensor.Shape{groups + 1} }, "scales"},
		{"too few zeros", func(p *planes) { p.zeros.Shape = tensor.Shape{groups + 1} }, "zeros"},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := good
			c.mut(&p)
			rt := newRuntime(t)
			b := rt.NewBuilder("refusal")
			a := tensor.Input(b, tensor.ValueDesc{
				Name: "a", DType: accel.F32, Shape: tensor.Shape{K},
			})
			tensor.Output(b, "out", tensor.Int4MatVec(b, a, tensor.Int4{
				Codes:   tensor.Weight(b, p.codes),
				Scales:  tensor.Weight(b, p.scales),
				Zeros:   tensor.Weight(b, p.zeros),
				Weights: weights,
			}))
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

// The activations a 4-bit matvec takes are refused when they are not a vector
// of the matrix's contraction width.
func TestAnInt4MatVecRefusesWrongActivations(t *testing.T) {
	const K, N = 128, 2
	weights := K * N
	words := (weights + 7) / 8
	groups := (weights + quant.Int4Group - 1) / quant.Int4Group

	for _, c := range []struct {
		name  string
		shape tensor.Shape
		dtype tensor.DType
		want  string
	}{
		{"f16 activations", tensor.Shape{K}, accel.F16, "reads f32"},
		{"a matrix of activations", tensor.Shape{2, K}, accel.F32, "takes a vector"},
		{"a width the matrix is not a multiple of", tensor.Shape{K + 1}, accel.F32, "multiple"},
	} {
		t.Run(c.name, func(t *testing.T) {
			rt := newRuntime(t)
			b := rt.NewBuilder("refusal")
			a := tensor.Input(b, tensor.ValueDesc{Name: "a", DType: c.dtype, Shape: c.shape})
			tensor.Output(b, "out", tensor.Int4MatVec(b, a, tensor.Int4{
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
			}))
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

// A 4-bit matmul through the public surface equals the matvec, row for row.
//
// specs/048-int4.md §5. The prefill shape against the decode shape over one
// weight matrix: the operators pick different kernels, and the answer is the
// same product either way. Compared against the matvec rather than against the
// exact product because §3's budget already covers the distance to exact, and
// what this asserts is that choosing the batched operator does not change the
// arithmetic a caller gets.
func TestAnInt4MatMulEqualsTheMatVecRowForRow(t *testing.T) {
	const K, N, M = 200, 20, 5
	rt := newRuntime(t)
	d := rt.Device()

	rng := rand.New(rand.NewPCG(53, 59))
	w := make([]float32, K*N)
	for i := range w {
		w[i] = rng.Float32()*2 - 1
	}
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(math.Sin(float64(i) * 0.23))
	}
	codes, scales, zeros := quant.Int4Quantize(w)

	run := func(label string, rows int, act []float32, batched bool) []float32 {
		t.Helper()
		b := rt.NewBuilder(label)
		shape := tensor.Shape{K}
		if batched {
			shape = tensor.Shape{rows, K}
		}
		av := tensor.Input(b, tensor.ValueDesc{
			Name: "a", DType: accel.F32, Shape: shape})
		cw := tensor.Weight(b, tensor.ValueDesc{
			Name: "codes", DType: accel.U32, Shape: tensor.Shape{len(codes)}})
		sw := tensor.Weight(b, tensor.ValueDesc{
			Name: "scales", DType: accel.F16, Shape: tensor.Shape{len(scales)}})
		zw := tensor.Weight(b, tensor.ValueDesc{
			Name: "zeros", DType: accel.F16, Shape: tensor.Shape{len(zeros)}})
		m := tensor.Int4{Codes: cw, Scales: sw, Zeros: zw, Weights: K * N}
		if batched {
			tensor.Output(b, "out", tensor.Int4MatMul(b, av, m))
		} else {
			tensor.Output(b, "out", tensor.Int4MatVec(b, av, m))
		}

		plan, err := b.Compile(rt, tensor.CompileOptions{Label: label})
		if err != nil {
			t.Fatalf("compile %s: %v", label, err)
		}
		defer plan.Close()

		out := f32Buffer(t, d, "out", make([]float32, rows*N))
		f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
			"a":      f32Buffer(t, d, "a", act),
			"codes":  u32Buffer(t, d, "codes", codes),
			"scales": f16Buffer(t, d, "scales", f32s(scales)),
			"zeros":  f16Buffer(t, d, "zeros", f32s(zeros)),
			"out":    out,
		}})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit %s: %v", label, err)
		}
		got := make([]float32, rows*N)
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("readback %s: %v", label, err)
		}
		return got
	}

	batched := run("int4mm", M, a, true)
	for m := range M {
		row := run("int4mv", 1, a[m*K:(m+1)*K], false)
		for n := range N {
			var mag float64
			for k := range K {
				mag += math.Abs(float64(a[m*K+k]) * float64(w[k*N+n]))
			}
			// The two sum in different orders, so specs/008-numerics.md §7's
			// reduction bound rather than equality.
			if e := math.Abs(float64(batched[m*N+n] - row[n])); e > mag*1e-6 {
				t.Fatalf("row %d column %d: batched %v, matvec %v, off by %v where "+
					"the reduction budget is %v", m, n, batched[m*N+n], row[n], e,
					mag*1e-6)
			}
		}
	}
}

// The batched operator refuses what the row operator takes, and says which.
func TestAnInt4MatMulRefusesAVector(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("int4mm")
	av := tensor.Input(b, tensor.ValueDesc{
		Name: "a", DType: accel.F32, Shape: tensor.Shape{64}})
	cw := tensor.Weight(b, tensor.ValueDesc{
		Name: "codes", DType: accel.U32, Shape: tensor.Shape{64 * 4 / 8}})
	sw := tensor.Weight(b, tensor.ValueDesc{
		Name: "scales", DType: accel.F16, Shape: tensor.Shape{2}})
	zw := tensor.Weight(b, tensor.ValueDesc{
		Name: "zeros", DType: accel.F16, Shape: tensor.Shape{2}})
	tensor.Int4MatMul(b, av, tensor.Int4{
		Codes: cw, Scales: sw, Zeros: zw, Weights: 64 * 4})

	err := b.Err()
	if err == nil {
		t.Fatal("a vector was accepted by the batched operator")
	}
	if !strings.Contains(err.Error(), "Int4MatVec") {
		t.Errorf("the refusal should name the operator that takes a vector, got %v", err)
	}
}
