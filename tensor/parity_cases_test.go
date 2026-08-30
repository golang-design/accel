// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/parity"
	"golang.design/x/accel/quant"
	"golang.design/x/accel/tensor"
)

// The cases specs/062-backend-parity.md section 6.6's gate enumerates against.
//
// Composed rather than one operator per plan, and that is deliberate: an
// operator alone is a shape the kernel corpus differential already compares.
// What these add is the operator *driving* a kernel -- the grid it computes,
// the view it materializes, the order it binds its operands in -- and a graph
// where one operator's result is another's operand is where that goes wrong.

// elementwiseParityCase is the arithmetic that has no view in it.
func elementwiseParityCase() tensorParityCase {
	const rows, width = 4, 64
	x := ramp(rows*width, 0)
	y := ramp(rows*width, 1.7)
	gain := ramp(width, 0.3)

	return tensorParityCase{
		name: "the elementwise operators and the norms",
		covers: parity.Covers{
			"Input", "Weight", "Output", "Scalar",
			"Add", "Mul", "Scale", "SiLU", "SwiGLU", "RMSNorm", "Softmax", "Cast",
		},
		// RMSNorm reaches a reciprocal square root over a sum, SiLU reaches
		// exp, and Softmax reaches exp and a division. specs/008-numerics.md
		// section 6 bounds each at 4 ULP; doubled for two implementations and
		// rounded the way specs/022-msl-target.md derives its ceilings.
		ceiling: parity.Ceiling{ULP: 16,
			Why: "RMSNorm's reciprocal square root, SiLU's exp, and Softmax's exp and division"},
		run: func(t *testing.T, d *accel.Device) []float32 {
			t.Helper()
			rt := parityRuntime(t, d)
			b := rt.NewBuilder("elementwise")
			tensor.Scalar(b, tensor.ScalarDesc{Name: "k", Kind: tensor.ScalarF32})

			xv := tensor.Input(b, tensor.ValueDesc{
				Name: "x", DType: accel.F32, Shape: tensor.Shape{rows, width}})
			yv := tensor.Input(b, tensor.ValueDesc{
				Name: "y", DType: accel.F32, Shape: tensor.Shape{rows, width}})
			gv := tensor.Weight(b, tensor.ValueDesc{
				Name: "gain", DType: accel.F32, Shape: tensor.Shape{width}})

			// A kernel's result as the next kernel's operand, four deep.
			sum := tensor.Add(b, xv, yv)
			scaled := tensor.Scale(b, sum, "k")
			normed := tensor.RMSNorm(b, scaled, gv, 1e-5)
			gated := tensor.SwiGLU(b, normed, tensor.SiLU(b, tensor.Mul(b, xv, yv)))
			// A narrowing round trip: the value goes to f16 and back, so the
			// comparison covers the conversion in both directions rather than
			// only the wider one.
			narrow := tensor.Cast(b, gated, accel.F16)
			wide := tensor.Cast(b, narrow, accel.F32)
			tensor.Output(b, "out", tensor.Softmax(b, wide, tensor.SoftmaxOptions{Axis: 1}))

			return runPlan(t, d, rt, b, "elementwise", "out", rows*width,
				tensor.Bindings{
					Buffers: map[string]accel.BufferView{
						"x":    f32Buffer(t, d, "x", x),
						"y":    f32Buffer(t, d, "y", y),
						"gain": f32Buffer(t, d, "gain", gain),
					},
					Scalars: map[string]tensor.ScalarValue{"k": tensor.F32(0.75)},
				})
		},
	}
}

// viewParityCase is the operators that move values without computing one, plus
// the two that index.
//
// A view feeding a kernel and a kernel feeding a view, in one graph. A backend
// that materialized the wrong stride would pass every kernel differential and
// fail here, because the corpus compares kernels and not the operators that
// select and drive them.
func viewParityCase() tensorParityCase {
	const rows, width, tableRows = 4, 32, 6
	table := ramp(tableRows*width, 0.11)
	bias := ramp(rows*width, 2.3)
	ids := []uint32{4, 0, 5, 1}
	pos := []uint32{0, 1, 2, 3}
	row := ramp(width, 0.9)

	return tensorParityCase{
		name: "the views, the gathers and the rotation",
		covers: parity.Covers{
			"Reshape", "Permute", "Transpose", "Slice", "Broadcast", "Contiguous",
			"GatherRows", "RoPE",
		},
		ceiling: parity.Ceiling{ULP: 16,
			Why: "RoPE's sine and cosine; the views and the gather are exact and add nothing"},
		run: func(t *testing.T, d *accel.Device) []float32 {
			t.Helper()
			rt := parityRuntime(t, d)
			b := rt.NewBuilder("views")
			tensor.Scalar(b, tensor.ScalarDesc{Name: "theta", Kind: tensor.ScalarF32})

			tbl := tensor.Weight(b, tensor.ValueDesc{
				Name: "table", DType: accel.F32, Shape: tensor.Shape{tableRows, width}})
			idv := tensor.Input(b, tensor.ValueDesc{
				Name: "ids", DType: accel.U32, Shape: tensor.Shape{rows}})
			posv := tensor.Input(b, tensor.ValueDesc{
				Name: "pos", DType: accel.U32, Shape: tensor.Shape{rows}})
			bv := tensor.Input(b, tensor.ValueDesc{
				Name: "bias", DType: accel.F32, Shape: tensor.Shape{rows, width}})
			rv := tensor.Input(b, tensor.ValueDesc{
				Name: "row", DType: accel.F32, Shape: tensor.Shape{1, width}})

			// A gather feeding an add, and a broadcast of one row feeding the
			// same add: a contiguous run repeated whole, which is the shape
			// this version materializes.
			g := tensor.Add(b, tensor.GatherRows(b, tbl, idv), bv)
			g = tensor.Add(b, g, tensor.Contiguous(b,
				tensor.Broadcast(b, rv, tensor.Shape{rows, width})))

			// Reshaped into the [tokens, heads, dim] RoPE takes -- one head,
			// because a position is per token and the operator refuses a
			// position count that does not match the rows -- rotated, then
			// permuted and packed back down.
			r := tensor.Reshape(b, g, tensor.Shape{rows, 1, width})
			r = tensor.RoPE(b, r, width/2, "theta", posv)
			r = tensor.Contiguous(b, tensor.Permute(b, r, 1, 0, 2))
			r = tensor.Reshape(b, r, tensor.Shape{rows, width})

			// A transposed operand materialized explicitly, then sliced: the
			// boundary this layer draws is that a copy of a matrix is asked
			// for and never implied.
			tr := tensor.Contiguous(b, tensor.Transpose(b, r, 0, 1))
			tensor.Output(b, "out", tensor.Contiguous(b,
				tensor.Slice(b, tr, 1, 1, rows)))

			n := width * (rows - 1)
			return runPlan(t, d, rt, b, "views", "out", n, tensor.Bindings{
				Buffers: map[string]accel.BufferView{
					"table": f32Buffer(t, d, "table", table),
					"ids":   u32Buffer(t, d, "ids", ids),
					"pos":   u32Buffer(t, d, "pos", pos),
					"bias":  f32Buffer(t, d, "bias", bias),
					"row":   f32Buffer(t, d, "row", row),
				},
				Scalars: map[string]tensor.ScalarValue{"theta": tensor.F32(10000)},
			})
		},
	}
}

// matmulParityCase covers the dense products.
func matmulParityCase() tensorParityCase {
	const m, k, n = 4, 64, 8
	x := ramp(m*k, 0)
	w := ramp(k*n, 1.1)
	bias := ramp(n, 0.4)

	return tensorParityCase{
		name:   "a dense product and a linear layer over it",
		covers: parity.Covers{"MatMul", "Linear"},
		// A dot product of length k, whose reduction order differs between a
		// tiled kernel and a flat one. specs/008-numerics.md section 7 bounds a
		// reduction rather than forbidding a different order.
		ceiling: parity.Ceiling{Abs: 1e-4,
			Why: "the reduction order of a length-64 dot product, which the two backends tile differently"},
		run: func(t *testing.T, d *accel.Device) []float32 {
			t.Helper()
			rt := parityRuntime(t, d)
			b := rt.NewBuilder("matmul")
			xv := tensor.Input(b, tensor.ValueDesc{
				Name: "x", DType: accel.F32, Shape: tensor.Shape{m, k}})
			wv := tensor.Weight(b, tensor.ValueDesc{
				Name: "w", DType: accel.F32, Shape: tensor.Shape{k, n}})
			w2 := tensor.Weight(b, tensor.ValueDesc{
				Name: "w2", DType: accel.F16, Shape: tensor.Shape{n, n}})
			bs := tensor.Weight(b, tensor.ValueDesc{
				Name: "bias", DType: accel.F32, Shape: tensor.Shape{n}})

			// The product feeds the linear layer, so a wrong grid in either is
			// visible in one output rather than needing two plans. Linear's
			// fused epilogue reads f16 on both operands and adds the bias in
			// f32, so the product is narrowed into it and widened back out.
			mm := tensor.Cast(b, tensor.MatMul(b, xv, wv), accel.F16)
			tensor.Output(b, "out",
				tensor.Cast(b, tensor.Linear(b, mm, w2, bs), accel.F32))

			return runPlan(t, d, rt, b, "matmul", "out", m*n, tensor.Bindings{
				Buffers: map[string]accel.BufferView{
					"x": f32Buffer(t, d, "x", x), "w": f32Buffer(t, d, "w", w),
					"w2":   f16Buffer(t, d, "w2", ramp(n*n, 2.2)),
					"bias": f32Buffer(t, d, "bias", bias),
				},
			})
		},
	}
}

// int4ParityCase covers the 4-bit weight path in both its shapes.
//
// The two spellings in one plan: the matrix-vector form and the matrix-matrix
// form over the same weights, summed. They select different kernels, so a
// disagreement in either is visible, and summing them means one output rather
// than two plans over the same quantisation.
func int4ParityCase() tensorParityCase {
	const K, N = 128, 4
	rng := rand.New(rand.NewPCG(41, 43))
	w := make([]float32, K*N)
	for i := range w {
		w[i] = rng.Float32()*2 - 1
	}
	codes, scales, zeros := quant.Int4Quantize(w)
	a := ramp(K, 0.29)

	return tensorParityCase{
		name:   "a 4-bit matrix, as a vector product and as a matrix product",
		covers: parity.Covers{"Int4MatVec", "Int4MatMul"},
		ceiling: parity.Ceiling{Abs: 1e-3,
			Why: "the reduction order of a length-128 4-bit product, tiled differently on the two backends"},
		run: func(t *testing.T, d *accel.Device) []float32 {
			t.Helper()
			rt := parityRuntime(t, d)
			b := rt.NewBuilder("int4")
			av := tensor.Input(b, tensor.ValueDesc{
				Name: "a", DType: accel.F32, Shape: tensor.Shape{K}})
			mv := tensor.Input(b, tensor.ValueDesc{
				Name: "m", DType: accel.F32, Shape: tensor.Shape{2, K}})
			cw := tensor.Weight(b, tensor.ValueDesc{
				Name: "codes", DType: accel.U32, Shape: tensor.Shape{len(codes)}})
			sw := tensor.Weight(b, tensor.ValueDesc{
				Name: "scales", DType: accel.F16, Shape: tensor.Shape{len(scales)}})
			zw := tensor.Weight(b, tensor.ValueDesc{
				Name: "zeros", DType: accel.F16, Shape: tensor.Shape{len(zeros)}})

			mat := tensor.Int4{Codes: cw, Scales: sw, Zeros: zw, Weights: K * N}
			vec := tensor.Reshape(b, tensor.Int4MatVec(b, av, mat), tensor.Shape{1, N})
			mm := tensor.Int4MatMul(b, mv, mat)
			tensor.Output(b, "out", tensor.Add(b, tensor.Contiguous(b,
				tensor.Broadcast(b, vec, tensor.Shape{2, N})), mm))

			m := make([]float32, 2*K)
			copy(m, a)
			copy(m[K:], ramp(K, 1.4))
			return runPlan(t, d, rt, b, "int4", "out", 2*N, tensor.Bindings{
				Buffers: map[string]accel.BufferView{
					"a":      f32Buffer(t, d, "a", a),
					"m":      f32Buffer(t, d, "m", m),
					"codes":  u32Buffer(t, d, "codes", codes),
					"scales": f16Buffer(t, d, "scales", f32s(scales)),
					"zeros":  f16Buffer(t, d, "zeros", f32s(zeros)),
				},
			})
		},
	}
}

// quantParityCase covers the 8-bit weight path and the gather over it.
func quantParityCase() tensorParityCase {
	const m, k, n = 2, 64, 4
	const tableRows = 6

	quants := make([]int8, k*n)
	for i := range quants {
		quants[i] = int8(i%251) - 125
	}
	scales := make([]float32, k*n/quant.Int8Block)
	for i := range scales {
		scales[i] = 0.01 + float32(i%7)*0.003
	}
	tableQuants := make([]int8, tableRows*k)
	for i := range tableQuants {
		tableQuants[i] = int8(i%239) - 119
	}
	tableScales := make([]float32, tableRows*k/quant.Int8Block)
	for i := range tableScales {
		tableScales[i] = 0.02 + float32(i%5)*0.004
	}

	return tensorParityCase{
		name:   "an 8-bit matrix and an 8-bit embedding table",
		covers: parity.Covers{"QuantMatMul", "QuantGatherRows"},
		ceiling: parity.Ceiling{Abs: 1e-2,
			Why: "the reduction order of a length-64 8-bit product, and the f16 accumulation both backends use"},
		run: func(t *testing.T, d *accel.Device) []float32 {
			t.Helper()
			rt := parityRuntime(t, d)
			b := rt.NewBuilder("quant")
			x := tensor.Input(b, tensor.ValueDesc{
				Name: "x", DType: accel.F16, Shape: tensor.Shape{m, k}})
			wq := tensor.Weight(b, tensor.ValueDesc{
				Name: "wq", DType: accel.I8, Shape: tensor.Shape{k, n}})
			ws := tensor.Weight(b, tensor.ValueDesc{
				Name: "ws", DType: accel.F16, Shape: tensor.Shape{len(scales)}})
			tq := tensor.Weight(b, tensor.ValueDesc{
				Name: "tq", DType: accel.I8, Shape: tensor.Shape{tableRows, k}})
			ts := tensor.Weight(b, tensor.ValueDesc{
				Name: "ts", DType: accel.F16, Shape: tensor.Shape{len(tableScales)}})
			ids := tensor.Input(b, tensor.ValueDesc{
				Name: "ids", DType: accel.U32, Shape: tensor.Shape{m}})

			// The gather's rows are the product's activations, so the two are
			// one graph and a wrong row is a wrong product rather than a second
			// assertion.
			rows := tensor.QuantGatherRows(b,
				tensor.Quantized{Quants: tq, Scales: ts}, ids)
			p1 := tensor.QuantMatMul(b, x, tensor.Quantized{Quants: wq, Scales: ws})
			p2 := tensor.QuantMatMul(b, tensor.Cast(b, rows, accel.F16),
				tensor.Quantized{Quants: wq, Scales: ws})
			tensor.Output(b, "out", tensor.Add(b, p1, p2))

			return runPlan(t, d, rt, b, "quant", "out", m*n, tensor.Bindings{
				Buffers: map[string]accel.BufferView{
					"x":   f16Buffer(t, d, "x", ramp(m*k, 0.5)),
					"wq":  i8Buffer(t, d, "wq", quants),
					"ws":  f16Buffer(t, d, "ws", scales),
					"tq":  i8Buffer(t, d, "tq", tableQuants),
					"ts":  f16Buffer(t, d, "ts", tableScales),
					"ids": u32Buffer(t, d, "ids", []uint32{4, 1}),
				},
			})
		},
	}
}

// groupedParityCase covers the mixture-of-experts products.
//
// The counts leave one expert empty, which is the case a backend computing its
// segment offsets differently gets wrong: an empty segment shifts every expert
// after it, and every non-empty arrangement hides that.
func groupedParityCase() tensorParityCase {
	const experts, K, N, tokens = 3, 64, 4, 5
	counts := []uint32{2, 0, 3}
	x := ramp(tokens*K, 0.19)
	w := make([]float32, experts*K*N)
	for e := range experts {
		for i := range K * N {
			w[e*K*N+i] = float32(math.Cos(float64(i)*0.13)) * float32(e+1)
		}
	}

	return tensorParityCase{
		name:   "a grouped product with an empty expert",
		covers: parity.Covers{"GroupedMatVec", "GroupedMatMul"},
		ceiling: parity.Ceiling{Abs: 1e-4,
			Why: "the reduction order of a length-64 dot product per expert"},
		run: func(t *testing.T, d *accel.Device) []float32 {
			t.Helper()
			rt := parityRuntime(t, d)
			b := rt.NewBuilder("grouped")
			xv := tensor.Input(b, tensor.ValueDesc{
				Name: "x", DType: accel.F32, Shape: tensor.Shape{tokens, K}})
			wv := tensor.Weight(b, tensor.ValueDesc{
				Name: "w", DType: accel.F32, Shape: tensor.Shape{experts, K, N}})
			cv := tensor.Input(b, tensor.ValueDesc{
				Name: "counts", DType: accel.U32, Shape: tensor.Shape{experts}})

			// Both spellings over the same operands, summed: they select
			// different kernels and must produce the same rows.
			tensor.Output(b, "out", tensor.Add(b,
				tensor.GroupedMatVec(b, xv, wv, cv),
				tensor.GroupedMatMul(b, xv, wv, cv)))

			return runPlan(t, d, rt, b, "grouped", "out", tokens*N, tensor.Bindings{
				Buffers: map[string]accel.BufferView{
					"x": f32Buffer(t, d, "x", x), "w": f32Buffer(t, d, "w", w),
					"counts": u32Buffer(t, d, "counts", counts),
				},
			})
		},
	}
}

// attentionParityCase covers the KV cache: a state, a layer view of it, a
// scatter into that view, an attention over what the scatter wrote, and a read
// of the state as a value.
func attentionParityCase() tensorParityCase {
	const (
		layers   = 2
		qHeads   = 4
		kvHeads  = 2
		headDim  = 8
		capacity = 8
	)
	return tensorParityCase{
		name: "a layered KV cache, scattered into and attended over",
		covers: parity.Covers{
			"NewState", "LayerState", "ScatterRows", "ReadState", "Attention",
		},
		ceiling: parity.Ceiling{ULP: 32,
			Why: "attention's exp and its reduction over the cached positions"},
		run: func(t *testing.T, d *accel.Device) []float32 {
			t.Helper()
			rt := parityRuntime(t, d)
			b := rt.NewBuilder("attention")
			tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
			tensor.Scalar(b, tensor.ScalarDesc{Name: "one", Kind: tensor.ScalarF32})

			q := tensor.Input(b, tensor.ValueDesc{
				Name: "q", DType: accel.F32, Shape: tensor.Shape{qHeads, headDim}})
			newK := tensor.Input(b, tensor.ValueDesc{
				Name: "newk", DType: accel.F32, Shape: tensor.Shape{1, kvHeads * headDim}})
			newV := tensor.Input(b, tensor.ValueDesc{
				Name: "newv", DType: accel.F32, Shape: tensor.Shape{1, kvHeads * headDim}})
			slot := tensor.Input(b, tensor.ValueDesc{
				Name: "slot", DType: accel.U32, Shape: tensor.Shape{1}})
			lengths := tensor.Input(b, tensor.ValueDesc{
				Name: "len", DType: accel.U32, Shape: tensor.Shape{1}})

			kc := tensor.NewState(b, tensor.StateDesc{Name: "kcache", DType: accel.F32,
				Shape: tensor.Shape{layers, capacity, kvHeads, headDim}})
			vc := tensor.NewState(b, tensor.StateDesc{Name: "vcache", DType: accel.F32,
				Shape: tensor.Shape{layers, capacity, kvHeads, headDim}})

			// Layer 1 rather than layer 0: a backend that ignored the layer
			// offset would read the layer it did not write, and layer 0 is the
			// one offset zero happens to be right for.
			k1, v1 := tensor.LayerState(b, kc, 1), tensor.LayerState(b, vc, 1)
			k2 := tensor.ScatterRows(b, k1, newK, slot)
			v2 := tensor.ScatterRows(b, v1, newV, slot)

			tensor.Output(b, "out", tensor.Attention(b, q, k2, v2,
				tensor.AttentionOptions{Lengths: lengths, ScaleName: "scale"}))
			// The scattered cache as a computed value. A view of a state is
			// still that state, and an output naming one is refused -- the
			// buffer is the caller's and already theirs to read -- so this goes
			// through a kernel. That is not a workaround: what makes the
			// scatter's placement comparable is a node that read every row of
			// the layer, and a scale is the cheapest one that does.
			tensor.Output(b, "rows", tensor.Scale(b, tensor.ReadState(b, k2), "one"))

			plan, err := b.Compile(rt, tensor.CompileOptions{Label: "attention"})
			if err != nil {
				t.Fatalf("compile attention on %v: %v", d.Info().Backend, err)
			}
			defer plan.Close()

			out := f32Buffer(t, d, "out", make([]float32, qHeads*headDim))
			cache := f32Buffer(t, d, "rows",
				make([]float32, capacity*kvHeads*headDim))
			kBuf := f32Buffer(t, d, "kcache",
				make([]float32, layers*capacity*kvHeads*headDim))
			vBuf := f32Buffer(t, d, "vcache",
				make([]float32, layers*capacity*kvHeads*headDim))

			// Three steps, so the attention reads positions a previous
			// submission wrote: a cache that did not survive between
			// submissions would agree with itself and differ from this.
			var got []float32
			for step := range 3 {
				f := plan.Submit(d.Queue(), tensor.Bindings{
					Buffers: map[string]accel.BufferView{
						"q":      f32Buffer(t, d, "q", ramp(qHeads*headDim, float64(step)*13)),
						"newk":   f32Buffer(t, d, "newk", ramp(kvHeads*headDim, float64(step)*7)),
						"newv":   f32Buffer(t, d, "newv", ramp(kvHeads*headDim, float64(step)*3)),
						"slot":   u32Buffer(t, d, "slot", []uint32{uint32(step)}),
						"len":    u32Buffer(t, d, "len", []uint32{uint32(step + 1)}),
						"kcache": kBuf, "vcache": vBuf,
						"out": out, "rows": cache,
					},
					Scalars: map[string]tensor.ScalarValue{
						"scale": tensor.F32(float32(1 / math.Sqrt(headDim))),
						"one":   tensor.F32(1)},
				})
				if err := f.Wait(); err != nil {
					t.Fatalf("submit attention step %d on %v: %v", step, d.Info().Backend, err)
				}
			}
			got = make([]float32, qHeads*headDim+capacity*kvHeads*headDim)
			if err := d.Queue().ReadBuffer(out.Buffer, 0, got[:qHeads*headDim]); err != nil {
				t.Fatalf("read out on %v: %v", d.Info().Backend, err)
			}
			if err := d.Queue().ReadBuffer(cache.Buffer, 0, got[qHeads*headDim:]); err != nil {
				t.Fatalf("read cache on %v: %v", d.Info().Backend, err)
			}
			return got
		},
	}
}

// linearAttentionParityCase covers the recurrent form and the state it carries
// between tokens.
func linearAttentionParityCase() tensorParityCase {
	const batch, heads, keyDim, valueDim, tokens = 1, 2, 6, 4, 5
	return tensorParityCase{
		name:   "a linear attention step over a carried state",
		covers: parity.Covers{"LinearAttention"},
		ceiling: parity.Ceiling{Abs: 1e-4,
			Why: "the recurrence accumulates a product per token, so the two orders differ by a reduction"},
		run: func(t *testing.T, d *accel.Device) []float32 {
			t.Helper()
			rt := parityRuntime(t, d)
			b := rt.NewBuilder("linear")
			tensor.Scalar(b, tensor.ScalarDesc{Name: "one", Kind: tensor.ScalarF32})
			q := tensor.Input(b, tensor.ValueDesc{
				Name: "q", DType: accel.F32, Shape: tensor.Shape{tokens, heads, keyDim}})
			k := tensor.Input(b, tensor.ValueDesc{
				Name: "k", DType: accel.F32, Shape: tensor.Shape{tokens, heads, keyDim}})
			v := tensor.Input(b, tensor.ValueDesc{
				Name: "v", DType: accel.F32, Shape: tensor.Shape{tokens, heads, valueDim}})
			alpha := tensor.Input(b, tensor.ValueDesc{
				Name: "alpha", DType: accel.F32, Shape: tensor.Shape{tokens}})
			beta := tensor.Input(b, tensor.ValueDesc{
				Name: "beta", DType: accel.F32, Shape: tensor.Shape{tokens}})
			extents := tensor.Input(b, tensor.ValueDesc{
				Name: "extents", DType: accel.U32, Shape: tensor.Shape{batch}})
			st := tensor.NewState(b, tensor.StateDesc{Name: "state", DType: accel.F32,
				Shape: tensor.Shape{batch, heads, valueDim, keyDim}})

			out, next := tensor.LinearAttention(b, q, k, v, st, tensor.LinearOptions{
				Alpha: alpha, Beta: beta, QueryExtents: extents,
			})
			tensor.Output(b, "out", out)
			// The state the step carried forward, through a kernel for the
			// reason the attention case gives: a view of a state is still that
			// state, and an output cannot name one.
			tensor.Output(b, "carried", tensor.Scale(b, tensor.ReadState(b, next), "one"))

			plan, err := b.Compile(rt, tensor.CompileOptions{Label: "linear"})
			if err != nil {
				t.Fatalf("compile linear on %v: %v", d.Info().Backend, err)
			}
			defer plan.Close()

			stateN := batch * heads * valueDim * keyDim
			outN := tokens * heads * valueDim
			outBuf := f32Buffer(t, d, "out", make([]float32, outN))
			stateOut := f32Buffer(t, d, "carried", make([]float32, stateN))
			state := f32Buffer(t, d, "state", make([]float32, stateN))

			gate := make([]float32, tokens)
			for i := range gate {
				gate[i] = 0.5 + float32(i)*0.05
			}
			f := plan.Submit(d.Queue(), tensor.Bindings{
				Buffers: map[string]accel.BufferView{
					"q":       f32Buffer(t, d, "q", ramp(tokens*heads*keyDim, 0.2)),
					"k":       f32Buffer(t, d, "k", ramp(tokens*heads*keyDim, 1.2)),
					"v":       f32Buffer(t, d, "v", ramp(tokens*heads*valueDim, 2.2)),
					"alpha":   f32Buffer(t, d, "alpha", gate),
					"beta":    f32Buffer(t, d, "beta", gate),
					"extents": u32Buffer(t, d, "extents", []uint32{tokens}),
					"state":   state, "out": outBuf, "carried": stateOut,
				},
				Scalars: map[string]tensor.ScalarValue{"one": tensor.F32(1)},
			})
			if err := f.Wait(); err != nil {
				t.Fatalf("submit linear on %v: %v", d.Info().Backend, err)
			}
			got := make([]float32, outN+stateN)
			if err := d.Queue().ReadBuffer(outBuf.Buffer, 0, got[:outN]); err != nil {
				t.Fatalf("read out on %v: %v", d.Info().Backend, err)
			}
			if err := d.Queue().ReadBuffer(stateOut.Buffer, 0, got[outN:]); err != nil {
				t.Fatalf("read state on %v: %v", d.Info().Backend, err)
			}
			return got
		},
	}
}

// samplingParityCase covers the token-selection operators.
//
// The masks and the draw are compared as *values* rather than as tokens where
// possible, because a token is an index and an index that differs by one says
// nothing about how far apart the two distributions were. The chosen token is
// compared too, since that is what a caller acts on.
func samplingParityCase() tensorParityCase {
	const rows, vocab = 3, 48
	logits := make([]float32, rows*vocab)
	for i := range logits {
		logits[i] = float32(math.Sin(float64(i)*0.37)) * 4
	}
	draws := []float32{0.1, 0.55, 0.93}

	return tensorParityCase{
		name: "the masks, the draw and the greedy choice",
		covers: parity.Covers{
			"Argmax", "SampleCategorical", "TopKMask", "TopPMask", "Sample",
			"DeclareSamplingScalars",
		},
		ceiling: parity.Ceiling{Abs: 1e-5,
			Why: "the softmax under the masks reaches exp and a division; the token indices are exact and contribute nothing"},
		run: func(t *testing.T, d *accel.Device) []float32 {
			t.Helper()
			rt := parityRuntime(t, d)
			b := rt.NewBuilder("sampling")

			x := tensor.Input(b, tensor.ValueDesc{
				Name: "x", DType: accel.F32, Shape: tensor.Shape{rows, vocab}})
			u := tensor.Input(b, tensor.ValueDesc{
				Name: "u", DType: accel.F32, Shape: tensor.Shape{rows}})
			// Sample composes a softmax with the default axis, which is the last
			// one only at rank 1, so the whole policy runs over a single row.
			// That is what the operator supports today rather than a shape
			// chosen for this case.
			one := tensor.Input(b, tensor.ValueDesc{
				Name: "one", DType: accel.F32, Shape: tensor.Shape{vocab}})
			oneDraw := tensor.Input(b, tensor.ValueDesc{
				Name: "onedraw", DType: accel.F32, Shape: tensor.Shape{1}})

			probs := tensor.Softmax(b, x, tensor.SoftmaxOptions{Axis: 1})
			masked := tensor.TopPMask(b, tensor.TopKMask(b, probs, 8), 0.6)
			tensor.Output(b, "mask", masked)

			opts := tensor.SamplingOptions{Temperature: 0.8, TopK: 8, TopP: 0.6}
			tensor.DeclareSamplingScalars(b, opts, "s")

			// Three token outputs, each u32 and read separately: there is no
			// registered u32-to-f32 conversion, so folding them into one float
			// buffer would need a kernel that does not exist. A token is exact
			// on both backends anyway, so nothing is lost by comparing the
			// three as their own arrays.
			tensor.Output(b, "drawn", tensor.SampleCategorical(b, masked, u))
			tensor.Output(b, "greedy", tensor.Argmax(b, x))
			tensor.Output(b, "policy",
				tensor.Sample(b, one, oneDraw, nil, nil, opts, "s"))

			plan, err := b.Compile(rt, tensor.CompileOptions{Label: "sampling"})
			if err != nil {
				t.Fatalf("compile sampling on %v: %v", d.Info().Backend, err)
			}
			defer plan.Close()

			scalars, err := opts.Scalars("s", uint32(vocab), 0)
			if err != nil {
				t.Fatalf("sampling scalars: %v", err)
			}

			mask := f32Buffer(t, d, "mask", make([]float32, rows*vocab))
			drawn := u32Buffer(t, d, "drawn", make([]uint32, rows))
			greedy := u32Buffer(t, d, "greedy", make([]uint32, rows))
			policy := u32Buffer(t, d, "policy", make([]uint32, 1))
			f := plan.Submit(d.Queue(), tensor.Bindings{
				Buffers: map[string]accel.BufferView{
					"x":       f32Buffer(t, d, "x", logits),
					"u":       f32Buffer(t, d, "u", draws),
					"one":     f32Buffer(t, d, "one", logits[:vocab]),
					"onedraw": f32Buffer(t, d, "onedraw", draws[:1]),
					"mask":    mask, "drawn": drawn, "greedy": greedy, "policy": policy,
				},
				Scalars: scalars,
			})
			if err := f.Wait(); err != nil {
				t.Fatalf("submit sampling on %v: %v", d.Info().Backend, err)
			}

			got := make([]float32, 0, rows*vocab+2*rows+1)
			values := make([]float32, rows*vocab)
			if err := d.Queue().ReadBuffer(mask.Buffer, 0, values); err != nil {
				t.Fatalf("read mask on %v: %v", d.Info().Backend, err)
			}
			got = append(got, values...)
			for _, tok := range []struct {
				name string
				view accel.BufferView
				n    int
			}{{"drawn", drawn, rows}, {"greedy", greedy, rows}, {"policy", policy, 1}} {
				ids := make([]uint32, tok.n)
				if err := d.Queue().ReadBuffer(tok.view.Buffer, 0, ids); err != nil {
					t.Fatalf("read %s on %v: %v", tok.name, d.Info().Backend, err)
				}
				// Shifted by one so a token of zero is still a non-zero value:
				// the degeneracy check refuses an all-zero result, and a greedy
				// choice of token zero is a legitimate answer.
				for _, id := range ids {
					got = append(got, float32(id)+1)
				}
			}
			return got
		},
	}
}
