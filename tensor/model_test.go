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

// A two-layer attention stack composes, on the CPU backend and on Metal.
//
// # What this is, and what it is not
//
// specs/009-sequencing.md's M7 criterion asks for a two-layer model producing
// *logits*. This is not that, and the gap is a missing operator rather than a
// missing test: the registered GEMM reads f16 and every other operator here is
// f32, so a projection cannot consume a normalization's output without a
// dtype-conversion operator, and specs/010-kernel-corpus.md registers no
// conversion kernel. An earlier version of this test papered over that by
// supplying the f16 operands from the host and discarding the f32 results,
// which computed every piece and composed none of them.
//
// So this composes what the dtypes allow, which is the whole attention path:
// an embedding lookup, then two layers of normalize, attend over a per-layer
// cache, and add the residual. Four tokens through one plan, with the caches
// bound once and everything else rewritten per step.
//
// The reference is computed in f64 beside it rather than committed as numbers.
// A committed golden would be digits nobody could evaluate: a failure would say
// the answer changed and not whether the new one is wrong.
//
// One state per layer rather than a layered cache, because a slot binds a whole
// resource and specs/026-tensor-decode.md records that gap.
func TestATwoLayerAttentionStackComposes(t *testing.T) {
	const (
		layers   = 2
		vocab    = 16
		qHeads   = 4
		kvHeads  = 2
		headDim  = 8
		width    = qHeads * headDim
		capacity = 8
	)

	weights := map[string][]float32{}
	fill := func(name string, n int, f func(i int) float32) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = f(i)
		}
		weights[name] = v
		return v
	}
	table := fill("table", vocab*width, func(i int) float32 {
		return float32(math.Sin(float64(i)*0.7)) * 0.5
	})
	for l := range layers {
		fill(gainName(l), width, func(i int) float32 { return 1 + float32(i%3)/16 })
	}

	run := func(t *testing.T, d *accel.Device, tokens []uint32) []float32 {
		t.Helper()
		rt, err := tensor.NewRuntime(d)
		if err != nil {
			t.Fatalf("runtime: %v", err)
		}
		defer func() {
			if err := rt.Close(); err != nil {
				t.Errorf("runtime close: %v", err)
			}
		}()

		b := rt.NewBuilder("stack")
		tensor.Scalar(b, tensor.ScalarDesc{Name: "len", Kind: tensor.ScalarU32})
		tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})

		ids := tensor.Input(b, tensor.ValueDesc{
			Name: "ids", DType: accel.U32, Shape: tensor.Shape{1},
		})
		slot := tensor.Input(b, tensor.ValueDesc{
			Name: "slot", DType: accel.U32, Shape: tensor.Shape{1},
		})
		tab := tensor.Weight(b, tensor.ValueDesc{
			Name: "table", DType: accel.F32, Shape: tensor.Shape{vocab, width},
		})

		h := tensor.Rows(b, tab, ids) // [1, width]

		for l := range layers {
			gain := tensor.Weight(b, tensor.ValueDesc{
				Name: gainName(l), DType: accel.F32, Shape: tensor.Shape{width},
			})
			kc := tensor.Persistent(b, tensor.StateDesc{
				Name: kName(l), DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
			})
			vc := tensor.Persistent(b, tensor.StateDesc{
				Name: vName(l), DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
			})
			kIn := tensor.Input(b, tensor.ValueDesc{
				Name: kInName(l), DType: accel.F32, Shape: tensor.Shape{1, kvHeads * headDim},
			})
			vIn := tensor.Input(b, tensor.ValueDesc{
				Name: vInName(l), DType: accel.F32, Shape: tensor.Shape{1, kvHeads * headDim},
			})

			// Normalize, reshape into heads, attend, add the residual. Every
			// value here is produced by the previous operator: nothing is
			// supplied from the host in the middle of the graph.
			normed := tensor.RMSNorm(b, h, gain, 1e-5)
			q := tensor.Reshape(b, normed, tensor.Shape{qHeads, headDim})
			attn := tensor.Attention(b, q,
				tensor.ScatterRows(b, kc, kIn, slot),
				tensor.ScatterRows(b, vc, vIn, slot),
				tensor.AttentionOptions{CurrentLengthName: "len", ScaleName: "scale"})
			h = tensor.Add(b, h, tensor.Reshape(b, attn, tensor.Shape{1, width}))
		}
		tensor.Output(b, "h", h)

		plan, err := b.Compile(rt, tensor.CompileOptions{Label: "stack"})
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		defer func() {
			if err := plan.Close(); err != nil {
				t.Errorf("plan close: %v", err)
			}
		}()

		bufs := map[string]accel.BufferView{
			"table": f32Buffer(t, d, "table", table),
		}
		for l := range layers {
			bufs[gainName(l)] = f32Buffer(t, d, gainName(l), weights[gainName(l)])
			bufs[kName(l)] = f32Buffer(t, d, kName(l), make([]float32, capacity*kvHeads*headDim))
			bufs[vName(l)] = f32Buffer(t, d, vName(l), make([]float32, capacity*kvHeads*headDim))
		}
		hOut := f32Buffer(t, d, "h", make([]float32, width))
		bufs["h"] = hOut

		var last []float32
		for step, tok := range tokens {
			bufs["ids"] = u32Buffer(t, d, "ids", []uint32{tok})
			bufs["slot"] = u32Buffer(t, d, "slot", []uint32{uint32(step)})
			for l := range layers {
				bufs[kInName(l)] = f32Buffer(t, d, kInName(l), layerKV(l, step, kvHeads*headDim, 1))
				bufs[vInName(l)] = f32Buffer(t, d, vInName(l), layerKV(l, step, kvHeads*headDim, 2))
			}
			f := plan.Submit(d.Queue(), tensor.Bindings{
				Buffers: bufs,
				Scalars: map[string]tensor.ScalarValue{
					"len":   tensor.U32(uint32(step + 1)),
					"scale": tensor.F32(float32(1 / math.Sqrt(headDim))),
				},
			})
			if err := f.Wait(); err != nil {
				t.Fatalf("step %d: %v", step, err)
			}
			last = make([]float32, width)
			if err := d.Queue().ReadBuffer(hOut.Buffer, 0, last); err != nil {
				t.Fatalf("readback: %v", err)
			}
		}
		return last
	}

	tokens := []uint32{3, 7, 1, 12}
	cpuDev, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open CPU: %v", err)
	}
	defer cpuDev.Close()
	cpu := run(t, cpuDev, tokens)

	want := referenceStack(table, weights, tokens, layers, vocab, width, qHeads, kvHeads, headDim)
	for i := range want {
		if math.Abs(float64(cpu[i])-want[i]) > 1e-3*(1+math.Abs(want[i])) {
			t.Fatalf("element %d is %v, want about %v", i, cpu[i], want[i])
		}
	}

	if gpu := runMetalModel(t, run, tokens); gpu != nil {
		for i := range cpu {
			if math.Abs(float64(cpu[i]-gpu[i])) > 1e-4*(1+math.Abs(float64(cpu[i]))) {
				t.Fatalf("element %d is %v on the CPU and %v on Metal", i, cpu[i], gpu[i])
			}
		}
	}
}

// referenceStack is the same computation in f64, stepped over the same tokens.
//
// Written from the model's definition rather than from the kernels, so it shares
// none of their structure: if a kernel and this agree, they agree about the
// operation rather than about an implementation.
func referenceStack(table []float32, weights map[string][]float32, tokens []uint32,
	layers, vocab, width, qHeads, kvHeads, headDim int) []float64 {

	keys := make([][][]float32, layers)
	vals := make([][][]float32, layers)
	var h []float64

	for step, tok := range tokens {
		h = make([]float64, width)
		for i := range h {
			h[i] = float64(table[int(tok)*width+i])
		}
		for l := range layers {
			keys[l] = append(keys[l], layerKV(l, step, kvHeads*headDim, 1))
			vals[l] = append(vals[l], layerKV(l, step, kvHeads*headDim, 2))

			// RMSNorm.
			var sq float64
			for _, v := range h {
				sq += v * v
			}
			inv := 1 / math.Sqrt(sq/float64(width)+1e-5)
			normed := make([]float64, width)
			gain := weights[gainName(l)]
			for i := range normed {
				normed[i] = h[i] * inv * float64(gain[i])
			}

			// Attention over everything cached so far.
			group := qHeads / kvHeads
			scale := 1 / math.Sqrt(float64(headDim))
			out := make([]float64, width)
			for head := range qHeads {
				kvHead := head / group
				scores := make([]float64, len(keys[l]))
				best := math.Inf(-1)
				for pos := range keys[l] {
					var acc float64
					for i := range headDim {
						acc += normed[head*headDim+i] *
							float64(keys[l][pos][kvHead*headDim+i])
					}
					scores[pos] = acc * scale
					best = math.Max(best, scores[pos])
				}
				var sum float64
				for i := range scores {
					scores[i] = math.Exp(scores[i] - best)
					sum += scores[i]
				}
				for i := range headDim {
					var acc float64
					for pos := range scores {
						acc += scores[pos] / sum * float64(vals[l][pos][kvHead*headDim+i])
					}
					out[head*headDim+i] = acc
				}
			}
			for i := range h {
				h[i] += out[i]
			}
		}
	}
	return h
}

func gainName(l int) string { return "gain" + itoa(l) }
func kName(l int) string    { return "k" + itoa(l) }
func vName(l int) string    { return "v" + itoa(l) }
func kInName(l int) string  { return "kin" + itoa(l) }
func vInName(l int) string  { return "vin" + itoa(l) }
func qInName(l int) string  { return "qin" + itoa(l) }
func aInName(l int) string  { return "ain" + itoa(l) }
func itoa(n int) string     { return string(rune('0' + n)) }

// layerKV generates deterministic per-step values, so both backends see the
// same inputs without a golden file.
func layerKV(layer, step, n, salt int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(math.Sin(float64(i+layer*31+step*17+salt*101) * 0.37))
	}
	return v
}
