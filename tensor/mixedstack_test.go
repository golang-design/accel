// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"
	"golang.design/x/accel/tensor"
)

// A decoder stack on f32 activations and narrow weights dispatches no Cast.
//
// # What this measures
//
// The count, which is the whole of accel issue 14. The report was not that a
// projection was wrong, it was that four dispatches per layer existed only to
// satisfy a dtype check -- 144 per forward pass at 36 layers, each a full pass
// over the activations, on a workload that is memory-bound. So the assertion is
// on the plan rather than on the numbers: the same graph is built twice, once
// the way the refusal forced and once the way the corpus now allows, and the
// difference is counted.
//
// Both weight widths are covered because both are what a model ships. f16 is
// what a small model is, int8 is what a large one becomes, and the second was
// the worse case: `auto` reaches for int8 *because* the model is large, so the
// configuration least able to afford a pass over its activations was the one
// that had to make four of them per layer.
func TestADecoderStackOnF32ActivationsNeedsNoCast(t *testing.T) {
	const layers, width, hidden = 3, 16, 32

	// Four projections per layer, which is what the report counted: three into
	// the attention block and one out of it. The MLP's are the same shape of
	// operand and would only multiply the number.
	const perLayer = 4

	for _, wt := range []struct {
		name       string
		quantized  bool
		wantKernel string
	}{
		// The kernel each expects at M=1, which is a decode step and is the
		// shape a model spends nearly all of its dispatches in. Both paths
		// have an M=1 kernel over f32 activations since 2026-09-02; before
		// that the unquantized one took the tile with seven rows idle.
		{"f16 weights", false, "MatVecF32F16"},
		{"int8 weights", true, "QuantMatVecF32"},
	} {
		t.Run(wt.name, func(t *testing.T) {
			// One set of weights, used by both graphs, so the two differ in
			// nothing but where the widths meet.
			ws := make([][]float32, layers*perLayer)
			for i := range ws {
				ws[i] = make([]float32, width*width)
				for j := range ws[i] {
					ws[i][j] = float32(math.Sin(float64(i*97+j)*0.19)) * 0.4
				}
			}
			gains := make([][]float32, layers)
			for l := range gains {
				gains[l] = make([]float32, width)
				for i := range gains[l] {
					gains[l][i] = 1 + float32(i%3)/16
				}
			}
			acts := make([]float32, width)
			for i := range acts {
				acts[i] = float32(math.Cos(float64(i)*0.23)) * 1.5
			}

			// build assembles the stack. narrowFirst is the shape the refusal
			// forced: every projection preceded by a conversion of its
			// activations, which is the dispatch this issue is about.
			build := func(rt *tensor.Runtime, label string, narrowFirst bool) *tensor.Plan {
				b := rt.NewBuilder(label)
				h := tensor.Input(b, tensor.ValueDesc{
					Name: "x", DType: accel.F32, Shape: tensor.Shape{1, width},
				})
				for l := range layers {
					gain := tensor.Weight(b, tensor.ValueDesc{
						Name:  fmt.Sprintf("g%d", l),
						DType: accel.F32, Shape: tensor.Shape{width},
					})
					normed := tensor.RMSNorm(b, h, gain, 1e-5)
					acc := h
					for p := range perLayer {
						x := normed
						if narrowFirst {
							x = tensor.Cast(b, normed, accel.F16)
						}
						name := fmt.Sprintf("w%d_%d", l, p)
						var proj *tensor.Tensor
						if wt.quantized {
							_, s := quant.Int8Quantize(ws[l*perLayer+p])
							proj = tensor.QuantMatMul(b, x, tensor.Quantized{
								Quants: tensor.Weight(b, tensor.ValueDesc{
									Name: name + "q", DType: accel.I8,
									Shape: tensor.Shape{width, width},
								}),
								Scales: tensor.Weight(b, tensor.ValueDesc{
									Name: name + "s", DType: accel.F16,
									Shape: tensor.Shape{len(s)},
								}),
							})
						} else {
							proj = tensor.MatMul(b, x, tensor.Weight(b, tensor.ValueDesc{
								Name: name, DType: accel.F16,
								Shape: tensor.Shape{width, width},
							}))
						}
						acc = tensor.Add(b, acc, proj)
					}
					h = acc
				}
				tensor.Output(b, "out", h)
				plan, err := b.Compile(rt, tensor.CompileOptions{Label: label})
				if err != nil {
					t.Fatalf("compile %s: %v", label, err)
				}
				return plan
			}

			rt := newRuntime(t)
			forced := build(rt, "forced", true)
			defer forced.Close()
			direct := build(rt, "direct", false)
			defer direct.Close()

			countCasts := func(p *tensor.Plan) int {
				n := 0
				for _, s := range p.Selections() {
					if strings.Contains(s.Op, "Cast") {
						n++
					}
				}
				return n
			}

			// The shape the refusal forced: one conversion per projection, all
			// of them a full pass over the activations.
			if got, want := countCasts(forced), layers*perLayer; got != want {
				t.Fatalf("the forced graph dispatches %d casts and should dispatch %d; "+
					"if it is not %d this test is no longer measuring what issue 14 "+
					"reported", got, want, want)
			}
			// And what the corpus now allows: none.
			if got := countCasts(direct); got != 0 {
				t.Errorf("the direct graph dispatches %d casts; at %d layers that is %d "+
					"per forward pass existing only to satisfy a dtype check",
					got, layers, got)
			}

			// Every projection really is on the mixed kernel, and not on a
			// narrow one that happens to have been handed a wide buffer.
			projections := 0
			for _, s := range direct.Selections() {
				if s.Kernel == wt.wantKernel {
					projections++
				}
			}
			if projections != layers*perLayer {
				t.Fatalf("%d of %d projections ran %s; selections: %+v",
					projections, layers*perLayer, wt.wantKernel, direct.Selections())
			}

			// And it runs. A plan that reported the right kernels and produced
			// nothing would pass everything above.
			d := rt.Device()
			binds := map[string]accel.BufferView{
				"x":   f32Buffer(t, d, "x", acts),
				"out": f32Buffer(t, d, "out", make([]float32, width)),
			}
			for l := range layers {
				binds[fmt.Sprintf("g%d", l)] = f32Buffer(t, d,
					fmt.Sprintf("g%d", l), gains[l])
				for p := range perLayer {
					name := fmt.Sprintf("w%d_%d", l, p)
					if wt.quantized {
						q, s := quant.Int8Quantize(ws[l*perLayer+p])
						binds[name+"q"] = i8Buffer(t, d, name+"q", q)
						binds[name+"s"] = f16BitsBuffer(t, d, name+"s", s)
					} else {
						binds[name] = f16Buffer(t, d, name, ws[l*perLayer+p])
					}
				}
			}
			if err := direct.Submit(d.Queue(), tensor.Bindings{Buffers: binds}).Wait(); err != nil {
				t.Fatalf("submit: %v", err)
			}
			got := make([]float32, width)
			if err := d.Queue().ReadBuffer(binds["out"].Buffer, 0, got); err != nil {
				t.Fatalf("readback: %v", err)
			}
			var moved bool
			for _, v := range got {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatalf("the stack produced %v", got)
				}
				if v != 0 {
					moved = true
				}
			}
			if !moved {
				t.Fatal("the stack produced all zeros, so running it proves nothing")
			}
		})
	}
}
