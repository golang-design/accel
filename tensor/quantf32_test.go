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

// f32 activations multiply int8 weights, at both shapes, with no Cast.
//
// # What this closes
//
// `auto` selects int8 whenever f16 does not fit, so the configuration a model
// lands in *because it is large* was the one that had to Cast its activations
// narrow before every projection and could least afford the pass (accel issue
// 14, which re-filed issue 11's asymmetry one level up).
//
// # Why both shapes are here
//
// M=1 is every decode step and larger M is every prefill, and they select
// different kernels whose reductions differ -- a tree over the lanes against a
// sequential fold. Admitting f32 on the general kernel alone would have closed
// the refusal and left decode, which is nearly every dispatch a model makes,
// running the unspecialized shape.
func TestF32ActivationsMultiplyInt8Weights(t *testing.T) {
	for _, c := range []struct {
		name       string
		m, k, n    int
		wantKernel string
	}{
		{"decode", 1, 64, 16, "QuantMatVecF32"},
		{"prefill", 4, 64, 16, "QuantMatMulTiledF32"},
	} {
		t.Run(c.name, func(t *testing.T) {
			acts := make([]float32, c.m*c.k)
			weights := make([]float32, c.k*c.n)
			for i := range acts {
				acts[i] = float32(math.Sin(float64(i)*0.31)) * 2
			}
			for i := range weights {
				weights[i] = float32(math.Cos(float64(i)*0.17)) * 1.5
			}
			wq, ws := quant.Int8Quantize(weights)

			rt := newRuntime(t)
			d := rt.Device()
			b := rt.NewBuilder(c.name)

			x := tensor.Input(b, tensor.ValueDesc{
				Name: "x", DType: accel.F32, Shape: tensor.Shape{c.m, c.k},
			})
			qw := tensor.Weight(b, tensor.ValueDesc{
				Name: "wq", DType: accel.I8, Shape: tensor.Shape{c.k, c.n},
			})
			sw := tensor.Weight(b, tensor.ValueDesc{
				Name: "ws", DType: accel.F16, Shape: tensor.Shape{len(ws)},
			})
			tensor.Output(b, "out",
				tensor.QuantMatMul(b, x, tensor.Quantized{Quants: qw, Scales: sw}))

			plan, err := b.Compile(rt, tensor.CompileOptions{Label: c.name})
			if err != nil {
				t.Fatalf("compile: %v", err)
			}
			defer plan.Close()

			// The kernel name and not only the numbers. The f16 and f32
			// variants of a pair differ only in binding 0's element type, so a
			// selection that ignored the activation's width would bind an f32
			// buffer to an f16 binding -- a buffer of the wrong element size
			// read as the right one, which is a plausible matrix rather than a
			// diagnostic.
			sel := plan.Selections()
			if len(sel) != 1 || sel[0].Kernel != c.wantKernel {
				t.Fatalf("selections are %+v, want %s", sel, c.wantKernel)
			}
			if !strings.Contains(sel[0].Reason, "f32 activations") {
				t.Errorf("the selection does not say the activations are f32: %q",
					sel[0].Reason)
			}
			for _, s := range sel {
				if strings.Contains(s.Op, "Cast") {
					t.Errorf("the plan contains %s; the f32 variants exist so that it "+
						"does not", s.Op)
				}
			}

			out := f32Buffer(t, d, "out", make([]float32, c.m*c.n))
			f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
				"x":   f32Buffer(t, d, "x", acts),
				"wq":  i8Buffer(t, d, "wq", wq),
				"ws":  f16BitsBuffer(t, d, "ws", ws),
				"out": out,
			}})
			if err := f.Wait(); err != nil {
				t.Fatalf("submit: %v", err)
			}
			got := make([]float32, c.m*c.n)
			if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
				t.Fatalf("readback: %v", err)
			}

			// The budget is the one TestAPlanMixesQuantizedAndUnquantized
			// derives, with the activation term dropped: the activations reach
			// the kernel unrounded now, so what is left is 027's quantization
			// error plus 008 section 7's f32 accumulation. The accumulation
			// term follows the kernel that ran -- a tree of depth log2(128) on
			// top of each lane's fold at M=1, a sequential fold over K
			// otherwise -- because that is what each kernel sums.
			const u = 1.0 / (1 << 24)
			for row := range c.m {
				for col := range c.n {
					var exact, magnitude, qErr float64
					for kk := range c.k {
						a := float64(acts[row*c.k+kk])
						wv := float64(weights[kk*c.n+col])
						exact += a * wv
						magnitude += math.Abs(a * wv)
						s := float64(ws[(kk*c.n+col)/quant.Int8Block].F32())
						qErr += math.Abs(a) * s / 2
					}
					depth := float64(c.k)
					if c.m == 1 {
						depth = math.Ceil(float64(c.k)/128) + 7
					}
					accErr := (depth * u) / (1 - depth*u) * magnitude
					i := row*c.n + col
					if diff := math.Abs(float64(got[i]) - exact); diff > qErr+accErr {
						t.Fatalf("(%d,%d) is %v against an exact %v: off by %v, budget "+
							"%v + %v", row, col, got[i], exact, diff, qErr, accErr)
					}
				}
			}
		})
	}
}

// An activation width no kernel reads is refused, and says which are registered.
func TestQuantMatMulRefusesAnUnregisteredActivationWidth(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("i8acts")
	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.I8, Shape: tensor.Shape{4, 32},
	})
	qw := tensor.Weight(b, tensor.ValueDesc{
		Name: "wq", DType: accel.I8, Shape: tensor.Shape{32, 8},
	})
	sw := tensor.Weight(b, tensor.ValueDesc{
		Name: "ws", DType: accel.F16, Shape: tensor.Shape{32 * 8 / quant.Int8Block},
	})
	tensor.QuantMatMul(b, x, tensor.Quantized{Quants: qw, Scales: sw})
	if err := b.Err(); err == nil {
		t.Fatal("i8 activations were accepted")
	} else if !strings.Contains(err.Error(), "f16 and f32 activations") {
		t.Errorf("the refusal should list the registered widths, got %v", err)
	}
}
