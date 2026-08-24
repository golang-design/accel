// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"fmt"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// What a layered cache costs at bind time.
//
// A window of a port is its own graph slot, so a 36-layer model has 72 slots
// where a whole-port binding has 2. specs/003-command-graph.md's check V24
// compares slots pairwise at every bind and a decode plan binds once per token,
// so the pairwise growth is a per-token cost and the question is whether it is
// one worth noticing.
//
// Measured on an M2, this is what it reports: 448 microseconds at one layer,
// 3.82 milliseconds at eight, and 13.0 at thirty-two. Thirty-two layers cost 29
// times one layer for 32 times the work, so the total is *sub*-linear in the
// layer count and the quadratic term is below the noise of the dispatches it
// guards. What this measures is therefore a bound rather than the check itself:
// V24 is not separable here because it is not large enough to separate.
func BenchmarkLayeredBind(b *testing.B) {
	const (
		qHeads   = 4
		kvHeads  = 2
		headDim  = 8
		capacity = 16
		perLayer = capacity * kvHeads * headDim
	)
	for _, layers := range []int{1, 8, 32} {
		b.Run(fmt.Sprintf("layers%d", layers), func(b *testing.B) {
			rt := newRuntimeB(b)
			d := rt.Device()
			bl := rt.NewBuilder("bindbench")
			tensor.Scalar(bl, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
			lengths := tensor.Input(bl, tensor.ValueDesc{
				Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
			})
			kc := tensor.NewState(bl, tensor.StateDesc{
				Name: "kcache", DType: accel.F32,
				Shape: tensor.Shape{layers, capacity, kvHeads, headDim},
			})
			vc := tensor.NewState(bl, tensor.StateDesc{
				Name: "vcache", DType: accel.F32,
				Shape: tensor.Shape{layers, capacity, kvHeads, headDim},
			})
			bufs := map[string]accel.BufferView{
				"kcache": f32BufferBB(b, d, "kcache", make([]float32, layers*perLayer)),
				"vcache": f32BufferBB(b, d, "vcache", make([]float32, layers*perLayer)),
				"len":    u32BufferBB(b, d, "len", []uint32{1}),
			}
			for l := range layers {
				q := tensor.Input(bl, tensor.ValueDesc{
					Name: fmt.Sprintf("q%d", l), DType: accel.F32,
					Shape: tensor.Shape{qHeads, headDim},
				})
				tensor.Output(bl, fmt.Sprintf("out%d", l),
					tensor.Attention(bl, q, tensor.LayerState(bl, kc, l),
						tensor.LayerState(bl, vc, l),
						tensor.AttentionOptions{Lengths: lengths, ScaleName: "scale"}))
				bufs[fmt.Sprintf("q%d", l)] = f32BufferBB(b, d, "q",
					make([]float32, qHeads*headDim))
				bufs[fmt.Sprintf("out%d", l)] = f32BufferBB(b, d, "out",
					make([]float32, qHeads*headDim))
			}
			plan, err := bl.Compile(rt, tensor.CompileOptions{Label: "bindbench"})
			if err != nil {
				b.Fatalf("compile: %v", err)
			}
			defer plan.Close()

			binds := tensor.Bindings{
				Buffers: bufs,
				Scalars: map[string]tensor.ScalarValue{"scale": tensor.F32(1)},
			}
			if err := plan.Submit(d.Queue(), binds).Wait(); err != nil {
				b.Fatalf("warmup: %v", err)
			}
			b.ResetTimer()
			for range b.N {
				if err := plan.Submit(d.Queue(), binds).Wait(); err != nil {
					b.Fatalf("submit: %v", err)
				}
			}
		})
	}
}

func newRuntimeB(b *testing.B) *tensor.Runtime {
	b.Helper()
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	b.Cleanup(func() { _ = d.Close() })
	rt, err := tensor.NewRuntime(d)
	if err != nil {
		b.Fatalf("runtime: %v", err)
	}
	return rt
}

func f32BufferBB(b *testing.B, d *accel.Device, label string, vals []float32) accel.BufferView {
	b.Helper()
	buf, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: len(vals), Label: label,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
	})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = buf.Close() })
	v, err := buf.View(0, len(vals))
	if err != nil {
		b.Fatal(err)
	}
	return v
}

func u32BufferBB(b *testing.B, d *accel.Device, label string, vals []uint32) accel.BufferView {
	b.Helper()
	buf, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.U32, Count: len(vals), Label: label,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
	})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = buf.Close() })
	if err := d.Queue().WriteBuffer(buf, 0, vals); err != nil {
		b.Fatal(err)
	}
	v, err := buf.View(0, len(vals))
	if err != nil {
		b.Fatal(err)
	}
	return v
}
