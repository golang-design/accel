// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"fmt"
	"math"
	"os"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
)

// What an empty block costs.
//
// specs/044-unbounded-context.md deviation 1 states the price of a
// workgroup-uniform loop bound: the loop walks the cache's capacity and masks
// the positions past its length, so an empty block still pays its barriers.
// This measures it.
//
// The length is fixed at 256 and the capacity varies, so every case does the
// same arithmetic over the same 256 positions and differs only in how many
// empty blocks it walks -- from none at cap256 to 62 at cap8192.
//
// One graph holds `repeats` dispatches so that submission latency, which is
// milliseconds and swamps the kernel, is amortized rather than measured. They
// write the same buffer, so the graph serializes them and none overlap.
func BenchmarkAttentionEmptyBlocks(b *testing.B) {
	const qHeads, kvHeads, headDim = 32, 8, 128
	const kvLen = 256
	const repeats = 200

	d := openMetalBenchDevice(b)
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &kernels.AttentionDecodeKernel, Label: "attn",
	})
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	dims := kernels.AttnDims{
		QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
		Scale: float32(1 / math.Sqrt(headDim)),
	}

	type run struct {
		capacity int
		length   uint32
	}
	runs := []run{
		// Fixed length, growing capacity: every extra block is empty.
		{256, kvLen}, {512, kvLen}, {1024, kvLen}, {2048, kvLen},
		{4096, kvLen}, {8192, kvLen},
		// Length equal to capacity: every block is full. The difference
		// between these and the row above at the same capacity is what the
		// arithmetic costs, and the rest is what a block costs regardless.
		{512, 512}, {1024, 1024}, {2048, 2048}, {4096, 4096},
	}
	for _, rc := range runs {
		capacity, kvLen := rc.capacity, rc.length
		b.Run(fmt.Sprintf("cap%d/len%d", capacity, kvLen), func(b *testing.B) {
			n := capacity * kvHeads * headDim
			q := f32BufferB(b, d, "q", make([]float32, qHeads*headDim))
			k := f32BufferB(b, d, "k", make([]float32, n))
			v := f32BufferB(b, d, "v", make([]float32, n))
			lens := u32BufferB(b, d, "len", []uint32{kvLen})
			out := f32BufferB(b, d, "out", make([]float32, qHeads*headDim))

			r := d.NewRecorder()
			for range repeats {
				r.Dispatch(p, []accel.Binding{
					{Index: 0, Buffer: q}, {Index: 1, Buffer: k}, {Index: 2, Buffer: v},
					{Index: 3, Buffer: lens}, {Index: 4, Buffer: out},
				}, []accel.UniformValue{{Value: dims}}, accel.WorkgroupCount{X: qHeads})
			}
			g, err := r.Build()
			if err != nil {
				b.Fatalf("build: %v", err)
			}
			defer g.Close()

			if err := d.Queue().Submit(g).Wait(); err != nil {
				b.Fatalf("warmup: %v", err)
			}
			b.ResetTimer()
			for range b.N {
				if err := d.Queue().Submit(g).Wait(); err != nil {
					b.Fatalf("submit: %v", err)
				}
			}
			b.StopTimer()
			// Per decode step, which is the unit a caller thinks in.
			b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*repeats),
				"ns/decode")
		})
	}
}

func openMetalBenchDevice(b *testing.B) *accel.Device {
	b.Helper()
	e := accel.Enumerate()
	for _, info := range e.Devices {
		if info.Backend != accel.BackendMetal {
			continue
		}
		d, err := accel.OpenDevice(info.ID)
		if err != nil {
			b.Fatalf("OpenDevice(%s): %v", info.Name, err)
		}
		b.Cleanup(func() { _ = d.Close() })
		return d
	}
	if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
		b.Fatalf("this job promises Metal and enumerated no adapter: %v", e.Diagnostics)
	}
	b.Skipf("no Metal adapter: %v", e.Diagnostics)
	return nil
}

func f32BufferB(b *testing.B, d *accel.Device, label string, vals []float32) accel.BufferView {
	b.Helper()
	buf, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: len(vals), Label: label,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
	})
	if err != nil {
		b.Fatalf("buffer %s: %v", label, err)
	}
	b.Cleanup(func() { _ = buf.Close() })
	if err := d.Queue().WriteBuffer(buf, 0, vals); err != nil {
		b.Fatalf("write %s: %v", label, err)
	}
	v, err := buf.View(0, len(vals))
	if err != nil {
		b.Fatalf("view %s: %v", label, err)
	}
	return v
}

func u32BufferB(b *testing.B, d *accel.Device, label string, vals []uint32) accel.BufferView {
	b.Helper()
	buf, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.U32, Count: len(vals), Label: label,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
	})
	if err != nil {
		b.Fatalf("buffer %s: %v", label, err)
	}
	b.Cleanup(func() { _ = buf.Close() })
	if err := d.Queue().WriteBuffer(buf, 0, vals); err != nil {
		b.Fatalf("write %s: %v", label, err)
	}
	v, err := buf.View(0, len(vals))
	if err != nil {
		b.Fatalf("view %s: %v", label, err)
	}
	return v
}
