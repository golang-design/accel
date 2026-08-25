// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel_test

import (
	"fmt"
	"testing"

	"golang.design/x/accel/internal/kernel"
)

// scaleKernel is the cheapest kernel a dispatch can run: one multiply per
// invocation, no shared memory, no scheduler.
//
// The cheapest one is the one the threshold has to be measured on. A pool's
// overhead is fixed and a workgroup's work is not, so the dispatch size at
// which a pool starts paying for itself is highest when the work per invocation
// is lowest. A threshold that is right here is conservative for every heavier
// kernel.
func scaleKernel(order bool) *kernel.Kernel {
	return &kernel.Kernel{
		Name: "Scale", Generator: kernel.ABIVersion,
		WorkgroupSize:    kernel.ID3{X: 64, Y: 1, Z: 1},
		OrderIndependent: order,
		Bindings: []kernel.Binding{
			{Name: "in", DType: kernel.F32, Access: kernel.Read},
			{Name: "out", DType: kernel.F32, Access: kernel.Write},
		},
		Flat: func(t kernel.Thread, a kernel.Args) {
			in := kernel.Slice[float32](a, 0)
			out := kernel.Slice[float32](a, 1)
			i := t.GlobalID().X
			if i < uint32(len(in)) {
				out[i] = in[i] * 2
			}
		},
	}
}

func scaleArgs(n int) kernel.Args {
	in := make([]float32, n)
	for i := range in {
		in[i] = float32(i)
	}
	return kernel.Args{Slices: []any{in, make([]float32, n)}}
}

// BenchmarkDispatchScale is the elementwise throughput a flat dispatch reaches,
// serially and on a pool, at the sizes the threshold has to separate.
func BenchmarkDispatchScale(b *testing.B) {
	k := scaleKernel(true)
	for _, n := range []int{128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 1 << 20} {
		args := scaleArgs(n)
		count := kernel.ID3{X: uint32(n / 64)}
		for _, w := range []struct {
			name    string
			workers int
		}{{"serial", 1}, {"forced", 8}, {"auto", 0}} {
			workers := w.workers
			b.Run(fmt.Sprintf("n=%d/%s", n, w.name), func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if err := kernel.DispatchWith(k, count, args,
						kernel.Options{Workers: workers}); err != nil {
						b.Fatal(err)
					}
				}
				b.ReportMetric(float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e6, "Melem/s")
			})
		}
	}
}
