// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel_test

import (
	"testing"

	"golang.design/x/accel/internal/kernel"
)

// BenchmarkSharedTracker is what developer mode pays per shared access: one
// epoch of every invocation writing its own slice of an array, then one of
// every invocation reading across it, which is the publish-then-read shape a
// tiled kernel has.
func BenchmarkSharedTracker(b *testing.B) {
	const n, invocations = 1024, 64
	const per = n / invocations
	tr := kernel.NewSharedTracker("bench", kernel.ID3{}, []int{n})
	b.ReportAllocs()
	for b.Loop() {
		tr.Reset(kernel.ID3{})
		for inv := range invocations {
			tr.Begin(kernel.ID3{X: uint32(inv)})
			for e := inv * per; e < (inv+1)*per; e++ {
				tr.Write(0, e)
			}
		}
		tr.Epoch()
		for inv := range invocations {
			tr.Begin(kernel.ID3{X: uint32(inv)})
			for e := 0; e < n; e += per {
				tr.Read(0, e)
			}
		}
		tr.Epoch()
		if ds := tr.Diagnostics(); len(ds) != 0 {
			b.Fatal(ds)
		}
	}
	b.ReportMetric(float64(n+invocations*(n/per))*float64(b.N)/b.Elapsed().Seconds()/1e6,
		"Maccess/s")
}

// reduceFrame is the shape a generated lowering keeps per invocation: a
// program counter and the locals that survive a suspension.
type reduceFrame struct {
	pc  int
	l   uint32
	sum float32
}

// Reset is what the emitter generates, so the scheduler can reuse the frame.
func (f *reduceFrame) Reset() { *f = reduceFrame{} }

// reduceKernel is a generated-style cooperative kernel: each invocation
// publishes to shared memory, a barrier, and lane 0 reduces the tile. Its
// state lives in a frame the entry point allocates on first call, exactly as
// the emitter's Cooperative wrapper does, which is what makes the allocation
// per invocation per workgroup visible here.
func reduceKernel(size int) *kernel.Kernel {
	return &kernel.Kernel{
		Name: "Reduce", Generator: kernel.ABIVersion,
		WorkgroupSize:    kernel.ID3{X: uint32(size), Y: 1, Z: 1},
		OrderIndependent: true,
		SharedSizes:      []int{size}, Suspensions: 1,
		Bindings: []kernel.Binding{
			{Name: "in", DType: kernel.F32, Access: kernel.Read},
			{Name: "out", DType: kernel.F32, Access: kernel.Write},
		},
		NewShared: func() []any {
			sh := make([]float32, size)
			kernel.Poison(sh)
			return []any{&sh}
		},
		Cooperative: func(t kernel.Thread, a kernel.Args, slot *kernel.Frame) bool {
			f, _ := slot.State.(*reduceFrame)
			if f == nil {
				f = &reduceFrame{}
				slot.State = f
			}
			sh := *kernel.SharedSlice[[]float32](a, 0)
			in := kernel.Slice[float32](a, 0)
			out := kernel.Slice[float32](a, 1)
			switch f.pc {
			case 0:
				f.l = t.LocalIndex()
				slot.Shared.Write(0, int(f.l))
				sh[f.l] = in[t.GlobalIndex()]
				f.pc = 1
				slot.Barrier = kernel.BarrierID{Index: 0}
				return true
			case 1:
				if f.l == 0 {
					f.sum = 0
					for i := range uint32(size) {
						f.sum += sh[slot.Shared.ReadAt(0, int(i))]
					}
					out[t.GroupIndex()] = f.sum
				}
			}
			return false
		},
	}
}

// BenchmarkDispatchCooperative is the scheduler's cost per invocation, with
// and without the instrumentation, on a pool and serially.
func BenchmarkDispatchCooperative(b *testing.B) {
	const size, groups = 64, 256
	k := reduceKernel(size)
	in := make([]float32, size*groups)
	for i := range in {
		in[i] = float32(i)
	}
	args := kernel.Args{Slices: []any{in, make([]float32, groups)}}
	for _, c := range []struct {
		name    string
		diag    bool
		workers int
	}{
		{"diagnostics/serial", true, 1},
		{"diagnostics/pool", true, 0},
		{"unchecked/serial", false, 1},
		{"unchecked/pool", false, 0},
	} {
		b.Run(c.name, func(b *testing.B) {
			opts := kernel.Options{Diagnostics: c.diag, Workers: c.workers}
			b.ReportAllocs()
			for b.Loop() {
				if err := kernel.DispatchCooperativeWith(k, kernel.ID3{X: groups}, args, opts); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(size*groups)*float64(b.N)/b.Elapsed().Seconds()/1e6, "Minv/s")
		})
	}
	// The benchmark is only evidence if the kernel computes what it claims.
	out := args.Slices[1].([]float32)
	if want := float32(size * (size - 1) / 2); out[0] != want {
		b.Fatalf("workgroup 0 reduced to %v, want %v", out[0], want)
	}
}
