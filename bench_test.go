// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"fmt"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

func benchDevice(b *testing.B) *accel.Device {
	b.Helper()
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { d.Close() })
	return d
}

// BenchmarkPoolAlloc is the number spec 001 section 5.1's workload cares about:
// a model loads thousands of weight planes in one burst, and this is what each
// one costs.
func BenchmarkPoolAlloc(b *testing.B) {
	d := benchDevice(b)
	p, err := d.NewPool(accel.PoolDescriptor{Kind: accel.MemoryDevice, Bytes: 1 << 30})
	if err != nil {
		b.Fatal(err)
	}
	defer p.Close()

	desc := accel.BufferDescriptor{
		DType: accel.F16, Count: 4096, Usage: accel.BufferStorage | accel.BufferCopyDst,
		Label: "weight",
	}

	b.ReportAllocs()
	var live []*accel.Buffer
	for b.Loop() {
		buf, err := p.AllocBuffer(desc)
		if err != nil {
			b.StopTimer()
			for _, x := range live {
				x.Close()
			}
			live = live[:0]
			b.StartTimer()
			continue
		}
		live = append(live, buf)
	}
	b.StopTimer()
	for _, x := range live {
		x.Close()
	}
}

// BenchmarkPoolAllocFree measures the churn case, which is the staging ring and
// the render-target rebuild rather than the weight burst.
func BenchmarkPoolAllocFree(b *testing.B) {
	d := benchDevice(b)
	p, err := d.NewPool(accel.PoolDescriptor{Kind: accel.MemoryDevice, Bytes: 1 << 24})
	if err != nil {
		b.Fatal(err)
	}
	defer p.Close()

	desc := accel.BufferDescriptor{DType: accel.F32, Count: 1024, Usage: accel.BufferStorage}

	b.ReportAllocs()
	for b.Loop() {
		buf, err := p.AllocBuffer(desc)
		if err != nil {
			b.Fatal(err)
		}
		if err := buf.Close(); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkView measures slicing, which spec 007 does by the thousand: a view
// is meant to be free enough to cache one per attention head.
func BenchmarkView(b *testing.B) {
	d := benchDevice(b)
	p, err := d.NewPool(accel.PoolDescriptor{Kind: accel.MemoryDevice, Bytes: 1 << 20})
	if err != nil {
		b.Fatal(err)
	}
	defer p.Close()
	buf, err := p.AllocBuffer(accel.BufferDescriptor{DType: accel.F16, Count: 8192, Usage: accel.BufferStorage})
	if err != nil {
		b.Fatal(err)
	}
	defer buf.Close()

	b.ReportAllocs()
	for b.Loop() {
		if _, err := buf.View(128, 128); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkTransfer measures the two memory kinds against each other, which is
// the whole argument for MemoryShared being a kind rather than a hint: on
// unified memory the copy vanishes, and this is where that shows up as a number.
func BenchmarkTransfer(b *testing.B) {
	for _, kind := range []accel.MemoryKind{accel.MemoryDevice, accel.MemoryShared} {
		for _, n := range []int{1 << 10, 1 << 16} {
			b.Run(fmt.Sprintf("%v/%dKiB", kind, n*4>>10), func(b *testing.B) {
				d := benchDevice(b)
				p, err := d.NewPool(accel.PoolDescriptor{Kind: kind, Bytes: 1 << 22})
				if err != nil {
					b.Fatal(err)
				}
				defer p.Close()
				buf, err := p.AllocBuffer(accel.BufferDescriptor{
					DType: accel.F32, Count: n,
					Usage: accel.BufferCopyDst | accel.BufferCopySrc,
				})
				if err != nil {
					b.Fatal(err)
				}
				defer buf.Close()

				data := make([]float32, n)
				q := d.Queue()

				b.SetBytes(int64(n * 4))
				b.ReportAllocs()
				for b.Loop() {
					if err := q.WriteBuffer(buf, 0, data); err != nil {
						b.Fatal(err)
					}
					if err := q.Flush().Wait(); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkReadBuffer measures the blocking path, which spec 001 documents as
// wrong in a hot loop. The number is here so that claim has evidence.
func BenchmarkReadBuffer(b *testing.B) {
	d := benchDevice(b)
	p, err := d.NewPool(accel.PoolDescriptor{Kind: accel.MemoryDevice, Bytes: 1 << 22})
	if err != nil {
		b.Fatal(err)
	}
	defer p.Close()
	const n = 1 << 12
	buf, err := p.AllocBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: n, Usage: accel.BufferCopyDst | accel.BufferCopySrc,
	})
	if err != nil {
		b.Fatal(err)
	}
	defer buf.Close()

	into := make([]float32, n)
	q := d.Queue()
	b.SetBytes(int64(n * 4))
	b.ReportAllocs()
	for b.Loop() {
		if err := q.ReadBuffer(buf, 0, into); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkOpenCPU measures device open, which every test in the suite pays.
func BenchmarkOpenCPU(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		d, err := accel.OpenCPU(accel.CPUOptions{})
		if err != nil {
			b.Fatal(err)
		}
		if err := d.Close(); err != nil {
			b.Fatal(err)
		}
	}
}

// Build cost and submit cost are separated because specs/003-command-graph.md's
// claim is that the second is small: validation, planning and lowering happen
// once, and a submission after that is replay. A number for each is what makes
// specs/016-graph-execution.md's added planning cost visible rather than
// absorbed.
func BenchmarkGraphBuild(b *testing.B) {
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	defer d.Close()
	dst, src := benchBuffer(b, d, "dst"), benchBuffer(b, d, "src")

	b.ReportAllocs()
	for b.Loop() {
		r := d.NewRecorder()
		for range 32 {
			r.CopyBuffer(benchView(b, dst), benchView(b, src))
		}
		g, err := r.Build()
		if err != nil {
			b.Fatal(err)
		}
		if err := g.Close(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGraphSubmit(b *testing.B) {
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	defer d.Close()
	dst, src := benchBuffer(b, d, "dst"), benchBuffer(b, d, "src")

	r := d.NewRecorder()
	for range 32 {
		r.CopyBuffer(benchView(b, dst), benchView(b, src))
	}
	g, err := r.Build()
	if err != nil {
		b.Fatal(err)
	}
	defer g.Close()

	q := d.Queue()
	b.ReportAllocs()
	for b.Loop() {
		if err := q.Submit(g).Wait(); err != nil {
			b.Fatal(err)
		}
	}
}

// Rebinding is the hot path a replayable graph exists for, so its cost is
// measured rather than assumed cheap.
func BenchmarkGraphRebind(b *testing.B) {
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	defer d.Close()
	dst, src := benchBuffer(b, d, "dst"), benchBuffer(b, d, "src")

	r := d.NewRecorder()
	in := r.Slot(accel.SlotDescriptor{
		Name: "in", Kind: accel.BindingStorageBuffer,
		DType: accel.F32, Access: accel.AccessRead, MinCount: 1024,
	})
	for range 8 {
		r.CopyFromSlot(benchView(b, dst), in, 0, 1024)
	}
	g, err := r.Build()
	if err != nil {
		b.Fatal(err)
	}
	defer g.Close()

	bind := []accel.SlotBinding{{Slot: in, Buffer: benchView(b, src)}}
	b.ReportAllocs()
	for b.Loop() {
		if err := g.Bind(bind...); err != nil {
			b.Fatal(err)
		}
	}
}

func benchBuffer(b *testing.B, d *accel.Device, label string) *accel.Buffer {
	b.Helper()
	buf, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: 1024, Label: label,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
	})
	if err != nil {
		b.Fatalf("buffer %q: %v", label, err)
	}
	b.Cleanup(func() { _ = buf.Close() })
	return buf
}

func benchView(b *testing.B, buf *accel.Buffer) accel.BufferView {
	b.Helper()
	v, err := buf.View(0, buf.Count())
	if err != nil {
		b.Fatalf("view: %v", err)
	}
	return v
}

// Inference and barrier planning against node count. Spec 003 claims the cost
// is O(N·k·Ā) with Ā small, and that claim needs a number rather than an
// argument — especially since it is what a caller pays at build to save at
// every submission.
func BenchmarkGraphPlanning(b *testing.B) {
	for _, n := range []int{16, 64, 256} {
		b.Run(fmt.Sprintf("nodes=%d", n), func(b *testing.B) {
			d, err := accel.OpenCPU(accel.CPUOptions{})
			if err != nil {
				b.Fatalf("open: %v", err)
			}
			defer d.Close()

			// A chain, which is the shape with the most hazards per node: each
			// node reads what the last one wrote.
			bufs := make([]*accel.Buffer, 8)
			for i := range bufs {
				bufs[i] = benchBuffer(b, d, fmt.Sprintf("b%d", i))
			}

			b.ReportAllocs()
			for b.Loop() {
				r := d.NewRecorder()
				for i := range n {
					dst, src := bufs[i%len(bufs)], bufs[(i+1)%len(bufs)]
					r.CopyBuffer(benchView(b, dst), benchView(b, src))
				}
				g, err := r.Build()
				if err != nil {
					b.Fatal(err)
				}
				if err := g.Close(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// A dispatch through a graph against the same kernel run directly. The gap is
// what the graph costs per submission, and spec 003's claim is that it is
// small: everything expensive happened at build.
func BenchmarkDispatchThroughAGraph(b *testing.B) {
	const n = 4096
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	defer d.Close()

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "add",
	})
	if err != nil {
		b.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	mk := func(label string) *accel.Buffer {
		buf, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: n, Usage: storage, Label: label,
		})
		if err != nil {
			b.Fatalf("buffer: %v", err)
		}
		b.Cleanup(func() { _ = buf.Close() })
		return buf
	}
	in, out := mk("in"), mk("out")

	r := d.NewRecorder()
	r.Dispatch(p, []accel.Binding{
		{Index: 0, Buffer: benchView(b, in)},
		{Index: 1, Buffer: benchView(b, in)},
		{Index: 2, Buffer: benchView(b, out)},
	}, nil, accel.WorkgroupCount{X: n / 64})
	g, err := r.Build()
	if err != nil {
		b.Fatalf("build: %v", err)
	}
	defer g.Close()

	q := d.Queue()
	b.ReportAllocs()
	for b.Loop() {
		if err := q.Submit(g).Wait(); err != nil {
			b.Fatal(err)
		}
	}
}

// Packing cost against transient count. Spec 003 claims O(n² log n), dominated
// by the scan of placed transients per placement, and calls it acceptable at
// 200 transients — a claim that needs a number rather than an argument, since
// it is paid at build for every graph a caller keeps.
func BenchmarkTransientPacking(b *testing.B) {
	for _, n := range []int{8, 32, 128} {
		b.Run(fmt.Sprintf("transients=%d", n), func(b *testing.B) {
			d, err := accel.OpenCPU(accel.CPUOptions{})
			if err != nil {
				b.Fatalf("open: %v", err)
			}
			defer d.Close()
			in := benchBuffer(b, d, "in")

			b.ReportAllocs()
			for b.Loop() {
				r := d.NewRecorder()
				// A chain, so every transient is compatible with every other
				// one it does not share a node with: the case where the
				// compatibility scan does the most work.
				prev := benchView(b, in)
				for range n {
					v := r.Transient(accel.BufferDescriptor{
						DType: accel.F32, Count: 1024, // matches benchBuffer
						Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
					})
					r.CopyBuffer(v, prev)
					prev = v
				}
				r.CopyBuffer(benchView(b, in), prev)
				g, err := r.Build()
				if err != nil {
					b.Fatal(err)
				}
				if err := g.Close(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
