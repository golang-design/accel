// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"fmt"
	"testing"

	"golang.design/x/accel"
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
	p, err := d.NewPool(accel.MemoryDevice, 1<<30)
	if err != nil {
		b.Fatal(err)
	}
	defer p.Close()

	desc := accel.BufferDescriptor{
		DType: accel.F16, Count: 4096, Usage: accel.UsageStorage | accel.UsageCopyDst,
		Label: "weight",
	}

	b.ReportAllocs()
	var live []*accel.Buffer
	for b.Loop() {
		buf, err := p.Alloc(desc)
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
	p, err := d.NewPool(accel.MemoryDevice, 1<<24)
	if err != nil {
		b.Fatal(err)
	}
	defer p.Close()

	desc := accel.BufferDescriptor{DType: accel.F32, Count: 1024, Usage: accel.UsageStorage}

	b.ReportAllocs()
	for b.Loop() {
		buf, err := p.Alloc(desc)
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
	p, err := d.NewPool(accel.MemoryDevice, 1<<20)
	if err != nil {
		b.Fatal(err)
	}
	defer p.Close()
	buf, err := p.Alloc(accel.BufferDescriptor{DType: accel.F16, Count: 8192, Usage: accel.UsageStorage})
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
				p, err := d.NewPool(kind, 1<<22)
				if err != nil {
					b.Fatal(err)
				}
				defer p.Close()
				buf, err := p.Alloc(accel.BufferDescriptor{
					DType: accel.F32, Count: n,
					Usage: accel.UsageCopyDst | accel.UsageCopySrc,
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
	p, err := d.NewPool(accel.MemoryDevice, 1<<22)
	if err != nil {
		b.Fatal(err)
	}
	defer p.Close()
	const n = 1 << 12
	buf, err := p.Alloc(accel.BufferDescriptor{
		DType: accel.F32, Count: n, Usage: accel.UsageCopyDst | accel.UsageCopySrc,
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

	bind := []accel.Binding{{Slot: in, Buffer: benchView(b, src)}}
	b.ReportAllocs()
	for b.Loop() {
		if err := g.Rebind(bind); err != nil {
			b.Fatal(err)
		}
	}
}

func benchBuffer(b *testing.B, d *accel.Device, label string) *accel.Buffer {
	b.Helper()
	buf, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: 1024, Label: label,
		Usage: accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst,
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
