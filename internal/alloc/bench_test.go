// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package alloc_test

import (
	"testing"

	"golang.design/x/accel/internal/alloc"
)

// BenchmarkTLSFAlloc is the burst spec 001 section 5.1 says dominates: a model
// loads thousands of weight planes at once and frees none of them until
// shutdown, so the free lists stay short and this is the case TLSF handles best.
func BenchmarkTLSFAlloc(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		b.StopTimer()
		a, err := alloc.NewTLSF(1<<24, granularity)
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		for range 1000 {
			if _, err := a.Alloc(granularity, 4); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkTLSFAllocFree is the churn case: the staging ring recycles a few
// fixed sizes continuously, which is the other half of the workload.
func BenchmarkTLSFAllocFree(b *testing.B) {
	a, err := alloc.NewTLSF(1<<20, granularity)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		al, err := a.Alloc(4096, 4)
		if err != nil {
			b.Fatal(err)
		}
		if err := a.Free(al); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkTLSFAllocFragmented measures placement once the pool has holes,
// which is the state spec 001 section 5.3 says a long-running general pool ends
// up in permanently. A good-fit allocator should not degrade here; a first-fit
// list would.
func BenchmarkTLSFAllocFragmented(b *testing.B) {
	a, err := alloc.NewTLSF(1<<22, granularity)
	if err != nil {
		b.Fatal(err)
	}
	var live []*alloc.Allocation
	for range 2000 {
		al, err := a.Alloc(granularity, 4)
		if err != nil {
			break
		}
		live = append(live, al)
	}
	for i := 0; i < len(live); i += 2 {
		if err := a.Free(live[i]); err != nil {
			b.Fatal(err)
		}
	}

	b.ReportAllocs()
	for b.Loop() {
		al, err := a.Alloc(granularity, 4)
		if err != nil {
			b.Fatal(err)
		}
		if err := a.Free(al); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkTLSFAllocAligned measures the over-allocate-and-trim path a texture
// placement alignment takes, which is far coarser than any buffer alignment.
func BenchmarkTLSFAllocAligned(b *testing.B) {
	a, err := alloc.NewTLSF(1<<24, granularity)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		al, err := a.Alloc(4096, 65536)
		if err != nil {
			b.Fatal(err)
		}
		if err := a.Free(al); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkBumpAlloc is the linear pool, which a graph's transients use because
// spec 003 has already solved placement offline. It is the number that says how
// much a runtime allocator costs by comparison.
func BenchmarkBumpAlloc(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		b.StopTimer()
		a, err := alloc.NewBump(1<<24, granularity)
		if err != nil {
			b.Fatal(err)
		}
		b.StartTimer()
		for range 1000 {
			if _, err := a.Alloc(granularity, 4); err != nil {
				b.Fatal(err)
			}
		}
	}
}

// BenchmarkStats measures the occupancy read, which a caller polls to see
// fragmentation coming. It scans one size class rather than every block, and
// this is the number that says whether that mattered.
func BenchmarkStats(b *testing.B) {
	a, err := alloc.NewTLSF(1<<22, granularity)
	if err != nil {
		b.Fatal(err)
	}
	var live []*alloc.Allocation
	for range 2000 {
		al, err := a.Alloc(granularity, 4)
		if err != nil {
			break
		}
		live = append(live, al)
	}
	for i := 0; i < len(live); i += 2 {
		a.Free(live[i])
	}

	b.ReportAllocs()
	for b.Loop() {
		if s := a.Stats(); s.Size == 0 {
			b.Fatal("empty stats")
		}
	}
}
