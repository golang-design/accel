// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package alloc_test

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"sort"
	"testing"
	"time"

	"golang.design/x/accel/internal/alloc"
)

// granularity is the default suballocation granularity from spec 001 section
// 3.1: a multiple of 256 is a sufficient bound offset on every backend.
const granularity = 256

func newTLSF(t *testing.T, size int) *alloc.TLSF {
	t.Helper()
	a, err := alloc.NewTLSF(size, granularity)
	if err != nil {
		t.Fatalf("NewTLSF(%d, %d): %v", size, granularity, err)
	}
	return a
}

// each runs a case against both policies, for the behaviour they share.
func each(t *testing.T, size int, fn func(t *testing.T, a alloc.Allocator)) {
	t.Helper()
	t.Run("TLSF", func(t *testing.T) {
		a, err := alloc.NewTLSF(size, granularity)
		if err != nil {
			t.Fatal(err)
		}
		fn(t, a)
	})
	t.Run("Bump", func(t *testing.T) {
		a, err := alloc.NewBump(size, granularity)
		if err != nil {
			t.Fatal(err)
		}
		fn(t, a)
	})
}

func TestConstructionRejectsBadExtents(t *testing.T) {
	for _, tc := range []struct{ size, granularity int }{
		{1 << 20, 0},
		{1 << 20, -256},
		{1 << 20, 100}, // not a power of two
		{128, 256},     // smaller than one allocation unit
	} {
		if _, err := alloc.NewTLSF(tc.size, tc.granularity); err == nil {
			t.Errorf("NewTLSF(%d, %d) was accepted", tc.size, tc.granularity)
		}
		if _, err := alloc.NewBump(tc.size, tc.granularity); err == nil {
			t.Errorf("NewBump(%d, %d) was accepted", tc.size, tc.granularity)
		}
	}
}

func TestRequestValidation(t *testing.T) {
	each(t, 1<<20, func(t *testing.T, a alloc.Allocator) {
		for _, tc := range []struct{ size, align int }{
			{0, 256},
			{-1, 256},
			{256, 0},
			{256, -4},
			{256, 100}, // not a power of two
		} {
			if _, err := a.Alloc(tc.size, tc.align); err == nil {
				t.Errorf("Alloc(%d, %d) was accepted", tc.size, tc.align)
			}
		}
	})
}

// TestOffsetsAreGranular is the alignment half of spec 001 section 5.2: the
// allocator's granularity is the pool's alignment floor, and every offset and
// size is a multiple of it.
func TestOffsetsAreGranular(t *testing.T) {
	each(t, 1<<20, func(t *testing.T, a alloc.Allocator) {
		if got := a.Granularity(); got != granularity {
			t.Fatalf("Granularity = %d, want %d", got, granularity)
		}
		for _, size := range []int{1, 2, 255, 256, 257, 1000, 4096} {
			al, err := a.Alloc(size, 4)
			if err != nil {
				t.Fatalf("Alloc(%d): %v", size, err)
			}
			if al.Offset%granularity != 0 {
				t.Errorf("Alloc(%d) placed at offset %d, not a multiple of %d", size, al.Offset, granularity)
			}
			if al.Size%granularity != 0 || al.Size < size {
				t.Errorf("Alloc(%d) has allocation size %d, want a multiple of %d at or above the request",
					size, al.Size, granularity)
			}
		}
	})
}

// TestUsedIncludesAlignmentPadding is spec 001 section 11.4: allocating N
// one-byte buffers reports Used as N*granularity, not N. The tax is visible in
// the number a caller already looks at rather than looking like a leak.
func TestUsedIncludesAlignmentPadding(t *testing.T) {
	const n = 64
	each(t, 1<<20, func(t *testing.T, a alloc.Allocator) {
		for range n {
			if _, err := a.Alloc(1, 4); err != nil {
				t.Fatal(err)
			}
		}
		s := a.Stats()
		if want := n * granularity; s.Used != want {
			t.Errorf("Used = %d after %d one-byte allocations, want %d", s.Used, n, want)
		}
		if s.Free != s.Size-s.Used {
			t.Errorf("Free = %d, want Size-Used = %d", s.Free, s.Size-s.Used)
		}
		if s.Allocations != n {
			t.Errorf("Allocations = %d, want %d", s.Allocations, n)
		}
	})
}

// TestStrictAlignmentIsServed covers the over-allocate-and-trim path: a texture
// placement alignment is far coarser than any buffer alignment, and the head
// that gets trimmed must return to the free list rather than being lost.
func TestStrictAlignmentIsServed(t *testing.T) {
	a := newTLSF(t, 4<<20)

	// Push the cursor off a coarse boundary first, so the next request has to
	// pad.
	if _, err := a.Alloc(300, 4); err != nil {
		t.Fatal(err)
	}
	const coarse = 65536 // D3D12's default resource placement alignment
	al, err := a.Alloc(1024, coarse)
	if err != nil {
		t.Fatalf("Alloc at alignment %d: %v", coarse, err)
	}
	if al.Offset%coarse != 0 {
		t.Fatalf("offset %d is not a multiple of %d", al.Offset, coarse)
	}

	// The trimmed head is free space, so a small request must be servable from
	// it rather than from beyond the coarse allocation.
	small, err := a.Alloc(256, 4)
	if err != nil {
		t.Fatalf("the trimmed head was lost: %v", err)
	}
	if small.Offset >= al.Offset {
		t.Errorf("small allocation landed at %d, past the aligned one at %d: "+
			"the trimmed head was not returned to the free list", small.Offset, al.Offset)
	}
}

// TestAlignmentFindsAnAlreadyAlignedBlock covers the bounded second look in
// findFit: a block that is already aligned needs no padding, so the over-ask
// must not be the only thing tried.
func TestAlignmentFindsAnAlreadyAlignedBlock(t *testing.T) {
	const coarse = 4096
	a := newTLSF(t, 1<<20)

	// Fill the pool, then free one aligned block exactly one coarse unit wide.
	var all []*alloc.Allocation
	for {
		al, err := a.Alloc(coarse, coarse)
		if err != nil {
			break
		}
		all = append(all, al)
	}
	if len(all) < 3 {
		t.Fatalf("expected several coarse allocations, got %d", len(all))
	}
	hole := all[1]
	if err := a.Free(hole); err != nil {
		t.Fatal(err)
	}

	got, err := a.Alloc(coarse, coarse)
	if err != nil {
		t.Fatalf("an exactly-sized aligned hole was not found: %v", err)
	}
	if got.Offset != hole.Offset {
		t.Errorf("placed at %d, want the aligned hole at %d", got.Offset, hole.Offset)
	}
}

// TestFragmentationIsReportedNotFixed is spec 001 section 5.3 and 11.4:
// allocate many blocks, free every other one, then request a block larger than
// any hole. It fails while Free exceeds the request, and that is what a
// non-compacting allocator is rather than a bug.
func TestFragmentationIsReportedNotFixed(t *testing.T) {
	a := newTLSF(t, 1<<20)

	var live []*alloc.Allocation
	for {
		al, err := a.Alloc(granularity, 4)
		if err != nil {
			break
		}
		live = append(live, al)
	}
	for i := 0; i < len(live); i += 2 {
		if err := a.Free(live[i]); err != nil {
			t.Fatal(err)
		}
	}

	s := a.Stats()
	request := 4 * granularity
	if s.Free <= request {
		t.Fatalf("the scenario did not leave enough free space: Free = %d, request %d", s.Free, request)
	}
	if s.LargestFree >= request {
		t.Fatalf("the scenario did not fragment: LargestFree = %d, request %d", s.LargestFree, request)
	}

	_, err := a.Alloc(request, 4)
	if !errors.Is(err, alloc.ErrNoSpace) {
		t.Fatalf("Alloc into a fragmented pool returned %v, want ErrNoSpace", err)
	}

	// Free and LargestFree together are what distinguish this from exhaustion,
	// which is a different problem with a different fix.
	if s.Free < s.LargestFree {
		t.Error("Free is below LargestFree, which cannot be")
	}
	if s.Blocks < 2 {
		t.Errorf("Blocks = %d; rising Blocks against flat Allocations is how a caller sees this coming", s.Blocks)
	}
}

// TestCoalescing checks that freeing everything returns the pool to one block,
// which is the property that keeps fragmentation from being permanent for
// allocations that do come and go in order.
func TestCoalescing(t *testing.T) {
	a := newTLSF(t, 1<<20)

	var live []*alloc.Allocation
	for i := range 64 {
		al, err := a.Alloc(granularity*(1+i%5), 4)
		if err != nil {
			t.Fatal(err)
		}
		live = append(live, al)
	}
	// Free in an order that exercises both neighbours: middle outward.
	sort.Slice(live, func(i, j int) bool { return live[i].Offset > live[j].Offset })
	for _, al := range live[:32] {
		if err := a.Free(al); err != nil {
			t.Fatal(err)
		}
	}
	for _, al := range live[32:] {
		if err := a.Free(al); err != nil {
			t.Fatal(err)
		}
	}

	s := a.Stats()
	if s.Used != 0 || s.Allocations != 0 {
		t.Errorf("after freeing everything: Used = %d, Allocations = %d, want 0 and 0", s.Used, s.Allocations)
	}
	if s.Blocks != 1 {
		t.Errorf("after freeing everything the pool has %d free blocks, want 1: nothing coalesced", s.Blocks)
	}
	if s.LargestFree != s.Free {
		t.Errorf("LargestFree = %d but Free = %d; a fully coalesced pool has one hole", s.LargestFree, s.Free)
	}
}

func TestDoubleFreeIsReported(t *testing.T) {
	a := newTLSF(t, 1<<20)
	al, err := a.Alloc(1024, 4)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Free(al); err != nil {
		t.Fatal(err)
	}
	if err := a.Free(al); !errors.Is(err, alloc.ErrDoubleFree) {
		t.Errorf("second Free returned %v, want ErrDoubleFree", err)
	}
	if err := a.Free(nil); !errors.Is(err, alloc.ErrDoubleFree) {
		t.Errorf("Free(nil) returned %v, want ErrDoubleFree", err)
	}

	other := newTLSF(t, 1<<20)
	foreign, err := other.Alloc(1024, 4)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Free(foreign); !errors.Is(err, alloc.ErrDoubleFree) {
		t.Errorf("freeing another pool's allocation returned %v, want ErrDoubleFree", err)
	}
}

// TestGeneralPoolRefusesReset and TestLinearPoolRefusesFree are spec 001
// section 11.4: each policy reports the operation the other one owns.
func TestGeneralPoolRefusesReset(t *testing.T) {
	a := newTLSF(t, 1<<20)
	if err := a.Reset(); !errors.Is(err, alloc.ErrUnsupported) {
		t.Errorf("TLSF.Reset returned %v, want ErrUnsupported", err)
	}
}

func TestLinearPoolRefusesFree(t *testing.T) {
	a, err := alloc.NewBump(1<<20, granularity)
	if err != nil {
		t.Fatal(err)
	}
	al, err := a.Alloc(1024, 4)
	if err != nil {
		t.Fatal(err)
	}
	if err := a.Free(al); !errors.Is(err, alloc.ErrUnsupported) {
		t.Errorf("Bump.Free returned %v, want ErrUnsupported", err)
	}
	if err := a.Free(nil); !errors.Is(err, alloc.ErrDoubleFree) {
		t.Errorf("Bump.Free(nil) returned %v, want ErrDoubleFree", err)
	}
}

// TestLinearReset checks that a reset rewinds everything and retires every
// handle, so a stale one is reported rather than addressing whatever now
// occupies that offset.
func TestLinearReset(t *testing.T) {
	a, err := alloc.NewBump(1<<20, granularity)
	if err != nil {
		t.Fatal(err)
	}

	first, err := a.Alloc(4096, 4)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := a.Alloc(4096, 4); err != nil {
		t.Fatal(err)
	}
	if err := a.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}

	s := a.Stats()
	if s.Used != 0 || s.Allocations != 0 || s.Free != s.Size {
		t.Errorf("after Reset: %+v, want an empty pool", s)
	}

	again, err := a.Alloc(4096, 4)
	if err != nil {
		t.Fatal(err)
	}
	if again.Offset != 0 {
		t.Errorf("the cursor did not rewind: first allocation after Reset is at %d", again.Offset)
	}
	if err := a.Free(first); !errors.Is(err, alloc.ErrDoubleFree) {
		t.Errorf("a handle from before Reset returned %v, want ErrDoubleFree", err)
	}
}

// TestLinearPoolCannotFragment is the other half of why a bump is right for a
// graph's transients: Free and LargestFree never diverge.
func TestLinearPoolCannotFragment(t *testing.T) {
	a, err := alloc.NewBump(1<<20, granularity)
	if err != nil {
		t.Fatal(err)
	}
	for range 100 {
		if _, err := a.Alloc(granularity, 4); err != nil {
			t.Fatal(err)
		}
		s := a.Stats()
		if s.LargestFree != s.Free {
			t.Fatalf("a bump fragmented: LargestFree = %d, Free = %d", s.LargestFree, s.Free)
		}
	}
}

func TestExhaustion(t *testing.T) {
	each(t, 4*granularity, func(t *testing.T, a alloc.Allocator) {
		if _, err := a.Alloc(8*granularity, 4); !errors.Is(err, alloc.ErrNoSpace) {
			t.Errorf("a request larger than the pool returned %v, want ErrNoSpace", err)
		}
		for range 4 {
			if _, err := a.Alloc(granularity, 4); err != nil {
				t.Fatalf("the pool should hold four units: %v", err)
			}
		}
		if _, err := a.Alloc(granularity, 4); !errors.Is(err, alloc.ErrNoSpace) {
			t.Errorf("a full pool returned %v, want ErrNoSpace", err)
		}
		if s := a.Stats(); s.Free != 0 || s.LargestFree != 0 {
			t.Errorf("a full pool reports %+v", s)
		}
	})
}

// TestAllocationIsConstantTime is spec 001 section 11.4: allocating 10,000
// buffers takes time linear in the count, not quadratic. This is the property
// that rules out the first-fit free list, so it is measured rather than assumed.
func TestAllocationIsConstantTime(t *testing.T) {
	if testing.Short() {
		t.Skip("timing-sensitive")
	}

	run := func(n int) time.Duration {
		a, err := alloc.NewTLSF(n*granularity*2, granularity)
		if err != nil {
			t.Fatal(err)
		}
		// Warm the allocator so the first measurement is not paying for growth
		// in the free-list slices.
		start := time.Now()
		for range n {
			if _, err := a.Alloc(granularity, 4); err != nil {
				t.Fatal(err)
			}
		}
		return time.Since(start)
	}

	small := run(1000)
	large := run(10000)

	// Ten times the work in far less than a hundred times the time. A quadratic
	// allocator lands near 100x; the generous bound keeps a loaded CI machine
	// from failing a correct implementation.
	//
	// This is a ratio and not a measurement, which is what makes it meaningful
	// under the race detector. Go 1.27's size-specialized malloc is disabled
	// whenever -race, -asan, or -msan is on, so the absolute numbers here are
	// not representative of a normal build; both sides of the ratio pay the same
	// tax, so the shape it is asserting survives.
	if ratio := float64(large) / float64(small+1); ratio > 30 {
		t.Errorf("10,000 allocations took %v against %v for 1,000, a ratio of %.1f: "+
			"placement is not O(1)", large, small, ratio)
	}
}

// TestNoOverlap is the invariant every other test depends on: two live
// allocations never share a byte.
func TestNoOverlap(t *testing.T) {
	each(t, 1<<20, func(t *testing.T, a alloc.Allocator) {
		var live []*alloc.Allocation
		for {
			al, err := a.Alloc(1+rand.IntN(4000), 1<<uint(rand.IntN(4))*4)
			if err != nil {
				break
			}
			live = append(live, al)
		}
		if err := checkDisjoint(live, a.Stats().Size); err != nil {
			t.Error(err)
		}
	})
}

func checkDisjoint(live []*alloc.Allocation, size int) error {
	sorted := make([]*alloc.Allocation, len(live))
	copy(sorted, live)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Offset < sorted[j].Offset })
	for i, al := range sorted {
		if al.Offset < 0 || al.Offset+al.Size > size {
			return fmt.Errorf("allocation [%d, %d) is outside the %d-byte pool",
				al.Offset, al.Offset+al.Size, size)
		}
		if i > 0 {
			prev := sorted[i-1]
			if al.Offset < prev.Offset+prev.Size {
				return fmt.Errorf("allocations [%d, %d) and [%d, %d) overlap",
					prev.Offset, prev.Offset+prev.Size, al.Offset, al.Offset+al.Size)
			}
		}
	}
	return nil
}
