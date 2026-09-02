// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package alloc_test

import (
	"math/rand/v2"
	"testing"

	"golang.design/x/accel/internal/alloc"
)

// FuzzTLSF is spec 001 section 11.4's allocator fuzz: random allocate and free
// sequences never overlap two live allocations, never exceed the pool, and
// always coalesce back to one free block when everything is freed.
//
// Overlap is checked two ways. The offsets are checked directly, and a
// per-allocation byte pattern is written and read back, which catches a placement
// that is arithmetically disjoint but whose *size* the allocator got wrong.
func FuzzTLSF(f *testing.F) {
	f.Add(uint64(1), uint(1<<20), uint(256), uint(200))
	f.Add(uint64(7), uint(1<<16), uint(256), uint(500))
	f.Add(uint64(99), uint(1<<18), uint(16), uint(300))
	f.Add(uint64(1234), uint(1<<14), uint(4096), uint(50))

	f.Fuzz(func(t *testing.T, seed uint64, rawSize, rawGran, rawOps uint) {
		granularity := 1 << (rawGran%9 + 2) // 4 .. 1024
		size := int(rawSize%(1<<20)) + granularity
		ops := int(rawOps % 2000)

		a, err := alloc.NewTLSF(size, granularity)
		if err != nil {
			t.Skip() // rejected construction is its own test
		}

		rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b9))
		memory := make([]byte, size)

		type live struct {
			al      *alloc.Allocation
			pattern byte
		}
		var held []live
		var next byte = 1
		// The bytes the test believes are allocated, kept independently of
		// the allocator's own counter so the statistics are checked against
		// something other than themselves.
		liveBytes := 0

		for range ops {
			if len(held) > 0 && rng.IntN(3) == 0 {
				i := rng.IntN(len(held))
				h := held[i]
				held = append(held[:i], held[i+1:]...)

				// The bytes must still read back as this allocation's pattern:
				// nothing else was ever allowed to write here.
				for _, b := range memory[h.al.Offset : h.al.Offset+h.al.Size] {
					if b != h.pattern {
						t.Fatalf("allocation at [%d, %d) was overwritten: saw %d, want %d",
							h.al.Offset, h.al.Offset+h.al.Size, b, h.pattern)
					}
				}
				clear(memory[h.al.Offset : h.al.Offset+h.al.Size])
				if err := a.Free(h.al); err != nil {
					t.Fatalf("Free: %v", err)
				}
				liveBytes -= h.al.Size
				if s := a.Stats(); s.Used != liveBytes || s.Free != size-liveBytes {
					t.Fatalf("after a free with %d bytes live, stats are %+v", liveBytes, s)
				}
				continue
			}

			want := 1 + rng.IntN(size)
			align := 1 << rng.IntN(6) // 1 .. 32, raised to the granularity
			al, err := a.Alloc(want, align)
			if err != nil {
				continue // exhaustion and fragmentation are legal outcomes
			}

			switch {
			case al.Offset < 0 || al.Offset+al.Size > size:
				t.Fatalf("allocation [%d, %d) is outside the %d-byte pool",
					al.Offset, al.Offset+al.Size, size)
			case al.Offset%granularity != 0:
				t.Fatalf("offset %d is not a multiple of the %d granularity", al.Offset, granularity)
			case al.Offset%align != 0:
				t.Fatalf("offset %d does not satisfy the requested alignment %d", al.Offset, align)
			case al.Size < want:
				t.Fatalf("allocation size %d is below the %d requested", al.Size, want)
			}

			for _, b := range memory[al.Offset : al.Offset+al.Size] {
				if b != 0 {
					t.Fatalf("allocation at [%d, %d) overlaps a live one holding pattern %d",
						al.Offset, al.Offset+al.Size, b)
				}
			}
			pattern := next
			next++
			if next == 0 {
				next = 1
			}
			for i := al.Offset; i < al.Offset+al.Size; i++ {
				memory[i] = pattern
			}
			held = append(held, live{al: al, pattern: pattern})
			liveBytes += al.Size

			if s := a.Stats(); s.Used != liveBytes || s.Free != size-liveBytes ||
				s.LargestFree > s.Free || s.Allocations != len(held) {
				t.Fatalf("with %d bytes in %d allocations live, stats are %+v",
					liveBytes, len(held), s)
			}
		}

		for _, h := range held {
			if err := a.Free(h.al); err != nil {
				t.Fatalf("draining the pool: %v", err)
			}
		}

		s := a.Stats()
		if s.Used != 0 || s.Allocations != 0 {
			t.Fatalf("after freeing everything: Used = %d, Allocations = %d", s.Used, s.Allocations)
		}
		if s.Blocks != 1 {
			t.Fatalf("after freeing everything the pool has %d free blocks, want 1: coalescing is incomplete", s.Blocks)
		}
	})
}

// FuzzBump checks the linear policy's much smaller contract: offsets only
// advance, nothing overlaps, and the cursor never passes the end.
func FuzzBump(f *testing.F) {
	f.Add(uint64(1), uint(1<<16), uint(256), uint(100))
	f.Add(uint64(5), uint(1<<12), uint(4), uint(400))

	f.Fuzz(func(t *testing.T, seed uint64, rawSize, rawGran, rawOps uint) {
		granularity := 1 << (rawGran%9 + 2)
		size := int(rawSize%(1<<20)) + granularity

		a, err := alloc.NewBump(size, granularity)
		if err != nil {
			t.Skip()
		}

		rng := rand.New(rand.NewPCG(seed, seed^0x5851f42d))
		var held []*alloc.Allocation
		end := 0

		for range int(rawOps % 2000) {
			if rng.IntN(20) == 0 {
				if err := a.Reset(); err != nil {
					t.Fatalf("Reset: %v", err)
				}
				held, end = held[:0], 0
				continue
			}
			al, err := a.Alloc(1+rng.IntN(size), 1<<rng.IntN(13)) // 1 .. 4096, above the granularity too, so padding happens
			if err != nil {
				continue
			}
			if al.Offset < end {
				t.Fatalf("a bump moved backwards: placed at %d with the cursor at %d", al.Offset, end)
			}
			if al.Offset+al.Size > size {
				t.Fatalf("allocation [%d, %d) is outside the %d-byte pool", al.Offset, al.Offset+al.Size, size)
			}
			end = al.Offset + al.Size
			held = append(held, al)

			// A bump charges alignment padding to Used, so what is used is
			// exactly where the cursor is, and everything past it is one free
			// block. The cursor is the test's own, advanced from the offsets
			// the allocator returned.
			if s := a.Stats(); s.Used != end || s.Free != size-end ||
				s.LargestFree != size-end || s.Allocations != len(held) {
				t.Fatalf("with the cursor at %d of %d after %d allocations, stats are %+v",
					end, size, len(held), s)
			}
		}

		if err := checkDisjoint(held, size); err != nil {
			t.Fatal(err)
		}
	})
}
