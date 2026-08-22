// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package alloc

import "fmt"

// Bump allocates by moving a cursor and frees only by resetting the whole
// extent: a linear pool.
//
// It is twenty lines against TLSF's two hundred and fifty, with zero internal
// fragmentation and no free operation at all, and that trade is right for
// exactly one workload: a graph's transient pool. Spec 003 computes every
// transient's offset at build time by packing an interference relation, so by
// the time the pool exists placement is already solved and there is nothing for
// a runtime allocator to decide. Its contents also die together, which is the
// other half of what makes a bump correct there and wrong anywhere a caller
// closes one resource at a time.
//
// A Bump is not safe for concurrent use; the pool that owns one serialises it.
type Bump struct {
	allocator

	cursor int
	live   []*Allocation // handles to retire on Reset
}

// NewBump returns a linear allocator over [0, size) placing at multiples of
// granularity.
func NewBump(size, granularity int) (*Bump, error) {
	if err := checkGranularity(size, granularity); err != nil {
		return nil, err
	}
	return &Bump{allocator: allocator{size: size, granularity: granularity}}, nil
}

// Alloc places size bytes at a multiple of align by advancing the cursor.
func (b *Bump) Alloc(size, align int) (*Allocation, error) {
	size, align, err := b.checkParams(size, align)
	if err != nil {
		return nil, err
	}

	offset := roundUp(b.cursor, align)
	if offset+size > b.size {
		return nil, fmt.Errorf("%w: %d bytes at alignment %d leaves %d of %d",
			ErrNoSpace, size, align, b.size-b.cursor, b.size)
	}

	// The alignment padding is charged to Used, because it is space the pool can
	// no longer hand out and spec 001 section 3.4 wants that tax visible in the
	// number a caller already looks at.
	b.used += offset + size - b.cursor
	b.cursor = offset + size
	b.allocations++

	a := &Allocation{Offset: offset, Size: size, owner: &b.allocator}
	b.live = append(b.live, a)
	return a, nil
}

// Free reports ErrUnsupported. A linear pool gives up individual frees in
// exchange for its bump, and reporting that is better than silently accepting a
// call that does nothing: a caller who reaches for it wanted a general pool.
func (b *Bump) Free(a *Allocation) error {
	if err := b.checkOwner(a); err != nil {
		return err
	}
	return fmt.Errorf("%w: a linear pool frees by Reset, not one allocation at a time", ErrUnsupported)
}

// Reset releases every allocation at once and rewinds the cursor. Every handle
// the pool ever returned is retired, so a use afterwards is reported rather
// than silently addressing whatever now occupies that offset.
func (b *Bump) Reset() error {
	for _, a := range b.live {
		a.owner = nil
	}
	b.live = b.live[:0]
	b.cursor = 0
	b.used = 0
	b.allocations = 0
	return nil
}

// Stats reports occupancy. A linear pool has exactly one free block, the tail,
// so LargestFree is what remains ahead of the cursor and never diverges from
// Free: a bump cannot fragment.
func (b *Bump) Stats() Stats {
	remaining := b.size - b.cursor
	blocks := 1
	if remaining == 0 {
		blocks = 0
	}
	return Stats{
		Size:        b.size,
		Used:        b.used,
		Free:        b.size - b.used,
		LargestFree: remaining,
		Allocations: b.allocations,
		Blocks:      blocks,
	}
}
