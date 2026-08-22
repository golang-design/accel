// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package alloc

import (
	"fmt"
	"math/bits"
)

// slBits is how many mantissa bits split each power-of-two size class, so each
// class is divided into 1<<slBits sub-classes. Four gives at most about 6
// percent overshoot on a request, which is the internal fragmentation bound
// spec 001 section 5.2 trades for O(1) placement.
const slBits = 4

const slCount = 1 << slBits

// TLSF is a two-level segregated fit allocator: a general pool.
//
// It keeps a two-dimensional array of free lists indexed by the exponent and
// the top slBits mantissa bits of a block's size, plus a bitmap over each
// level. Allocation finds the smallest size class *guaranteed* to satisfy the
// request, takes its head, and splits the remainder. Free coalesces with
// physically adjacent free neighbours through the block list and pushes the
// result onto its class. Both are a handful of bit operations and pointer
// writes, which is the property that rules out the O(n) first-fit list: loading
// thousands of weight tensors must not be quadratic in the tensor count.
//
// It does not compact and cannot. A device address is baked into descriptor
// sets, descriptor heap entries, argument buffers, and recorded commands, so
// moving an allocation would mean rewriting every binding naming it. External
// fragmentation inside a pool is therefore permanent for that pool's life, and
// the mitigation is separating pools by lifetime class rather than a cleverer
// allocator. See spec 001 section 5.3.
//
// A TLSF is not safe for concurrent use; the pool that owns one serialises it.
type TLSF struct {
	allocator

	flCount int

	flBitmap uint64
	slBitmap []uint32
	free     [][]*block

	head *block // first block in physical order

	blocks int // free block count, for Stats
}

// block is one range in physical order, free or live. Physical neighbours are
// linked so that Free can coalesce without searching.
type block struct {
	offset int
	size   int
	free   bool

	prevPhys, nextPhys *block
	prevFree, nextFree *block
}

// NewTLSF returns a general allocator over [0, size) placing at multiples of
// granularity.
func NewTLSF(size, granularity int) (*TLSF, error) {
	if err := checkGranularity(size, granularity); err != nil {
		return nil, err
	}

	// Classes are indexed by size in granularity units rather than in bytes, so
	// the smallest class is always exponent zero whatever the granularity is.
	// The one above the whole pool is reachable by the round-up in mappingUp and
	// is allocated so that a search for it lands on an empty class rather than
	// out of range.
	count := bits.Len(uint(size/granularity)) + 1

	t := &TLSF{
		allocator: allocator{size: size, granularity: granularity},
		flCount:   count,
		slBitmap:  make([]uint32, count),
		free:      make([][]*block, count),
	}
	for i := range t.free {
		t.free[i] = make([]*block, slCount)
	}

	// The pool starts as one free block covering everything the granularity can
	// address. A tail smaller than the granularity is unusable by construction,
	// so it is not tracked and does not appear as free space that cannot be
	// handed out.
	usable := size &^ (granularity - 1)
	t.head = &block{offset: 0, size: usable, free: true}
	t.insert(t.head)
	return t, nil
}

// mapping places a size in the (fl, sl) grid. size must be a granularity
// multiple, which every block's is.
//
// The exponent is taken over granularity units, not bytes. An exponent below
// slBits then spans fewer distinct sizes than there are sub-classes, so the
// mantissa is shifted up instead of down: each such size gets its own
// sub-class, which makes a hit there exact rather than merely sufficient.
func (t *TLSF) mapping(size int) (fl, sl int) {
	u := size / t.granularity
	fl = bits.Len(uint(u)) - 1
	if fl < slBits {
		sl = (u << uint(slBits-fl)) & (slCount - 1)
	} else {
		sl = (u >> uint(fl-slBits)) & (slCount - 1)
	}
	return fl, sl
}

// mappingUp places a size in the smallest class every member of which is at
// least that size. Rounding first is what makes a hit on that class a guarantee
// rather than a search: without it the class could contain blocks smaller than
// the request.
//
// Below slBits the class is already exact, so there is nothing to round.
func (t *TLSF) mappingUp(size int) (fl, sl int) {
	u := size / t.granularity
	if e := bits.Len(uint(u)) - 1; e > slBits {
		u += (1 << uint(e-slBits)) - 1
	}
	return t.mapping(u * t.granularity)
}

func (t *TLSF) insert(b *block) {
	fl, sl := t.mapping(b.size)
	b.prevFree = nil
	b.nextFree = t.free[fl][sl]
	if b.nextFree != nil {
		b.nextFree.prevFree = b
	}
	t.free[fl][sl] = b
	t.flBitmap |= 1 << uint(fl)
	t.slBitmap[fl] |= 1 << uint(sl)
	t.blocks++
}

func (t *TLSF) remove(b *block) {
	fl, sl := t.mapping(b.size)
	if b.prevFree != nil {
		b.prevFree.nextFree = b.nextFree
	} else {
		t.free[fl][sl] = b.nextFree
	}
	if b.nextFree != nil {
		b.nextFree.prevFree = b.prevFree
	}
	b.prevFree, b.nextFree = nil, nil
	if t.free[fl][sl] == nil {
		t.slBitmap[fl] &^= 1 << uint(sl)
		if t.slBitmap[fl] == 0 {
			t.flBitmap &^= 1 << uint(fl)
		}
	}
	t.blocks--
}

// search finds the head of the first non-empty class at or above (fl, sl).
func (t *TLSF) search(fl, sl int) *block {
	if fl >= t.flCount {
		return nil
	}
	// Sub-classes at or above sl within this exponent.
	if m := t.slBitmap[fl] &^ (1<<uint(sl) - 1); m != 0 {
		return t.free[fl][bits.TrailingZeros32(m)]
	}
	// Otherwise the first exponent above this one with anything in it.
	m := t.flBitmap &^ (1<<uint(fl+1) - 1)
	if m == 0 {
		return nil
	}
	fl = bits.TrailingZeros64(m)
	return t.free[fl][bits.TrailingZeros32(t.slBitmap[fl])]
}

// Alloc places size bytes at a multiple of align.
//
// An alignment stricter than the granularity is served by taking a block large
// enough to hold the padding and trimming the head back onto the free list, so
// no space is lost to the alignment beyond what the granularity already costs.
func (t *TLSF) Alloc(size, align int) (*Allocation, error) {
	size, align, err := t.checkParams(size, align)
	if err != nil {
		return nil, err
	}

	b := t.findFit(size, align)
	if b == nil {
		return nil, fmt.Errorf("%w: %d bytes at alignment %d", ErrNoSpace, size, align)
	}
	t.remove(b)

	// Trim the head to reach the alignment, if the block does not already start
	// at one. The trimmed part goes back as a free block rather than being lost.
	if pad := roundUp(b.offset, align) - b.offset; pad > 0 {
		head := &block{offset: b.offset, size: pad, free: true, prevPhys: b.prevPhys, nextPhys: b}
		if b.prevPhys != nil {
			b.prevPhys.nextPhys = head
		} else {
			t.head = head
		}
		b.prevPhys = head
		b.offset += pad
		b.size -= pad
		t.insert(head)
	}

	// Trim the tail if what is left over is itself addressable.
	if rest := b.size - size; rest >= t.granularity {
		tail := &block{offset: b.offset + size, size: rest, free: true, prevPhys: b, nextPhys: b.nextPhys}
		if b.nextPhys != nil {
			b.nextPhys.prevPhys = tail
		}
		b.nextPhys = tail
		b.size = size
		t.insert(tail)
	}

	b.free = false
	t.used += b.size
	t.allocations++
	return &Allocation{Offset: b.offset, Size: b.size, blk: b, owner: &t.allocator}, nil
}

// findFit returns a free block that can hold size at align, or nil.
func (t *TLSF) findFit(size, align int) *block {
	// Without a stricter alignment the class search is exact: a hit is a block
	// guaranteed to be big enough, which is the whole point of rounding up.
	if align <= t.granularity {
		fl, sl := t.mappingUp(size)
		return t.search(fl, sl)
	}

	// With one, a block also has to absorb up to align-granularity of head
	// padding. Asking for that much more keeps the search O(1) and can only
	// over-ask, never under-ask.
	fl, sl := t.mappingUp(size + align - t.granularity)
	if b := t.search(fl, sl); b != nil {
		return b
	}

	// The over-ask can miss a block that happens to be aligned already. That is
	// the one case worth a bounded second look: scan the exact class, which is
	// short because a class holds blocks of nearly one size.
	fl, sl = t.mappingUp(size)
	for b := t.search(fl, sl); b != nil; b = b.nextFree {
		if roundUp(b.offset, align)+size <= b.offset+b.size {
			return b
		}
	}
	return nil
}

// Free releases an allocation and coalesces it with free physical neighbours.
func (t *TLSF) Free(a *Allocation) error {
	if err := t.checkOwner(a); err != nil {
		return err
	}
	b := a.blk
	if b.free {
		return fmt.Errorf("%w: offset %d", ErrDoubleFree, a.Offset)
	}

	t.used -= b.size
	t.allocations--
	b.free = true

	if n := b.nextPhys; n != nil && n.free {
		t.remove(n)
		b.size += n.size
		b.nextPhys = n.nextPhys
		if n.nextPhys != nil {
			n.nextPhys.prevPhys = b
		}
	}
	if p := b.prevPhys; p != nil && p.free {
		t.remove(p)
		p.size += b.size
		p.nextPhys = b.nextPhys
		if b.nextPhys != nil {
			b.nextPhys.prevPhys = p
		}
		b = p
	}
	t.insert(b)

	a.owner = nil
	a.blk = nil
	return nil
}

// Reset is not offered by a general allocator: it has nothing Free does not
// already cover one allocation at a time, and offering it would invite using a
// general pool where a linear one was meant.
func (t *TLSF) Reset() error {
	return fmt.Errorf("%w: a general pool frees one allocation at a time", ErrUnsupported)
}

// Stats reports occupancy, including the LargestFree that predicts the
// fragmentation failure of spec 001 section 5.3 rather than reporting it
// afterwards.
func (t *TLSF) Stats() Stats {
	// The largest free block lives in the highest non-empty class, and a class
	// holds blocks within about 6 percent of one size, so this scans one short
	// list rather than every block in the pool.
	largest := 0
	if t.flBitmap != 0 {
		fl := 63 - bits.LeadingZeros64(t.flBitmap)
		sl := 31 - bits.LeadingZeros32(t.slBitmap[fl])
		for b := t.free[fl][sl]; b != nil; b = b.nextFree {
			if b.size > largest {
				largest = b.size
			}
		}
	}
	return Stats{
		Size:        t.size,
		Used:        t.used,
		Free:        t.size - t.used,
		LargestFree: largest,
		Allocations: t.allocations,
		Blocks:      t.blocks,
	}
}
