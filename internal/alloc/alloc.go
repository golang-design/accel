// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package alloc carves a pool into allocations.
//
// It works in offsets and sizes and knows nothing about devices, dtypes, or
// buffers: a pool is one device allocation and this decides where inside it
// each resource goes. Keeping it separate is what makes the fragmentation
// behaviour spec 001 section 5.3 promises testable directly rather than only
// through a device.
//
// Two allocators exist because two workloads do. See specs/001-device-resources.md
// section 5.2 for the comparison against buddy, first-fit, and fixed size
// classes that this choice came out of.
package alloc

import (
	"errors"
	"fmt"
)

// ErrNoSpace reports that an allocator cannot serve a request.
//
// It does not distinguish exhaustion from fragmentation, because the numbers
// that do are in [Stats] and the caller already has them: Free at or above the
// request while LargestFree is below it is fragmentation, and those are
// different problems with different fixes.
var ErrNoSpace = errors.New("accel: allocator has no space for the request")

// ErrDoubleFree reports an allocation released twice, or released to an
// allocator it did not come from.
var ErrDoubleFree = errors.New("accel: allocation is already free")

// ErrUnsupported reports an operation the allocator's policy does not offer,
// such as freeing one allocation from a linear pool.
var ErrUnsupported = errors.New("accel: the allocator does not support that operation")

// Allocation is one placed range. Offset and Size are both multiples of the
// allocator's granularity, so Size is the *allocation size* of spec 001 section
// 3.4 and is at or above the caller's buffer size.
type Allocation struct {
	Offset int
	Size   int

	blk   *block     // nil for a linear allocator, which has nothing to give back
	owner *allocator // set on placement, cleared on free
}

// Stats reports occupancy. It is the shape accel.PoolStats is built from.
type Stats struct {
	Size        int
	Used        int
	Free        int
	LargestFree int
	Allocations int
	Blocks      int
}

// Allocator places allocations inside a fixed extent.
//
// Nothing here grows. A pool is exactly one device allocation, no backend can
// resize one in place, and growing by reallocating and copying would invalidate
// every address already handed out. See spec 001 section 5.5.
type Allocator interface {
	// Alloc places size bytes at an offset that is a multiple of align. align
	// must be a positive power of two; a value below the granularity is raised to
	// it, since every offset is a granularity multiple anyway.
	Alloc(size, align int) (*Allocation, error)

	// Free releases one allocation. A linear allocator reports ErrUnsupported:
	// individual frees are exactly what it trades away for its bump.
	Free(a *Allocation) error

	// Reset releases everything at once. A general allocator reports
	// ErrUnsupported: it has nothing to offer that Free does not already cover
	// one allocation at a time.
	Reset() error

	// Stats reports occupancy.
	Stats() Stats

	// Granularity reports the multiple every offset and size is rounded to. It is
	// the pool's alignment floor, so an allocation's size includes the alignment
	// padding spec 001 section 3.4 makes visible in PoolStats.Used.
	Granularity() int
}

// allocator is the state both policies share, so that Allocation can name its
// owner without the two implementations knowing about each other.
type allocator struct {
	size        int
	granularity int
	used        int
	allocations int
}

func (a *allocator) Granularity() int { return a.granularity }

// checkParams validates a request the same way for both policies.
func (a *allocator) checkParams(size, align int) (int, int, error) {
	if size <= 0 {
		return 0, 0, fmt.Errorf("accel: allocation size %d is not positive", size)
	}
	if align <= 0 || align&(align-1) != 0 {
		return 0, 0, fmt.Errorf("accel: alignment %d is not a positive power of two", align)
	}
	if align < a.granularity {
		align = a.granularity
	}
	return roundUp(size, a.granularity), align, nil
}

// checkOwner rejects an allocation that did not come from this allocator, or
// that has already been given back.
func (a *allocator) checkOwner(al *Allocation) error {
	if al == nil {
		return fmt.Errorf("%w: nil allocation", ErrDoubleFree)
	}
	if al.owner != a {
		return fmt.Errorf("%w: allocation at offset %d does not belong to this pool",
			ErrDoubleFree, al.Offset)
	}
	return nil
}

func roundUp(v, to int) int { return (v + to - 1) &^ (to - 1) }

// checkGranularity validates a pool's own construction parameters.
func checkGranularity(size, granularity int) error {
	if granularity <= 0 || granularity&(granularity-1) != 0 {
		return fmt.Errorf("accel: pool granularity %d is not a positive power of two", granularity)
	}
	if size < granularity {
		return fmt.Errorf("accel: pool size %d is below the %d granularity", size, granularity)
	}
	return nil
}
