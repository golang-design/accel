// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import "golang.design/x/accel"

// SegmentDims is a segmented extent's row count.
type SegmentDims struct {
	// Rows is how many segments there are, so the offsets are Rows+1 long.
	Rows uint32
}

// SegmentOffsets turns a count per row into the offsets those counts imply.
//
// specs/046-segmented-extents.md §1. A segmented value is a flat buffer, a
// count per row, and the exclusive prefix sum of those counts:
//
//	n   = [ 3 , 1 , 4 ]
//	off = [ 0 , 3 , 4 , 8 ]
//
// # Why Rows+1 entries
//
// A kernel that owns row r needs both ends of it. With the extra entry the
// end is off[r+1], and the row's own count is off[r+1]-off[r] -- so nothing
// downstream binds the counts as well, and there is no second buffer that
// could disagree with the first. The last entry is also the total, which is
// what lets the host refuse a flat buffer whose length does not match.
//
// # Why one invocation
//
// Rows is a batch size: tens of entries. A parallel scan of tens of integers
// costs more in barriers than the serial loop costs in adds, and this runs once
// before a dispatch that reads whole caches. The shape that would need a real
// scan is a segmented axis over thousands of rows, which nothing asks for --
// specs/046-segmented-extents.md §6 is where that would be argued.
//
//accel:kernel workgroup=1
func SegmentOffsets(t accel.Thread, d SegmentDims, counts []uint32, offsets []uint32) {
	sum := uint32(0)
	for r := uint32(0); r < d.Rows; r++ {
		offsets[r] = sum
		// A count of zero is legal and lands here as a repeated offset, which
		// is what makes a row that contributes nothing an ordinary member of a
		// batch rather than a case anything below has to test for.
		sum = sum + counts[r]
	}
	offsets[d.Rows] = sum
}
