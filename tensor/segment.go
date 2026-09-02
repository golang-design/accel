// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
)

// The segmented extent, in its own file because it belongs to no operator.
//
// specs/046-segmented-extents.md §1 argues the primitive before any caller, and
// this file is that argument in the code: a count per row is not attention's,
// and it lived in attention.go only because attention happened to need it
// first. It has two callers already -- a ragged attention step and a gated
// delta scan -- and issue 18's grouped GEMM is the third, whose counts are
// tokens routed per expert rather than tokens contributed per sequence.

// segmentOffsets records the exclusive prefix sum of a count-per-row tensor.
//
// specs/046-segmented-extents.md §1.1: the offsets are a function of the
// counts, so a caller who supplied both could supply two that disagree. They
// are derived by the operator that needs them and are not part of the public
// surface. A later caller that genuinely needs offsets it cannot derive is what
// would change that, and there is none.
//
// The result is rows+1 long, so a kernel that owns row r has both ends of it
// and the row's own count is off[r+1]-off[r]. The last entry is the total.
func (b *Builder) segmentOffsets(counts *Tensor, rows int) *Tensor {
	if poisoned(counts) {
		return b.poison()
	}
	return b.record(node{
		op: "SegmentOffsets", inputs: []*Tensor{counts},
		kernel: &kernels.SegmentOffsetsKernel,
		uniform: func(map[string]ScalarValue) any {
			return kernels.SegmentDims{Rows: uint32(rows)}
		},
		grid: func(*Tensor) accel.WorkgroupCount {
			// One workgroup of one invocation: the scan is serial and rows is a
			// batch size. See the kernel for why a parallel scan would cost
			// more than it saves at this size.
			return accel.WorkgroupCount{X: 1}
		},
		reason: "the serial prefix sum: rows is a batch size, so the scan is a handful " +
			"of adds before a dispatch that reads whole caches",
	}, accel.U32, Shape{rows + 1})
}
