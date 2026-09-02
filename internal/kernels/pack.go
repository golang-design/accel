// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import "golang.design/x/accel"

// PackRank is the highest rank Pack handles.
//
// A fixed bound because the strides travel in a std140 uniform block, whose
// size is decided at generation: a rank-parameterised block would be a
// different block per call site. Eight is well past what a transformer reaches
// — a batched attention operand is rank four — and a shape beyond it is refused
// naming this number rather than silently packing the first eight axes.
const PackRank = 8

// PackParams describes a strided source and the contiguous destination.
//
// Extents and strides are separate arrays rather than one interleaved array
// because std140 gives an array member a sixteen-byte stride whatever its
// element type: two arrays of eight uint32 cost what one array of sixteen
// would, and each is indexed by axis rather than by axis-times-two.
type PackParams struct {
	// Rank is how many of the eight extents and strides are live. Axes beyond
	// it are ignored rather than required to be one, so a caller fills what
	// they have.
	Rank uint32

	// Count is how many elements the destination holds, which is the product of
	// the live extents. Passed rather than recomputed so the bounds check is
	// one comparison.
	Count uint32

	// Offset is the source's first element, in elements.
	Offset uint32

	Extent [PackRank]uint32
	Stride [PackRank]uint32
}

// Pack copies a strided view into contiguous storage.
//
// # Why this exists
//
// A view is free: a transpose or a slice changes strides and copies nothing,
// which is what makes a head split cost nothing. But a kernel that indexes
// contiguously cannot read one, so an operand that has been transposed reaches
// a matmul as a refusal. Until this kernel existed there was no way to convert
// it — specs/025-tensor-operators.md named the operator in four error messages,
// one of which told a caller to insert it, and it did not exist.
//
// # How the index is computed
//
// Each invocation owns one destination element. Its linear index decomposes
// into coordinates from the last axis backwards, and each coordinate multiplies
// that axis's source stride. The loop runs to PackRank rather than to Rank so
// the trip count is a constant the lowering can unroll; axes past Rank
// contribute nothing because their extent is treated as one.
//
//accel:kernel workgroup=64
func Pack(t accel.Thread, p PackParams, src []float32, dst []float32) {
	i := t.GlobalID().X
	if i < p.Count {
		rem := i
		at := p.Offset
		for axis := int32(PackRank - 1); axis >= 0; axis-- {
			a := uint32(axis)
			if a < p.Rank {
				e := p.Extent[a]
				if e > 0 {
					c := rem % e
					rem = rem / e
					at = at + c*p.Stride[a]
				}
			}
		}
		dst[i] = src[at]
	}
}
