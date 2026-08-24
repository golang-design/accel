// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import "golang.design/x/accel/internal/testkernels"

// Contiguous packs a strided view into fresh contiguous storage.
//
// # Why a caller reaches for this
//
// A view is free. [Permute], [Transpose], [Slice] and [Broadcast] change
// strides and copy nothing, which is what makes a head split cost nothing. But
// a kernel that indexes contiguously cannot read one, so a transposed operand
// reaches a matmul as a refusal — and until this existed there was no way to
// convert it. specs/025-tensor-operators.md named this operator in four error
// messages, one of which told a caller to insert it.
//
// # Why it is explicit and not automatic
//
// Inserting the copy silently is the choice specs/007-tensor-layer.md declines:
// *"a copy nobody asked for is a cost nobody can see."* A transformer's
// reshaping is free precisely because nothing materializes behind the caller's
// back, and an operator that quietly packed would make the free case and the
// expensive case look identical in the source.
//
// So a strided operand is still refused, and the refusal names this. Calling it
// is how a caller says the copy is worth it.
//
// # What it costs
//
// One element read and one written, per element, plus the index arithmetic:
// each destination element decomposes its linear index into coordinates and
// dots them with the source strides. An operand that is already contiguous is
// returned unchanged rather than copied, so calling this defensively is free.
func Contiguous(b *Builder, x *Tensor) *Tensor {
	if poisoned(x) {
		return b.poison()
	}
	// Already packed: nothing to do, and returning the operand rather than a
	// copy is what makes a defensive call cost nothing. A caller wrapping every
	// matmul operand should not pay for the ones that did not need it.
	if x.contiguousLayout() {
		return x
	}
	if !elementwiseDType(x.dtype) {
		return b.fail(1, "Contiguous", "%v is not a dtype the packing kernel moves",
			x.dtype)
	}
	rank := len(x.shape)
	if rank > testkernels.PackRank {
		return b.fail(1, "Contiguous", "%v has rank %d and the packing kernel carries %d "+
			"extents and strides in one std140 block, so a higher rank would be a "+
			"different block per call site", x.shape, rank, testkernels.PackRank)
	}
	if rank == 0 {
		return b.fail(1, "Contiguous", "a rank-zero operand has no layout to pack")
	}

	var p testkernels.PackParams
	p.Rank = uint32(rank)
	p.Count = uint32(x.shape.Elements())
	p.Offset = uint32(x.offset)
	for i := range rank {
		if x.shape[i] <= 0 {
			return b.fail(1, "Contiguous", "axis %d of %v is %d, and the packing kernel "+
				"needs every extent to compute an index", i, x.shape, x.shape[i])
		}
		if x.strides[i] < 0 {
			return b.fail(1, "Contiguous", "axis %d of %v has stride %d; a negative "+
				"stride would read backwards from the offset and this kernel indexes "+
				"forward", i, x.shape, x.strides[i])
		}
		p.Extent[i] = uint32(x.shape[i])
		p.Stride[i] = uint32(x.strides[i])
	}

	return b.record(node{
		op: "Contiguous", inputs: []*Tensor{x},
		kernel:  &testkernels.PackKernel,
		strided: true,
		uniform: func(map[string]ScalarValue) any { return p },
		reason: "the strided gather of specs/042-surface-completion.md section 2.1; " +
			"the operand's strides are baked into the node because a view's layout is " +
			"structure rather than contents",
	}, x.dtype, x.shape)
}
