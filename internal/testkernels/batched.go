// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import (
	"golang.design/x/accel"
	"golang.design/x/accel/kmath"
)

// BatchedDims is a batched paged decode's shape.
type BatchedDims struct {
	// Batch is how many sequences step together.
	Batch uint32

	QHeads  uint32
	KVHeads uint32
	HeadDim uint32

	// Block is how many positions one physical block holds, and MaxPages how
	// many page-table entries each sequence has room for. The page table is a
	// Batch x MaxPages array, because a ragged one has no dtype.
	Block    uint32
	MaxPages uint32

	Scale float32
}

// AttentionDecodeBatched steps several sequences in one dispatch.
//
// # Why this is a kernel rather than a loop over the paged one
//
// A decode step is memory-bound: it reads a whole cache to produce one token.
// Running four sequences as four submissions reads four caches in four
// dispatches, each with its own launch and its own tail where most of the
// device is idle. Running them together fills the device with work that was
// going to happen anyway.
//
// # Sequences have different lengths, and that is the whole difficulty
//
// Each workgroup handles one (sequence, head) pair and reads that sequence's
// own length and its own page table. Nothing is padded to a common length: a
// short sequence's lanes past its end contribute the identity to each reduction,
// exactly as they do in the unbatched kernel, so a batch of one long and three
// short sequences costs what the long one costs rather than four times it.
//
// specs/030-paged-kv.md's pool is what makes the page tables independent.
//
//accel:kernel workgroup=128
func AttentionDecodeBatched(t accel.Thread, d BatchedDims, q []float32, k []float32,
	v []float32, pages []uint32, lengths []uint32, out []float32,
	scores *[128]float32, red *[128]float32) {

	group := t.GroupID().X
	lane := t.LocalID().X

	// One workgroup per (sequence, head). Sequence-major, so the heads of one
	// sequence are adjacent and read the same page table.
	seq := group / d.QHeads
	h := group % d.QHeads

	kvHead := h / (d.QHeads / d.KVHeads)
	kvLen := lengths[seq]
	pageBase := seq * d.MaxPages
	qBase := (seq*d.QHeads + h) * d.HeadDim

	s := float32(0)
	if lane < kvLen {
		phys := pages[pageBase+lane/d.Block]*d.Block + lane%d.Block
		acc := float32(0)
		for i := uint32(0); i < d.HeadDim; i++ {
			qi := q[qBase+i]
			ki := k[phys*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i]
			acc = acc + qi*ki
		}
		s = acc * d.Scale
	}
	scores[lane] = s

	m := s
	if lane >= kvLen {
		m = float32(-3.4e38)
	}
	red[lane] = m
	t.Barrier()

	for stride := uint32(64); stride > 0; stride /= 2 {
		if lane < stride {
			a := red[lane]
			b := red[lane+stride]
			if b > a {
				red[lane] = b
			}
		}
		t.Barrier()
	}
	best := red[0]
	t.Barrier()

	e := float32(0)
	if lane < kvLen {
		e = kmath.Exp(scores[lane] - best)
	}
	scores[lane] = e
	red[lane] = e
	t.Barrier()

	for stride := uint32(64); stride > 0; stride /= 2 {
		if lane < stride {
			red[lane] = red[lane] + red[lane+stride]
		}
		t.Barrier()
	}
	total := red[0]

	if lane < d.HeadDim {
		acc := float32(0)
		for j := uint32(0); j < kvLen; j++ {
			phys := pages[pageBase+j/d.Block]*d.Block + j%d.Block
			acc = acc + scores[j]*v[phys*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
		}
		out[qBase+lane] = acc / total
	}
}
