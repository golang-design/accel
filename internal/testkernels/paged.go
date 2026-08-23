// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import (
	"golang.design/x/accel"
	"golang.design/x/accel/kmath"
)

// PagedDims is a paged decode step's shape.
//
// The same as [AttnDims] plus the block size, because that is the only thing
// the addressing needs that the contiguous form does not.
type PagedDims struct {
	QHeads  uint32
	KVHeads uint32
	HeadDim uint32

	// KVLen is how many logical positions hold real tokens.
	KVLen uint32

	// Block is how many positions one physical block holds.
	Block uint32

	Scale float32
}

// AttentionDecodePaged is one decode step over a cache addressed through a page
// table.
//
//	physical(j) = pages[j / Block] * Block + (j mod Block)
//
// # What changed from the contiguous form, and what did not
//
// One indirection, and nothing else. The scoring, the shifted-exponential
// softmax, the weighted sum, and the treatment of lanes past the cache are all
// [AttentionDecode]'s, unchanged -- because a paged cache is a different
// *addressing* rather than a different attention, and specs/030-paged-kv.md
// makes that the property to preserve.
//
// The page table is a binding rather than a uniform. It varies per sequence and
// per step, and a uniform would make every sequence its own plan, which is what
// paging exists to avoid.
//
//accel:kernel workgroup=128
func AttentionDecodePaged(t accel.Thread, d PagedDims, q []float32, k []float32,
	v []float32, pages []uint32, out []float32, scores *[128]float32, red *[128]float32) {

	h := t.GroupID().X
	lane := t.LocalID().X

	group := d.QHeads / d.KVHeads
	kvHead := h / group

	// Each lane scores one cached position, reached through its page.
	s := float32(0)
	if lane < d.KVLen {
		phys := pages[lane/d.Block]*d.Block + lane%d.Block
		acc := float32(0)
		for i := uint32(0); i < d.HeadDim; i++ {
			qi := q[h*d.HeadDim+i]
			ki := k[phys*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i]
			acc = acc + qi*ki
		}
		s = acc * d.Scale
	}
	scores[lane] = s

	m := s
	if lane >= d.KVLen {
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
	if lane < d.KVLen {
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

	// The weighted sum of V, parallel over the head's dimensions. Each lane
	// owns one output element and walks every cached position, so the page
	// lookup happens per position here rather than per lane.
	if lane < d.HeadDim {
		acc := float32(0)
		for j := uint32(0); j < d.KVLen; j++ {
			phys := pages[j/d.Block]*d.Block + j%d.Block
			acc = acc + scores[j]*v[phys*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
		}
		out[h*d.HeadDim+lane] = acc / total
	}
}
