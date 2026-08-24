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
	v []float32, pages []uint32, lengths []uint32, out []float32,
	scores *[AttnBlock]float32, red *[AttnBlock]float32) {
	h := t.GroupID().X
	// One sequence, so its length is the first entry. See AttentionDecode.
	kvLen := lengths[0]
	lane := t.LocalID().X

	group := d.QHeads / d.KVHeads
	kvHead := h / group
	// How many positions the page table can address. From the table's extent
	// rather than from the pool's: the pool holds every sequence's blocks and
	// is sized for total concurrency, so looping over it would walk other
	// sequences' caches. specs/002-compute-model.md section 3.3 makes both
	// len() and a uniform field workgroup-uniform, which is what lets the loop
	// below hold a barrier -- see [AttentionDecode] for why kvLen cannot.
	capacity := uint32(len(pages)) * d.Block

	// The length is clamped to what the binding can reach. A caller's length is
	// device data and nothing above has checked it against the cache -- and the
	// loop bound below limits `base`, not `base+lane`, so an unclamped length
	// past the reach is scored by the lanes of the last block rather than
	// stopped by the loop. For the paged kernels that means reading the *next*
	// sequence's page-table row, which is the failure
	// specs/040-batch-scheduler.md names; for the contiguous ones it is a read
	// past the end of the cache.
	//
	// Clamping truncates: the answer attends over a prefix. That is wrong, and
	// it is the wrong that can be bounded here -- the kernel cannot tell a
	// length that is too large from one that is right.
	if kvLen > capacity {
		kvLen = capacity
	}

	// The running softmax, carried across blocks. See [AttentionDecode] for the
	// recurrence and for why one block reproduces the single-pass form exactly.
	m := float32(-3.4e38)
	l := float32(0)

	// The output accumulator is a local, not a shared array: each lane owns
	// exactly one element of the row and no other lane reads it, so there is
	// nothing to publish and no barrier to pay. The resumable lowering carries
	// a local across a suspension point, which is what makes this available
	// inside a loop that holds barriers.
	o := float32(0)

	for base := uint32(0); base < capacity; base += AttnBlock {
		pos := base + lane

		// Each lane scores one cached position, reached through its page.
		s := float32(-3.4e38)
		if pos < kvLen {
			phys := pages[pos/d.Block]*d.Block + pos%d.Block
			dot := float32(0)
			for i := uint32(0); i < d.HeadDim; i++ {
				dot = dot + q[h*d.HeadDim+i]*k[phys*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i]
			}
			s = dot * d.Scale
		}

		// The shared arrays are loop-carried, so this pass's writes have to be
		// ordered against the previous pass's reads of them. See
		// [AttentionDecode].
		t.Barrier()
		scores[lane] = s
		red[lane] = s
		t.Barrier()

		for stride := uint32(AttnBlock / 2); stride > 0; stride /= 2 {
			if lane < stride {
				red[lane] = kmath.Max(red[lane], red[lane+stride])
			}
			t.Barrier()
		}
		blockMax := red[0]

		next := kmath.Max(m, blockMax)
		alpha := kmath.Exp(m - next)

		e := float32(0)
		if pos < kvLen {
			e = kmath.Exp(s - next)
		}
		t.Barrier()
		scores[lane] = e
		red[lane] = e
		t.Barrier()

		for stride := uint32(AttnBlock / 2); stride > 0; stride /= 2 {
			if lane < stride {
				red[lane] = red[lane] + red[lane+stride]
			}
			t.Barrier()
		}
		l = alpha*l + red[0]
		m = next

		// The weighted sum of V, parallel over the head's dimensions. Each lane
		// owns one output element and walks the block's positions, so the page
		// lookup happens per position here rather than per lane.
		//
		// The bound on j is a range check and not a mask. A position past the
		// cache already has a probability of zero, so dropping it changes no
		// arithmetic -- it reads V past its end, which the CPU backend reports
		// as an out-of-range index and a device would not report at all.
		if lane < d.HeadDim {
			o = alpha * o
			for j := uint32(0); j < AttnBlock; j++ {
				if base+j < kvLen {
					phys := pages[(base+j)/d.Block]*d.Block + (base+j)%d.Block
					o = o + scores[j]*v[phys*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
				}
			}
		}
	}

	if lane < d.HeadDim {
		out[h*d.HeadDim+lane] = o / l
	}
}
