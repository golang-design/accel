// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import (
	"golang.design/x/accel"
	"golang.design/x/accel/kmath"
)

// AttentionPrefillPagedF16 is [AttentionPrefillPaged] over an f16 cache.
//
// [#25](https://github.com/golang-design/accel/issues/25). The narrow cache
// existed for a contiguous prefill and a paged *decode*, and not for the
// combination -- which is the one a consumer actually reaches, because a shared
// block pool is addressed through a page table by construction and every
// conversation starts with a prefill.
//
// # Derived, not written
//
// [AttentionPrefillPaged]'s body with three edits and no others: k and v become
// accel.Float16 in the signature, and the two loads that read them gain .F32().
// The same three that make [AttentionPrefillF16] from [AttentionPrefill] and
// [AttentionDecodePagedF16] from [AttentionDecodePaged]. Everything numeric is
// the f32 kernel's -- the accumulator, the running maximum, the block loop --
// because specs/008-numerics.md section 4 makes a narrow cache storage that
// widens on load rather than arithmetic at a narrower width.
//
//accel:kernel workgroup=128
func AttentionPrefillPagedF16(t accel.Thread, d PagedPrefillDims, q []float32,
	k []accel.Float16, v []accel.Float16, pages []uint32, lengths []uint32,
	out []float32,
	scores *[AttnBlock]float32, red *[AttnBlock]float32) {
	group := t.GroupID().X
	// A prefill is one sequence, so its cache length is the first entry. A
	// binding rather than a uniform so that there is exactly one way to say
	// "how much of the cache is real" across every attention kernel --
	// specs/043-per-row-values.md. Batched prefill is
	// specs/040-batch-scheduler.md's, and when it arrives this indexes by
	// sequence like the decode kernels do.
	kvLen := lengths[0]
	lane := t.LocalID().X

	// The workgroup's output row: which query position, and which head.
	s := group / d.QHeads
	h := group % d.QHeads

	kvHead := h / (d.QHeads / d.KVHeads)

	// The last cached position this query may see. Causal masking is this
	// bound and nothing else: a lane past it contributes nothing.
	limit := d.Base + s

	// How far the loop walks. The causal limit, or what the page table can
	// reach if the limit goes past it -- both workgroup-uniform, which is what
	// lets the loop hold barriers.
	//
	// The table's reach and not the pool's, for [AttentionDecodePaged]'s
	// reason: a pool holds every sequence's blocks and is sized for total
	// concurrency, so its extent is not this sequence's.
	capacity := uint32(len(pages)) * d.Block
	bound := limit + 1
	if bound > capacity {
		bound = capacity
	}

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

	for base := uint32(0); base < bound; base += AttnBlock {
		pos := base + lane

		// A masked lane contributes a value that cannot win the maximum: the
		// identity is negative infinity and the smallest finite f32 serves,
		// because a real score below it would already have overflowed.
		score := float32(-3.4e38)
		visible := pos <= limit && pos < kvLen
		if visible {
			// The page lookup is per position, not per element of the head:
			// one indirection for the whole row rather than HeadDim of them.
			phys := pages[pos/d.Block]*d.Block + pos%d.Block
			dot := float32(0)
			for i := uint32(0); i < d.HeadDim; i++ {
				qi := q[(s*d.QHeads+h)*d.HeadDim+i]
				ki := k[phys*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i].F32()
				dot = dot + qi*ki
			}
			score = dot * d.Scale
		}

		// The shared arrays are loop-carried, so this pass's writes have to be
		// ordered against the previous pass's reads of them. See
		// [AttentionDecode].
		t.Barrier()
		scores[lane] = score
		red[lane] = score
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

		// A masked lane contributes zero, which for a sum is the identity and
		// is therefore correct where it was not for the maximum.
		e := float32(0)
		if visible {
			e = kmath.Exp(score - next)
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

		// The weighted sum of V, parallel over the head's dimensions: every
		// lane can read every probability now, and each owns one output
		// element.
		//
		// The bound on j is a range check and not a mask. A position past the
		// cache already has a probability of zero, so dropping it changes no
		// arithmetic -- it reads V past its end, which the CPU backend reports
		// as an out-of-range index and a device would not report at all.
		if lane < d.HeadDim {
			o = alpha * o
			for j := uint32(0); j < AttnBlock; j++ {
				if base+j <= limit && base+j < kvLen {
					phys := pages[(base+j)/d.Block]*d.Block + (base+j)%d.Block
					o = o + scores[j]*v[phys*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane].F32()
				}
			}
		}
	}

	if lane < d.HeadDim {
		out[(s*d.QHeads+h)*d.HeadDim+lane] = o / l
	}
}
