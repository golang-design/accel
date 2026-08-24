// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import (
	"golang.design/x/accel"
	"golang.design/x/accel/kmath"
)

// PrefillDims is a prefill attention's shape.
//
// The same fields as [AttnDims] plus the query length, which is what makes this
// a different kernel rather than the same one with a bigger grid: a decode step
// attends one token over the whole cache, and a prefill attends many tokens
// each over a prefix of it.
type PrefillDims struct {
	QHeads  uint32
	KVHeads uint32
	HeadDim uint32

	// QSeq is how many query positions there are.
	QSeq uint32

	// Base is the position of the first query token within the cache, so a
	// prefill that extends an existing cache masks correctly.
	Base uint32

	Scale float32
}

// AttentionPrefill scores a sequence of queries against a cache, causally.
//
//	out[s, h] = Σⱼ≤base+s softmax(q[s,h]·kⱼ · scale)ⱼ · vⱼ
//
// # Why causal masking is built in rather than an option
//
// A prefill that did not mask would let a token attend to its own future, which
// is not a slower answer but a different model. specs/007-tensor-layer.md makes
// causal a compile-time attribute for that reason, and this kernel is the
// causal one: a non-causal prefill is a different kernel and would be selected
// separately.
//
// # Why one workgroup per (query position, head)
//
// The same reason the decode kernel uses one per head: the softmax over a row
// of scores is a reduction, and a reduction whose participants share workgroup
// memory cannot be split across workgroups. Each workgroup owns one output row
// and reduces over the cache within it.
//
// The grid is therefore QSeq*QHeads workgroups, and the cache is walked a block
// at a time with a running softmax, exactly as it is for decode.
//
// # This kernel's loop bound is the causal limit, not the cache
//
// A query at position base+s may see positions 0..base+s and no others, so
// there is nothing past that limit for the loop to reach. The limit is
// workgroup-uniform -- Base is a field of the uniform struct and s comes from
// the group id -- so it can bound a loop holding barriers where kvLen, being a
// load, cannot. The first query of a prefill therefore scores one block and the
// last scores as many as the sequence needs, which is the triangle the causal
// mask describes rather than a square with half of it discarded.
//
//accel:kernel workgroup=128
func AttentionPrefill(t accel.Thread, d PrefillDims, q []float32, k []float32, v []float32,
	lengths []uint32, out []float32, scores *[AttnBlock]float32, red *[AttnBlock]float32) {
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

	// How far the loop walks. The causal limit, or the cache's extent if the
	// limit reaches past it -- both workgroup-uniform, which is what lets the
	// loop hold barriers. See the note above.
	capacity := uint32(len(k)) / (d.KVHeads * d.HeadDim)
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
			dot := float32(0)
			for i := uint32(0); i < d.HeadDim; i++ {
				qi := q[(s*d.QHeads+h)*d.HeadDim+i]
				ki := k[pos*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i]
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
					o = o + scores[j]*v[(base+j)*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
				}
			}
		}
	}

	if lane < d.HeadDim {
		out[(s*d.QHeads+h)*d.HeadDim+lane] = o / l
	}
}
