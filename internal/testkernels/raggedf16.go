// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import (
	"golang.design/x/accel"
	"golang.design/x/accel/kmath"
)

// AttentionRaggedF16 is [AttentionRagged] over an f16 cache.
//
// # Why this variant exists at all
//
// accel issue 23, filed by a consumer against the ragged step the day it
// landed, and the report is about a collision rather than a missing dtype. An
// f16 cache halves the largest allocation a serving process has after the
// weights and the only one that scales with both concurrency and context
// (issue 13). The ragged step is the only way to express a batched prefill or
// to share a dispatch between a prefill chunk and decodes (issue 16). Without
// this kernel a server has to give up one to have the other: batching costs the
// halving back, and keeping the halving forbids batching a prefill.
//
// Their arithmetic: per-sequence cache traffic is about L x 288 KiB at f32, and
// doubling it halves both the batch size worth reaching and the throughput
// ceiling -- a crossover near B=7 becoming near B=3.5 at L=2048.
//
// # What differs from the f32 form, and what does not
//
// Three lines: the two cache bindings and the two loads that widen. Everything
// else -- the segment lookup, the causal position, the block loop, the running
// softmax -- is [AttentionRagged]'s and is not restated, because a second copy
// of a recurrence is a second thing to keep correct.
//
// The arithmetic is f32 throughout: the halves widen on load, which is
// specs/002-compute-model.md's rule that narrow dtypes are storage. So this
// kernel's numeric bound is the f32 kernel's, and the two are compared against
// each other on values f16 holds exactly.
//
//accel:uniform offsets, lengths
//accel:kernel workgroup=128
func AttentionRaggedF16(t accel.Thread, d RaggedDims, q []float32, k []accel.Float16,
	v []accel.Float16, pages []uint32, lengths []uint32, offsets []uint32, out []float32,
	scores *[AttnBlock]float32, red *[AttnBlock]float32) {

	group := t.GroupID().X
	lane := t.LocalID().X

	tok := group / d.QHeads
	h := group % d.QHeads
	kvHead := h / (d.QHeads / d.KVHeads)

	qBase := (tok*d.QHeads + h) * d.HeadDim

	// Which sequence this token belongs to, as a *count* rather than a search:
	// the number of rows that end at or before it. For offsets [0,3,3,7] and
	// token 3 that is two -- rows 0 and 1 both end at or before 3 -- and row 2
	// is where token 3 lives.
	//
	// Written this way because the obvious form is ambiguous in a way that
	// hides. Testing `offsets[r] <= tok && tok < offsets[r+1]` and keeping the
	// match is correct, and testing it with `<=` on the upper bound is *also*
	// correct here only because the loop keeps the last match: a reader who
	// added a break would turn a harmless typo into a token attributed to the
	// row before its own. A count has no such reading -- every row either ends
	// before this token or does not.
	//
	// A row that contributed nothing has offsets[r] equal to offsets[r+1], so
	// it ends wherever it began and is counted for every token at or after it.
	// That is what makes an empty row cost nothing here rather than a case.
	seq := uint32(0)
	for r := uint32(0); r < d.Batch; r++ {
		if offsets[r+1] <= tok {
			seq = seq + 1
		}
	}

	// A token past the last segment is padding, and scores nothing.
	// [AttentionRagged]'s guard, for its reason: without it `seq` reaches Batch,
	// one past the end of `offsets`, `lengths`, and the page table's rows. The
	// cache being f16 changes nothing about the lookup, which is why this is the
	// same six lines.
	if seq >= d.Batch {
		if lane < d.HeadDim {
			out[qBase+lane] = 0
		}
		return
	}

	// The token's index within its own sequence, and from that its position in
	// the sequence -- which is not its index in the flat buffer.
	//
	// specs/046-segmented-extents.md §2.2: with L cached positions after this
	// step and n tokens contributed, this step's tokens occupy the last n of
	// them, so token i sits at L-n+i. Getting this wrong by one lets a token
	// read the token after it, which is the leak causal masking exists to
	// prevent and which reads as a fluent continuation rather than as a fault.
	i := tok - offsets[seq]
	n := offsets[seq+1] - offsets[seq]
	kvLen := lengths[seq]

	// A token with no position, because its sequence's length is smaller than
	// its count: [AttentionRagged]'s guard, for its reason. The unsigned form
	// of L-n+i wrapped and removed the mask for the whole sequence.
	if kvLen+i < n {
		if lane < d.HeadDim {
			out[qBase+lane] = 0
		}
		return
	}
	limit := kvLen + i - n

	pageBase := seq * d.MaxPages

	// The table's reach, not the pool's: a pool holds every sequence's blocks,
	// so looping over it would walk another sequence's cache. Same argument as
	// [AttentionDecodeBatched].
	capacity := d.MaxPages * d.Block
	bound := limit + 1
	if bound > capacity {
		bound = capacity
	}
	// specs/044-unbounded-context.md deviation 6's clamp, for its reason: the
	// loop bounds `base` and not `base+lane`, so a length past what the page
	// table can address would be scored by the lanes of the last block rather
	// than stopped by the loop -- and for a paged kernel that reads the next
	// sequence's page-table row.
	if kvLen > capacity {
		kvLen = capacity
	}

	m := float32(-3.4e38)
	l := float32(0)
	o := float32(0)

	for base := uint32(0); base < bound; base += AttnBlock {
		pos := base + lane

		score := float32(-3.4e38)
		visible := pos <= limit && pos < kvLen
		if visible {
			phys := pages[pageBase+pos/d.Block]*d.Block + pos%d.Block
			dot := float32(0)
			for j := uint32(0); j < d.HeadDim; j++ {
				dot = dot + q[qBase+j]*k[phys*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+j].F32()
			}
			score = dot * d.Scale
		}

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

		if lane < d.HeadDim {
			o = alpha * o
			for j := uint32(0); j < AttnBlock; j++ {
				if base+j <= limit && base+j < kvLen {
					phys := pages[pageBase+(base+j)/d.Block]*d.Block + (base+j)%d.Block
					o = o + scores[j]*v[phys*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane].F32()
				}
			}
		}
	}

	if lane < d.HeadDim {
		out[qBase+lane] = o / l
	}
}
