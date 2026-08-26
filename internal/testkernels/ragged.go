// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import (
	"golang.design/x/accel"
	"golang.design/x/accel/kmath"
)

// RaggedDims is a ragged step's shape.
//
// Batch is here and the per-sequence token counts are not: a count differs per
// row, so specs/043-per-row-values.md makes it a binding. What is uniform is
// how many rows there are.
type RaggedDims struct {
	// Batch is how many sequences contribute to this step. A sequence
	// contributing zero tokens is still one of them.
	Batch uint32

	QHeads  uint32
	KVHeads uint32
	HeadDim uint32

	Block    uint32
	MaxPages uint32

	Scale float32
}

// AttentionRagged steps several sequences that contribute different numbers of
// tokens.
//
// specs/046-segmented-extents.md §2. This is what a batched *prefill* is, and
// what lets a prefill chunk share a dispatch with decode steps: the query
// buffer is flat and one offsets binding says which rows belong to whom.
//
//	offsets = [0, 512, 513, 514, 515]     one 512-token chunk, three decodes
//	q       = [515, qHeads, headDim]
//
// # Why one workgroup per (token, head) and not per (sequence, head)
//
// [AttentionDecodeBatched] puts a workgroup on a sequence because a decode step
// gives each sequence exactly one query row. Here a sequence has as many rows as
// it contributed, and they are causal against different positions, so the unit
// of work is a token. That is the whole difference between this kernel and that
// one, and it is why this is a separate entry rather than a flag on it.
//
// # The segment lookup
//
// A workgroup knows its flat token index and has to find whose it is. The scan
// below is linear over Batch, which is tens of entries: a few comparisons
// before a loop that reads whole caches. Everything it produces is
// workgroup-uniform -- the token index comes from GroupID -- which is what lets
// the block loop below hold barriers at all
// (specs/002-compute-model.md §3.3).
//
//accel:kernel workgroup=128
func AttentionRagged(t accel.Thread, d RaggedDims, q []float32, k []float32,
	v []float32, pages []uint32, lengths []uint32, offsets []uint32, out []float32,
	scores *[AttnBlock]float32, red *[AttnBlock]float32) {

	group := t.GroupID().X
	lane := t.LocalID().X

	tok := group / d.QHeads
	h := group % d.QHeads
	kvHead := h / (d.QHeads / d.KVHeads)

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
	limit := kvLen - n + i

	pageBase := seq * d.MaxPages
	qBase := (tok*d.QHeads + h) * d.HeadDim

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
				dot = dot + q[qBase+j]*k[phys*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+j]
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
					o = o + scores[j]*v[phys*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
				}
			}
		}
	}

	if lane < d.HeadDim {
		out[qBase+lane] = o / l
	}
}
