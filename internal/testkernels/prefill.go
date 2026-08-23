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

	// KVLen is how many cached positions there are. For the prefill that
	// produces a cache, it equals QSeq.
	KVLen uint32

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
// The grid is therefore QSeq*QHeads workgroups, and the cache length is bounded
// by the shared array exactly as it is for decode. A longer one needs the
// chunked variant with an online softmax, which is post-v0.
//
//accel:kernel workgroup=128
func AttentionPrefill(t accel.Thread, d PrefillDims, q []float32, k []float32, v []float32,
	out []float32, scores *[128]float32, red *[128]float32) {

	group := t.GroupID().X
	lane := t.LocalID().X

	// The workgroup's output row: which query position, and which head.
	s := group / d.QHeads
	h := group % d.QHeads

	kvHead := h / (d.QHeads / d.KVHeads)

	// The last cached position this query may see. Causal masking is this
	// bound and nothing else: a lane past it contributes nothing.
	limit := d.Base + s

	score := float32(0)
	visible := lane <= limit && lane < d.KVLen
	if visible {
		acc := float32(0)
		for i := uint32(0); i < d.HeadDim; i++ {
			qi := q[(s*d.QHeads+h)*d.HeadDim+i]
			ki := k[lane*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i]
			acc = acc + qi*ki
		}
		score = acc * d.Scale
	}
	scores[lane] = score

	// The maximum, over the visible lanes only. A masked lane contributes a
	// value that cannot win: for a *maximum* the identity is negative infinity,
	// and the smallest finite f32 serves, because a real score below it would
	// already have overflowed.
	m := score
	if !visible {
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

	// The shifted exponentials. A masked lane contributes zero, which for a sum
	// is the identity and is therefore correct where it was not for the maximum.
	e := float32(0)
	if visible {
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

	// The weighted sum of V, parallel over the head's dimensions: every lane
	// can read every probability now, and each owns one output element.
	if lane < d.HeadDim {
		acc := float32(0)
		for j := uint32(0); j < d.KVLen; j++ {
			acc = acc + scores[j]*v[j*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
		}
		out[(s*d.QHeads+h)*d.HeadDim+lane] = acc / total
	}
}
