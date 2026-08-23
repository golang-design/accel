// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import (
	"golang.design/x/accel"
	"golang.design/x/accel/kmath"
)

// AttnDims is a decode-step attention's shape.
type AttnDims struct {
	// QHeads and KVHeads are the query and key/value head counts. QHeads must
	// be a multiple of KVHeads: several query heads share one KV head, which is
	// what grouped-query attention is and what makes a decode cache small
	// enough to keep.
	QHeads  uint32
	KVHeads uint32

	// HeadDim is divisible by eight and at most 128, per spec 010.
	HeadDim uint32

	// KVLen is how many cached positions there are.
	KVLen uint32

	// Scale multiplies the scores, conventionally 1/sqrt(headDim). It is a
	// parameter rather than computed, because a model may ship a different one
	// and computing it here would silently override what the weights expect.
	Scale float32
}

// AttnMaxKV is the longest cache this fused kernel handles.
//
// The scores live in one shared array, so the cache length is bounded by it. A
// longer cache needs the chunked variant with an online softmax, which is
// post-v0: spec 010 makes the fused path an *optional selection* precisely so
// that a shape it cannot take falls back to the composed path rather than being
// computed wrongly.
const AttnMaxKV = 128

// AttentionDecode is one decode step of attention, fused.
//
//	out[h] = Σⱼ softmax(qₕ·kⱼ · scale)ⱼ · vⱼ
//
// # Why fused rather than composed
//
// The composed path is MatMul → Softmax → MatMul, and it writes the whole score
// matrix to memory between the first two. For a decode step that matrix is one
// row per head and the traffic dominates the arithmetic. Fusing keeps the scores
// in shared memory, which is the entire reason this kernel exists.
//
// It is not mandatory. Spec 010 makes the composed path the definition and this
// an optional selection, so a shape or device this cannot take falls back
// rather than being computed wrongly — and the test compares the two, which is
// what makes the fallback a fallback rather than a second implementation.
//
// One workgroup per query head.
//
//accel:kernel workgroup=128
func AttentionDecode(t accel.Thread, d AttnDims, q []float32, k []float32, v []float32,
	out []float32, scores *[128]float32, red *[128]float32) {

	h := t.GroupID().X
	lane := t.LocalID().X

	// Several query heads share one KV head. Integer division rather than a
	// lookup: the grouping is contiguous by construction, which is what lets a
	// cache hold one entry per KV head rather than per query head.
	group := d.QHeads / d.KVHeads
	kvHead := h / group

	// Each lane scores one cached position.
	s := float32(0)
	if lane < d.KVLen {
		acc := float32(0)
		for i := uint32(0); i < d.HeadDim; i++ {
			qi := q[h*d.HeadDim+i]
			ki := k[lane*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i]
			acc = acc + qi*ki
		}
		s = acc * d.Scale
	}
	scores[lane] = s

	// A lane past the cache contributes negative infinity to the maximum rather
	// than zero: a zero would win over a row of genuinely negative scores and
	// shift every exponent, which changes the answer rather than only the
	// intermediate.
	red[lane] = s
	if lane >= d.KVLen {
		red[lane] = -3.4e38
	}
	t.Barrier()

	for stride := uint32(64); stride > 0; stride /= 2 {
		if lane < stride {
			red[lane] = kmath.Max(red[lane], red[lane+stride])
		}
		t.Barrier()
	}
	m := red[0]
	t.Barrier()

	// The shifted exponentials. A lane past the cache contributes zero, which
	// for a *sum* is the identity and therefore correct where it was not for
	// the maximum.
	e := float32(0)
	if lane < d.KVLen {
		e = kmath.Exp(scores[lane] - m)
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

	// The weighted sum of V, parallel over the head's dimensions rather than
	// over the cache: the probabilities are in shared memory now, so every lane
	// can read all of them and each owns one output element.
	if lane < d.HeadDim {
		acc := float32(0)
		for j := uint32(0); j < d.KVLen; j++ {
			acc = acc + scores[j]*v[j*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
		}
		out[h*d.HeadDim+lane] = acc / total
	}
}
