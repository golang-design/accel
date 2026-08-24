// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import "golang.design/x/accel"

// SampleDims is a sampler's shape.
type SampleDims struct {
	// Vocab is how many logits each row holds.
	Vocab uint32

	// Rows is how many independent sequences are sampled together. One is a
	// single sequence, which is the same path rather than a special case.
	Rows uint32
}

// The draw is a binding rather than a field here, and specs/043-per-row-values.md
// says why.
//
// specs/028-sampling.md's decision to make the randomness an *input* is the
// right one -- a token is reproducible only if the caller supplies it, and two
// backends agree on a token only if neither runs a PRNG. What was wrong was the
// *shape*: one draw per dispatch.
//
// A shared draw keeps reproducibility and destroys independence. Two sequences
// whose distributions are similar -- common for a well-trained model answering
// related prompts -- draw against the same u and emit the same token. Their
// contexts then converge, so the next distributions are more similar still, and
// two users get the same answer. Every test passed, because reproducibility is
// what they check and reproducibility is exactly what was preserved.

// SampleArgmax writes the index of the largest logit.
//
//	out[r] = argmaxᵢ logits[r][i]
//
// # Ties go to the lowest index
//
// Equal logits are ordinary: an untrained model produces them everywhere and a
// trained one produces them at saturation. Leaving the answer to whichever lane
// happened to compare which pair would make two backends reducing at different
// widths return different tokens, so the reduction keeps the lower index on
// every equal comparison and the result is the same anywhere.
//
// One workgroup: the whole vocabulary reduces together, because a maximum split
// across workgroups would need a second pass and a tie rule that survived it.
//
//accel:kernel workgroup=128
func SampleArgmax(t accel.Thread, d SampleDims, logits []float32, out []uint32,
	best *[128]float32, at *[128]uint32) {

	lane := t.LocalID().X

	// One workgroup per row, so the rows of a batch reduce independently and
	// never share a partial. The index kept is within the row, which is what a
	// caller wants back: a token id, not an offset into the batch.
	r := t.GroupID().X
	base := r * d.Vocab

	// Each lane's own best over its strided slice, scanning upward so the
	// first of an equal pair is the one kept.
	v := float32(-3.4e38)
	idx := uint32(0)
	for i := lane; i < d.Vocab; i += RowWidth {
		x := logits[base+i]
		if x > v {
			v = x
			idx = i
		}
	}
	best[lane] = v
	at[lane] = idx
	t.Barrier()

	for stride := uint32(64); stride > 0; stride /= 2 {
		if lane < stride {
			a := best[lane]
			b := best[lane+stride]
			// Strictly greater, so an equal pair keeps the left one -- which
			// holds the lower index, because the lane's own scan went upward
			// and this half owns the lower lanes.
			if b > a {
				best[lane] = b
				at[lane] = at[lane+stride]
			} else if b == a && at[lane+stride] < at[lane] {
				at[lane] = at[lane+stride]
			}
		}
		t.Barrier()
	}

	if lane == 0 {
		out[r] = at[0]
	}
}

// SampleCategorical draws an index from a distribution.
//
//	out[r] = min{ i : Σⱼ≤ᵢ probs[r][j] > draws[r] }
//
// # Why the walk is sequential
//
// A parallel prefix sum would be faster and would put the boundary in a
// different place when two probabilities are equal, because the partial sums it
// forms are not the ones an in-order walk forms. specs/028-sampling.md takes
// reproducibility over speed here, which is the same trade
// specs/008-numerics.md makes by refusing a tolerance.
//
// One invocation does the walk. A vocabulary is thousands of entries and this
// runs once per token, next to a model step that is millions of operations.
//
// # The weights need not be normalized
//
// The walk compares against draw times the total rather than against the draw,
// which makes it correct for any vector of non-negative weights rather than
// only for a distribution summing to one.
//
// That is not generality for its own sake. A top-k or top-p mask zeroes most of
// a distribution and leaves the rest summing to less than one, and the
// alternative was a renormalizing pass whose only purpose was to satisfy this
// kernel. It also subsumes the case this originally special-cased: Softmax
// divides by a sum computed in f32, so its output can land a few ulps below
// one, and scaling by the actual total handles that without a rule about it.
//
// # The draw may be outside [0, 1)
//
// Clamped rather than rejected, because a kernel cannot report an error and an
// unclamped draw reads past the end. A draw of one lands exactly on the total,
// which no partial sum exceeds, so it is clamped just below.
//
//accel:kernel workgroup=1
func SampleCategorical(t accel.Thread, d SampleDims, probs []float32, draws []float32,
	out []uint32) {

	r := t.GlobalID().X
	if r >= d.Rows {
		return
	}
	base := r * d.Vocab

	draw := draws[r]
	if draw < float32(0) {
		draw = float32(0)
	}
	if draw > float32(0.99999994) {
		draw = float32(0.99999994)
	}

	total := float32(0)
	for i := uint32(0); i < d.Vocab; i++ {
		total = total + probs[base+i]
	}
	target := draw * total

	acc := float32(0)
	chosen := d.Vocab - 1
	found := false
	for i := uint32(0); i < d.Vocab; i++ {
		acc = acc + probs[base+i]
		// Strictly greater, so a draw landing exactly on a cumulative boundary
		// moves on to the next index rather than stopping short of it. A
		// zero-weight entry can never be chosen, because its partial sum does
		// not increase.
		if !found && acc > target {
			chosen = i
			found = true
		}
	}
	out[r] = chosen
}
