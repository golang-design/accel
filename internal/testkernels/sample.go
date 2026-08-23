// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import "golang.design/x/accel"

// SampleDims is a sampler's shape.
type SampleDims struct {
	// Vocab is how many logits there are.
	Vocab uint32

	// Draw is a uniform value in [0, 1), supplied by the caller rather than
	// generated here. specs/028-sampling.md gives the reason: a token is
	// reproducible only if the randomness is an input, and the two backends can
	// agree on a token only if neither is running a PRNG.
	Draw float32
}

// SampleArgmax writes the index of the largest logit.
//
//	out[0] = argmaxᵢ logits[i]
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

	// Each lane's own best over its strided slice, scanning upward so the
	// first of an equal pair is the one kept.
	v := float32(-3.4e38)
	idx := uint32(0)
	for i := lane; i < d.Vocab; i += RowWidth {
		x := logits[i]
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
		out[0] = at[0]
	}
}

// SampleCategorical draws an index from a distribution.
//
//	out[0] = min{ i : Σⱼ≤ᵢ probs[j] > draw }
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
// # Two things it must not assume
//
// The draw may be outside [0, 1): it is clamped rather than rejected, because a
// kernel cannot report an error and an unclamped draw reads past the end.
//
// The probabilities may sum to slightly below one, because Softmax divides by a
// sum computed in f32. The walk therefore returns the last index if it reaches
// the end, rather than falling off it.
//
//accel:kernel workgroup=1
func SampleCategorical(t accel.Thread, d SampleDims, probs []float32, out []uint32) {
	if t.GlobalID().X != 0 {
		return
	}

	draw := d.Draw
	if draw < float32(0) {
		draw = float32(0)
	}
	// Just below one. A draw of exactly one exceeds every partial sum, and the
	// index that would come back is whatever the loop left behind.
	if draw > float32(0.99999994) {
		draw = float32(0.99999994)
	}

	acc := float32(0)
	chosen := d.Vocab - 1
	found := false
	for i := uint32(0); i < d.Vocab; i++ {
		acc = acc + probs[i]
		if !found && acc > draw {
			chosen = i
			found = true
		}
	}
	out[0] = chosen
}
