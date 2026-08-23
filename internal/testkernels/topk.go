// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import "golang.design/x/accel"

// TopDims is a truncation's shape.
type TopDims struct {
	Vocab uint32

	// K is how many entries top-k keeps. Ignored by top-p.
	K uint32

	// P is the cumulative mass top-p keeps, in (0, 1]. Ignored by top-k.
	P float32
}

// TopMaxRounds bounds how many entries either truncation can keep.
//
// Both walk the distribution from the largest downward, one entry per round, so
// the bound is a real limit rather than a buffer size: a top-k above it keeps
// fewer than asked, and a top-p that has not reached its mass stops here.
//
// 128 because that is the workgroup, and because a nucleus wider than 128
// tokens is not a nucleus. Stated rather than left implicit, since a truncation
// that silently kept fewer entries than asked would change what a model
// samples without changing what it reports.
const TopMaxRounds = 128

// TopKMask keeps the k largest logits and drives the rest to zero weight.
//
// # Why repeated extraction rather than a sort
//
// A full sort of a vocabulary is thousands of comparisons to answer a question
// about its first few dozen entries. Extraction asks the question directly:
// each round finds the largest entry below the previous round's, which is one
// workgroup reduction, and k rounds give the k-th largest exactly.
//
// Exactly, which a threshold search would not. Bisecting the value range would
// be fewer rounds and would land between two logits that differ in their last
// bit, and the count on each side would then depend on where the bisection
// happened to stop.
//
// # Ties
//
// The comparison is lexicographic on (value, index) descending, so an entry
// ties with another only against itself. That makes "the k largest" a set of
// exactly k entries whatever the data, and makes it the same set on both
// backends -- which specs/028-sampling.md requires for the same reason it
// requires argmax's tie rule.
//
// The output is a weight rather than a logit: kept entries carry their input
// value and dropped ones carry zero, so [SampleCategorical] can walk the result
// without a renormalizing pass.
//
//accel:kernel workgroup=128
func TopKMask(t accel.Thread, d TopDims, weights []float32, out []float32,
	best *[128]float32, at *[128]uint32) {

	lane := t.LocalID().X

	// The running frontier: the (value, index) of the smallest entry kept so
	// far. The first round has no frontier, which is what the huge initial
	// value means -- everything is below it.
	frontV := float32(3.4e38)
	frontI := uint32(0)
	rounds := d.K
	if rounds > TopMaxRounds {
		rounds = TopMaxRounds
	}

	for r := uint32(0); r < rounds; r++ {
		// Each lane's largest entry strictly below the frontier.
		v := float32(-3.4e38)
		idx := d.Vocab
		for i := lane; i < d.Vocab; i += RowWidth {
			w := weights[i]
			below := w < frontV
			if w == frontV && i > frontI {
				below = true
			}
			if below {
				better := w > v
				if w == v && i < idx {
					better = true
				}
				if better {
					v = w
					idx = i
				}
			}
		}
		best[lane] = v
		at[lane] = idx
		t.Barrier()

		for stride := uint32(64); stride > 0; stride /= 2 {
			if lane < stride {
				a := best[lane]
				b := best[lane+stride]
				if b > a {
					best[lane] = b
					at[lane] = at[lane+stride]
				} else if b == a && at[lane+stride] < at[lane] {
					at[lane] = at[lane+stride]
				}
			}
			t.Barrier()
		}
		frontV = best[0]
		frontI = at[0]
		t.Barrier()
	}

	// The frontier now names the k-th largest. Everything at or above it, in
	// the same lexicographic order, is kept.
	for i := lane; i < d.Vocab; i += RowWidth {
		w := weights[i]
		keep := w > frontV
		if w == frontV && i <= frontI {
			keep = true
		}
		if keep {
			out[i] = w
		} else {
			out[i] = float32(0)
		}
	}
}

// TopPMask keeps the smallest set of largest entries whose mass reaches p.
//
// The nucleus, and the same walk as [TopKMask] with a different stopping rule:
// rather than counting entries, it accumulates their weight and stops once the
// running total reaches p of the whole. The entry that crosses the threshold is
// kept, which is what makes the set the *smallest* one reaching p rather than
// the largest one below it.
//
// The mass is a fraction of the input's own total rather than of one, so this
// composes after a top-k mask or on unnormalized weights, for the same reason
// [SampleCategorical] scales its draw by the total.
//
//accel:kernel workgroup=128
func TopPMask(t accel.Thread, d TopDims, weights []float32, out []float32,
	best *[128]float32, at *[128]uint32) {

	lane := t.LocalID().X

	// The total, so the threshold is a fraction of what is actually there.
	sum := float32(0)
	for i := lane; i < d.Vocab; i += RowWidth {
		sum = sum + weights[i]
	}
	best[lane] = sum
	t.Barrier()
	for stride := uint32(64); stride > 0; stride /= 2 {
		if lane < stride {
			best[lane] = best[lane] + best[lane+stride]
		}
		t.Barrier()
	}
	target := best[0] * d.P
	t.Barrier()

	frontV := float32(3.4e38)
	frontI := uint32(0)
	kept := float32(0)

	// Every round runs, and only the frontier's *advance* is conditional.
	//
	// The obvious shape -- stop the loop once enough mass is kept -- puts a
	// barrier inside a branch, which specs/018-cooperative-lowering.md refuses
	// and specs/002-compute-model.md section 3.1 forbids anyway: a barrier must
	// sit in workgroup-uniform control flow. The compiler said so by name,
	// which is the diagnostic doing its job.
	//
	// So the reductions and their barriers stay at the top level and the test
	// that stops the walk guards only the assignment. Once the mass is reached
	// the extraction keeps finding the same entry and nothing takes it, which
	// costs rounds and buys a lowering that exists.
	for r := uint32(0); r < TopMaxRounds; r++ {
		v := float32(-3.4e38)
		idx := d.Vocab
		for i := lane; i < d.Vocab; i += RowWidth {
			w := weights[i]
			below := w < frontV
			if w == frontV && i > frontI {
				below = true
			}
			if below {
				better := w > v
				if w == v && i < idx {
					better = true
				}
				if better {
					v = w
					idx = i
				}
			}
		}
		best[lane] = v
		at[lane] = idx
		t.Barrier()

		for stride := uint32(64); stride > 0; stride /= 2 {
			if lane < stride {
				a := best[lane]
				b := best[lane+stride]
				if b > a {
					best[lane] = b
					at[lane] = at[lane+stride]
				} else if b == a && at[lane+stride] < at[lane] {
					at[lane] = at[lane+stride]
				}
			}
			t.Barrier()
		}

		// kept, target, best[0] and at[0] are all workgroup-uniform, so this
		// branch is too and every invocation takes the same side of it.
		if kept < target {
			frontV = best[0]
			frontI = at[0]
			kept = kept + frontV
		}
		t.Barrier()
	}

	for i := lane; i < d.Vocab; i += RowWidth {
		w := weights[i]
		keep := w > frontV
		if w == frontV && i <= frontI {
			keep = true
		}
		if keep {
			out[i] = w
		} else {
			out[i] = float32(0)
		}
	}
}
