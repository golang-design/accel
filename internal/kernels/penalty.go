// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import "golang.design/x/accel"

// PenaltyDims is one penalty pass's shape and coefficients.
//
// The counts live in a buffer rather than here because there is one per
// vocabulary entry; what is here is the handful of numbers a decode step
// rebinds every token, which specs/039-sampling-policy.md section 7 prices at
// nothing because Submit rewrites every uniform on every submission anyway.
type PenaltyDims struct {
	Vocab uint32

	// History is the ring's capacity, and Count is how much of it is filled:
	// min(tokens so far, History). Both are needed because the ring is bound at
	// a fixed length -- binding it at its current length instead would change
	// the input shape every token, which is a new plan every token.
	History uint32
	Count   uint32

	// Repetition divides a positive logit and multiplies a non-positive one.
	// 1 is off, and so is 0: neither is a penalty, and refusing one of them
	// while accepting the other would be a trap.
	Repetition float32

	// Presence is subtracted once from any token that occurs at all, Frequency
	// once per occurrence. 0 is off for both.
	Presence  float32
	Frequency float32
}

// PenaltyCount counts how often each token id appears in a history ring.
//
// This is pass one of two, and the split is not an optimization.
//
// # Why counting first is the only correct order
//
// The history holds duplicate ids by construction -- that is exactly what a
// frequency penalty counts. One invocation per history entry doing
// `logits[id] -= penalty` therefore has several invocations writing the same
// address. Unsynchronised, updates are lost. With [accel.AddF32],
// specs/008-numerics.md section 2 class E makes the result non-deterministic
// against itself, and specs/011-conformance-harness.md section 9 excludes it
// from bit comparison, so the differential would stop being able to see the
// bug. Even a serial loop is order-dependent in f32, because l-p-p and l-2p
// are different numbers.
//
// [accel.AddU32] is exact and order-independent, so this pass is deterministic
// whatever order the workgroups run in, and [PenaltyApply] then does one update
// per distinct token from a count that is already final.
//
// The wrong version of this is not loud. It gives a different token on rerun
// from the same prompt and seed, but only when a token repeats.
//
//accel:kernel workgroup=64
func PenaltyCount(t accel.Thread, d PenaltyDims, history []uint32, counts []uint32) {
	i := t.GlobalID().X

	// Count rather than len(history): the ring is bound at its full capacity
	// and is only partly filled until the sequence is longer than it. Reading
	// the whole buffer would count the zeros past the end as occurrences of
	// token 0, which is a real id.
	if i >= d.Count {
		return
	}
	id := history[i]

	// A token id outside the vocabulary cannot index counts. It is dropped
	// rather than clamped, because clamping would attribute someone else's
	// repeat to the last token in the vocabulary. The host refuses this before
	// submission; this is the kernel-side half, which is what runs under the
	// CPU backend's strict execution.
	if id < d.Vocab {
		accel.AddU32(counts, id, 1)
	}
}

// PenaltyApply subtracts the penalties from the logits, one invocation per
// vocabulary entry.
//
// # The sign asymmetry is stated, not discovered
//
// The repetition penalty is divisive, and dividing a *negative* logit by r > 1
// moves it toward zero, which is upward: it rewards the repeat it was meant to
// punish. Logits have no fixed zero, so this is not a rare case. The negative
// branch multiplies instead, which moves it down.
//
// The honest limit is that r stays scale-dependent even so: two models whose
// logits sit at different offsets need different r for the same effect. The
// subtractive pair does not have that problem, which is a reason to prefer it
// and not a reason to hide the divisive one.
//
//accel:kernel workgroup=64
func PenaltyApply(t accel.Thread, d PenaltyDims, logits []float32, counts []uint32,
	out []float32) {

	i := t.GlobalID().X
	if i >= d.Vocab {
		return
	}
	l := logits[i]
	c := counts[i]

	// A token that never occurred is copied through untouched. Not "penalised
	// by zero": the subtractive terms would be exact, but dividing by
	// Repetition when it is off would still round, and a token the caller never
	// generated must come out of this pass bit-identical to what went in.
	if c == 0 {
		out[i] = l
		return
	}

	if d.Repetition != 0 && d.Repetition != 1 {
		if l > 0 {
			l = l / d.Repetition
		} else {
			l = l * d.Repetition
		}
	}
	out[i] = l - d.Presence - d.Frequency*float32(c)
}

// PenaltyClear zeroes the count buffer.
//
// It exists because the counts are accumulated with an atomic add, which reads
// what is already there. A decode step reuses the same buffer every token, so
// without this the second token is penalised by the first token's history as
// well as its own -- and the sequence still decodes, just increasingly wrongly,
// which is the shape of bug this corpus exists to keep out.
//
//accel:kernel workgroup=64
func PenaltyClear(t accel.Thread, d PenaltyDims, counts []uint32) {
	i := t.GlobalID().X
	if i < d.Vocab {
		counts[i] = 0
	}
}
