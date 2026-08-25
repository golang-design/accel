// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"math"
	"testing"

	"golang.design/x/accel/tensor"
)

// A draw is never 1.0, over the whole range the generator can produce.
//
// specs/039-sampling-policy.md section 2: float32(rng.Float64()) rounds up to
// exactly 1.0 about once in 2^24, and specs/028-sampling.md's walk clamps that
// down rather than failing, so the last token in the vocabulary quietly
// receives the extra mass and every backend agrees about it. Nothing reports
// the bug. The bound is checked here instead.
//
// Sampled across the step space rather than exhaustively over 2^64, but with
// enough steps that a mantissa rounding to one would have to hide in a
// vanishing fraction of them.
func TestADrawIsNeverOne(t *testing.T) {
	s := tensor.Stream{Seed: 0x243F6A8885A308D3}
	var max float32
	for step := range uint64(1 << 21) {
		d := s.Draw(step)
		if d < 0 || d >= 1 {
			t.Fatalf("step %d drew %v, which is outside [0,1)", step, d)
		}
		if d > max {
			max = d
		}
	}
	// The largest representable output. Reaching it is what says the top of the
	// range is used rather than merely not exceeded.
	const ceiling = float32(1<<24-1) / (1 << 24)
	if max != ceiling {
		t.Fatalf("the largest draw over 2^21 steps was %v, want %v: either the "+
			"top of the range is unreachable or the divisor is wrong", max, ceiling)
	}
}

// The draw for a step depends on the step, not on how many draws were taken.
//
// This is the property that lets a caller turn temperature off for one token
// without moving every later one, and it fails for any design that keeps a
// position and advances it.
func TestADrawDependsOnTheStepAndNotOnTheOrderAsked(t *testing.T) {
	s := tensor.Stream{Seed: 7}
	forward := make([]float32, 64)
	for i := range forward {
		forward[i] = s.Draw(uint64(i))
	}
	// Backwards, and skipping every other one on the way, so neither an
	// advancing position nor a cached previous step could reproduce it.
	for i := len(forward) - 1; i >= 0; i-- {
		if got := s.Draw(uint64(i)); got != forward[i] {
			t.Fatalf("step %d drew %v in order and %v out of order", i, forward[i], got)
		}
	}
	for i := 0; i < len(forward); i += 2 {
		if got := s.Draw(uint64(i)); got != forward[i] {
			t.Fatalf("step %d drew %v in order and %v when skipped to", i, forward[i], got)
		}
	}
}

// Copying a Stream copies a number, so two copies cannot interleave.
//
// The failure this rules out is not hypothetical: a policy struct holding a
// *rand.Rand copies the pointer, and the two sequences then share one
// generator. Written as an interleave because that is the shape that catches
// it -- two copies drawing alternately still each produce their own sequence.
func TestTwoCopiesOfAStreamDoNotShareAnything(t *testing.T) {
	a := tensor.Stream{Seed: 99}
	b := a
	want := make([]float32, 32)
	for i := range want {
		want[i] = a.Draw(uint64(i))
	}
	for i := range want {
		if got := b.Draw(uint64(i)); got != want[i] {
			t.Fatalf("step %d: the original drew %v and the copy drew %v after "+
				"interleaving, so the two share state", i, want[i], got)
		}
		if got := a.Draw(uint64(i)); got != want[i] {
			t.Fatalf("step %d: the original drew %v and then %v, so a draw "+
				"advanced something", i, want[i], got)
		}
	}
}

// Adjacent seeds and adjacent steps decorrelate.
//
// Without the avalanche, draw(seed, j) is a function of seed+j: adjacent seeds
// produce shifted copies of one sequence, and adjacent steps drift
// monotonically. Both look fine in a histogram, which is why the property is
// asserted directly.
func TestAdjacentSeedsAndStepsDoNotTrackEachOther(t *testing.T) {
	// Two seeds one apart must not produce shifted copies of one sequence.
	x, y := tensor.Stream{Seed: 1000}, tensor.Stream{Seed: 1001}
	same := 0
	for i := range uint64(256) {
		if x.Draw(i+1) == y.Draw(i) {
			same++
		}
	}
	if same > 1 {
		t.Fatalf("%d of 256 draws from seed 1001 matched the next draw from seed "+
			"1000: the seeds differ by a shift rather than by an avalanche", same)
	}

	// Consecutive steps must not drift in one direction. A monotone stream
	// rises or falls on nearly every step; an even one changes direction about
	// half the time.
	s := tensor.Stream{Seed: 4242}
	rises := 0
	const n = 4096
	prev := s.Draw(0)
	for i := uint64(1); i < n; i++ {
		d := s.Draw(i)
		if d > prev {
			rises++
		}
		prev = d
	}
	if frac := float64(rises) / (n - 1); math.Abs(frac-0.5) > 0.05 {
		t.Fatalf("draws rose on %.1f%% of consecutive steps, want near 50%%: the "+
			"stream drifts rather than mixing", frac*100)
	}
}

// Two sequences of a batch get different streams, and sequence 0 is not the
// root.
func TestDeriveGivesEachSequenceItsOwnStream(t *testing.T) {
	const root = 0xDEADBEEF
	zero := tensor.Derive(root, 0)
	if zero.Seed == root {
		t.Fatal("sequence 0 derived to the root seed itself, so a one-sequence " +
			"batch and an unbatched run would draw the same numbers by accident")
	}
	seen := map[uint64]uint64{}
	for seq := range uint64(64) {
		s := tensor.Derive(root, seq)
		if prev, ok := seen[s.Seed]; ok {
			t.Fatalf("sequences %d and %d derived to the same seed %#x", prev, seq, s.Seed)
		}
		seen[s.Seed] = seq
	}
}

// The value stream is pinned, because pinning it is the point.
//
// A change to the multipliers, the shift, or the divisor is a change to every
// token every caller generates from a saved seed. It should be a test failure
// and a decision, not a diff nobody notices.
func TestTheDrawSequenceIsPinned(t *testing.T) {
	s := tensor.Stream{Seed: 1}
	got := make([]float32, 8)
	for i := range got {
		got[i] = s.Draw(uint64(i))
	}
	// Computed from the formula in specs/039-sampling-policy.md section 2 by a
	// separate implementation of SplitMix64, not copied out of this one. A
	// golden recorded from the code it guards only proves the code did not
	// change; these also say it was right when it was written.
	want := []float32{
		0.3381666, 0.5665615, 0.7457817, 0.9710027,
		0.44435918, 0.44426465, 0.76289433, 0.87734866,
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("step %d drew %v, want %v: the generator's value stream changed, "+
				"so every sequence a caller saved a seed for now decodes differently",
				i, got[i], want[i])
		}
	}
}
