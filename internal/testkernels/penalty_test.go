// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"testing"

	"golang.design/x/accel/internal/testkernels"
	"golang.design/x/accel/kernelabi"
)

// penalise runs the three passes the way a decode step does, and returns the
// penalised logits.
//
// Written as a helper because the order is the thing under test everywhere
// below: clear, count, apply. A test that assembled it differently each time
// would be testing its own arrangement.
func penalise(t *testing.T, d testkernels.PenaltyDims, logits []float32, history []uint32) []float32 {
	t.Helper()
	counts := make([]uint32, d.Vocab)
	out := make([]float32, d.Vocab)
	runFlat(t, &testkernels.PenaltyClearKernel, int(d.Vocab),
		kernelabi.Args{Uniforms: []any{d}, Slices: []any{counts}})
	runFlat(t, &testkernels.PenaltyCountKernel, len(history),
		kernelabi.Args{Uniforms: []any{d}, Slices: []any{history, counts}})
	runFlat(t, &testkernels.PenaltyApplyKernel, int(d.Vocab),
		kernelabi.Args{Uniforms: []any{d}, Slices: []any{logits, counts, out}})
	return out
}

// A token that never occurred comes out bit-identical.
//
// Not "penalised by zero". The subtractive terms would be exact, but dividing
// by Repetition rounds, so an implementation that applies the arithmetic
// unconditionally moves every logit in the vocabulary by an ulp or two. That
// changes which token wins a close comparison, on tokens the caller never
// generated.
func TestAnUnseenTokenIsUntouchedByAPenalty(t *testing.T) {
	logits := []float32{0.1, -0.3, 2.7, -1.9, 0.30000001, 1e-30, -1e-30, 3.3}
	d := testkernels.PenaltyDims{
		Vocab: uint32(len(logits)), History: 4, Count: 2,
		Repetition: 1.3, Presence: 0.5, Frequency: 0.25,
	}
	// Only ids 2 and 5 occur.
	got := penalise(t, d, logits, []uint32{2, 5, 0, 0})
	for i, want := range logits {
		if i == 2 || i == 5 {
			continue
		}
		if got[i] != want {
			t.Fatalf("token %d was not in the history and moved from %v to %v: an "+
				"unpenalised logit must come through bit-identical", i, want, got[i])
		}
	}
}

// The divisive penalty moves a negative logit down, not up.
//
// Dividing a negative number by r > 1 moves it toward zero, which is upward:
// it rewards the repeat. Logits have no fixed zero, so the negative branch is
// ordinary rather than a corner.
func TestTheDivisivePenaltyMovesANegativeLogitDown(t *testing.T) {
	logits := []float32{2.0, -2.0}
	d := testkernels.PenaltyDims{Vocab: 2, History: 2, Count: 2, Repetition: 2}
	got := penalise(t, d, logits, []uint32{0, 1})

	if got[0] >= logits[0] {
		t.Fatalf("a positive logit went from %v to %v: the penalty must lower it",
			logits[0], got[0])
	}
	if got[1] >= logits[1] {
		t.Fatalf("a negative logit went from %v to %v: dividing it by r moves it "+
			"toward zero, which rewards the repeat the penalty was meant to punish",
			logits[1], got[1])
	}
	if want := float32(-4); got[1] != want {
		t.Fatalf("a negative logit under r=2 gave %v, want %v: the branch multiplies",
			got[1], want)
	}
}

// The frequency penalty scales with the count and the presence penalty does not.
//
// The mutation this catches is the pair being swapped, which is invisible at a
// count of one and is the count that most tests would use.
func TestFrequencyScalesWithTheCountAndPresenceDoesNot(t *testing.T) {
	logits := []float32{0, 0, 0}
	d := testkernels.PenaltyDims{
		Vocab: 3, History: 8, Count: 6,
		Presence: 1, Frequency: 0.5,
	}
	// id 0 once, id 1 twice, id 2 three times.
	got := penalise(t, d, logits, []uint32{0, 1, 1, 2, 2, 2, 0, 0})
	want := []float32{-1.5, -2, -2.5}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d occurring %d times gave %v, want %v", i, i+1, got[i], want[i])
		}
	}
}

// Only the filled part of the ring is counted.
//
// The ring is bound at its full capacity because binding it at its current
// length would change the input shape every token, which is a new plan every
// token. So the tail is whatever was there before, and reading it counts those
// entries as occurrences. Token 0 is a real id, so the usual leftover -- zero --
// is not detectably wrong from the output.
func TestOnlyTheFilledPartOfTheHistoryIsCounted(t *testing.T) {
	logits := []float32{0, 0, 0, 0}
	d := testkernels.PenaltyDims{Vocab: 4, History: 8, Count: 3, Presence: 1}
	// Three real entries; the rest is a previous sequence's leftovers.
	got := penalise(t, d, logits, []uint32{1, 2, 3, 0, 0, 0, 0, 0})
	if got[0] != 0 {
		t.Fatalf("token 0 was penalised by %v with a Count of 3: the unfilled tail "+
			"of the ring was counted, and its zeros are a real token id", -got[0])
	}
	for i := 1; i < 4; i++ {
		if got[i] != -1 {
			t.Fatalf("token %d was penalised by %v, want 1", i, -got[i])
		}
	}
}

// A history entry outside the vocabulary is dropped, not clamped.
//
// Clamping would attribute one sequence's repeat to the last token in the
// vocabulary, which is a plausible-looking penalty on a token nobody generated.
func TestAnOutOfRangeHistoryEntryIsDropped(t *testing.T) {
	logits := []float32{0, 0, 0, 0}
	d := testkernels.PenaltyDims{Vocab: 4, History: 4, Count: 4, Presence: 1}
	got := penalise(t, d, logits, []uint32{9, 4, 1, 4})
	if got[3] != 0 {
		t.Fatalf("the last token was penalised by %v: an id past the vocabulary was "+
			"clamped into it rather than dropped", -got[3])
	}
	if got[1] != -1 {
		t.Fatalf("token 1 was penalised by %v, want 1", -got[1])
	}
}

// Counting the same history twice is the bug clearing exists to stop.
//
// The counts accumulate with an atomic add, so a step that reuses the buffer
// without clearing it is penalised by every previous token as well as its own.
// The sequence still decodes; it decodes increasingly wrongly.
func TestCountsAreClearedBetweenSteps(t *testing.T) {
	const vocab = 4
	d := testkernels.PenaltyDims{Vocab: vocab, History: 2, Count: 2, Presence: 1}
	history := []uint32{1, 1}
	counts := make([]uint32, vocab)
	logits := []float32{0, 0, 0, 0}
	out := make([]float32, vocab)

	step := func() float32 {
		runFlat(t, &testkernels.PenaltyClearKernel, vocab,
			kernelabi.Args{Uniforms: []any{d}, Slices: []any{counts}})
		runFlat(t, &testkernels.PenaltyCountKernel, len(history),
			kernelabi.Args{Uniforms: []any{d}, Slices: []any{history, counts}})
		runFlat(t, &testkernels.PenaltyApplyKernel, vocab,
			kernelabi.Args{Uniforms: []any{d}, Slices: []any{logits, counts, out}})
		return out[1]
	}
	first := step()
	if second := step(); second != first {
		t.Fatalf("the same history penalised token 1 by %v and then %v: the counts "+
			"carried over from the previous step", -first, -second)
	}
}

// The counts do not depend on the order the history is walked.
//
// This is why pass one counts with an integer atomic instead of subtracting
// from the logits directly. The property is asserted on the arithmetic: the
// penalty for three occurrences is one subtraction of 3f, not three
// subtractions of f, and in f32 those are different numbers.
func TestAPenaltyIsOneUpdatePerTokenAndNotOnePerOccurrence(t *testing.T) {
	// A logit and a coefficient whose repeated subtraction rounds differently
	// from the single one.
	const f = 0.1
	logits := []float32{1}
	d := testkernels.PenaltyDims{Vocab: 1, History: 3, Count: 3, Frequency: f}
	got := penalise(t, d, logits, []uint32{0, 0, 0})

	once := float32(1) - f*float32(3)
	thrice := float32(1)
	for range 3 {
		thrice -= f
	}
	if once == thrice {
		t.Skip("this platform's f32 does not distinguish the two spellings, so " +
			"the assertion below cannot fail for the right reason")
	}
	if got[0] != once {
		t.Fatalf("three occurrences gave %v, want %v (the repeated-subtraction "+
			"spelling gives %v): the penalty must be applied once from a final count",
			got[0], once, thrice)
	}
}
