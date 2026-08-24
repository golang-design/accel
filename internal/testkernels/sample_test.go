// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"golang.design/x/accel/kernelabi"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
)

func argmax(t *testing.T, logits []float32) uint32 {
	t.Helper()
	out := make([]uint32, 1)
	err := kernel.DispatchCooperative(&testkernels.SampleArgmaxKernel, accel.ID3{X: 1},
		kernelabi.Args{
			Slices: []any{logits, out},
			Uniforms: []any{testkernels.SampleDims{
				Vocab: uint32(len(logits)), Rows: 1,
			}},
		})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	return out[0]
}

func categorical(t *testing.T, probs []float32, draw float32) uint32 {
	t.Helper()
	out := make([]uint32, 1)
	err := kernel.Dispatch(&testkernels.SampleCategoricalKernel, accel.ID3{X: 1},
		kernelabi.Args{
			Slices: []any{probs, []float32{draw}, out},
			Uniforms: []any{testkernels.SampleDims{
				Vocab: uint32(len(probs)), Rows: 1,
			}},
		})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	return out[0]
}

// Argmax finds the largest logit, and gives ties to the lowest index.
//
// The tie rule is the substance. Equal logits are ordinary -- an untrained model
// produces them everywhere -- and leaving the answer to whichever lane compared
// which pair would make two backends reducing at different widths return
// different tokens. A test that only used distinct values would pass for an
// implementation with no tie rule at all.
func TestArgmaxAndItsTieRule(t *testing.T) {
	for _, tc := range []struct {
		name   string
		logits []float32
		want   uint32
	}{
		{"a clear winner", []float32{1, 5, 2, 3}, 1},
		{"the winner is first", []float32{9, 1, 2}, 0},
		{"the winner is last", []float32{1, 2, 9}, 2},
		{"all equal", []float32{3, 3, 3, 3, 3}, 0},
		{"a plateau at the top", []float32{1, 7, 7, 7, 2}, 1},
		{"a plateau spanning the reduction's halves", func() []float32 {
			// Equal maxima far apart, so the pair that meets them is formed
			// deep in the tree rather than at the first step. An
			// implementation whose tie rule only worked adjacent-to-adjacent
			// passes the case above and fails this one.
			v := make([]float32, 200)
			for i := range v {
				v[i] = -1
			}
			v[7] = 5
			v[150] = 5
			return v
		}(), 7},
		{"negatives only", []float32{-5, -2, -9}, 1},
		{"a single logit", []float32{42}, 0},
		{"more logits than lanes", func() []float32 {
			v := make([]float32, 1000)
			for i := range v {
				v[i] = float32(math.Sin(float64(i) * 0.1))
			}
			v[613] = 10
			return v
		}(), 613},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := argmax(t, tc.logits); got != tc.want {
				t.Errorf("argmax = %d, want %d", got, tc.want)
			}
		})
	}
}

// Categorical sampling returns the index the cumulative distribution names, at
// every boundary.
//
// Checked at boundaries rather than at random draws, because a boundary is
// where an off-by-one lives: a walk using >= instead of > shifts every answer
// by one index for exactly the draws that land on a cumulative sum.
func TestCategoricalWalksTheDistribution(t *testing.T) {
	// Cumulative: 0.1, 0.3, 0.6, 1.0
	probs := []float32{0.1, 0.2, 0.3, 0.4}

	for _, tc := range []struct {
		draw float32
		want uint32
	}{
		{0, 0},    // the very bottom
		{0.05, 0}, // inside the first
		{0.1, 1},  // exactly the first boundary: > means it moves on
		{0.15, 1},
		{0.3, 2}, // the second boundary
		{0.45, 2},
		{0.6, 3}, // the third
		{0.99, 3},
		{0.999999, 3}, // just below one
	} {
		if got := categorical(t, probs, tc.draw); got != tc.want {
			t.Errorf("a draw of %v gave %d, want %d", tc.draw, got, tc.want)
		}
	}

	// A draw outside the range clamps rather than reading past the end. A
	// kernel cannot report an error, so the only alternative is undefined.
	if got := categorical(t, probs, -0.5); got != 0 {
		t.Errorf("a negative draw gave %d, want the first index", got)
	}
	if got := categorical(t, probs, 1.5); got != 3 {
		t.Errorf("a draw above one gave %d, want the last index", got)
	}
	if got := categorical(t, probs, 1); got != 3 {
		t.Errorf("a draw of exactly one gave %d, want the last index", got)
	}

	// A distribution summing below one returns the last index rather than
	// falling off the end. Softmax divides by a sum computed in f32, so its
	// output really can land a few ulps short.
	short := []float32{0.25, 0.25, 0.25, 0.2499}
	if got := categorical(t, short, 0.9999); got != 3 {
		t.Errorf("a distribution summing to %v gave %d for a draw past its total, want the "+
			"last index", 0.9999, got)
	}
}

// Sampling composes with softmax the way a decode step uses it: temperature,
// then probabilities, then a draw.
//
// The check is distributional rather than exact: over many draws the sampled
// indices must follow the distribution, which is the property a caller relies
// on and which no single draw can show.
func TestSamplingFollowsTheDistribution(t *testing.T) {
	const vocab, rows = 4, 1
	logits := []float32{0, 1, 2, 3}
	probs := make([]float32, vocab)
	// Softmax through the corpus kernel, so this tests the pair rather than a
	// host-side distribution the kernel never sees.
	err := kernel.DispatchCooperative(&testkernels.SoftmaxKernel, accel.ID3{X: rows},
		kernelabi.Args{
			Slices:   []any{logits, probs},
			Uniforms: []any{testkernels.RowDims{Rows: rows, Width: vocab}},
		})
	if err != nil {
		t.Fatalf("softmax: %v", err)
	}

	const draws = 4000
	counts := make([]int, vocab)
	for i := range draws {
		// A deterministic sweep rather than a random source: the draws are an
		// input, so the test supplies them, and a sweep covers the distribution
		// exactly rather than approximately.
		d := (float32(i) + 0.5) / draws
		counts[categorical(t, probs, d)]++
	}
	for i := range vocab {
		want := float64(probs[i]) * draws
		if got := float64(counts[i]); math.Abs(got-want) > 2 {
			t.Errorf("index %d was drawn %v times over a sweep and its probability predicts "+
				"%v", i, got, want)
		}
	}
}

// Two rows with identical distributions and different draws emit different
// tokens.
//
// This is the failure a shared draw produced and no test caught. specs/028's
// choice to make the randomness an input is right — a token is reproducible
// only if the caller supplies it — but one draw per *dispatch* keeps
// reproducibility and destroys **independence**: two sequences whose
// distributions are similar, which is common for a well-trained model answering
// related prompts, draw against the same u and emit the same token. Their
// contexts converge, the next distributions are more similar still, and two
// users get the same answer.
//
// Every existing test passed, because reproducibility is what they check and
// reproducibility is exactly what was preserved. So the assertion here is the
// one they could not make: identical rows, different draws, different tokens.
func TestABatchedSampleDrawsIndependentlyPerRow(t *testing.T) {
	const vocab, rows = 8, 3
	// The same distribution in every row, so nothing but the draw can separate
	// the results.
	probs := make([]float32, rows*vocab)
	for r := range rows {
		for i := range vocab {
			probs[r*vocab+i] = 1.0 / vocab
		}
	}
	// Draws in different buckets of a uniform distribution over eight.
	draws := []float32{0.05, 0.45, 0.95}

	out := make([]uint32, rows)
	err := kernel.Dispatch(&testkernels.SampleCategoricalKernel, accel.ID3{X: rows},
		kernelabi.Args{
			Slices:   []any{probs, draws, out},
			Uniforms: []any{testkernels.SampleDims{Vocab: vocab, Rows: rows}},
		})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	if out[0] == out[1] && out[1] == out[2] {
		t.Fatalf("three rows with identical distributions and draws %v all emitted token "+
			"%d; a shared draw makes two users get the same answer", draws, out[0])
	}
	// And each is the token its own draw selects, so this is per-row rather
	// than merely varied.
	for r := range rows {
		want := uint32(float64(draws[r]) * vocab)
		if out[r] != want {
			t.Errorf("row %d drew %v over a uniform distribution and got token %d, want %d",
				r, draws[r], out[r], want)
		}
	}
}

// A batched argmax reduces each row independently and returns an index within
// that row.
//
// Within the row, because what a caller wants back is a token id rather than an
// offset into the batch — and a reduction that shared partials across rows
// would return the batch's best for every row, which looks like a model that
// suddenly agrees with itself.
func TestABatchedArgmaxIsPerRow(t *testing.T) {
	const vocab, rows = 16, 3
	logits := make([]float32, rows*vocab)
	want := []uint32{2, 11, 7}
	for r := range rows {
		for i := range vocab {
			logits[r*vocab+i] = float32(i%5) - 3
		}
		logits[r*vocab+int(want[r])] = 100
	}

	out := make([]uint32, rows)
	err := kernel.DispatchCooperative(&testkernels.SampleArgmaxKernel, accel.ID3{X: rows},
		kernelabi.Args{
			Slices:   []any{logits, out},
			Uniforms: []any{testkernels.SampleDims{Vocab: vocab, Rows: rows}},
		})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	for r := range rows {
		if out[r] != want[r] {
			t.Errorf("row %d picked %d, want %d; an index outside [0,%d) would be an "+
				"offset into the batch rather than a token id", r, out[r], want[r], vocab)
		}
	}
}
