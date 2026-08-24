// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"golang.design/x/accel/kernelabi"
	"math"
	"sort"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
)

func topK(t *testing.T, w []float32, k int) []float32 {
	t.Helper()
	out := make([]float32, len(w))
	err := kernel.DispatchCooperative(&testkernels.TopKMaskKernel, accel.ID3{X: 1},
		kernelabi.Args{
			Slices: []any{w, out},
			Uniforms: []any{testkernels.TopDims{
				Vocab: uint32(len(w)), K: uint32(k),
			}},
		})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	return out
}

func topP(t *testing.T, w []float32, p float32) []float32 {
	t.Helper()
	out := make([]float32, len(w))
	err := kernel.DispatchCooperative(&testkernels.TopPMaskKernel, accel.ID3{X: 1},
		kernelabi.Args{
			Slices: []any{w, out},
			Uniforms: []any{testkernels.TopDims{
				Vocab: uint32(len(w)), P: p,
			}},
		})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	return out
}

// kept reports which indices survived a mask, in order.
func kept(out []float32) []int {
	var idx []int
	for i, v := range out {
		if v != 0 {
			idx = append(idx, i)
		}
	}
	return idx
}

// Top-k keeps exactly k entries, and the right ones.
//
// "Exactly k" is the part worth asserting. A threshold-based implementation
// keeps however many happen to sit above the threshold, which is k only when no
// two entries tie -- and a distribution with ties is the normal case near the
// tail, not an edge case.
func TestTopKKeepsExactlyKAndTheLargest(t *testing.T) {
	for _, tc := range []struct {
		name string
		w    []float32
		k    int
		want []int
	}{
		{"distinct values", []float32{0.1, 0.5, 0.2, 0.4, 0.3}, 2, []int{1, 3}},
		{"k of one", []float32{0.1, 0.5, 0.2}, 1, []int{1}},
		{"k equal to the vocabulary", []float32{0.3, 0.1, 0.2}, 3, []int{0, 1, 2}},
		// Ties: the lowest indices win, so exactly k survive rather than four.
		{"a tie at the boundary", []float32{5, 5, 5, 5, 1}, 2, []int{0, 1}},
		{"all equal", []float32{2, 2, 2, 2}, 3, []int{0, 1, 2}},
		{"zeros among the values", []float32{0, 3, 0, 1, 0}, 2, []int{1, 3}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := topK(t, tc.w, tc.k)
			got := kept(out)
			if len(got) != len(tc.want) {
				t.Fatalf("kept %v, want %v", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Fatalf("kept %v, want %v", got, tc.want)
				}
			}
			// A kept entry carries its input weight, so the result feeds the
			// sampler without a renormalizing pass.
			for _, i := range got {
				if out[i] != tc.w[i] {
					t.Errorf("index %d was kept as %v and its input was %v",
						i, out[i], tc.w[i])
				}
			}
		})
	}
}

// Top-k over a real distribution keeps the k largest, checked against a sort.
//
// The sort is the reference: written from the definition of "the k largest"
// rather than from the kernel's extraction, so agreement is about the answer
// and not about a shared method.
func TestTopKMatchesASort(t *testing.T) {
	const n, k = 500, 17
	w := make([]float32, n)
	for i := range w {
		w[i] = float32(math.Abs(math.Sin(float64(i) * 0.37)))
	}

	type pair struct {
		v float32
		i int
	}
	ps := make([]pair, n)
	for i, v := range w {
		ps[i] = pair{v, i}
	}
	sort.SliceStable(ps, func(a, b int) bool {
		if ps[a].v != ps[b].v {
			return ps[a].v > ps[b].v
		}
		return ps[a].i < ps[b].i
	})
	want := map[int]bool{}
	for _, p := range ps[:k] {
		want[p.i] = true
	}

	got := kept(topK(t, w, k))
	if len(got) != k {
		t.Fatalf("kept %d entries, want %d", len(got), k)
	}
	for _, i := range got {
		if !want[i] {
			t.Errorf("index %d was kept and is not among the %d largest", i, k)
		}
	}
}

// Top-p keeps the smallest set of largest entries whose mass reaches p.
//
// Smallest, which is why the entry that crosses the threshold is kept and not
// the one before it: a nucleus that stopped short would hold less than p of the
// mass, which is the opposite of what the parameter asks for.
func TestTopPKeepsTheNucleus(t *testing.T) {
	// Sorted mass: 0.4, 0.3, 0.2, 0.1 at indices 3, 2, 1, 0.
	w := []float32{0.1, 0.2, 0.3, 0.4}

	for _, tc := range []struct {
		p    float32
		want []int
	}{
		{0.3, []int{3}},       // 0.4 alone already reaches 0.3
		{0.4, []int{3}},       // exactly 0.4: the first entry reaches it
		{0.5, []int{2, 3}},    // 0.4 is short, 0.4+0.3 crosses
		{0.7, []int{2, 3}},    // exactly 0.7
		{0.9, []int{1, 2, 3}}, // needs the third
		{1.0, []int{0, 1, 2, 3}},
	} {
		got := kept(topP(t, w, tc.p))
		if len(got) != len(tc.want) {
			t.Fatalf("p=%v kept %v, want %v", tc.p, got, tc.want)
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Fatalf("p=%v kept %v, want %v", tc.p, got, tc.want)
			}
		}
	}
}

// Top-p works on unnormalized weights, because the threshold is a fraction of
// the input's own total.
//
// That is what lets it compose after a top-k mask, which leaves a distribution
// summing to less than one.
func TestTopPIsRelativeToItsInput(t *testing.T) {
	normalized := []float32{0.1, 0.2, 0.3, 0.4}
	scaled := make([]float32, len(normalized))
	for i, v := range normalized {
		scaled[i] = v * 37 // an arbitrary total
	}
	a := kept(topP(t, normalized, 0.5))
	b := kept(topP(t, scaled, 0.5))
	if len(a) != len(b) {
		t.Fatalf("normalized kept %v and scaled kept %v", a, b)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("normalized kept %v and scaled kept %v", a, b)
		}
	}

	// And composed: a top-k mask followed by a top-p over what it left.
	w := make([]float32, 64)
	for i := range w {
		w[i] = float32(i + 1)
	}
	afterK := topK(t, w, 8)
	afterP := kept(topP(t, afterK, 0.5))
	if len(afterP) == 0 || len(afterP) > 8 {
		t.Fatalf("a top-p over a top-k of 8 kept %d entries", len(afterP))
	}
	// Everything it kept must have survived the top-k too.
	inK := map[int]bool{}
	for _, i := range kept(afterK) {
		inK[i] = true
	}
	for _, i := range afterP {
		if !inK[i] {
			t.Errorf("index %d survived top-p and not top-k", i)
		}
	}
}

// A truncation asking for more than the kernel can walk keeps what it can and
// does not read past its bound.
//
// The cap is a real limit rather than a buffer size, and a truncation that
// silently kept fewer entries than asked would change what a model samples
// without changing what it reports -- so the behaviour is pinned here.
func TestTruncationRespectsItsBound(t *testing.T) {
	const n = 400
	w := make([]float32, n)
	for i := range w {
		w[i] = float32(n - i) // strictly descending, so the answer is a prefix
	}
	got := kept(topK(t, w, n))
	if len(got) != testkernels.TopMaxRounds {
		t.Fatalf("a top-k of %d over a vocabulary of %d kept %d, and the bound is %d",
			n, n, len(got), testkernels.TopMaxRounds)
	}
	for i, idx := range got {
		if idx != i {
			t.Fatalf("the weights descend, so the kept set should be the first %d indices; "+
				"position %d is index %d", testkernels.TopMaxRounds, i, idx)
		}
	}
}

// Each row of a batch is truncated against its own distribution.
//
// The masks used to index from zero and their whole grid was one workgroup, so
// a batch was not expressible at all. Now the grid is one workgroup per row and
// the row offset comes from the group id, which is the arrangement
// [testkernels.SampleArgmax] has always used.
//
// The rows here differ in *scale* rather than in shape, which is what makes
// this catch the interesting fault. Top-p accumulates a fraction of its row's
// own total, so a sum loop left indexing from zero gives row 1 row 0's
// threshold: with row 1 ten times larger, that threshold is reached by its
// first entry alone and the mask keeps one token instead of the nucleus. The
// result is a plausible mask over the right vocabulary, and only a comparison
// against the same row run alone finds it.
func TestTruncationBatchesByRow(t *testing.T) {
	rowA := []float32{1, 2, 3, 4, 5, 6}
	rowB := []float32{60, 10, 50, 20, 40, 30}

	batched := func(k *accel.Kernel, d testkernels.TopDims, rows ...[]float32) [][]float32 {
		t.Helper()
		vocab := len(rows[0])
		in := make([]float32, 0, len(rows)*vocab)
		for _, r := range rows {
			in = append(in, r...)
		}
		out := make([]float32, len(in))
		d.Vocab = uint32(vocab)
		if err := kernel.DispatchCooperative(k, accel.ID3{X: uint32(len(rows))},
			kernelabi.Args{
				Slices: []any{in, out}, Uniforms: []any{d},
			}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		split := make([][]float32, len(rows))
		for i := range rows {
			split[i] = out[i*vocab : (i+1)*vocab]
		}
		return split
	}

	t.Run("TopKMask", func(t *testing.T) {
		got := batched(&testkernels.TopKMaskKernel, testkernels.TopDims{K: 2}, rowA, rowB)
		// Each row alone is the oracle: a batch of two must equal two batches
		// of one, element for element.
		for i, row := range [][]float32{rowA, rowB} {
			want := topK(t, row, 2)
			for j := range want {
				if got[i][j] != want[j] {
					t.Fatalf("row %d element %d: batched %v, alone %v",
						i, j, got[i][j], want[j])
				}
			}
		}
	})

	t.Run("TopPMask", func(t *testing.T) {
		got := batched(&testkernels.TopPMaskKernel, testkernels.TopDims{P: 0.5}, rowA, rowB)
		for i, row := range [][]float32{rowA, rowB} {
			want := topP(t, row, 0.5)
			for j := range want {
				if got[i][j] != want[j] {
					t.Fatalf("row %d element %d: batched %v, alone %v",
						i, j, got[i][j], want[j])
				}
			}
		}
	})
}
