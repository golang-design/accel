// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
	"golang.design/x/accel/kernelabi"
)

func groupedFixture(d testkernels.GroupedDims, counts []uint32) (x, w []float32,
	offsets []uint32) {

	offsets = make([]uint32, d.Experts+1)
	sum := uint32(0)
	for r, c := range counts {
		offsets[r] = sum
		sum += c
	}
	offsets[d.Experts] = sum

	x = make([]float32, sum*d.K)
	for i := range x {
		x[i] = float32(math.Sin(float64(i) * 0.23))
	}
	// Each expert's matrix is scaled by its index, so a token multiplied by the
	// wrong expert's weights is wrong by a factor rather than by a rounding.
	w = make([]float32, d.Experts*d.K*d.N)
	for e := uint32(0); e < d.Experts; e++ {
		for i := uint32(0); i < d.K*d.N; i++ {
			w[e*d.K*d.N+i] = float32(math.Cos(float64(i)*0.17)) * float32(e+1)
		}
	}
	return x, w, offsets
}

func runGrouped(t *testing.T, d testkernels.GroupedDims, x, w []float32,
	offsets []uint32) []float32 {

	t.Helper()
	tokens := offsets[d.Experts]
	out := make([]float32, tokens*d.N)
	if err := kernel.DispatchCooperative(&testkernels.GroupedMatVecKernel,
		accel.ID3{X: tokens * d.N},
		kernelabi.Args{
			Uniforms: []any{d},
			Slices:   []any{x, w, offsets, out},
		}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	return out
}

// Each expert's tokens equal that expert's matrix multiplied alone.
//
// specs/049-grouped-gemm.md §4's first assertion, and the whole claim: a
// grouped product is right when each segment is the ungrouped product of its
// own rows against its own weights.
func TestAGroupedProductMatchesEachExpertAlone(t *testing.T) {
	d := testkernels.GroupedDims{Experts: 3, K: 64, N: 5}
	counts := []uint32{2, 1, 3}
	x, w, offsets := groupedFixture(d, counts)

	got := runGrouped(t, d, x, w, offsets)

	for e := uint32(0); e < d.Experts; e++ {
		for tok := offsets[e]; tok < offsets[e+1]; tok++ {
			for n := uint32(0); n < d.N; n++ {
				// The reference: this token against this expert's matrix, in
				// f64, with no grouping anywhere in it.
				want := 0.0
				for k := uint32(0); k < d.K; k++ {
					want += float64(x[tok*d.K+k]) * float64(w[e*d.K*d.N+k*d.N+n])
				}
				if e := math.Abs(float64(got[tok*d.N+n]) - want); e > 1e-4 {
					t.Fatalf("token %d column %d is %v, want %v", tok, n,
						got[tok*d.N+n], want)
				}
			}
		}
	}
}

// An expert nothing routed to contributes nothing and shifts nothing.
//
// Top-k routing produces this on every step: with two of eight, six experts get
// no tokens. A naive implementation divides by the count.
func TestAGroupedProductAcceptsAnEmptyExpert(t *testing.T) {
	d := testkernels.GroupedDims{Experts: 4, K: 64, N: 3}
	counts := []uint32{2, 0, 2, 0}
	x, w, offsets := groupedFixture(d, counts)

	got := runGrouped(t, d, x, w, offsets)
	if want := int(offsets[d.Experts] * d.N); len(got) != want {
		t.Fatalf("the output is %d elements for %d tokens", len(got), offsets[d.Experts])
	}

	// Token 2 is the first of expert 2, and an off-by-one in the lookup gives
	// it expert 1's matrix -- which is empty, and whose weights are a different
	// multiple.
	for n := uint32(0); n < d.N; n++ {
		want := 0.0
		for k := uint32(0); k < d.K; k++ {
			want += float64(x[2*d.K+k]) * float64(w[2*d.K*d.N+k*d.N+n])
		}
		if e := math.Abs(float64(got[2*d.N+n]) - want); e > 1e-4 {
			t.Fatalf("the first token after an empty expert took the wrong matrix: "+
				"column %d is %v, want %v", n, got[2*d.N+n], want)
		}
	}
}

// Two experts give different answers for the same token.
//
// The mutation this names is a kernel that indexes the weight tensor by
// anything other than the segment it looked up -- including one that always
// reads expert zero, which passes every test with a single expert.
func TestAGroupedProductUsesTheExpertItLookedUp(t *testing.T) {
	d := testkernels.GroupedDims{Experts: 2, K: 64, N: 2}
	x, w, offsets := groupedFixture(d, []uint32{1, 1})

	got := runGrouped(t, d, x, w, offsets)

	// The fixture scales expert e's matrix by e+1, and the two tokens have
	// different x -- so compare each against its own expert rather than against
	// each other.
	for tok, e := range []uint32{0, 1} {
		for n := uint32(0); n < d.N; n++ {
			mine, theirs := 0.0, 0.0
			other := 1 - e
			for k := uint32(0); k < d.K; k++ {
				xv := float64(x[uint32(tok)*d.K+k])
				mine += xv * float64(w[e*d.K*d.N+k*d.N+n])
				theirs += xv * float64(w[other*d.K*d.N+k*d.N+n])
			}
			g := float64(got[uint32(tok)*d.N+n])
			if math.Abs(g-mine) > 1e-4 {
				t.Fatalf("token %d column %d is %v, want %v (its own expert %d)",
					tok, n, g, mine, e)
			}
			if math.Abs(mine-theirs) < 1e-3 {
				t.Fatalf("the two experts' answers for token %d column %d are %v and "+
					"%v, which are too close for this test to tell them apart",
					tok, n, mine, theirs)
			}
		}
	}
}

// The authored grouped kernel and its generated lowering agree.
func TestTheAuthoredGroupedKernelMatchesItsLowering(t *testing.T) {
	d := testkernels.GroupedDims{Experts: 3, K: 64, N: 4}
	x, w, offsets := groupedFixture(d, []uint32{2, 1, 2})
	tokens := offsets[d.Experts]

	authored := make([]float32, tokens*d.N)
	groups := kernel.ID3{X: tokens * d.N, Y: 1, Z: 1}
	for g := range groups.X {
		var sh [128]float32
		kernel.RunAuthored(kernel.ID3{X: 128, Y: 1, Z: 1}, kernel.ID3{X: g},
			groups, 128, func(th kernel.Thread) {
				testkernels.GroupedMatVec(th, d, x, w, offsets, authored, &sh)
			})
	}

	generated := runGrouped(t, d, x, w, offsets)
	for i := range authored {
		if math.Abs(float64(authored[i]-generated[i])) > 1e-5 {
			t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
		}
	}
}

// A token past the last expert's segment is padding: it writes zero and reads
// no weights out of bounds.
//
// [#24](https://github.com/golang-design/accel/issues/24) was filed against the
// ragged attention kernel; this is the same lookup in the same shape, and a fix
// to one kernel and not the other is the divergence sharing a primitive invites.
// Here the stray index lands on `wBase`, so the read is a matrix past the weight
// tensor rather than a neighbouring sequence's cache.
func TestAGroupedTokenPastTheLastExpertIsPadding(t *testing.T) {
	d := testkernels.GroupedDims{Experts: 3, K: 64, N: 4}
	// x holds five tokens; the counts claim four.
	x, w, _ := groupedFixture(d, []uint32{2, 1, 2})
	short := []uint32{0, 2, 2, 4}

	const rows = 5
	out := make([]float32, rows*d.N)
	if err := kernel.DispatchCooperative(&testkernels.GroupedMatVecKernel,
		accel.ID3{X: rows * d.N},
		kernelabi.Args{
			Uniforms: []any{d},
			Slices:   []any{x, w, short, out},
		}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	// The four real tokens still take their own expert's matrix.
	for e := uint32(0); e < d.Experts; e++ {
		for tok := short[e]; tok < short[e+1]; tok++ {
			for n := uint32(0); n < d.N; n++ {
				want := 0.0
				for k := uint32(0); k < d.K; k++ {
					want += float64(x[tok*d.K+k]) * float64(w[e*d.K*d.N+k*d.N+n])
				}
				if got := float64(out[tok*d.N+n]); math.Abs(got-want) > 1e-4 {
					t.Fatalf("padding changed a real token: %d column %d is %v, want %v",
						tok, n, got, want)
				}
			}
		}
	}

	// The pad row is zero, and asserted rather than assumed unwritten.
	for i := int(short[d.Experts]) * int(d.N); i < len(out); i++ {
		if out[i] != 0 {
			t.Fatalf("the pad row is not zero: element %d is %v", i, out[i])
		}
	}

	// The authored form takes the same branch. The differential against the
	// lowering only runs where Metal does, so without this the guard is
	// uncovered on the platform the coverage gate measures.
	authored := make([]float32, len(out))
	groups := kernel.ID3{X: rows * d.N, Y: 1, Z: 1}
	for g := range groups.X {
		var sh [128]float32
		kernel.RunAuthored(kernel.ID3{X: 128, Y: 1, Z: 1}, kernel.ID3{X: g},
			groups, 128, func(th kernel.Thread) {
				testkernels.GroupedMatVec(th, d, x, w, short, authored, &sh)
			})
	}
	for i := range authored {
		if math.Abs(float64(authored[i]-out[i])) > 1e-5 {
			t.Fatalf("element %d: authored %v, generated %v", i, authored[i], out[i])
		}
	}
}
