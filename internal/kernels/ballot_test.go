// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels_test

import (
	"fmt"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/kernels"
	"golang.design/x/accel/kernelabi"
)

// ballotRun dispatches the ballot kernel and returns its five answers per lane.
type ballotOut struct {
	count, lower []int32
	lowest, bit  []uint32
	any          []uint32
	width        int
	size, below  uint32
}

func runBallot(t *testing.T, size, below uint32) ballotOut {
	t.Helper()
	width := int(kernels.BallotKernel.WorkgroupSize.X)
	o := ballotOut{
		count:  make([]int32, width),
		lowest: make([]uint32, width),
		lower:  make([]int32, width),
		bit:    make([]uint32, width),
		any:    make([]uint32, width),
		width:  width, size: size, below: below,
	}
	err := kernel.DispatchCooperativeWith(&kernels.BallotKernel, accel.ID3{X: 1},
		kernelabi.Args{
			Uniforms: []any{kernels.BallotDims{Below: below}},
			Slices:   []any{o.count, o.lowest, o.lower, o.bit, o.any},
		},
		kernel.Options{Diagnostics: true, SubgroupSize: size})
	if err != nil {
		t.Fatalf("dispatch at subgroup size %d, below %d: %v", size, below, err)
	}
	return o
}

// Every one of the mask's five methods answers what the predicate says.
//
// specs/058-ballot.md §5. The reference is computed from the predicate rather
// than from the mask, so a method returning a plausible number -- LowestSet
// giving zero for an empty mask rather than the width -- disagrees.
//
// Swept across emulated subgroup sizes and thresholds, because most single
// combinations are satisfied by more than one wrong implementation: at
// below >= size every lane votes true and Count is the subgroup size whether
// or not the ballot combined anything.
func TestTheBallotAnswersEveryQuestionAboutItsPredicate(t *testing.T) {
	for _, size := range []uint32{1, 4, 32, 64} {
		for _, below := range []uint32{0, 1, 3, 32, 64} {
			t.Run(fmt.Sprintf("size=%d/below=%d", size, below), func(t *testing.T) {
				o := runBallot(t, size, below)

				for i := range o.width {
					lane := uint32(i) % size

					// The predicate is lane < below, and a subgroup has `size`
					// lanes, so this many voted true.
					set := min(below, size)

					if got, want := o.count[i], int32(set); got != want {
						t.Fatalf("lane %d: Count is %d, want %d", i, got, want)
					}
					// The lowest set lane is 0 when anything voted, and the
					// mask's width when nothing did.
					want := uint32(128)
					if set > 0 {
						want = 0
					}
					if got := o.lowest[i]; got != want {
						t.Fatalf("lane %d: LowestSet is %d, want %d", i, got, want)
					}
					// Every set bit is below `set`, so the count under this
					// lane is the lane index capped there.
					if got, want := o.lower[i], int32(min(lane, set)); got != want {
						t.Fatalf("lane %d: CountLower(%d) is %d, want %d", i, lane, got, want)
					}
					if got, want := o.bit[i], b32(lane < below); got != want {
						t.Fatalf("lane %d: Bit(%d) is %d, want %d", i, lane, got, want)
					}
					if got, want := o.any[i], b32(set > 0); got != want {
						t.Fatalf("lane %d: Any is %d, want %d", i, got, want)
					}
				}
			})
		}
	}
}

func b32(v bool) uint32 {
	if v {
		return 1
	}
	return 0
}

// The mask is broadcast to every lane, not computed per lane.
//
// Four of the five answers are subgroup-uniform, so a lowering that gave each
// lane a mask holding only its own bit would report Count == 1 everywhere. That
// is indistinguishable from a correct Count at below == 1, which is why this
// asserts at a threshold where the two differ and asserts the *uniformity*
// rather than the value.
func TestTheBallotIsBroadcastToEveryLane(t *testing.T) {
	const size = 4
	o := runBallot(t, size, 3)

	for i := range o.width {
		first := i - i%size // the lowest lane of this subgroup
		if o.count[i] != o.count[first] {
			t.Fatalf("lane %d reports Count %d and lane %d of its own subgroup reports "+
				"%d: the mask was not broadcast", i, o.count[i], first, o.count[first])
		}
		if o.lowest[i] != o.lowest[first] {
			t.Fatalf("lane %d reports LowestSet %d and lane %d reports %d",
				i, o.lowest[i], first, o.lowest[first])
		}
	}
	// And the value it agrees on is not 1, which is what a per-lane mask would
	// produce and what would make the agreement above vacuous.
	if o.count[0] != 3 {
		t.Fatalf("Count is %d at a threshold of 3, so this test cannot tell a "+
			"broadcast mask from a per-lane one", o.count[0])
	}
}

// CountLower over the ballot equals an exclusive scan of the predicate.
//
// The use the method exists for, and the one that catches an off-by-one:
// including the lane's own bit would make every voting lane's answer one too
// high, which every other assertion here still passes.
func TestCountLowerIsAnExclusiveScanOverTheBallot(t *testing.T) {
	const size = 8
	o := runBallot(t, size, 5)

	for i := range o.width {
		lane := uint32(i) % size
		// The scan, computed lane by lane from the predicate.
		want := int32(0)
		for l := range lane {
			if l < o.below {
				want++
			}
		}
		if got := o.lower[i]; got != want {
			t.Fatalf("lane %d: CountLower is %d and the exclusive scan of the predicate "+
				"is %d", i, got, want)
		}
	}
}

// The Metal lowering is refused, naming what Metal lacks.
//
// specs/058-ballot.md §3. This is the first kernel-visible capability the first
// backend does not have, so the refusal has to read as "this device cannot"
// rather than as "this compiler has not implemented it" -- 004's rule that a
// target-specific rejection names the target.
func TestTheBallotIsRefusedOnMetal(t *testing.T) {
	if kernels.BallotKernel.MSL != "" {
		t.Fatal("the ballot kernel carries MSL, and MSL cannot spell a ballot: " +
			"simd_ballot returns a simd_vote (specs/022-msl-target.md section 5)")
	}
	// The kernel still declares the capability, which is what a device query
	// answers false for. Without this the absent MSL would be
	// indistinguishable from a kernel that failed to lower for some unrelated
	// reason.
	want := uint32(accel.CapSubgroupBallot)
	if got := kernels.BallotKernel.Caps; got&want == 0 {
		t.Errorf("the ballot kernel declares caps %#x, which does not include "+
			"subgroup_ballot (%#x)", got, want)
	}
}

// The authored ballot and its generated lowering agree, at a subgroup of one.
//
// specs/010-kernel-corpus.md §6, run the way the subgroup reduction's authored
// half is run and for the same reason: `Thread.SubgroupBallot` returns a mask
// holding only the calling lane's bit, because combining is the scheduler's
// job and not the method's. At subgroup size 1 that **is** the right answer --
// a subgroup of one lane ballots to one bit -- so the comparison is against a
// correct expectation rather than against a stub's artefact.
//
// What it checks is therefore the five methods and the plumbing around them,
// not the rendezvous. The rendezvous is checked by the tests above, where the
// generated path runs at real subgroup widths and every lane must see the whole
// subgroup's vote.
func TestTheAuthoredBallotMatchesItsLoweringAtOneLane(t *testing.T) {
	width := int(kernels.BallotKernel.WorkgroupSize.X)
	d := kernels.BallotDims{Below: 5}

	a := ballotOut{
		count:  make([]int32, width),
		lowest: make([]uint32, width),
		lower:  make([]int32, width),
		bit:    make([]uint32, width),
		any:    make([]uint32, width),
	}
	kernel.RunAuthored(&kernels.BallotKernel, kernel.ID3{}, kernel.ID3{X: 1}, 1,
		func(th kernel.Thread) {
			kernels.Ballot(th, d, a.count, a.lowest, a.lower, a.bit, a.any)
		})

	g := runBallot(t, 1, 5)
	for i := range width {
		if a.count[i] != g.count[i] || a.lowest[i] != g.lowest[i] ||
			a.lower[i] != g.lower[i] || a.bit[i] != g.bit[i] || a.any[i] != g.any[i] {
			t.Fatalf("lane %d: authored (%d %d %d %d %d), generated (%d %d %d %d %d)",
				i, a.count[i], a.lowest[i], a.lower[i], a.bit[i], a.any[i],
				g.count[i], g.lowest[i], g.lower[i], g.bit[i], g.any[i])
		}
	}
	// Not five zeros agreeing: at a subgroup of one, lane 0 votes true and
	// every lane's own bit is set, so Count is 1 rather than 0.
	if a.count[0] != 1 {
		t.Fatalf("Count is %d at a subgroup of one, so the comparison says nothing",
			a.count[0])
	}
}
