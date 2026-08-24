// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel_test

import (
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernel"
)

// The inactive-lane rules of specs/002-compute-model.md section 5.2.
//
// # Why these are driven by hand rather than by a compiled kernel
//
// An active set with a hole in it — lanes 0, 2 and 3 taking part while lane 1
// does not — is what the five rules are written about, and no generated
// lowering produces one: a subgroup operation inside a conditional is refused,
// because the state machine has no way to resume inside a branch. So the
// scheduler is driven directly, which is also the level the rules live at. The
// generated path is exercised by the corpus kernel in internal/testkernels.

// lanes is one lane's view of a rendezvous.
type laneView struct {
	f float32
	b bool
	m kernel.Mask
}

// runPartial runs one rendezvous over a single subgroup in which only the named
// lanes take part, and reports what each of them read back.
//
// A lane that is not in the set returns without suspending, which is what makes
// it inactive: it is not at this operation. Nothing else in the scheduler is
// special-cased, so what this measures is the combination step itself.
func runPartial(t *testing.T, width uint32, active []uint32, op kernel.SubgroupOp,
	give func(lane uint32) (float32, bool, uint32), diag bool) (map[uint32]laneView, error) {
	t.Helper()

	out := map[uint32]laneView{}
	k := &kernel.Kernel{
		Name: "Partial", WorkgroupSize: kernel.ID3{X: width, Y: 1, Z: 1},
		Generator: kernel.ABIVersion, Suspensions: 1,
		Cooperative: func(th kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			lane := th.SubgroupLane()
			if !slices.Contains(active, lane) {
				return false
			}
			if f.Pass == 0 {
				f.Pass = 1
				f.Sub = op
				f.SubF32, f.SubBool, f.SubLane = give(lane)
				f.Barrier = kernel.BarrierID{Index: 0}
				return true
			}
			out[lane] = laneView{f: f.SubF32, b: f.SubBool, m: f.SubMask}
			return false
		},
	}
	err := kernel.DispatchCooperativeWith(k, kernel.ID3{X: 1}, kernel.Args{},
		kernel.Options{SubgroupSize: width, Diagnostics: diag})
	return out, err
}

// Rule 1: an inactive lane is not present, so "the lowest lane" means the
// lowest *active* one.
//
// Lane 0 sits out, and both operations that name a lane by position have to
// name lane 1. An implementation that read position zero of its own bookkeeping
// would pass a test where the active set is a prefix, which is why the hole is
// at the bottom here.
func TestTheLowestLaneMeansTheLowestActiveLane(t *testing.T) {
	give := func(lane uint32) (float32, bool, uint32) { return float32(lane) + 10, false, 0 }

	elected, err := runPartial(t, 4, []uint32{1, 2, 3}, kernel.SubElect, give, true)
	if err != nil {
		t.Fatalf("elect: %v", err)
	}
	for lane, got := range elected {
		if want := lane == 1; got.b != want {
			t.Errorf("lane %d is elected=%v, want %v: lane 0 is not at this operation, so "+
				"it is not a candidate", lane, got.b, want)
		}
	}

	first, err := runPartial(t, 4, []uint32{1, 2, 3}, kernel.SubBroadcastFirstF32, give, true)
	if err != nil {
		t.Fatalf("broadcast first: %v", err)
	}
	for lane, got := range first {
		if got.f != 11 {
			t.Errorf("lane %d received %v, want lane 1's 11: an inactive lane holds no "+
				"value to broadcast", lane, got.f)
		}
	}
}

// Rule 2: Ballot reports 0 for an inactive lane, so Ballot(true) is not
// all-ones.
//
// The count is the *active* count, which is usually what was wanted and
// occasionally a bug — and a mask that set a bit for every lane of the subgroup
// would make the two indistinguishable.
func TestABallotReportsZeroForAnInactiveLane(t *testing.T) {
	give := func(lane uint32) (float32, bool, uint32) { return 0, true, 0 }
	got, err := runPartial(t, 4, []uint32{1, 2, 3}, kernel.SubBallot, give, true)
	if err != nil {
		t.Fatalf("ballot: %v", err)
	}
	for lane, v := range got {
		if v.m.Bit(0) {
			t.Errorf("lane %d sees lane 0's bit set, and lane 0 is not at this operation: "+
				"an inactive lane's predicate is not false, it is absent", lane)
		}
		if n := v.m.Count(); n != 3 {
			t.Errorf("lane %d counts %d set bits over three active lanes, all of which set "+
				"their predicate; Ballot(true).Count() is the active count", lane, n)
		}
		if l := v.m.LowestSet(); l != 1 {
			t.Errorf("lane %d reads LowestSet %d, want 1", lane, l)
		}
	}
}

// Rule 3: reading an inactive lane of this subgroup is undefined, and the
// oracle says so rather than handing back a number.
//
// It names both lanes because either one can be the mistake: the reading lane
// is where the code is, and the requested lane is what the code assumed was
// there.
func TestReadingAnInactiveLaneIsReported(t *testing.T) {
	// Lane 1 sits out and lane 0 asks for it.
	give := func(lane uint32) (float32, bool, uint32) {
		if lane == 0 {
			return 100, false, 1
		}
		return float32(lane) + 100, false, lane
	}
	_, err := runPartial(t, 4, []uint32{0, 2, 3}, kernel.SubShuffleF32, give, true)
	if err == nil {
		t.Fatal("reading a lane that is not at the operation is undefined and was not " +
			"reported: a plausible number propagating out of it is a wrong answer nothing " +
			"else would catch")
	}
	var ds kernel.Diagnostics
	if !errors.As(err, &ds) {
		t.Fatalf("want a diagnostic, got %T: %v", err, err)
	}
	if len(ds) != 1 {
		t.Fatalf("want one diagnostic, got %d: %v", len(ds), ds)
	}
	if ds[0].Kind != kernel.DiagUndefinedLane {
		t.Errorf("kind is %v, want the inactive-lane read", ds[0].Kind)
	}
	for _, want := range []string{"lane 0 read lane 1", "ShuffleF32", "undefined"} {
		if !strings.Contains(ds[0].Error(), want) {
			t.Errorf("the report should say %q, and says:\n%v", want, ds[0].Error())
		}
	}
}

// And with the instrumentation off the value is undefined rather than zero.
//
// Zero is the answer that would let a kernel relying on it look correct, and
// specs/002-compute-model.md section 5.2 rule 3 says it is not the answer: the
// oracle hands back a quiet NaN, which nobody mistakes for a result.
func TestAnUndefinedLaneReadIsNotZero(t *testing.T) {
	give := func(lane uint32) (float32, bool, uint32) {
		if lane == 0 {
			return 100, false, 1
		}
		return float32(lane) + 100, false, lane
	}
	got, err := runPartial(t, 4, []uint32{0, 2, 3}, kernel.SubShuffleF32, give, false)
	if err != nil {
		t.Fatalf("with diagnostics off the dispatch should run: %v", err)
	}
	if v := got[0].f; !math.IsNaN(float64(v)) {
		t.Errorf("lane 0 read undefined lane 1 and got %v, want a quiet NaN: zero is a "+
			"value a kernel could compute, so returning it makes the mistake survive", v)
	}
	// The lanes that read an active lane are unaffected, or the check above
	// would be measuring a broken combination step.
	if v := got[2].f; v != 102 {
		t.Errorf("lane 2 read itself and got %v, want 102", v)
	}
}

// A lane index outside the subgroup is undefined and is *not* reported.
//
// This is the one place the oracle stops short of rule 3, and it is deliberate.
// A shuffle up by one has lane 0 read lane -1 on every device, and a scan
// written with one discards that lane's answer rather than avoiding the call —
// which it cannot do, since a subgroup operation inside a conditional does not
// lower. Reporting it would refuse the kernel the operation exists for. What is
// still reported is a lane that is *there* and not taking part, which is the
// case the rule was written about. See specs/002-compute-model.md section 5.2.
func TestReadingOutsideTheSubgroupIsNotReported(t *testing.T) {
	give := func(lane uint32) (float32, bool, uint32) { return float32(lane) + 100, false, 1 }
	got, err := runPartial(t, 4, []uint32{0, 1, 2, 3}, kernel.SubShuffleUpF32, give, true)
	if err != nil {
		t.Fatalf("a shuffle up past the bottom of the subgroup should not be reported: %v", err)
	}
	if v := got[0].f; !math.IsNaN(float64(v)) {
		t.Errorf("lane 0 shuffled up past the subgroup and got %v, want a quiet NaN", v)
	}
	for lane := uint32(1); lane < 4; lane++ {
		if v, want := got[lane].f, float32(lane)+99; v != want {
			t.Errorf("lane %d read its predecessor and got %v, want %v", lane, v, want)
		}
	}
}

// Rule 5, over an active set that is a subset rather than a workgroup of one.
//
// The witness is negative zero, because for every finite v, 0 + v is exactly v:
// an accumulator seeded with zero passes any test using ordinary values. 0 +
// (-0) is +0, and a sign that flips changes the sign of a later division.
func TestAReductionOverOneActiveLaneKeepsItsSign(t *testing.T) {
	negZero := float32(math.Copysign(0, -1))
	give := func(lane uint32) (float32, bool, uint32) { return negZero, false, 0 }

	got, err := runPartial(t, 4, []uint32{2}, kernel.SubAddF32, give, true)
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if bits := math.Float32bits(got[2].f); bits != math.Float32bits(negZero) {
		t.Errorf("the one active lane read back 0x%08X, want 0x%08X: a reduction over an "+
			"active set of one returns that lane's value, not v plus an identity the "+
			"inactive lanes were assumed to hold", bits, math.Float32bits(negZero))
	}
}

// A broadcast's lane operand must be dynamically uniform, and a disagreement is
// reported rather than resolved.
//
// Resolving it would be the wrong answer to give: on hardware the winner is the
// device's choice, so a kernel whose output depends on it is already wrong, and
// an oracle that picked one would make one device's answer look correct.
func TestABroadcastLaneMustBeUniform(t *testing.T) {
	give := func(lane uint32) (float32, bool, uint32) {
		return float32(lane) + 100, false, lane % 2
	}
	_, err := runPartial(t, 4, []uint32{0, 1, 2, 3}, kernel.SubBroadcastF32, give, true)
	if err == nil {
		t.Fatal("two lanes asked a broadcast for different lanes and nothing was reported")
	}
	if !strings.Contains(err.Error(), "dynamically uniform") {
		t.Errorf("the report should say the lane must be dynamically uniform, and says:\n%v", err)
	}

	// And it is instrumentation, so it is off with the rest of it. Every check
	// in the combination step lives in the developer mode together: one that
	// stayed on would make the same kernel pass or fail on a switch it does not
	// name, which is worse than either answer.
	got, err := runPartial(t, 4, []uint32{0, 1, 2, 3}, kernel.SubBroadcastF32, give, false)
	if err != nil {
		t.Fatalf("with the instrumentation off the dispatch should run: %v", err)
	}
	if v := got[0].f; v != 100 {
		t.Errorf("lane 0 asked for lane 0 and read %v, want 100: with the check off, each "+
			"lane still reads the lane it named", v)
	}
}

// A shuffle moves bits, so it is compared as bits.
//
// Negative zero and a NaN with a payload are the witnesses: both compare equal
// to something else under ordinary arithmetic comparison — -0 == 0, and a NaN
// is equal to nothing at all — so an implementation that reconstructed the
// value instead of moving it would pass a test written with ordinary numbers.
func TestAShuffleMovesBits(t *testing.T) {
	payload := math.Float32frombits(0x7FC0BEEF)
	negZero := float32(math.Copysign(0, -1))
	values := []float32{negZero, payload, 3, -7}

	give := func(lane uint32) (float32, bool, uint32) {
		// Reverse the subgroup: lane i reads lane 3-i.
		return values[lane], false, 3 - lane
	}
	got, err := runPartial(t, 4, []uint32{0, 1, 2, 3}, kernel.SubShuffleF32, give, true)
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	for lane := uint32(0); lane < 4; lane++ {
		want := math.Float32bits(values[3-lane])
		if bits := math.Float32bits(got[lane].f); bits != want {
			t.Errorf("lane %d read lane %d and got bits 0x%08X, want 0x%08X",
				lane, 3-lane, bits, want)
		}
	}
}

// Rule 4: a scan skips an inactive lane rather than treating it as an identity
// element, and each lane's prefix is over the active lanes below it.
//
// The values are powers of two, so a sum names its terms: over active lanes
// {0, 2, 3} lane 3's exclusive scan is 5, which is lanes 0 and 2 and could be
// no other subset. That is what fixes *which* lanes each prefix covers.
//
// It is not, on its own, a witness for skip against identity-fill — see
// [TestAScanSkippingALaneIsVisibleOnlyInTheSign] for that, and for why the
// distinction needs a signed zero to be observable at all.
func TestAScanSkipsInactiveLanesRatherThanZeroingThem(t *testing.T) {
	// 1, 2, 4, 8 by lane, so a sum names its terms.
	give := func(lane uint32) (float32, bool, uint32) { return float32(int(1) << lane), false, 0 }
	active := []uint32{0, 2, 3}

	excl, err := runPartial(t, 4, active, kernel.SubExclusiveAddF32, give, true)
	if err != nil {
		t.Fatalf("exclusive: %v", err)
	}
	for _, c := range []struct {
		lane uint32
		want float32
		why  string
	}{
		{0, 0, "the lowest active lane sums nothing, and nothing is the identity"},
		{2, 1, "lane 0 only: lane 1 is not active, so it contributes no 2"},
		{3, 5, "lanes 0 and 2, which is 1+4 and not 1+2+4"},
	} {
		if got := excl[c.lane].f; got != c.want {
			t.Errorf("lane %d's exclusive scan is %v, want %v: %s", c.lane, got, c.want, c.why)
		}
	}

	incl, err := runPartial(t, 4, active, kernel.SubInclusiveAddF32, give, true)
	if err != nil {
		t.Fatalf("inclusive: %v", err)
	}
	for _, c := range []struct {
		lane uint32
		want float32
	}{{0, 1}, {2, 5}, {3, 13}} {
		if got := incl[c.lane].f; got != c.want {
			t.Errorf("lane %d's inclusive scan is %v, want %v", c.lane, got, c.want)
		}
	}
}

// The lowest active lane's exclusive scan is the identity, and the lowest
// active lane's inclusive scan is its own value.
//
// The same witness as the reduction's, read twice, because the two rows of
// specs/002-compute-model.md section 5.2 disagree on purpose: an inclusive scan
// over one lane is that lane's value, so a negative zero survives it, and an
// exclusive scan over one lane is the sum of nothing, which is +0 whatever that
// lane holds. An implementation that seeded both with the same accumulator gets
// one of them wrong.
func TestTheEndsOfAScanDifferOnNegativeZero(t *testing.T) {
	negZero := float32(math.Copysign(0, -1))
	give := func(lane uint32) (float32, bool, uint32) { return negZero, false, 0 }

	incl, err := runPartial(t, 4, []uint32{1, 2}, kernel.SubInclusiveAddF32, give, true)
	if err != nil {
		t.Fatalf("inclusive: %v", err)
	}
	if bits := math.Float32bits(incl[1].f); bits != math.Float32bits(negZero) {
		t.Errorf("the lowest active lane's inclusive scan is 0x%08X, want 0x%08X: it is a "+
			"prefix of one element, which is that lane's value and not zero plus it",
			bits, math.Float32bits(negZero))
	}

	excl, err := runPartial(t, 4, []uint32{1, 2}, kernel.SubExclusiveAddF32, give, true)
	if err != nil {
		t.Fatalf("exclusive: %v", err)
	}
	if bits := math.Float32bits(excl[1].f); bits != 0 {
		t.Errorf("the lowest active lane's exclusive scan is 0x%08X, want +0: its prefix is "+
			"empty, so it receives the identity rather than anything it holds", bits)
	}
}

// And the difference between skipping a lane and adding an identity for it is
// visible only in the sign of a zero.
//
// This is worth stating rather than assuming, because it bounds what the rule
// above can prove: `x + 0` is exactly `x` for every finite x, so an
// implementation that summed an identity element for each inactive lane would
// produce the same number as one that skipped it — on every input but one.
// `-0 + 0` is `+0`. So the witness is a prefix whose whole content is a
// negative zero, and the mistake it catches is a sign that flips, which changes
// the sign of a later division and nothing else about the answer.
func TestAScanSkippingALaneIsVisibleOnlyInTheSign(t *testing.T) {
	negZero := float32(math.Copysign(0, -1))
	give := func(lane uint32) (float32, bool, uint32) {
		if lane == 0 {
			return negZero, false, 0
		}
		return 1, false, 0
	}
	// Lane 1 sits out, so lane 2's exclusive prefix is lane 0 alone.
	got, err := runPartial(t, 4, []uint32{0, 2}, kernel.SubExclusiveAddF32, give, true)
	if err != nil {
		t.Fatalf("exclusive: %v", err)
	}
	if bits := math.Float32bits(got[2].f); bits != math.Float32bits(negZero) {
		t.Errorf("lane 2's exclusive scan is 0x%08X, want 0x%08X: its prefix is lane 0 "+
			"alone, and adding an identity element for the lane that is not there turns "+
			"that lane's negative zero positive", bits, math.Float32bits(negZero))
	}
}
