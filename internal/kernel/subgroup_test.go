// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel_test

import (
	"math"
	"testing"

	"golang.design/x/accel/internal/kernel"
)

// A reduction over an active set of one returns that lane's value, not v + 0.
//
// Spec 002 section 5.2's rule 5, and it needs the right witness: for every
// finite v, 0 + v is exactly v, so an implementation seeding its accumulator
// with zero passes any test using ordinary values. Negative zero is where the
// two differ — 0 + (-0) is +0 — and a sign that flips changes the sign of a
// later division, which is the kind of difference nobody looks for.
func TestAReductionOverOneLaneReturnsThatLanesValue(t *testing.T) {
	negZero := float32(math.Copysign(0, -1))

	k := &kernel.Kernel{
		Name: "One", WorkgroupSize: kernel.ID3{X: 1, Y: 1, Z: 1},
		Generator: kernel.ABIVersion, Suspensions: 1,
		Bindings: []kernel.Binding{{Name: "out", DType: kernel.F32, Access: kernel.Write}},
		Cooperative: func(th kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			out := a.Slices[0].([]float32)
			switch f.Pass {
			case 0:
				f.Pass = 1
				f.Sub = kernel.SubAddF32
				f.SubF32 = negZero
				f.Barrier = kernel.BarrierID{Index: 0}
				return true
			default:
				out[0] = f.SubF32
				return false
			}
		},
	}
	out := make([]float32, 1)
	if err := kernel.DispatchCooperative(k, kernel.ID3{X: 1},
		kernel.Args{Slices: []any{out}}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	if got := math.Float32bits(out[0]); got != math.Float32bits(negZero) {
		t.Errorf("a reduction over one lane holding -0 gave bits 0x%08X, want 0x%08X: "+
			"seeding the accumulator with zero turns -0 into +0, and a sign that flips "+
			"changes the sign of a later division",
			got, math.Float32bits(negZero))
	}
}

// The mask reports what each lane set, and its accessors agree with each other.
func TestABallotMaskReportsEachLane(t *testing.T) {
	k := &kernel.Kernel{
		Name: "Ballot", WorkgroupSize: kernel.ID3{X: 8, Y: 1, Z: 1},
		Generator: kernel.ABIVersion, Suspensions: 1,
		Bindings: []kernel.Binding{{Name: "out", DType: kernel.U32, Access: kernel.Write}},
		Cooperative: func(th kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			out := a.Slices[0].([]uint32)
			switch f.Pass {
			case 0:
				f.Pass = 1
				f.Sub = kernel.SubBallot
				// Odd lanes set their predicate.
				f.SubBool = th.SubgroupLane()%2 == 1
				f.Barrier = kernel.BarrierID{Index: 0}
				return true
			default:
				lane := th.SubgroupLane()
				out[lane] = uint32(f.SubMask.Count())
				if f.SubMask.Bit(lane) != (lane%2 == 1) {
					t.Errorf("lane %d's own bit is %v", lane, f.SubMask.Bit(lane))
				}
				if lane == 0 {
					if got := f.SubMask.LowestSet(); got != 1 {
						t.Errorf("LowestSet is %d, want 1", got)
					}
					if got := f.SubMask.CountLower(5); got != 2 {
						t.Errorf("CountLower(5) is %d, want 2", got)
					}
					if !f.SubMask.Any() {
						t.Error("Any is false for a mask with bits set")
					}
				}
				return false
			}
		},
	}
	out := make([]uint32, 8)
	if err := kernel.DispatchCooperative(k, kernel.ID3{X: 1},
		kernel.Args{Slices: []any{out}}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	for lane, got := range out {
		if got != 4 {
			t.Errorf("lane %d saw %d set bits, want 4", lane, got)
		}
	}
}

// An inactive lane's bit is zero, so Ballot(true) is not all ones.
//
// specs/002-compute-model.md §5.1's rule 2, which that section says is the one
// everyone gets wrong: a lane that does not reach the ballot contributes
// nothing -- it is not zero, not the identity, not present -- so
// `Ballot(true).Count()` is the *active* count and not the subgroup size.
//
// # Why this is here and not in the corpus
//
// The only way to make a lane inactive is to put the ballot inside a
// conditional, and specs/018-cooperative-lowering.md's split cannot resume
// inside a branch, so that kernel is refused by the compiler
// (specs/058-ballot.md §6.4). This drives the scheduler directly, which is the
// same seam the mask's other rules are checked through above, and it is the
// only place the rule has a witness at all today.
//
// The predicate is a constant true so the question is what an *absent* lane
// contributes rather than what a false vote does: every lane that reaches the
// ballot sets its bit, so Count is exactly how many reached it.
func TestAnInactiveLaneContributesNoBit(t *testing.T) {
	const width, voting = 8, 3
	k := &kernel.Kernel{
		Name: "BallotInactive", WorkgroupSize: kernel.ID3{X: width, Y: 1, Z: 1},
		Generator: kernel.ABIVersion, Suspensions: 1,
		Bindings: []kernel.Binding{{Name: "out", DType: kernel.U32, Access: kernel.Write}},
		Cooperative: func(th kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			out := a.Slices[0].([]uint32)
			lane := th.SubgroupLane()
			switch f.Pass {
			case 0:
				f.Pass = 1
				// The lanes above the threshold never suspend at the ballot,
				// which is exactly what an inactive lane is. The sentinel says
				// they took this path rather than voting false.
				if lane >= voting {
					out[lane] = 999
					return false
				}
				f.Sub = kernel.SubBallot
				f.SubBool = true
				f.Barrier = kernel.BarrierID{Index: 0}
				return true
			default:
				out[lane] = uint32(f.SubMask.Count())
				// And the absent lanes' bits are clear, read from a voting
				// lane's copy of the mask.
				for l := uint32(voting); l < width; l++ {
					if f.SubMask.Bit(l) {
						t.Errorf("lane %d did not reach the ballot and its bit is set", l)
					}
				}
				return false
			}
		},
	}
	out := make([]uint32, width)
	if err := kernel.DispatchCooperative(k, kernel.ID3{X: 1},
		kernel.Args{Slices: []any{out}}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	for lane := range voting {
		if out[lane] != voting {
			t.Errorf("lane %d saw %d set bits and %d lanes voted: an inactive lane was "+
				"counted", lane, out[lane], voting)
		}
	}
	for lane := voting; lane < width; lane++ {
		if out[lane] != 999 {
			t.Errorf("lane %d wrote %d, and it should not have reached the ballot",
				lane, out[lane])
		}
	}
}

// An empty mask reports nothing set, and LowestSet says so rather than naming
// lane zero.
func TestAnEmptyMaskNamesNoLane(t *testing.T) {
	var m kernel.Mask
	if m.Any() {
		t.Error("an empty mask reports Any")
	}
	if m.Count() != 0 {
		t.Errorf("an empty mask counts %d", m.Count())
	}
	if got := m.LowestSet(); got != 128 {
		t.Errorf("LowestSet on an empty mask is %d, want the width: naming lane 0 "+
			"would be indistinguishable from lane 0 being set", got)
	}
	if m.Bit(0) || m.Bit(200) {
		t.Error("an empty mask has a bit set")
	}
	if m.String() == "" {
		t.Error("a mask describes itself")
	}
}

// Min, max, broadcast, any and all, each over a subgroup.
func TestTheRestOfTheSubgroupOperations(t *testing.T) {
	cases := []struct {
		name  string
		op    kernel.SubgroupOp
		give  func(lane uint32) (float32, bool)
		check func(t *testing.T, lane uint32, f float32, b bool)
	}{{
		name: "min",
		op:   kernel.SubMinF32,
		give: func(lane uint32) (float32, bool) { return float32(8 - lane), false },
		check: func(t *testing.T, lane uint32, f float32, _ bool) {
			if f != 1 {
				t.Errorf("lane %d got min %v, want 1", lane, f)
			}
		},
	}, {
		name: "max",
		op:   kernel.SubMaxF32,
		give: func(lane uint32) (float32, bool) { return float32(lane), false },
		check: func(t *testing.T, lane uint32, f float32, _ bool) {
			if f != 7 {
				t.Errorf("lane %d got max %v, want 7", lane, f)
			}
		},
	}, {
		name: "broadcast first",
		op:   kernel.SubBroadcastFirstF32,
		give: func(lane uint32) (float32, bool) { return float32(lane) + 10, false },
		check: func(t *testing.T, lane uint32, f float32, _ bool) {
			if f != 10 {
				t.Errorf("lane %d got %v, want the lowest lane's 10", lane, f)
			}
		},
	}, {
		name: "elect",
		op:   kernel.SubElect,
		give: func(lane uint32) (float32, bool) { return 0, false },
		check: func(t *testing.T, lane uint32, _ float32, b bool) {
			if b != (lane == 0) {
				t.Errorf("lane %d is elected=%v, want %v: accel pins the choice to the "+
					"lowest lane, since hardware guarantees only that one is chosen",
					lane, b, lane == 0)
			}
		},
	}, {
		name: "any",
		op:   kernel.SubAny,
		give: func(lane uint32) (float32, bool) { return 0, lane == 3 },
		check: func(t *testing.T, lane uint32, _ float32, b bool) {
			if !b {
				t.Errorf("lane %d got Any=false with one lane's predicate true", lane)
			}
		},
	}, {
		name: "all",
		op:   kernel.SubAll,
		give: func(lane uint32) (float32, bool) { return 0, lane != 3 },
		check: func(t *testing.T, lane uint32, _ float32, b bool) {
			if b {
				t.Errorf("lane %d got All=true with one lane's predicate false", lane)
			}
		},
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			k := &kernel.Kernel{
				Name: "Op", WorkgroupSize: kernel.ID3{X: 8, Y: 1, Z: 1},
				Generator: kernel.ABIVersion, Suspensions: 1,
				Cooperative: func(th kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
					lane := th.SubgroupLane()
					if f.Pass == 0 {
						f.Pass = 1
						f.Sub = c.op
						f.SubF32, f.SubBool = c.give(lane)
						f.Barrier = kernel.BarrierID{Index: 0}
						return true
					}
					c.check(t, lane, f.SubF32, f.SubBool)
					return false
				},
			}
			if err := kernel.DispatchCooperative(k, kernel.ID3{X: 1}, kernel.Args{}); err != nil {
				t.Fatalf("dispatch: %v", err)
			}
		})
	}
}

func TestSubgroupOpsName(t *testing.T) {
	for _, c := range []struct {
		op   kernel.SubgroupOp
		want string
	}{
		{kernel.SubNone, "barrier"},
		{kernel.SubAddF32, "SubgroupAddF32"},
		{kernel.SubMinF32, "SubgroupMinF32"},
		{kernel.SubMaxF32, "SubgroupMaxF32"},
		{kernel.SubBroadcastFirstF32, "BroadcastFirstF32"},
		{kernel.SubElect, "Elect"},
		{kernel.SubAny, "Any"},
		{kernel.SubAll, "All"},
		{kernel.SubBallot, "Ballot"},
		{kernel.SubgroupOp(99), "barrier"},
	} {
		if got := c.op.String(); got != c.want {
			t.Errorf("got %q, want %q", got, c.want)
		}
	}
}
