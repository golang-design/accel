// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import "golang.design/x/accel"

// BallotDims is the predicate's threshold.
type BallotDims struct {
	// Below is the lane index under which the predicate is true, so a caller
	// can vary the active set without recompiling.
	Below uint32
}

// Ballot writes each of the mask's five methods, per lane.
//
// specs/058-ballot.md §5. Each lane votes `lane < Below` and then asks the mask
// the five questions 002 §5.2 says it must answer, so a method that returned a
// plausible number rather than the right one disagrees with a reference
// computed from the same predicate.
//
// # Why every lane writes every answer
//
// Four of the five are subgroup-uniform: the whole subgroup gets one mask, so
// Count, LowestSet and Any are the same for every lane of it. Writing them per
// lane is what says the *broadcast* happened -- a lowering that gave each lane
// its own single-bit mask would produce a Count of 1 everywhere, which a
// one-lane-writes test could not tell from a correct Count of 1 at Below=1.
//
//accel:kernel workgroup=64
//accel:requires subgroup_ballot, subgroup_basic
func Ballot(t accel.Thread, d BallotDims, count []int32, lowest []uint32,
	lower []int32, bit []uint32, any []uint32) {

	lane := t.SubgroupLane()
	m := t.SubgroupBallot(lane < d.Below)
	i := t.LocalID().X

	count[i] = int32(m.Count())
	lowest[i] = m.LowestSet()
	lower[i] = int32(m.CountLower(lane))

	// Bit and Any are bools, and the dtype set stores them as u32 here so one
	// binding shape covers all five answers.
	bit[i] = 0
	if m.Bit(lane) {
		bit[i] = 1
	}
	any[i] = 0
	if m.Any() {
		any[i] = 1
	}
}
