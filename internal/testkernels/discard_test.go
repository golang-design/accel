// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import (
	"testing"

	"golang.design/x/accel"
)

// The generated discarding lowering agrees with its authored source, in both
// what it returns and whether it discarded.
//
// A fresh Fragment per call, because a discard is recorded in the receiver and
// one reused across a sweep would stay set after the first column that
// discards -- which would make every later column agree for the wrong reason.
func TestTheGeneratedDiscardingStageAgreesWithItsSource(t *testing.T) {
	var discarded, kept int
	for x := range 8 {
		coord := accel.Vec4{float32(x) + 0.5, 0.5, 0.25, 1}

		af := accel.NewFragmentForTest(coord, true)
		want := DiscardFS(af, accel.NoVaryings{})

		gf := accel.NewFragmentForTest(coord, true)
		got := discardFSFlat(gf, accel.NoVaryings{})

		if got != want {
			t.Errorf("column %d: generated %+v, authored %+v", x, got, want)
		}
		if gf.Discarded() != af.Discarded() {
			t.Errorf("column %d: generated discarded=%t, authored discarded=%t",
				x, gf.Discarded(), af.Discarded())
		}
		if af.Discarded() {
			discarded++
		} else {
			kept++
		}
	}
	// Both halves have to occur, or the comparison above is two constants
	// agreeing: a stage that discarded every column and one that discarded none
	// would each pass it against itself.
	if discarded == 0 || kept == 0 {
		t.Fatalf("%d columns discarded and %d kept; the sweep must cover both", discarded, kept)
	}
}

// The stage record carries the discard, which is what stops a backend
// promising an early depth test the stage cannot have.
//
// specs/032-stage-abi.md section 4.2 and section 6. Asserted against a stage
// that does not discard as well, because a field that is always true carries no
// more information than one that is always false.
func TestAStageRecordsWhetherItDiscards(t *testing.T) {
	if !DiscardFSStage.Discards {
		t.Error("DiscardFS reaches Discard and its record says it does not")
	}
	if SolidFSStage.Discards {
		t.Error("SolidFS reaches no Discard and its record says it does")
	}
}
