// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package emit

import (
	"testing"

	"golang.design/x/accel/internal/kernelc/intrin"
	"golang.design/x/accel/internal/kernelc/ir"
)

// Every subgroup rendezvous is registered everywhere it has to be.
//
// # Why one walk instead of four checklists
//
// A rendezvous opcode is spread across four tables: the intrinsic that makes it
// callable, the carrier that says what travels in the frame, the runtime
// constant that says what to combine, and the Metal spelling. Missing one is
// not a compile error in any of them. Missing the carrier or the name used to
// mean a suspension that combined nothing and resumed reading the lane's own
// contribution — a plausible number, which is the failure
// specs/020-cooperative-atomics.md section 6.1 records three of.
//
// So the set is walked rather than listed. An opcode added to the IR is covered
// by this test the moment it exists, which a list of names is not.
func TestEverySubgroupRendezvousIsRegistered(t *testing.T) {
	// Refused on purpose, with the reason. Metal's simd_ballot returns a
	// simd_vote rather than an integer and the conversion is family-dependent,
	// so specs/022-msl-target.md section 5 refuses the operation by name rather
	// than lowering it to something else.
	refusedByMSL := map[ir.Opcode]string{
		ir.OpBallot: "simd_ballot returns a simd_vote (specs/022-msl-target.md section 5)",
	}

	found, unreachable := 0, 0
	for op := ir.Opcode(0); op < 256; op++ {
		if !op.IsSubgroupRendezvous() {
			continue
		}
		found++
		// An opcode no intrinsic lowers to is one nothing can emit, so the
		// tables below cannot be reached and their absence is not a hole. The
		// exemption is keyed on the table rather than on a name, so adding the
		// intrinsic is what turns the rest of this on.
		if !intrin.HasOp(op) {
			unreachable++
			if op != ir.OpBallot {
				t.Errorf("%v is a rendezvous no intrinsic reaches, so a kernel author "+
					"cannot call it: either add the intrinsic or take the opcode out", op)
			}
			continue
		}
		t.Run(op.String(), func(t *testing.T) {
			if _, ok := subOpName(op); !ok {
				t.Errorf("no runtime constant names it, so the generated lowering would "+
					"suspend as an ordinary barrier and combine nothing (%v)", op)
			}
			if subCarrier(op) == carrierNone {
				t.Errorf("no carrier says what travels in the frame, so its result would "+
					"be whatever the frame last held (%v)", op)
			}
			_, hasMSL := mslSubgroup[op]
			if !hasMSL {
				_, hasMSL = mslLaneRead[op]
			}
			if !hasMSL {
				_, hasMSL = mslSubgroupNullary[op]
			}
			if !hasMSL && refusedByMSL[op] == "" {
				t.Errorf("Metal neither spells it nor refuses it by name: add the spelling, "+
					"or add it to this test's refusal list with the reason (%v)", op)
			}
			if hasMSL && refusedByMSL[op] != "" {
				t.Errorf("it is both spelled and listed as refused, so one of the two is "+
					"stale (%v)", op)
			}
		})
	}
	if found == 0 {
		t.Fatal("no rendezvous opcodes were walked, so this gate checks nothing: the " +
			"IsSubgroupRendezvous range has moved")
	}
	// Ballot is the one unreachable rendezvous, and it is listed here so that a
	// second one is noticed rather than silently exempted along with it.
	if unreachable != 1 {
		t.Errorf("%d rendezvous opcodes are unreachable from a kernel, want the one "+
			"(Ballot): each is an operation the IR names and nobody can call", unreachable)
	}
}

// The lane-addressed reads carry their lane operand, and the reductions do not.
//
// The two families share a field in the frame, so a reduction that started
// writing SubLane would be writing a number the scheduler reads for a different
// operation.
func TestOnlyLaneReadsCarryALane(t *testing.T) {
	for op := ir.Opcode(0); op < 256; op++ {
		if !op.IsSubgroupRendezvous() {
			continue
		}
		if got, want := subCarrier(op) == carrierF32Lane, op.IsSubgroupLaneRead(); got != want {
			t.Errorf("%v carries a lane operand: %v, and is a lane read: %v", op, got, want)
		}
	}
}
