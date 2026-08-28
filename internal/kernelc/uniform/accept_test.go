// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package uniform_test

import (
	"fmt"
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/front"
	"golang.design/x/accel/internal/kernelc/uniform"
)

// accept compiles a kernel body containing a barrier and reports the
// rejections. The body is wrapped the same way the level tests wrap theirs.
func accept(t *testing.T, body string) []uniform.Rejection {
	t.Helper()
	src := "package k\n\nimport \"golang.design/x/accel\"\n\n" +
		"//accel:kernel workgroup=64\n" +
		"func K(t accel.Thread, in []float32, out []float32) {\n" +
		"\tseed := in[t.GlobalID().X]\n\tout[t.GlobalID().X] = seed\n" +
		body + "\n}\n"

	pkg := checkSource(t, src)
	if pkg == nil {
		t.Fatalf("the source did not type-check:\n%s", src)
	}
	fns, diags := front.Check(pkg)
	if len(diags) > 0 {
		t.Fatalf("the front end rejected it: %v\n%s", diags, src)
	}
	return uniform.AcceptBarriers(fns[0])
}

// A barrier in straight-line code, and one under a uniform predicate, are
// accepted. Without these the rejections below would be passing against a rule
// that refuses every barrier.
func TestBarriersInUniformControlFlowAreAccepted(t *testing.T) {
	cases := []struct {
		name string
		body string
	}{
		{"straight-line", "\tt.Barrier()"},
		{"under a uniform predicate", `
	if t.GroupIndex() < 4 {
		t.Barrier()
	}`},
		{"in a loop with a literal trip count", `
	for i := uint32(0); i < 8; i++ {
		t.Barrier()
	}`},
		{"in a loop bounded by a uniform value", `
	for i := uint32(0); i < t.GroupIndex(); i++ {
		t.Barrier()
	}`},
		{"after a uniformly controlled break", `
	for i := uint32(0); i < 8; i++ {
		if t.GroupIndex() > 2 {
			break
		}
		t.Barrier()
	}`},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := accept(t, c.body); len(got) != 0 {
				t.Errorf("rejected a uniform barrier: %v", got)
			}
		})
	}
}

// A subgroup barrier's scope is one subgroup, so subgroup-uniform control
// around it is legal where the same control around a workgroup barrier is not.
//
// specs/002-compute-model.md §12 names this exact pair as a test: "a
// SubgroupBarrier controlled by SubgroupID is accepted while the same control
// around a workgroup barrier is rejected. A predicate involving
// SubgroupInvocationID rejects both."
//
// The three rows are one body each, so the difference between them is only
// which barrier or which predicate -- which is what makes the comparison
// evidence rather than three unrelated results.
func TestASubgroupBarrierIsCheckedAtSubgroupScope(t *testing.T) {
	const underSubgroupIndex = `
	if t.SubgroupIndex() < 2 {
		%s
	}`
	const underLane = `
	if t.SubgroupLane() < 2 {
		%s
	}`

	for _, c := range []struct {
		name    string
		body    string
		reject  bool
		mention string
	}{
		// The pair 002 §12 names. Same predicate, different barrier.
		{
			name: "subgroup barrier under SubgroupIndex",
			body: fmt.Sprintf(underSubgroupIndex, "t.SubgroupBarrier()"),
		},
		{
			name:    "workgroup barrier under SubgroupIndex",
			body:    fmt.Sprintf(underSubgroupIndex, "t.Barrier()"),
			reject:  true,
			mention: "workgroup-uniform",
		},
		// A per-lane predicate is below subgroup scope, so it rejects both.
		{
			name:    "subgroup barrier under SubgroupLane",
			body:    fmt.Sprintf(underLane, "t.SubgroupBarrier()"),
			reject:  true,
			mention: "subgroup-uniform",
		},
		{
			name:    "workgroup barrier under SubgroupLane",
			body:    fmt.Sprintf(underLane, "t.Barrier()"),
			reject:  true,
			mention: "workgroup-uniform",
		},
		// Straight-line, so the accepted row above is not passing against a
		// rule that accepts every subgroup barrier for the wrong reason.
		{
			name: "subgroup barrier in straight-line code",
			body: "\tt.SubgroupBarrier()",
		},

		// Clause 3 at subgroup scope: a uniform loop that some *lanes* leave
		// early. The trip count is uniform and the break is not, so the lanes
		// that took it never arrive -- the same reasoning as the workgroup
		// clause, one level down the lattice.
		{
			name: "subgroup barrier after a per-lane break",
			body: `
	for i := uint32(0); i < 8; i++ {
		if t.SubgroupLane() > 2 {
			break
		}
		t.SubgroupBarrier()
	}`,
			reject:  true,
			mention: "never arrive",
		},
		// And the same escape at subgroup scope is *not* an escape for a
		// subgroup barrier: every lane of a subgroup takes the break together.
		// This is the row that says the clause reads the escape's level rather
		// than rejecting every escape.
		{
			name: "subgroup barrier after a per-subgroup break",
			body: `
	for i := uint32(0); i < 8; i++ {
		if t.SubgroupIndex() > 2 {
			break
		}
		t.SubgroupBarrier()
	}`,
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := accept(t, c.body)
			switch {
			case c.reject && len(got) == 0:
				t.Fatalf("accepted, and this shape must be rejected:\n%s", c.body)
			case !c.reject && len(got) != 0:
				t.Fatalf("rejected a legal shape: %v\n%s", got, c.body)
			case !c.reject:
				return
			}
			if !strings.Contains(got[0].Msg, c.mention) {
				t.Errorf("the message should say %q, got: %v", c.mention, got[0].Msg)
			}
		})
	}
}

// A barrier under a mask method is rejected, because a mask is not uniform.
//
// specs/058-ballot.md. The five methods lower to ordinary calls rather than
// rendezvous, so `IsSubgroupRendezvous` does not reach them, and `Count`,
// `LowestSet` and `Any` take **no arguments** -- the join over zero operands is
// workgroup-uniform, so a barrier under `if m.Count() > 1` was accepted. It is
// not: a ballot's result is subgroup-uniform at best, and different subgroups
// can take different branches.
//
// Two of the five were rejected before the fix and for the wrong reason: `Bit`
// and `CountLower` take a lane operand, and the test passes `SubgroupLane`,
// which is non-uniform on its own. So a test written with only those two would
// have reported the analysis correct. All five are here, and the three nullary
// ones are the ones that matter.
func TestABarrierUnderAMaskMethodIsRejected(t *testing.T) {
	const ballot = "\n\tm := t.SubgroupBallot(t.SubgroupLane() < 3)"
	for _, c := range []struct{ name, cond string }{
		// No arguments: nothing but the receiver can carry the level.
		{"Count", "m.Count() > 1"},
		{"Any", "m.Any()"},
		{"LowestSet", "m.LowestSet() > 1"},
		// A lane operand as well, so these would reject either way -- kept so
		// the set is the whole method table rather than the interesting half.
		{"Bit", "m.Bit(t.SubgroupLane())"},
		{"CountLower", "m.CountLower(t.SubgroupLane()) > 1"},
	} {
		t.Run(c.name, func(t *testing.T) {
			body := ballot + "\n\tif " + c.cond + " {\n\t\tt.Barrier()\n\t}"
			got := accept(t, body)
			if len(got) == 0 {
				t.Fatalf("a workgroup barrier under %s was accepted, and a mask is "+
					"subgroup-uniform at best:\n%s", c.cond, body)
			}
			if !strings.Contains(got[0].Msg, "workgroup-uniform") {
				t.Errorf("the message should say workgroup-uniform, got: %v", got[0].Msg)
			}
		})
	}
}

// The three clauses of spec 002 section 3.3's rule, each with a case that fails
// only it.
func TestBarriersInNonUniformControlFlowAreRejected(t *testing.T) {
	cases := []struct {
		name   string
		clause string
		body   string
		says   string
	}{{
		name:   "under a non-uniform predicate",
		clause: "1: every predicate in the control dependence set is uniform",
		body: `
	if t.LocalIndex() < 4 {
		t.Barrier()
	}`,
		says: "reached under non-uniform control",
	}, {
		name:   "in a loop with a non-uniform trip count",
		clause: "2: every enclosing loop's trip count is uniform",
		body: `
	for i := uint32(0); i < t.LocalIndex(); i++ {
		t.Barrier()
	}`,
		says: "reached under non-uniform control",
	}, {
		name:   "after a non-uniformly controlled break",
		clause: "3: no break or continue under a less uniform predicate",
		body: `
	for i := uint32(0); i < 8; i++ {
		if t.LocalIndex() > 2 {
			break
		}
		t.Barrier()
	}`,
		says: "a break under non-uniform control can skip this one",
	}, {
		name:   "after a non-uniformly controlled continue",
		clause: "3, in its other form",
		body: `
	for i := uint32(0); i < 8; i++ {
		if t.LocalIndex() > 2 {
			continue
		}
		t.Barrier()
	}`,
		says: "a continue under non-uniform control can skip this one",
	}, {
		name:   "nested inside a non-uniform predicate",
		clause: "1, through two levels",
		body: `
	if t.GroupIndex() < 4 {
		if t.LocalIndex() < 2 {
			t.Barrier()
		}
	}`,
		says: "reached under non-uniform control",
	}, {
		name:   "under a predicate derived from a load",
		clause: "1, where the predicate reads memory another invocation may have written",
		body: `
	if in[0] > 0 {
		t.Barrier()
	}`,
		says: "reached under non-uniform control",
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := accept(t, c.body)
			if len(got) != 1 {
				t.Fatalf("clause %s: got %d rejections, want 1: %v", c.clause, len(got), got)
			}
			if !strings.Contains(got[0].Error(), c.says) {
				t.Errorf("the message should say %q, got:\n%v", c.says, got[0])
			}
			if got[0].Pos == 0 {
				t.Error("a rejection must carry the barrier's position")
			}
			if got[0].Because == 0 {
				t.Error("a rejection must carry the position of what made the flow non-uniform")
			}
		})
	}
}

// Clause 3 is the one an analysis checking only predicates and trip counts
// misses, so it gets its own statement: the loop's trip count is a literal and
// every predicate enclosing the barrier is uniform, and the barrier is still
// unreachable for some invocations.
func TestAUniformLoopWithANonUniformEscapeIsStillRejected(t *testing.T) {
	got := accept(t, `
	for i := uint32(0); i < 8; i++ {
		if t.LocalIndex() > 2 {
			break
		}
		t.Barrier()
	}`)
	if len(got) != 1 {
		t.Fatalf("got %d rejections, want 1: the loop's trip count is uniform and the "+
			"barrier's own predicate is uniform, so an analysis checking only those "+
			"admits this barrier -- and the invocations that broke never arrive", len(got))
	}
}

// A break in one loop does not make a barrier in a later sibling loop
// unreachable: it left before that loop began.
func TestAnEscapeDoesNotEscapeItsOwnLoop(t *testing.T) {
	got := accept(t, `
	for i := uint32(0); i < 8; i++ {
		if t.LocalIndex() > 2 {
			break
		}
	}
	for i := uint32(0); i < 8; i++ {
		t.Barrier()
	}`)
	if len(got) != 0 {
		t.Errorf("a break in an earlier loop should not reject a barrier in a later "+
			"one: %v", got)
	}
}

// The two false rejections spec 002 section 3.3 names as the cost of rejecting
// when it cannot decide. They are asserted as rejections so the price is
// visible in the test suite rather than described in prose, and so a later
// escape hatch has something to change.
func TestTheKnownFalseRejections(t *testing.T) {
	t.Run("a predicate uniform by construction through a load", func(t *testing.T) {
		// Every invocation writes the same value, so the predicate is uniform in
		// fact. The analysis cannot see it, because a load is non-uniform.
		got := accept(t, `
	if in[t.GroupIndex()] > 0 {
		t.Barrier()
	}`)
		if len(got) != 1 {
			t.Fatalf("this is a known false rejection and should still be rejected; "+
				"if it now passes, spec 002 section 3.3's cost paragraph is out of date: %v", got)
		}
	})

	t.Run("a loop bound read from a binding", func(t *testing.T) {
		got := accept(t, `
	for i := float32(0); i < in[0]; i++ {
		t.Barrier()
	}`)
		if len(got) != 1 {
			t.Fatalf("this is a known false rejection and should still be rejected: %v", got)
		}
	})
}

func TestARejectionCarriesItsMessage(t *testing.T) {
	got := accept(t, `
	if t.LocalIndex() < 4 {
		t.Barrier()
	}`)
	if len(got) != 1 {
		t.Fatalf("got %d rejections, want 1", len(got))
	}
	if got[0].Msg != got[0].Error() {
		t.Error("Error should report Msg")
	}
	if !strings.Contains(got[0].Error(), "specs/002-compute-model.md") {
		t.Error("a rejection should name the rule it enforces, so a reader can look it up")
	}
}
