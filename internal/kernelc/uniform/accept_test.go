// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package uniform_test

import (
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
