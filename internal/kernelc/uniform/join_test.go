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

// acceptWith is [accept] with package-level declarations in front of the
// kernel, for the cases that need a helper.
func acceptWith(t *testing.T, decls, body string) []uniform.Rejection {
	t.Helper()
	src := "package k\n\nimport \"golang.design/x/accel\"\n\n" + decls + "\n\n" +
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

// A return under a non-uniform predicate is an escape from the whole
// function: the invocations that took it never arrive at any barrier after it.
//
// It is the third clause's third form. A break leaves a loop and a continue
// leaves an iteration, and both are scoped to the loop that encloses them; a
// return leaves everything, so it is not reset at a loop boundary and it
// applies to a barrier at the top level as much as to one inside a loop.
func TestABarrierAfterANonUniformReturnIsRejected(t *testing.T) {
	for _, c := range []struct {
		name string
		body string
	}{
		{"at the top level", `
	if t.LocalIndex() > 2 {
		return
	}
	t.Barrier()`},
		{"before a loop holding the barrier", `
	if t.LocalIndex() > 2 {
		return
	}
	for i := uint32(0); i < 8; i++ {
		t.Barrier()
	}`},
		{"inside a loop, before a barrier after the loop", `
	for i := uint32(0); i < 8; i++ {
		if t.LocalIndex() > i {
			return
		}
	}
	t.Barrier()`},
		{"under a predicate derived from a load", `
	if in[0] > 0 {
		return
	}
	t.Barrier()`},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := accept(t, c.body)
			if len(got) != 1 {
				t.Fatalf("got %d rejections, want 1: the invocations that returned "+
					"never arrive at the barrier:\n%s\n%v", len(got), c.body, got)
			}
			if !strings.Contains(got[0].Msg, "a return under non-uniform control") {
				t.Errorf("the message should name the return, got: %v", got[0].Msg)
			}
			if got[0].Because == 0 || got[0].Because == got[0].Pos {
				t.Error("a rejection must point at the return that made arrival non-uniform")
			}
		})
	}

	// The accepting half: a return every invocation takes together is not an
	// escape, and one after the barrier cannot skip it.
	for _, c := range []struct {
		name string
		body string
	}{
		{"under a uniform predicate", `
	if t.GroupIndex() > 2 {
		return
	}
	t.Barrier()`},
		{"after the barrier", `
	t.Barrier()
	if t.LocalIndex() > 2 {
		return
	}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := accept(t, c.body); len(got) != 0 {
				t.Errorf("rejected a uniform barrier: %v", got)
			}
		})
	}
}

// A helper's result is as uniform as what its body computes, not only as its
// arguments. A helper that reads a per-invocation source -- an id, a load --
// returns something non-uniform whatever it was passed, and a barrier under it
// is one some invocations never reach.
func TestABarrierUnderAHelperReadingAPerInvocationSourceIsRejected(t *testing.T) {
	for _, c := range []struct {
		name  string
		decls string
		cond  string
	}{
		{"an id through the thread", `//accel:helper
func lane(t accel.Thread) uint32 { return t.LocalIndex() }`, "lane(t) > 2"},
		{"a load through a binding", `//accel:helper
func first(in []float32) float32 { return in[0] }`, "first(in) > 0"},
		{"a literal returned under divergent control", `//accel:helper
func pick(t accel.Thread) uint32 {
	if t.LocalIndex() > 2 {
		return 1
	}
	return 2
}`, "pick(t) > 1"},
		{"a per-invocation value two helpers deep", `//accel:helper
func lane(t accel.Thread) uint32 { return t.LocalIndex() }

//accel:helper
func twiceLane(t accel.Thread) uint32 { return lane(t) * 2 }`, "twiceLane(t) > 2"},
	} {
		t.Run(c.name, func(t *testing.T) {
			body := "\n\tif " + c.cond + " {\n\t\tt.Barrier()\n\t}"
			got := acceptWith(t, c.decls, body)
			if len(got) != 1 {
				t.Fatalf("got %d rejections, want 1: the helper's result is per-invocation "+
					"and the barrier sits under it:\n%s%s\n%v", len(got), c.decls, body, got)
			}
			if !strings.Contains(got[0].Msg, "non-uniform control") {
				t.Errorf("the message should say non-uniform control, got: %v", got[0].Msg)
			}
		})
	}

	// The accepting half, so the rule is about what the body reads rather than
	// about helpers: arithmetic on a uniform argument is uniform.
	got := acceptWith(t, `//accel:helper
func twice(x uint32) uint32 { return x * 2 }`, `
	if twice(t.GroupIndex()) > 2 {
		t.Barrier()
	}`)
	if len(got) != 0 {
		t.Errorf("a helper computing only from a uniform argument was rejected: %v", got)
	}
}

// A loop some invocations leave early runs a different number of iterations
// for each of them, so a local assigned anywhere in its body holds a different
// value after the loop even though every definition is a literal and the
// loop's own control is uniform. A continue changes no trip count, so it only
// affects what follows it in the iteration. A barrier under such a local after
// the loop is a barrier some invocations never reach.
func TestALocalAssignedInALoopWithANonUniformEscapeIsNonUniform(t *testing.T) {
	for _, c := range []struct {
		name string
		body string
	}{
		{"after a break", `
	last := uint32(0)
	for i := uint32(0); i < 8; i++ {
		if t.LocalIndex() > i {
			break
		}
		last = i
	}
	if last > 3 {
		t.Barrier()
	}`},
		{"before a break", `
	last := uint32(0)
	for i := uint32(0); i < 8; i++ {
		last = i
		if t.LocalIndex() > i {
			break
		}
	}
	if last > 3 {
		t.Barrier()
	}`},
		{"after a continue", `
	last := uint32(0)
	for i := uint32(0); i < 8; i++ {
		if t.LocalIndex() > i {
			continue
		}
		last = i
	}
	if last > 3 {
		t.Barrier()
	}`},
		{"after a break in a nested block", `
	last := uint32(0)
	for i := uint32(0); i < 8; i++ {
		{
			if t.LocalIndex() > i {
				break
			}
		}
		last = i
	}
	if last > 3 {
		t.Barrier()
	}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := accept(t, c.body)
			if len(got) != 1 {
				t.Fatalf("got %d rejections, want 1: the invocations that left the loop "+
					"early hold a different value of last:\n%s\n%v", len(got), c.body, got)
			}
			if !strings.Contains(got[0].Msg, "non-uniform control") {
				t.Errorf("the message should say non-uniform control, got: %v", got[0].Msg)
			}
		})
	}

	// The accepting half: an escape every invocation takes together leaves the
	// local uniform, and a continue skips only what follows it.
	for _, c := range []struct {
		name string
		body string
	}{
		{"uniform break", `
	last := uint32(0)
	for i := uint32(0); i < 8; i++ {
		if t.GroupIndex() > i {
			break
		}
		last = i
	}
	if last > 3 {
		t.Barrier()
	}`},
		{"assignment before a continue", `
	last := uint32(0)
	for i := uint32(0); i < 8; i++ {
		last = i
		if t.LocalIndex() > i {
			continue
		}
	}
	if last > 3 {
		t.Barrier()
	}`},
		{"escape confined to an inner loop", `
	last := uint32(0)
	for i := uint32(0); i < 8; i++ {
		for j := uint32(0); j < 8; j++ {
			if t.LocalIndex() > j {
				break
			}
		}
		last = i
	}
	if last > 3 {
		t.Barrier()
	}`},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := accept(t, c.body); len(got) != 0 {
				t.Errorf("rejected a uniform barrier: %v", got)
			}
		})
	}
}

// A non-uniform break skips every barrier in its loop, the ones before it in
// the body included: an invocation that broke in one iteration does not arrive
// at the barrier of the next while the others do. A continue skips only what
// follows it in the iteration, so a barrier before one is reached by all.
func TestABarrierBeforeANonUniformBreakIsRejected(t *testing.T) {
	got := accept(t, `
	for i := uint32(0); i < 8; i++ {
		t.Barrier()
		if t.LocalIndex() > i {
			break
		}
	}`)
	if len(got) != 1 {
		t.Fatalf("got %d rejections, want 1: an invocation that broke does not arrive at "+
			"the next iteration's barrier: %v", len(got), got)
	}
	if !strings.Contains(got[0].Msg, "a break under non-uniform control can skip this one") {
		t.Errorf("the message should name the break, got: %v", got[0].Msg)
	}

	got = accept(t, `
	for i := uint32(0); i < 8; i++ {
		t.Barrier()
		if t.LocalIndex() > i {
			continue
		}
	}`)
	if len(got) != 0 {
		t.Errorf("a barrier before a continue is reached by every invocation in every "+
			"iteration, and was rejected: %v", got)
	}
}

// The value-level statements behind the two acceptor cases above, so the
// levels are asserted where they are computed rather than only through a
// barrier.
func TestHelperResultsAndEscapedLocalsAreLevelledByWhatTheyRead(t *testing.T) {
	levels := levelsOf(t, "package k\n\nimport \"golang.design/x/accel\"\n\n"+`//accel:helper
func lane(t accel.Thread) uint32 { return t.LocalIndex() }

//accel:helper
func first(in []float32) float32 { return in[0] }

//accel:helper
func twice(x uint32) uint32 { return x * 2 }

//accel:kernel workgroup=64
func K(t accel.Thread, in []float32, out []float32) {
	seed := in[t.GlobalID().X]
	out[t.GlobalID().X] = seed
	viaLane := lane(t)
	viaLoad := first(in)
	viaArg := twice(t.GroupIndex())
	last := uint32(0)
	for i := uint32(0); i < 8; i++ {
		if t.LocalIndex() > i {
			break
		}
		last = i
	}
	kept := uint32(0)
	for i := uint32(0); i < 8; i++ {
		kept = i
		if t.LocalIndex() > i {
			break
		}
	}
	before := uint32(0)
	for i := uint32(0); i < 8; i++ {
		before = i
		if t.LocalIndex() > i {
			continue
		}
	}
}
`)
	for name, want := range map[string]uniform.Level{
		"viaLane": uniform.Non,
		"viaLoad": uniform.Non,
		"viaArg":  uniform.Workgroup,
		"last":    uniform.Non,
		"kept":    uniform.Non,
		"before":  uniform.Workgroup,
	} {
		if got := levels[name]; got != want {
			t.Errorf("%s is %v, want %v", name, got, want)
		}
	}
}
