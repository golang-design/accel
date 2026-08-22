// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"fmt"
	"testing"

	"golang.design/x/accel"
)

// Two nodes writing disjoint halves of one buffer have no hazard. Written as a
// pair with the overlapping case, because a planner that compares whole
// resources passes the second alone — and whole-resource comparison is not a
// missed optimization, it is the one that matters: a tiled workload is a stream
// of nodes touching disjoint slices of one allocation.
func TestSubRangesAreComparedNotResources(t *testing.T) {
	d := openDevice(t)

	build := func(t *testing.T, aOff, aLen, bOff, bLen int) *accel.Graph {
		t.Helper()
		buf := newBuffer(t, d, "shared", 16, accel.UsageStorage|accel.UsageCopyDst)
		r := d.NewRecorder()
		for _, c := range [][2]int{{aOff, aLen}, {bOff, bLen}} {
			v, err := buf.View(c[0], c[1])
			if err != nil {
				t.Fatalf("view: %v", err)
			}
			r.CopyToBuffer(v, make([]float32, c[1]))
		}
		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		t.Cleanup(func() { _ = g.Close() })
		return g
	}

	disjoint := build(t, 0, 8, 8, 8)
	if got := disjoint.Hazards(); got != 0 {
		t.Errorf("two writes to disjoint halves have no hazard, got %d", got)
	}
	if e := disjoint.Edges()[0]; len(e) != 0 {
		t.Errorf("disjoint writes should produce no edge, got %v", e)
	}

	// One element of overlap is a hazard. Elements, so this is one float32 of
	// four bytes rather than a byte, which is the smallest overlap a view can
	// express.
	touching := build(t, 0, 8, 7, 8)
	if got := touching.Hazards(); got != 1 {
		t.Errorf("writes overlapping by one element are one hazard, got %d", got)
	}
	if e := touching.Edges()[0]; len(e) != 1 || e[0] != 1 {
		t.Errorf("overlapping writes should produce edge 0 -> 1, got %v", e)
	}
}

// Two readers of one range may overlap freely. A read-only resource generates
// zero edges however many nodes read it, which is the largest single saving in
// inference and is free: weight buffers are read-only and are most of the
// accesses in a model graph.
func TestReadsDoNotHazardAgainstReads(t *testing.T) {
	d := openDevice(t)
	src := newBuffer(t, d, "weights", 8, accel.UsageStorage|accel.UsageCopySrc)

	r := d.NewRecorder()
	for range 5 {
		dst := newBuffer(t, d, "out", 8, accel.UsageStorage|accel.UsageCopyDst)
		r.CopyBuffer(whole(t, dst), whole(t, src))
	}
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	if got := g.Hazards(); got != 0 {
		t.Errorf("five readers of one buffer have no hazard, got %d", got)
	}
	if got := g.Barriers(); got != 1 {
		t.Errorf("got %d barriers, want only the head-of-submission one", got)
	}
}

// The three hazards, each in isolation, so that a planner emitting the wrong
// kind is distinguishable from one emitting none.
func TestEachHazardProducesItsEdge(t *testing.T) {
	d := openDevice(t)

	cases := []struct {
		name string
		rec  func(r *accel.Recorder, a, b accel.BufferView)
	}{
		{"read after write", func(r *accel.Recorder, a, b accel.BufferView) {
			r.CopyToBuffer(a, make([]float32, a.Count)) // writes a
			r.CopyBuffer(b, a)                          // reads a
		}},
		{"write after write", func(r *accel.Recorder, a, b accel.BufferView) {
			r.CopyToBuffer(a, make([]float32, a.Count))
			r.CopyToBuffer(a, make([]float32, a.Count))
		}},
		{"write after read", func(r *accel.Recorder, a, b accel.BufferView) {
			r.CopyBuffer(b, a)                          // reads a
			r.CopyToBuffer(a, make([]float32, a.Count)) // writes a
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			a := newBuffer(t, d, "a", 4,
				accel.UsageStorage|accel.UsageCopySrc|accel.UsageCopyDst)
			b := newBuffer(t, d, "b", 4,
				accel.UsageStorage|accel.UsageCopySrc|accel.UsageCopyDst)
			r := d.NewRecorder()
			c.rec(r, whole(t, a), whole(t, b))
			g, err := r.Build()
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			defer g.Close()

			if got := g.Hazards(); got != 1 {
				t.Errorf("got %d hazards, want 1", got)
			}
			if e := g.Edges()[0]; len(e) != 1 || e[0] != 1 {
				t.Errorf("want edge 0 -> 1, got %v", e)
			}
			if n := g.NodeStats(1); n.BarriersBefore != 1 {
				t.Errorf("the dependent node needs a barrier, got %d", n.BarriersBefore)
			}
		})
	}
}

// A node that reads and writes one range must not hazard against itself. That
// is what classifying all of a node's accesses before committing any of them
// buys, and without it every in-place operation depends on itself.
func TestANodeDoesNotHazardAgainstItself(t *testing.T) {
	d := openDevice(t)
	buf := newBuffer(t, d, "buf", 8,
		accel.UsageStorage|accel.UsageCopySrc|accel.UsageCopyDst)

	lo, err := buf.View(0, 4)
	if err != nil {
		t.Fatalf("view: %v", err)
	}
	hi, err := buf.View(4, 4)
	if err != nil {
		t.Fatalf("view: %v", err)
	}
	r := d.NewRecorder()
	r.CopyBuffer(lo, hi) // one node, reading and writing the same buffer
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	if got := g.Hazards(); got != 0 {
		t.Errorf("a single node cannot hazard against itself, got %d", got)
	}
	if e := g.Edges()[0]; len(e) != 0 {
		t.Errorf("got a self edge %v", e)
	}
}

// A barrier is queue-wide, so one emitted for a hazard on one resource also
// covers every hazard whose source precedes it and whose destination follows.
// This is what makes the barrier count far below the hazard count, and it is
// the "also covers" column of specs/003-command-graph.md's worked example.
//
// The shape matters: n3's hazard is on b, written by n1, and n1 precedes the
// barrier emitted before n2. A planner that emits per hazard rather than
// clearing what a queue-wide barrier already satisfied emits one before n3 too.
func TestOneBarrierCoversUnrelatedHazards(t *testing.T) {
	d := openDevice(t)
	a := newBuffer(t, d, "a", 4, accel.UsageStorage|accel.UsageCopySrc|accel.UsageCopyDst)
	b := newBuffer(t, d, "b", 4, accel.UsageStorage|accel.UsageCopySrc|accel.UsageCopyDst)
	outA := newBuffer(t, d, "outA", 4, accel.UsageStorage|accel.UsageCopyDst)
	outB := newBuffer(t, d, "outB", 4, accel.UsageStorage|accel.UsageCopyDst)

	r := d.NewRecorder()
	r.CopyToBuffer(whole(t, a), []float32{1, 2, 3, 4}) // n0 writes a
	r.CopyToBuffer(whole(t, b), []float32{5, 6, 7, 8}) // n1 writes b, independent of n0
	r.CopyBuffer(whole(t, outA), whole(t, a))          // n2 reads a: RAW sourced at n0
	r.CopyBuffer(whole(t, outB), whole(t, b))          // n3 reads b: RAW sourced at n1
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	if got := g.Hazards(); got != 2 {
		t.Errorf("got %d hazards, want a read-after-write on each of a and b", got)
	}
	if n := g.NodeStats(1); n.BarriersBefore != 0 {
		t.Errorf("n1 is independent of n0 and needs no barrier, got %d", n.BarriersBefore)
	}
	if n := g.NodeStats(2); n.BarriersBefore != 1 {
		t.Errorf("n2 reads what n0 wrote and needs a barrier, got %d", n.BarriersBefore)
	}
	if n := g.NodeStats(3); n.BarriersBefore != 0 {
		t.Errorf("n3's hazard is sourced at n1, which precedes n2's queue-wide "+
			"barrier, so it needs none of its own: got %d", n.BarriersBefore)
	}
	if got := g.Barriers(); got != 2 {
		t.Errorf("got %d barriers for 2 hazards plus the submission head, want 2", got)
	}
}

// Inference and barrier planning must be identical across builds. Not for
// aesthetics: the plan comparisons in specs/017-graph-aliasing.md compare plans,
// and one that depended on map iteration order would fail one run in ten and be
// marked flaky rather than investigated. This is the shape that caught the
// kernel front end's map-order bug.
func TestPlansAreDeterministic(t *testing.T) {
	d := openDevice(t)

	plan := func() string {
		bufs := make([]*accel.Buffer, 6)
		for i := range bufs {
			bufs[i] = newBuffer(t, d, fmt.Sprintf("b%d", i), 16,
				accel.UsageStorage|accel.UsageCopySrc|accel.UsageCopyDst)
		}
		r := d.NewRecorder()
		for i := range 12 {
			dst, src := bufs[i%len(bufs)], bufs[(i*5+3)%len(bufs)]
			if dst == src {
				src = bufs[(i+1)%len(bufs)]
			}
			r.CopyBuffer(whole(t, dst), whole(t, src))
		}
		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()

		out := fmt.Sprintf("barriers=%d hazards=%d\n", g.Barriers(), g.Hazards())
		for i, e := range g.Edges() {
			out += fmt.Sprintf("%d -> %v before=%d\n", i, e, g.NodeStats(accel.NodeID(i)).BarriersBefore)
		}
		return out
	}

	first := plan()
	for i := range 20 {
		if got := plan(); got != first {
			t.Fatalf("build %d produced a different plan:\n%s\nwant:\n%s", i, got, first)
		}
	}
}

// Slots hazard against each other by slot identity, because their eventual
// resource is unknown at build. Two nodes touching one slot are ordered even
// though nothing is bound yet; V24 is what checks that two slots did not later
// land on the same bytes.
func TestSlotsHazardByIdentity(t *testing.T) {
	d := openDevice(t)
	src := newBuffer(t, d, "src", 4, accel.UsageStorage|accel.UsageCopySrc)

	r := d.NewRecorder()
	s := r.Slot(accel.SlotDescriptor{
		Name: "io", Kind: accel.BindingStorageBuffer,
		DType: accel.F32, Access: accel.AccessReadWrite, MinCount: 4,
	})
	other := r.Slot(accel.SlotDescriptor{
		Name: "other", Kind: accel.BindingStorageBuffer,
		DType: accel.F32, Access: accel.AccessWrite, MinCount: 4,
	})
	r.CopyToSlot(s, 0, 4, whole(t, src))     // n0 writes s
	r.CopyToSlot(other, 0, 4, whole(t, src)) // n1 writes a different slot
	r.CopyToSlot(s, 0, 4, whole(t, src))     // n2 writes s again: WAW

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	if got := g.Hazards(); got != 1 {
		t.Errorf("got %d hazards, want the one write-after-write on the shared slot", got)
	}
	if e := g.Edges()[0]; len(e) != 1 || e[0] != 2 {
		t.Errorf("want edge 0 -> 2, got %v", e)
	}
	if e := g.Edges()[1]; len(e) != 0 {
		t.Errorf("the other slot is independent, got %v", e)
	}
}
