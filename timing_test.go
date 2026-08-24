// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"testing"
	"time"

	"golang.design/x/accel"
)

// A graph that asked for timings reports how long the device took, and one that
// did not reports nothing.
//
// specs/003-command-graph.md listed this as a hole and said so plainly: "at v0
// a caller asking 'which node is slow' has no answer from this library, on a
// library whose reason to exist is throughput." This answers the smaller
// question — how long did this take — and does not pretend to answer the
// larger one.
//
// Opt-in for the reason the run-time counters are: it costs the backend a query
// it would not otherwise make, and a throughput library should not spend a
// caller's time measuring itself unless asked.
func TestASubmissionReportsDeviceTimeWhenAsked(t *testing.T) {
	const n = 4096
	d := openDevice(t)
	q := d.Queue()
	p := addPipeline(t, d)

	build := func(timed bool, dispatches int) *accel.Graph {
		storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
		a := newBuffer(t, d, "a", n, storage)
		b := newBuffer(t, d, "b", n, storage)
		out := newBuffer(t, d, "out", n, storage)

		r := d.NewRecorder()
		if timed {
			r.CollectTimings(true)
		}
		// Several dispatches, so the device has something to spend time on.
		for range dispatches {
			r.Dispatch(p, []accel.Binding{
				{Index: 0, Buffer: whole(t, a)},
				{Index: 1, Buffer: whole(t, b)},
				{Index: 2, Buffer: whole(t, out)},
			}, nil, p.Workgroups(n))
		}
		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		t.Cleanup(func() { _ = g.Close() })
		return g
	}

	// The work is sized from the platform's clock rather than fixed, because a
	// zero here is ambiguous in a way this test cannot resolve otherwise:
	// Elapsed is zero both when no timing was collected and when the work
	// finished inside one tick of the monotonic clock. Windows' tick is coarse
	// -- hundreds of microseconds where Linux and macOS are tens of nanoseconds
	// -- and eight adds over 4096 elements finish well inside it, so this test
	// failed there while testing nothing about the timer.
	//
	// Doubling until the timer resolves the run is self-calibrating and
	// terminates: the cap is what fails, and it fails saying how much work went
	// unmeasured rather than reporting a zero whose cause is unknown.
	tick := clockTick()
	var stats accel.SubmissionStats
	dispatches := 8
	for {
		timed := build(true, dispatches)
		f := q.Submit(timed)
		if err := f.Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		var err error
		stats, err = f.Stats()
		if err != nil {
			t.Fatalf("stats: %v", err)
		}
		if stats.Elapsed > 0 || dispatches >= 8192 {
			break
		}
		dispatches *= 2
	}
	if stats.Elapsed <= 0 {
		t.Errorf("a graph that asked for timings reported %v after %d dispatches over "+
			"%d elements; this platform's monotonic clock resolves %v, so the work was "+
			"not too small to measure", stats.Elapsed, dispatches, n, tick)
	}

	// And silence when nobody asked, so the cost is only paid on request.
	quiet := build(false, dispatches)
	f2 := q.Submit(quiet)
	if err := f2.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	stats2, err := f2.Stats()
	if err != nil {
		t.Fatalf("stats: %v", err)
	}
	if stats2.Elapsed != 0 {
		t.Errorf("a graph that did not ask for timings reported %v", stats2.Elapsed)
	}
}

// Timings are read before the submission completes at a caller's peril, and
// Stats says so rather than returning a stale number.
func TestTimingsAreNotReadableBeforeCompletion(t *testing.T) {
	d := openDevice(t)
	q := d.Queue()
	p := addPipeline(t, d)
	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	a := newBuffer(t, d, "a", 64, storage)
	b := newBuffer(t, d, "b", 64, storage)
	out := newBuffer(t, d, "out", 64, storage)

	r := d.NewRecorder()
	r.CollectTimings(true)
	r.Dispatch(p, []accel.Binding{
		{Index: 0, Buffer: whole(t, a)},
		{Index: 1, Buffer: whole(t, b)},
		{Index: 2, Buffer: whole(t, out)},
	}, nil, p.Workgroups(64))
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	f := q.Submit(g)
	if !f.Done() {
		if _, err := f.Stats(); err == nil {
			t.Error("Stats returned a figure for a submission that has not completed")
		}
	}
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
}

// clockTick is the smallest non-zero interval this platform's monotonic clock
// reports.
//
// Go's clock resolution is not uniform across platforms -- it is tens of
// nanoseconds on Linux and macOS and hundreds of microseconds or worse on
// Windows -- so a test that asserts a duration is positive is asserting
// something about the platform unless it knows this number.
func clockTick() time.Duration {
	start := time.Now()
	for {
		if d := time.Since(start); d > 0 {
			return d
		}
	}
}
