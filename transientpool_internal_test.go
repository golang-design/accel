// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"errors"
	"strings"
	"testing"
)

// poolGraph builds a graph of copies through a transient, into pool.
func poolGraph(t *testing.T, d *Device, pool *TransientPool, n int) *Graph {
	t.Helper()
	usage := UsageStorage | UsageCopySrc | UsageCopyDst
	src, err := d.NewBuffer(BufferDescriptor{DType: F32, Count: n, Usage: usage, Label: "src"})
	if err != nil {
		t.Fatalf("src: %v", err)
	}
	t.Cleanup(func() { _ = src.Close() })
	dst, err := d.NewBuffer(BufferDescriptor{DType: F32, Count: n, Usage: usage, Label: "dst"})
	if err != nil {
		t.Fatalf("dst: %v", err)
	}
	t.Cleanup(func() { _ = dst.Close() })

	r := d.NewRecorder()
	r.UseTransientPool(pool)
	mid := r.Transient(BufferDescriptor{DType: F32, Count: n, Usage: usage, Label: "mid"})
	sv, err := src.View(0, n)
	if err != nil {
		t.Fatalf("src view: %v", err)
	}
	dv, err := dst.View(0, n)
	if err != nil {
		t.Fatalf("dst view: %v", err)
	}
	r.CopyBuffer(mid, sv)
	r.CopyBuffer(dv, mid)
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	t.Cleanup(func() { _ = g.Close() })
	return g
}

// The exclusivity rule is the whole cost of sharing a pool, and it is white-box
// tested because at v0 no backend reports a second queue: one queue runs its
// submissions in turn, so nothing reachable through Submit alone can overlap.
// Holding the claim by hand is what a second queue would do, and the check is
// that both submission entry points then refuse.
//
// Both, because a claim taken in Queue.Submit covers Submit only, and
// Queue.SubmitAfter reaches the same graph by another road. Taking it inside
// Graph.run instead is what makes one rule cover every road.
func TestSharedPoolRefusesAnOverlappingSubmission(t *testing.T) {
	d, err := OpenCPU(CPUOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()

	pool, err := d.NewTransientPool("p")
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	g := poolGraph(t, d, pool, 64)

	for _, tc := range []struct {
		name   string
		submit func() *Fence
	}{
		{"Submit", func() *Fence { return d.Queue().Submit(g) }},
		{"SubmitAfter", func() *Fence { return d.Queue().SubmitAfter(g) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := pool.begin(); err != nil {
				t.Fatalf("claim: %v", err)
			}
			err := tc.submit().Wait()
			pool.end()
			if err == nil {
				t.Fatal("a graph ran while another held its shared pool")
			}
			if !strings.Contains(err.Error(), "cannot overlap") {
				t.Errorf("the refusal should explain the rule: %v", err)
			}
		})
	}

	// And the claim is released on the way out, not leaked: a leaked claim
	// kills the pool for good, since nothing can clear it.
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("after two refusals the pool is unusable: %v", err)
	}
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("after a successful submission the pool is unusable: %v", err)
	}
}

// Growing the pool moves the memory a running graph is reading, so a build
// while anything executes is refused. Same reachability caveat as above.
func TestSharedPoolRefusesABuildWhileInFlight(t *testing.T) {
	d, err := OpenCPU(CPUOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()

	pool, err := d.NewTransientPool("p")
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	poolGraph(t, d, pool, 64)
	if err := pool.begin(); err != nil {
		t.Fatalf("claim: %v", err)
	}
	r := d.NewRecorder()
	r.UseTransientPool(pool)
	r.Transient(BufferDescriptor{DType: F32, Count: 4096, Usage: UsageStorage, Label: "big"})
	_, err = r.Build()
	pool.end()
	if err == nil {
		t.Fatal("a graph was built into a pool while a graph was executing out of it")
	}
	var lifetime *LifetimeError
	if !errors.As(err, &lifetime) {
		t.Fatalf("want a LifetimeError, got %T: %v", err, err)
	}

	// The refusal leaves the pool usable rather than half-grown.
	poolGraph(t, d, pool, 128)
}

// A pool's graph count has to match the graphs that actually reserved from it,
// and the two ways it can drift are opposite and both fatal.
//
// Too high and the pool never closes again: nothing can clear the count, so a
// Build that failed after reserving kills the pool for the rest of the program.
// Too low and the pool closes while a live graph still holds offsets into the
// memory it frees, which is a use-after-free whose symptom is another graph's
// results.
func TestSharedPoolCountsTheGraphsThatReserved(t *testing.T) {
	d, err := OpenCPU(CPUOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()

	pool, err := d.NewTransientPool("p")
	if err != nil {
		t.Fatal(err)
	}
	defer pool.Close()

	// A graph with no transients reserves nothing, so closing it must not
	// decrement a count it never incremented.
	empty := d.NewRecorder()
	empty.UseTransientPool(pool)
	g, err := empty.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if pool.Graphs() != 0 {
		t.Errorf("a graph with no transients counts %d against the pool", pool.Graphs())
	}
	// It must not claim the pool when it runs either, or closing the pool would
	// break a graph that needs nothing from it.
	if err := pool.begin(); err != nil {
		t.Fatalf("claim: %v", err)
	}
	err = d.Queue().Submit(g).Wait()
	pool.end()
	if err != nil {
		t.Errorf("a graph with no transients claimed the pool it takes nothing from: %v", err)
	}
	if err := g.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if n := pool.Graphs(); n != 0 {
		t.Fatalf("closing a graph that reserved nothing left the count at %d", n)
	}

	// A build that fails after reserving must give the reservation back. That
	// failure is driven directly rather than provoked, because every Build
	// error reachable from outside happens either before placement or not at
	// Build at all -- V19's closed-resource check runs at Submit. The call is
	// the one Build makes on its own error path, which is the thing under test.
	g2 := poolGraph(t, d, pool, 64)
	if pool.Graphs() != 1 {
		t.Fatalf("a graph with transients counts %d against the pool", pool.Graphs())
	}
	g2.releaseTransients()
	if n := pool.Graphs(); n != 0 {
		t.Fatalf("a failed build left %d graph(s) counted against the pool, which then "+
			"never closes", n)
	}
	// And the graph's own Close runs the same call again, so it has to be
	// harmless the second time or the count drifts the other way.
	if err := g2.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if n := pool.Graphs(); n != 0 {
		t.Fatalf("releasing twice left the count at %d", n)
	}
	if err := pool.Close(); err != nil {
		t.Fatalf("the pool is unusable after a failed build: %v", err)
	}

}
