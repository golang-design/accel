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
