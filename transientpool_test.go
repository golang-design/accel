// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// buildInto records a graph of n-element work, optionally into a shared pool,
// and returns it with its output buffer.
func buildInto(t *testing.T, d *accel.Device, pool *accel.TransientPool, n int) (*accel.Graph, *accel.Buffer) {
	t.Helper()
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "add",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })

	in := newBuffer(t, d, "in", n, storage)
	out := newBuffer(t, d, "out", n, storage)
	ones := make([]float32, n)
	for i := range ones {
		ones[i] = 1
	}
	if err := d.Queue().WriteBuffer(in, 0, ones); err != nil {
		t.Fatalf("write: %v", err)
	}

	r := d.NewRecorder()
	if pool != nil {
		r.UseTransientPool(pool)
	}
	// An intermediate, so the graph actually has transients to place.
	mid := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: n, Usage: storage, Label: "mid",
	})
	r.Dispatch(p, []accel.Binding{
		{Index: 0, Buffer: whole(t, in)},
		{Index: 1, Buffer: whole(t, in)},
		{Index: 2, Buffer: mid},
	}, nil, accel.WorkgroupCount{X: (n + 63) / 64})
	r.Dispatch(p, []accel.Binding{
		{Index: 0, Buffer: mid},
		{Index: 1, Buffer: mid},
		{Index: 2, Buffer: whole(t, out)},
	}, nil, accel.WorkgroupCount{X: (n + 63) / 64})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	t.Cleanup(func() { _ = g.Close() })
	return g, out
}

// Graphs sharing a pool allocate once, sized to the largest, and compute what
// they would have computed alone.
//
// The saving is the point: a set of plans over one model holds one plan's
// transients rather than all of them, which specs/007-tensor-layer.md puts at
// a gigabyte for five prefill buckets.
func TestGraphsShareOneTransientPool(t *testing.T) {
	d := openDevice(t)
	pool, err := d.NewTransientPool("buckets")
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	defer pool.Close()

	sizes := []int{64, 256, 128}
	var graphs []*accel.Graph
	var outs []*accel.Buffer
	var largest int
	for _, n := range sizes {
		g, out := buildInto(t, d, pool, n)
		graphs = append(graphs, g)
		outs = append(outs, out)
		if m := g.Memory().TransientBytes; m > largest {
			largest = m
		}
	}

	if pool.Graphs() != len(sizes) {
		t.Errorf("the pool reports %d graphs, want %d", pool.Graphs(), len(sizes))
	}
	// Sized to the largest, not to the sum: that difference is the feature.
	if pool.Bytes() != largest {
		t.Errorf("the pool allocated %d bytes and its largest graph needs %d",
			pool.Bytes(), largest)
	}
	var sum int
	for _, g := range graphs {
		sum += g.Memory().TransientBytes
	}
	if sum <= pool.Bytes() {
		t.Fatalf("the graphs need %d bytes separately and the pool holds %d, so this "+
			"configuration does not demonstrate sharing", sum, pool.Bytes())
	}

	// Each graph computes 1+1 = 2, then 2+2 = 4, whatever else has run.
	for i, g := range graphs {
		if err := d.Queue().Submit(g).Wait(); err != nil {
			t.Fatalf("graph %d: %v", i, err)
		}
		for j, v := range readback(t, d, outs[i]) {
			if v != 4 {
				t.Fatalf("graph %d element %d is %v, want 4", i, j, v)
			}
		}
	}

	// And again in reverse, because the property that would break if the
	// offsets were not per-graph is that an earlier graph's leftovers do not
	// change a later one's answer.
	for i := len(graphs) - 1; i >= 0; i-- {
		if err := d.Queue().Submit(graphs[i]).Wait(); err != nil {
			t.Fatalf("graph %d on replay: %v", i, err)
		}
		for j, v := range readback(t, d, outs[i]) {
			if v != 4 {
				t.Fatalf("graph %d element %d is %v after another graph used the pool",
					i, j, v)
			}
		}
	}
}

// A graph sharing a pool computes exactly what it computes with its own.
//
// The sharing has to be invisible to a caller, and this is the comparison that
// says so rather than an argument that it should be.
func TestSharingAPoolChangesNothing(t *testing.T) {
	d := openDevice(t)
	const n = 128

	alone, aloneOut := buildInto(t, d, nil, n)
	if err := d.Queue().Submit(alone).Wait(); err != nil {
		t.Fatalf("alone: %v", err)
	}
	want := readback(t, d, aloneOut)

	pool, err := d.NewTransientPool("p")
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	defer pool.Close()
	// Another graph first, so the pool's memory holds something before the one
	// under test runs.
	other, _ := buildInto(t, d, pool, 256)
	if err := d.Queue().Submit(other).Wait(); err != nil {
		t.Fatalf("other: %v", err)
	}
	shared, sharedOut := buildInto(t, d, pool, n)
	if err := d.Queue().Submit(shared).Wait(); err != nil {
		t.Fatalf("shared: %v", err)
	}
	got := readback(t, d, sharedOut)

	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("element %d is %v with a shared pool and %v with its own",
				i, got[i], want[i])
		}
	}
}

// The pool's lifetime rules, each of which prevents a use-after-free whose
// symptom would be a wrong answer rather than a crash.
func TestTransientPoolLifetime(t *testing.T) {
	d := openDevice(t)
	pool, err := d.NewTransientPool("p")
	if err != nil {
		t.Fatalf("pool: %v", err)
	}

	g, _ := buildInto(t, d, pool, 64)

	// Closing a pool with graphs still open is refused: they hold offsets into
	// memory it would free.
	if err := pool.Close(); err == nil {
		t.Error("a pool closed with a graph still open")
	} else if !strings.Contains(err.Error(), "close them first") {
		t.Errorf("the refusal should say what to do: %v", err)
	}

	if err := g.Close(); err != nil {
		t.Fatalf("graph close: %v", err)
	}
	if pool.Graphs() != 0 {
		t.Errorf("the pool reports %d graphs after its only one closed", pool.Graphs())
	}
	// Closing a graph does not free the pool: it is the caller's, which is what
	// makes it shareable.
	if pool.Bytes() == 0 {
		t.Error("closing a graph freed the pool it shared")
	}

	if err := pool.Close(); err != nil {
		t.Fatalf("pool close: %v", err)
	}
	if err := pool.Close(); err != nil {
		t.Errorf("closing a pool twice must be harmless: %v", err)
	}
	if pool.Bytes() != 0 {
		t.Errorf("a closed pool reports %d bytes", pool.Bytes())
	}

	// And building into a closed pool is refused rather than silently
	// allocating a graph's own.
	r := d.NewRecorder()
	r.UseTransientPool(pool)
	r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: 16, Usage: accel.UsageStorage, Label: "t",
	})
	if _, err := r.Build(); err == nil {
		t.Error("a graph built into a closed pool")
	}
}
