// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"fmt"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// worked builds specs/003-command-graph.md's worked example: eight nodes, one
// genuine diamond, six transients, one attention block simplified.
//
// It is the strongest assertion this milestone can carry, because the spec
// wrote down the edge set and the barrier positions before the code existed. A
// test comparing results cannot tell a graph that overlapped correctly from one
// that serialized and got the same answer; this one can.
//
// The sizes are scaled down from the spec's 4 MiB tensors. The shape is what
// the assertions are about and it is identical; the byte counts belong to
// specs/017-graph-aliasing.md, which asserts the spec's own numbers.
type workedGraph struct {
	g              *accel.Graph
	x, kv, y, prms accel.Slot
}

func worked(t *testing.T, d *accel.Device) workedGraph {
	t.Helper()
	const n = 64 // stands in for the spec's 1 Mi elements
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst

	// One pipeline stands in for the spec's five: the assertions are about the
	// graph's shape, and five identical kernels would say nothing more.
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "op",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })

	r := d.NewRecorder()
	slot := func(name string, a accel.Access, count int) accel.Slot {
		return r.Slot(accel.SlotDescriptor{
			Name: name, Kind: accel.BindingStorageBuffer,
			DType: accel.F32, Access: a, MinCount: count,
		})
	}
	params := slot("params", accel.AccessReadWrite, n)
	x := slot("x", accel.AccessRead, n)
	kv := slot("kv", accel.AccessRead, n)
	y := slot("y", accel.AccessWrite, n)

	transient := func(label string, count int) accel.BufferView {
		return r.Transient(accel.BufferDescriptor{
			DType: accel.F32, Count: count, Usage: storage, Label: label,
		})
	}
	t0 := transient("t0", n) // normed
	t1 := transient("t1", n) // q
	t2 := transient("t2", n) // k
	t3 := transient("t3", n) // q after rope
	// The spec's t4 is half the width of the others. Here every transient is
	// full width, because Add indexes its three bindings together and needs
	// them equal; the differing sizes are what specs/017-graph-aliasing.md's
	// packing assertions are about, and the shape this test asserts -- the
	// diamond, the edges, the barrier positions -- is identical either way.
	t4 := transient("t4", n)
	t5 := transient("t5", n) // attn out

	wQ := newBuffer(t, d, "wQ", n, storage)
	wK := newBuffer(t, d, "wK", n, storage)
	count := accel.WorkgroupCount{X: 1}
	dispatch := func(a, b, out accel.Binding) {
		a.Index, b.Index, out.Index = 0, 1, 2
		r.Dispatch(p, []accel.Binding{a, b, out}, count)
	}

	// n0: the host write to params.
	r.CopyToBufferSlot(params, 0, n, make([]float32, n))
	// n1: t0 = norm(x, params)
	dispatch(accel.Binding{Slot: x}, accel.Binding{Slot: params}, accel.Binding{Buffer: t0})
	// n2: t1 = t0 @ wQ, and n3: t2 = t0 @ wK. The diamond's two arms.
	dispatch(accel.Binding{Buffer: t0}, accel.Binding{Buffer: whole(t, wQ)}, accel.Binding{Buffer: t1})
	dispatch(accel.Binding{Buffer: t0}, accel.Binding{Buffer: whole(t, wK)}, accel.Binding{Buffer: t2})
	// n4: t3 = rope(t1, params), extending one arm.
	dispatch(accel.Binding{Buffer: t1}, accel.Binding{Slot: params}, accel.Binding{Buffer: t3})
	// n5: t4 = scores(t3, t2), where the arms join.
	dispatch(accel.Binding{Buffer: t3}, accel.Binding{Buffer: t2}, accel.Binding{Buffer: t4})
	// n6: t5 = attend(t4, kv)
	dispatch(accel.Binding{Buffer: t4}, accel.Binding{Slot: kv}, accel.Binding{Buffer: t5})
	// n7: y = x + t5
	dispatch(accel.Binding{Buffer: t5}, accel.Binding{Slot: x}, accel.Binding{Slot: y})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	t.Cleanup(func() { _ = g.Close() })
	return workedGraph{g: g, x: x, kv: kv, y: y, prms: params}
}

// The inferred edge set, asserted against specs/003-command-graph.md's table.
//
// The absences are the point and are asserted as such: no edge between n2 and
// n3, which are the two GEMMs and the pair whose overlap matters most; and none
// from wQ, wK, or x, because nothing writes them.
func TestWorkedGraphEdges(t *testing.T) {
	d := openDevice(t)
	w := worked(t, d)

	want := map[string]bool{
		"0->1": true, "0->4": true, // params, read after the host write
		"1->2": true, "1->3": true, // t0, the diamond fanning out
		"2->4": true, // t1
		"4->5": true, // t3
		"3->5": true, // t2, the arms joining
		"5->6": true, // t4
		"6->7": true, // t5
	}
	got := map[string]bool{}
	for from, succ := range w.g.Edges() {
		for _, to := range succ {
			got[fmt.Sprintf("%d->%d", from, int(to))] = true
		}
	}

	for e := range want {
		if !got[e] {
			t.Errorf("edge %s is missing", e)
		}
	}
	for e := range got {
		if !want[e] {
			t.Errorf("edge %s should not exist", e)
		}
	}
	// Stated separately because it is the assertion the whole design is for.
	if got["2->3"] || got["3->2"] {
		t.Error("n2 and n3 are the two GEMMs and must not be ordered against each other")
	}
}

// The barrier positions, asserted against the spec's table. A planner emitting
// the right count in the wrong places would match a count assertion and lose
// the point entirely.
func TestWorkedGraphBarrierPositions(t *testing.T) {
	d := openDevice(t)
	w := worked(t, d)

	// Per the spec's table: a barrier before n0 (head of submission), n1, n2,
	// n4, n5, n6, and n7 — and none before n3, because n3 reads t0 and n2's
	// barrier already covers it.
	want := map[int]bool{0: true, 1: true, 2: true, 3: false, 4: true, 5: true, 6: true, 7: true}
	for node, expect := range want {
		got := w.g.NodeStats(accel.NodeID(node)).BarriersBefore
		if (got > 0) != expect {
			t.Errorf("node %d has %d barriers before it, want %v", node, got, expect)
		}
	}
	if got := w.g.Barriers(); got != 7 {
		t.Errorf("got %d barriers, want the spec's seven", got)
	}
	if got := w.g.Hazards(); got != 9 {
		t.Errorf("got %d hazards, want the spec's nine", got)
	}
}

// The sub-range variant of the spec's own note: two nodes writing disjoint
// halves of one transient produce no write-after-write edge, so the two GEMMs
// still overlap. Under whole-resource comparison there is an edge, a barrier,
// and the block's two largest dispatches serialize.
func TestWorkedGraphSubRangeVariant(t *testing.T) {
	const n = 64
	d := openDevice(t)
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "gemm",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	r := d.NewRecorder()
	qk := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: n, Usage: storage, Label: "qk",
	})
	in := newBuffer(t, d, "in", n, storage)
	lo, hi := qk, qk
	lo.Count = n / 2
	hi.Offset, hi.Count = n/2, n/2

	for _, out := range []accel.BufferView{lo, hi} {
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: whole(t, in)},
			{Index: 1, Buffer: whole(t, in)},
			{Index: 2, Buffer: out},
		}, accel.WorkgroupCount{X: 1})
	}
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	if got := g.Hazards(); got != 0 {
		t.Errorf("writes to disjoint halves of one transient have no hazard, got %d", got)
	}
	if e := g.Edges()[0]; len(e) != 0 {
		t.Errorf("the two GEMMs must still overlap, got edge %v", e)
	}
}

// The worked graph runs, and its result is what a serial execution of the same
// nodes produces. The plan assertions above say the shape is right; this says
// the shape is also executable.
func TestWorkedGraphRuns(t *testing.T) {
	const n = 64
	d := openDevice(t)
	w := worked(t, d)
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst

	bind := func(s accel.Slot, label string, fill float32) *accel.Buffer {
		b := newBuffer(t, d, label, n, storage)
		vals := make([]float32, n)
		for i := range vals {
			vals[i] = fill
		}
		if err := d.Queue().WriteBuffer(b, 0, vals); err != nil {
			t.Fatalf("write %s: %v", label, err)
		}
		if err := w.g.Bind(accel.Binding{Slot: s, Buffer: whole(t, b)}); err != nil {
			t.Fatalf("bind %s: %v", label, err)
		}
		return b
	}
	bind(w.prms, "params", 0)
	bind(w.x, "x", 1)
	bind(w.kv, "kv", 3)
	y := bind(w.y, "y", 0)

	if err := d.Queue().Submit(w.g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	// Every node is Add, and the buffers the graph does not fill are zero, so
	// the value follows the dependency chain the edges above assert:
	//
	//	t0 = x + params = 1    n1
	//	t1 = t0 + wQ    = 1    n2
	//	t2 = t0 + wK    = 1    n3
	//	t3 = t1 + params = 1   n4
	//	t4 = t3 + t2    = 2    n5, where the diamond's arms join
	//	t5 = t4 + kv    = 5    n6
	//	y  = t5 + x     = 6    n7
	//
	// Written out rather than asserted as a bare constant: the number is only
	// meaningful as the chain, and a reader who cannot check it cannot tell a
	// correct result from one a mis-ordered graph produced.
	got := readback(t, d, y)
	for i, v := range got {
		if v != 6 {
			t.Fatalf("element %d is %v, want 6", i, v)
		}
	}
}
