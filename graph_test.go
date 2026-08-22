// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"golang.design/x/accel"
)

func openDevice(t *testing.T) *accel.Device {
	t.Helper()
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = d.Close() })
	return d
}

func newBuffer(t *testing.T, d *accel.Device, label string, n int, u accel.BufferUsage) *accel.Buffer {
	t.Helper()
	b, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: n, Usage: u, Label: label,
	})
	if err != nil {
		t.Fatalf("new buffer %q: %v", label, err)
	}
	t.Cleanup(func() { _ = b.Close() })
	return b
}

func whole(t *testing.T, b *accel.Buffer) accel.BufferView {
	t.Helper()
	v, err := b.View(0, b.Count())
	if err != nil {
		t.Fatalf("view: %v", err)
	}
	return v
}

func readback(t *testing.T, d *accel.Device, b *accel.Buffer) []float32 {
	t.Helper()
	out := make([]float32, b.Count())
	if err := d.Queue().ReadBuffer(b, 0, out); err != nil {
		t.Fatalf("readback: %v", err)
	}
	return out
}

// The child's end-to-end criterion: record a graph of copies, submit it, wait,
// read back; then rebind a slot and resubmit the same graph, with no rebuild.
func TestAGraphOfCopiesRunsAndReplays(t *testing.T) {
	d := openDevice(t)
	q := d.Queue()

	a := newBuffer(t, d, "a", 4, accel.UsageStorage|accel.UsageCopySrc)
	b := newBuffer(t, d, "b", 4, accel.UsageStorage|accel.UsageCopySrc)
	out := newBuffer(t, d, "out", 4, accel.UsageStorage|accel.UsageCopyDst)
	if err := q.WriteBuffer(a, 0, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("write a: %v", err)
	}
	if err := q.WriteBuffer(b, 0, []float32{5, 6, 7, 8}); err != nil {
		t.Fatalf("write b: %v", err)
	}

	r := d.NewRecorder()
	in := r.Slot(accel.SlotDescriptor{
		Name: "in", Kind: accel.BindingStorageBuffer,
		DType: accel.F32, Access: accel.AccessRead, MinCount: 4,
	})
	mid := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: 4,
		Usage: accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst, Label: "mid",
	})
	r.CopyFromSlot(mid, in, 0, 4)
	r.CopyBuffer(whole(t, out), mid)

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	for _, c := range []struct {
		label string
		buf   *accel.Buffer
		want  []float32
	}{{"a", a, []float32{1, 2, 3, 4}}, {"b", b, []float32{5, 6, 7, 8}}} {
		if err := g.Bind(accel.Binding{Slot: in, Buffer: whole(t, c.buf)}); err != nil {
			t.Fatalf("bind %s: %v", c.label, err)
		}
		if err := q.Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		got := readback(t, d, out)
		for i := range c.want {
			if got[i] != c.want[i] {
				t.Fatalf("after binding %s got %v, want %v", c.label, got, c.want)
			}
		}
	}
}

// A host write baked into the graph is rewritten on every submission, which is
// what makes a graph carrying a small constant replayable.
func TestAHostWriteIsOwnedByTheGraph(t *testing.T) {
	d := openDevice(t)
	q := d.Queue()
	out := newBuffer(t, d, "out", 4, accel.UsageStorage|accel.UsageCopyDst)

	src := []float32{1, 2, 3, 4}
	r := d.NewRecorder()
	r.CopyToBuffer(whole(t, out), src)
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	// Mutating the caller's slice after recording must not change what the graph
	// writes: a Graph is immutable and holding the slice would break that.
	src[0] = 99
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	if got := readback(t, d, out); got[0] != 1 {
		t.Fatalf("the graph should own its bytes, got %v", got)
	}
}

// Memory reports all three fields from this milestone, with the pool pinned to
// the unaliased total. specs/017-graph-aliasing.md separates them, and pinning
// them here is what makes that a measured difference rather than a new number.
func TestMemoryReportsAllThreeFields(t *testing.T) {
	d := openDevice(t)
	r := d.NewRecorder()
	dst := newBuffer(t, d, "dst", 64, accel.UsageStorage|accel.UsageCopyDst)

	const n = 3
	views := make([]accel.BufferView, n)
	for i := range views {
		views[i] = r.Transient(accel.BufferDescriptor{
			DType: accel.F32, Count: 64,
			Usage: accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst,
		})
	}
	// A chain: each transient is written then read, so at most two are live at
	// any record-order point and the peak is below the total.
	for i := range views {
		r.CopyToBuffer(views[i], make([]float32, 64))
	}
	r.CopyBuffer(whole(t, dst), views[n-1])

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	m := g.Memory()
	if m.UnaliasedBytes != m.TransientBytes {
		t.Errorf("this milestone does not alias, so the pool is the unaliased total: "+
			"%d vs %d", m.TransientBytes, m.UnaliasedBytes)
	}
	if m.PeakBytes > m.TransientBytes {
		t.Errorf("peak %d cannot exceed the allocated %d", m.PeakBytes, m.TransientBytes)
	}
	if m.UnaliasedBytes < n*64*4 {
		t.Errorf("three 256-byte transients should total at least %d, got %d", n*64*4, m.UnaliasedBytes)
	}
}

// The record-order plan puts a barrier before every node. That is its
// definition, not an incidental fact, so it is asserted directly: 016 lowering
// the count has to be a visible change.
func TestTheRecordOrderPlanBarriersEveryNode(t *testing.T) {
	d := openDevice(t)
	dst := newBuffer(t, d, "dst", 4, accel.UsageStorage|accel.UsageCopyDst)

	r := d.NewRecorder()
	for range 4 {
		r.CopyToBuffer(whole(t, dst), []float32{1, 2, 3, 4})
	}
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	if got := g.Barriers(); got != 4 {
		t.Errorf("got %d barriers for 4 nodes, want 4", got)
	}
	nodes := g.Nodes()
	if len(nodes) != 4 {
		t.Fatalf("got %d nodes, want 4", len(nodes))
	}
	for i, n := range nodes {
		if n.BarriersBefore != 1 {
			t.Errorf("node %d has %d barriers before it, want 1", i, n.BarriersBefore)
		}
		if n.Kind != accel.NodeHostWrite {
			t.Errorf("node %d is %v, want a host write", i, n.Kind)
		}
	}
}

func TestSlotsAreDiscoverable(t *testing.T) {
	d := openDevice(t)
	r := d.NewRecorder()
	a := r.Slot(accel.SlotDescriptor{Name: "a", DType: accel.F32, MinCount: 4, Access: accel.AccessRead})
	b := r.Slot(accel.SlotDescriptor{DType: accel.U32, MinCount: 8, Access: accel.AccessWrite})
	dst := newBuffer(t, d, "dst", 4, accel.UsageStorage|accel.UsageCopyDst)
	r.CopyFromSlot(whole(t, dst), a, 0, 4)
	r.CopyToSlot(b, 0, 8, mustViewAs(t, newBuffer(t, d, "src", 8, accel.UsageStorage|accel.UsageCopySrc), accel.U32))

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	slots := g.Slots()
	if len(slots) != 2 {
		t.Fatalf("got %d slots, want 2", len(slots))
	}
	if slots[0].Slot != a || slots[0].Descriptor.Name != "a" {
		t.Errorf("first slot is %+v", slots[0])
	}
	// An unnamed slot gets a name, so every error about it can name something.
	if slots[1].Slot != b || slots[1].Descriptor.Name == "" {
		t.Errorf("second slot is %+v", slots[1])
	}
}

func mustViewAs(t *testing.T, b *accel.Buffer, d accel.DType) accel.BufferView {
	t.Helper()
	v, err := b.ViewAs(d, 0, b.Bytes()/d.Size())
	if err != nil {
		t.Fatalf("view as %v: %v", d, err)
	}
	return v
}

func TestBuildReportsEveryErrorAtOnce(t *testing.T) {
	d := openDevice(t)
	dst := newBuffer(t, d, "dst", 4, accel.UsageStorage|accel.UsageCopyDst)

	r := d.NewRecorder()
	r.CopyToBuffer(accel.BufferView{}, []float32{1})                                          // names no buffer
	r.CopyToBuffer(accel.BufferView{Buffer: dst, DType: accel.F32, Count: 999}, []float32{1}) // out of range
	r.CopyToBuffer(whole(t, dst), []int32{1, 2, 3, 4})                                        // wrong host slice

	_, err := r.Build()
	if err == nil {
		t.Fatal("expected a build failure")
	}
	for _, want := range []string{"names no buffer", "outside the buffer", "must be []float32"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the report should mention %q, got:\n%v", want, err)
		}
	}
}

func TestARecorderIsUsedOnce(t *testing.T) {
	d := openDevice(t)
	r := d.NewRecorder()
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	if _, err := r.Build(); err == nil {
		t.Error("a second Build should fail")
	}
	if s := r.Slot(accel.SlotDescriptor{DType: accel.F32}); s != 0 {
		t.Error("declaring a slot after Build should fail")
	}
	if v := r.Transient(accel.BufferDescriptor{DType: accel.F32, Count: 1}); v.Buffer != nil {
		t.Error("declaring a transient after Build should fail")
	}
}

func TestAGraphIsSubmittedOneAtATime(t *testing.T) {
	d := openDevice(t)
	dst := newBuffer(t, d, "dst", 1<<14, accel.UsageStorage|accel.UsageCopyDst)
	src := newBuffer(t, d, "src", 1<<14, accel.UsageStorage|accel.UsageCopySrc)

	want := make([]float32, 1<<14)
	for i := range want {
		want[i] = float32(i)
	}
	if err := d.Queue().WriteBuffer(src, 0, want); err != nil {
		t.Fatalf("write: %v", err)
	}

	r := d.NewRecorder()
	for range 32 {
		r.CopyBuffer(whole(t, dst), whole(t, src))
	}
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	// Hammer it. Whatever the interleaving, a refused submission must be refused
	// as in-flight and never half-run: two overlapping submissions would write
	// each other's transients, so silently queueing the second is the bug.
	var wg sync.WaitGroup
	var refusals atomic.Int64
	var wrong atomic.Int64
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 25 {
				err := d.Queue().Submit(g).Wait()
				switch {
				case err == nil:
				case errors.Is(err, accel.ErrGraphInFlight):
					refusals.Add(1)
				default:
					wrong.Add(1)
					t.Error(err)
				}
			}
		}()
	}
	wg.Wait()

	if n := wrong.Load(); n != 0 {
		t.Errorf("%d submissions failed for a reason other than being in flight", n)
	}
	if refusals.Load() == 0 {
		t.Log("no submission was refused; the machine serialized them all")
	}
	got := readback(t, d, dst)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("element %d is %v, want %v", i, got[i], want[i])
		}
	}
}

// Rebinding while a submission is in flight is refused for the same reason:
// half the graph would see one resource and half the other.
func TestRebindDuringASubmissionIsRefused(t *testing.T) {
	d := openDevice(t)
	dst := newBuffer(t, d, "dst", 1<<14, accel.UsageStorage|accel.UsageCopyDst)
	src := newBuffer(t, d, "src", 1<<14, accel.UsageStorage|accel.UsageCopySrc)

	r := d.NewRecorder()
	in := r.Slot(accel.SlotDescriptor{
		Name: "in", Kind: accel.BindingStorageBuffer,
		DType: accel.F32, Access: accel.AccessRead, MinCount: 1 << 14,
	})
	for range 64 {
		r.CopyFromSlot(whole(t, dst), in, 0, 1<<14)
	}
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := g.Bind(accel.Binding{Slot: in, Buffer: whole(t, src)}); err != nil {
		t.Fatalf("bind: %v", err)
	}

	var wg sync.WaitGroup
	var wrong atomic.Int64
	wg.Add(1)
	go func() {
		defer wg.Done()
		for range 50 {
			err := g.Bind(accel.Binding{Slot: in, Buffer: whole(t, src)})
			if err != nil && !errors.Is(err, accel.ErrGraphInFlight) {
				wrong.Add(1)
			}
		}
	}()
	for range 20 {
		if err := d.Queue().Submit(g).Wait(); err != nil && !errors.Is(err, accel.ErrGraphInFlight) {
			t.Errorf("submit: %v", err)
		}
	}
	wg.Wait()
	if n := wrong.Load(); n != 0 {
		t.Errorf("%d rebinds failed for a reason other than being in flight", n)
	}
}

func TestClosingAGraphReleasesItsTransients(t *testing.T) {
	d := openDevice(t)
	r := d.NewRecorder()
	r.Transient(accel.BufferDescriptor{DType: accel.F32, Count: 1024, Usage: accel.UsageStorage})
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	// A device cannot close while a graph is live: the graph owns memory and an
	// executable, and a device closing under it would strand both.
	if err := d.Close(); err == nil {
		t.Error("closing a device with a live graph should fail")
	}
	if err := g.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := g.Close(); err != nil {
		t.Errorf("Close should be idempotent: %v", err)
	}
}

// Two submissions to one queue are fully ordered: the second begins no earlier
// than the first ends, and the first's writes are visible to it with no caller
// action. specs/003-command-graph.md states this as a deliberate cost, so it is
// asserted rather than assumed.
func TestSubmissionsOnOneQueueAreOrdered(t *testing.T) {
	d := openDevice(t)
	q := d.Queue()
	dst := newBuffer(t, d, "dst", 1<<14, accel.UsageStorage|accel.UsageCopyDst)

	// Each graph is long enough that an unordered implementation genuinely
	// overlaps them rather than finishing before the next call is made.
	build := func(v float32) *accel.Graph {
		t.Helper()
		payload := make([]float32, 1<<14)
		for i := range payload {
			payload[i] = v
		}
		r := d.NewRecorder()
		for range 64 {
			r.CopyToBuffer(whole(t, dst), payload)
		}
		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		t.Cleanup(func() { _ = g.Close() })
		return g
	}
	first, second := build(1), build(2)

	// Submitted back to back without waiting on the first.
	f1 := q.Submit(first)
	f2 := q.Submit(second)
	if err := f2.Wait(); err != nil {
		t.Fatalf("second: %v", err)
	}
	if err := f1.Wait(); err != nil {
		t.Fatalf("first: %v", err)
	}
	got := readback(t, d, dst)
	for i := range got {
		if got[i] != 2 {
			t.Fatalf("element %d is %v: the second submission did not run last", i, got[i])
		}
	}
}

// A host write joins the same stream rather than being flushed alongside it. An
// implementation that flushed on the caller's goroutine would write device
// memory while a submission already running was reading it.
func TestAHostWriteIsOrderedAgainstSubmissions(t *testing.T) {
	d := openDevice(t)
	q := d.Queue()
	src := newBuffer(t, d, "src", 1<<14, accel.UsageStorage|accel.UsageCopySrc)
	dst := newBuffer(t, d, "dst", 1<<14, accel.UsageStorage|accel.UsageCopyDst)

	ones := make([]float32, 1<<14)
	for i := range ones {
		ones[i] = 1
	}
	if err := q.WriteBuffer(src, 0, ones); err != nil {
		t.Fatalf("write: %v", err)
	}

	r := d.NewRecorder()
	for range 64 {
		r.CopyBuffer(whole(t, dst), whole(t, src))
	}
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	f := q.Submit(g)

	// Queued behind the submission, so the copies above all see ones.
	twos := make([]float32, 1<<14)
	for i := range twos {
		twos[i] = 2
	}
	if err := q.WriteBuffer(src, 0, twos); err != nil {
		t.Fatalf("write: %v", err)
	}
	after := q.Flush()

	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	if err := after.Wait(); err != nil {
		t.Fatalf("flush: %v", err)
	}
	got := readback(t, d, dst)
	for i := range got {
		if got[i] != 1 {
			t.Fatalf("element %d is %v: a later write reached memory the submission was reading",
				i, got[i])
		}
	}
}
