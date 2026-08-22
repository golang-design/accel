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

// The diamond of specs/017-graph-aliasing.md section 2, which is the case
// aliasing exists to get right.
//
// n1 writes t0; n2 and n3 read it; n4 writes t3. In record order t0 occupies
// [n1, n3] and t3 occupies [n4, ...] — disjoint, so an interval planner aliases
// them. But n3 reads t0 and n4 writes t3, and nothing orders n3 before n4, so a
// backend that runs the two arms at once corrupts t0.
//
// The assertion is on the placement, not on the output: an output comparison on
// the CPU backend can pass while the placement is wrong, because this backend
// executes serially and cannot observe the race it would create.
func TestTheDiamondDoesNotAliasUnorderedTransients(t *testing.T) {
	const n = 64
	d := openDevice(t)
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "op",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	in := newBuffer(t, d, "in", n, storage)
	sink := newBuffer(t, d, "sink", n, storage)
	r := d.NewRecorder()
	transient := func(label string) accel.BufferView {
		return r.Transient(accel.BufferDescriptor{
			DType: accel.F32, Count: n, Usage: storage, Label: label,
		})
	}
	t0, t1, t3 := transient("t0"), transient("t1"), transient("t3")
	count := accel.WorkgroupCount{X: 1}
	dispatch := func(a, b, out accel.BufferView) {
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: a}, {Index: 1, Buffer: b}, {Index: 2, Buffer: out},
		}, count)
	}

	dispatch(whole(t, in), whole(t, in), t0) // n0 writes t0
	dispatch(t0, whole(t, in), t1)           // n1 reads t0
	dispatch(t0, whole(t, in), whole(t, sink))
	// n2 reads t0; the two arms are unordered
	dispatch(t1, whole(t, in), t3) // n3 writes t3, unordered against n2

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	place := g.TransientPlacement()
	if len(place) != 3 {
		t.Fatalf("got %d placements, want 3", len(place))
	}
	byLabel := map[string]accel.TransientPlacement{}
	for _, pl := range place {
		byLabel[pl.Label] = pl
	}
	a, b := byLabel["t0"], byLabel["t3"]
	if a.Offset < b.Offset+b.Bytes && b.Offset < a.Offset+a.Bytes {
		t.Errorf("t0 at [%d, %d) and t3 at [%d, %d) share bytes: n2 reads t0 and n3 "+
			"writes t3, and nothing orders them, so a backend that runs the diamond's "+
			"two arms at once corrupts t0",
			a.Offset, a.Offset+a.Bytes, b.Offset, b.Offset+b.Bytes)
	}

	// And the graph still runs: the placement assertion says the shape is safe,
	// this says it is executable.
	ones := make([]float32, n)
	for i := range ones {
		ones[i] = 1
	}
	if err := d.Queue().WriteBuffer(in, 0, ones); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	for i, v := range readback(t, d, sink) {
		if v != 3 { // sink = t0 + in = (in+in) + in
			t.Fatalf("element %d is %v, want 3", i, v)
		}
	}
}

// Transients whose users are all ordered against each other do alias. Without
// this the diamond test above would pass against a planner that simply never
// aliases anything.
func TestOrderedTransientsAlias(t *testing.T) {
	const n = 64
	d := openDevice(t)
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "op",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	in := newBuffer(t, d, "in", n, storage)
	out := newBuffer(t, d, "out", n, storage)
	r := d.NewRecorder()
	transient := func(label string) accel.BufferView {
		return r.Transient(accel.BufferDescriptor{
			DType: accel.F32, Count: n, Usage: storage, Label: label,
		})
	}
	// A strict chain: every node depends on the last, so every pair of
	// transients has all its users ordered.
	a, b, c := transient("a"), transient("b"), transient("c")
	count := accel.WorkgroupCount{X: 1}
	dispatch := func(x, y, z accel.BufferView) {
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: x}, {Index: 1, Buffer: y}, {Index: 2, Buffer: z},
		}, count)
	}
	dispatch(whole(t, in), whole(t, in), a)
	dispatch(a, whole(t, in), b)
	dispatch(b, whole(t, in), c)
	dispatch(c, whole(t, in), whole(t, out))

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	m := g.Memory()
	if m.TransientBytes >= m.UnaliasedBytes {
		t.Errorf("a chain of three transients should alias: pool %d, unaliased %d",
			m.TransientBytes, m.UnaliasedBytes)
	}
	// a and c have no user in common and every user of one precedes every user
	// of the other, so they share bytes.
	place := map[string]accel.TransientPlacement{}
	for _, pl := range g.TransientPlacement() {
		place[pl.Label] = pl
	}
	if place["a"].Offset != place["c"].Offset {
		t.Errorf("a is at %d and c at %d; they are fully ordered and should share bytes",
			place["a"].Offset, place["c"].Offset)
	}

	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
}

// A node using both transients makes them incompatible, whatever the
// reachability says about the rest of their users. A relation that counted a
// shared user as ordered would alias the input and output of one node, and
// every chain of dispatches would corrupt itself.
//
// This is spec 003's "t0, t1: no, n2 uses both" row, and it is what the strict
// reachability in the compatibility test buys.
func TestTransientsSharingAUserDoNotAlias(t *testing.T) {
	const n = 64
	d := openDevice(t)
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "op",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	in := newBuffer(t, d, "in", n, storage)
	out := newBuffer(t, d, "out", n, storage)
	r := d.NewRecorder()
	mk := func(label string) accel.BufferView {
		return r.Transient(accel.BufferDescriptor{
			DType: accel.F32, Count: n, Usage: storage, Label: label,
		})
	}
	a, b := mk("a"), mk("b")
	count := accel.WorkgroupCount{X: 1}
	dispatch := func(x, y, z accel.BufferView) {
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: x}, {Index: 1, Buffer: y}, {Index: 2, Buffer: z},
		}, count)
	}
	dispatch(whole(t, in), whole(t, in), a) // n0 writes a
	dispatch(a, whole(t, in), b)            // n1 reads a and writes b
	dispatch(b, whole(t, in), whole(t, out))

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	place := map[string]accel.TransientPlacement{}
	for _, pl := range g.TransientPlacement() {
		place[pl.Label] = pl
	}
	pa, pb := place["a"], place["b"]
	if pa.Offset < pb.Offset+pb.Bytes && pb.Offset < pa.Offset+pa.Bytes {
		t.Fatalf("a at [%d, %d) and b at [%d, %d) share bytes, and node 1 reads a "+
			"while writing b", pa.Offset, pa.Offset+pa.Bytes, pb.Offset, pb.Offset+pb.Bytes)
	}

	// And the result proves it: with aliasing, b's write lands on the bytes the
	// same invocation is reading a from.
	ones := make([]float32, n)
	for i := range ones {
		ones[i] = 1
	}
	if err := d.Queue().WriteBuffer(in, 0, ones); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	for i, v := range readback(t, d, out) {
		if v != 4 { // a=2, b=a+1=3, out=b+1=4
			t.Fatalf("element %d is %v, want 4", i, v)
		}
	}
}

// The peak is a lower bound over the record-order linearization, so it can
// never exceed the pool. When it does, the packer aliased something the
// linearization says is simultaneously live — which is how the shared-user bug
// above was found.
func TestThePeakNeverExceedsThePool(t *testing.T) {
	const n = 64
	d := openDevice(t)
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "op",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	in := newBuffer(t, d, "in", n, storage)
	out := newBuffer(t, d, "out", n, storage)
	for _, chain := range []int{2, 4, 8, 16} {
		r := d.NewRecorder()
		prev := whole(t, in)
		count := accel.WorkgroupCount{X: 1}
		for range chain {
			v := r.Transient(accel.BufferDescriptor{
				DType: accel.F32, Count: n, Usage: storage,
			})
			r.Dispatch(p, []accel.Binding{
				{Index: 0, Buffer: prev}, {Index: 1, Buffer: whole(t, in)}, {Index: 2, Buffer: v},
			}, count)
			prev = v
		}
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: prev}, {Index: 1, Buffer: whole(t, in)},
			{Index: 2, Buffer: whole(t, out)},
		}, count)

		g, err := r.Build()
		if err != nil {
			t.Fatalf("build chain of %d: %v", chain, err)
		}
		m := g.Memory()
		if m.PeakBytes > m.TransientBytes {
			t.Errorf("chain of %d: peak %d exceeds pool %d, so the packer aliased "+
				"transients the record-order linearization says are both live",
				chain, m.PeakBytes, m.TransientBytes)
		}
		if err := g.Close(); err != nil {
			t.Fatalf("close: %v", err)
		}
	}
}

// GraphMemory's three fields separate here, and the gap between the peak and
// the pool is the price of DAG-safe aliasing rather than fragmentation.
func TestMemoryFieldsSeparateUnderAliasing(t *testing.T) {
	const n = 64
	d := openDevice(t)
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "op",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	in := newBuffer(t, d, "in", n, storage)
	out := newBuffer(t, d, "out", n, storage)
	r := d.NewRecorder()
	var views []accel.BufferView
	for range 4 {
		views = append(views, r.Transient(accel.BufferDescriptor{
			DType: accel.F32, Count: n, Usage: storage,
		}))
	}
	count := accel.WorkgroupCount{X: 1}
	prev := whole(t, in)
	for _, v := range views {
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: prev}, {Index: 1, Buffer: whole(t, in)}, {Index: 2, Buffer: v},
		}, count)
		prev = v
	}
	r.Dispatch(p, []accel.Binding{
		{Index: 0, Buffer: prev}, {Index: 1, Buffer: whole(t, in)}, {Index: 2, Buffer: whole(t, out)},
	}, count)

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	m := g.Memory()
	if m.TransientBytes >= m.UnaliasedBytes {
		t.Errorf("aliasing bought nothing: pool %d, unaliased %d",
			m.TransientBytes, m.UnaliasedBytes)
	}
	if m.PeakBytes > m.TransientBytes {
		t.Errorf("the peak %d is a lower bound and cannot exceed the pool %d",
			m.PeakBytes, m.TransientBytes)
	}
	if m.PeakBytes == 0 {
		t.Error("a graph with live transients has a non-zero peak")
	}
}

// V20: a pool larger than the device's budget is rejected, checked against a
// mimicked profile with a small budget so the path does not wait for hardware
// that has one.
func TestV20RejectsAnOversizedPool(t *testing.T) {
	base, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	small := accel.DeviceProfile{Info: base.Info()}
	small.Info.Limits.MaxPoolBytes = 1024
	if err := base.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	d, err := accel.OpenCPU(accel.CPUOptions{Mode: accel.CPUMimic, Mimic: &small})
	if err != nil {
		t.Fatalf("open mimicking: %v", err)
	}
	defer d.Close()

	r := d.NewRecorder()
	dst := newBuffer(t, d, "dst", 4096, accel.UsageStorage|accel.UsageCopyDst)
	v := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: 4096, Usage: accel.UsageStorage | accel.UsageCopySrc,
		Label: "big",
	})
	r.CopyBuffer(whole(t, dst), v)
	_, err = r.Build()
	if err == nil {
		t.Fatal("a pool over the device's budget should be rejected")
	}
	for _, want := range []string{"after aliasing", "pool budget"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the message should say %q, got:\n%v", want, err)
		}
	}
}
