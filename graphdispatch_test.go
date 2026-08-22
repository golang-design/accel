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

func addPipeline(t *testing.T, d *accel.Device) *accel.ComputePipeline {
	t.Helper()
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "add",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })
	return p
}

// M3's end-to-end criterion: a public recorder uploads to a buffer, dispatches
// a flat Add over it, and reads back; the graph is retained, an input rebound,
// and replayed, producing the second input's result with no rebuild.
func TestGraphUploadDispatchReadback(t *testing.T) {
	const n = 256
	d := openDevice(t)
	q := d.Queue()
	p := addPipeline(t, d)

	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst
	first := newBuffer(t, d, "first", n, storage)
	second := newBuffer(t, d, "second", n, storage)
	out := newBuffer(t, d, "out", n, storage)

	ones, twos := make([]float32, n), make([]float32, n)
	for i := range ones {
		ones[i], twos[i] = 1, 2
	}
	if err := q.WriteBuffer(first, 0, ones); err != nil {
		t.Fatalf("write first: %v", err)
	}
	if err := q.WriteBuffer(second, 0, twos); err != nil {
		t.Fatalf("write second: %v", err)
	}

	r := d.NewRecorder()
	// b is uploaded by the graph itself, so the recording exercises a transfer
	// and a dispatch in one plan with a hazard between them.
	uploaded := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: n, Usage: storage, Label: "uploaded",
	})
	tens := make([]float32, n)
	for i := range tens {
		tens[i] = 10
	}
	r.CopyToBuffer(uploaded, tens)

	a := r.Slot(accel.SlotDescriptor{
		Name: "a", Kind: accel.BindingStorageBuffer,
		DType: accel.F32, Access: accel.AccessRead, MinCount: n,
	})
	r.Dispatch(p, []accel.Binding{
		{Index: 0, Slot: a},
		{Index: 1, Buffer: uploaded},
		{Index: 2, Buffer: whole(t, out)},
	}, accel.WorkgroupCount{X: n / 64})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	// The upload writes what the dispatch reads, so there is a hazard and a
	// barrier between them, and none anywhere else.
	if got := g.Hazards(); got != 1 {
		t.Errorf("got %d hazards, want the one between the upload and the dispatch", got)
	}
	if n := g.NodeStats(1); n.BarriersBefore != 1 {
		t.Errorf("the dispatch reads what the upload wrote and needs a barrier, got %d",
			n.BarriersBefore)
	}

	for _, c := range []struct {
		label string
		in    *accel.Buffer
		want  float32
	}{{"first", first, 11}, {"second", second, 12}} {
		if err := g.Bind(accel.Binding{Slot: a, Buffer: whole(t, c.in)}); err != nil {
			t.Fatalf("bind %s: %v", c.label, err)
		}
		if err := q.Submit(g).Wait(); err != nil {
			t.Fatalf("submit %s: %v", c.label, err)
		}
		got := readback(t, d, out)
		for i := range got {
			if got[i] != c.want {
				t.Fatalf("with %s bound, element %d is %v, want %v", c.label, i, got[i], c.want)
			}
		}
	}
}

// A dispatch over a transient: the builder owns the memory, and the kernel
// writes into it and reads it back out through a second dispatch.
func TestADispatchChainThroughATransient(t *testing.T) {
	const n = 128
	d := openDevice(t)
	q := d.Queue()
	p := addPipeline(t, d)

	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst
	in := newBuffer(t, d, "in", n, storage)
	out := newBuffer(t, d, "out", n, storage)
	ones := make([]float32, n)
	for i := range ones {
		ones[i] = 1
	}
	if err := q.WriteBuffer(in, 0, ones); err != nil {
		t.Fatalf("write: %v", err)
	}

	r := d.NewRecorder()
	mid := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: n, Usage: storage, Label: "mid",
	})
	count := accel.WorkgroupCount{X: n / 64}
	r.Dispatch(p, []accel.Binding{
		{Index: 0, Buffer: whole(t, in)},
		{Index: 1, Buffer: whole(t, in)},
		{Index: 2, Buffer: mid},
	}, count) // mid = 2
	r.Dispatch(p, []accel.Binding{
		{Index: 0, Buffer: mid},
		{Index: 1, Buffer: whole(t, in)},
		{Index: 2, Buffer: whole(t, out)},
	}, count) // out = 3

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	for i, v := range readback(t, d, out) {
		if v != 3 {
			t.Fatalf("element %d is %v, want 3", i, v)
		}
	}
}

// Two dispatches writing different buffers from the same read-only input are
// independent, which is the case a whole-resource planner would serialize.
func TestIndependentDispatchesAreNotSeparated(t *testing.T) {
	const n = 64
	d := openDevice(t)
	p := addPipeline(t, d)
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst
	in := newBuffer(t, d, "in", n, storage)

	r := d.NewRecorder()
	for range 3 {
		out := newBuffer(t, d, "out", n, storage)
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: whole(t, in)},
			{Index: 1, Buffer: whole(t, in)},
			{Index: 2, Buffer: whole(t, out)},
		}, accel.WorkgroupCount{X: 1})
	}
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	if got := g.Hazards(); got != 0 {
		t.Errorf("three dispatches reading one buffer and writing three have no hazard, got %d", got)
	}
	if got := g.Barriers(); got != 1 {
		t.Errorf("got %d barriers, want only the head-of-submission one", got)
	}
}

// The dispatch validation rows. Spec 003's V1, V3, V6, V8, V10, and V17.
func TestDispatchValidationRows(t *testing.T) {
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst

	cases := []struct {
		row  string
		what string
		says string
		run  func(t *testing.T, d *accel.Device) error
	}{{
		row:  "V1",
		what: "a layout entry with nothing bound",
		says: `binding "out" at index 2 has no resource bound`,
		run: func(t *testing.T, d *accel.Device) error {
			p := addPipeline(t, d)
			in := newBuffer(t, d, "in", 64, storage)
			r := d.NewRecorder()
			r.Dispatch(p, []accel.Binding{
				{Index: 0, Buffer: whole(t, in)},
				{Index: 1, Buffer: whole(t, in)},
			}, accel.WorkgroupCount{X: 1})
			_, err := r.Build()
			return err
		},
	}, {
		row:  "V1",
		what: "a binding index outside the layout",
		says: "outside the kernel's 3 entries",
		run: func(t *testing.T, d *accel.Device) error {
			p := addPipeline(t, d)
			in := newBuffer(t, d, "in", 64, storage)
			r := d.NewRecorder()
			r.Dispatch(p, []accel.Binding{{Index: 7, Buffer: whole(t, in)}}, accel.WorkgroupCount{X: 1})
			_, err := r.Build()
			return err
		},
	}, {
		row:  "V1",
		what: "one entry bound twice",
		says: "is bound twice",
		run: func(t *testing.T, d *accel.Device) error {
			p := addPipeline(t, d)
			in := newBuffer(t, d, "in", 64, storage)
			r := d.NewRecorder()
			r.Dispatch(p, []accel.Binding{
				{Index: 0, Buffer: whole(t, in)},
				{Index: 0, Buffer: whole(t, in)},
			}, accel.WorkgroupCount{X: 1})
			_, err := r.Build()
			return err
		},
	}, {
		row:  "V3",
		what: "a view whose dtype differs from the layout's",
		says: "is f32 and the view is u32",
		run: func(t *testing.T, d *accel.Device) error {
			p := addPipeline(t, d)
			in := newBuffer(t, d, "in", 64, storage)
			r := d.NewRecorder()
			r.Dispatch(p, []accel.Binding{
				{Index: 0, Buffer: mustViewAs(t, in, accel.U32)},
				{Index: 1, Buffer: whole(t, in)},
				{Index: 2, Buffer: whole(t, in)},
			}, accel.WorkgroupCount{X: 1})
			_, err := r.Build()
			return err
		},
	}, {
		row:  "V6",
		what: "a buffer created without storage usage",
		says: "needs UsageStorage",
		run: func(t *testing.T, d *accel.Device) error {
			p := addPipeline(t, d)
			in := newBuffer(t, d, "in", 64, storage)
			plain := newBuffer(t, d, "plain", 64, accel.UsageCopyDst)
			r := d.NewRecorder()
			r.Dispatch(p, []accel.Binding{
				{Index: 0, Buffer: whole(t, in)},
				{Index: 1, Buffer: whole(t, in)},
				{Index: 2, Buffer: whole(t, plain)},
			}, accel.WorkgroupCount{X: 1})
			_, err := r.Build()
			return err
		},
	}, {
		row:  "V8",
		what: "a workgroup count of zero",
		says: "is a recording mistake rather than a skip",
		run: func(t *testing.T, d *accel.Device) error {
			p := addPipeline(t, d)
			in := newBuffer(t, d, "in", 64, storage)
			r := d.NewRecorder()
			r.Dispatch(p, []accel.Binding{
				{Index: 0, Buffer: whole(t, in)},
				{Index: 1, Buffer: whole(t, in)},
				{Index: 2, Buffer: whole(t, in)},
			}, accel.WorkgroupCount{})
			_, err := r.Build()
			return err
		},
	}, {
		row:  "V8",
		what: "a workgroup count past the device limit",
		says: "workgroup count X is",
		run: func(t *testing.T, d *accel.Device) error {
			p := addPipeline(t, d)
			in := newBuffer(t, d, "in", 64, storage)
			r := d.NewRecorder()
			r.Dispatch(p, []accel.Binding{
				{Index: 0, Buffer: whole(t, in)},
				{Index: 1, Buffer: whole(t, in)},
				{Index: 2, Buffer: whole(t, in)},
			}, accel.WorkgroupCount{X: 1 << 30})
			_, err := r.Build()
			return err
		},
	}, {
		row:  "V19",
		what: "a pipeline from a different device",
		says: "belongs to a different device",
		run: func(t *testing.T, d *accel.Device) error {
			p := addPipeline(t, openDevice(t))
			in := newBuffer(t, d, "in", 64, storage)
			r := d.NewRecorder()
			r.Dispatch(p, []accel.Binding{{Index: 0, Buffer: whole(t, in)}}, accel.WorkgroupCount{X: 1})
			_, err := r.Build()
			return err
		},
	}}

	for _, c := range cases {
		t.Run(c.row+"/"+c.what, func(t *testing.T) {
			d := openDevice(t)
			err := c.run(t, d)
			if err == nil {
				t.Fatalf("%s: expected a rejection", c.row)
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Fatalf("%s: the message should say %q, got:\n%v", c.row, c.says, err)
			}
		})
	}
}

// V10 and V11 are pipeline-creation rows, and they are checked against a
// mimicked profile with small limits so the path does not wait for hardware
// that has them. This is the same reasoning spec 014 used for the uniform
// block limit.
func TestPipelineLimitRows(t *testing.T) {
	small := accel.DeviceProfile{}
	base, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	small.Info = base.Info()
	small.Info.Limits.MaxWorkgroupInvocations = 8
	small.Info.Limits.MaxWorkgroupSize = [3]int{8, 8, 8}
	if err := base.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	d, err := accel.OpenCPU(accel.CPUOptions{Mode: accel.CPUMimic, Mimic: &small})
	if err != nil {
		t.Fatalf("open mimicking: %v", err)
	}
	defer d.Close()

	// Add's workgroup is 64 wide, which is over both limits.
	_, err = d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "add",
	})
	if err == nil {
		t.Fatal("a 64-wide workgroup should be rejected on an 8-wide device")
	}
	for _, want := range []string{"MaxWorkgroupSize[0]", "MaxWorkgroupInvocations", "add"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the message should say %q, got:\n%v", want, err)
		}
	}
}

func TestPipelineRejectsAMalformedDescriptor(t *testing.T) {
	d := openDevice(t)

	if _, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{}); err == nil ||
		!strings.Contains(err.Error(), "names no kernel") {
		t.Errorf("a descriptor with no kernel should be rejected, got %v", err)
	}

	stale := testkernels.AddKernel
	stale.Generator = accel.KernelABIVersion + 1
	if _, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: &stale}); err == nil ||
		!strings.Contains(err.Error(), "re-run go generate") {
		t.Errorf("a kernel from another ABI should be rejected, got %v", err)
	}

	coop := testkernels.AddKernel
	coop.Flat = nil
	if _, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: &coop}); err == nil ||
		!strings.Contains(err.Error(), "arrives at M4") {
		t.Errorf("a cooperative kernel should be rejected naming M4, got %v", err)
	}

	zero := testkernels.AddKernel
	zero.WorkgroupSize = accel.ID3{X: 0, Y: 1, Z: 1}
	if _, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: &zero}); err == nil ||
		!strings.Contains(err.Error(), "dispatches nothing") {
		t.Errorf("a zero workgroup extent should be rejected, got %v", err)
	}
}

func TestADeviceWillNotCloseUnderAPipeline(t *testing.T) {
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: &testkernels.AddKernel})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	if err := d.Close(); err == nil {
		t.Error("closing a device with a live pipeline should fail")
	}
	if err := p.Close(); err != nil {
		t.Fatalf("close pipeline: %v", err)
	}
	if err := p.Close(); err != nil {
		t.Errorf("Close should be idempotent: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Errorf("close device: %v", err)
	}
}
