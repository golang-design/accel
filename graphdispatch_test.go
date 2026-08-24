// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"golang.design/x/accel/kernelabi"
	"math"
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

	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
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
	r.UploadToBuffer(uploaded, tens)

	a := r.Slot(accel.SlotDescriptor{
		Name: "a", Kind: accel.BindingStorageBuffer,
		DType: accel.F32, Access: accel.AccessRead, MinCount: n,
	})
	r.Dispatch(p, []accel.Binding{
		{Index: 0, Slot: a},
		{Index: 1, Buffer: uploaded},
		{Index: 2, Buffer: whole(t, out)},
	}, nil, accel.WorkgroupCount{X: n / 64})

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
		if err := g.Bind(accel.SlotBinding{Slot: a, Buffer: whole(t, c.in)}); err != nil {
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

	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
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
	}, nil, count) // mid = 2
	r.Dispatch(p, []accel.Binding{
		{Index: 0, Buffer: mid},
		{Index: 1, Buffer: whole(t, in)},
		{Index: 2, Buffer: whole(t, out)},
	}, nil, count) // out = 3

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
	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	in := newBuffer(t, d, "in", n, storage)

	r := d.NewRecorder()
	for range 3 {
		out := newBuffer(t, d, "out", n, storage)
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: whole(t, in)},
			{Index: 1, Buffer: whole(t, in)},
			{Index: 2, Buffer: whole(t, out)},
		}, nil, accel.WorkgroupCount{X: 1})
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
	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst

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
			}, nil, accel.WorkgroupCount{X: 1})
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
			r.Dispatch(p, []accel.Binding{{Index: 7, Buffer: whole(t, in)}}, nil, accel.WorkgroupCount{X: 1})
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
			}, nil, accel.WorkgroupCount{X: 1})
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
			}, nil, accel.WorkgroupCount{X: 1})
			_, err := r.Build()
			return err
		},
	}, {
		row:  "V6",
		what: "a buffer created without storage usage",
		says: "needs BufferStorage",
		run: func(t *testing.T, d *accel.Device) error {
			p := addPipeline(t, d)
			in := newBuffer(t, d, "in", 64, storage)
			plain := newBuffer(t, d, "plain", 64, accel.BufferCopyDst)
			r := d.NewRecorder()
			r.Dispatch(p, []accel.Binding{
				{Index: 0, Buffer: whole(t, in)},
				{Index: 1, Buffer: whole(t, in)},
				{Index: 2, Buffer: whole(t, plain)},
			}, nil, accel.WorkgroupCount{X: 1})
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
			}, nil, accel.WorkgroupCount{})
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
			}, nil, accel.WorkgroupCount{X: 1 << 30})
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
			r.Dispatch(p, []accel.Binding{{Index: 0, Buffer: whole(t, in)}}, nil, accel.WorkgroupCount{X: 1})
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
	stale.Generator = kernelabi.Version + 1
	if _, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: &stale}); err == nil ||
		!strings.Contains(err.Error(), "re-run go generate") {
		t.Errorf("a kernel from another ABI should be rejected, got %v", err)
	}

	// A record with neither entry point is an incomplete generated file, not a
	// cooperative kernel: a kernel has exactly one, chosen by whether its body
	// reaches a barrier, shared memory, or a subgroup operation.
	neither := testkernels.AddKernel
	neither.Flat = nil
	if _, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: &neither}); err == nil ||
		!strings.Contains(err.Error(), "re-run go generate") {
		t.Errorf("a record with no entry point should say the file is incomplete, got %v", err)
	}

	// And a cooperative one is accepted, since the resumable lowering exists.
	if p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.ExchangeKernel,
	}); err != nil {
		t.Errorf("a cooperative kernel should be accepted: %v", err)
	} else if err := p.Close(); err != nil {
		t.Errorf("close: %v", err)
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

// A kernel requiring a capability the device lacks is refused at pipeline
// creation, naming the capability and the device.
//
// Checked against a mimicked profile rather than against hardware, which is the
// same reasoning spec 014 used for the uniform block limit: a rule that waits
// for a device nobody here owns is a rule nobody runs. Spec 000's decision 6 is
// what this implements — an absent feature is a typed answer before anything is
// dispatched, not a failure at dispatch time.
func TestAKernelRequiringAnAbsentCapabilityIsRefused(t *testing.T) {
	base, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	profile := accel.DeviceProfile{Info: base.Info()}
	if err := base.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	// A device with no subgroups at all, which several real ones are.
	profile.Info.Capabilities.Subgroups = false
	profile.Info.Capabilities.SubgroupOps = 0

	d, err := accel.OpenCPU(accel.CPUOptions{Mode: accel.CPUMimic, Mimic: &profile})
	if err != nil {
		t.Fatalf("open mimicking: %v", err)
	}
	defer d.Close()

	_, err = d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.SubgroupReduceKernel, Label: "reduce",
	})
	if err == nil {
		t.Fatal("a kernel reducing across lanes should be refused on a device with no " +
			"subgroups")
	}
	for _, want := range []string{"reduce", "SubgroupReduce", "capability"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the message should say %q, got:\n%v", want, err)
		}
	}

	// And the fallback, which requires nothing, is accepted on the same device.
	// Without this the refusal above would be passing against a device that
	// refuses everything.
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.SubgroupReduceFallbackKernel, Label: "fallback",
	})
	if err != nil {
		t.Fatalf("the fallback requires no capability and should be accepted: %v", err)
	}
	if err := p.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}

// The capability the device does have is accepted, so the refusal is about the
// capability rather than about subgroups generally.
func TestAKernelRequiringAPresentCapabilityIsAccepted(t *testing.T) {
	d := openDevice(t)
	if !d.Info().Capabilities.Subgroups {
		t.Skip("the CPU backend's default profile reports no subgroups")
	}
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.SubgroupReduceKernel, Label: "reduce",
	})
	if err != nil {
		t.Fatalf("the CPU backend emulates subgroups and should accept this: %v", err)
	}
	if err := p.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
}

// M5's end-to-end criterion: a public graph runs upload → tiled GEMM →
// readback, in strict mode.
//
// Strict mode is the point of the "in strict mode" clause: it is the profile
// that reports the intersection of what every target allows, so a kernel that
// runs there runs anywhere. It also turns the instrumentation off, which means
// this is the path a caller actually gets rather than the checked one.
func TestGraphRunsTheTiledGEMMInStrictMode(t *testing.T) {
	const m, n, k = 17, 19, 23 // no dimension a multiple of any tile dimension

	d, err := accel.OpenCPU(accel.CPUOptions{
		Mode:          accel.CPUStrict,
		StrictTargets: []accel.Backend{accel.BackendMetal},
	})
	if err != nil {
		t.Fatalf("open strict: %v", err)
	}
	defer d.Close()

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.MatMulTiledKernel, Label: "gemm",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	mk := func(label string, count int, dt accel.DType) *accel.Buffer {
		b, err := d.NewBuffer(accel.BufferDescriptor{
			DType: dt, Count: count, Usage: storage, Label: label,
		})
		if err != nil {
			t.Fatalf("buffer %q: %v", label, err)
		}
		t.Cleanup(func() { _ = b.Close() })
		return b
	}
	aBuf := mk("a", m*k, accel.F16)
	bBuf := mk("b", k*n, accel.F16)
	outBuf := mk("out", m*n, accel.F32)

	// The inputs go up as f16 bits, which is what the format is on the wire.
	aBits := make([]uint16, m*k)
	bBits := make([]uint16, k*n)
	for i := range aBits {
		aBits[i] = accel.ToFloat16(float32(i%7) - 3).Bits()
	}
	for i := range bBits {
		bBits[i] = accel.ToFloat16(float32(i%5) - 2).Bits()
	}
	if err := d.Queue().WriteBuffer(aBuf, 0, aBits); err != nil {
		t.Fatalf("upload a: %v", err)
	}
	if err := d.Queue().WriteBuffer(bBuf, 0, bBits); err != nil {
		t.Fatalf("upload b: %v", err)
	}

	r := d.NewRecorder()
	r.Dispatch(p, []accel.Binding{
		{Index: 0, Buffer: whole(t, aBuf)},
		{Index: 1, Buffer: whole(t, bBuf)},
		{Index: 2, Buffer: whole(t, outBuf)},
	}, []accel.UniformValue{{Index: 0, Value: testkernels.GEMMDims{M: m, N: n, K: k}}}, accel.WorkgroupCount{
		X: (n + testkernels.TileN - 1) / testkernels.TileN,
		Y: (m + testkernels.TileM - 1) / testkernels.TileM,
	})
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	got := make([]float32, m*n)
	if err := d.Queue().ReadBuffer(outBuf, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}

	// Against the same independent reference the kernel tests use.
	for i := range got {
		row, col := i/n, i%n
		var want float64
		for kk := range k {
			want += float64(accel.Float16FromBits(aBits[row*k+kk]).F32()) *
				float64(accel.Float16FromBits(bBits[kk*n+col]).F32())
		}
		if diff := math.Abs(float64(got[i]) - want); diff > 1e-3 {
			t.Fatalf("element (%d,%d) is %v, want about %v", row, col, got[i], want)
		}
	}
}

// Persistent state mutated by a kernel is tracked by the graph's hazards, which
// is the precondition M7's whole risk row rests on.
//
// Spec 009 says: "Tensor state mutation escapes graph hazards | M7
// versioned-state negatives and prefill/decode parity | Fix State lowering;
// never add an untracked in-place escape hatch." The question that decides
// whether a tensor State needs new machinery or can route through what exists
// is answerable now, with two kernels that read and write one buffer.
//
// It can: a scatter followed by a gather over the same state produces a
// read-after-write edge and a barrier, because a dispatch's accesses come from
// the kernel's binding layout and the compiler inferred them from the body. No
// escape hatch is needed, and this test is what would notice one being added.
func TestKernelMutatedStateIsTrackedByTheGraph(t *testing.T) {
	d := openDevice(t)
	scatter, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.ScatterRowsKernel, Label: "scatter"})
	if err != nil {
		t.Fatalf("scatter: %v", err)
	}
	defer scatter.Close()
	gather, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.GatherRowsKernel, Label: "gather"})
	if err != nil {
		t.Fatalf("gather: %v", err)
	}
	defer gather.Close()

	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	const width, capacity, count = 4, 8, 4
	state := newBuffer(t, d, "state", capacity*width, storage)
	rows := newBuffer(t, d, "rows", count*width, storage)
	out := newBuffer(t, d, "out", count*width, storage)
	idsBuf, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.U32, Count: count, Usage: storage, Label: "ids"})
	if err != nil {
		t.Fatalf("ids: %v", err)
	}
	defer idsBuf.Close()
	ids, err := idsBuf.View(0, count)
	if err != nil {
		t.Fatalf("view: %v", err)
	}

	p := testkernels.RowParams{Rows: count, Width: width, Capacity: capacity}
	r := d.NewRecorder()
	r.Dispatch(scatter, []accel.Binding{
		{Index: 0, Buffer: whole(t, rows)},
		{Index: 1, Buffer: ids},
		{Index: 2, Buffer: whole(t, state)}, // written
	}, []accel.UniformValue{{Index: 0, Value: p}}, accel.WorkgroupCount{X: 1})
	r.Dispatch(gather, []accel.Binding{
		{Index: 0, Buffer: whole(t, state)}, // read
		{Index: 1, Buffer: ids},
		{Index: 2, Buffer: whole(t, out)},
	}, []accel.UniformValue{{Index: 0, Value: p}}, accel.WorkgroupCount{X: 1})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	if g.Hazards() != 1 {
		t.Errorf("got %d hazards, want the read-after-write on the state buffer",
			g.Hazards())
	}
	if e := g.Edges()[0]; len(e) != 1 || e[0] != 1 {
		t.Errorf("want edge 0 -> 1, got %v: the gather reads what the scatter wrote", e)
	}
	if n := g.NodeStats(1); n.BarriersBefore != 1 {
		t.Errorf("the gather needs a barrier after the scatter, got %d", n.BarriersBefore)
	}

	// And it runs, producing what was scattered. A hazard the plan records and
	// the execution ignores would be worse than no hazard at all.
	vals := make([]float32, count*width)
	for i := range vals {
		vals[i] = float32(i) + 1
	}
	if err := d.Queue().WriteBuffer(rows, 0, vals); err != nil {
		t.Fatalf("write rows: %v", err)
	}
	if err := d.Queue().WriteBuffer(idsBuf, 0, []uint32{3, 1, 6, 0}); err != nil {
		t.Fatalf("write ids: %v", err)
	}
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := readback(t, d, out)
	for i := range got {
		if got[i] != vals[i] {
			t.Fatalf("element %d came back as %v, want the %v that was scattered in",
				i, got[i], vals[i])
		}
	}
}

// A by-value parameter can be replaced between submissions, and the next
// submission computes with the new one.
//
// This is what lets a plan carry a value that varies every step -- a softmax
// scale, a RoPE base, a sequence length -- without rebuilding the graph. The
// line specs/007-tensor-layer.md draws is structural: a value that changes the
// *shape* of the work still needs another plan, because the barriers and the
// transient layout were computed from it.
//
// The test submits twice with different factors and checks the second result,
// because a SetUniform that wrote somewhere the plan does not read would pass
// any test that submitted once.
func TestSetUniformChangesWhatTheNextSubmissionComputes(t *testing.T) {
	const n = 64
	d := openDevice(t)
	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.ElemScaleKernel, Label: "scale",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	in := newBuffer(t, d, "in", n, storage)
	out := newBuffer(t, d, "out", n, storage)
	vals := make([]float32, n)
	for i := range vals {
		vals[i] = float32(i) + 1
	}
	if err := d.Queue().WriteBuffer(in, 0, vals); err != nil {
		t.Fatalf("write: %v", err)
	}

	r := d.NewRecorder()
	node := r.Dispatch(p, []accel.Binding{
		{Index: 0, Buffer: whole(t, in)},
		{Index: 1, Buffer: whole(t, out)},
	}, []accel.UniformValue{{Index: 0, Value: testkernels.ScaleParams{Factor: 2}}}, accel.WorkgroupCount{X: 1})
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	submit := func(want float32) {
		t.Helper()
		if err := d.Queue().Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		got := readback(t, d, out)
		for i := range got {
			if got[i] != vals[i]*want {
				t.Fatalf("element %d is %v, want %v at a factor of %v",
					i, got[i], vals[i]*want, want)
			}
		}
	}
	submit(2)
	if err := g.SetUniform(node, 0, testkernels.ScaleParams{Factor: 5}); err != nil {
		t.Fatalf("SetUniform: %v", err)
	}
	submit(5)

	// The refusals. Each is a way to write a value the plan would not read, or
	// would read as something else.
	for _, tc := range []struct {
		name string
		call func() error
		want string
	}{{
		name: "a node that is not a dispatch",
		call: func() error { return g.SetUniform(node+1, 0, testkernels.ScaleParams{}) },
		want: "node 1 of 1",
	}, {
		name: "a parameter index the kernel does not have",
		call: func() error { return g.SetUniform(node, 3, testkernels.ScaleParams{}) },
		want: "takes 1 by-value parameters",
	}, {
		name: "a different type of the same shape",
		call: func() error {
			type lookalike struct{ Factor float32 }
			return g.SetUniform(node, 0, lookalike{Factor: 5})
		},
		// The one that matters: this encodes identically today and diverges the
		// first time either type gains a field, so the check is on the type.
		want: "would encode identically",
	}} {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.call()
			if err == nil {
				t.Fatalf("%s was accepted", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("the error should say %q, got %v", tc.want, err)
			}
		})
	}
}
