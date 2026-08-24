// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"testing"

	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/kernelabi"
)

// The stage classification of specs/003-command-graph.md, asserted on the plan.
//
// These are in-package tests because a stage mask is not a public value: it is
// what a barrier names on each side, and nothing on the CPU backend or on Metal
// reads it. That is the whole reason it went eight months classifying every
// render pass as a transfer, so the assertion has to reach the field.

// stageTestVS and stageTestFS are the smallest pair of stages a pipeline
// accepts. Nothing runs them: every test here stops at the plan, and the
// package cannot import internal/testkernels because that package imports this
// one.
var (
	stageTestVS = Stage{
		Name: "stageTestVS", Kind: kernel.StageVertex, Varyings: "stageTestVaryings",
		Attributes: []kernel.StageAttribute{{Name: "pos", Index: 0, Components: 3}},
		RunVertex: func(kernel.Vertex, []any, [][]float32) (kernel.Clip, []float32) {
			return kernel.Clip{}, nil
		},
	}
	stageTestFS = Stage{
		Name: "stageTestFS", Kind: kernel.StageFragment, Varyings: "stageTestVaryings",
		Outputs: []kernel.StageOutput{{Name: "colour", Index: 0}},
		RunFragment: func(kernel.Fragment, []any, []float32) [][4]float32 {
			return nil
		},
	}
)

// stageTestKernel is the smallest kernel a pipeline accepts, for the same
// reason the stages above are hand-written: the corpus lives in a package that
// imports this one.
var stageTestKernel = Kernel{
	Name: "stageTestKernel", WorkgroupSize: kernel.ID3{X: 1, Y: 1, Z: 1},
	Bindings:  []kernel.Binding{{Name: "out", DType: kernel.F32, Access: kernel.Write}},
	Generator: kernelabi.Version,
	Flat:      func(kernel.Thread, kernel.Args) {},
}

func stageTestDevice(t *testing.T) *Device {
	t.Helper()
	d, err := OpenCPU(CPUOptions{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { _ = d.Close() })
	return d
}

func stageTestBuffer(t *testing.T, d *Device, label string, n int) BufferView {
	t.Helper()
	b, err := d.NewBuffer(BufferDescriptor{
		DType: F32, Count: n, Label: label,
		Usage: BufferStorage | BufferCopySrc | BufferCopyDst | BufferIndirect,
	})
	if err != nil {
		t.Fatalf("new buffer %q: %v", label, err)
	}
	t.Cleanup(func() { _ = b.Close() })
	v, err := b.View(0, n)
	if err != nil {
		t.Fatalf("view %q: %v", label, err)
	}
	return v
}

func stageTestPipeline(t *testing.T, d *Device) *RenderPipeline {
	t.Helper()
	p, err := d.NewRenderPipeline(RenderPipelineDescriptor{
		Vertex: &stageTestVS, Fragment: &stageTestFS,
		VertexBuffers: []VertexBufferLayout{{
			Stride: 12, StepMode: StepVertex,
			Attributes: []VertexAttribute{{Location: 0, Format: AttrFloat32x3, Offset: 0}},
		}},
		Targets: []ColorTargetState{{Format: RGBA32Float}},
		Label:   "stage test",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })
	return p
}

// stageOf returns the stage recorded for the access naming v on node id.
func stageOf(t *testing.T, r *Recorder, id NodeID, v BufferView) stage {
	t.Helper()
	var found []stage
	for _, a := range r.state.nodes[id].accesses {
		if a.res.buf == v.Buffer {
			found = append(found, a.stage)
		}
	}
	if len(found) != 1 {
		t.Fatalf("node %d declares %d accesses of %q, want exactly one",
			int(id), len(found), v.Buffer.desc.Label)
	}
	return found[0]
}

// Every access a render pass declares carries the stage that touches it.
//
// This is the whole finding of specs/042-surface-completion.md section 5.3: a
// pass declares five kinds of access and they were one node-wide bit, so all
// five read as a transfer. A vertex fetch, an argument fetch, a colour write
// and a depth test are four different points in a pipeline, and a barrier that
// names the wrong one is placed against a stage the hazard is not in.
func TestEveryRenderPassAccessNamesItsOwnStage(t *testing.T) {
	const w, h = 4, 4
	d := stageTestDevice(t)

	colour := stageTestBuffer(t, d, "colour", w*h*4)
	depth := stageTestBuffer(t, d, "depth", w*h)
	verts := stageTestBuffer(t, d, "vertices", 9)
	index := stageTestBuffer(t, d, "indices", 3)
	args := stageTestBuffer(t, d, "args", 4)

	r := d.NewRecorder()
	p := r.RenderPass(RenderPassDescriptor{
		Color: []ColorAttachment{{View: colour, Load: LoadClear}},
		Depth: &DepthAttachment{View: depth, Load: LoadClear, Clear: 1},
		Width: w, Height: h, Label: "every stage",
	})
	p.SetPipeline(stageTestPipeline(t, d))
	p.SetVertexBuffer(0, verts)
	p.SetIndexBuffer(index, Index32)
	p.DrawIndexed(DrawIndexed{IndexCount: 3})
	p.DrawIndirect(args, Draw{VertexCount: 3})

	for _, c := range []struct {
		what string
		view BufferView
		want stage
		why  string
	}{{
		what: "the colour attachment", view: colour, want: stageColourOutput,
		why: "a colour write happens after the fragment shader, in the blend and " +
			"output stage, which is the stage a barrier against it names",
	}, {
		what: "the depth attachment", view: depth,
		want: stageEarlyDepth | stageLateDepth,
		why: "where the depth test runs depends on the pipeline's discard and " +
			"depth-write behaviour, so a depth barrier names both fragment-test stages",
	}, {
		what: "a vertex buffer", view: verts, want: stageVertexInput,
		why: "attributes are fetched by the vertex input stage, before the vertex " +
			"shader runs, so a write to them must be visible earlier than a shader read",
	}, {
		what: "an index buffer", view: index, want: stageVertexInput,
		why: "an index fetch is vertex input for the same reason an attribute fetch is",
	}, {
		what: "an indirect argument buffer", view: args, want: stageIndirectFetch,
		why: "the arguments are read before the draw is issued, which is its own " +
			"stage on Vulkan and D3D12",
	}} {
		if got := stageOf(t, r, p.Node(), c.view); got != c.want {
			t.Errorf("%s is in the %v stage, want %v: %s", c.what, got, c.want, c.why)
		}
	}
}

// An indirect dispatch reads its count in the fetch stage, not the compute one.
//
// The doc on Recorder.indirectImpl has claimed this since the call was written,
// and the code could not do it: the stage was a property of the node and the
// node is a dispatch, so the count read was classified compute -- exactly the
// case the doc says would be a hazard the barrier plan does not see.
func TestAnIndirectCountIsFetchedRatherThanComputed(t *testing.T) {
	d := stageTestDevice(t)
	count, err := d.NewBuffer(BufferDescriptor{
		DType: U32, Count: 3, Usage: BufferIndirect | BufferCopyDst, Label: "count",
	})
	if err != nil {
		t.Fatalf("count buffer: %v", err)
	}
	t.Cleanup(func() { _ = count.Close() })
	cv, err := count.View(0, 3)
	if err != nil {
		t.Fatalf("count view: %v", err)
	}

	pipe, err := d.NewComputePipeline(ComputePipelineDescriptor{Kernel: &stageTestKernel})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Cleanup(func() { _ = pipe.Close() })
	out := stageTestBuffer(t, d, "out", 4)

	r := d.NewRecorder()
	id := r.DispatchIndirect(pipe, []Binding{{Index: 0, Buffer: out}}, nil, cv,
		WorkgroupCount{X: 1, Y: 1, Z: 1})

	if got := stageOf(t, r, id, cv); got != stageIndirectFetch {
		t.Errorf("the count is read in the %v stage, want %v: the fetch happens "+
			"before the dispatch, so a node writing the count is ordered against the "+
			"fetch rather than against the kernel", got, stageIndirectFetch)
	}
	if got := stageOf(t, r, id, out); got != stageCompute {
		t.Errorf("the kernel's binding is in the %v stage, want %v", got, stageCompute)
	}
	if want := stageIndirectFetch | stageCompute; r.state.nodes[id].stage != want {
		t.Errorf("the node's stage is %v, want the union of its accesses %v",
			r.state.nodes[id].stage, want)
	}
}

// One range fetched in two roles is one access whose mask names both.
//
// The mask exists for this. Splitting on stage instead would make a buffer
// declared twice two accesses, which is the duplicate declareRead exists to
// prevent: the hazard count a caller reads would double against one earlier
// writer, and the access index build consumes positionally would shift.
func TestOneRangeFetchedInTwoRolesIsOneAccess(t *testing.T) {
	const w, h = 4, 4
	d := stageTestDevice(t)

	colour := stageTestBuffer(t, d, "colour", w*h*4)
	// Nine float32 is thirty-six bytes: three vertices of a float32x3 attribute
	// at stride 12, and more than the sixteen an argument buffer needs. So one
	// range is legal in both roles and the pass declares it in both.
	shared := stageTestBuffer(t, d, "shared", 9)

	r := d.NewRecorder()
	r.UploadToBuffer(shared, make([]float32, 9))
	p := r.RenderPass(RenderPassDescriptor{
		Color: []ColorAttachment{{View: colour, Load: LoadClear}},
		Width: w, Height: h, Label: "two roles",
	})
	p.SetPipeline(stageTestPipeline(t, d))
	p.SetVertexBuffer(0, shared)
	p.DrawIndirect(shared, Draw{VertexCount: 3})

	if got := stageOf(t, r, p.Node(), shared); got != stageIndirectFetch|stageVertexInput {
		t.Errorf("the shared range is in the %v stage, want %v: one range in two "+
			"roles is one access naming both, which is what makes this a mask",
			got, stageIndirectFetch|stageVertexInput)
	}

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer func() { _ = g.Close() }()

	if got := g.Hazards(); got != 1 {
		t.Errorf("got %d hazards, want the one read-after-write on the shared range; "+
			"a duplicated access counts its hazard against the upload twice", got)
	}
}

// Every bit of the mask prints its own name, and a set of them reads as a set.
func TestStageNamesEveryBitItHolds(t *testing.T) {
	for _, c := range []struct {
		s    stage
		want string
	}{
		{0, "none"},
		{stageTransfer, "transfer"},
		{stageIndirectFetch, "indirect fetch"},
		{stageVertexInput, "vertex input"},
		{stageEarlyDepth, "early depth"},
		{stageLateDepth, "late depth"},
		{stageColourOutput, "colour output"},
		{stageCompute, "compute"},
		{stageEarlyDepth | stageLateDepth, "early depth and late depth"},
		{stageTransfer | stageCompute, "transfer and compute"},
		{stageAll, "transfer and indirect fetch and vertex input and early depth " +
			"and late depth and colour output and compute"},
	} {
		if got := c.s.String(); got != c.want {
			t.Errorf("stage(%d) prints %q, want %q", uint8(c.s), got, c.want)
		}
	}
}

// A barrier names the stage of the access that needs it, not the whole node's.
//
// A pass that waits for a vertex buffer waits at vertex input. Naming the pass
// instead names the colour and depth stages too, which stalls two stages on a
// hazard neither one has -- and on a backend with real stage masks that is a
// pipeline drain per frame, not a lost optimization.
func TestABarrierNamesTheStageOfTheAccessThatNeedsIt(t *testing.T) {
	const w, h = 4, 4
	d := stageTestDevice(t)

	colour := stageTestBuffer(t, d, "colour", w*h*4)
	verts := stageTestBuffer(t, d, "vertices", 9)

	r := d.NewRecorder()
	// Node 0 writes the vertices. Node 1 is the pass, whose only hazard is the
	// read-after-write on them: its attachment is cleared and nothing wrote it.
	r.UploadToBuffer(verts, make([]float32, 9))
	p := r.RenderPass(RenderPassDescriptor{
		Color: []ColorAttachment{{View: colour, Load: LoadClear}},
		Width: w, Height: h, Label: "waits at vertex input",
	})
	p.SetPipeline(stageTestPipeline(t, d))
	p.SetVertexBuffer(0, verts)
	p.Draw(Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer func() { _ = g.Close() }()

	b := g.barriersBefore[p.Node()]
	if b == nil {
		t.Fatal("no barrier before the pass, and it reads what node 0 wrote")
	}
	if b.src != stageTransfer {
		t.Errorf("the barrier's source is %v, want %v: an upload is a staged blit", b.src, stageTransfer)
	}
	if b.dst != stageVertexInput {
		t.Errorf("the barrier's destination is %v, want %v: the pass waits where it "+
			"fetches attributes, and its colour stage has no hazard to wait for",
			b.dst, stageVertexInput)
	}
}

// The head-of-submission barrier covers every stage.
//
// It stands for the previous submission, whose work this one knows nothing
// about, so anything narrower is a hole: a graphics stage left out of it would
// be one this submission never waits for and no test on either current backend
// could see.
func TestTheHeadOfSubmissionBarrierCoversEveryStage(t *testing.T) {
	d := stageTestDevice(t)
	src := stageTestBuffer(t, d, "src", 4)
	dst := stageTestBuffer(t, d, "dst", 4)

	r := d.NewRecorder()
	r.CopyBuffer(dst, src)
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer func() { _ = g.Close() }()

	b := g.barriersBefore[0]
	if b == nil {
		t.Fatal("no head-of-submission barrier")
	}
	if b.src != stageAll || b.dst != stageAll {
		t.Errorf("the head barrier orders %v against %v, want %v on both sides",
			b.src, b.dst, stageAll)
	}
}
