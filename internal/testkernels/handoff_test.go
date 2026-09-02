// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// specs/035-cpu-rasterizer.md section 7's "handoff stays on device" entry.
//
// The predecessor's failure it names: a G-buffer that went to the host and back
// every frame. That is not a wrong picture -- the images are identical either
// way -- so no value comparison can see it, and this asserts on the graph's
// *nodes* instead. It is the entry section 7 marks "exact, structural", and it
// says in the same row that it is not sufficient alone, which is why the origin
// entry checks values.
//
// # Why the attachment is a slot
//
// A compute kernel cannot read a texture: specs/032-stage-abi.md section 5.1
// refuses one by name, because a dispatch's argument set cannot carry a
// texture binding. So the handoff a caller can actually build today is a pass
// writing into a buffer-backed attachment and a dispatch reading those bytes,
// which is the shape this asserts on.
func TestADeferredHandoffStaysOnDevice(t *testing.T) {
	const w, h = 8, 8
	t.Run("cpu", func(t *testing.T) {
		d, err := accel.OpenCPU(accel.CPUOptions{})
		if err != nil {
			t.Fatalf("OpenCPU: %v", err)
		}
		defer d.Close()
		checkHandoffStaysOnDevice(t, d, w, h)
	})
}

// checkHandoffStaysOnDevice builds the deferred graph on one device and asserts
// on its nodes and its values.
func checkHandoffStaysOnDevice(t *testing.T, d *accel.Device, w, h int) {
	n := w * h * 4
	{
		t.Helper()
		q := d.Queue()

		// RowFS rather than SolidFS: a G-buffer of one constant is the same
		// bytes in any row order, so a handoff that transposed or flipped
		// the attachment would still tonemap to the right answer.
		geometry, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
			Vertex:   &testkernels.FullScreenVSStage,
			Fragment: &testkernels.RowFSStage,
			Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
			Label:    "geometry",
		})
		if err != nil {
			t.Fatalf("geometry pipeline: %v", err)
		}
		defer geometry.Close()

		tonemap, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
			Kernel: &testkernels.ScaleKernel, Label: "tonemap",
		})
		if err != nil {
			t.Fatalf("tonemap pipeline: %v", err)
		}
		defer tonemap.Close()

		storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
		gbuf, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: n, Usage: storage, Label: "gbuffer",
		})
		if err != nil {
			t.Fatalf("gbuffer: %v", err)
		}
		defer gbuf.Close()
		toned, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: n, Usage: storage, Label: "toned",
		})
		if err != nil {
			t.Fatalf("toned: %v", err)
		}
		defer toned.Close()

		gv, err := gbuf.View(0, gbuf.Count())
		if err != nil {
			t.Fatalf("gbuffer view: %v", err)
		}
		tv, err := toned.View(0, toned.Count())
		if err != nil {
			t.Fatalf("toned view: %v", err)
		}

		r := d.NewRecorder()
		slot := r.Slot(accel.SlotDescriptor{
			Name: "gbuffer", Kind: accel.BindingStorageBuffer,
			DType: accel.F32, Access: accel.AccessReadWrite, MinCount: n,
		})
		p := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{Slot: slot, Load: accel.LoadClear}},
			Width: w, Height: h, Label: "geometry",
		})
		p.SetPipeline(geometry)
		p.Draw(accel.Draw{VertexCount: 3})

		r.Dispatch(tonemap, []accel.Binding{
			{Index: 0, Slot: slot},
			{Index: 1, Buffer: tv},
		}, nil, tonemap.Workgroups(n))

		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		if err := g.Bind(accel.SlotBinding{Slot: slot, Buffer: gv}); err != nil {
			t.Fatalf("bind: %v", err)
		}

		// The structural assertion. Two nodes, a pass and a dispatch, and no
		// transfer of any kind between them: the tonemap reads what the pass
		// wrote, where it wrote it.
		nodes := g.Nodes()
		if len(nodes) != 2 {
			t.Fatalf("the graph has %d nodes, want 2: a pass and a dispatch", len(nodes))
		}
		if nodes[0].Kind != accel.NodeRenderPass || nodes[1].Kind != accel.NodeDispatch {
			t.Fatalf("the nodes are %v then %v, want a render pass then a dispatch",
				nodes[0].Kind, nodes[1].Kind)
		}
		for _, node := range nodes {
			switch node.Kind {
			case accel.NodeCopyBuffer, accel.NodeCopyTextureToBuffer,
				accel.NodeCopyBufferToTexture:
				t.Errorf("node %q is a %v: the handoff went through a transfer",
					node.Label, node.Kind)
			}
		}
		// And the graph knows the dispatch depends on the pass. Without the
		// edge the two could run in either order and the picture would be
		// right by luck on a backend that happens to serialize.
		if g.Hazards() == 0 {
			t.Error("the dispatch reads what the pass wrote and the graph found no hazard")
		}

		if err := q.Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}

		// The values, so "no transfer" is not satisfied by a graph that did
		// nothing. RowFS writes (x+0.5, y+0.5, 0, 1) at pixel (x, y) in
		// top-origin row order, and Scale doubles.
		got := make([]float32, n)
		if err := q.ReadBuffer(toned, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}
		for p := range w * h {
			x, y := p%w, p/w
			want := [4]float32{2 * (float32(x) + 0.5), 2 * (float32(y) + 0.5), 0, 2}
			if [4]float32(got[p*4:p*4+4]) != want {
				t.Fatalf("pixel (%d,%d) is %v, want %v: the tonemap did not read the "+
					"pass's output where the pass wrote it", x, y, got[p*4:p*4+4], want)
			}
		}
	}

}
