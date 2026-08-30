// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// specs/035-cpu-rasterizer.md section 7's "depth readback through a transfer
// node" entry.
//
// # The constraint it covers
//
// A depth texture is device-private on macOS, so accel refuses Queue.ReadTexture
// for every depth format rather than offering a path that works on one platform.
// The supported route is a recorded copy: a transfer node inside the graph puts
// the bytes in a buffer, and a buffer is readable anywhere. Nothing verified
// that the route exists.
//
// # What it asserts
//
// A full-screen triangle at a known clip depth, and the buffer holding the
// window depth the convention says it should -- z_window = (z_ndc + 1)/2, so
// clip z = 0 at w = 1 is 0.5. Exact: the depth write is a store, not an
// interpolation of anything the two backends may weight differently.
func TestADepthAttachmentIsReadBackThroughATransferNode(t *testing.T) {
	const w, h = 8, 8
	// The clip depth the triangle is drawn at, and the window depth it becomes.
	// Neither is the depth clear of 1, so a buffer that was never written by
	// the pass fails rather than matching by accident.
	const clipZ, wantWindow = 0.25, 0.625

	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer d.Close()
	checkDepthReadbackThroughATransfer(t, d, w, h, clipZ, wantWindow)
}

// checkDepthReadbackThroughATransfer runs the entry on one device.
//
// Shared with the Metal case rather than duplicated. Both backends lower a
// texture copy, so neither skips -- the guard written for that possibility was
// removed rather than left in, since a skip that can never fire is a comparison
// nobody notices is not happening.
func checkDepthReadbackThroughATransfer(t *testing.T, d *accel.Device, w, h int, clipZ, wantWindow float32) {
	t.Helper()
	q := d.Queue()

	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.ScaledVSStage,
		Fragment: &testkernels.SolidFSStage,
		VertexBuffers: []accel.VertexBufferLayout{{
			Stride:     12,
			Attributes: []accel.VertexAttribute{{Location: 0, Format: accel.AttrFloat32x3}},
		}},
		Targets: []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		DepthStencil: &accel.DepthStencilState{
			Format: accel.Depth32Float, Test: true, Write: true,
			Compare: accel.CompareLess,
		},
		Label: "depth",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	verts := []float32{-1, -1, clipZ, 3, -1, clipZ, -1, 3, clipZ}
	vb, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: len(verts),
		Usage: accel.BufferVertex | accel.BufferCopyDst, Label: "verts",
	})
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer vb.Close()
	if err := q.WriteBuffer(vb, 0, verts); err != nil {
		t.Fatalf("write: %v", err)
	}
	vv, err := vb.View(0, vb.Count())
	if err != nil {
		t.Fatalf("view: %v", err)
	}

	colour, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.RGBA32Float, Size: accel.Extent{Width: w, Height: h},
		Usage: accel.TextureRenderTarget | accel.TextureCopySrc | accel.TextureCopyDst,
		Kind:  accel.MemoryDevice, Label: "colour",
	})
	if err != nil {
		t.Fatalf("colour: %v", err)
	}
	defer colour.Close()
	depth, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.Depth32Float, Size: accel.Extent{Width: w, Height: h},
		Usage: accel.TextureRenderTarget | accel.TextureCopySrc | accel.TextureCopyDst,
		Kind:  accel.MemoryDevice, Label: "depth",
	})
	if err != nil {
		t.Fatalf("depth: %v", err)
	}
	defer depth.Close()

	// The direct path is refused, and that refusal is the reason this entry
	// exists. Asserted here so the transfer route is shown to be the *only*
	// one rather than merely one that works.
	scratch := make([]byte, w*h*4)
	if err := q.ReadTexture(depth, scratch); err == nil {
		t.Error("ReadTexture accepted a depth texture; the format is device-private on " +
			"at least one backend and the API refuses to offer a path that works on one")
	} else if !strings.Contains(err.Error(), "depth") {
		t.Errorf("the refusal does not mention depth: %v", err)
	}

	out, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: w * h,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
		Label: "depth bytes",
	})
	if err != nil {
		t.Fatalf("out: %v", err)
	}
	defer out.Close()
	ov, err := out.View(0, out.Count())
	if err != nil {
		t.Fatalf("out view: %v", err)
	}

	cv, err := colour.Whole()
	if err != nil {
		t.Fatalf("colour view: %v", err)
	}
	dv, err := depth.Whole()
	if err != nil {
		t.Fatalf("depth view: %v", err)
	}

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: cv, Load: accel.LoadClear}},
		Depth: &accel.DepthAttachment{View: dv, Load: accel.LoadClear, Clear: 1},
		Width: w, Height: h, Label: "depth",
	})
	p.SetPipeline(pipe)
	p.SetVertexBuffer(0, vv)
	p.SetVertexUniform(0, testkernels.StageTransform{Scale: 1, Offset: accel.Vec2{0, 0}})
	p.Draw(accel.Draw{VertexCount: 3})
	r.CopyTextureToBuffer(ov, depth)

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	// The route is a transfer node, and the graph must say so.
	var copies int
	for _, n := range g.Nodes() {
		if n.Kind == accel.NodeCopyTextureToBuffer {
			copies++
		}
	}
	if copies != 1 {
		t.Fatalf("the graph has %d texture-to-buffer copies, want 1", copies)
	}
	// And it must be ordered after the pass that wrote the depth.
	if g.Hazards() == 0 {
		t.Error("the copy reads what the pass wrote and the graph found no hazard")
	}

	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, w*h)
	if err := q.ReadBuffer(out, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}
	for i, z := range got {
		if z != wantWindow {
			t.Fatalf("pixel %d has depth %v, want %v: window depth is (z_ndc+1)/2 and "+
				"the triangle was drawn at clip z %v", i, z, wantWindow, clipZ)
		}
	}
}
