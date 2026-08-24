// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// The load action decides the edge, and it is asserted on the graph rather than
// on an image.
//
// specs/033-render-api.md section 7. An image cannot tell the two apart: a pass
// that keeps and a pass that does not care both leave the previous contents in
// memory, because not caring is permission to leave them, not an instruction to
// change them. What differs is the edge, and the edge is what lets the builder
// alias the memory out from under the previous writer.
func TestTheLoadActionDecidesTheReadAfterWriteEdge(t *testing.T) {
	const w, h = 4, 4
	for _, c := range []struct {
		name     string
		load     accel.LoadOp
		hazards  int
		barriers int
		why      string
	}{{
		name: "Keep reads what the previous node wrote", load: accel.LoadKeep,
		hazards: 2, barriers: 1,
		why: "keeping is a read and a write, so it is both a read-after-write and a " +
			"write-after-write against the upload",
	}, {
		name: "Clear does not read", load: accel.LoadClear,
		hazards: 1, barriers: 1,
		why: "a clear overwrites, which is still a write-after-write and still ordered",
	}, {
		name: "DontCare does not read", load: accel.LoadDontCare,
		hazards: 1, barriers: 1,
		why: "not caring removes the read, and the write-after-write remains",
	}} {
		t.Run(c.name, func(t *testing.T) {
			d := openDevice(t)
			target := newBuffer(t, d, "target", w*h*4,
				accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)

			r := d.NewRecorder()
			// Node 0 writes the attachment. Node 1 is the pass.
			r.UploadToBuffer(whole(t, target), make([]float32, w*h*4))
			pass := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: whole(t, target), Load: c.load}},
				Width: w, Height: h, Label: "pass",
			})
			pass.SetPipeline(solidPipeline(t, d))
			pass.Draw(accel.Draw{VertexCount: 3})

			g, err := r.Build()
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			defer g.Close()

			if got := g.Hazards(); got != c.hazards {
				t.Errorf("%d hazards, want %d: %s", got, c.hazards, c.why)
			}
			if got := g.NodeStats(pass.Node()).BarriersBefore; got != c.barriers {
				t.Errorf("%d barriers before the pass, want %d: %s", got, c.barriers, c.why)
			}

			// Edges is a successor list, so the upload's entry is where the
			// pass appears. Every load action leaves the write-after-write
			// edge, so the pass is a successor of the upload whichever it is.
			succ := g.Edges()[0]
			if len(succ) != 1 || succ[0] != pass.Node() {
				t.Errorf("the upload's successors are %v, want the pass: every load "+
					"action leaves the write-after-write edge", succ)
			}
		})
	}
}

// Two passes writing the same attachment stay ordered whatever the load action
// is, because a write-after-write is an ordering constraint no load action
// removes.
//
// The trap this guards: reading "DontCare removes the edge" as removing every
// edge. It removes the read. Two passes that both write, with the second not
// caring what the first left, still cannot run in either order — the second's
// output must be what survives.
func TestTwoPassesWritingOneAttachmentStayOrdered(t *testing.T) {
	const w, h = 4, 4
	d := openDevice(t)
	target := newBuffer(t, d, "target", w*h*4,
		accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)
	pipe := solidPipeline(t, d)

	r := d.NewRecorder()
	var ids []accel.NodeID
	for _, load := range []accel.LoadOp{accel.LoadClear, accel.LoadDontCare} {
		p := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{View: whole(t, target), Load: load}},
			Width: w, Height: h, Label: "pass",
		})
		p.SetPipeline(pipe)
		p.Draw(accel.Draw{VertexCount: 3})
		ids = append(ids, p.Node())
	}

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	succ := g.Edges()[ids[0]]
	if len(succ) != 1 || succ[0] != ids[1] {
		t.Errorf("the first pass's successors are %v, want the second: DontCare removes "+
			"the read-after-write edge, not the write-after-write one", succ)
	}
}

// A pass reading one attachment and writing another is ordered against both,
// and the two edges come from different halves of the declaration.
//
// Through GeometryVS and ShadeFS, which is the corpus pair that writes two
// attachments and needs both an attribute and a by-value parameter — so this is
// also the first test where the whole stage input path carries a real pipeline.
func TestAPassDeclaresEveryAttachmentItTouches(t *testing.T) {
	const w, h = 4, 4
	d := openDevice(t)
	q := d.Queue()
	usage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	kept := newBuffer(t, d, "kept", w*h*4, usage)
	cleared := newBuffer(t, d, "cleared", w*h*4, usage)

	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.GeometryVSStage,
		Fragment: &testkernels.ShadeFSStage,
		VertexBuffers: []accel.VertexBufferLayout{{
			Stride: 20,
			Attributes: []accel.VertexAttribute{
				{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
				{Location: 1, Format: accel.AttrFloat32x2, Offset: 12},
			},
		}},
		Targets: []accel.ColorTargetState{
			{Format: accel.RGBA32Float}, {Format: accel.RGBA32Float},
		},
		Label: "geometry",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	verts := []float32{
		-1, -1, 0, 0, 0,
		1, -1, 0, 1, 0,
		-1, 1, 0, 0, 1,
	}
	vb := newBuffer(t, d, "verts", len(verts), accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(vb, 0, verts); err != nil {
		t.Fatalf("write: %v", err)
	}

	r := d.NewRecorder()
	r.UploadToBuffer(whole(t, kept), make([]float32, w*h*4))
	r.UploadToBuffer(whole(t, cleared), make([]float32, w*h*4))
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{
			{View: whole(t, kept), Load: accel.LoadKeep},
			{View: whole(t, cleared), Load: accel.LoadClear},
		},
		Width: w, Height: h, Label: "mixed",
	})
	p.SetPipeline(pipe)
	p.SetVertexBuffer(0, whole(t, vb))
	p.SetVertexUniform(0, testkernels.StageTransform{Scale: 1})
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	edges := g.Edges()
	for _, upload := range []accel.NodeID{0, 1} {
		if len(edges[upload]) != 1 || edges[upload][0] != p.Node() {
			t.Errorf("upload %d's successors are %v, want the pass; a pass is ordered "+
				"against every attachment it declares", upload, edges[upload])
		}
	}
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	// ShadeFS writes the interpolated colour to attachment 0 and the UV plus
	// depth to attachment 1. Two attachments carrying different values is the
	// assertion that MRT reached both rather than one twice.
	a, b := readback(t, d, kept), readback(t, d, cleared)
	same := true
	for i := range a {
		if a[i] != b[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("both attachments hold the same values, so the two struct fields did " +
			"not reach different attachments")
	}
}
