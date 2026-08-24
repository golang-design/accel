// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"testing"

	"golang.design/x/accel"
)

// A transient a render pass reads is live at that pass.
//
// It was not. Every node kind had to call Recorder.touch for itself, and the
// render pass — the first kind added after that rule existed — did not, so a
// transient it touched had a live range that covered the upload and the
// readback and not the pass in between. With no readback the range was one node
// long, and the aliasing pass was free to put another transient over bytes the
// pass uses.
//
// Asserted on the placement rather than on pixels, because the corruption needs
// a second transient of the right size in the right span to appear at all — the
// bug is that the *permission* exists, and that is what this reads.
//
// # What this covered before, and what it covers now
//
// It read the same property through a transient *attachment*, which
// specs/045-texture-attachments.md ended: an attachment is a texture view and a
// texture cannot be a transient. The regression this pins is Recorder.touch on
// a render pass and that is unchanged, but the attachment path specifically is
// no longer reachable from a transient, so nothing here exercises it.
func TestATransientAPassReadsIsLiveAtThatPass(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)
	r := d.NewRecorder()

	verts := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: len(triangleVertices()),
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
		Label: "vertices",
	})
	r.UploadToBuffer(verts, triangleVertices())
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{
			View: view(t, colourTarget(t, d, "colour", w, h)), Load: accel.LoadClear,
		}},
		Width: w, Height: h, Label: "pass",
	})
	p.SetPipeline(attributePipeline(t, d))
	p.SetVertexBuffer(0, verts)
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	places := g.TransientPlacement()
	if len(places) != 1 {
		t.Fatalf("%d placements, want the one transient", len(places))
	}
	var atPass bool
	for _, n := range places[0].Users {
		if n == p.Node() {
			atPass = true
		}
	}
	if !atPass {
		t.Errorf("%q is live at nodes %v and the pass is node %d; a transient the pass "+
			"reads is not in its own live range, so another transient may be placed "+
			"over it", places[0].Label, places[0].Users, p.Node())
	}
}

// Two transients a pass each touches share memory when one is dead before the
// other is written.
//
// specs/033-render-api.md section 7's aliasing consequence, stated as what the
// placement shows: two transients at one offset is aliasing, and it is sound
// only when every user of one is ordered against every user of the other. The
// ordering here is a real chain — render, read the result back, feed it to the
// next pass — because without one the two passes may run concurrently and must
// *not* share bytes, which is the case that first ran.
//
// The transients are the passes' vertex data rather than their render targets,
// which is what this read before specs/045-texture-attachments.md made an
// attachment a texture. A texture cannot be a transient, so the aliasing of two
// render *targets* is no longer expressible; the relation being checked is the
// same one.
func TestTwoTransientsARenderPassTouchesAliasWhenTheirRangesDoNot(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)
	q := d.Queue()
	usage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	verts := triangleVertices()

	first := colourTarget(t, d, "first", w, h)
	second := colourTarget(t, d, "second", w, h)
	bridge := newBuffer(t, d, "bridge", w*h*4, usage)

	r := d.NewRecorder()
	firstVerts := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: len(verts), Usage: usage, Label: "first vertices",
	})
	secondVerts := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: len(verts), Usage: usage, Label: "second vertices",
	})

	// Pass 0 reads firstVerts, which is dead after it.
	r.UploadToBuffer(firstVerts, verts)
	p0 := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: view(t, first), Load: accel.LoadClear}},
		Width: w, Height: h, Label: "first pass",
	})
	p0.SetPipeline(attributePipeline(t, d))
	p0.SetVertexBuffer(0, firstVerts)
	p0.Draw(accel.Draw{VertexCount: 3})

	// The chain: the pass's output is read back into a buffer, and that buffer
	// writes the second transient. Every user of firstVerts is ordered before
	// every user of secondVerts, which is exactly what makes aliasing sound.
	r.CopyTextureToBuffer(whole(t, bridge), first)
	src, err := bridge.View(0, len(verts))
	if err != nil {
		t.Fatalf("view: %v", err)
	}
	r.CopyBuffer(secondVerts, src)

	p1 := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: view(t, second), Load: accel.LoadClear}},
		Width: w, Height: h, Label: "second pass",
	})
	p1.SetPipeline(attributePipeline(t, d))
	p1.SetVertexBuffer(0, secondVerts)
	p1.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	places := g.TransientPlacement()
	if len(places) != 2 {
		t.Fatalf("%d placements, want two", len(places))
	}
	if places[0].Offset != places[1].Offset {
		t.Errorf("%q is at %d and %q at %d; every user of the first is ordered before "+
			"every user of the second, so they should share bytes", places[0].Label,
			places[0].Offset, places[1].Label, places[1].Offset)
	}
	if got, want := g.Memory().TransientBytes, g.Memory().UnaliasedBytes; got >= want {
		t.Errorf("aliasing saved nothing: %d bytes against %d unaliased", got, want)
	}

	// And it still runs, because a placement that aliased unsoundly would have
	// the copy writing over vertices the first pass had not yet read.
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	if got := readback(t, d, bridge); got[(h-1)*w*4+3] != 1 {
		t.Errorf("the first pass's output has alpha %v at the bottom left, want 1: every "+
			"vertex supplied 1, so the pass did not draw through its transient",
			got[(h-1)*w*4+3])
	}
}

// triangleVertices is the interleaved position-and-colour triangle the
// attribute pipeline reads, with its three vertices on viewport corners.
func triangleVertices() []float32 {
	var v []float32
	v = append(v, interleaved(-1, -1, 0, [4]float32{1, 0, 0, 1})...)
	v = append(v, interleaved(1, -1, 0, [4]float32{0, 1, 0, 1})...)
	v = append(v, interleaved(-1, 1, 0, [4]float32{0, 0, 1, 1})...)
	return v
}
