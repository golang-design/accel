// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"testing"

	"golang.design/x/accel"
)

// A transient used as a render attachment is live at the pass that writes it.
//
// It was not. Every node kind had to call Recorder.touch for itself, and the
// render pass — the first kind added after that rule existed — did not, so a
// transient attachment's live range covered the upload and the readback and not
// the pass in between. With no readback the range was one node long, and the
// aliasing pass was free to put another transient over bytes the pass writes.
//
// Asserted on the placement rather than on pixels, because the corruption needs
// a second transient of the right size in the right span to appear at all — the
// bug is that the *permission* exists, and that is what this reads.
func TestATransientAttachmentIsLiveAtItsPass(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)
	r := d.NewRecorder()
	usage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst

	att := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: w * h * 4, Usage: usage, Label: "attachment",
	})
	r.UploadToBuffer(att, make([]float32, w*h*4))
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: att, Load: accel.LoadClear}},
		Width: w, Height: h, Label: "pass",
	})
	p.SetPipeline(solidPipeline(t, d))
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	places := g.TransientPlacement()
	if len(places) != 1 {
		t.Fatalf("%d placements, want the one attachment", len(places))
	}
	var atPass bool
	for _, n := range places[0].Users {
		if n == p.Node() {
			atPass = true
		}
	}
	if !atPass {
		t.Errorf("%q is live at nodes %v and the pass is node %d; a transient the pass "+
			"writes is not in its own live range, so another transient may be placed "+
			"over it", places[0].Label, places[0].Users, p.Node())
	}
}

// Two render targets share memory when one is dead before the other is written.
//
// specs/033-render-api.md section 7's aliasing consequence, stated as what the
// placement shows: two transients at one offset is aliasing, and it is sound
// only when every user of one is ordered against every user of the other. The
// ordering here is a real chain — render, read the result back, use it as the
// next pass's geometry — because without one the two passes may run
// concurrently and must *not* share bytes, which is the case that first ran.
func TestTwoRenderTargetsAliasWhenTheirRangesDoNot(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)
	q := d.Queue()
	usage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	mid := newBuffer(t, d, "mid", w*h*4, usage)
	out := newBuffer(t, d, "out", w*h*4, usage)

	r := d.NewRecorder()
	first := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: w * h * 4, Usage: usage, Label: "first",
	})
	second := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: w * h * 4, Usage: usage, Label: "second",
	})

	// Pass 0 renders into first; the copy reads it, and first is dead after.
	p0 := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: first, Load: accel.LoadClear}},
		Width: w, Height: h, Label: "first pass",
	})
	p0.SetPipeline(solidPipeline(t, d))
	p0.Draw(accel.Draw{VertexCount: 3})
	r.CopyBuffer(whole(t, mid), first)

	// Pass 1 renders into second, taking its geometry from what the copy
	// wrote. That read is the edge that orders it after every user of first.
	p1 := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: second, Load: accel.LoadClear}},
		Width: w, Height: h, Label: "second pass",
	})
	p1.SetPipeline(attributePipeline(t, d))
	p1.SetVertexBuffer(0, whole(t, mid))
	p1.Draw(accel.Draw{VertexCount: 3})
	r.CopyBuffer(whole(t, out), second)

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
	// the second pass writing over what the first copy had not yet read.
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	if got := readback(t, d, mid); got[(h-1)*w*4] != 0.25 {
		t.Errorf("the first pass's copy holds %v, want its drawn colour",
			got[(h-1)*w*4:(h-1)*w*4+4])
	}
}
