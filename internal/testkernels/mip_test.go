// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// Two mips of one texture are written and read independently.
//
// specs/045-texture-attachments.md section 2 makes a view name a subresource,
// and until 2026-08-30 that name reached the feedback rule and nothing else:
// the allocation, the hazard range and the fetch extent were all the base
// level's. This is the end-to-end assertion that the name is now an address.
//
// # Why the values are read through a fetch
//
// Queue.ReadTexture returns the base level, so it cannot see mip 1 at all. The
// second level is read the way a stage would read it -- an on-device fetch into
// a buffer-backed attachment -- which is also the path whose extent and pitch
// are the level's rather than the base's.
//
// # What a wrong offset looks like
//
// Overlap. Mip 1 of an 8x8 RGBA32Float texture is 4x4, and if its offset were
// zero it would sit on top of mip 0's first rows: the two passes would write
// each other's bytes and the fetch would return whichever ran last. The colours
// are distinct so that outcome is a different number rather than a coincidence.
func TestTwoMipsAreWrittenAndReadIndependently(t *testing.T) {
	const w, h = 8, 8

	t.Run("cpu", func(t *testing.T) {
		d, err := accel.OpenCPU(accel.CPUOptions{})
		if err != nil {
			t.Fatalf("OpenCPU: %v", err)
		}
		defer d.Close()
		checkMipsAreIndependent(t, d, w, h)
	})
}

// checkMipsAreIndependent renders a different colour into each of two mips and
// reads the second one back through a fetch.
func checkMipsAreIndependent(t *testing.T, d *accel.Device, w, h int) {
	t.Helper()
	q := d.Queue()

	tex, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.RGBA32Float, Size: accel.Extent{Width: w, Height: h},
		MipLevels: 2,
		Usage: accel.TextureRenderTarget | accel.TextureSampled |
			accel.TextureCopySrc | accel.TextureCopyDst,
		Kind: accel.MemoryReadback, Label: "mipped",
	})
	if err != nil {
		t.Fatalf("texture: %v", err)
	}
	defer tex.Close()

	base, err := tex.View(accel.TextureViewDesc{Mip: 0})
	if err != nil {
		t.Fatalf("base view: %v", err)
	}
	second, err := tex.View(accel.TextureViewDesc{Mip: 1})
	if err != nil {
		t.Fatalf("second view: %v", err)
	}
	sub := second.Subresource()

	paint, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.FullScreenVSStage,
		Fragment: &testkernels.TintedFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		Label:    "paint",
	})
	if err != nil {
		t.Fatalf("paint pipeline: %v", err)
	}
	defer paint.Close()

	fetch, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.FullScreenVSStage,
		Fragment: &testkernels.BlitFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		Label:    "fetch",
	})
	if err != nil {
		t.Fatalf("fetch pipeline: %v", err)
	}
	defer fetch.Close()

	out, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: sub.Width * sub.Height * 4,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
		Label: "mip one",
	})
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer out.Close()
	ov, err := out.View(0, out.Count())
	if err != nil {
		t.Fatalf("buffer view: %v", err)
	}

	baseColour := accel.Vec4{1, 0, 0, 1}
	secondColour := accel.Vec4{0, 0.5, 0.25, 1}

	r := d.NewRecorder()
	slot := r.Slot(accel.SlotDescriptor{
		Name: "mip one", Kind: accel.BindingStorageBuffer,
		DType: accel.F32, Access: accel.AccessWrite,
		MinCount: sub.Width * sub.Height * 4,
	})
	for _, c := range []struct {
		view   accel.TextureView
		colour accel.Vec4
		w, h   int
	}{
		{base, baseColour, w, h},
		{second, secondColour, sub.Width, sub.Height},
	} {
		p := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{View: c.view, Load: accel.LoadClear}},
			Width: c.w, Height: c.h, Label: "paint",
		})
		p.SetPipeline(paint)
		p.SetFragmentUniform(0, testkernels.StageTint{Colour: c.colour})
		p.Draw(accel.Draw{VertexCount: 3})
	}

	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{Slot: slot, Load: accel.LoadClear}},
		Width: sub.Width, Height: sub.Height, Label: "read mip one",
	})
	p.SetPipeline(fetch)
	p.SetTexture(0, second)
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := g.Bind(accel.SlotBinding{Slot: slot, Buffer: ov}); err != nil {
		t.Fatalf("bind: %v", err)
	}
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	got := make([]float32, sub.Width*sub.Height*4)
	if err := q.ReadBuffer(out, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}
	for i := range got {
		want := secondColour[i%4]
		if got[i] != want {
			t.Fatalf("mip 1 texel %d channel %d is %v, want %v (mip 0 holds %v): the two "+
				"levels are not disjoint", i/4, i%4, got[i], want, baseColour[i%4])
		}
	}

	// And mip 0 still holds its own colour, which is the other half of
	// "independently": a level 1 write that landed at offset zero would have
	// overwritten it.
	raw := make([]byte, w*h*tex.Format().BytesPerPixel())
	if err := q.ReadTexture(tex, raw); err != nil {
		t.Fatalf("read texture: %v", err)
	}
	first := float32sOf(raw)
	for i := range first {
		if want := baseColour[i%4]; first[i] != want {
			t.Fatalf("mip 0 texel %d channel %d is %v, want %v: writing mip 1 moved it",
				i/4, i%4, first[i], want)
		}
	}
}
