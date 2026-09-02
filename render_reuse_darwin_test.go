// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package accel_test

import (
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
	"golang.design/x/accel/internal/mtl"
)

// A replayed render pass reuses its Metal textures rather than allocating
// them per submission.
//
// The backend used to give every attachment and every sampled texture a fresh
// MTLTexture per pass and close it at the pass's end, so a frame paid an
// allocation per texture per frame. The executable now caches them by what
// they are a view of and sweeps what a submission did not use. Measured on an
// M2: a 32-draw pass's submit went from 271 to 186 us. The count of live
// textures after the first submission is the count after the third, and Close
// releases them.
func TestAReplayedPassReusesItsTextures(t *testing.T) {
	const w, h = 8, 8
	d := openMetal(t)
	draw := texturePipeline(t, d, &kernels.FullScreenVSStage, &kernels.RowFSStage)
	defer draw.Close()
	blit := texturePipeline(t, d, &kernels.FullScreenVSStage, &kernels.BlitFSStage)
	defer blit.Close()
	first := renderTexture(t, d, "first", w, h)
	second := renderTexture(t, d, "second", w, h)

	r := d.NewRecorder()
	p1 := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole2D(t, first), Load: accel.LoadClear}},
		Width: w, Height: h, Label: "draw",
	})
	p1.SetPipeline(draw)
	p1.Draw(accel.Draw{VertexCount: 3})
	p2 := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole2D(t, second), Load: accel.LoadClear}},
		Width: w, Height: h, Label: "blit",
	})
	p2.SetPipeline(blit)
	p2.SetTexture(0, whole2D(t, first))
	p2.Draw(accel.Draw{VertexCount: 3})
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	before := mtl.LiveTextures()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	afterFirst := mtl.LiveTextures()
	if afterFirst <= before {
		t.Fatalf("the first submission created no textures (%d before, %d after)", before, afterFirst)
	}
	for i := 0; i < 2; i++ {
		if err := d.Queue().Submit(g).Wait(); err != nil {
			t.Fatalf("submit %d: %v", i+2, err)
		}
	}
	if got := mtl.LiveTextures(); got != afterFirst {
		t.Fatalf("%d textures live after three submissions, %d after one: the pass "+
			"allocates per submission", got, afterFirst)
	}
	// And the picture is still the fetch, not a stale texture.
	a := readRenderTexture(t, d, first)
	b := readRenderTexture(t, d, second)
	for p := range w * h {
		if [4]float32(b[p*4:p*4+4]) != [4]float32(a[p*4:p*4+4]) {
			t.Fatalf("pixel %d differs between the passes after replay", p)
		}
	}
	if err := g.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if got := mtl.LiveTextures(); got != before {
		t.Fatalf("%d textures live after Close, %d before the graph existed", got, before)
	}
}
