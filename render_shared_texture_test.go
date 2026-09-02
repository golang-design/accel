// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
)

// Two draws of one pass binding the same texture see the same texels, and the
// pass produces the picture one draw does.
//
// The CPU backend decodes a bound texture once per pass rather than once per
// draw. The picture is the check that the second draw reads the decode the
// first one made and not something else: BlitFS copies its own pixel, so any
// difference between the draws' inputs would show as a pixel that differs from
// the source.
func TestTwoDrawsSharingATextureSeeTheSameTexels(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)

	draw := texturePipeline(t, d, &kernels.FullScreenVSStage, &kernels.SolidFSStage)
	defer draw.Close()
	blit := texturePipeline(t, d, &kernels.FullScreenVSStage, &kernels.BlitFSStage)
	defer blit.Close()

	source := renderTexture(t, d, "source", w, h)
	once := renderTexture(t, d, "once", w, h)
	twice := renderTexture(t, d, "twice", w, h)

	r := d.NewRecorder()
	p1 := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole2D(t, source), Load: accel.LoadClear}},
		Width: w, Height: h, Label: "draw",
	})
	p1.SetPipeline(draw)
	p1.Draw(accel.Draw{VertexCount: 3})

	// One draw, the reference.
	p2 := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole2D(t, once), Load: accel.LoadClear}},
		Width: w, Height: h, Label: "blit once",
	})
	p2.SetPipeline(blit)
	p2.SetTexture(0, whole2D(t, source))
	p2.Draw(accel.Draw{VertexCount: 3})

	// Two draws binding the same texture, which the backend decodes once.
	p3 := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole2D(t, twice), Load: accel.LoadClear}},
		Width: w, Height: h, Label: "blit twice",
	})
	p3.SetPipeline(blit)
	p3.SetTexture(0, whole2D(t, source))
	p3.Draw(accel.Draw{VertexCount: 3})
	p3.SetTexture(0, whole2D(t, source))
	p3.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	src := readRenderTexture(t, d, source)
	a := readRenderTexture(t, d, once)
	b := readRenderTexture(t, d, twice)
	var nonZero int
	for i := range src {
		if src[i] != 0 {
			nonZero++
		}
		if a[i] != src[i] || b[i] != src[i] {
			t.Fatalf("element %d is %v after one draw and %v after two, want the source's %v",
				i, a[i], b[i], src[i])
		}
	}
	if nonZero == 0 {
		t.Fatal("the source pass drew nothing, so the comparison is vacuous")
	}
}
