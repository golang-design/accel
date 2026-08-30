// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// Stencil written in one pass is read by the next.
//
// specs/033-render-api.md section 2.1's state, reachable at last: internal/raster
// implemented the whole of it -- per-face compare, three outcome operations,
// read and write masks, the dynamic reference -- and accel.DepthStencilState
// carried none of it, so no caller could reach a line.
//
// # Why across two passes
//
// One pass would pass against a scratch array. The stencil buffer used to be
// allocated per pass and thrown away, so a technique that marks in one pass and
// tests in the next -- which is what stencil is *for* -- could not have worked
// even once the state was reachable. Two passes is the assertion that the
// aspect round-trips through the attachment.
//
// # Why the result is read as colour
//
// Queue.ReadTexture refuses every depth format, because what a device stores
// for one is device-defined. So the stencil is observed through the pipeline
// that consumes it: pass 2 draws over the whole target and passes its stencil
// test only where pass 1 marked, and its coverage is the stencil buffer made
// visible.
func TestAStencilWrittenInOnePassIsReadByTheNext(t *testing.T) {
	const w, h = 16, 16
	d := openDevice(t)

	// Pass 1 marks where it covers: always pass the stencil test, replace with
	// the reference. HalfTriangleVS covers the lower-left half.
	mark, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.HalfTriangleVSStage,
		Fragment: &testkernels.SolidFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		DepthStencil: &accel.DepthStencilState{
			Format: accel.Depth32FloatStencil8,
			Stencil: accel.StencilState{
				Enabled: true,
				Front: accel.StencilFace{
					Compare: accel.CompareAlways, ReadMask: 0xff, WriteMask: 0xff,
					Fail: accel.StencilKeep, DepthFail: accel.StencilKeep,
					Pass: accel.StencilReplace,
				},
				Back: accel.StencilFace{
					Compare: accel.CompareAlways, ReadMask: 0xff, WriteMask: 0xff,
					Fail: accel.StencilKeep, DepthFail: accel.StencilKeep,
					Pass: accel.StencilReplace,
				},
			},
		},
		Label: "mark",
	})
	if err != nil {
		t.Fatalf("mark pipeline: %v", err)
	}
	defer mark.Close()

	// Pass 2 covers everything and keeps only what pass 1 marked.
	test, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.FullScreenVSStage,
		Fragment: &testkernels.SolidFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		DepthStencil: &accel.DepthStencilState{
			Format: accel.Depth32FloatStencil8,
			Stencil: accel.StencilState{
				Enabled: true,
				Front: accel.StencilFace{
					Compare: accel.CompareEqual, ReadMask: 0xff, WriteMask: 0,
					Fail: accel.StencilKeep, DepthFail: accel.StencilKeep,
					Pass: accel.StencilKeep,
				},
				Back: accel.StencilFace{
					Compare: accel.CompareEqual, ReadMask: 0xff, WriteMask: 0,
					Fail: accel.StencilKeep, DepthFail: accel.StencilKeep,
					Pass: accel.StencilKeep,
				},
			},
		},
		Label: "test",
	})
	if err != nil {
		t.Fatalf("test pipeline: %v", err)
	}
	defer test.Close()

	marked := renderTexture(t, d, "marked", w, h)
	shown := renderTexture(t, d, "shown", w, h)
	depth, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.Depth32FloatStencil8, Size: accel.Extent{Width: w, Height: h},
		Usage: accel.TextureRenderTarget | accel.TextureCopySrc | accel.TextureCopyDst,
		Kind:  accel.MemoryDevice, Label: "depth",
	})
	if err != nil {
		t.Fatalf("depth: %v", err)
	}
	defer depth.Close()
	dv, err := depth.Whole()
	if err != nil {
		t.Fatalf("depth view: %v", err)
	}

	r := d.NewRecorder()
	p1 := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole2D(t, marked), Load: accel.LoadClear}},
		Depth: &accel.DepthAttachment{View: dv, Load: accel.LoadClear, Clear: 1},
		Width: w, Height: h, Label: "mark",
	})
	p1.SetPipeline(mark)
	p1.SetStencilReference(1)
	p1.Draw(accel.Draw{VertexCount: 3})

	// LoadKeep on the depth attachment: the second pass needs what the first
	// wrote, which is the whole point.
	p2 := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole2D(t, shown), Load: accel.LoadClear}},
		Depth: &accel.DepthAttachment{View: dv, Load: accel.LoadKeep},
		Width: w, Height: h, Label: "test",
	})
	p2.SetPipeline(test)
	p2.SetStencilReference(1)
	p2.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	first := readRenderTexture(t, d, marked)
	second := readRenderTexture(t, d, shown)

	// Where pass 1 drew, pass 2 must have drawn. Where it did not, pass 2 must
	// not have -- and pass 2 covered the whole target, so every pixel it did
	// not write was stopped by the stencil test.
	var covered, shownCount, mismatched int
	for i := 0; i < len(first); i += 4 {
		a := first[i+3] != 0
		b := second[i+3] != 0
		if a {
			covered++
		}
		if b {
			shownCount++
		}
		if a != b {
			mismatched++
		}
	}
	if covered == 0 {
		t.Fatal("the marking pass drew nothing, so there is no stencil to test")
	}
	if covered == w*h {
		t.Fatal("the marking pass covered everything, so a stencil test that always " +
			"passed would agree with it")
	}
	if mismatched > 0 {
		t.Errorf("%d of %d pixels disagree: the marking pass covered %d and the "+
			"stencil test kept %d", mismatched, w*h, covered, shownCount)
	}
}

// A stencil state against a depth format with no stencil aspect is refused at
// creation.
//
// specs/033-render-api.md section 2.2. Without it the pipeline compiles, the
// draw runs, and the stencil state changes nothing -- which reads as the
// technique being wrong rather than the format.
func TestAStencilStateNeedsAStencilAspect(t *testing.T) {
	d := openDevice(t)
	_, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.FullScreenVSStage,
		Fragment: &testkernels.SolidFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		DepthStencil: &accel.DepthStencilState{
			Format:  accel.Depth32Float,
			Stencil: accel.StencilState{Enabled: true},
		},
		Label: "aspectless",
	})
	if err == nil {
		t.Fatal("a stencil state against Depth32Float was accepted, so the state is " +
			"silently inert")
	}
	for _, want := range []string{"Depth32Float", "no stencil aspect", "Depth32FloatStencil8"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal does not name %q: %v", want, err)
		}
	}
	// The accepting half: the same state against a format that has the aspect.
	p, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.FullScreenVSStage,
		Fragment: &testkernels.SolidFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		DepthStencil: &accel.DepthStencilState{
			Format:  accel.Depth32FloatStencil8,
			Stencil: accel.StencilState{Enabled: true},
		},
		Label: "aspectful",
	})
	if err != nil {
		t.Fatalf("a stencil state against Depth32FloatStencil8 was refused: %v", err)
	}
	p.Close()
}

// Depth32FloatStencil8 reports the layout it promises.
//
// The codec's own round trip lives beside the codec, in internal/cpu; this is
// the half a caller can see, and BytesPerPixel is the number they size a
// readback from.
func TestTheStencilFormatReportsItsLayout(t *testing.T) {
	d := openDevice(t)
	info := d.FormatInfo(accel.Depth32FloatStencil8)
	if !info.IsDepth || !info.IsStencil {
		t.Errorf("Depth32FloatStencil8 reports depth=%t stencil=%t", info.IsDepth, info.IsStencil)
	}
	// The depth plane's, because the format is planar: BytesPerPixel is what a
	// row pitch is computed from and the two planes have different ones.
	if info.BytesPerPixel != 4 {
		t.Errorf("BytesPerPixel is %d, want 4: it is the depth plane's, and the stencil "+
			"plane has its own pitch", info.BytesPerPixel)
	}
	if !accel.Depth32FloatStencil8.Planar() {
		t.Error("Depth32FloatStencil8 does not report itself planar, so a caller sizing " +
			"a subresource from BytesPerPixel alone would be wrong by a whole plane")
	}
	if accel.Depth32Float.Planar() {
		t.Error("Depth32Float reports itself planar and has one aspect")
	}
	// Depth24PlusStencil8 stays device-defined, and adding a stencil format is
	// not a reason to relax that: "24 plus" still has two defensible encodings.
	if bpp := d.FormatInfo(accel.Depth24PlusStencil8).BytesPerPixel; bpp != 0 {
		t.Errorf("Depth24PlusStencil8 reports %d bytes per pixel, and its layout is "+
			"device-defined", bpp)
	}
}
