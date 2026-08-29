// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// discardPass records specs/032-stage-abi.md section 4.2's whole claim as one
// picture: a discard writes no attachment *and no depth*.
//
// # How the depth half is observed
//
// A depth attachment cannot be read back -- Queue.ReadTexture refuses every
// depth format, because what a device stores for one is device-defined. So the
// depth write is observed through the pipeline that consumes it instead, which
// is a better assertion than a readback anyway: it asks what the depth buffer
// *does*, not what it holds.
//
// Draw 1 covers everything at z = 0 and discards the left columns. Draw 2
// covers everything at z = 0.5 under CompareLess. Where draw 1 was kept, depth
// is 0 and draw 2 is rejected. Where draw 1 discarded, depth is still the clear
// and draw 2 passes. So the final image is:
//
//	left  columns -> draw 2's colour   (draw 1 wrote neither colour nor depth)
//	right columns -> draw 1's colour   (draw 1 wrote both)
//
// and three separate failures are distinguishable in it. A discard that was
// ignored puts draw 1's colour everywhere. A discard that suppressed the colour
// but still wrote depth leaves the clear colour on the left, since draw 2 would
// then be rejected there too. A discard that took everything puts draw 2's
// colour everywhere.
func discardImage(t *testing.T, d *accel.Device, w, h int) []float32 {
	t.Helper()
	q := d.Queue()

	depthState := &accel.DepthStencilState{
		Format: accel.Depth32Float, Test: true, Write: true,
		Compare: accel.CompareLess,
	}
	front, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:       &testkernels.FullScreenVSStage,
		Fragment:     &testkernels.DiscardFSStage,
		Targets:      []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		DepthStencil: depthState,
		Label:        "discarding",
	})
	if err != nil {
		t.Fatalf("discarding pipeline: %v", err)
	}
	defer front.Close()

	behind, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.ScaledVSStage,
		Fragment: &testkernels.TintedFSStage,
		VertexBuffers: []accel.VertexBufferLayout{{
			Stride:     12,
			Attributes: []accel.VertexAttribute{{Location: 0, Format: accel.AttrFloat32x3}},
		}},
		Targets:      []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		DepthStencil: depthState,
		Label:        "behind",
	})
	if err != nil {
		t.Fatalf("behind pipeline: %v", err)
	}
	defer behind.Close()

	// A full-screen triangle at z = 0.5, behind draw 1's z = 0 and in front of
	// the depth clear.
	verts := []float32{-1, -1, 0.5, 3, -1, 0.5, -1, 3, 0.5}
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
		Kind:  accel.MemoryReadback, Label: "colour",
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
		// A clear colour neither stage writes, so "nothing drew here" is
		// distinguishable from either of them having drawn.
		Color: []accel.ColorAttachment{{
			View: cv, Load: accel.LoadClear, Clear: [4]float32{1, 0, 1, 1},
		}},
		Depth: &accel.DepthAttachment{View: dv, Load: accel.LoadClear, Clear: 1},
		Width: w, Height: h, Label: "discard",
	})
	p.SetPipeline(front)
	p.Draw(accel.Draw{VertexCount: 3})
	p.SetPipeline(behind)
	p.SetVertexBuffer(0, vv)
	p.SetVertexUniform(0, testkernels.StageTransform{Scale: 1, Offset: accel.Vec2{0, 0}})
	p.SetFragmentUniform(0, testkernels.StageTint{Colour: accel.Vec4{0, 1, 0, 1}})
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	raw := make([]byte, w*h*colour.Format().BytesPerPixel())
	if err := q.ReadTexture(colour, raw); err != nil {
		t.Fatalf("read: %v", err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = float32frombytes(raw[i*4 : i*4+4])
	}
	return out
}

// discardWant is the picture discardPass must produce.
//
// DiscardFS discards where the pixel centre's x is below 4, so column 3 and
// below take the behind draw's colour and column 4 and above take the
// discarding stage's own.
func discardWant(x int) [4]float32 {
	if x < 4 {
		return [4]float32{0, 1, 0, 1}
	}
	return [4]float32{0.25, 0.5, 0.75, 1}
}

// specs/032-stage-abi.md section 4.2 on the oracle.
func TestADiscardWritesNeitherColourNorDepth(t *testing.T) {
	const w, h = 8, 4
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer d.Close()
	got := discardImage(t, d, w, h)

	for y := range h {
		for x := range w {
			want := discardWant(x)
			at := (y*w + x) * 4
			for c := range 4 {
				if got[at+c] != want[c] {
					t.Fatalf("pixel (%d,%d) is %v, want %v -- %s", x, y,
						got[at:at+4], want, discardDiagnosis(got[at:at+4]))
				}
			}
		}
	}
}

// discardDiagnosis names which of the three failures a pixel shows, so a
// wrong picture reports what went wrong rather than only that it did.
func discardDiagnosis(px []float32) string {
	switch {
	case px[0] == 1 && px[1] == 0 && px[2] == 1:
		return "this is the clear colour: the discard suppressed the colour " +
			"write but still wrote depth, so the draw behind it was rejected too"
	case px[0] == 0 && px[1] == 1:
		return "this is the behind draw's colour: the discard took a fragment " +
			"it should have kept"
	default:
		return "this is the discarding stage's colour: the discard did not happen"
	}
}

func float32frombytes(b []byte) float32 {
	return math.Float32frombits(uint32(b[0]) | uint32(b[1])<<8 |
		uint32(b[2])<<16 | uint32(b[3])<<24)
}
