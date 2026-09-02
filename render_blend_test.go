// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
)

// blendPipeline draws a full-viewport triangle in a by-value colour, blended.
func blendPipeline(t *testing.T, d *accel.Device, blend accel.BlendState) *accel.RenderPipeline {
	t.Helper()
	p, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &kernels.ScaledVSStage,
		Fragment: &kernels.TintedFSStage,
		VertexBuffers: []accel.VertexBufferLayout{{
			Stride: 12,
			Attributes: []accel.VertexAttribute{
				{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
			},
		}},
		Targets: []accel.ColorTargetState{{Format: accel.RGBA32Float, Blend: blend}},
		Label:   "blended",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })
	return p
}

// Draws execute in the order recorded, and the builder never reorders them.
//
// specs/033-render-api.md section 7. Checked with two overlapping blended draws
// whose result depends on the order: "over" is not commutative, so swapping the
// two produces a different colour, and a builder that reordered would produce
// the other one. A test using opaque draws could not tell — the last one wins
// either way, and "last" is what a reorder changes.
// blendedOverResult draws two half-transparent triangles in the order given and
// returns the pixel. Package-level so the capability test below drives the same
// path rather than a second copy of it.
func blendedOverResult(t *testing.T, first, second [4]float32) [4]float32 {
	t.Helper()
	const w, h = 8, 8
	{
		d := openDevice(t)
		q := d.Queue()
		pipe := blendPipeline(t, d, accel.AlphaBlend())

		pos := []float32{-1, -1, 0, 3, -1, 0, -1, 3, 0}
		vb := newBuffer(t, d, "pos", len(pos), accel.BufferStorage|accel.BufferCopyDst)
		if err := q.WriteBuffer(vb, 0, pos); err != nil {
			t.Fatalf("write: %v", err)
		}
		target := colourTarget(t, d, "colour", w, h)

		r := d.NewRecorder()
		pass := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{
				View: view(t, target), Load: accel.LoadClear,
				Clear: [4]float32{0, 0, 0, 0},
			}},
			Width: w, Height: h, Label: "order",
		})
		pass.SetPipeline(pipe)
		pass.SetVertexBuffer(0, whole(t, vb))
		pass.SetVertexUniform(0, kernels.StageTransform{Scale: 1})
		for _, c := range [][4]float32{first, second} {
			pass.SetFragmentUniform(0, kernels.StageTint{Colour: c})
			pass.Draw(accel.Draw{VertexCount: 3})
		}

		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		if err := q.Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		got := readTarget(t, d, target)
		return [4]float32{got[0], got[1], got[2], got[3]}
	}
}

func TestBlendedDrawsRunInRecordedOrder(t *testing.T) {
	// Half-transparent red over half-transparent blue, and the reverse. The
	// "over" operator gives src + dst*(1-srcAlpha) with premultiplied-style
	// weights, so the second draw dominates and the two orders differ.
	red := [4]float32{1, 0, 0, 0.5}
	blue := [4]float32{0, 0, 1, 0.5}

	redFirst := blendedOverResult(t, red, blue)
	blueFirst := blendedOverResult(t, blue, red)

	if redFirst == blueFirst {
		t.Fatalf("both orders produced %v, so the result does not depend on the order "+
			"and this test proves nothing about reordering", redFirst)
	}

	// The arithmetic, so the assertion is about the right thing rather than
	// about two values merely differing. "over" is
	// src*srcAlpha + dst*(1-srcAlpha) per channel.
	over := func(src, dst [4]float32) [4]float32 {
		var out [4]float32
		for i := range 3 {
			out[i] = src[i]*src[3] + dst[i]*(1-src[3])
		}
		out[3] = src[3] + dst[3]*(1-src[3])
		return out
	}
	want := over(blue, over(red, [4]float32{0, 0, 0, 0}))
	for i := range want {
		if math.Abs(float64(redFirst[i]-want[i])) > 1e-6 {
			t.Errorf("red then blue gives %v, and the over operator applied twice gives "+
				"%v", redFirst, want)
			break
		}
	}
}

// A blended, depth-tested pass is accepted rather than rejected as feedback.
//
// specs/033-render-api.md section 3: fixed-function attachment
// read-modify-write is ordered by the ROP, so describing a blended pass as
// "reading a resource it writes" would reject every blended or depth-tested
// pass ever written. The refusal that does exist is for a *stage* reading an
// attachment through a binding, which is a different thing.
func TestABlendedDepthTestedPassIsAccepted(t *testing.T) {
	const w, h = 4, 4
	d := openDevice(t)
	q := d.Queue()

	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &kernels.ScaledVSStage,
		Fragment: &kernels.TintedFSStage,
		VertexBuffers: []accel.VertexBufferLayout{{
			Stride: 12,
			Attributes: []accel.VertexAttribute{
				{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
			},
		}},
		DepthStencil: &accel.DepthStencilState{
			Format: accel.Depth32Float, Test: true, Write: true,
			Compare: accel.CompareLess,
		},
		Targets: []accel.ColorTargetState{{
			Format: accel.RGBA32Float, Blend: accel.AlphaBlend(),
		}},
		Label: "blended and depth tested",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	pos := []float32{-1, -1, 0, 3, -1, 0, -1, 3, 0}
	vb := newBuffer(t, d, "pos", len(pos), accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(vb, 0, pos); err != nil {
		t.Fatalf("write: %v", err)
	}
	colour := colourTarget(t, d, "colour", w, h)
	depth := depthTarget(t, d, "depth", w, h)

	r := d.NewRecorder()
	// Both attachments loaded Keep, which is the case a naive feedback check
	// rejects: the pass reads and writes each of them.
	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: view(t, colour), Load: accel.LoadKeep}},
		Depth: &accel.DepthAttachment{View: view(t, depth), Load: accel.LoadKeep},
		Width: w, Height: h, Label: "rmw",
	})
	pass.SetPipeline(pipe)
	pass.SetVertexBuffer(0, whole(t, vb))
	pass.SetVertexUniform(0, kernels.StageTransform{Scale: 1})
	pass.SetFragmentUniform(0, kernels.StageTint{Colour: [4]float32{1, 1, 1, 0.5}})
	pass.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("a blended, depth-tested pass was refused: %v", err)
	}
	defer g.Close()
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
}

// Blending off replaces, which is what a target that says nothing about
// blending must do.
//
// The zero BlendState is the common case, and inferring "enabled" from
// non-zero factors would make the zero value multiply everything by zero.
func TestTheZeroBlendStateReplaces(t *testing.T) {
	const w, h = 4, 4
	d := openDevice(t)
	q := d.Queue()
	pipe := blendPipeline(t, d, accel.BlendState{})

	pos := []float32{-1, -1, 0, 3, -1, 0, -1, 3, 0}
	vb := newBuffer(t, d, "pos", len(pos), accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(vb, 0, pos); err != nil {
		t.Fatalf("write: %v", err)
	}
	target := colourTarget(t, d, "colour", w, h)

	tint := [4]float32{0.25, 0.5, 0.75, 0.5}
	r := d.NewRecorder()
	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{
			View: view(t, target), Load: accel.LoadClear, Clear: [4]float32{1, 1, 1, 1},
		}},
		Width: w, Height: h, Label: "replace",
	})
	pass.SetPipeline(pipe)
	pass.SetVertexBuffer(0, whole(t, vb))
	pass.SetVertexUniform(0, kernels.StageTransform{Scale: 1})
	pass.SetFragmentUniform(0, kernels.StageTint{Colour: tint})
	pass.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := readTarget(t, d, target)
	px := [4]float32{got[0], got[1], got[2], got[3]}
	if px != tint {
		t.Errorf("pixel (0,0) is %v, want exactly the fragment's %v; the zero BlendState "+
			"combined with the clear instead of replacing it", px, tint)
	}
}

// The CPU device reports rasterizer-ordered access, and this is what licenses
// it until a fragment stage can write storage.
//
// specs/005-graphics.md's ROA paragraph, and a correction to how it was
// recorded. specs/STATUS.md first called this "a capability advertised and not
// emulated". That was too strong: the CPU rasterizer processes primitives
// sequentially, so primitive-ordered access holds by construction, and the bit
// is honest. What is true is narrower — the capability is **unreachable**,
// because no fragment stage can bind a written slice, so nothing can observe
// the ordering the bit promises.
//
// So the claim is tied to the nearest thing that *is* observable: blended draws
// landing in recorded order, which is the same sequential execution ROA would
// rest on. If the rasterizer ever grows parallelism across primitives, that
// assertion fails here and the bit must be re-examined rather than quietly
// becoming false.
func TestTheCPUReportsOrderedAccessOnlyWhileItRasterizesInOrder(t *testing.T) {
	if !openDevice(t).Capabilities().RasterizerOrderedAccess {
		t.Skip("the CPU device no longer reports rasterizer-ordered access; if " +
			"that is deliberate, this test and specs/005's paragraph go together")
	}

	// The observable half, driven through the same path
	// TestBlendedDrawsRunInRecordedOrder uses so the two cannot drift apart.
	red := [4]float32{1, 0, 0, 0.5}
	blue := [4]float32{0, 0, 1, 0.5}
	if blendedOverResult(t, red, blue) == blendedOverResult(t, blue, red) {
		t.Fatal("the device reports rasterizer-ordered access and swapping two " +
			"overlapping draws changes nothing, so primitive order is not " +
			"preserved and the capability is false")
	}
}
