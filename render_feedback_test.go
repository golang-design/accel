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

// A stage that fetches the subresource its pass is writing is refused.
//
// specs/033-render-api.md §3.3, and specs/045-texture-attachments.md §5 puts it
// last because it needs both an attachment and a fetch to have anything to
// reject. Both exist now.
//
// The result of a stage reading what the pass is writing is undefined on every
// target — not wrong in a way that shows, but a picture that is right on the
// machine it was written on. That is the whole reason it is a build error
// rather than a documented hazard.
func TestAStageFetchingTheAttachmentIsFeedback(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)

	blit := texturePipeline(t, d, &testkernels.FullScreenVSStage,
		&testkernels.BlitFSStage)
	defer blit.Close()

	target := renderTexture(t, d, "target", w, h)

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole2D(t, target), Load: accel.LoadClear}},
		Width: w, Height: h, Label: "feedback",
	})
	p.SetPipeline(blit)
	// The same texture the pass renders into.
	p.SetTexture(0, whole2D(t, target))
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err == nil {
		g.Close()
		t.Fatal("a pass whose fragment stage fetches its own colour attachment built; " +
			"the result of that draw is undefined on every target, and it is a picture " +
			"that happens to be right on the machine it was written on")
	}
	for _, want := range []string{"feedback", "colour attachment 0"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("refused with %q, which does not mention %q; the message has to "+
				"name both views or a caller cannot tell which binding to move", err, want)
		}
	}
}

// Fixed-function read-modify-write is not feedback.
//
// LoadKeep, blending and the depth test all read the attachment, and they are
// ordered by the raster operations rather than by the shader. A rule that
// looked at "does this pass read what it writes" instead of at what the *stage*
// fetches would reject every blended or depth-tested pass ever written — which
// is the accepting half, and the half that decides whether the rule can be
// built at all.
func TestFixedFunctionAttachmentReadsAreNotFeedback(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)

	solid := texturePipeline(t, d, &testkernels.FullScreenVSStage,
		&testkernels.SolidFSStage)
	defer solid.Close()

	target := renderTexture(t, d, "target", w, h)

	r := d.NewRecorder()
	// LoadKeep reads what was in the attachment, and two draws blend over each
	// other. Both are reads of the thing this pass writes.
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole2D(t, target), Load: accel.LoadKeep}},
		Width: w, Height: h, Label: "blended",
	})
	p.SetPipeline(solid)
	p.Draw(accel.Draw{VertexCount: 3})
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("a pass that keeps its attachment's contents and draws over them was "+
			"refused: %v. The raster operations order that read, and refusing it would "+
			"reject every blended pass there is", err)
	}
	g.Close()
}

// Binding a texture the pass does not write is not feedback either.
//
// The companion to the case above: this is the ordinary two-pass shape, and it
// says the rule compares the fetched view against *this* pass's attachments
// rather than against every texture in the graph.
func TestFetchingAnotherPassesTargetIsNotFeedback(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)

	draw := texturePipeline(t, d, &testkernels.FullScreenVSStage,
		&testkernels.SolidFSStage)
	defer draw.Close()
	blit := texturePipeline(t, d, &testkernels.FullScreenVSStage,
		&testkernels.BlitFSStage)
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
		t.Fatalf("reading an earlier pass's target was refused as feedback: %v", err)
	}
	g.Close()
}

// A disjoint subresource is legal, and this test skips until one can be built.
//
// specs/033-render-api.md §3.3 permits a different mip or array layer, because
// it is different storage. specs/045-texture-attachments.md §8.3 still refuses
// MipLevels above one, so today every view of a texture names mip 0 and the
// permission cannot be exercised.
//
// The rule compares ranges anyway rather than texture handles, because writing
// the degenerate form would make the day mips land the day this silently starts
// refusing legal draws. But an untested accepting half is the shape two rules
// in this project were withdrawn in — V23, and 033 §6's undeclared-slot rule —
// so this exists and **self-activates**: the day a second mip is admissible it
// stops skipping and asserts the permission, rather than waiting for someone to
// remember that it should.
func TestADisjointSubresourceIsNotFeedback(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)

	tex, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.RGBA32Float, Size: accel.Extent{Width: w, Height: h},
		MipLevels: 2,
		Usage: accel.TextureRenderTarget | accel.TextureSampled |
			accel.TextureCopySrc | accel.TextureCopyDst,
		Kind: accel.MemoryReadback, Label: "mipped",
	})
	if err != nil {
		// It self-activated on 2026-08-30 and the skip is gone with the
		// refusal it watched for. A skip that can never fire is a comparison
		// nobody notices is not happening, so a two-mip texture being refused
		// again is a failure rather than a quiet pass.
		t.Fatalf("a texture with two mips was refused, so the permission this test "+
			"exists for is unreachable again: %v", err)
	}
	defer tex.Close()

	attachment, err := tex.View(accel.TextureViewDesc{Mip: 0})
	if err != nil {
		t.Fatalf("mip 0 view: %v", err)
	}
	fetched, err := tex.View(accel.TextureViewDesc{Mip: 1})
	if err != nil {
		t.Fatalf("mip 1 view: %v", err)
	}

	blit := texturePipeline(t, d, &testkernels.FullScreenVSStage,
		&testkernels.BlitFSStage)
	defer blit.Close()

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: attachment, Load: accel.LoadClear}},
		Width: w, Height: h, Label: "disjoint",
	})
	p.SetPipeline(blit)
	p.SetTexture(0, fetched)
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("writing mip 0 while fetching mip 1 of the same texture was refused "+
			"as feedback: %v. They are different storage, and refusing this is the "+
			"case comparing texture handles instead of view ranges gets wrong", err)
	}
	g.Close()
}

// A draw parameterised by a UniformBuffer, which is not yet possible.
//
// specs/042-surface-completion.md §3.1 records this as the one instance of "an
// exported declaration that reaches nothing" that this project shipped itself:
// UniformBuffer allocates, encodes correctly, and hands back a BufferView no
// draw takes. The mechanism is specs/033-render-api.md §6's draw at a recorded
// byte offset, which that spec's deviation 1 removed and did not replace.
//
// It is a test rather than a sentence for the reason the same day's audit
// found twice over: a gap recorded in prose has no accepting half, so nothing
// makes it resume and nothing notices when it closes. This skips with the
// reason and **self-activates** the day a draw can name a uniform offset, at
// which point it is the first caller of that channel rather than a rewrite
// somebody has to remember to do.
//
// The skip is deliberately not conditional on a feature flag. It reads the
// surface: if RenderPass ever grows a call taking a BufferView for a stage
// uniform, this stops describing the library and should be written out.
func TestADrawCanBeParameterisedByAUniformBuffer(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)

	ub, err := accel.NewUniformBuffer[testkernels.StageTint](d, testkernels.StageTintCodec{})
	if err != nil {
		t.Fatalf("NewUniformBuffer: %v", err)
	}
	defer ub.Close()

	want := accel.Vec4{0.125, 0.25, 0.5, 1}
	if err := ub.Write(d.Queue(), testkernels.StageTint{Colour: want}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	view, err := ub.View()
	if err != nil {
		t.Fatalf("View: %v", err)
	}

	pipe := texturePipeline(t, d, &testkernels.FullScreenVSStage, &testkernels.TintedFSStage)
	defer pipe.Close()
	tex := renderTexture(t, d, "target", w, h)

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole2D(t, tex), Load: accel.LoadClear}},
		Width: w, Height: h, Label: "uniformed",
	})
	p.SetPipeline(pipe)
	p.SetFragmentUniformBuffer(0, view)
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	got := readRenderTexture(t, d, tex)
	for i := range got {
		if got[i] != want[i%4] {
			t.Fatalf("element %d is %v, want %v: the stage did not read the block the "+
				"buffer holds", i, got[i], want[i%4])
		}
	}
}
