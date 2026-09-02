// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
)

// Binding a texture to a stage, per specs/032-stage-abi.md section 5.

// A texture slot outside what a stage can hold is a diagnostic, not a panic.
//
// SetVertexBuffer's rule, and it is here for that method's reason rather than
// by analogy: a negative slot skips the grow loop and indexes the slice, which
// takes the caller's process down from inside a recording call. Every other
// method on RenderPass reports through the recorder, and a panic is the one
// diagnostic a caller cannot handle, cannot attribute to a slot, and cannot see
// beside the rest of a build's errors.
func TestATextureSlotOutsideWhatAStageHoldsIsRefused(t *testing.T) {
	d := openDevice(t)
	limit := d.Info().Limits.MaxTexturesPerStage
	if limit <= 0 {
		t.Fatal("the device reports no per-stage texture limit, so nothing constrains a slot")
	}
	for _, c := range []struct {
		name string
		slot int
		says string
	}{
		{"negative", -1, "cannot be negative"},
		// The ceiling is the device's Limits.MaxTexturesPerStage rather
		// than a constant, and the refusal names the device.
		{"at the ceiling", limit, d.Info().Name},
		{"far past the ceiling", 4096, "the ceiling"},
	} {
		t.Run(c.name, func(t *testing.T) {
			target := colourTarget(t, d, "colour", 4, 4)
			src := colourTarget(t, d, "src", 4, 4)

			r := d.NewRecorder()
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, target)}},
				Width: 4, Height: 4, Label: "slots",
			})
			// No recover(): a panic here fails the test by taking it down,
			// which is the behaviour this rule prevents.
			p.SetTexture(c.slot, view(t, src))

			_, err := r.Build()
			if err == nil {
				t.Fatalf("texture slot %d was accepted", c.slot)
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Fatalf("the refusal should say %q, got %v", c.says, err)
			}
			// And it names the slot, so a caller with several knows which.
			if !strings.Contains(err.Error(), fmt.Sprint(c.slot)) {
				t.Errorf("the refusal should name slot %d: %v", c.slot, err)
			}
		})
	}
}

// The last slot a stage can hold is accepted.
//
// The accepting half of the ceiling, and it is not decoration: a rule written
// as ">" where it meant ">=" refuses nothing a caller writes, and a rule
// written the other way refuses the highest legal slot. Only a test at the
// boundary tells the two apart, and this project has withdrawn three rules
// that were never tested from the accepting side.
func TestTheHighestTextureSlotIsAccepted(t *testing.T) {
	d := openDevice(t)
	target := colourTarget(t, d, "colour", 4, 4)
	src := colourTarget(t, d, "src", 4, 4)

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: view(t, target)}},
		Width: 4, Height: 4, Label: "ceiling",
	})
	p.SetPipeline(solidPipeline(t, d))
	p.SetTexture(15, view(t, src))
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("slot 15 is the highest a stage can hold and it was refused: %v", err)
	}
	defer g.Close()
}

// A second pass reads what the first pass drew.
//
// This is the capability the whole texel-fetch chain exists for: deferred
// shading, shadow maps, post-processing and material textures are all one pass
// reading another's output. Until a pass could bind a texture, accel could
// write multiple attachments and nothing could read them back in a stage.
//
// BlitFS fetches at its own pixel coordinate, so the second pass reproduces the
// first pass's image exactly. Exactly, not within a budget: a fetch returns the
// texel, and a copy that is off by a row or a channel is a different picture
// rather than a rounder one.
//
// The first pass draws RowFS, which writes each pixel's own coordinate, rather
// than a solid colour: a solid image is equal to itself flipped, transposed or
// shifted by a row, so a fetch with any of those faults reproduced it exactly.
func TestAPassReadsWhatAnEarlierPassDrew(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)

	draw := texturePipeline(t, d, &kernels.FullScreenVSStage,
		&kernels.RowFSStage)
	defer draw.Close()
	blit := texturePipeline(t, d, &kernels.FullScreenVSStage,
		&kernels.BlitFSStage)
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
	defer g.Close()

	// The second pass reads what the first wrote, so the graph must order them.
	// A test that only compared pixels could not tell a correct order from a
	// lucky one.
	if g.Hazards() == 0 {
		t.Error("the two passes share a texture and one writes it; that is a hazard")
	}
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	a := readRenderTexture(t, d, first)
	b := readRenderTexture(t, d, second)
	for p := range w * h {
		x, y := p%w, p/w
		// RowFS writes (x+0.5, y+0.5, 0, 1) at pixel (x, y), in the caller's
		// top-origin row order. This is what makes the comparison below
		// about the fetch: the first image is known, not merely non-zero.
		if want := [4]float32{float32(x) + 0.5, float32(y) + 0.5, 0, 1}; [4]float32(a[p*4:p*4+4]) != want {
			t.Fatalf("pixel (%d,%d) of the first pass is %v, want %v", x, y, a[p*4:p*4+4], want)
		}
		if [4]float32(b[p*4:p*4+4]) != [4]float32(a[p*4:p*4+4]) {
			t.Fatalf("pixel (%d,%d) is %v in the second pass and %v in the first",
				x, y, b[p*4:p*4+4], a[p*4:p*4+4])
		}
	}
}

func texturePipeline(t *testing.T, d *accel.Device, vs, fs *accel.Stage) *accel.RenderPipeline {
	t.Helper()
	p, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex: vs, Fragment: fs,
		Targets: []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		Label:   "texture",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	return p
}

func renderTexture(t *testing.T, d *accel.Device, label string, w, h int) *accel.Texture {
	t.Helper()
	tex, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.RGBA32Float, Size: accel.Extent{Width: w, Height: h},
		Usage: accel.TextureRenderTarget | accel.TextureSampled |
			accel.TextureCopySrc | accel.TextureCopyDst,
		Kind: accel.MemoryReadback, Label: label,
	})
	if err != nil {
		t.Fatalf("texture %s: %v", label, err)
	}
	t.Cleanup(func() { _ = tex.Close() })
	return tex
}

func whole2D(t *testing.T, tex *accel.Texture) accel.TextureView {
	t.Helper()
	v, err := tex.Whole()
	if err != nil {
		t.Fatalf("view: %v", err)
	}
	return v
}

func readRenderTexture(t *testing.T, d *accel.Device, tex *accel.Texture) []float32 {
	t.Helper()
	sz := tex.Size()
	raw := make([]byte, sz.Width*sz.Height*tex.Format().BytesPerPixel())
	if err := d.Queue().ReadTexture(tex, raw); err != nil {
		t.Fatalf("read texture: %v", err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 |
			uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24)
	}
	return out
}

// A clear value with a load op that never clears is refused rather than
// discarded.
//
// specs/033-render-api.md section 3.1 names this as an error and section 6's
// table lists it at graph build; neither existed, so the value was appended to
// the plan unconditionally and dropped by both backends. The caller wrote what
// they wanted the attachment to start as and got whatever was there.
func TestAClearValueWithoutLoadClearIsRefused(t *testing.T) {
	const w, h = 4, 4
	d := openDevice(t)
	tex := renderTexture(t, d, "target", w, h)
	depth, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.Depth32Float, Size: accel.Extent{Width: w, Height: h},
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

	for _, c := range []struct {
		name string
		desc accel.RenderPassDescriptor
		want string
	}{
		{
			name: "colour clear with LoadKeep",
			desc: accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{
					View: whole2D(t, tex), Load: accel.LoadKeep,
					Clear: [4]float32{1, 0, 0, 1},
				}},
				Width: w, Height: h, Label: "keep",
			},
			want: "colour 0 sets a clear value and loads",
		},
		{
			name: "depth clear with LoadDontCare",
			desc: accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{
					View: whole2D(t, tex), Load: accel.LoadClear,
				}},
				Depth: &accel.DepthAttachment{
					View: dv, Load: accel.LoadDontCare, Clear: 1,
				},
				Width: w, Height: h, Label: "dontcare",
			},
			want: "depth sets a clear value and loads",
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			r := d.NewRecorder()
			r.RenderPass(c.desc)
			_, err := r.Build()
			if err == nil {
				t.Fatal("the pass was accepted, so the clear value is silently discarded")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("the refusal does not say %q: %v", c.want, err)
			}
		})
	}

	// The accepting half, and it is the one that constrains the rule: a zero
	// clear value is also the field's zero value, so a pass that says nothing
	// about clearing must still be legal with any load op.
	pipe := texturePipeline(t, d, &kernels.FullScreenVSStage, &kernels.SolidFSStage)
	defer pipe.Close()
	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole2D(t, tex), Load: accel.LoadKeep}},
		Width: w, Height: h, Label: "silent",
	})
	p.SetPipeline(pipe)
	p.Draw(accel.Draw{VertexCount: 3})
	if _, err := r.Build(); err != nil {
		t.Errorf("a pass with no clear value and LoadKeep was refused: %v", err)
	}
}
