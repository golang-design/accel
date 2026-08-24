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

// specs/045-texture-attachments.md: an attachment is a texture view, the view's
// format decides how the bytes are read and written, and sRGB converts there.

// newByteBuffer is a buffer the host writes bytes into, for filling a texture
// whose texels are not float32.
func newByteBuffer(t *testing.T, d *accel.Device, label string, data []byte) *accel.Buffer {
	t.Helper()
	b, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.U8, Count: len(data), Label: label,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
	})
	if err != nil {
		t.Fatalf("new byte buffer %q: %v", label, err)
	}
	t.Cleanup(func() { _ = b.Close() })
	if err := d.Queue().WriteBuffer(b, 0, data); err != nil {
		t.Fatalf("write %q: %v", label, err)
	}
	return b
}

// fillBytes copies tightly packed bytes into a texture, which is the only way
// to give one contents: there is no host write to a texture.
func fillBytes(t *testing.T, d *accel.Device, tex *accel.Texture, data []byte) {
	t.Helper()
	src := newByteBuffer(t, d, "fill", data)
	r := d.NewRecorder()
	r.CopyBufferToTexture(tex, whole(t, src))
	g, err := r.Build()
	if err != nil {
		t.Fatalf("fill build: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("fill submit: %v", err)
	}
}

func unormTarget(t *testing.T, d *accel.Device, label string, w, h int) *accel.Texture {
	t.Helper()
	return newTexture(t, d, label, w, h, accel.RGBA8Unorm,
		accel.TextureRenderTarget|accel.TextureCopySrc|accel.TextureCopyDst,
		accel.MemoryReadback)
}

func unormPipeline(t *testing.T, d *accel.Device) *accel.RenderPipeline {
	t.Helper()
	p, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.HalfTriangleVSStage,
		Fragment: &testkernels.SolidFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA8Unorm}},
		Label:    "unorm solid",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })
	return p
}

// An RGBA8Unorm target holds unorm8 bytes, not float32.
//
// This is the symptom specs/042-surface-completion.md section 5.2 named first:
// the CPU backend reinterpreted attachment bytes as []float32 whatever the
// caller declared, so RGBA8Unorm was secretly RGBA32Float and no test could
// see it. The bytes are asserted rather than the components, because a decode
// and an encode that are both wrong the same way agree with each other.
//
// It also covers the row pitch. A 4-wide RGBA8Unorm row is sixteen bytes and
// the device pads rows to 256, so an implementation that stepped
// width*bytes-per-pixel reads fifteen rows of padding as texels.
func TestARenderTargetStoresItsOwnFormat(t *testing.T) {
	const w, h = 4, 4
	d := openDevice(t)
	target := unormTarget(t, d, "unorm", w, h)

	r := d.NewRecorder()
	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{
			View: view(t, target), Load: accel.LoadClear, Clear: [4]float32{1, 0, 0, 1},
		}},
		Width: w, Height: h, Label: "unorm",
	})
	pass.SetPipeline(unormPipeline(t, d))
	pass.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	got := readTargetBytes(t, d, target)
	if len(got) != w*h*4 {
		t.Fatalf("read %d bytes, want %d", len(got), w*h*4)
	}
	// SolidFS returns (0.25, 0.5, 0.75, 1), and unorm8 is v/255 rounded to
	// nearest: 63.75, 127.5, 191.25, 255. The clear is opaque red.
	covered := [4]byte{64, 128, 191, 255}
	cleared := [4]byte{255, 0, 0, 255}
	for y := range h {
		for x := range w {
			want := cleared
			if y > x {
				want = covered
			}
			px := [4]byte(got[(y*w+x)*4 : (y*w+x)*4+4])
			if px != want {
				t.Fatalf("pixel (%d,%d) is %v, want %v", x, y, px, want)
			}
		}
	}
}

// A pass writes only the render area, and the rest of a larger target is left
// alone.
//
// The accepting half of the extent check, and the one that reads the row pitch
// twice: the area is four texels of a row that is eight texels wide and 256
// bytes long, so a write that ignored either number lands in the wrong row.
func TestAPassWritesOnlyItsRenderArea(t *testing.T) {
	const tw, th = 8, 8
	const w, h = 4, 4
	d := openDevice(t)
	target := unormTarget(t, d, "larger", tw, th)

	prior := make([]byte, tw*th*4)
	for i := range prior {
		prior[i] = 0xAA
	}
	fillBytes(t, d, target, prior)

	r := d.NewRecorder()
	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{
			View: view(t, target), Load: accel.LoadClear, Clear: [4]float32{1, 0, 0, 1},
		}},
		Width: w, Height: h, Label: "corner",
	})
	pass.SetPipeline(unormPipeline(t, d))
	pass.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	got := readTargetBytes(t, d, target)
	for y := range th {
		for x := range tw {
			px := [4]byte(got[(y*tw+x)*4 : (y*tw+x)*4+4])
			if x >= w || y >= h {
				if px != [4]byte{0xAA, 0xAA, 0xAA, 0xAA} {
					t.Fatalf("pixel (%d,%d) is %v and lies outside the %dx%d render "+
						"area, so it should still hold what was there", x, y, px, w, h)
				}
				continue
			}
			want := [4]byte{255, 0, 0, 255}
			if y > x {
				want = [4]byte{64, 128, 191, 255}
			}
			if px != want {
				t.Fatalf("pixel (%d,%d) is %v, want %v", x, y, px, want)
			}
		}
	}
}

// sRGB converts on write and on read, and the view's format is what decides.
//
// specs/035-cpu-rasterizer.md section 5 has said so since it was written and
// nothing owned the conversion. specs/045-texture-attachments.md section 2.1
// gives it an owner: one texture, two views, the same bytes and different
// values. That is the case the view format exists for, and it is checked here
// against hand-computed IEC 61966-2-1 constants rather than against a round
// trip, because an encode and a decode that are both wrong agree.
//
// The pass adds the fragment's colour to what the attachment holds, so both
// directions are in one number: the destination has to be *decoded* to be added
// to, and the sum has to be *encoded* to be stored.
func TestAnSRGBViewConvertsOnWriteAndOnRead(t *testing.T) {
	const w, h = 4, 4

	// Byte 137 is the sRGB encoding of linear 0.25: 1.055*0.25^(1/2.4) - 0.055
	// is 0.53710, and 0.53710*255 is 136.96.
	prior := make([]byte, w*h*4)
	for i := range prior {
		prior[i] = 137
		if i%4 == 3 {
			prior[i] = 255
		}
	}

	// src + dst, so the destination is read as well as written. Alpha takes the
	// source only, which keeps it at one and out of the comparison.
	add := accel.BlendState{
		Enabled:  true,
		SrcColor: accel.FactorOne, DstColor: accel.FactorOne, ColorOp: accel.BlendAdd,
		SrcAlpha: accel.FactorOne, DstAlpha: accel.FactorZero, AlphaOp: accel.BlendAdd,
	}

	run := func(t *testing.T, format accel.Format) []byte {
		t.Helper()
		d := openDevice(t)
		q := d.Queue()
		target := unormTarget(t, d, "shared bytes", w, h)
		fillBytes(t, d, target, prior)

		pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
			Vertex:   &testkernels.ScaledVSStage,
			Fragment: &testkernels.TintedFSStage,
			VertexBuffers: []accel.VertexBufferLayout{{
				Stride: 12,
				Attributes: []accel.VertexAttribute{
					{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
				},
			}},
			Targets: []accel.ColorTargetState{{Format: format, Blend: add}},
			Label:   "adding",
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

		r := d.NewRecorder()
		pass := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{
				View: viewAs(t, target, format), Load: accel.LoadKeep,
			}},
			Width: w, Height: h, Label: "adding",
		})
		pass.SetPipeline(pipe)
		pass.SetVertexBuffer(0, whole(t, vb))
		pass.SetVertexUniform(0, testkernels.StageTransform{Scale: 1})
		pass.SetFragmentUniform(0, testkernels.StageTint{
			Colour: [4]float32{0.25, 0.25, 0.25, 1},
		})
		pass.Draw(accel.Draw{VertexCount: 3})

		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		if err := q.Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		return readTargetBytes(t, d, target)
	}

	// Through the sRGB view: 137 decodes to linear 0.2502, plus 0.25 is 0.5002,
	// and 1.055*0.5002^(1/2.4) - 0.055 is 0.73545, which stores as 188.
	srgb := run(t, accel.RGBA8UnormSRGB)
	// Through the texture's own linear view: 137/255 is 0.53725, plus 0.25 is
	// 0.78725, which stores as 201. Same bytes in, different bytes out.
	linear := run(t, accel.RGBA8Unorm)

	for i := range w * h {
		px := [4]byte(srgb[i*4 : i*4+4])
		if want := ([4]byte{188, 188, 188, 255}); px != want {
			t.Fatalf("through the sRGB view, texel %d is %v, want %v: sRGB converts on "+
				"write and on read, and the view's format is what decides", i, px, want)
		}
		lin := [4]byte(linear[i*4 : i*4+4])
		if want := ([4]byte{201, 201, 201, 255}); lin != want {
			t.Fatalf("through the linear view, texel %d is %v, want %v", i, lin, want)
		}
	}
}

// Every refusal an attachment's texture owns, and each names what is wrong.
//
// The two aspect cases depend on an ordering, and it is worth saying so where a
// reader will see it: check V13 would refuse the same two recordings, because a
// pipeline that agreed with a wrong-aspect attachment cannot be constructed --
// NewRenderPipeline rejects a depth format as a colour target and the reverse.
// The attachment check runs first and gives the better message, so these assert
// that message. If V13 ever moved ahead of it these would still pass, on V13's
// wording, with the check they exist for gone; the comment on checkAttachment
// says the same thing from the other side.
func TestTextureAttachmentRefusals(t *testing.T) {
	const w, h = 4, 4
	for _, c := range []struct {
		name   string
		record func(t *testing.T, d *accel.Device, r *accel.Recorder)
		says   string
	}{{
		name: "a texture that is not a render target",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder) {
			tex := newTexture(t, d, "sampled", w, h, accel.RGBA8Unorm,
				accel.TextureSampled|accel.TextureCopyDst, accel.MemoryDevice)
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, tex)}},
				Width: w, Height: h, Label: "sampled",
			})
			p.SetPipeline(unormPipeline(t, d))
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: "needs TextureRenderTarget and was created with",
	}, {
		name: "a depth format as a colour attachment",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder) {
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{
					View: view(t, depthTarget(t, d, "depth", w, h)),
				}},
				Width: w, Height: h, Label: "wrong aspect",
			})
			p.SetPipeline(unormPipeline(t, d))
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: "a colour attachment does not take a depth format",
	}, {
		name: "a colour format as a depth attachment",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder) {
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, unormTarget(t, d, "c", w, h))}},
				Depth: &accel.DepthAttachment{
					View: view(t, unormTarget(t, d, "not depth", w, h)),
				},
				Width: w, Height: h, Label: "wrong aspect",
			})
			p.SetPipeline(depthPipeline(t, d))
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: "a depth attachment takes a depth format",
	}, {
		name: "a device-defined layout",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder) {
			tex := newTexture(t, d, "packed", w, h, accel.Depth24PlusStencil8,
				accel.TextureRenderTarget, accel.MemoryDevice)
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, unormTarget(t, d, "c", w, h))}},
				Depth: &accel.DepthAttachment{View: view(t, tex)},
				Width: w, Height: h, Label: "packed",
			})
			p.SetPipeline(depthPipeline(t, d))
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: "whose layout is device-defined",
	}, {
		name: "a view naming no texture",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder) {
			r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{}},
				Width: w, Height: h, Label: "empty",
			})
		},
		says: "names no resource",
	}} {
		t.Run(c.name, func(t *testing.T) {
			d := openDevice(t)
			r := d.NewRecorder()
			c.record(t, d, r)
			g, err := r.Build()
			if err == nil {
				_ = g.Close()
				t.Fatal("Build accepted a recording it should refuse")
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Fatalf("the error should say %q, got %v", c.says, err)
			}
		})
	}
}

// Check V13: a pipeline compiled for one attachment format is refused against
// another, and the refusal names the pipeline, the node and the index.
//
// specs/003-command-graph.md has carried V13 since it was written and it could
// not be implemented while an attachment was a buffer view: there was no format
// on one side to compare. The index is asserted with a *second* attachment
// mismatching, because an off-by-one there is invisible with one attachment.
func TestV13RefusesAPipelineCompiledForAnotherFormat(t *testing.T) {
	const w, h = 4, 4
	for _, c := range []struct {
		name   string
		record func(t *testing.T, d *accel.Device, r *accel.Recorder)
		says   []string
	}{{
		name: "a single attachment of another format",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder) {
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, unormTarget(t, d, "8bit", w, h))}},
				Width: w, Height: h, Label: "mismatched",
			})
			// solidPipeline declares RGBA32Float.
			p.SetPipeline(solidPipeline(t, d))
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: []string{`the pipeline "solid"`, "colour target 0 as RGBA32Float",
			"attachment 0 is RGBA8Unorm", "check V13"},
	}, {
		name: "the second attachment of a pair",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder) {
			mrt, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
				Vertex:   &testkernels.GeometryVSStage,
				Fragment: &testkernels.ShadeFSStage,
				VertexBuffers: []accel.VertexBufferLayout{{
					Stride: 20,
					Attributes: []accel.VertexAttribute{
						{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
						{Location: 1, Format: accel.AttrFloat32x2, Offset: 12},
					},
				}},
				Targets: []accel.ColorTargetState{
					{Format: accel.RGBA32Float}, {Format: accel.RGBA32Float},
				},
				Label: "mrt",
			})
			if err != nil {
				t.Fatalf("pipeline: %v", err)
			}
			t.Cleanup(func() { _ = mrt.Close() })
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{
					{View: view(t, colourTarget(t, d, "first", w, h))},
					{View: view(t, unormTarget(t, d, "second", w, h))},
				},
				Width: w, Height: h, Label: "mrt",
			})
			p.SetPipeline(mrt)
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: []string{"colour target 1 as RGBA32Float", "attachment 1 is RGBA8Unorm"},
	}, {
		name: "a depth attachment of another format",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder) {
			pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
				Vertex:   &testkernels.HalfTriangleVSStage,
				Fragment: &testkernels.SolidFSStage,
				Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
				DepthStencil: &accel.DepthStencilState{
					Format: accel.Depth24PlusStencil8, Test: true, Write: true,
					Compare: accel.CompareLess,
				},
				Label: "packed depth",
			})
			if err != nil {
				t.Fatalf("pipeline: %v", err)
			}
			t.Cleanup(func() { _ = pipe.Close() })
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, colourTarget(t, d, "c", w, h))}},
				Depth: &accel.DepthAttachment{View: view(t, depthTarget(t, d, "z", w, h))},
				Width: w, Height: h, Label: "depth format",
			})
			p.SetPipeline(pipe)
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: []string{"declares depth as Depth24PlusStencil8",
			"the depth attachment is Depth32Float"},
	}, {
		name: "a pipeline compiled for the texture rather than the view",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder) {
			// The texture is RGBA8Unorm and the view reads it as sRGB, so the
			// writes go through sRGB and a pipeline compiled linear is wrong.
			// This is the case that proves V13 reads the view.
			tex := unormTarget(t, d, "linear texture", w, h)
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{
					View: viewAs(t, tex, accel.RGBA8UnormSRGB),
				}},
				Width: w, Height: h, Label: "view format",
			})
			p.SetPipeline(unormPipeline(t, d))
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: []string{"colour target 0 as RGBA8Unorm", "attachment 0 is RGBA8UnormSRGB"},
	}} {
		t.Run(c.name, func(t *testing.T) {
			d := openDevice(t)
			r := d.NewRecorder()
			c.record(t, d, r)
			g, err := r.Build()
			if err == nil {
				_ = g.Close()
				t.Fatal("Build accepted a pipeline compiled for another format")
			}
			for _, want := range append(c.says, "node") {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("the error should say %q, got %v", want, err)
				}
			}
		})
	}
}

// V13's accepting half, and the half that proves it reads the view.
//
// A pipeline compiled for RGBA8UnormSRGB against an RGBA8Unorm texture viewed
// as sRGB must build and must produce sRGB bytes. A check that compared the
// texture's own format would refuse this, which is the one case the view format
// exists for; a check that compared nothing would accept the previous case too.
func TestV13AcceptsAPipelineCompiledForTheViewsFormat(t *testing.T) {
	const w, h = 4, 4
	d := openDevice(t)
	tex := unormTarget(t, d, "linear texture", w, h)

	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.HalfTriangleVSStage,
		Fragment: &testkernels.SolidFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA8UnormSRGB}},
		Label:    "srgb solid",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	r := d.NewRecorder()
	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{
			View: viewAs(t, tex, accel.RGBA8UnormSRGB), Load: accel.LoadClear,
			Clear: [4]float32{0, 0, 0, 1},
		}},
		Width: w, Height: h, Label: "srgb",
	})
	pass.SetPipeline(pipe)
	pass.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("a pipeline compiled for the view's format was refused: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	// SolidFS returns linear (0.25, 0.5, 0.75, 1), which through an sRGB view
	// stores as 137, 188 and 225: 1.055*v^(1/2.4) - 0.055, times 255, rounded.
	got := readTargetBytes(t, d, tex)
	var covered bool
	for y := range h {
		for x := range w {
			px := [4]byte(got[(y*w+x)*4 : (y*w+x)*4+4])
			want := [4]byte{0, 0, 0, 255}
			if y > x {
				want = [4]byte{137, 188, 225, 255}
				covered = true
			}
			if px != want {
				t.Fatalf("pixel (%d,%d) is %v, want %v", x, y, px, want)
			}
		}
	}
	if !covered {
		t.Fatal("nothing was drawn, so the encoding proves nothing")
	}
}

// A view that reinterprets a texture's format is accepted as an attachment,
// and a view of a format outside the family is refused before it exists.
//
// The accepting half is the one that matters: the compatibility rule lives on
// Texture.View and a pass that refused what View accepted would make the field
// unusable. The refusal is checked at the view, which is where it belongs --
// a view outside a family is wrong whether or not anyone attaches it.
func TestAReinterpretingViewIsAnAttachmentAndAnIncompatibleOneIsNot(t *testing.T) {
	const w, h = 4, 4
	d := openDevice(t)
	target := unormTarget(t, d, "reinterpreted", w, h)

	if _, err := target.View(accel.TextureViewDesc{Format: accel.RGBA32Float}); err == nil {
		t.Error("an RGBA32Float view of an RGBA8Unorm texture was accepted; a view " +
			"reinterprets bytes and does not convert them")
	}

	r := d.NewRecorder()
	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.HalfTriangleVSStage,
		Fragment: &testkernels.SolidFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA8UnormSRGB}},
		Label:    "srgb solid",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{
			View: viewAs(t, target, accel.RGBA8UnormSRGB), Load: accel.LoadClear,
		}},
		Width: w, Height: h, Label: "srgb view",
	})
	pass.SetPipeline(pipe)
	pass.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("an sRGB view of a linear texture was refused as an attachment: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
}
