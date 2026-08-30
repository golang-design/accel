// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"encoding/binary"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// The normalized integer vertex formats convert as specs/033-render-api.md
// states, and both backends produce the same numbers.
//
// # Why these formats exist
//
// Every real mesh packs normals, tangents and colours as bytes or shorts. With
// only the float32 widths a caller either quadruples their vertex bandwidth or
// cannot use the vertex path, which is what
// specs/042-surface-completion.md section 5.3 calls the CPU oracle setting the
// public API's ceiling.
//
// # What is asserted, and why exactly
//
// A conversion is a division by a constant, not an interpolation, so the two
// backends have nothing to weight differently and the comparison is exact. The
// values are chosen at the ends and at the awkward middle: 0 and 255 for
// unorm8, and -128 for snorm8, which is the one input where the clamp shows --
// -128/127 is below -1, and a backend that let it through produces a normal
// slightly too long on exactly one value, which is invisible in an image and
// wrong in a lighting term.
func TestNormalizedVertexAttributesConvertTheSameOnBothBackends(t *testing.T) {
	for _, c := range []struct {
		name   string
		format accel.AttrFormat
		raw    []byte
		want   [4]float32
	}{
		{
			name: "unorm8x4 at both ends", format: accel.AttrUnorm8x4,
			raw:  []byte{0, 255, 128, 1},
			want: [4]float32{0, 1, 128.0 / 255, 1.0 / 255},
		},
		{
			name: "snorm8x4 clamps its most negative value", format: accel.AttrSnorm8x4,
			raw:  []byte{0x80, 0x81, 0x7f, 0x00}, // -128, -127, 127, 0
			want: [4]float32{-1, -1, 1, 0},
		},
		{
			name: "unorm16x2", format: accel.AttrUnorm16x2,
			raw:  []byte{0x00, 0x00, 0xff, 0xff},
			want: [4]float32{0, 1, 0, 0},
		},
		{
			name: "snorm16x2 clamps", format: accel.AttrSnorm16x2,
			raw:  []byte{0x00, 0x80, 0xff, 0x7f}, // -32768, 32767
			want: [4]float32{-1, 1, 0, 0},
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			cpu, err := accel.OpenCPU(accel.CPUOptions{})
			if err != nil {
				t.Fatalf("OpenCPU: %v", err)
			}
			defer cpu.Close()
			got := fetchAttribute(t, cpu, c.format, c.raw)
			n := c.format.Components()
			for i := range n {
				if got[i] != c.want[i] {
					t.Errorf("component %d is %v, want %v", i, got[i], c.want[i])
				}
			}
			checkAttributeAgreesOnMetal(t, c.format, c.raw, got)
		})
	}
}

// fetchAttribute draws one vertex whose attribute holds raw, and returns the
// floats the stage received.
//
// The attribute reaches the fragment stage through the varyings, so what is
// read back is what the *stage* got rather than what the buffer held -- which
// is the conversion under test.
func fetchAttribute(t *testing.T, d *accel.Device, f accel.AttrFormat, raw []byte) [4]float32 {
	t.Helper()
	q := d.Queue()

	// The stage is chosen by the attribute's width, because pipeline creation
	// checks the declared format against the parameter's type: AttributeVS
	// takes a Vec4 at location 1 and GeometryVS a Vec2, and a mismatch is
	// refused -- correctly, since a fetch of the wrong width deforms geometry
	// rather than losing it.
	vs, fs := &testkernels.AttributeVSStage, &testkernels.TintFSStage
	targets := []accel.ColorTargetState{{Format: accel.RGBA32Float}}
	if f.Components() == 2 {
		vs, fs = &testkernels.GeometryVSStage, &testkernels.ShadeFSStage
		targets = append(targets, accel.ColorTargetState{Format: accel.RGBA32Float})
	}
	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   vs,
		Fragment: fs,
		VertexBuffers: []accel.VertexBufferLayout{{
			// Position first as float32x3, then the attribute under test.
			Stride: 12 + 16,
			Attributes: []accel.VertexAttribute{
				{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
				{Location: 1, Format: f, Offset: 12},
			},
		}},
		Targets: targets,
		Label:   "attribute",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	// Three vertices with the same attribute, so interpolation cannot change
	// it: a constant field interpolates to itself under any weights.
	stride := 12 + 16
	verts := make([]byte, stride*3)
	pos := [3][3]float32{{-1, -1, 0}, {3, -1, 0}, {-1, 3, 0}}
	for v := range 3 {
		for i, p := range pos[v] {
			binary.LittleEndian.PutUint32(verts[v*stride+i*4:], mathBits(p))
		}
		copy(verts[v*stride+12:], raw)
	}

	vb, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.U8, Count: len(verts),
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

	const w, h = 4, 4
	tex, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.RGBA32Float, Size: accel.Extent{Width: w, Height: h},
		Usage: accel.TextureRenderTarget | accel.TextureCopySrc | accel.TextureCopyDst,
		Kind:  accel.MemoryReadback, Label: "out",
	})
	if err != nil {
		t.Fatalf("texture: %v", err)
	}
	defer tex.Close()
	tv, err := tex.Whole()
	if err != nil {
		t.Fatalf("view: %v", err)
	}
	colour := []accel.ColorAttachment{{View: tv, Load: accel.LoadClear}}
	read := tex
	if len(targets) == 2 {
		// ShadeFS writes the interpolated colour to attachment 0 and the uv to
		// attachment 1, so the attribute under test is in the second one.
		second, err := d.NewTexture(accel.TextureDescriptor{
			Format: accel.RGBA32Float, Size: accel.Extent{Width: w, Height: h},
			Usage: accel.TextureRenderTarget | accel.TextureCopySrc | accel.TextureCopyDst,
			Kind:  accel.MemoryReadback, Label: "uv",
		})
		if err != nil {
			t.Fatalf("second texture: %v", err)
		}
		defer second.Close()
		sv, err := second.Whole()
		if err != nil {
			t.Fatalf("second view: %v", err)
		}
		colour = append(colour, accel.ColorAttachment{View: sv, Load: accel.LoadClear})
		read = second
	}

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: colour,
		Width: w, Height: h, Label: "attribute",
	})
	p.SetPipeline(pipe)
	p.SetVertexBuffer(0, vv)
	if len(targets) == 2 {
		p.SetVertexUniform(0, testkernels.StageTransform{
			Scale: 1, Offset: accel.Vec2{0, 0},
		})
	}
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	rawOut := make([]byte, w*h*read.Format().BytesPerPixel())
	if err := q.ReadTexture(read, rawOut); err != nil {
		t.Fatalf("read: %v", err)
	}
	px := float32sOf(rawOut)
	return [4]float32{px[0], px[1], px[2], px[3]}
}
