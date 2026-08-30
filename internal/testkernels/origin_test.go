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

// The origin corpus entry of specs/035-cpu-rasterizer.md section 7, in its
// discriminating form.
//
// # What it is for
//
// specs/005-graphics.md guarantees row 0 is the top row in three places -- the
// rasterizer's window mapping, an on-device texel fetch, and host readback --
// and says which of the three a backend corrects is the backend's choice, while
// their agreeing is not. Nothing checked that they agree.
// docs/conventions.md records the predecessor's actual bug here: its
// compute-path test passed while the texture path was mirrored.
//
// # The two paths, and why one must end in a buffer
//
//	A  device reads the texture at row 0 through a texel fetch, writes into a
//	   buffer-backed attachment, and the host reads the buffer
//	B  the host reads the texture and takes row 0
//
// Path A ends in a *buffer*, which has rows only by arithmetic and so has no
// origin convention to get wrong. Path B ends in a texture readback, which
// does. A mirrored readback moves both paths' answer to the other end of the
// image, so `A == B` alone would pass with both wrong -- which is why section 7
// requires the second assertion, that both hold the *top* row's value.
//
// # It is a fragment fetch rather than the compute one section 7 names
//
// specs/032-stage-abi.md section 5.1 refuses a texture in a compute kernel by
// name, deliberately: a dispatch's argument set cannot carry one, so a kernel
// declaring a texture would compile to a binding no caller could supply. The
// fragment path is the on-device fetch that exists, and it crosses the same
// seam -- the device's own view of the texture's rows against the host's.
func TestTheDeviceAndTheHostAgreeAboutRowZero(t *testing.T) {
	// Twelve wide so the row pitch pads, five tall so a flip cannot coincide
	// with the identity.
	const w, h = 12, 5
	t.Run("cpu", func(t *testing.T) {
		d, err := accel.OpenCPU(accel.CPUOptions{})
		if err != nil {
			t.Fatalf("OpenCPU: %v", err)
		}
		defer d.Close()
		checkRowZeroAgreement(t, d, w, h)
	})
}

// checkRowZeroAgreement runs the two paths on one device and compares them.
//
// Shared with the Metal case rather than duplicated: the assertion is about a
// convention, and a convention checked with two different fixtures is two
// conventions.
func checkRowZeroAgreement(t *testing.T, d *accel.Device, w, h int) {
	{
		t.Helper()
		q := d.Queue()

		encode, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
			Vertex:   &testkernels.FullScreenVSStage,
			Fragment: &testkernels.RowFSStage,
			Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
			Label:    "encode",
		})
		if err != nil {
			t.Fatalf("encode pipeline: %v", err)
		}
		defer encode.Close()

		fetch, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
			Vertex:   &testkernels.FullScreenVSStage,
			Fragment: &testkernels.TopRowFSStage,
			Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
			Label:    "fetch",
		})
		if err != nil {
			t.Fatalf("fetch pipeline: %v", err)
		}
		defer fetch.Close()

		target, err := d.NewTexture(accel.TextureDescriptor{
			Format: accel.RGBA32Float, Size: accel.Extent{Width: w, Height: h},
			Usage: accel.TextureRenderTarget | accel.TextureSampled |
				accel.TextureCopySrc | accel.TextureCopyDst,
			Kind: accel.MemoryReadback, Label: "rows",
		})
		if err != nil {
			t.Fatalf("texture: %v", err)
		}
		defer target.Close()
		tv, err := target.Whole()
		if err != nil {
			t.Fatalf("view: %v", err)
		}

		// Path A's destination is a buffer, reached through a slot-bound
		// attachment. One row tall, so the buffer holds exactly what the device
		// read out of the texture's row zero.
		out, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: w * 4,
			Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
			Label: "row zero",
		})
		if err != nil {
			t.Fatalf("buffer: %v", err)
		}
		defer out.Close()
		ov, err := out.View(0, out.Count())
		if err != nil {
			t.Fatalf("buffer view: %v", err)
		}

		r := d.NewRecorder()
		slot := r.Slot(accel.SlotDescriptor{
			Name: "row zero", Kind: accel.BindingStorageBuffer,
			DType: accel.F32, Access: accel.AccessWrite, MinCount: w * 4,
		})

		p1 := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{View: tv, Load: accel.LoadClear}},
			Width: w, Height: h, Label: "encode",
		})
		p1.SetPipeline(encode)
		p1.Draw(accel.Draw{VertexCount: 3})

		p2 := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{Slot: slot, Load: accel.LoadClear}},
			Width: w, Height: 1, Label: "fetch row zero",
		})
		p2.SetPipeline(fetch)
		p2.SetTexture(0, tv)
		p2.Draw(accel.Draw{VertexCount: 3})

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

		pathA := make([]float32, w*4)
		if err := q.ReadBuffer(out, 0, pathA); err != nil {
			t.Fatalf("read buffer: %v", err)
		}

		raw := make([]byte, w*h*target.Format().BytesPerPixel())
		if err := q.ReadTexture(target, raw); err != nil {
			t.Fatalf("read texture: %v", err)
		}
		whole := make([]float32, len(raw)/4)
		for i := range whole {
			whole[i] = math.Float32frombits(uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 |
				uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24)
		}
		pathB := whole[:w*4]

		// Exactly. A fetch is an indexed load and a readback is a copy, so a
		// difference here is a different texel rather than a rounder number.
		for i := range pathA {
			if pathA[i] != pathB[i] {
				t.Fatalf("column %d channel %d is %v from the device's fetch and %v from "+
					"the host's readback: the two disagree about which row is row zero",
					i/4, i%4, pathA[i], pathB[i])
			}
		}

		// And both hold the *top* row. Without this, a readback mirrored in
		// both paths agrees with itself: RowFS writes each pixel's own window
		// coordinate, so the top row's y is 0.5 and the bottom row's is h-0.5.
		for x := range w {
			gotX, gotY := pathA[x*4], pathA[x*4+1]
			if wantX := float32(x) + 0.5; gotX != wantX {
				t.Errorf("column %d reports x = %v, want %v: the row is transposed or "+
					"shifted rather than merely flipped", x, gotX, wantX)
			}
			if gotY != 0.5 {
				t.Errorf("column %d reports y = %v, want 0.5: row zero holds row %v's "+
					"value, so the origin is at the bottom on both paths",
					x, gotY, gotY-0.5)
			}
		}
	}

}

// float32sOf reinterprets little-endian bytes as float32s.
func float32sOf(raw []byte) []float32 {
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 |
			uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24)
	}
	return out
}

// mathBits is math.Float32bits, named so the attribute fixture reads as bytes.
func mathBits(f float32) uint32 { return math.Float32bits(f) }
