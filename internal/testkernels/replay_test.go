// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// specs/035-cpu-rasterizer.md section 7's "per-object replay" entry, and the
// case specs/033-render-api.md section 4.1 says decides whether the design
// works under specs/003-command-graph.md's immutability.
//
// N objects are recorded once, each with a fixed byte offset into one uniform
// buffer. The transforms are then rewritten and the *same graph* is submitted
// again. Nothing is re-recorded, and the second image is the one the second set
// of transforms asks for.
//
// # The stride is the device's, not the struct's
//
// Section 4.1 is explicit and this is where it bites: a caller who wrote
// i*sizeof(T) gets garbage for every object but the first on any device whose
// MinUniformBufferOffsetAlignment exceeds the block. Device.UniformStride is
// what computes it, and binding at the wrong stride is caught here as a colour
// that belongs to another object.
func TestObjectsReplayAtRecordedUniformOffsets(t *testing.T) {
	const w, h, objects = 8, 8, 3
	t.Run("cpu", func(t *testing.T) {
		d, err := accel.OpenCPU(accel.CPUOptions{})
		if err != nil {
			t.Fatalf("OpenCPU: %v", err)
		}
		defer d.Close()
		checkPerObjectReplay(t, d, w, h, objects)
	})
}

// checkPerObjectReplay records N objects at recorded offsets, submits, rewrites
// the transforms, and submits the same graph again.
func checkPerObjectReplay(t *testing.T, d *accel.Device, w, h, objects int) {
	{
		t.Helper()
		q := d.Queue()

		codec := testkernels.StageTintCodec{}
		stride := d.UniformStride(codec.EncodedSize())
		if stride < codec.EncodedSize() {
			t.Fatalf("the stride is %d and a block is %d", stride, codec.EncodedSize())
		}

		buf, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.U8, Count: stride * objects,
			Usage: accel.BufferUniform | accel.BufferCopyDst | accel.BufferCopySrc,
			Label: "transforms",
		})
		if err != nil {
			t.Fatalf("buffer: %v", err)
		}
		defer buf.Close()

		// The transforms live in their own buffer at their own recorded
		// offsets, so both stages read a uniform from a buffer -- which is
		// section 4.1's actual scene, and is what exercises the vertex side of
		// the binding as well as the fragment one.
		xf := testkernels.StageTransformCodec{}
		xfStride := d.UniformStride(xf.EncodedSize())
		xfBuf, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.U8, Count: xfStride * objects,
			Usage: accel.BufferUniform | accel.BufferCopyDst | accel.BufferCopySrc,
			Label: "xf",
		})
		if err != nil {
			t.Fatalf("transform buffer: %v", err)
		}
		defer xfBuf.Close()

		// Each object covers a different quarter, so which colour landed where
		// says which offset each draw read.
		quarters := []accel.Vec2{{-0.5, -0.5}, {0.5, -0.5}, {-0.5, 0.5}}
		xfRaw := make([]byte, xfStride*objects)
		for i, q := range quarters {
			if err := xf.Encode(xfRaw[i*xfStride:(i+1)*xfStride],
				testkernels.StageTransform{Scale: 0.5, Offset: q}); err != nil {
				t.Fatalf("encode transform %d: %v", i, err)
			}
		}
		if err := q.WriteBuffer(xfBuf, 0, xfRaw); err != nil {
			t.Fatalf("write transforms: %v", err)
		}

		write := func(colours []accel.Vec4) {
			raw := make([]byte, stride*objects)
			for i, c := range colours {
				if err := codec.Encode(raw[i*stride:(i+1)*stride],
					testkernels.StageTint{Colour: c}); err != nil {
					t.Fatalf("encode %d: %v", i, err)
				}
			}
			if err := q.WriteBuffer(buf, 0, raw); err != nil {
				t.Fatalf("write: %v", err)
			}
		}

		first := []accel.Vec4{{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}}
		second := []accel.Vec4{{0, 0, 1, 1}, {1, 0, 0, 1}, {0, 1, 0, 1}}
		write(first)

		pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
			Vertex:   &testkernels.ScaledVSStage,
			Fragment: &testkernels.TintedFSStage,
			VertexBuffers: []accel.VertexBufferLayout{{
				Stride:     12,
				Attributes: []accel.VertexAttribute{{Location: 0, Format: accel.AttrFloat32x3}},
			}},
			Targets: []accel.ColorTargetState{{Format: accel.RGBA32Float}},
			Label:   "objects",
		})
		if err != nil {
			t.Fatalf("pipeline: %v", err)
		}
		defer pipe.Close()

		// A quad in a quarter-sized box, moved per object by its transform.
		quad := []float32{
			-1, -1, 0, 1, -1, 0, 1, 1, 0,
			-1, -1, 0, 1, 1, 0, -1, 1, 0,
		}
		vb, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: len(quad),
			Usage: accel.BufferVertex | accel.BufferCopyDst, Label: "quad",
		})
		if err != nil {
			t.Fatalf("quad: %v", err)
		}
		defer vb.Close()
		if err := q.WriteBuffer(vb, 0, quad); err != nil {
			t.Fatalf("write quad: %v", err)
		}
		vv, err := vb.View(0, vb.Count())
		if err != nil {
			t.Fatalf("view: %v", err)
		}

		target := colourTexture(t, d, "objects", w, h)
		r := d.NewRecorder()
		p := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{
				View: wholeOf(t, target), Load: accel.LoadClear,
			}},
			Width: w, Height: h, Label: "objects",
		})
		p.SetPipeline(pipe)
		p.SetVertexBuffer(0, vv)
		for i := range objects {
			// The offset is structure and is baked in here. The bytes at that
			// offset are variation and change between submissions.
			view, err := buf.View(i*stride, codec.EncodedSize())
			if err != nil {
				t.Fatalf("uniform view %d: %v", i, err)
			}
			p.SetFragmentUniformBuffer(0, view)
			xfView, err := xfBuf.View(i*xfStride, xf.EncodedSize())
			if err != nil {
				t.Fatalf("transform view %d: %v", i, err)
			}
			p.SetVertexUniformBuffer(0, xfView)
			p.Draw(accel.Draw{VertexCount: 6})
		}

		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()

		read := func() []float32 {
			if err := q.Submit(g).Wait(); err != nil {
				t.Fatalf("submit: %v", err)
			}
			return readColourTexture(t, d, target)
		}

		before := read()

		// Every object's colour must be somewhere in the image. Without this
		// the test passes with every draw reading offset zero: all three
		// objects then hold object 0's colour, the rewrite rotates it the same
		// way, and the permutation check below agrees with itself.
		present := map[accel.Vec4]bool{}
		for at := range w * h {
			present[colourOf(before, at)] = true
		}
		for i, c := range first {
			if !present[c] {
				t.Fatalf("object %d's colour %v is nowhere in the image: the draws are "+
					"reading one offset rather than their own", i, c)
			}
		}

		write(second)
		after := read()

		// The same graph, replayed. Every object's quarter must now hold the
		// colour the second write put at *its* offset -- which is another
		// object's first colour, so a draw reading the wrong offset produces a
		// picture that is right for the wrong reason unless the permutation is
		// checked.
		var changed int
		for i := range before {
			if before[i] != after[i] {
				changed++
			}
		}
		if changed == 0 {
			t.Fatal("rewriting the buffer changed nothing, so the draws are not reading it")
		}
		// And the second image is the first one's colours permuted, since the
		// second set is the first rotated: every pixel that was object i's is
		// now object i-1's colour.
		var checked int
		for at := range w * h {
			b, a := colourOf(before, at), colourOf(after, at)
			if b == (accel.Vec4{}) {
				continue
			}
			for i := range objects {
				if b == first[i] {
					if a != second[i] {
						t.Fatalf("pixel %d was object %d's %v and is now %v, want %v",
							at, i, b, a, second[i])
					}
					checked++
				}
			}
		}
		if checked == 0 {
			t.Fatal("no pixel held any object's colour, so the replay proved nothing")
		}
	}

}

// colourOf is one pixel of a readback.
func colourOf(px []float32, at int) accel.Vec4 {
	return accel.Vec4{px[at*4], px[at*4+1], px[at*4+2], px[at*4+3]}
}
