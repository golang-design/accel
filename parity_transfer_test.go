// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/parity"
	"golang.design/x/accel/internal/testkernels"
)

// specs/062-backend-parity.md sections 6.1, 6.2 and 6.8: every dtype and every
// format, compared between the two backends rather than run on one.

// dtypeParityCases is one case per dtype: a host write, a recorded copy, and a
// readback, compared byte for byte.
//
// Byte for byte and not within anything. A transfer moves bytes and never
// converts, so a byte that comes back different is a defect rather than a
// rounding difference, and a tolerance here would hide the one class of bug
// this path has. The copy is in the middle because a dtype whose *storage*
// worked and whose element stride was wrong would round-trip through the queue
// and corrupt the moment a graph moved it.
func dtypeParityCases() []parityCase {
	return []parityCase{
		dtypeCase(accel.F32, func(i int) float32 { return float32(i)*1.5 - 8 }),
		dtypeCase(accel.F16, func(i int) uint16 { return uint16(i)*0x0102 + 3 }),
		dtypeCase(accel.BF16, func(i int) uint16 { return uint16(i)*0x0301 + 9 }),
		dtypeCase(accel.I32, func(i int) int32 { return int32(i)*7 - 100 }),
		dtypeCase(accel.U32, func(i int) uint32 { return uint32(i)*0x01020304 + 5 }),
		dtypeCase(accel.I8, func(i int) int8 { return int8(i) - 32 }),
		dtypeCase(accel.U8, func(i int) uint8 { return uint8(i) + 7 }),
	}
}

// dtypeCase builds the case for one dtype.
//
// The patterns have distinct bytes wherever the width allows, so a transposed
// or truncated element is visible rather than accidentally equal to itself.
func dtypeCase[T comparable](dt accel.DType, at func(int) T) parityCase {
	return parityCase{
		name:   "a " + dt.String() + " buffer through a copy",
		covers: parity.Covers{"DType." + dtypeConstName(dt)},
		run: func(t *testing.T, d *accel.Device) []byte {
			t.Helper()
			const n = 64
			usage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
			return dtypeRoundTrip(t, d, dt, n, usage, at)
		},
	}
}

// dtypeRoundTrip writes, copies through a graph, and reads back at an offset.
//
// The offset read is not decoration. specs/001-device-resources.md section 3.2
// makes a storage buffer a tightly packed array of one dtype, so the element
// stride is the dtype's size; a read that ignored its offset passes at zero and
// fails on every real suballocation.
func dtypeRoundTrip[T comparable](t *testing.T, d *accel.Device, dt accel.DType,
	n int, usage accel.BufferUsage, at func(int) T) []byte {
	t.Helper()

	mk := func(label string) *accel.Buffer {
		b, err := d.NewBuffer(accel.BufferDescriptor{
			DType: dt, Count: n, Usage: usage, Label: label,
		})
		if err != nil {
			t.Fatalf("buffer %s: %v", label, err)
		}
		t.Cleanup(func() { _ = b.Close() })
		return b
	}
	src, dst := mk(dt.String()+" src"), mk(dt.String()+" dst")

	values := make([]T, n)
	for i := range values {
		values[i] = at(i)
	}
	if err := d.Queue().WriteBuffer(src, 0, values); err != nil {
		t.Fatalf("write %v: %v", dt, err)
	}

	r := d.NewRecorder()
	r.CopyBuffer(whole(t, dst), whole(t, src))
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build %v: %v", dt, err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit %v: %v", dt, err)
	}

	const off = 16
	all, tail := make([]T, n), make([]T, n-off)
	if err := d.Queue().ReadBuffer(dst, 0, all); err != nil {
		t.Fatalf("read %v: %v", dt, err)
	}
	if err := d.Queue().ReadBuffer(dst, off, tail); err != nil {
		t.Fatalf("read %v at %d: %v", dt, off, err)
	}

	var out bytes.Buffer
	if err := binary.Write(&out, binary.LittleEndian, all); err != nil {
		t.Fatalf("encode %v: %v", dt, err)
	}
	if err := binary.Write(&out, binary.LittleEndian, tail); err != nil {
		t.Fatalf("encode %v tail: %v", dt, err)
	}
	return out.Bytes()
}

// dtypeConstName is the constant's Go name, which is what the surface
// enumeration holds. DType.String is the dtype's spelling in a diagnostic --
// "f32" -- and the two are deliberately different.
func dtypeConstName(dt accel.DType) string {
	switch dt {
	case accel.F32:
		return "F32"
	case accel.F16:
		return "F16"
	case accel.BF16:
		return "BF16"
	case accel.I32:
		return "I32"
	case accel.U32:
		return "U32"
	case accel.I8:
		return "I8"
	case accel.U8:
		return "U8"
	}
	return "DType(" + dt.String() + ")"
}

// formatParityCases renders one solid full-viewport triangle into an
// attachment of each format and compares the texels byte for byte.
//
// Byte for byte across nine formats is a stronger bar than it looks, and it is
// the right one: the fragment writes constants, so nothing interpolates and
// there is no arithmetic for two rasterizers to round differently. What is left
// is the encoding -- the unorm quantisation rule, the sRGB transfer function,
// the channel order of BGRA, the half-float rounding -- and those must agree
// exactly or a caller's image differs by backend.
//
// This enumeration is what found the missing Metal pixel formats: the format
// table marks all nine renderable, the CPU rasterizer renders all nine, and
// five of them were refused at submit on Metal because their constants were
// absent from internal/mtl. A device reporting a format renderable and then
// refusing it is a capability answer that is not true.
func formatParityCases() []parityCase {
	const w, h = 4, 4
	colour := []accel.Format{
		accel.RGBA8Unorm, accel.RGBA8UnormSRGB, accel.BGRA8Unorm,
		accel.R16Float, accel.RG16Float, accel.RGBA16Float,
		accel.R32Float, accel.RG32Float, accel.RGBA32Float,
	}
	cases := make([]parityCase, 0, len(colour)+1)
	for _, f := range colour {
		cases = append(cases, parityCase{
			name:   "a solid draw into " + f.String(),
			covers: parity.Covers{"Format." + f.String()},
			run: func(t *testing.T, d *accel.Device) []byte {
				t.Helper()
				tex := newTexture(t, d, f.String(), w, h, f,
					accel.TextureRenderTarget|accel.TextureCopySrc|accel.TextureCopyDst,
					accel.MemoryReadback)
				pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
					Vertex:   &testkernels.FullScreenVSStage,
					Fragment: &testkernels.SolidFSStage,
					Targets:  []accel.ColorTargetState{{Format: f}},
					Label:    "parity " + f.String(),
				})
				if err != nil {
					t.Fatalf("pipeline for %v: %v", f, err)
				}
				defer pipe.Close()

				r := d.NewRecorder()
				p := r.RenderPass(accel.RenderPassDescriptor{
					Color: []accel.ColorAttachment{{
						View: view(t, tex), Load: accel.LoadClear,
						Clear: [4]float32{0, 0, 0, 0},
					}},
					Width: w, Height: h, Label: "parity " + f.String(),
				})
				p.SetPipeline(pipe)
				p.Draw(accel.Draw{VertexCount: 3})
				submitOne(t, d, r)
				return readTargetBytes(t, d, tex)
			},
		})
	}

	// The depth aspect, read through a recorded copy: a depth format is not
	// host-copyable on either backend, which is itself a parity claim.
	cases = append(cases, parityCase{
		name:   "a depth attachment written and copied back",
		covers: parity.Covers{"Format.Depth32Float"},
		run: func(t *testing.T, d *accel.Device) []byte {
			t.Helper()
			return f32Bytes(depthOfTwoFlatTriangles(t, d))
		},
	})
	return cases
}

// formatParityExclusions are the three formats no case can compare, and why.
func formatParityExclusions() []parity.Excluded {
	return []parity.Excluded{
		{Name: "Format.FormatInvalid", Why: "the zero-value sentinel for an optional " +
			"format constraint, not a creatable format: there is nothing to render into"},
		{Name: "Format.Depth24PlusStencil8", Why: "the CPU backend refuses it by name " +
			"(internal/cpu/texel.go): \"24 plus\" has two defensible encodings and the " +
			"oracle will not assert one. A comparison needs an oracle"},
		{Name: "Format.Depth32FloatStencil8", Why: "the Metal backend does not lower " +
			"stencil state (internal/metal/render_darwin.go) and has no pixel format for " +
			"the stencil-bearing depth attachment. Delete this entry when it does"},
	}
}

// submitOne builds and submits a recorder's single graph.
func submitOne(t *testing.T, d *accel.Device, r *accel.Recorder) {
	t.Helper()
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
}

// f32Bytes packs float32 components little-endian, which is the encoding every
// other case returns and the one compareParity decodes.
func f32Bytes(v []float32) []byte {
	out := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(f))
	}
	return out
}
