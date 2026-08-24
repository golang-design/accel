// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// interleaved is one vertex of the layout the tests below declare: three floats
// of position then four of colour, in one buffer at a stride of 28 bytes.
func interleaved(x, y, z float32, c [4]float32) []float32 {
	return []float32{x, y, z, c[0], c[1], c[2], c[3]}
}

func attributePipeline(t *testing.T, d *accel.Device) *accel.RenderPipeline {
	t.Helper()
	p, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.AttributeVSStage,
		Fragment: &testkernels.TintFSStage,
		VertexBuffers: []accel.VertexBufferLayout{{
			Stride: 28, StepMode: accel.StepVertex,
			Attributes: []accel.VertexAttribute{
				{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
				{Location: 1, Format: accel.AttrFloat32x4, Offset: 12},
			},
		}},
		Targets: []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		Label:   "attributes",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })
	return p
}

// A triangle whose positions and colours come from a vertex buffer, with the
// colour interpolated across it.
//
// This is the whole attribute path: a declared layout, a bound buffer, a fetch
// per vertex, and perspective-correct interpolation of what the stage returned.
// The three corners assert the fetch — each reconstructs its own vertex colour
// where the barycentric weight is 1 — and the centroid asserts the
// interpolation, because a fetch that worked and an interpolation that returned
// the first vertex's varyings agree at every corner and nowhere else.
func TestATriangleFromAVertexBuffer(t *testing.T) {
	const w, h = 16, 16
	d := openDevice(t)
	q := d.Queue()
	pipe := attributePipeline(t, d)

	// A triangle whose three vertices land exactly on viewport corners, so the
	// pixel nearest each corner is almost entirely that vertex's colour. A
	// triangle larger than the viewport would put no corner near vertices 1
	// and 2 and the fetch assertion would have nothing to stand on.
	verts := []float32{}
	verts = append(verts, interleaved(-1, -1, 0, [4]float32{1, 0, 0, 1})...)
	verts = append(verts, interleaved(1, -1, 0, [4]float32{0, 1, 0, 1})...)
	verts = append(verts, interleaved(-1, 1, 0, [4]float32{0, 0, 1, 1})...)

	vb := newBuffer(t, d, "verts", len(verts), accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(vb, 0, verts); err != nil {
		t.Fatalf("write: %v", err)
	}
	target := newBuffer(t, d, "colour", w*h*4,
		accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)

	r := d.NewRecorder()
	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{
			View: whole(t, target), Load: accel.LoadClear, Clear: [4]float32{0, 0, 0, 0},
		}},
		Width: w, Height: h, Label: "attributes",
	})
	pass.SetPipeline(pipe)
	pass.SetVertexBuffer(0, whole(t, vb))
	pass.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := readback(t, d, target)

	at := func(x, y int) [4]float32 {
		i := (y*w + x) * 4
		return [4]float32{got[i], got[i+1], got[i+2], got[i+3]}
	}

	// The clip triangle (-1,-1), (1,-1), (-1,1) maps to screen (0,h), (w,h),
	// (0,0) — the three corners of the viewport's lower-left half. The pixel
	// nearest each is almost entirely that vertex's colour.
	for _, c := range []struct {
		name    string
		x, y    int
		channel int
	}{
		{"vertex 0, red, at the bottom left", 0, h - 1, 0},
		{"vertex 1, green, at the bottom right", w - 2, h - 1, 1},
		{"vertex 2, blue, at the top left", 0, 1, 2},
	} {
		px := at(c.x, c.y)
		if px[c.channel] < 0.5 {
			t.Errorf("%s: pixel (%d,%d) is %v, and channel %d should dominate — the "+
				"attribute fetch read the wrong vertex", c.name, c.x, c.y, px, c.channel)
		}
		if px[3] != 1 {
			t.Errorf("%s: alpha is %v, want 1; every vertex supplied 1, so anything else "+
				"is a fetch at the wrong offset", c.name, px[3])
		}
	}

	// The interpolation, at a point interior to all three edges. The diagonal
	// runs corner to corner, so a point on it is a fill-rule case rather than
	// an interpolation one; this is well inside.
	mid := at(2, 12)
	var sum float32
	for i := range 3 {
		sum += mid[i]
		if mid[i] > 0.95 {
			t.Errorf("the interior point is %v: channel %d is nearly saturated, so the varyings "+
				"were taken from one vertex rather than interpolated", mid, i)
		}
	}
	if math.Abs(float64(sum-1)) > 1e-5 {
		t.Errorf("the interior point is %v and its three channels sum to %v; the barycentric "+
			"weights sum to one, so the interpolated colours must too", mid, sum)
	}
}

// Per-instance stepping: the attribute advances with the instance rather than
// with the vertex, which is how a per-object value reaches a stage with no
// uniform.
//
// Two instances of the same three vertices, each with its own colour, and both
// halves of the target hold their own instance's colour. A step mode ignored
// would paint both with instance zero's.
func TestPerInstanceAttributes(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)
	q := d.Queue()

	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.AttributeVSStage,
		Fragment: &testkernels.TintFSStage,
		VertexBuffers: []accel.VertexBufferLayout{
			{
				Stride: 12, StepMode: accel.StepVertex,
				Attributes: []accel.VertexAttribute{
					{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
				},
			},
			{
				Stride: 16, StepMode: accel.StepInstance,
				Attributes: []accel.VertexAttribute{
					{Location: 1, Format: accel.AttrFloat32x4, Offset: 0},
				},
			},
		},
		Targets: []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		Label:   "instanced",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	// Three vertices, shared by both instances: a position is indexed by the
	// vertex, so the second instance draws the same triangle. What differs
	// between them is the second buffer alone, which is the point.
	pos := []float32{-1, -1, 0, 1, -1, 0, -1, 1, 0}
	tints := []float32{1, 0, 0, 1, 0, 0, 1, 1}

	pb := newBuffer(t, d, "pos", len(pos), accel.BufferStorage|accel.BufferCopyDst)
	ib := newBuffer(t, d, "tints", len(tints), accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(pb, 0, pos); err != nil {
		t.Fatalf("write pos: %v", err)
	}
	if err := q.WriteBuffer(ib, 0, tints); err != nil {
		t.Fatalf("write tints: %v", err)
	}
	target := newBuffer(t, d, "colour", w*h*4,
		accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)

	r := d.NewRecorder()
	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole(t, target), Load: accel.LoadClear}},
		Width: w, Height: h, Label: "instanced",
	})
	pass.SetPipeline(pipe)
	pass.SetVertexBuffer(0, whole(t, pb))
	pass.SetVertexBuffer(1, whole(t, ib))
	// Three vertices, drawn twice; the second instance reads vertices 0..2
	// again and tint 1. FirstVertex does not advance with the instance.
	pass.Draw(accel.Draw{VertexCount: 3, InstanceCount: 2})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := readback(t, d, target)

	// Both instances draw the same triangle, so instance 1 overdraws instance
	// 0 everywhere and the result is instance 1's colour. That is the
	// assertion: if the step mode were ignored, both would use tint 0 and the
	// result would be red.
	i := ((h-1)*w + 0) * 4
	px := [4]float32{got[i], got[i+1], got[i+2], got[i+3]}
	if px != ([4]float32{0, 0, 1, 1}) {
		t.Errorf("pixel (0,%d) is %v, want instance 1's blue; red means the per-instance "+
			"attribute did not advance with the instance", h-1, px)
	}
}

// Every refusal the vertex layout owns, each because its absence is silent.
func TestVertexLayoutRefusals(t *testing.T) {
	base := func() accel.RenderPipelineDescriptor {
		return accel.RenderPipelineDescriptor{
			Vertex:   &testkernels.AttributeVSStage,
			Fragment: &testkernels.TintFSStage,
			VertexBuffers: []accel.VertexBufferLayout{{
				Stride: 28,
				Attributes: []accel.VertexAttribute{
					{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
					{Location: 1, Format: accel.AttrFloat32x4, Offset: 12},
				},
			}},
			Targets: []accel.ColorTargetState{{Format: accel.RGBA32Float}},
			Label:   "layout",
		}
	}
	for _, c := range []struct {
		name string
		edit func(d *accel.RenderPipelineDescriptor)
		says string
	}{{
		name: "no stride",
		edit: func(d *accel.RenderPipelineDescriptor) { d.VertexBuffers[0].Stride = 0 },
		says: "a stride of 0",
	}, {
		name: "a buffer with no attributes",
		edit: func(d *accel.RenderPipelineDescriptor) { d.VertexBuffers[0].Attributes = nil },
		says: "declares no attributes",
	}, {
		name: "an unset format",
		edit: func(d *accel.RenderPipelineDescriptor) {
			d.VertexBuffers[0].Attributes[0].Format = accel.AttrInvalid
		},
		says: "an unset attribute format",
	}, {
		name: "an attribute past the stride",
		edit: func(d *accel.RenderPipelineDescriptor) {
			d.VertexBuffers[0].Attributes[1].Offset = 20
		},
		says: "ends at 36 in a stride of 28",
	}, {
		name: "a location declared twice",
		edit: func(d *accel.RenderPipelineDescriptor) {
			d.VertexBuffers[0].Attributes[1].Location = 0
		},
		says: "location 0 is declared twice",
	}, {
		name: "a stage attribute the layout omits",
		edit: func(d *accel.RenderPipelineDescriptor) {
			d.VertexBuffers[0].Attributes = d.VertexBuffers[0].Attributes[:1]
		},
		says: "layout declares 1 attributes and AttributeVS reads 2",
	}, {
		name: "a location the stage does not read",
		edit: func(d *accel.RenderPipelineDescriptor) {
			d.VertexBuffers[0].Attributes[1].Location = 7
		},
		says: `reads attribute "tint" at location 1 and the layout declares no attribute`,
	}, {
		name: "a format of the wrong width",
		edit: func(d *accel.RenderPipelineDescriptor) {
			d.VertexBuffers[0].Attributes[1].Format = accel.AttrFloat32x2
		},
		says: "is [4]float32 in AttributeVS and float32x2 in the layout",
	}, {
		name: "a layout for a stage that reads no attribute",
		edit: func(d *accel.RenderPipelineDescriptor) {
			d.Vertex, d.Fragment = &testkernels.HalfTriangleVSStage, &testkernels.SolidFSStage
		},
		says: "layout declares 2 attributes and HalfTriangleVS reads 0",
	}} {
		t.Run(c.name, func(t *testing.T) {
			d := openDevice(t)
			desc := base()
			c.edit(&desc)
			p, err := d.NewRenderPipeline(desc)
			if err == nil {
				_ = p.Close()
				t.Fatalf("NewRenderPipeline accepted a layout it should refuse")
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Fatalf("the error should say %q, got %v", c.says, err)
			}
		})
	}
}

// A slot the layout reads and no draw bound is refused at build.
//
// Refused rather than fetched as zeros: zeroed attributes put every vertex at
// the origin, which reads as a broken transform rather than as a missing
// binding — and the caller would look at the stage.
func TestADrawWithNoVertexBufferBound(t *testing.T) {
	d := openDevice(t)
	target := newBuffer(t, d, "colour", 4*4*4,
		accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)

	r := d.NewRecorder()
	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole(t, target)}},
		Width: 4, Height: 4, Label: "unbound",
	})
	pass.SetPipeline(attributePipeline(t, d))
	pass.Draw(accel.Draw{VertexCount: 3})

	_, err := r.Build()
	if err == nil {
		t.Fatal("a draw with no vertex buffer bound was accepted")
	}
	if !strings.Contains(err.Error(), "no buffer is bound there") {
		t.Errorf("the error should name the unbound slot, got %v", err)
	}
}

// A draw that walks past the end of its vertex buffer is refused at build.
//
// Checked against the draw's counts rather than against the buffer: a buffer
// large enough for the geometry and a draw that reads more of it than exists is
// the common shape, and it reads as corruption.
func TestADrawPastTheEndOfItsVertexBuffer(t *testing.T) {
	d := openDevice(t)
	q := d.Queue()
	target := newBuffer(t, d, "colour", 4*4*4,
		accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)
	// Two vertices' worth of a 28-byte stride, for a draw of three.
	vb := newBuffer(t, d, "verts", 14, accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(vb, 0, make([]float32, 14)); err != nil {
		t.Fatalf("write: %v", err)
	}

	r := d.NewRecorder()
	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole(t, target)}},
		Width: 4, Height: 4, Label: "short",
	})
	pass.SetPipeline(attributePipeline(t, d))
	pass.SetVertexBuffer(0, whole(t, vb))
	pass.Draw(accel.Draw{VertexCount: 3})

	_, err := r.Build()
	if err == nil {
		t.Fatal("a draw past the end of its vertex buffer was accepted")
	}
	if !strings.Contains(err.Error(), "which needs 84") {
		t.Errorf("the error should say how many bytes the draw needs, got %v", err)
	}
}
