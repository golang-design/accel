// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
)

// drawArgs packs the four uint32 an indirect draw reads, in the layout Vulkan,
// D3D12 and Metal share.
func drawArgs(vertexCount, instanceCount, firstVertex, firstInstance uint32) []float32 {
	return []float32{
		math.Float32frombits(vertexCount),
		math.Float32frombits(instanceCount),
		math.Float32frombits(firstVertex),
		math.Float32frombits(firstInstance),
	}
}

// An indirect draw matches the equivalent direct draw.
//
// specs/033-render-api.md section 7. The comparison is the assertion: an
// argument buffer read at the wrong offset, or in the wrong order, produces a
// different picture and the direct draw says which.
func TestAnIndirectDrawMatchesTheDirectOne(t *testing.T) {
	const w, h = 16, 16
	verts := []float32{
		-1, -1, 0, 1, 0, 0, 1,
		1, -1, 0, 0, 1, 0, 1,
		-1, 1, 0, 0, 0, 1, 1,
	}

	render := func(t *testing.T, indirect bool) []float32 {
		d := openDevice(t)
		q := d.Queue()
		vb := newBuffer(t, d, "verts", len(verts), accel.BufferStorage|accel.BufferCopyDst)
		if err := q.WriteBuffer(vb, 0, verts); err != nil {
			t.Fatalf("write: %v", err)
		}
		ab := newBuffer(t, d, "args", 4, accel.BufferStorage|accel.BufferCopyDst)
		if err := q.WriteBuffer(ab, 0, drawArgs(3, 1, 0, 0)); err != nil {
			t.Fatalf("write args: %v", err)
		}
		target := newBuffer(t, d, "colour", w*h*4,
			accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)

		r := d.NewRecorder()
		p := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{View: whole(t, target), Load: accel.LoadClear}},
			Width: w, Height: h, Label: "indirect",
		})
		p.SetPipeline(attributePipeline(t, d))
		p.SetVertexBuffer(0, whole(t, vb))
		if indirect {
			p.DrawIndirect(whole(t, ab), accel.Draw{VertexCount: 3, InstanceCount: 1})
		} else {
			p.Draw(accel.Draw{VertexCount: 3})
		}
		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		if err := q.Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		return readback(t, d, target)
	}

	direct, indirect := render(t, false), render(t, true)
	var lit int
	for i := range direct {
		if direct[i] != indirect[i] {
			t.Fatalf("float %d is %v directly and %v indirectly", i, direct[i], indirect[i])
		}
		if i%4 == 0 && direct[i] != 0 {
			lit++
		}
	}
	if lit == 0 {
		t.Fatal("nothing was drawn, so matching the direct draw proves nothing")
	}
}

// A device-written count above the build-time maximum is clamped, and clamped
// whether or not anyone asked to be told.
//
// specs/033-render-api.md section 4.2. Correctness cannot depend on a debug
// flag, and exceeding a backend's limit is undefined rather than a clean error
// — so a graph that never called CollectRunStats is still protected, and what it
// gives up is knowing.
func TestAnIndirectDrawClampsToItsMaximum(t *testing.T) {
	const w, h = 16, 16
	// Three vertices in the buffer, and an argument buffer asking for thirty.
	// Unclamped, the fetch reads past the end and the pass fails; clamped, it
	// draws the triangle.
	verts := []float32{
		-1, -1, 0, 1, 0, 0, 1,
		1, -1, 0, 0, 1, 0, 1,
		-1, 1, 0, 0, 0, 1, 1,
	}
	d := openDevice(t)
	q := d.Queue()
	vb := newBuffer(t, d, "verts", len(verts), accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(vb, 0, verts); err != nil {
		t.Fatalf("write: %v", err)
	}
	ab := newBuffer(t, d, "args", 4, accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(ab, 0, drawArgs(30, 7, 0, 0)); err != nil {
		t.Fatalf("write args: %v", err)
	}
	target := newBuffer(t, d, "colour", w*h*4,
		accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole(t, target), Load: accel.LoadClear}},
		Width: w, Height: h, Label: "clamped",
	})
	p.SetPipeline(attributePipeline(t, d))
	p.SetVertexBuffer(0, whole(t, vb))
	p.DrawIndirect(whole(t, ab), accel.Draw{VertexCount: 3, InstanceCount: 1})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	// No CollectRunStats: the clamp must happen anyway.
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("a count of 30 was not clamped to the maximum of 3: %v", err)
	}
	// The bottom-left pixel, which the triangle covers. The top-left corner sits
	// exactly on the hypotenuse and the top-left fill rule excludes it, so it
	// would read as "nothing drawn" however well the clamp worked.
	got := readback(t, d, target)
	i := (h - 1) * w * 4
	if got[i] == 0 && got[i+1] == 0 && got[i+2] == 0 {
		t.Errorf("pixel (0,%d) is %v: nothing was drawn, so the clamp took the count to "+
			"zero rather than to the maximum", h-1, got[i:i+4])
	}
}

// A count below the maximum is used as given, so the clamp is a ceiling and not
// a replacement.
//
// Without this, a backend that ignored the buffer and always drew the maximum
// would pass the test above.
func TestAnIndirectCountBelowTheMaximumIsUsed(t *testing.T) {
	const w, h = 16, 16
	d := openDevice(t)
	q := d.Queue()

	// Six vertices: two triangles. The maximum is six and the device asks for
	// three, so only the first triangle is drawn.
	verts := []float32{
		-1, -1, 0, 1, 0, 0, 1,
		1, -1, 0, 1, 0, 0, 1,
		-1, 1, 0, 1, 0, 0, 1,

		1, 1, 0, 0, 1, 0, 1,
		1, -1, 0, 0, 1, 0, 1,
		-1, 1, 0, 0, 1, 0, 1,
	}
	vb := newBuffer(t, d, "verts", len(verts), accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(vb, 0, verts); err != nil {
		t.Fatalf("write: %v", err)
	}
	ab := newBuffer(t, d, "args", 4, accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(ab, 0, drawArgs(3, 1, 0, 0)); err != nil {
		t.Fatalf("write args: %v", err)
	}
	target := newBuffer(t, d, "colour", w*h*4,
		accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{
			View: whole(t, target), Load: accel.LoadClear, Clear: [4]float32{0, 0, 1, 1},
		}},
		Width: w, Height: h, Label: "under",
	})
	p.SetPipeline(attributePipeline(t, d))
	p.SetVertexBuffer(0, whole(t, vb))
	p.DrawIndirect(whole(t, ab), accel.Draw{VertexCount: 6, InstanceCount: 1})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := readback(t, d, target)

	// The first triangle is the lower-left half in red; the second would cover
	// the upper-right in green. The upper-right must still be the clear.
	i := (1*w + w - 2) * 4
	if got[i+1] != 0 {
		t.Errorf("the upper-right is %v and has green in it, so the second triangle was "+
			"drawn: the maximum replaced the device's count instead of bounding it",
			got[i:i+4])
	}
}

// An indirect draw declares its argument buffer as a read, so a kernel that
// writes the counts is ordered before the pass.
//
// This is the case indirect draws exist for: a compute pass decides how much to
// draw. Without the edge the pass reads counts the kernel has not written.
func TestAnIndirectDrawDeclaresItsArguments(t *testing.T) {
	const w, h = 4, 4
	d := openDevice(t)
	q := d.Queue()
	vb := newBuffer(t, d, "verts", 21, accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(vb, 0, make([]float32, 21)); err != nil {
		t.Fatalf("write: %v", err)
	}
	target := newBuffer(t, d, "colour", w*h*4,
		accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)

	r := d.NewRecorder()
	args := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: 4,
		Usage: accel.BufferStorage | accel.BufferCopyDst, Label: "args",
	})
	r.UploadToBuffer(args, drawArgs(3, 1, 0, 0))
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole(t, target), Load: accel.LoadClear}},
		Width: w, Height: h, Label: "ordered",
	})
	p.SetPipeline(attributePipeline(t, d))
	p.SetVertexBuffer(0, whole(t, vb))
	p.DrawIndirect(args, accel.Draw{VertexCount: 3, InstanceCount: 1})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	succ := g.Edges()[0]
	found := false
	for _, s := range succ {
		if s == p.Node() {
			found = true
		}
	}
	if !found {
		t.Errorf("the upload's successors are %v and do not include the pass; an "+
			"indirect draw reads its arguments, and the node that writes them must "+
			"come first", succ)
	}
}

// Every refusal an indirect draw owns.
func TestIndirectDrawRefusals(t *testing.T) {
	for _, c := range []struct {
		name   string
		record func(t *testing.T, p *accel.RenderPass, args accel.BufferView)
		says   string
	}{{
		name: "no argument buffer",
		record: func(t *testing.T, p *accel.RenderPass, args accel.BufferView) {
			p.DrawIndirect(accel.BufferView{}, accel.Draw{VertexCount: 3})
		},
		says: "DrawIndirect with no argument buffer",
	}, {
		name: "no maximum",
		record: func(t *testing.T, p *accel.RenderPass, args accel.BufferView) {
			p.DrawIndirect(args, accel.Draw{VertexCount: 0})
		},
		says: "a maximum of 0 vertices",
	}} {
		t.Run(c.name, func(t *testing.T) {
			d := openDevice(t)
			q := d.Queue()
			ab := newBuffer(t, d, "args", 4, accel.BufferStorage|accel.BufferCopyDst)
			if err := q.WriteBuffer(ab, 0, drawArgs(3, 1, 0, 0)); err != nil {
				t.Fatalf("write: %v", err)
			}
			vb := newBuffer(t, d, "verts", 21, accel.BufferStorage|accel.BufferCopyDst)
			if err := q.WriteBuffer(vb, 0, make([]float32, 21)); err != nil {
				t.Fatalf("write: %v", err)
			}
			target := newBuffer(t, d, "colour", 4*4*4,
				accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)

			r := d.NewRecorder()
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: whole(t, target)}},
				Width: 4, Height: 4, Label: "refused",
			})
			p.SetPipeline(attributePipeline(t, d))
			p.SetVertexBuffer(0, whole(t, vb))
			c.record(t, p, whole(t, ab))

			g, err := r.Build()
			if err == nil {
				_ = g.Close()
				t.Fatalf("Build accepted a recording it should refuse")
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Fatalf("the error should say %q, got %v", c.says, err)
			}
		})
	}

	t.Run("an argument buffer too small for four uint32", func(t *testing.T) {
		d := openDevice(t)
		q := d.Queue()
		small := newBuffer(t, d, "args", 2, accel.BufferStorage|accel.BufferCopyDst)
		if err := q.WriteBuffer(small, 0, make([]float32, 2)); err != nil {
			t.Fatalf("write: %v", err)
		}
		vb := newBuffer(t, d, "verts", 21, accel.BufferStorage|accel.BufferCopyDst)
		if err := q.WriteBuffer(vb, 0, make([]float32, 21)); err != nil {
			t.Fatalf("write: %v", err)
		}
		target := newBuffer(t, d, "colour", 4*4*4,
			accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)
		r := d.NewRecorder()
		p := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{View: whole(t, target)}},
			Width: 4, Height: 4, Label: "short args",
		})
		p.SetPipeline(attributePipeline(t, d))
		p.SetVertexBuffer(0, whole(t, vb))
		p.DrawIndirect(whole(t, small), accel.Draw{VertexCount: 3})
		g, err := r.Build()
		if err == nil {
			_ = g.Close()
			t.Fatal("a two-element argument buffer was accepted")
		}
		if !strings.Contains(err.Error(), "four uint32 arguments need 16") {
			t.Errorf("got %v", err)
		}
	})
}
