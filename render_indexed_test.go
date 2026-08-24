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

// An indexed draw produces what the equivalent non-indexed one does.
//
// The comparison is the assertion: an index buffer that was ignored, or read
// with the wrong width, gives a different picture, and comparing against a
// direct draw of the same triangle says which. Run for both index widths,
// because a uint16 buffer read as uint32 gives indices built from two entries
// at once and lands somewhere plausible.
func TestAnIndexedDrawMatchesTheDirectOne(t *testing.T) {
	const w, h = 16, 16

	// Four vertices of a quad, drawn as two triangles. The indices name them
	// out of order so a draw that ignored the buffer and used 0,1,2,3,4,5
	// would read past the end of a four-vertex buffer and be refused --
	// which is a different failure from a wrong picture, and a stronger one.
	verts := []float32{
		-1, -1, 0, 1, 0, 0, 1,
		1, -1, 0, 0, 1, 0, 1,
		1, 1, 0, 0, 0, 1, 1,
		-1, 1, 0, 1, 1, 0, 1,
	}
	order := []uint32{0, 1, 2, 0, 2, 3}

	// The same six vertices written out, which is what the indexed draw must
	// reproduce.
	var flat []float32
	for _, i := range order {
		flat = append(flat, verts[i*7:i*7+7]...)
	}

	direct := func(t *testing.T) []float32 {
		d := openDevice(t)
		q := d.Queue()
		vb := newBuffer(t, d, "verts", len(flat), accel.BufferStorage|accel.BufferCopyDst)
		if err := q.WriteBuffer(vb, 0, flat); err != nil {
			t.Fatalf("write: %v", err)
		}
		target := newBuffer(t, d, "colour", w*h*4,
			accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)
		r := d.NewRecorder()
		p := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{View: whole(t, target), Load: accel.LoadClear}},
			Width: w, Height: h, Label: "direct",
		})
		p.SetPipeline(attributePipeline(t, d))
		p.SetVertexBuffer(0, whole(t, vb))
		p.Draw(accel.Draw{VertexCount: 6})
		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		if err := q.Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		return readback(t, d, target)
	}(t)

	for _, c := range []struct {
		name   string
		format accel.IndexFormat
		pack   func([]uint32) ([]float32, int)
	}{{
		name: "uint32 indices", format: accel.Index32,
		pack: func(ix []uint32) ([]float32, int) {
			out := make([]float32, len(ix))
			for i, v := range ix {
				out[i] = f32bits(v)
			}
			return out, len(ix)
		},
	}, {
		name: "uint16 indices", format: accel.Index16,
		pack: func(ix []uint32) ([]float32, int) {
			// Two uint16 per float32 element.
			out := make([]float32, (len(ix)+1)/2)
			for i := 0; i < len(ix); i += 2 {
				lo := uint32(ix[i])
				var hi uint32
				if i+1 < len(ix) {
					hi = uint32(ix[i+1])
				}
				out[i/2] = f32bits(lo | hi<<16)
			}
			return out, len(out)
		},
	}} {
		t.Run(c.name, func(t *testing.T) {
			d := openDevice(t)
			q := d.Queue()

			vb := newBuffer(t, d, "verts", len(verts), accel.BufferStorage|accel.BufferCopyDst)
			if err := q.WriteBuffer(vb, 0, verts); err != nil {
				t.Fatalf("write verts: %v", err)
			}
			packed, n := c.pack(order)
			ib := newBuffer(t, d, "indices", n, accel.BufferStorage|accel.BufferCopyDst)
			if err := q.WriteBuffer(ib, 0, packed); err != nil {
				t.Fatalf("write indices: %v", err)
			}
			target := newBuffer(t, d, "colour", w*h*4,
				accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)

			r := d.NewRecorder()
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: whole(t, target), Load: accel.LoadClear}},
				Width: w, Height: h, Label: "indexed",
			})
			p.SetPipeline(attributePipeline(t, d))
			p.SetVertexBuffer(0, whole(t, vb))
			p.SetIndexBuffer(whole(t, ib), c.format)
			p.DrawIndexed(accel.DrawIndexed{IndexCount: 6})

			g, err := r.Build()
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			defer g.Close()
			if err := q.Submit(g).Wait(); err != nil {
				t.Fatalf("submit: %v", err)
			}

			got := readback(t, d, target)
			for i := range direct {
				if got[i] != direct[i] {
					t.Fatalf("float %d is %v and the direct draw gives %v; the index "+
						"buffer was not read as %v", i, got[i], direct[i], c.format)
				}
			}
			// And it drew something, so two identical blank images do not pass.
			var lit int
			for i := 0; i < len(got); i += 4 {
				if got[i] != 0 || got[i+1] != 0 || got[i+2] != 0 {
					lit++
				}
			}
			if lit == 0 {
				t.Fatal("nothing was drawn, so matching the direct draw proves nothing")
			}
		})
	}
}

// An indexed draw declares its index and vertex buffers as reads, so a node
// that writes them is ordered before the pass.
//
// The pass node exists before any draw is recorded, so these are declared when
// the draw is. A pass that did not declare them would run unordered against
// whatever uploaded the geometry — and the picture would be right most of the
// time.
func TestAnIndexedDrawDeclaresItsBuffers(t *testing.T) {
	const w, h = 4, 4
	d := openDevice(t)
	usage := accel.BufferStorage | accel.BufferCopyDst

	r := d.NewRecorder()
	vb := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: 21, Usage: usage, Label: "verts",
	})
	ib := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: 3, Usage: usage, Label: "indices",
	})
	r.UploadToBuffer(vb, make([]float32, 21))
	r.UploadToBuffer(ib, []float32{f32bits(0), f32bits(1), f32bits(2)})

	target := newBuffer(t, d, "colour", w*h*4,
		accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole(t, target), Load: accel.LoadClear}},
		Width: w, Height: h, Label: "ordered",
	})
	p.SetPipeline(attributePipeline(t, d))
	p.SetVertexBuffer(0, vb)
	p.SetIndexBuffer(ib, accel.Index32)
	p.DrawIndexed(accel.DrawIndexed{IndexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	edges := g.Edges()
	for _, upload := range []accel.NodeID{0, 1} {
		found := false
		for _, s := range edges[upload] {
			if s == p.Node() {
				found = true
			}
		}
		if !found {
			t.Errorf("upload %d's successors are %v and do not include the pass; a draw's "+
				"vertex and index buffers are reads the pass declares", upload, edges[upload])
		}
	}
}

// Every refusal an indexed draw owns.
func TestIndexedDrawRefusals(t *testing.T) {
	for _, c := range []struct {
		name   string
		record func(t *testing.T, d *accel.Device, p *accel.RenderPass, ib accel.BufferView)
		says   string
	}{{
		name: "no index buffer",
		record: func(t *testing.T, d *accel.Device, p *accel.RenderPass, ib accel.BufferView) {
			p.DrawIndexed(accel.DrawIndexed{IndexCount: 3})
		},
		says: "an indexed draw with no index buffer",
	}, {
		name: "no indices",
		record: func(t *testing.T, d *accel.Device, p *accel.RenderPass, ib accel.BufferView) {
			p.SetIndexBuffer(ib, accel.Index32)
			p.DrawIndexed(accel.DrawIndexed{IndexCount: 0})
		},
		says: "an indexed draw of 0 indices",
	}, {
		name: "a negative base vertex",
		record: func(t *testing.T, d *accel.Device, p *accel.RenderPass, ib accel.BufferView) {
			p.SetIndexBuffer(ib, accel.Index32)
			p.DrawIndexed(accel.DrawIndexed{IndexCount: 3, BaseVertex: -1})
		},
		says: "base vertex (-1)",
	}, {
		name: "an index buffer with no buffer",
		record: func(t *testing.T, d *accel.Device, p *accel.RenderPass, ib accel.BufferView) {
			p.SetIndexBuffer(accel.BufferView{}, accel.Index32)
		},
		says: "SetIndexBuffer with no buffer",
	}, {
		name: "a draw past the end of the index buffer",
		record: func(t *testing.T, d *accel.Device, p *accel.RenderPass, ib accel.BufferView) {
			p.SetIndexBuffer(ib, accel.Index32)
			p.DrawIndexed(accel.DrawIndexed{IndexCount: 99})
		},
		says: "the draw reads 99 uint32 indices",
	}} {
		t.Run(c.name, func(t *testing.T) {
			d := openDevice(t)
			q := d.Queue()
			ib := newBuffer(t, d, "indices", 6, accel.BufferStorage|accel.BufferCopyDst)
			if err := q.WriteBuffer(ib, 0, make([]float32, 6)); err != nil {
				t.Fatalf("write: %v", err)
			}
			vb := newBuffer(t, d, "verts", 28, accel.BufferStorage|accel.BufferCopyDst)
			if err := q.WriteBuffer(vb, 0, make([]float32, 28)); err != nil {
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
			c.record(t, d, p, whole(t, ib))

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
}

// f32bits reinterprets an index as the float32 element a buffer of F32 holds.
//
// Index buffers are bytes, and the buffer type here is F32 because that is what
// the test helpers make; the bits are what matter, and reinterpreting is how a
// caller packs indices into whatever buffer they have.
func f32bits(v uint32) float32 { return math.Float32frombits(v) }

// An index reaching past the vertex buffer is an error naming the index, not a
// crash and not a wrong picture.
//
// Build cannot catch it: an indexed draw's vertex range is decided by the index
// values, which are data rather than structure. The backend checks it once per
// draw against the indices it has already decoded.
func TestAnIndexPastTheVertexBuffer(t *testing.T) {
	d := openDevice(t)
	q := d.Queue()

	// Three vertices in the buffer, and an index naming a fourth.
	verts := make([]float32, 21)
	vb := newBuffer(t, d, "verts", len(verts), accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(vb, 0, verts); err != nil {
		t.Fatalf("write: %v", err)
	}
	ib := newBuffer(t, d, "indices", 3, accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(ib, 0, []float32{f32bits(0), f32bits(1), f32bits(9)}); err != nil {
		t.Fatalf("write: %v", err)
	}
	target := newBuffer(t, d, "colour", 4*4*4,
		accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole(t, target), Load: accel.LoadClear}},
		Width: 4, Height: 4, Label: "overrun",
	})
	p.SetPipeline(attributePipeline(t, d))
	p.SetVertexBuffer(0, whole(t, vb))
	p.SetIndexBuffer(whole(t, ib), accel.Index32)
	p.DrawIndexed(accel.DrawIndexed{IndexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	err = q.Submit(g).Wait()
	if err == nil {
		t.Fatal("an index past the end of the vertex buffer was accepted")
	}
	if !strings.Contains(err.Error(), "index 9 with base vertex 0") {
		t.Errorf("the error should name the index, got %v", err)
	}
}

// BaseVertex offsets the attribute fetch and not the index the stage sees.
//
// Two draws of the same indices over a buffer holding two copies of the
// geometry, the second offset by BaseVertex, produce the two copies. If
// BaseVertex reached the stage's VertexIndex instead, a stage computing from
// the index would move and a stage reading attributes would not -- and
// specs/032-stage-abi.md section 2.1 declines to expose a base-vertex built-in
// precisely because backends disagree about which one theirs reports.
func TestBaseVertexOffsetsTheFetchOnly(t *testing.T) {
	const w, h = 16, 16
	d := openDevice(t)
	q := d.Queue()

	// Two triangles at different places, back to back in one buffer.
	first := []float32{
		-1, -1, 0, 1, 0, 0, 1,
		0, -1, 0, 1, 0, 0, 1,
		-1, 0, 0, 1, 0, 0, 1,
	}
	second := []float32{
		0, 0, 0, 0, 0, 1, 1,
		1, 0, 0, 0, 0, 1, 1,
		0, 1, 0, 0, 0, 1, 1,
	}
	verts := append(append([]float32{}, first...), second...)
	vb := newBuffer(t, d, "verts", len(verts), accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(vb, 0, verts); err != nil {
		t.Fatalf("write: %v", err)
	}
	ib := newBuffer(t, d, "indices", 3, accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(ib, 0, []float32{f32bits(0), f32bits(1), f32bits(2)}); err != nil {
		t.Fatalf("write: %v", err)
	}
	target := newBuffer(t, d, "colour", w*h*4,
		accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: whole(t, target), Load: accel.LoadClear}},
		Width: w, Height: h, Label: "base vertex",
	})
	p.SetPipeline(attributePipeline(t, d))
	p.SetVertexBuffer(0, whole(t, vb))
	p.SetIndexBuffer(whole(t, ib), accel.Index32)
	p.DrawIndexed(accel.DrawIndexed{IndexCount: 3})
	p.DrawIndexed(accel.DrawIndexed{IndexCount: 3, BaseVertex: 3})

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

	// The first triangle is red in the lower-left quadrant; the second is blue
	// in the upper-right. The same three indices produced both, which only
	// happens if BaseVertex moved the fetch.
	if px := at(2, 13); px[0] == 0 {
		t.Errorf("the lower-left is %v and should be the first triangle's red", px)
	}
	if px := at(10, 6); px[2] == 0 {
		t.Errorf("the upper-right is %v and should be the second triangle's blue; "+
			"BaseVertex did not offset the attribute fetch", px)
	}
}
