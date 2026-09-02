// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
)

// flatImage renders IndexedVS/IndexedFS, whose integer varying is tagged flat.
//
// Red carries the id, green carries an ordinary interpolated varying, so one
// picture says both that the flat rule held and that the stage is otherwise
// running normally.
func flatImage(t *testing.T, d *accel.Device, w, h int) []float32 {
	t.Helper()
	q := d.Queue()

	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &kernels.IndexedVSStage,
		Fragment: &kernels.IndexedFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		Label:    "flat",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	tex, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.RGBA32Float, Size: accel.Extent{Width: w, Height: h},
		Usage: accel.TextureRenderTarget | accel.TextureCopySrc | accel.TextureCopyDst,
		Kind:  accel.MemoryReadback, Label: "flat",
	})
	if err != nil {
		t.Fatalf("texture: %v", err)
	}
	defer tex.Close()
	view, err := tex.Whole()
	if err != nil {
		t.Fatalf("view: %v", err)
	}

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: view, Load: accel.LoadClear}},
		Width: w, Height: h, Label: "flat",
	})
	p.SetPipeline(pipe)
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	raw := make([]byte, w*h*tex.Format().BytesPerPixel())
	if err := q.ReadTexture(tex, raw); err != nil {
		t.Fatalf("read: %v", err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 |
			uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24)
	}
	return out
}

// flatIDs is the distinct ids a rendered image carries, over covered pixels.
//
// Covered is decided by the green channel rather than by red, because red *is*
// the value under test and an id of zero would read as uncovered.
func flatIDs(px []float32, w int) (ids map[float32]int, covered int) {
	ids = map[float32]int{}
	for i := 0; i < len(px); i += 4 {
		if px[i+1] == 0 {
			continue
		}
		covered++
		ids[px[i]]++
	}
	return ids, covered
}

// A flat varying takes one vertex's value over the whole primitive.
//
// specs/032-stage-abi.md section 3.1 and section 8's exact list. The three
// vertices carry three different ids, so an interpolated varying would produce
// a gradient of values across the triangle and a flat one produces exactly one.
// That is the assertion, and it is convention-independent: *which* of the three
// a target picks is the provoking-vertex rule, which no spec here fixes and
// which the cross-backend test below is what would catch.
func TestAFlatVaryingIsConstantAcrossThePrimitive(t *testing.T) {
	const w, h = 16, 16
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer d.Close()

	ids, covered := flatIDs(flatImage(t, d, w, h), w)
	if covered == 0 {
		t.Fatal("nothing was covered, so there is no varying to be flat")
	}
	if len(ids) != 1 {
		t.Fatalf("%d distinct ids over %d covered pixels, want 1: %v -- a flat varying "+
			"is not interpolated", len(ids), covered, ids)
	}
	// And it is one of the three the vertex stage wrote, rather than a fourth
	// value an interpolation happened to land on everywhere.
	want := map[float32]bool{}
	for i := range uint32(3) {
		_, vary := kernels.IndexedVS(accel.NewVertexForTest(i, 0))
		want[float32(vary.ID)] = true
	}
	for got := range ids {
		if !want[got] {
			t.Errorf("the flat id is %v, which no vertex wrote: %v", got, want)
		}
	}
}

// perspectiveImage renders PerspectiveVS/PerspectiveFS: red is the
// perspective-correct interpolation of a value and green is the screen-linear
// interpolation of the same value.
func perspectiveImage(t *testing.T, d *accel.Device, w, h int) []float32 {
	t.Helper()
	q := d.Queue()

	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &kernels.PerspectiveVSStage,
		Fragment: &kernels.PerspectiveFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		Label:    "perspective",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	tex, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.RGBA32Float, Size: accel.Extent{Width: w, Height: h},
		Usage: accel.TextureRenderTarget | accel.TextureCopySrc | accel.TextureCopyDst,
		Kind:  accel.MemoryReadback, Label: "perspective",
	})
	if err != nil {
		t.Fatalf("texture: %v", err)
	}
	defer tex.Close()
	view, err := tex.Whole()
	if err != nil {
		t.Fatalf("view: %v", err)
	}

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: view, Load: accel.LoadClear}},
		Width: w, Height: h, Label: "perspective",
	})
	p.SetPipeline(pipe)
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	raw := make([]byte, w*h*tex.Format().BytesPerPixel())
	if err := q.ReadTexture(tex, raw); err != nil {
		t.Fatalf("read: %v", err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 |
			uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24)
	}
	return out
}

// noperspective is a different operation from the default, and the picture
// proves it without a second implementation of either formula.
//
// specs/032-stage-abi.md section 3.1's second row. One value is carried in two
// fields, one tagged and one not, over a triangle whose vertices have different
// w. Perspective-correct interpolation divides by the interpolated 1/w and
// screen-linear does not, so the two must disagree somewhere -- and where they
// do not, nothing was applied.
func TestNoPerspectiveInterpolatesDifferently(t *testing.T) {
	const w, h = 32, 32
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer d.Close()

	px := perspectiveImage(t, d, w, h)
	var covered, differing int
	var maxGap float64
	for i := 0; i < len(px); i += 4 {
		// Blue is the second component of the same varying, which is 1 at every
		// vertex, so it is 1 wherever the triangle covered and 0 elsewhere --
		// a coverage channel that does not depend on the values under test.
		if px[i+2] == 0 {
			continue
		}
		covered++
		if gap := math.Abs(float64(px[i] - px[i+1])); gap > 0 {
			differing++
			maxGap = math.Max(maxGap, gap)
		}
	}
	if covered == 0 {
		t.Fatal("nothing was covered, so there is no interpolation to compare")
	}
	if differing == 0 {
		t.Fatalf("the two interpolations of one value agree at all %d covered pixels: "+
			"the noperspective qualifier reached nothing", covered)
	}
	t.Logf("%d of %d covered pixels differ, by at most %v", differing, covered, maxGap)
}

// renderTwiceFromOneGraph submits one built graph twice and returns both
// images.
//
// One graph rather than two identical ones: specs/003-command-graph.md's
// guarantee is that a *replay* produces the same result, and rebuilding would
// also be re-planning, which is a different claim.
func renderTwiceFromOneGraph(t *testing.T, d *accel.Device, w, h int) (first, second []float32) {
	t.Helper()
	q := d.Queue()

	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &kernels.PerspectiveVSStage,
		Fragment: &kernels.PerspectiveFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		Label:    "determinism",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	tex, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.RGBA32Float, Size: accel.Extent{Width: w, Height: h},
		Usage: accel.TextureRenderTarget | accel.TextureCopySrc | accel.TextureCopyDst,
		Kind:  accel.MemoryReadback, Label: "determinism",
	})
	if err != nil {
		t.Fatalf("texture: %v", err)
	}
	defer tex.Close()
	view, err := tex.Whole()
	if err != nil {
		t.Fatalf("view: %v", err)
	}

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: view, Load: accel.LoadClear}},
		Width: w, Height: h, Label: "determinism",
	})
	p.SetPipeline(pipe)
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	read := func() []float32 {
		if err := q.Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		raw := make([]byte, w*h*tex.Format().BytesPerPixel())
		if err := q.ReadTexture(tex, raw); err != nil {
			t.Fatalf("read: %v", err)
		}
		out := make([]float32, len(raw)/4)
		for i := range out {
			out[i] = math.Float32frombits(uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 |
				uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24)
		}
		return out
	}
	return read(), read()
}

// assertDeterministic compares two replays of one graph, bit for bit.
func assertDeterministic(t *testing.T, first, second []float32) {
	t.Helper()
	var covered int
	for i := range first {
		if first[i] != 0 {
			covered++
		}
		if math.Float32bits(first[i]) != math.Float32bits(second[i]) {
			t.Fatalf("element %d is %v on the first submission and %v on the second",
				i, first[i], second[i])
		}
	}
	// Two blank images are identical, and would say nothing.
	if covered == 0 {
		t.Fatal("the graph drew nothing, so replaying it identically proves nothing")
	}
}

// specs/035-cpu-rasterizer.md section 7's determinism corpus entry, which had
// no test: specs/003-command-graph.md guarantees a replay produces the same
// result, and TestPlansAreDeterministic is about plan-cache identity rather
// than pixels.
//
// Bit for bit rather than within a bound. Two runs of one program over one
// input have no licence to differ at all, and a bound here would hide exactly
// the non-determinism the entry exists to catch.
func TestARenderGraphReplaysIdentically(t *testing.T) {
	const w, h = 32, 32
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer d.Close()
	a, b := renderTwiceFromOneGraph(t, d, w, h)
	assertDeterministic(t, a, b)
}

// The render helpers below are portable on purpose.
//
// They lived in render_darwin_test.go and were called from this file and from
// the other portable corpus entries, which builds on a Mac and fails everywhere
// else -- twice in one day. A helper used by a portable test belongs in a
// portable file, and `GOOS=linux go vet ./...` is what says so before CI does.

// colourTexture is a render target the host can read back.
func colourTexture(t *testing.T, d *accel.Device, label string, w, h int) *accel.Texture {
	t.Helper()
	tex, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.RGBA32Float, Size: accel.Extent{Width: w, Height: h},
		Usage: accel.TextureRenderTarget | accel.TextureCopySrc | accel.TextureCopyDst,
		Kind:  accel.MemoryReadback, Label: label,
	})
	if err != nil {
		t.Fatalf("texture %s: %v", label, err)
	}
	t.Cleanup(func() { _ = tex.Close() })
	return tex
}

func wholeOf(t *testing.T, tex *accel.Texture) accel.TextureView {
	t.Helper()
	v, err := tex.Whole()
	if err != nil {
		t.Fatalf("view: %v", err)
	}
	return v
}

func readColourTexture(t *testing.T, d *accel.Device, tex *accel.Texture) []float32 {
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
