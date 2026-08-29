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

// flatImage renders IndexedVS/IndexedFS, whose integer varying is tagged flat.
//
// Red carries the id, green carries an ordinary interpolated varying, so one
// picture says both that the flat rule held and that the stage is otherwise
// running normally.
func flatImage(t *testing.T, d *accel.Device, w, h int) []float32 {
	t.Helper()
	q := d.Queue()

	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.IndexedVSStage,
		Fragment: &testkernels.IndexedFSStage,
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
		_, vary := testkernels.IndexedVS(accel.NewVertexForTest(i, 0))
		want[float32(vary.ID)] = true
	}
	for got := range ids {
		if !want[got] {
			t.Errorf("the flat id is %v, which no vertex wrote: %v", got, want)
		}
	}
}
