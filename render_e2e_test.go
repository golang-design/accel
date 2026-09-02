// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
)

// specs/035-cpu-rasterizer.md §8 step 1: a clip-space triangle renders to an
// offscreen target through the public API, and the interior pixels match.
//
// The triangle covers half the target rather than all of it, so the assertion
// has two halves that fail for different reasons: a covered pixel holding the
// clear colour is a rasterizer that dropped coverage, and an uncovered pixel
// holding the shaded colour is one that ignored it.
func TestATriangleRendersOffscreen(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)
	q := d.Queue()

	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &kernels.HalfTriangleVSStage,
		Fragment: &kernels.SolidFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		Label:    "solid half",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	target := colourTarget(t, d, "colour", w, h)

	r := d.NewRecorder()
	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{
			View:  view(t, target),
			Load:  accel.LoadClear,
			Clear: [4]float32{1, 0, 0, 1},
		}},
		Width: w, Height: h, Label: "offscreen",
	})
	pass.SetPipeline(pipe)
	pass.Draw(accel.Draw{VertexCount: 3, InstanceCount: 1})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	got := readTarget(t, d, target)

	// The stage places its vertices at clip (-1,-1), (1,-1) and (-1,1), which
	// after the viewport transform is the screen-space triangle (0,h), (w,h),
	// (0,0). Its hypotenuse runs from (0,0) to (w,h), so it passes exactly
	// through the centre of every pixel where x == y, and the fill rule rather
	// than the coverage arithmetic decides those. The interior lies to the
	// edge's left, which makes it a right edge, and the top-left rule excludes
	// them: strictly y > x, not y >= x.
	var covered, cleared int
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			px := got[(y*w+x)*4 : (y*w+x)*4+4]
			inside := y > x
			want := [4]float32{1, 0, 0, 1}
			if inside {
				want = [4]float32{0.25, 0.5, 0.75, 1}
				covered++
			} else {
				cleared++
			}
			for c := range want {
				if px[c] != want[c] {
					t.Fatalf("pixel (%d,%d) is %v; it is %s, so it should be %v",
						x, y, px, coverageWord(inside), want)
				}
			}
		}
	}
	// The counts are asserted too, because a triangle that covered nothing and a
	// clear that painted everything agree with each other pixel by pixel only if
	// one of the two loops above never ran.
	// A whole diagonal is at stake in that strictness, so it is worth naming
	// what the counts have to be: 28 covered of 64, not the 36 an inclusive
	// rule would give.
	if covered != 28 || cleared != 36 {
		t.Errorf("%d covered and %d cleared: an %dx%d target under the top-left rule "+
			"has 28 and 36", covered, cleared, w, h)
	}
	if covered == 0 || cleared == 0 {
		t.Fatalf("%d covered and %d cleared pixels: the assertion above checked "+
			"only one case", covered, cleared)
	}
}

func coverageWord(inside bool) string {
	if inside {
		return "inside the triangle"
	}
	return "outside the triangle"
}

// The depth attachment: cleared, tested against, and written through.
//
// Read back as the assertion rather than inferred from colour, because a depth
// buffer that is tested but never written and one that is written but never
// read give the same picture for a single draw. The values themselves separate
// them: covered pixels hold the stage's z and uncovered ones hold the clear.
func TestARenderPassClearsAndWritesDepth(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)
	q := d.Queue()

	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &kernels.HalfTriangleVSStage,
		Fragment: &kernels.SolidFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		DepthStencil: &accel.DepthStencilState{
			Format: accel.Depth32Float, Test: true, Write: true,
			Compare: accel.CompareLess,
		},
		Label: "depth",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	colour := colourTarget(t, d, "colour", w, h)
	depth := depthTarget(t, d, "depth", w, h)

	r := d.NewRecorder()
	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: view(t, colour), Load: accel.LoadClear}},
		Depth: &accel.DepthAttachment{
			View: view(t, depth), Load: accel.LoadClear, Clear: 1,
		},
		Width: w, Height: h, Label: "depth pass",
	})
	pass.SetPipeline(pipe)
	pass.Draw(accel.Draw{VertexCount: 3, InstanceCount: 1})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	// The stage puts every vertex at clip z 0.5 with w 1. specs/032-stage-abi.md
	// section 2.3 fixes clip depth as -w <= z <= w, so that is NDC 0.5, and the
	// viewport transform maps [-1, 1] onto the [0, 1] depth range: 0.75. Writing
	// 0.5 here is the [0, 1]-clip convention of Metal and Vulkan, and
	// docs/conventions.md records that mixing the two reads as a broken
	// transform rather than as a convention mismatch.
	got := readDepth(t, d, depth)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			want := float32(1)
			if y > x {
				want = 0.75
			}
			if got[y*w+x] != want {
				t.Fatalf("depth at (%d,%d) is %v, want %v", x, y, got[y*w+x], want)
			}
		}
	}
}

// LoadKeep keeps what was there, which is the whole difference between it and
// LoadClear and the reason an attachment loaded Keep is recorded as a read.
func TestARenderPassKeepsWhatWasThere(t *testing.T) {
	const w, h = 4, 4
	d := openDevice(t)
	q := d.Queue()

	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &kernels.HalfTriangleVSStage,
		Fragment: &kernels.SolidFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		Label:    "keep",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	target := colourTarget(t, d, "colour", w, h)
	prior := make([]float32, w*h*4)
	for i := range prior {
		prior[i] = 9
	}
	fillTarget(t, d, target, prior)

	r := d.NewRecorder()
	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: view(t, target), Load: accel.LoadKeep}},
		Width: w, Height: h, Label: "keep pass",
	})
	pass.SetPipeline(pipe)
	pass.Draw(accel.Draw{VertexCount: 3, InstanceCount: 1})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			got := readTarget(t, d, target)[(y*w+x)*4]
			want := float32(9)
			if y > x {
				want = 0.25
			}
			if got != want {
				t.Fatalf("red at (%d,%d) is %v, want %v", x, y, got, want)
			}
		}
	}
}

// A stage that panics becomes an error naming the pass, because on a GPU the
// same mistake is undefined rather than a crash: taking the caller's process
// down would make the oracle the least usable of the backends at the one thing
// it exists to do.
func TestAPanickingStageBecomesAnError(t *testing.T) {
	d := openDevice(t)

	boom := accel.Stage{
		Name: "Boom", Kind: accel.StageVertex, Varyings: "NoVaryings",
		RunVertex: func(accel.Vertex, []any, [][]float32, []accel.Texture2D) (accel.Clip, []float32) {
			panic("a stage read past the end of something")
		},
	}
	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &boom,
		Fragment: &kernels.SolidFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		Label:    "boom",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	target := colourTarget(t, d, "colour", 4, 4)
	r := d.NewRecorder()
	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: view(t, target), Load: accel.LoadClear}},
		Width: 4, Height: 4, Label: "exploding",
	})
	pass.SetPipeline(pipe)
	pass.Draw(accel.Draw{VertexCount: 3, InstanceCount: 1})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	err = d.Queue().Submit(g).Wait()
	if err == nil {
		t.Fatal("a panicking stage was reported as a successful submission")
	}
	for _, want := range []string{"exploding", "read past the end"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the error should name %q, got %v", want, err)
		}
	}
}
