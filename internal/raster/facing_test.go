// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package raster_test

import (
	"testing"

	"golang.design/x/accel/internal/raster"
)

// backQuadAt is quadAt with two vertices swapped, so it winds clockwise in
// clip space and is a back face under CounterClockwise.
func backQuadAt(z float32) [3]raster.Vertex {
	q := quadAt(z)
	q[1], q[2] = q[2], q[1]
	return q
}

// A back-facing fragment takes the Back stencil state, and a front-facing one
// the Front state.
//
// Every other stencil test in this package sets Back = Front, so a rasterizer
// that always selected Front would pass all of them. This one gives the two
// faces different operations and draws each winding once: the face bit is what
// chooses between 7 and 3, and a wrong choice is the other number.
func TestStencilSelectsTheFaceState(t *testing.T) {
	const n = 4
	for _, tc := range []struct {
		name string
		tri  [3]raster.Vertex
		want uint8
	}{
		{"front face replaces with the reference", quadAt(0), 7},
		{"back face increments the clear", backQuadAt(0), 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fb := &raster.Framebuffer{
				Color: []*raster.ColorTarget{raster.NewColorTarget(n, n, [4]float32{})},
				Depth: raster.NewDepthTarget(n, n, 1, 2),
			}
			ps := pass(n, n)
			ps.Stencil = raster.StencilState{
				Enabled:   true,
				Reference: 7,
				Front: raster.StencilFace{
					Compare: raster.CompareAlways, ReadMask: 0xFF, WriteMask: 0xFF,
					Fail: raster.StencilKeep, DepthFail: raster.StencilKeep,
					Pass: raster.StencilReplace,
				},
				Back: raster.StencilFace{
					Compare: raster.CompareAlways, ReadMask: 0xFF, WriteMask: 0xFF,
					Fail: raster.StencilKeep, DepthFail: raster.StencilKeep,
					Pass: raster.StencilIncrementClamp,
				},
			}
			if raster.Draw(ps, fb, tc.tri, constant([4]float32{1, 1, 1, 1})) != n*n {
				t.Fatal("the triangle did not cover the target, so no face was tested")
			}
			for i, got := range fb.Depth.Stencil {
				if got != tc.want {
					t.Fatalf("stencil[%d] is %d, want %d: the fragment took the other "+
						"face's operation", i, got, tc.want)
				}
			}
		})
	}
}

// The back face's compare is its own too: a back-facing draw can be rejected
// by Back.Compare while Front.Compare would have accepted it.
func TestStencilBackFaceCompareRejectsOnItsOwn(t *testing.T) {
	const n = 4
	fb := &raster.Framebuffer{
		Color: []*raster.ColorTarget{raster.NewColorTarget(n, n, [4]float32{})},
		Depth: raster.NewDepthTarget(n, n, 1, 0),
	}
	ps := pass(n, n)
	ps.Stencil = raster.StencilState{
		Enabled:   true,
		Reference: 0,
		Front: raster.StencilFace{
			Compare: raster.CompareAlways, ReadMask: 0xFF, WriteMask: 0xFF,
			Fail: raster.StencilKeep, DepthFail: raster.StencilKeep, Pass: raster.StencilKeep,
		},
		Back: raster.StencilFace{
			Compare: raster.CompareNever, ReadMask: 0xFF, WriteMask: 0xFF,
			Fail: raster.StencilKeep, DepthFail: raster.StencilKeep, Pass: raster.StencilKeep,
		},
	}
	if got := raster.Draw(ps, fb, backQuadAt(0), constant([4]float32{1, 1, 1, 1})); got != 0 {
		t.Fatalf("%d fragments of a back face wrote colour past a Back compare of Never", got)
	}
	if got := raster.Draw(ps, fb, quadAt(0), constant([4]float32{1, 1, 1, 1})); got != n*n {
		t.Fatalf("%d of %d fragments of a front face wrote colour under a Front compare of "+
			"Always", got, n*n)
	}
}

// A viewport with a non-zero origin places NDC inside its own rectangle.
//
// Every other test here has the viewport at (0, 0), so a transform that
// dropped X and Y would pass all of them. A full-viewport triangle through a
// viewport at (2, 3) of size 4x4 covers exactly [2, 6) x [3, 7) of an 8x8
// target, and nothing outside it.
func TestViewportOriginOffsetsCoverage(t *testing.T) {
	const n = 8
	red := [4]float32{1, 0, 0, 1}
	tgt := raster.NewColorTarget(n, n, [4]float32{})
	fb := &raster.Framebuffer{Color: []*raster.ColorTarget{tgt}}

	ps := pass(n, n)
	ps.Viewport = raster.Viewport{X: 2, Y: 3, W: 4, H: 4, MinDepth: 0, MaxDepth: 1}
	if got := raster.Draw(ps, fb, quadAt(0), constant(red)); got != 16 {
		t.Fatalf("%d fragments written through a 4x4 viewport, want 16", got)
	}
	for y := range n {
		for x := range n {
			inside := x >= 2 && x < 6 && y >= 3 && y < 7
			got := tgt.At(x, y)
			if inside && got != red {
				t.Errorf("(%d,%d) inside the viewport is %v, want %v", x, y, got, red)
			}
			if !inside && got != ([4]float32{}) {
				t.Errorf("(%d,%d) outside the viewport is %v, want the clear", x, y, got)
			}
		}
	}
}
