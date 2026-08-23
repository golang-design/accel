// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package raster_test

import (
	"math"
	"testing"

	"golang.design/x/accel/internal/conformance/numeq"
	"golang.design/x/accel/internal/raster"
)

func vp(w, h int) raster.Viewport {
	return raster.Viewport{W: w, H: h, MinDepth: 0, MaxDepth: 1}
}

func state(w, h int) raster.State {
	return raster.State{Viewport: vp(w, h), Front: raster.CounterClockwise, Cull: raster.CullNone}
}

// at builds a clip-space vertex from an NDC position with w = 1, which is the
// orthographic case: no perspective, so a test that wants to isolate coverage
// from interpolation uses it.
func at(x, y, z float32, vary ...float32) raster.Vertex {
	return raster.Vertex{Pos: raster.Clip{X: x, Y: y, Z: z, W: 1}, Varyings: vary}
}

type pixel struct{ X, Y int }

func cover(t *testing.T, st raster.State, tri [3]raster.Vertex) map[pixel]int {
	t.Helper()
	out := map[pixel]int{}
	raster.Rasterize(st, tri, func(f raster.Fragment) { out[pixel{f.X, f.Y}]++ })
	return out
}

// The fill rule, checked against specs/035-cpu-rasterizer.md section 2's
// formulation directly rather than by rendering and counting.
//
// Two triangles sharing an edge must cover each shared pixel **exactly once**.
// A pixel count would pass when both triangles claim the edge and one simply
// overwrites the other, which is the double-shading the rule exists to prevent
// and is invisible in an image whose two halves are the same colour.
func TestTopLeftRuleCoversSharedEdgesExactlyOnce(t *testing.T) {
	st := state(16, 16)

	// A unit square split along its diagonal. The diagonal is the shared edge,
	// and both triangles are counter-clockwise in NDC.
	lower := [3]raster.Vertex{at(-1, -1, 0), at(1, -1, 0), at(1, 1, 0)}
	upper := [3]raster.Vertex{at(-1, -1, 0), at(1, 1, 0), at(-1, 1, 0)}

	total := map[pixel]int{}
	for _, tri := range [][3]raster.Vertex{lower, upper} {
		for p, n := range cover(t, st, tri) {
			total[p] += n
		}
	}

	var twice, none []pixel
	for y := range 16 {
		for x := range 16 {
			switch total[pixel{x, y}] {
			case 1:
			case 0:
				none = append(none, pixel{x, y})
			default:
				twice = append(twice, pixel{x, y})
			}
		}
	}
	if len(twice) > 0 {
		t.Errorf("%d pixels covered more than once along the shared edge, first %v; the "+
			"top-left rule exists to make this impossible", len(twice), twice[0])
	}
	if len(none) > 0 {
		t.Errorf("%d pixels covered by neither triangle, first %v; a gap along a shared "+
			"edge is the other half of the same failure", len(none), none[0])
	}
}

// The rule is a statement about samples on an edge, so it is also checked where
// the edge passes exactly through pixel centres rather than only on a diagonal.
//
// With a 4x4 viewport, NDC x = 0 is window x = 2.0, which is a pixel boundary
// and not a centre; NDC x = -0.5 is window x = 1.0. A vertical edge at a pixel
// boundary is the case where "left edge" decides, and the two halves must still
// partition the square.
func TestTopLeftRuleOnAxisAlignedSharedEdges(t *testing.T) {
	st := state(4, 4)
	left := [3]raster.Vertex{at(-1, -1, 0), at(0, -1, 0), at(0, 1, 0)}
	leftUp := [3]raster.Vertex{at(-1, -1, 0), at(0, 1, 0), at(-1, 1, 0)}
	right := [3]raster.Vertex{at(0, -1, 0), at(1, -1, 0), at(1, 1, 0)}
	rightUp := [3]raster.Vertex{at(0, -1, 0), at(1, 1, 0), at(0, 1, 0)}

	total := map[pixel]int{}
	for _, tri := range [][3]raster.Vertex{left, leftUp, right, rightUp} {
		for p, n := range cover(t, st, tri) {
			total[p] += n
		}
	}
	for y := range 4 {
		for x := range 4 {
			if got := total[pixel{x, y}]; got != 1 {
				t.Fatalf("pixel (%d,%d) covered %d times across a square split by a "+
					"vertical shared edge, want 1", x, y, got)
			}
		}
	}
}

// Perspective-correct interpolation, against the closed form, under the derived
// budget of specs/008-numerics.md section 8.1.
//
// The triangle is deliberately in perspective -- the three w values differ --
// because a screen-linear interpolation agrees with the correct one exactly when
// they do not. A test with w = 1 everywhere passes with the formula wrong.
func TestPerspectiveCorrectInterpolation(t *testing.T) {
	st := state(32, 32)
	vertex := []float32{0, 10, 20}
	tri := [3]raster.Vertex{
		{Pos: raster.Clip{X: -1, Y: -1, Z: 0, W: 1}, Varyings: []float32{vertex[0]}},
		{Pos: raster.Clip{X: 3, Y: -3, Z: 0, W: 3}, Varyings: []float32{vertex[1]}},
		{Pos: raster.Clip{X: -2, Y: 2, Z: 0, W: 2}, Varyings: []float32{vertex[2]}},
	}

	var got, want []float32
	raster.Rasterize(st, tri, func(f raster.Fragment) {
		// The closed form, in float64, from the same barycentrics the
		// rasterizer used: this compares the arithmetic, not the coverage.
		var num, den float64
		for i, l := range f.Bary {
			iw := 1 / float64(tri[i].Pos.W)
			num += float64(l) * float64(vertex[i]) * iw
			den += float64(l) * iw
		}
		got = append(got, f.Varyings[0])
		want = append(want, float32(num/den))
	})
	if len(got) == 0 {
		t.Fatal("the triangle covered nothing")
	}

	if r := numeq.WithinInterpolation(got, want, vertex); !r.OK() {
		t.Errorf("interpolated varying: %v", r)
	}

	// And the screen-linear answer must be *outside* the budget somewhere, or
	// this test would pass with the perspective divide removed.
	var linear []float32
	raster.Rasterize(st, tri, func(f raster.Fragment) {
		var v float32
		for i, l := range f.Bary {
			v += l * vertex[i]
		}
		linear = append(linear, v)
	})
	if r := numeq.WithinInterpolation(linear, want, vertex); r.OK() {
		t.Error("a screen-linear interpolation is within the perspective budget, so this " +
			"triangle does not discriminate between the two formulas")
	}
}

// Depth is interpolated linearly in window space, not perspective-correctly.
//
// Checked against the closed form for a known plane rather than only against
// which surface a depth test picked: getting this backwards leaves the image
// correct and moves the winner only where two surfaces are close, which reads as
// z-fighting rather than as a formula error.
func TestDepthIsLinearInWindowSpace(t *testing.T) {
	st := state(32, 32)
	tri := [3]raster.Vertex{
		{Pos: raster.Clip{X: -1, Y: -1, Z: -1, W: 1}},
		{Pos: raster.Clip{X: 3, Y: -3, Z: 1.5, W: 3}},
		{Pos: raster.Clip{X: -2, Y: 2, Z: 0, W: 2}},
	}
	// Window depth per vertex: (z/w + 1)/2.
	var zw [3]float64
	for i, v := range tri {
		zw[i] = (float64(v.Pos.Z)/float64(v.Pos.W) + 1) * 0.5
	}

	var maxErr, maxPerspErr float64
	n := 0
	raster.Rasterize(st, tri, func(f raster.Fragment) {
		n++
		var linear, num, den float64
		for i, l := range f.Bary {
			linear += float64(l) * zw[i]
			iw := 1 / float64(tri[i].Pos.W)
			num += float64(l) * zw[i] * iw
			den += float64(l) * iw
		}
		maxErr = math.Max(maxErr, math.Abs(float64(f.Depth)-linear))
		maxPerspErr = math.Max(maxPerspErr, math.Abs(float64(f.Depth)-num/den))
	})
	if n == 0 {
		t.Fatal("the triangle covered nothing")
	}

	// The linear sum is three products and two additions, so section 7's
	// sequential bound over Σ|λᵢ zᵢ| applies; the magnitudes are at most 1 here.
	budget, ok := numeq.SumBudget(4, 1)
	if !ok {
		t.Fatal("no sum budget")
	}
	if maxErr > budget {
		t.Errorf("depth is %g from the linear closed form, past the %g budget", maxErr, budget)
	}
	if maxPerspErr <= budget {
		t.Error("depth also matches the perspective-correct form, so this triangle does " +
			"not discriminate between them")
	}
}

// Winding and culling. Reversing the front face with back-face culling on
// empties coverage entirely, which is the exact symptom conventions.md records
// for the Metal divergence: the silhouette survives while every attribute comes
// from the wrong surface, so it reads as a shading bug.
func TestWindingFlipEmptiesCoverage(t *testing.T) {
	tri := [3]raster.Vertex{at(-1, -1, 0), at(1, -1, 0), at(0, 1, 0)}

	st := state(16, 16)
	st.Cull = raster.CullBack
	if len(cover(t, st, tri)) == 0 {
		t.Fatal("a front-facing triangle was culled")
	}

	st.Front = raster.Clockwise
	if got := cover(t, st, tri); len(got) != 0 {
		t.Errorf("reversing the front face left %d covered pixels under back-face culling",
			len(got))
	}

	// Culling the other way is the mirror of it, so a rasterizer that ignored
	// the cull mode entirely would still fail one of these.
	st = state(16, 16)
	st.Cull = raster.CullFront
	if got := cover(t, st, tri); len(got) != 0 {
		t.Errorf("CullFront left %d pixels of a front-facing triangle", len(got))
	}
}

// A back-facing triangle that survives culling covers exactly what its
// front-facing counterpart covers.
//
// This is the assertion that makes the winding normalisation in rasterOne
// load-bearing. Its edge functions are all negative inside the triangle, so a
// rasterizer that only negated the area and left the vertices alone walks a
// triangle nothing is ever inside -- and every culling test still passes,
// because culling is decided before the walk.
func TestBackFacingSurvivesCullNone(t *testing.T) {
	st := state(16, 16)
	front := [3]raster.Vertex{at(-1, -1, 0), at(1, -1, 0), at(0, 1, 0)}
	back := [3]raster.Vertex{at(-1, -1, 0), at(0, 1, 0), at(1, -1, 0)}

	wantCover := cover(t, st, front)
	if len(wantCover) == 0 {
		t.Fatal("the front-facing triangle covered nothing")
	}
	gotCover := cover(t, st, back)
	if len(gotCover) == 0 {
		t.Fatal("a back-facing triangle covered nothing under CullNone; its edge " +
			"functions are negative inside, so the winding was not normalised")
	}
	for p := range wantCover {
		if gotCover[p] == 0 {
			t.Fatalf("pixel %v is covered front-facing and not back-facing", p)
		}
	}
	if len(gotCover) != len(wantCover) {
		t.Errorf("back-facing covered %d pixels and front-facing %d",
			len(gotCover), len(wantCover))
	}
}

// Row 0 is the top row. NDC y increases upward and window y increases downward,
// so a triangle in the upper half of NDC must land in the low-numbered rows.
//
// This is the first of specs/005-graphics.md's three places that must agree, and
// the one the other two are corrected against.
func TestRowZeroIsTheTopRow(t *testing.T) {
	st := state(16, 16)
	upper := [3]raster.Vertex{at(-1, 0.1, 0), at(1, 0.1, 0), at(0, 1, 0)}
	got := cover(t, st, upper)
	if len(got) == 0 {
		t.Fatal("the triangle covered nothing")
	}
	for p := range got {
		if p.Y >= 8 {
			t.Fatalf("a triangle in the upper half of NDC covered row %d of 16; NDC y "+
				"increases upward and window y downward, so it belongs above row 8", p.Y)
		}
	}
}

// Near-plane clipping. Geometry straddling the near plane keeps its near half
// and loses the rest, rather than covering nothing at all -- which is the
// symptom conventions.md describes for a [-1,1] projection on a [0,1] backend
// and which reads like a broken transform.
func TestNearPlaneStraddleKeepsTheNearHalf(t *testing.T) {
	st := state(32, 32)
	// One vertex behind the near plane (z < -w), two in front.
	tri := [3]raster.Vertex{
		{Pos: raster.Clip{X: -1, Y: -1, Z: -2, W: 1}},
		{Pos: raster.Clip{X: 1, Y: -1, Z: 0, W: 1}},
		{Pos: raster.Clip{X: 0, Y: 1, Z: 0, W: 1}},
	}
	got := cover(t, st, tri)
	if len(got) == 0 {
		t.Fatal("straddling geometry covered nothing, which is the exact failure this " +
			"test exists to catch")
	}

	// Every surviving fragment's depth is inside the stored range. A clip that
	// merely dropped the test rather than cutting the triangle would produce
	// depths outside it.
	raster.Rasterize(st, tri, func(f raster.Fragment) {
		if f.Depth < 0 || f.Depth > 1 {
			t.Fatalf("a fragment at (%d,%d) has window depth %g, outside [0,1]",
				f.X, f.Y, f.Depth)
		}
	})

	// Wholly behind the near plane is no coverage, which is the other side of
	// the same clip.
	behind := [3]raster.Vertex{
		{Pos: raster.Clip{X: -1, Y: -1, Z: -2, W: 1}},
		{Pos: raster.Clip{X: 1, Y: -1, Z: -3, W: 1}},
		{Pos: raster.Clip{X: 0, Y: 1, Z: -2, W: 1}},
	}
	if got := cover(t, st, behind); len(got) != 0 {
		t.Errorf("geometry entirely behind the near plane covered %d pixels", len(got))
	}
}

// Reverse-Z, which specs/032-stage-abi.md section 2.4 claims needs no API
// change: near maps to window 1.0 and far to 0.0 under the existing convention,
// so a reversed projection is a clear value and a compare function.
func TestReverseZMapsNearToOne(t *testing.T) {
	st := state(8, 8)
	near := [3]raster.Vertex{at(-1, -1, 1), at(1, -1, 1), at(0, 1, 1)}
	far := [3]raster.Vertex{at(-1, -1, -1), at(1, -1, -1), at(0, 1, -1)}

	depthOf := func(tri [3]raster.Vertex) float32 {
		var d float32
		var n int
		raster.Rasterize(st, tri, func(f raster.Fragment) { d += f.Depth; n++ })
		if n == 0 {
			t.Fatal("covered nothing")
		}
		return d / float32(n)
	}
	if got := depthOf(near); got != 1 {
		t.Errorf("clip z = +w stores window depth %g, want 1.0", got)
	}
	if got := depthOf(far); got != 0 {
		t.Errorf("clip z = -w stores window depth %g, want 0.0", got)
	}
}

// The scissor bounds coverage, which is how the side planes are handled instead
// of geometric clipping.
func TestScissorBoundsCoverage(t *testing.T) {
	st := state(16, 16)
	st.Scissor = raster.Rect{X: 4, Y: 4, W: 4, H: 4}
	full := [3]raster.Vertex{at(-1, -1, 0), at(1, -1, 0), at(1, 1, 0)}
	for p := range cover(t, st, full) {
		if p.X < 4 || p.X >= 8 || p.Y < 4 || p.Y >= 8 {
			t.Fatalf("a fragment at (%d,%d) escaped the scissor", p.X, p.Y)
		}
	}
}

// Degenerate and malformed input is refused rather than dividing by a zero area
// or interpolating over a prefix.
func TestRasterizeRefusesWhatItCannotDraw(t *testing.T) {
	st := state(8, 8)

	zeroArea := [3]raster.Vertex{at(-1, -1, 0), at(1, 1, 0), at(0, 0, 0)}
	if raster.Rasterize(st, zeroArea, func(raster.Fragment) {
		t.Error("a zero-area triangle emitted a fragment")
	}) {
		t.Error("a zero-area triangle reported coverage")
	}

	// Vertices disagreeing about their varying count: silently interpolating a
	// prefix would produce a plausible image from a caller bug.
	ragged := [3]raster.Vertex{at(-1, -1, 0, 1, 2), at(1, -1, 0, 1), at(0, 1, 0, 1, 2)}
	if raster.Rasterize(st, ragged, func(raster.Fragment) {
		t.Error("a ragged primitive emitted a fragment")
	}) {
		t.Error("a ragged primitive reported coverage")
	}
}

// A triangle wholly outside the viewport covers nothing and does not panic on
// its own bounding box.
func TestOffscreenTriangleCoversNothing(t *testing.T) {
	st := state(8, 8)
	off := [3]raster.Vertex{at(-9, -9, 0), at(-7, -9, 0), at(-8, -7, 0)}
	if got := cover(t, st, off); len(got) != 0 {
		t.Errorf("an offscreen triangle covered %d pixels", len(got))
	}
}

// A clipped vertex carries interpolated varyings, and the interpolation is
// linear in **clip** space -- correct precisely because it happens before the
// perspective divide, which the rasterizer's own correction then undoes.
//
// Without this, a straddling triangle's new vertices would carry whatever the
// first vertex had, and the visible symptom is a gradient that flattens against
// the near plane rather than an error.
func TestClippedVerticesCarryInterpolatedVaryings(t *testing.T) {
	st := state(32, 32)
	// The behind-the-near-plane vertex carries 0 and the two in front carry 100,
	// so any fragment reading 0 came from an uninterpolated clip vertex.
	tri := [3]raster.Vertex{
		{Pos: raster.Clip{X: -1, Y: -1, Z: -2, W: 1}, Varyings: []float32{0}},
		{Pos: raster.Clip{X: 1, Y: -1, Z: 0, W: 1}, Varyings: []float32{100}},
		{Pos: raster.Clip{X: 0, Y: 1, Z: 0, W: 1}, Varyings: []float32{100}},
	}

	var lo, hi float32 = 1e9, -1e9
	n := 0
	raster.Rasterize(st, tri, func(f raster.Fragment) {
		n++
		lo = min(lo, f.Varyings[0])
		hi = max(hi, f.Varyings[0])
	})
	if n == 0 {
		t.Fatal("the straddling triangle covered nothing")
	}
	// The clip parameter is computable: the behind vertex has z + w = -1 and
	// each front vertex has z + w = +1, so both new vertices land at t = 0.5 and
	// carry 0 + 0.5·(100 − 0) = 50. The clipped quadrilateral's four vertices
	// therefore hold 50, 100, 100, 50, and every interior sample is at least 50.
	//
	// That lower bound is the assertion, and it is what a weaker one misses: a
	// clip that copied one endpoint's varyings into the new vertex leaves 0 at
	// one corner, and every "is it above zero" check still passes because the
	// samples are pixel centres strictly inside the triangle.
	const want = 50
	if lo < want-1 {
		t.Errorf("the lowest surviving fragment reads %g, and the clip's own parameter "+
			"puts the minimum at %g; a clipped vertex kept an endpoint's value instead "+
			"of an interpolated one", lo, float32(want))
	}
	if hi > 100 {
		t.Errorf("a surviving fragment reads %g, above every vertex value", hi)
	}
	// And the gradient survives: a clip that collapsed both new vertices onto
	// one value would make every fragment read the same thing.
	if hi-lo < 1 {
		t.Errorf("every surviving fragment reads about %g, so the clipped vertices "+
			"carry no gradient", lo)
	}
}
