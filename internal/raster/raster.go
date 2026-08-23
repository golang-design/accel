// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package raster is the fixed-function half of the CPU reference rasterizer.
//
// It implements specs/035-cpu-rasterizer.md sections 2 to 4: the top-left fill
// rule, perspective-correct attribute interpolation, linear window-depth
// interpolation, and the viewport transform. It is a reference implementation
// and is not expected to be fast.
//
// # What it deliberately does not know
//
// This package never sees a kernel, an IR, or a Go struct. A caller hands it
// clip positions and a flat vector of per-vertex varyings, and it calls back per
// covered sample with those varyings already interpolated. Two reasons, and the
// second is the load-bearing one:
//
//   - the fill rule and the interpolation formulas are the arithmetic most
//     likely to be wrong, and they are testable here with closures rather than
//     with a compiled shader; and
//   - specs/032-stage-abi.md's mapping from a Go varyings struct to slots is the
//     caller's job. A rasterizer that knew about struct fields would need the
//     compiler to test it, and the compiler is the thing this is the oracle for.
package raster

import "math"

// Clip is a clip-space position, before the perspective divide.
//
// The convention is specs/032-stage-abi.md section 2.4: z in [-w, w], which
// becomes NDC z in [-1, 1] and window z in [0, 1]. A vertex kernel emits this
// range on every backend, and the backends whose native NDC range is [0, 1]
// fold the remap into emitted code rather than asking a caller to adjust a
// projection matrix.
type Clip struct{ X, Y, Z, W float32 }

// Vertex is one post-vertex-stage vertex: its clip position and its varyings as
// a flat float vector.
//
// Flat rather than typed, per the package doc. Every vertex of one primitive
// carries the same number of varyings, and Rasterize refuses a primitive where
// they disagree rather than interpolating whatever is shorter.
type Vertex struct {
	Pos      Clip
	Varyings []float32
}

// Fragment is one covered sample, handed to the callback.
type Fragment struct {
	// X and Y are integer pixel coordinates, top-origin: row 0 is the top row.
	// specs/005-graphics.md makes that a guarantee across the fragment stage, an
	// on-device texel fetch, and host readback, and this is the first of the
	// three.
	X, Y int

	// Depth is window depth in [0, 1], interpolated linearly -- see
	// [Rasterize]. It is what a depth compare uses and what a depth attachment
	// stores.
	Depth float32

	// InvW is 1/w at this sample, which specs/032-stage-abi.md exposes as the w
	// component of the fragment stage's window coordinate.
	InvW float32

	// Varyings are perspective-correct interpolated, in the order the vertices
	// carried them. The slice is reused between fragments: a callback that keeps
	// it must copy it.
	Varyings []float32

	// Front is whether this fragment came from a front-facing primitive, under
	// the pipeline's declared winding. specs/032-stage-abi.md exposes it as
	// Fragment.FrontFacing, and the per-face stencil state selects on it.
	Front bool

	// Bary are the screen-space barycentric weights, reported because
	// specs/035-cpu-rasterizer.md's diagnostic order asks for coverage evidence
	// before arithmetic, and because a test that checks the interpolation
	// formula needs the weights the formula used.
	Bary [3]float32
}

// Viewport is the window-space rectangle NDC maps onto.
//
// The depth range is part of it because the viewport transform is where NDC's
// [-1, 1] becomes window depth in [0, 1]; see [Rasterize].
type Viewport struct {
	X, Y, W, H         int
	MinDepth, MaxDepth float32
}

// FrontFace says which winding is front-facing.
//
// There is no zero value meaning "the backend's default". Metal's default
// disagrees with GL's, and getting it backwards keeps back faces instead of
// front faces: the silhouette stays right while every per-pixel attribute comes
// from the wrong surface, so it reads as a shading bug. A default would make
// that the easiest thing to write.
type FrontFace uint8

const (
	// CounterClockwise makes a triangle front-facing when its vertices wind
	// counter-clockwise in **clip space**, which is what a caller reasons
	// about. The viewport's y flip reverses the sign of the window-space area,
	// so the implementation's sign test is not the caller's convention and the
	// two are stated separately on purpose.
	CounterClockwise FrontFace = iota
	Clockwise
)

// Cull says which faces are discarded.
type Cull uint8

const (
	CullNone Cull = iota
	CullFront
	CullBack
)

// State is the fixed-function configuration one primitive is rasterized under.
type State struct {
	Viewport Viewport
	Front    FrontFace
	Cull     Cull

	// Scissor bounds coverage. specs/035-cpu-rasterizer.md section 4 clips only
	// the near and far planes geometrically and handles the side planes by
	// scissoring, because every clip-generated vertex needs its varyings
	// interpolated and every interpolation is a place for the oracle to
	// disagree with itself.
	Scissor Rect

	// Flat marks the varying slots that take the provoking vertex's value
	// rather than being interpolated. A nil or short slice leaves the remaining
	// slots smooth.
	//
	// specs/032-stage-abi.md section 3.1 makes integer varyings flat-*only*,
	// and that is not an accel choice: no target backend interpolates an
	// integer. The mask lives here rather than being inferred from a type
	// because this package deliberately has no types -- see the package doc.
	Flat []bool
}

// Rect is a window-space rectangle, top-origin, with X,Y its minimum corner.
type Rect struct{ X, Y, W, H int }

// window is a vertex after the perspective divide and the viewport transform.
type window struct {
	x, y, z float32 // window position, z in [minDepth, maxDepth]
	invW    float32
	vary    []float32
}

// Rasterize walks one triangle and calls emit for every covered sample.
//
// # The fill rule
//
// specs/035-cpu-rasterizer.md section 2 makes the top-left rule an accel
// guarantee rather than an implementation detail: without a stated rule, two
// triangles sharing an edge can double-shade or leave a gap, and any coverage
// comparison against the oracle is meaningless. With edge functions
//
//	E_i(x, y) = (y_{i+1} - y_i)(x - x_i) - (x_{i+1} - x_i)(y - y_i)
//
// a sample is inside when every E_i is positive, or zero on an edge the rule
// admits:
//
//	covered <=> for all i: E_i > 0 or (E_i == 0 and topLeft(i))
//
// # Interpolation
//
// Varyings are perspective-correct, per section 3:
//
//	a = sum(lambda_i * a_i / w_i) / sum(lambda_i / w_i)
//
// Depth is the exception and is interpolated linearly in window space, because
// window depth is already the post-divide value. Getting that backwards is a
// classic bug with a deceptive symptom: the image looks correct and the depth
// test picks the wrong surface only where two surfaces are close, which reads as
// z-fighting rather than as a formula error.
//
// Rasterize reports whether the triangle produced any coverage at all, which is
// what distinguishes "culled or clipped away" from "covered nothing", and it
// returns false for a degenerate triangle rather than dividing by a zero area.
func Rasterize(st State, tri [3]Vertex, emit func(Fragment)) bool {
	n := len(tri[0].Varyings)
	if len(tri[1].Varyings) != n || len(tri[2].Varyings) != n {
		// Refused rather than interpolated over the shorter one: a primitive
		// whose vertices disagree about their varying count is a caller bug, and
		// silently interpolating a prefix would produce a plausible image.
		return false
	}

	// The provoking vertex is the *original* primitive's first vertex, captured
	// before clipping. specs/035-cpu-rasterizer.md section 3 fixes it as first,
	// and capturing it here is not a micro-optimisation: clipping can remove
	// that vertex, and the fan hub is then a clip-generated vertex carrying an
	// interpolated value -- which is exactly what a flat varying must never be.
	provoking := tri[0].Varyings

	verts, ok := clipNearFar(tri)
	if !ok {
		return false
	}

	any := false
	for i := 1; i+1 < len(verts); i++ {
		if rasterOne(st, [3]Vertex{verts[0], verts[i], verts[i+1]}, n, provoking, emit) {
			any = true
		}
	}
	return any
}

// flatAt reports whether slot k takes the provoking vertex's value.
func (st State) flatAt(k int) bool { return k < len(st.Flat) && st.Flat[k] }

// rasterOne rasterizes one already-clipped triangle.
func rasterOne(st State, tri [3]Vertex, n int, provoking []float32, emit func(Fragment)) bool {
	var w [3]window
	for i, v := range tri {
		w[i] = toWindow(st.Viewport, v)
	}

	// The signed area decides both facing and the barycentric denominator, so it
	// is computed once and its sign carries all the way through.
	area := edge(w[0], w[1], float64(w[2].x), float64(w[2].y))
	if area == 0 {
		return false
	}
	front := area > 0
	if st.Front == Clockwise {
		front = !front
	}
	switch st.Cull {
	case CullFront:
		if front {
			return false
		}
	case CullBack:
		if !front {
			return false
		}
	}

	// A back-facing triangle that survives culling still has to be walked with
	// positive edge functions, so the winding is normalized by swapping two
	// vertices rather than by flipping every comparison -- which is where a
	// sign error would otherwise hide.
	if area < 0 {
		w[1], w[2] = w[2], w[1]
		area = -area
	}

	lo, hi := bounds(st, w)
	if lo.X >= hi.X || lo.Y >= hi.Y {
		return false
	}

	vary := make([]float32, n)
	any := false
	for y := lo.Y; y < hi.Y; y++ {
		for x := lo.X; x < hi.X; x++ {
			// The sample is the pixel centre, which is what makes the fill rule
			// a statement about a point rather than about an area.
			px, py := float64(x)+0.5, float64(y)+0.5

			e0 := edge(w[1], w[2], px, py)
			e1 := edge(w[2], w[0], px, py)
			e2 := edge(w[0], w[1], px, py)
			if !inside(e0, w[1], w[2]) || !inside(e1, w[2], w[0]) || !inside(e2, w[0], w[1]) {
				continue
			}

			l0 := float32(e0 / area)
			l1 := float32(e1 / area)
			l2 := float32(e2 / area)

			// Depth: linear in window space, no perspective divide.
			depth := l0*w[0].z + l1*w[1].z + l2*w[2].z

			// Varyings: perspective-correct. The denominator is computed once
			// and shared, which is also what the error bound in
			// specs/008-numerics.md section 8.1 counts.
			den := l0*w[0].invW + l1*w[1].invW + l2*w[2].invW
			for k := range vary {
				if st.flatAt(k) {
					vary[k] = provoking[k]
					continue
				}
				num := l0*w[0].vary[k]*w[0].invW +
					l1*w[1].vary[k]*w[1].invW +
					l2*w[2].vary[k]*w[2].invW
				vary[k] = num / den
			}

			any = true
			emit(Fragment{
				X: x, Y: y,
				Depth:    depth,
				InvW:     den,
				Front:    front,
				Varyings: vary,
				Bary:     [3]float32{l0, l1, l2},
			})
		}
	}
	return any
}

// edge is the edge function of the directed line a→b evaluated at (x, y).
//
// Computed in float64 so that the fill rule's "exactly zero" case is a decision
// about the geometry rather than about f32 rounding. A sample that lands on an
// edge is the case the rule exists to arbitrate, and an f32 edge function turns
// it into a coin toss between two triangles that would each round differently.
func edge(a, b window, x, y float64) float64 {
	return (float64(b.y)-float64(a.y))*(x-float64(a.x)) -
		(float64(b.x)-float64(a.x))*(y-float64(a.y))
}

// inside applies the top-left rule to one edge.
//
// An edge is a *top* edge when it is horizontal and the interior is below it,
// and a *left* edge when it descends -- under a top-origin y axis, with the
// winding already normalized so the interior is on the positive side.
func inside(e float64, a, b window) bool {
	if e > 0 {
		return true
	}
	if e < 0 {
		return false
	}
	dy := b.y - a.y
	dx := b.x - a.x
	top := dy == 0 && dx < 0
	left := dy > 0
	return top || left
}

// toWindow performs the perspective divide and the viewport transform.
//
// The y flip lives here, and it is why row 0 is the top row: NDC y increases
// upward and window y increases downward. Doing it here rather than at readback
// is what makes specs/005-graphics.md's three-way origin guarantee one
// correction instead of three.
func toWindow(vp Viewport, v Vertex) window {
	invW := 1 / v.Pos.W
	ndcX := v.Pos.X * invW
	ndcY := v.Pos.Y * invW
	ndcZ := v.Pos.Z * invW
	return window{
		x:    float32(vp.X) + (ndcX+1)*0.5*float32(vp.W),
		y:    float32(vp.Y) + (1-ndcY)*0.5*float32(vp.H),
		z:    vp.MinDepth + (ndcZ+1)*0.5*(vp.MaxDepth-vp.MinDepth),
		invW: invW,
		vary: v.Varyings,
	}
}

// bounds is the scissored integer sample range the triangle can cover.
func bounds(st State, w [3]window) (lo, hi Rect) {
	minX, minY := math.Inf(1), math.Inf(1)
	maxX, maxY := math.Inf(-1), math.Inf(-1)
	for _, v := range w {
		minX = math.Min(minX, float64(v.x))
		maxX = math.Max(maxX, float64(v.x))
		minY = math.Min(minY, float64(v.y))
		maxY = math.Max(maxY, float64(v.y))
	}
	sc := st.Scissor
	if sc.W == 0 && sc.H == 0 {
		sc = Rect{X: st.Viewport.X, Y: st.Viewport.Y, W: st.Viewport.W, H: st.Viewport.H}
	}
	x0 := max(int(math.Floor(minX)), sc.X)
	y0 := max(int(math.Floor(minY)), sc.Y)
	x1 := min(int(math.Ceil(maxX))+1, sc.X+sc.W)
	y1 := min(int(math.Ceil(maxY))+1, sc.Y+sc.H)
	return Rect{X: x0, Y: y0}, Rect{X: x1, Y: y1}
}

// clipNearFar clips a triangle against z = -w and z = +w.
//
// Only those two planes, per specs/035-cpu-rasterizer.md section 4: a vertex
// with w <= 0 has no meaningful NDC position, so near clipping is not optional,
// while the side planes are handled by the scissor in [bounds] because every
// clip-generated vertex needs its varyings interpolated and every interpolation
// is a place for the oracle to disagree with itself.
//
// Near-plane clipping is the subject of one of the sharper corpus tests: a
// [-1, 1] projection on a [0, 1] backend produces no coverage at all for near
// geometry, which reads like a broken transform rather than a convention
// mismatch.
func clipNearFar(tri [3]Vertex) ([]Vertex, bool) {
	poly := tri[:]
	poly = clipPlane(poly, func(v Vertex) float32 { return v.Pos.Z + v.Pos.W }) // z >= -w
	if len(poly) < 3 {
		return nil, false
	}
	poly = clipPlane(poly, func(v Vertex) float32 { return v.Pos.W - v.Pos.Z }) // z <= +w
	if len(poly) < 3 {
		return nil, false
	}
	return poly, true
}

// clipPlane is one Sutherland-Hodgman pass against dist >= 0.
func clipPlane(poly []Vertex, dist func(Vertex) float32) []Vertex {
	out := make([]Vertex, 0, len(poly)+1)
	for i := range poly {
		a := poly[i]
		b := poly[(i+1)%len(poly)]
		da, db := dist(a), dist(b)
		if da >= 0 {
			out = append(out, a)
		}
		if (da >= 0) != (db >= 0) {
			out = append(out, lerpVertex(a, b, da/(da-db)))
		}
	}
	return out
}

// lerpVertex interpolates a clipped vertex.
//
// Linearly in clip space, on both the position and the varyings, which is
// correct precisely because it happens *before* the perspective divide: the
// perspective correction in [rasterOne] is what recovers the right values from
// linearly interpolated clip-space quantities.
func lerpVertex(a, b Vertex, t float32) Vertex {
	v := Vertex{Pos: Clip{
		X: a.Pos.X + t*(b.Pos.X-a.Pos.X),
		Y: a.Pos.Y + t*(b.Pos.Y-a.Pos.Y),
		Z: a.Pos.Z + t*(b.Pos.Z-a.Pos.Z),
		W: a.Pos.W + t*(b.Pos.W-a.Pos.W),
	}}
	if n := len(a.Varyings); n > 0 {
		v.Varyings = make([]float32, n)
		for i := range v.Varyings {
			v.Varyings[i] = a.Varyings[i] + t*(b.Varyings[i]-a.Varyings[i])
		}
	}
	return v
}
