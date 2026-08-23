// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package raster

import "fmt"

// Primitive assembly: the walk from a draw's counts to the triangles [Draw]
// rasterizes.
//
// It lives here rather than in the backend so the oracle owns the geometry. A
// backend that assembled its own primitives would be a second implementation of
// strip winding and index fetch, and specs/000-decisions.md decision 3 makes
// this package the thing a backend is checked against, not a peer of it.

// Topology is how vertices group into primitives.
type Topology uint8

const (
	TriangleList Topology = iota
	TriangleStrip

	// LineList, LineStrip and PointList are named so a caller's value maps onto
	// specs/033-render-api.md's enumeration, and are refused by
	// [DrawPrimitives]. specs/035-cpu-rasterizer.md section 10 leaves their
	// rules open: lines have a diamond-exit rule on some backends and a
	// Bresenham-ish rule on others, and points have a size and a centre
	// convention. Guessing one would put an unstated rule in the oracle, which
	// is worse than refusing.
	LineList
	LineStrip
	PointList
)

func (t Topology) String() string {
	switch t {
	case TriangleList:
		return "triangle list"
	case TriangleStrip:
		return "triangle strip"
	case LineList:
		return "line list"
	case LineStrip:
		return "line strip"
	case PointList:
		return "point list"
	}
	return fmt.Sprintf("topology(%d)", uint8(t))
}

// DrawCall is one recorded draw's counts.
//
// Instancing is the instance count and not a separate call, which is why the
// non-instanced case is this one with a count of one: a caller never picks
// between two entry points for the same drawing.
type DrawCall struct {
	Topology Topology

	// Count is the vertex count, or the index count when Index is non-nil.
	Count int

	// Instances is how many times the whole primitive set is drawn.
	Instances int

	// First is the first vertex, or the first index when Index is non-nil.
	First int

	// FirstInstance offsets the instance index the stage sees.
	FirstInstance int

	// Index is the index buffer, or nil for a non-indexed draw.
	Index []uint32

	// BaseVertex is added to each fetched index before the attribute fetch, and
	// **not** to the index the stage sees. specs/032-stage-abi.md section 2.1
	// declines to expose a base-vertex built-in for exactly this reason:
	// backends disagree about whether their built-in reports the pre-offset or
	// post-offset value, so the ABI exposes only the one a caller can act on.
	BaseVertex int
}

// VertexFn produces one post-vertex-stage vertex.
//
// The index it receives is what specs/032-stage-abi.md section 2.1 calls
// VertexIndex: for a non-indexed draw it is First + i, and for an indexed draw
// it is the value read from the index buffer -- before BaseVertex, which applies
// to the attribute fetch the callee performs.
type VertexFn func(vertexIndex, instanceIndex, baseVertex uint32) Vertex

// DrawPrimitives assembles a draw's primitives and rasterizes each.
//
// It returns the number of fragments that reached an attachment write, summed
// over every primitive, and an error for a topology whose rules this rasterizer
// does not state.
//
// Primitives are emitted in order and never reordered, which
// specs/033-render-api.md makes caller-visible: blending is order dependent, and
// so is any reasoning about overdraw.
func DrawPrimitives(ps PassState, fb *Framebuffer, dc DrawCall,
	vs VertexFn, shade func(Fragment) Shaded) (int, error) {

	switch dc.Topology {
	case TriangleList, TriangleStrip:
	default:
		return 0, fmt.Errorf("raster: %v is not rasterizable: specs/035-cpu-rasterizer.md "+
			"section 10 leaves its fill rule unstated, and guessing one would put an "+
			"unstated rule in the oracle every backend is checked against", dc.Topology)
	}
	if dc.Count < 0 || dc.Instances < 0 || dc.First < 0 {
		return 0, fmt.Errorf("raster: a draw with a negative count, instance count or "+
			"first element: %+v", dc)
	}

	written := 0
	for inst := range dc.Instances {
		instance := uint32(dc.FirstInstance + inst)
		for _, idx := range triangles(dc) {
			var tri [3]Vertex
			for k, at := range idx {
				tri[k] = vs(at, instance, uint32(dc.BaseVertex))
			}
			written += Draw(ps, fb, tri, shade)
		}
	}
	return written, nil
}

// triangles enumerates the vertex-index triples one instance draws.
//
// Strip winding alternates: the second triangle of a strip has its first two
// vertices swapped, so that every triangle in the strip presents the same facing
// as the first. Emitting a strip without the swap makes every other triangle
// back-facing, and with culling on that produces a striped mesh -- a failure
// that looks like a geometry bug rather than a winding one.
func triangles(dc DrawCall) [][3]uint32 {
	fetch := func(i int) uint32 {
		if dc.Index == nil {
			return uint32(dc.First + i)
		}
		at := dc.First + i
		if at >= len(dc.Index) {
			// Past the end of the index buffer is a caller error the layer
			// above validates. Zero here keeps this function total rather than
			// panicking inside a rasterizer.
			return 0
		}
		return dc.Index[at]
	}

	var out [][3]uint32
	switch dc.Topology {
	case TriangleList:
		for i := 0; i+2 < dc.Count; i += 3 {
			out = append(out, [3]uint32{fetch(i), fetch(i + 1), fetch(i + 2)})
		}
	case TriangleStrip:
		for i := 0; i+2 < dc.Count; i++ {
			a, b, c := fetch(i), fetch(i+1), fetch(i+2)
			if i%2 == 1 {
				a, b = b, a
			}
			out = append(out, [3]uint32{a, b, c})
		}
	}
	return out
}
