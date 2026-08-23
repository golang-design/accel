// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package raster_test

import (
	"strings"
	"testing"

	"golang.design/x/accel/internal/raster"
)

// fromPositions returns a vertex function reading clip positions by index, and
// recording which (vertexIndex, instanceIndex) pairs it was asked for.
func fromPositions(pos []raster.Clip, seen *[][2]uint32) raster.VertexFn {
	return func(v, i, base uint32) raster.Vertex {
		if seen != nil {
			*seen = append(*seen, [2]uint32{v, i})
		}
		at := int(v) + int(base)
		if at < 0 || at >= len(pos) {
			return raster.Vertex{}
		}
		return raster.Vertex{Pos: pos[at]}
	}
}

func fbOf(n int) *raster.Framebuffer {
	return &raster.Framebuffer{Color: []*raster.ColorTarget{
		raster.NewColorTarget(n, n, [4]float32{}),
	}}
}

// A triangle list groups vertices in threes.
func TestTriangleListGroupsInThrees(t *testing.T) {
	pos := []raster.Clip{
		{X: -1, Y: -1, W: 1}, {X: 1, Y: -1, W: 1}, {X: -1, Y: 1, W: 1},
		{X: 1, Y: -1, W: 1}, {X: 1, Y: 1, W: 1}, {X: -1, Y: 1, W: 1},
	}
	var seen [][2]uint32
	if _, err := raster.DrawPrimitives(pass(8, 8), fbOf(8),
		raster.DrawCall{Topology: raster.TriangleList, Count: 6, Instances: 1},
		fromPositions(pos, &seen), constant([4]float32{1, 0, 0, 1})); err != nil {
		t.Fatal(err)
	}
	want := [][2]uint32{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}}
	if len(seen) != len(want) {
		t.Fatalf("the stage ran %d times, want %d", len(seen), len(want))
	}
	for i := range want {
		if seen[i] != want[i] {
			t.Errorf("call %d asked for %v, want %v", i, seen[i], want[i])
		}
	}

	// A count that is not a multiple of three drops the incomplete primitive
	// rather than reading past it.
	seen = nil
	if _, err := raster.DrawPrimitives(pass(8, 8), fbOf(8),
		raster.DrawCall{Topology: raster.TriangleList, Count: 5, Instances: 1},
		fromPositions(pos, &seen), constant([4]float32{1, 0, 0, 1})); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 3 {
		t.Errorf("a count of 5 ran the stage %d times, want 3: the incomplete "+
			"primitive is dropped", len(seen))
	}
}

// A triangle strip alternates its winding, so every triangle in the strip
// presents the same facing as the first.
//
// Without the swap every other triangle is back-facing, and with culling on that
// produces a striped mesh -- which reads as a geometry bug rather than a winding
// one, so a coverage test with culling off would not catch it.
func TestTriangleStripAlternatesWinding(t *testing.T) {
	// Four vertices of a quad in strip order, wound so the first triangle is
	// counter-clockwise in clip space and therefore front-facing.
	pos := []raster.Clip{
		{X: -1, Y: -1, W: 1}, {X: 1, Y: -1, W: 1},
		{X: -1, Y: 1, W: 1}, {X: 1, Y: 1, W: 1},
	}
	ps := pass(16, 16)
	ps.Cull = raster.CullBack

	fb := fbOf(16)
	n, err := raster.DrawPrimitives(ps, fb,
		raster.DrawCall{Topology: raster.TriangleStrip, Count: 4, Instances: 1},
		fromPositions(pos, nil), constant([4]float32{1, 0, 0, 1}))
	if err != nil {
		t.Fatal(err)
	}
	if n == 0 {
		t.Fatal("the strip covered nothing with back-face culling on")
	}

	// Both triangles survive culling, so the quad is filled: every pixel holds
	// the drawn colour. Without the alternating swap the second triangle is
	// culled and half the quad stays cleared.
	red := [4]float32{1, 0, 0, 1}
	for y := range 16 {
		for x := range 16 {
			if got := fb.Color[0].At(x, y); got != red {
				t.Fatalf("(%d,%d) is %v, want %v: half the strip was culled, which is "+
					"what an unalternated winding does", x, y, got, red)
			}
		}
	}
}

// The strip's triple order is asserted directly, because "the quad filled" is
// consistent with more than one correct-looking assembly.
func TestTriangleStripTripleOrder(t *testing.T) {
	pos := make([]raster.Clip, 5)
	for i := range pos {
		pos[i] = raster.Clip{W: 1}
	}
	var seen [][2]uint32
	if _, err := raster.DrawPrimitives(pass(4, 4), fbOf(4),
		raster.DrawCall{Topology: raster.TriangleStrip, Count: 5, Instances: 1},
		fromPositions(pos, &seen), constant([4]float32{})); err != nil {
		t.Fatal(err)
	}
	// Triangles 0,1,2 then 2,1,3 (swapped) then 2,3,4.
	want := []uint32{0, 1, 2, 2, 1, 3, 2, 3, 4}
	if len(seen) != len(want) {
		t.Fatalf("the stage ran %d times, want %d", len(seen), len(want))
	}
	for i, w := range want {
		if seen[i][0] != w {
			t.Fatalf("call %d asked for vertex %d, want %d; the strip order is "+
				"0,1,2 / 2,1,3 / 2,3,4", i, seen[i][0], w)
		}
	}
}

// An indexed draw reads its vertex indices from the index buffer, and First
// offsets into that buffer rather than into the vertices.
func TestIndexedDrawReadsTheIndexBuffer(t *testing.T) {
	pos := make([]raster.Clip, 8)
	for i := range pos {
		pos[i] = raster.Clip{W: 1}
	}
	var seen [][2]uint32
	if _, err := raster.DrawPrimitives(pass(4, 4), fbOf(4), raster.DrawCall{
		Topology: raster.TriangleList,
		Count:    3, Instances: 1, First: 2,
		Index: []uint32{9, 9, 5, 6, 7},
	}, fromPositions(pos, &seen), constant([4]float32{})); err != nil {
		t.Fatal(err)
	}
	want := []uint32{5, 6, 7}
	if len(seen) != 3 {
		t.Fatalf("the stage ran %d times, want 3", len(seen))
	}
	for i, w := range want {
		if seen[i][0] != w {
			t.Errorf("call %d asked for vertex %d, want %d: First offsets into the "+
				"index buffer, not into the vertices", i, seen[i][0], w)
		}
	}
}

// BaseVertex reaches the attribute fetch and not the index the stage sees.
//
// specs/032-stage-abi.md section 2.1 declines to expose a base-vertex built-in
// because backends disagree about whether theirs reports the pre-offset or
// post-offset value; this asserts which of the two accel means.
func TestBaseVertexOffsetsTheFetchNotTheIndex(t *testing.T) {
	pos := make([]raster.Clip, 8)
	for i := range pos {
		pos[i] = raster.Clip{W: 1}
	}
	var indices, fetched []uint32
	vs := func(v, i, base uint32) raster.Vertex {
		indices = append(indices, v)
		fetched = append(fetched, v+base)
		return raster.Vertex{Pos: pos[v+base]}
	}
	if _, err := raster.DrawPrimitives(pass(4, 4), fbOf(4), raster.DrawCall{
		Topology: raster.TriangleList,
		Count:    3, Instances: 1, BaseVertex: 4,
		Index: []uint32{0, 1, 2},
	}, vs, constant([4]float32{})); err != nil {
		t.Fatal(err)
	}
	for i, want := range []uint32{0, 1, 2} {
		if indices[i] != want {
			t.Errorf("the stage saw vertex index %d, want the un-offset %d",
				indices[i], want)
		}
	}
	for i, want := range []uint32{4, 5, 6} {
		if fetched[i] != want {
			t.Errorf("the fetch used %d, want the offset %d", fetched[i], want)
		}
	}
}

// Instancing replays the whole primitive set, with FirstInstance offsetting the
// index the stage sees.
func TestInstancingReplaysThePrimitiveSet(t *testing.T) {
	pos := []raster.Clip{{W: 1}, {W: 1}, {W: 1}}
	var seen [][2]uint32
	if _, err := raster.DrawPrimitives(pass(4, 4), fbOf(4), raster.DrawCall{
		Topology: raster.TriangleList,
		Count:    3, Instances: 3, FirstInstance: 10,
	}, fromPositions(pos, &seen), constant([4]float32{})); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 9 {
		t.Fatalf("the stage ran %d times, want 9: three vertices times three instances",
			len(seen))
	}
	for i, call := range seen {
		wantVertex := uint32(i % 3)
		wantInstance := uint32(10 + i/3)
		if call != [2]uint32{wantVertex, wantInstance} {
			t.Errorf("call %d is %v, want {%d %d}", i, call, wantVertex, wantInstance)
		}
	}

	// Zero instances draws nothing, which is what a graph recorded for a fixed
	// maximum uses for its absent objects.
	seen = nil
	if _, err := raster.DrawPrimitives(pass(4, 4), fbOf(4), raster.DrawCall{
		Topology: raster.TriangleList, Count: 3, Instances: 0,
	}, fromPositions(pos, &seen), constant([4]float32{})); err != nil {
		t.Fatal(err)
	}
	if len(seen) != 0 {
		t.Errorf("a zero-instance draw ran the stage %d times", len(seen))
	}
}

// The three topologies whose rules this rasterizer does not state are refused by
// name rather than approximated.
func TestUnstatedTopologiesAreRefused(t *testing.T) {
	for _, tp := range []raster.Topology{raster.LineList, raster.LineStrip, raster.PointList} {
		_, err := raster.DrawPrimitives(pass(4, 4), fbOf(4),
			raster.DrawCall{Topology: tp, Count: 3, Instances: 1},
			fromPositions(nil, nil), constant([4]float32{}))
		if err == nil {
			t.Errorf("%v was rasterized, and its fill rule is unstated", tp)
			continue
		}
		if !strings.Contains(err.Error(), tp.String()) {
			t.Errorf("the refusal for %v does not name it: %v", tp, err)
		}
	}
	// And an unknown value still prints something a reader can act on.
	if got := raster.Topology(99).String(); !strings.Contains(got, "99") {
		t.Errorf("an unknown topology prints %q", got)
	}
}

// Negative counts are refused rather than silently drawing nothing, because the
// two have different causes and a silent zero hides the arithmetic that produced
// it.
func TestNegativeCountsAreRefused(t *testing.T) {
	for _, dc := range []raster.DrawCall{
		{Topology: raster.TriangleList, Count: -1, Instances: 1},
		{Topology: raster.TriangleList, Count: 3, Instances: -1},
		{Topology: raster.TriangleList, Count: 3, Instances: 1, First: -1},
	} {
		if _, err := raster.DrawPrimitives(pass(4, 4), fbOf(4), dc,
			fromPositions(nil, nil), constant([4]float32{})); err == nil {
			t.Errorf("%+v was accepted", dc)
		}
	}
}

// An index past the end of the index buffer does not panic inside the
// rasterizer. Validating the range is the layer above's job; being total is
// this one's.
func TestIndexPastTheBufferDoesNotPanic(t *testing.T) {
	pos := []raster.Clip{{W: 1}}
	if _, err := raster.DrawPrimitives(pass(4, 4), fbOf(4), raster.DrawCall{
		Topology: raster.TriangleList,
		Count:    3, Instances: 1, First: 1,
		Index: []uint32{0, 0},
	}, fromPositions(pos, nil), constant([4]float32{})); err != nil {
		t.Fatal(err)
	}
}

// The reported fragment count sums over every primitive and instance, which is
// what tells a caller that a draw produced nothing because of a test rather than
// because it assembled nothing.
func TestDrawPrimitivesReportsFragments(t *testing.T) {
	const n = 8
	// One triangle covering the whole viewport, drawn twice as two instances.
	pos := []raster.Clip{
		{X: -1, Y: -1, W: 1}, {X: 3, Y: -1, W: 1}, {X: -1, Y: 3, W: 1},
	}
	got, err := raster.DrawPrimitives(pass(n, n), fbOf(n), raster.DrawCall{
		Topology: raster.TriangleList, Count: 3, Instances: 2,
	}, fromPositions(pos, nil), constant([4]float32{1, 0, 0, 1}))
	if err != nil {
		t.Fatal(err)
	}
	if want := 2 * n * n; got != want {
		t.Errorf("%d fragments written, want %d from two full-viewport instances",
			got, want)
	}
}
