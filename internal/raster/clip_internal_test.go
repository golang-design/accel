// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package raster

import (
	"math"
	"testing"
)

// throughTheEye is a triangle whose edge a-b passes through the clip-space
// origin: a is in front of the near plane, b behind it, and the crossing lands
// at z = w = 0 exactly.
//
// That point satisfies both plane tests -- z >= -w and z <= w are each 0 >= 0
// -- and has no window position, since dividing by w is dividing by zero.
func throughTheEye() [3]Vertex {
	return [3]Vertex{
		{Pos: Clip{X: -1, Y: -1, Z: 1, W: 1}},
		{Pos: Clip{X: 1, Y: 2, Z: -1, W: -1}},
		{Pos: Clip{X: 1, Y: 0, Z: 0.5, W: 1}},
	}
}

// Clipping never hands the perspective divide a vertex with w <= 0.
//
// A vertex at w = 0 is the projective point at infinity. It survives both
// plane tests, and toWindow then computes 1/0 and 0 * Inf: an infinite 1/w and
// NaN window coordinates, which bounds turns into an integer through int(NaN),
// a conversion whose result differs between arm64 and amd64. Coverage of a
// primitive through the eye must not depend on which one ran the test.
func TestClipNeverYieldsAVertexWithoutAWindowPosition(t *testing.T) {
	verts, ok := clipNearFar(throughTheEye())
	if !ok {
		t.Fatal("a triangle with two vertices in front of the near plane covers " +
			"something, so clipping must keep it")
	}
	for i, v := range verts {
		if !(v.Pos.W > 0) {
			t.Errorf("clipped vertex %d has w = %v: it has no window position and the "+
				"divide would produce %v", i, v.Pos.W, 1/v.Pos.W)
		}
		w := toWindow(Viewport{W: 8, H: 8, MaxDepth: 1}, v)
		if math.IsNaN(float64(w.x)) || math.IsNaN(float64(w.y)) || math.IsInf(float64(w.invW), 0) {
			t.Errorf("clipped vertex %d maps to window (%v, %v) with 1/w = %v", i, w.x, w.y, w.invW)
		}
	}
}
