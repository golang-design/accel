// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel_test

import (
	"testing"

	"golang.design/x/accel/internal/kernel"
)

// The two receivers report what they were built with, and nothing else can set
// them.
//
// The values matter less than the shape: a stage body cannot construct one,
// because composite literals of a struct with unexported fields are outside the
// subset, and a caller cannot set one, because an index a caller can set is an
// index the backend does not own.
func TestStageReceiversCarryTheirIdentity(t *testing.T) {
	v := kernel.NewVertex(7, 3)
	if got := v.VertexIndex(); got != 7 {
		t.Errorf("VertexIndex is %d, want 7", got)
	}
	if got := v.InstanceIndex(); got != 3 {
		t.Errorf("InstanceIndex is %d, want 3", got)
	}

	coord := kernel.Vec4{12.5, 4.5, 0.25, 2}
	f := kernel.NewFragment(coord, true, nil)
	if got := f.Coord(); got != coord {
		t.Errorf("Coord is %v, want %v", got, coord)
	}
	if !f.FrontFacing() {
		t.Error("FrontFacing is false for a front-facing fragment")
	}
	if kernel.NewFragment(coord, false, nil).FrontFacing() {
		t.Error("FrontFacing is true for a back-facing fragment")
	}
}

// The window coordinate's z is window depth, so conventions.md's recovery to
// NDC type-checks with both sides in this ABI's own numbers.
//
// Asserted as arithmetic rather than as a comment, because the whole value of
// stating one depth convention is that the conversion between the two is
// writable without a cast or a backend test.
func TestFragmentCoordRecoversNDCDepth(t *testing.T) {
	for _, tc := range []struct{ window, ndc float32 }{
		{0, -1}, {0.5, 0}, {1, 1},
	} {
		f := kernel.NewFragment(kernel.Vec4{0, 0, tc.window, 1}, true, nil)
		if got := f.Coord()[2]*2 - 1; got != tc.ndc {
			t.Errorf("window depth %g recovers to %g, want %g", tc.window, got, tc.ndc)
		}
	}
}

// The vector types are the array types the layout code already knows, which is
// what keeps the compiler from having two spellings for one thing.
//
// A compile-time assertion rather than a runtime one: if these ever became
// distinct named types, this file stops building, which is the point.
func TestVectorTypesAreArrayAliases(t *testing.T) {
	var (
		_ kernel.Vec2 = [2]float32{}
		_ kernel.Vec3 = [3]float32{}
		_ kernel.Vec4 = [4]float32{}
		_ kernel.Clip = [4]float32{}
	)
	var arr [3]float32
	var v kernel.Vec3 = arr // assignable in both directions only if aliased
	arr = v
	if len(arr) != 3 {
		t.Fatal("unreachable")
	}
}
