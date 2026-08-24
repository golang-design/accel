// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import (
	"testing"

	"golang.design/x/accel"
)

// The generated lowering of a graphics stage agrees with the source it was
// generated from.
//
// This is the same obligation specs/012-kernel-pipeline.md puts on a compute
// kernel, and it exists for the same reason: the CPU path is an oracle only
// because one IR produces both the code that runs here and the shader a GPU
// runs. A mistake in IR construction is wrong identically everywhere, so the
// authored function is still run, by this.
//
// The lowerings are unexported, so this test is in the package rather than
// beside the others in package testkernels_test.
func TestGeneratedStagesAgreeWithTheirSource(t *testing.T) {
	t.Run("vertex", func(t *testing.T) {
		v := accel.NewVertexForTest(3, 1)
		xf := StageTransform{Scale: 2, Offset: accel.Vec2{0.5, -0.25}}
		pos := accel.Vec3{1, -2, 0.5}
		uv := accel.Vec2{0.25, 0.75}

		wantPos, wantVary := GeometryVS(v, xf, pos, uv)
		gotPos, gotVary := geometryVSFlat(v, xf, pos, uv)

		if gotPos != wantPos {
			t.Errorf("clip position: generated %v, authored %v", gotPos, wantPos)
		}
		if gotVary != wantVary {
			t.Errorf("varyings: generated %v, authored %v", gotVary, wantVary)
		}
	})

	t.Run("vertex with no varyings", func(t *testing.T) {
		// Every vertex of the full-screen triangle, because the stage's whole
		// body is a branch on the index.
		for i := range uint32(3) {
			v := accel.NewVertexForTest(i, 0)
			wantPos, _ := FullScreenVS(v)
			gotPos, _ := fullScreenVSFlat(v)
			if gotPos != wantPos {
				t.Errorf("vertex %d: generated %v, authored %v", i, gotPos, wantPos)
			}
		}
	})

	t.Run("fragment", func(t *testing.T) {
		f := accel.NewFragmentForTest(accel.Vec4{12.5, 4.5, 0.25, 2}, true)
		in := Varyings{Colour: accel.Vec4{0.1, 0.2, 0.3, 1}, UV: accel.Vec2{0.4, 0.6}}

		want := ShadeFS(f, in)
		got := shadeFSFlat(f, in)
		if got != want {
			t.Errorf("attachments: generated %+v, authored %+v", got, want)
		}
	})
}

// A fragment stage's result struct maps field-for-field onto the attachments,
// in declaration order. That mapping is exact, per specs/035-cpu-rasterizer.md
// section 6, so it is asserted rather than compared within a bound.
func TestFragmentFieldsAreDistinctAttachments(t *testing.T) {
	f := accel.NewFragmentForTest(accel.Vec4{0, 0, 0.5, 1}, true)
	in := Varyings{Colour: accel.Vec4{1, 0, 0, 1}, UV: accel.Vec2{0, 1}}

	out := shadeFSFlat(f, in)
	if out.Albedo == out.Normal {
		t.Fatal("both attachments hold the same value, which a single-target test " +
			"could not distinguish from a correct one")
	}
	if out.Albedo != in.Colour {
		t.Errorf("attachment 0 is %v, want the varying's colour %v", out.Albedo, in.Colour)
	}
}
