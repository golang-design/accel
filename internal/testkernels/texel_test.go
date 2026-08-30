// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import (
	"testing"

	"golang.design/x/accel"
)

// The texel fetch of specs/032-stage-abi.md section 5, checked the way every
// other corpus entry is: the generated lowering is run against the authored
// source it was generated from.
//
// The lowerings are unexported, so this test is in the package rather than
// beside the others in package testkernels_test.

// checker3x2 is three columns by two rows, each texel naming its own position.
//
// Distinct values per channel and per texel, so a fetch that reads the wrong
// texel, transposes x and y, or returns a constant is a different number rather
// than the same one.
func checker3x2() accel.Texture2D {
	texels := make([]float32, 3*2*4)
	for y := range 2 {
		for x := range 3 {
			i := (y*3 + x) * 4
			texels[i] = float32(10*y + x)
			texels[i+1] = float32(100 + 10*y + x)
			texels[i+2] = float32(200 + 10*y + x)
			texels[i+3] = 1
		}
	}
	return accel.NewTexture2D(3, 2, texels)
}

// The generated fragment lowering agrees with its authored source, at every
// texel of the texture and at every coordinate just outside it.
func TestGeneratedFetchingStagesAgreeWithTheirSource(t *testing.T) {
	tex := checker3x2()
	f := accel.NewFragmentForTest(accel.Vec4{0.5, 0.5, 0.25, 1}, true)

	t.Run("fragment", func(t *testing.T) {
		for y := -1; y <= 2; y++ {
			for x := -1; x <= 3; x++ {
				in := TexelVaryings{Texel: accel.Vec2{float32(x), float32(y)}}
				want := SampledFS(f, in, tex)
				got := sampledFSFlat(f, in, tex)
				if got != want {
					t.Errorf("texel (%d, %d): generated %+v, authored %+v",
						x, y, got, want)
				}
			}
		}
	})

	t.Run("vertex", func(t *testing.T) {
		// Vertex 3 is past the last column, so its fetch is out of range and
		// its position is the origin with w = 1.
		for i := range uint32(4) {
			v := accel.NewVertexForTest(i, 0)
			wantPos, wantVary := DisplacedVS(v, tex)
			gotPos, gotVary := displacedVSFlat(v, tex)
			if gotPos != wantPos {
				t.Errorf("vertex %d: position generated %v, authored %v", i, gotPos, wantPos)
			}
			if gotVary != wantVary {
				t.Errorf("vertex %d: varyings generated %v, authored %v", i, gotVary, wantVary)
			}
		}
	})
}

// A fetch in range returns that texel, and the neighbour fetch at column zero
// returns zero because x = -1 is outside the texture.
//
// The agreement test above would pass if both halves fetched nothing, since
// both call the same oracle. This says what the values are.
func TestAFetchingStageReadsTheTextureAndZeroOutsideIt(t *testing.T) {
	tex := checker3x2()
	f := accel.NewFragmentForTest(accel.Vec4{0.5, 0.5, 0.25, 1}, true)

	// Column one of row one: in range, and its left neighbour is in range too.
	got := sampledFSFlat(f, TexelVaryings{Texel: accel.Vec2{1, 1}}, tex)
	want := accel.Vec4{10 + 1, 100 + 10 + 1, 200 + 10 + 1, 10 + 0}
	if got.Colour != want {
		t.Errorf("texel (1, 1) = %v, want %v", got.Colour, want)
	}

	// Column zero: in range, and its left neighbour is at x = -1.
	got = sampledFSFlat(f, TexelVaryings{Texel: accel.Vec2{0, 1}}, tex)
	want = accel.Vec4{10, 110, 210, 0}
	if got.Colour != want {
		t.Errorf("texel (0, 1) = %v, want %v: the alpha carries the fetch at x = -1, "+
			"which specs/032-stage-abi.md section 5 fixes at zero", got.Colour, want)
	}

	// A row past the last one: both fetches are out of range.
	got = sampledFSFlat(f, TexelVaryings{Texel: accel.Vec2{1, 2}}, tex)
	if got.Colour != (accel.Vec4{}) {
		t.Errorf("texel (1, 2) = %v, want the zero vector: row 2 is outside a "+
			"two-row texture", got.Colour)
	}
}

// Every stage carries a flat adapter, and one that declares a texture carries
// the texture in its record too.
//
// This asserted the opposite until the flat form gained a texture channel. The
// reasoning then was sound and is worth keeping visible: the form took a
// uniform slice and interpolated floats and had nowhere to put a texture, so
// emitting an adapter for a fetching stage would have passed an empty one,
// every fetch would have been out of range, and the pass would have produced
// black without failing anything. The answer was to withhold the adapter, which
// made the stage unrunnable on the backend that is the oracle.
//
// The channel exists now, so the adapter is emitted and the guarantee moves: a
// stage that declares a texture must still *declare* it, because that record is
// what the pass binds against and what refuses a draw that bound nothing.
func TestAStageWithATextureDeclaresIt(t *testing.T) {
	for _, s := range Stages {
		if s.RunVertex == nil && s.RunFragment == nil {
			t.Errorf("%s has no flat adapter, so the CPU backend cannot run it", s.Name)
		}
		if len(s.Textures) == 0 {
			continue
		}
		for i, tx := range s.Textures {
			if tx.Index != i {
				t.Errorf("%s: texture %q is at index %d, want %d", s.Name, tx.Name, tx.Index, i)
			}
			if !tx.Reads {
				t.Errorf("%s: texture %q is declared and never read", s.Name, tx.Name)
			}
		}
	}
}

// The origin corpus entry's two stages, generated against authored.
//
// specs/010-kernel-corpus.md section 6: every corpus entry's authored form is
// run against its generated lowering. It is asserted here, in a portable file,
// because the Metal differential is what exercises a lowering on a Mac and it
// does not run anywhere else -- so a stage with no portable caller reads as
// covered on darwin and drops the Linux gate. That has happened three times.
func TestTheOriginStagesAgreeWithTheirSource(t *testing.T) {
	tex := checker3x2()

	t.Run("RowFS", func(t *testing.T) {
		// Every pixel of a small grid, because the stage's whole content is
		// its own coordinate and a constant would agree with anything.
		for y := range 3 {
			for x := range 4 {
				f := accel.NewFragmentForTest(
					accel.Vec4{float32(x) + 0.5, float32(y) + 0.5, 0.25, 1}, true)
				want := RowFS(f, accel.NoVaryings{})
				if got := rowFSFlat(f, accel.NoVaryings{}); got != want {
					t.Errorf("(%d,%d): generated %+v, authored %+v", x, y, got, want)
				}
				if want.Colour[0] != float32(x)+0.5 || want.Colour[1] != float32(y)+0.5 {
					t.Errorf("(%d,%d) reports %v, and a target encoding row position "+
						"has to name where it is", x, y, want.Colour)
				}
			}
		}
	})

	t.Run("TopRowFS", func(t *testing.T) {
		// Rows above zero as well, since the stage's point is that it fetches
		// row zero whatever row it is shading.
		for y := range 3 {
			for x := range 4 {
				f := accel.NewFragmentForTest(
					accel.Vec4{float32(x) + 0.5, float32(y) + 0.5, 0.25, 1}, true)
				want := TopRowFS(f, accel.NoVaryings{}, tex)
				if got := topRowFSFlat(f, accel.NoVaryings{}, tex); got != want {
					t.Errorf("(%d,%d): generated %+v, authored %+v", x, y, got, want)
				}
			}
		}
		// And it is row zero rather than its own row: the checker's rows differ
		// by ten, so shading row 1 and fetching row 1 would be a different
		// number.
		top := TopRowFS(accel.NewFragmentForTest(accel.Vec4{0.5, 0.5, 0, 1}, true),
			accel.NoVaryings{}, tex)
		lower := TopRowFS(accel.NewFragmentForTest(accel.Vec4{0.5, 1.5, 0, 1}, true),
			accel.NoVaryings{}, tex)
		if top != lower {
			t.Errorf("shading row 0 gives %v and row 1 gives %v; the fetch is at its own "+
				"row rather than at row zero", top.Colour, lower.Colour)
		}
	})
}
