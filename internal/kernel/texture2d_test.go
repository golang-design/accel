// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel

import "testing"

// texture3x2 is three columns by two rows, each texel naming its own position
// so a fetch that reads the wrong one says which one it read.
func texture3x2() Texture2D {
	texels := make([]float32, 3*2*4)
	for y := range 2 {
		for x := range 3 {
			i := (y*3 + x) * 4
			texels[i] = float32(x)
			texels[i+1] = float32(y)
			texels[i+2] = float32(y*3 + x)
			texels[i+3] = 1
		}
	}
	return NewTexture2D(3, 2, texels)
}

// Every in-range coordinate returns its own texel, row zero being the top row.
//
// The accepting half of the bounds rule. A fetch that returned zero everywhere
// would satisfy the out-of-range test below and nothing else, which is why the
// two are separate tests rather than one.
func TestFetchReadsTheTexelAtTheCoordinate(t *testing.T) {
	tex := texture3x2()
	if got, want := tex.Width(), 3; got != want {
		t.Errorf("Width() = %d, want %d", got, want)
	}
	if got, want := tex.Height(), 2; got != want {
		t.Errorf("Height() = %d, want %d", got, want)
	}
	for y := range int32(2) {
		for x := range int32(3) {
			want := Vec4{float32(x), float32(y), float32(y*3 + x), 1}
			if got := Fetch(tex, x, y); got != want {
				t.Errorf("Fetch(%d, %d) = %v, want %v", x, y, got, want)
			}
		}
	}
}

// A coordinate outside the texture returns zero, in all four directions.
//
// specs/032-stage-abi.md section 5 fixes the rule and this is the oracle for
// it. The negative cases are the ones that matter most: a coordinate held as
// unsigned would wrap -1 into an enormous index, and an unguarded fetch would
// return whatever is adjacent in memory rather than failing.
func TestFetchOutsideTheTextureIsZero(t *testing.T) {
	tex := texture3x2()
	for _, c := range []struct {
		name string
		x, y int32
	}{
		{"left of column zero", -1, 0},
		{"far left", -1000, 1},
		{"above row zero", 0, -1},
		{"far above", 2, -1000},
		{"right of the last column", 3, 0},
		{"far right", 1 << 20, 1},
		{"below the last row", 0, 2},
		{"far below", 1, 1 << 20},
		{"both negative", -1, -1},
		{"the corner past both extents", 3, 2},
	} {
		if got := Fetch(tex, c.x, c.y); got != (Vec4{}) {
			t.Errorf("%s: Fetch(%d, %d) = %v, want the zero vector",
				c.name, c.x, c.y, got)
		}
	}
}

// A texture with no texels answers every fetch with zero rather than panicking.
//
// It is the shape a stage sees when nothing was bound, and the out-of-range
// rule already says what it returns, so the empty case needs no second rule.
func TestFetchOnAnEmptyTextureIsZero(t *testing.T) {
	for _, tex := range []Texture2D{
		{},
		NewTexture2D(0, 0, nil),
		NewTexture2D(-1, 4, make([]float32, 16)),
		NewTexture2D(2, -1, make([]float32, 16)),
	} {
		if got := Fetch(tex, 0, 0); got != (Vec4{}) {
			t.Errorf("Fetch on %+v = %v, want the zero vector", tex, got)
		}
	}
}

// A texel slice shorter than the declared extent shortens the extent, so every
// coordinate the texture reports as in range has a texel behind it.
//
// The alternative is an index past the end of the slice on the last row, which
// is a panic inside a fragment rather than a diagnosable bind-time error.
func TestATruncatedTextureLosesWholeRows(t *testing.T) {
	// Five texels of storage for a three-wide texture: one whole row and two
	// texels of a second.
	tex := NewTexture2D(3, 2, make([]float32, 5*4))
	if got, want := tex.Height(), 1; got != want {
		t.Fatalf("Height() = %d, want %d: a partial row is not a row", got, want)
	}
	if got := Fetch(tex, 2, 1); got != (Vec4{}) {
		t.Errorf("Fetch(2, 1) = %v, want the zero vector", got)
	}
}
