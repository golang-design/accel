// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package std140_test

import (
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/std140"
)

// A matrix is an array of column vectors, and its two extents are kept apart.
//
// The layout kept one number, the column count, and the encoder and the MSL
// declaration read it as the row count too: [2][4]float32 encoded M[c][0..1]
// and left the rest zero, and [3][8]float32 got 416 bytes of block, a
// `float A[3][4]` in MSL and a codec writing three of its eight rows while the
// body indexed A[2][7].
func TestAMatrixCarriesItsColumnsAndRowsSeparately(t *testing.T) {
	for _, tc := range []struct {
		name         string
		decl         string
		cols, rows   int
		size, stride int
	}{
		{"2 columns of 4", "type T struct{ M [2][4]float32 }", 2, 4, 32, 16},
		{"4 columns of 2", "type T struct{ M [4][2]float32 }", 4, 2, 64, 16},
		{"3 columns of 3", "type T struct{ M [3][3]int32 }", 3, 3, 48, 16},
		{"2 columns of 1", "type T struct{ M [2][1]uint32 }", 2, 1, 32, 16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			l, err := std140.Of("T", structNamed(t, tc.decl, "T"))
			if err != nil {
				t.Fatalf("Of: %v", err)
			}
			f := l.Fields[0]
			if f.Kind != std140.KMatrix {
				t.Fatalf("kind is %v, want a matrix", f.Kind)
			}
			if f.Len != tc.cols || f.Rows != tc.rows {
				t.Errorf("%d columns of %d, want %d of %d", f.Len, f.Rows, tc.cols, tc.rows)
			}
			if f.Size != tc.size || f.Stride != tc.stride || f.Align != 16 {
				t.Errorf("size %d stride %d align %d, want %d, %d, 16", f.Size, f.Stride, f.Align, tc.size, tc.stride)
			}
			if !strings.Contains(l.String(), "columns of") {
				t.Errorf("the description does not say it is a matrix:\n%s", l)
			}
		})
	}
}

// An array of scalars carries its stride, which is the layout's number and
// not one the front end restates.
func TestAnArrayCarriesItsStride(t *testing.T) {
	for _, tc := range []struct {
		decl      string
		n, stride int
	}{
		{"type T struct{ A [1]float32 }", 1, 16},
		{"type T struct{ A [6]float32 }", 6, 16},
		{"type T struct{ A [64]uint32 }", 64, 16},
	} {
		l, err := std140.Of("T", structNamed(t, tc.decl, "T"))
		if err != nil {
			t.Fatalf("Of: %v", err)
		}
		f := l.Fields[0]
		if f.Kind != std140.KArray || f.Len != tc.n || f.Stride != tc.stride || f.Size != tc.n*tc.stride {
			t.Errorf("%s: %+v, want an array of %d at stride %d", tc.decl, f, tc.n, tc.stride)
		}
	}
}

// std140 has no matrix with more than four rows and no matrix of matrices.
// Each was laid out as a matrix whose rows past the fourth the encoder and the
// shader disagreed on; each is refused naming the field.
func TestWideAndNestedMatricesAreRefused(t *testing.T) {
	for _, tc := range []struct {
		name, decl, want string
	}{
		{"3 columns of 8", "type T struct{ A [3][8]float32 }", "at most four rows"},
		{"1 column of 5", "type T struct{ A [1][5]int32 }", "at most four rows"},
		{"array of matrices", "type T struct{ A [2][4][4]float32 }", "array of matrices"},
		{"matrix of vectors of vectors", "type T struct{ A [2][2][2]float32 }", "array of matrices"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := std140.Of("T", structNamed(t, tc.decl, "T"))
			if err == nil {
				t.Fatalf("%s was laid out, and no std140 matrix has that shape", tc.decl)
			}
			if !strings.Contains(err.Error(), "field A") || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("the refusal should name field A and say %q: %v", tc.want, err)
			}
		})
	}
}
