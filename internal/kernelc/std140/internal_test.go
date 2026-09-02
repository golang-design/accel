// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package std140

import (
	"go/types"
	"strings"
	"testing"
)

// TestScalarStrings covers what a diagnostic prints for each element type.
func TestScalarStrings(t *testing.T) {
	for _, tc := range []struct {
		s    Scalar
		want string
	}{
		{SF32, "float32"},
		{SI32, "int32"},
		{SU32, "uint32"},
		{SInvalid, "invalid"},
		{Scalar(99), "invalid"},
	} {
		if got := tc.s.String(); got != tc.want {
			t.Errorf("Scalar(%d).String() = %q, want %q", int(tc.s), got, tc.want)
		}
	}

	// Every scalar std140 admits is four bytes, which is why a uniform block has
	// no narrow types: there is no width to pack them into.
	for _, s := range []Scalar{SF32, SI32, SU32} {
		if got := s.Size(); got != 4 {
			t.Errorf("%v.Size() = %d, want 4", s, got)
		}
	}
}

// TestAlignEdges covers the rounding every offset in a block goes through.
func TestAlignEdges(t *testing.T) {
	for _, tc := range []struct{ offset, to, want int }{
		{0, 16, 0},
		{1, 16, 16},
		{16, 16, 16},
		{17, 16, 32},
		{4, 4, 4},
		{5, 8, 8},
		// A non-positive alignment leaves the offset alone rather than dividing
		// by zero. It cannot arise from a layout and the guard is what keeps a
		// malformed one from crashing a generator.
		{7, 0, 7},
		{7, -4, 7},
	} {
		if got := align(tc.offset, tc.to); got != tc.want {
			t.Errorf("align(%d, %d) = %d, want %d", tc.offset, tc.to, got, tc.want)
		}
	}
}

// TestFieldDescriptions covers each shape's rendering, which is what a reader
// sees when an offset is not where they expected.
func TestFieldDescriptions(t *testing.T) {
	for _, tc := range []struct {
		f    Field
		want string
	}{
		{Field{Kind: KScalar, Scalar: SF32, Size: 4, Align: 4}, "float32, 4 bytes"},
		{Field{Kind: KVector, Scalar: SF32, Len: 3, Size: 12, Align: 16}, "3-vector"},
		{Field{Kind: KArray, Scalar: SI32, Len: 8, Size: 128, Align: 16}, "stride 16"},
		{Field{Kind: KMatrix, Scalar: SF32, Len: 4, Size: 64, Align: 16}, "4 columns"},
		{Field{Kind: KStruct, Size: 16, Align: 16}, "nested struct"},
		{Field{Kind: Kind(99)}, "unknown"},
	} {
		if got := tc.f.describe(); !strings.Contains(got, tc.want) {
			t.Errorf("describe() = %q, want it to contain %q", got, tc.want)
		}
	}
}

// TestZeroLengthArray covers a shape Go permits and a uniform block cannot use.
func TestZeroLengthArray(t *testing.T) {
	arr := types.NewArray(types.Typ[types.Float32], 0)
	if _, err := layoutArray("A", arr, 0); err == nil {
		t.Error("a zero-length array was accepted")
	}
}

// TestForbiddenCoversEveryReason exercises the explanations directly, since
// each names a different fix and a reader told only "unsupported" has to guess.
func TestForbiddenCoversEveryReason(t *testing.T) {
	for _, tc := range []struct {
		t    types.Type
		want string
	}{
		{types.Typ[types.Bool], "one byte in Go"},
		{types.Typ[types.Int], "platform-width"},
		{types.Typ[types.Float64], "no f64 dtype"},
		{types.Typ[types.String], "no memory model"},
		{types.Typ[types.Int16], "only 4-byte scalars"},
		{types.Typ[types.Complex128], "no complex dtype"},
		{types.NewSlice(types.Typ[types.Float32]), "no device representation"},
		{types.NewChan(types.SendRecv, types.Typ[types.Int32]), "no device representation"},
		{types.NewSignatureType(nil, nil, nil, nil, nil, false), "no device representation"},
		{types.NewInterfaceType(nil, nil), "no device representation"},
		{types.Typ[types.UnsafePointer], "not a scalar a uniform block can hold"},
		{types.NewTuple(), "not a scalar, vector, matrix"},
	} {
		if got := forbidden(tc.t); !strings.Contains(got, tc.want) {
			t.Errorf("forbidden(%s) = %q, want it to contain %q", tc.t, got, tc.want)
		}
	}
}

// TestArrayOfArraysOfArrays covers the three-dimensional array, which nothing
// in the corpus declares.
//
// It was laid out as a matrix of 2 columns with a 64-byte stride, and the
// encoder then wrote A[c][r] for c and r below 2 -- four of its thirty-two
// floats -- against an MSL declaration of `float A[2][16]`. An array of
// matrices is refused as an array of structs is, until something needs it.
func TestArrayOfArraysOfArrays(t *testing.T) {
	inner := types.NewArray(types.Typ[types.Float32], 4)
	mid := types.NewArray(inner, 4)
	outer := types.NewArray(mid, 2)

	_, err := layoutArray("A", outer, 0)
	if err == nil {
		t.Fatal("an array of matrices was laid out, and the encoder had no shape for it")
	}
	if !strings.Contains(err.Error(), "array of matrices") {
		t.Errorf("the refusal should say what it is: %v", err)
	}
}

// TestArrayOfUnsupportedElements covers the leaf that is neither scalar,
// nested array, nor struct.
func TestArrayOfUnsupportedElements(t *testing.T) {
	arr := types.NewArray(types.NewSlice(types.Typ[types.Float32]), 2)
	if _, err := layoutArray("A", arr, 0); err == nil {
		t.Error("an array of slices was accepted")
	}
}
