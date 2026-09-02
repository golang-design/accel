// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package std140 lays out a Go struct the way a uniform block is read.
//
// # Why std140 and not something tighter
//
// It is the portable intersection. GLSL ES 3.1 permits std140 on uniform blocks
// and does not permit std430, so choosing the tighter layout would leave the GL
// backend unable to express a uniform block at all, and the only remedy would be
// promoting every uniform to a storage buffer, losing the constant-cache path
// that makes uniforms worth having.
//
// # Why a generated codec and not an unsafe cast
//
// The padding is real and it is not Go's. std140 rounds a three-component
// vector to sixteen bytes, aligns a nested struct to sixteen, and gives an array
// of scalars a stride of sixteen; Go does none of those. An unsafe cast from a
// caller's struct to uniform bytes is silently correct for a struct of four
// floats and silently wrong for the first one containing a three-float member,
// which is the worst possible distribution of failure: it works until somebody
// trusts it.
//
// So the GPU layout owns the padding and the Go side conforms, and the caller's
// struct declares no padding field. See specs/001-device-resources.md section
// 3.3.
package std140

import (
	"fmt"
	"go/types"
	"strings"
)

// MaxNesting bounds how deep a uniform struct may nest.
//
// Not because std140 has a limit, but because a type can refer to itself
// through a pointer and this walk has no pointer to follow: without a bound a
// malformed input would recurse until the stack ran out, and a generator that
// crashes is worse than one that refuses.
const MaxNesting = 8

// Field is one member's placement in the block.
type Field struct {
	Name string

	// Offset is the byte offset from the start of the block, and Size is how
	// many bytes the member occupies there. They are the device's numbers and
	// have nothing to do with the Go struct's own layout.
	Offset int
	Size   int

	// Align is the alignment std140 gave this member, kept so a diagnostic can
	// say why an offset is where it is rather than only what it is.
	Align int

	// Kind describes the member for the encoder, which has to know whether it is
	// writing a scalar, a vector, an array, or a nested struct.
	Kind Kind

	// Elem describes a nested struct's layout.
	Elem *Layout

	// Len is an array's length, a vector's component count, or a matrix's
	// column count.
	Len int

	// Rows is a matrix's row count: the components of each column vector.
	// Separate from Len because a matrix need not be square, and a layout
	// that kept one number for both encoded a 2x4 as a 2x2.
	Rows int

	// Stride is the byte distance between an array's elements or a matrix's
	// columns, which std140 rounds up to sixteen. Zero for a scalar, a vector
	// and a nested struct, which have no elements to stride over.
	Stride int

	// Scalar is the element type for a scalar, a vector, or an array of them.
	Scalar Scalar
}

// Kind is what shape a member has.
type Kind int

const (
	KScalar Kind = iota
	KVector
	KArray
	KMatrix
	KStruct
)

// Scalar is a member's element type. The set is exactly what std140 and the
// kernel subset agree on.
type Scalar int

const (
	SInvalid Scalar = iota
	SF32
	SI32
	SU32
)

func (s Scalar) String() string {
	switch s {
	case SF32:
		return "float32"
	case SI32:
		return "int32"
	case SU32:
		return "uint32"
	}
	return "invalid"
}

// Size is a scalar's width. Every scalar std140 admits is four bytes, which is
// why a uniform block has no narrow types: there is no width to pack them into.
func (Scalar) Size() int { return 4 }

// Layout is a whole block's placement.
type Layout struct {
	Name   string
	Fields []Field

	// Size is the block's total size, rounded up to a multiple of sixteen,
	// which is the rule for the block itself.
	Size int

	// Align is the block's alignment, always sixteen.
	Align int
}

// Of computes the std140 layout of a Go struct.
//
// It walks the type with go/types rather than reflect, because the generator
// runs at build time over source it has type-checked and never over a value.
func Of(name string, t types.Type) (*Layout, error) {
	st, ok := types.Unalias(t).Underlying().(*types.Struct)
	if !ok {
		return nil, fmt.Errorf("accel: a uniform parameter is a struct, and %s is %s", name, t)
	}
	return layoutStruct(name, st, 0)
}

func layoutStruct(name string, st *types.Struct, depth int) (*Layout, error) {
	if depth > MaxNesting {
		return nil, fmt.Errorf("accel: uniform struct %s nests more than %d deep", name, MaxNesting)
	}

	out := &Layout{Name: name, Align: 16}
	offset := 0
	for f := range st.Fields() {
		if !f.Exported() {
			// The generator cannot address an unexported field from the package
			// it generates into, and skipping one silently would produce a block
			// whose size is right and whose contents are shifted.
			return nil, fmt.Errorf("accel: uniform struct %s has an unexported field %s: the "+
				"generator cannot address it, and skipping it would silently shift every "+
				"member after it", name, f.Name())
		}
		if f.Anonymous() {
			return nil, fmt.Errorf("accel: uniform struct %s embeds %s: an embedded field has no "+
				"name to place, and promoting its members would make the block's shape depend "+
				"on a type the caller did not write out", name, f.Name())
		}

		fd, err := layoutField(f.Name(), f.Type(), depth)
		if err != nil {
			return nil, fmt.Errorf("accel: uniform struct %s field %s: %w", name, f.Name(), err)
		}
		offset = align(offset, fd.Align)
		fd.Offset = offset
		offset += fd.Size
		out.Fields = append(out.Fields, *fd)
	}

	// The block itself rounds up to sixteen. Without it, an array of blocks
	// would have its second element misaligned, and a caller who sized a buffer
	// from the unrounded number would be short.
	out.Size = align(offset, 16)
	return out, nil
}

// layoutField places one member.
func layoutField(name string, t types.Type, depth int) (*Field, error) {
	t = types.Unalias(t)

	if s, ok := scalarOf(t); ok {
		return &Field{Name: name, Kind: KScalar, Scalar: s, Align: 4, Size: 4}, nil
	}

	switch u := t.Underlying().(type) {
	case *types.Array:
		return layoutArray(name, u, depth)
	case *types.Struct:
		inner, err := layoutStruct(name, u, depth+1)
		if err != nil {
			return nil, err
		}
		// A nested struct aligns to sixteen and consumes its own size rounded up
		// to sixteen, which layoutStruct already did.
		return &Field{Name: name, Kind: KStruct, Align: 16, Size: inner.Size, Elem: inner}, nil
	}

	return nil, fmt.Errorf("%s is not a type a uniform block can hold: %s", name, forbidden(t))
}

// layoutArray places an array, a vector, or a matrix.
//
// std140 does not distinguish them by name; it distinguishes them by shape. A
// [3]float32 is a three-component vector, which consumes twelve bytes aligned
// to sixteen; a [4][4]float32 is a matrix, which is an array of four column
// vectors; and a [64]float32 is an array, whose element stride rounds up to
// sixteen and therefore occupies 1024 bytes rather than 256.
//
// A matrix is an array of column vectors and nothing else: [N][M] with M at
// most four. [3][8]float32 is not a matrix in std140, it is an array of arrays
// of scalars, and each of those scalars would take its own sixteen-byte slot
// -- 384 bytes for 96 bytes of data, indexed nowhere the way a matrix is. It
// is refused, as an array of structs is, rather than laid out as a matrix
// whose rows past the fourth the encoder and the shader would disagree on.
func layoutArray(name string, a *types.Array, depth int) (*Field, error) {
	n := int(a.Len())
	if n <= 0 {
		return nil, fmt.Errorf("%s has length %d", name, n)
	}
	elem := types.Unalias(a.Elem())

	// A vector: a short array of scalars, which std140 packs rather than
	// striding. Two, three, and four components have their own rules; anything
	// longer is an array.
	if s, ok := scalarOf(elem); ok && n <= 4 {
		switch n {
		case 1:
			return &Field{Name: name, Kind: KArray, Scalar: s, Len: 1, Align: 16, Size: 16, Stride: 16}, nil
		case 2:
			return &Field{Name: name, Kind: KVector, Scalar: s, Len: 2, Align: 8, Size: 8}, nil
		case 3:
			// Twelve bytes consumed, aligned to sixteen. The next member aligns
			// by its own rule, so a scalar after a three-vector occupies the tail
			// of the same sixteen-byte slot, and that is the case an unsafe cast
			// gets wrong.
			return &Field{Name: name, Kind: KVector, Scalar: s, Len: 3, Align: 16, Size: 12}, nil
		default:
			return &Field{Name: name, Kind: KVector, Scalar: s, Len: 4, Align: 16, Size: 16}, nil
		}
	}

	// A matrix: an array of column vectors, each aligned to sixteen. The
	// column count and the row count are kept apart, because a 2x4 and a 4x2
	// have the same column stride and nothing else in common.
	if inner, ok := elem.Underlying().(*types.Array); ok {
		rows := int(inner.Len())
		s, scalar := scalarOf(types.Unalias(inner.Elem()))
		if !scalar {
			return nil, fmt.Errorf("%s is an array of arrays of arrays: std140 has no matrix "+
				"of matrices, and an array of matrices is excluded until something needs "+
				"it (specs/004-kernel-authoring.md)", name)
		}
		if rows < 1 || rows > 4 {
			return nil, fmt.Errorf("%s is [%d][%d]%s, and a std140 matrix has at most four "+
				"rows: an array of %d-element arrays gives every scalar its own sixteen-byte "+
				"slot, so put it in a storage buffer", name, n, rows, s, rows)
		}
		return &Field{
			Name: name, Kind: KMatrix, Scalar: s, Len: n, Rows: rows,
			Align: 16, Size: 16 * n, Stride: 16,
		}, nil
	}

	// An ordinary array. Its stride rounds up to sixteen whatever the element
	// is, which is why an array of 64 floats occupies 1024 bytes and why arrays
	// belong in storage buffers rather than here.
	if s, ok := scalarOf(elem); ok {
		return &Field{Name: name, Kind: KArray, Scalar: s, Len: n, Align: 16, Size: 16 * n, Stride: 16}, nil
	}

	if _, ok := elem.Underlying().(*types.Struct); ok {
		return nil, fmt.Errorf("%s is an array of structs: legal in std140 and a padding trap, "+
			"excluded until something needs it (specs/004-kernel-authoring.md)", name)
	}
	return nil, fmt.Errorf("%s has elements a uniform block cannot hold: %s", name, forbidden(elem))
}

// scalarOf reports a type's scalar kind, if it has one a uniform admits.
func scalarOf(t types.Type) (Scalar, bool) {
	b, ok := types.Unalias(t).Underlying().(*types.Basic)
	if !ok {
		return SInvalid, false
	}
	switch b.Kind() {
	case types.Float32:
		return SF32, true
	case types.Int32:
		return SI32, true
	case types.Uint32:
		return SU32, true
	}
	return SInvalid, false
}

// forbidden explains why a type cannot be in a uniform block.
//
// Each reason is specific because each is different, and a reader told only
// "unsupported type" has to guess whether the fix is a conversion, a different
// field, or a storage buffer.
func forbidden(t types.Type) string {
	b, ok := types.Unalias(t).Underlying().(*types.Basic)
	if !ok {
		switch types.Unalias(t).Underlying().(type) {
		case *types.Pointer, *types.Slice, *types.Map, *types.Chan, *types.Interface, *types.Signature:
			return "pointers, slices, maps, channels, interfaces, and functions have no device " +
				"representation"
		}
		return "it is not a scalar, vector, matrix, array, or nested struct"
	}
	switch b.Kind() {
	case types.Bool:
		return "bool is one byte in Go and four on every device, so a block containing one has " +
			"two layouts"
	case types.Int, types.Uint, types.Uintptr:
		return "int, uint, and uintptr are platform-width, so a struct holding one has no single " +
			"device layout: use int32 or uint32"
	case types.Float64:
		return "there is no f64 dtype in the compute model (specs/002-compute-model.md)"
	case types.String:
		return "a string has no memory model on a GPU"
	case types.Int8, types.Int16, types.Int64, types.Uint8, types.Uint16, types.Uint64:
		return "std140 admits only 4-byte scalars, so a narrow or wide integer has no slot: " +
			"use int32 or uint32"
	case types.Complex64, types.Complex128:
		return "there is no complex dtype"
	}
	return "it is not a scalar a uniform block can hold"
}

func align(offset, to int) int {
	if to <= 0 {
		return offset
	}
	return (offset + to - 1) / to * to
}

// String describes a layout the way a diagnostic wants it, which is offset
// first: the question a reader has is where a member landed and why.
func (l *Layout) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s (%d bytes, align %d)", l.Name, l.Size, l.Align)
	for _, f := range l.Fields {
		fmt.Fprintf(&b, "\n  %4d  %-12s %s", f.Offset, f.Name, f.describe())
	}
	return b.String()
}

func (f Field) describe() string {
	switch f.Kind {
	case KScalar:
		return fmt.Sprintf("%s, %d bytes, align %d", f.Scalar, f.Size, f.Align)
	case KVector:
		return fmt.Sprintf("%d-vector of %s, %d bytes, align %d", f.Len, f.Scalar, f.Size, f.Align)
	case KArray:
		return fmt.Sprintf("[%d]%s at stride 16, %d bytes, align %d", f.Len, f.Scalar, f.Size, f.Align)
	case KMatrix:
		return fmt.Sprintf("%d columns of %d %s at stride %d, %d bytes, align %d",
			f.Len, f.Rows, f.Scalar, f.Stride, f.Size, f.Align)
	case KStruct:
		return fmt.Sprintf("nested struct, %d bytes, align %d", f.Size, f.Align)
	}
	return "unknown"
}
