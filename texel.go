// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import "golang.design/x/accel/internal/kernel"

// Texture2D is a texture a stage reads, as a stage signature spells it.
//
//	//accel:fragment
//	func Blit(f accel.Fragment, in accel.NoVaryings, src accel.Texture2D) Solid {
//		c := f.Coord()
//		return Solid{Colour: accel.Fetch(src, int32(c[0]), int32(c[1]))}
//	}
//
// A distinct type from a slice binding and from a by-value uniform, which is
// what lets the compiler tell an image binding from a storage buffer without
// asking the author to say which is which.
//
// It names one subresource. A mip level and an array layer belong to the
// [TextureView] a pipeline binds, not to the fetch: one shape names a
// subresource, so the feedback rule that compares an attachment against a
// shader-visible binding has one thing to compare.
type Texture2D = kernel.Texture2D

// Fetch reads the texel at an integer coordinate. There is no sampler.
//
// Integer fetch is admitted and filtering is not, because a fetch has one
// definition every backend agrees on and a hardware sampler does not: its
// half-texel addressing, its LOD rounding and its per-tap integer lerps are
// per-vendor, so a filtered sample is a feature the CPU oracle cannot check. A
// stage that wants filtering builds it from fetches, where the arithmetic is
// the stage's own and is compared like any other.
//
// **A coordinate outside the texture returns Vec4{}**, on every backend, with
// negative coordinates included — the coordinates are signed so that the
// ordinary neighbourhood read at the left or top edge is representable rather
// than wrapping to an enormous unsigned index. See
// specs/032-stage-abi.md section 5.
func Fetch(t Texture2D, x, y int32) Vec4 { return kernel.Fetch(t, x, y) }

// NewTexture2D builds a texture binding over four-component texels in row-major
// order, row zero being the top row.
//
// It is for a backend binding a texture to a stage and for the tests that check
// the two lowerings agree, the way [NewVertexForTest] is: a stage body cannot
// construct one, because a texture a stage invented is a texture no backend
// bound.
func NewTexture2D(width, height int, texels []float32) Texture2D {
	return kernel.NewTexture2D(width, height, texels)
}
