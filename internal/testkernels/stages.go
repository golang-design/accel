// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import (
	"golang.design/x/accel"
	"golang.design/x/accel/kmath"
)

// The graphics stages of specs/032-stage-abi.md, in the corpus so the front end
// is exercised by code a reader can also read.

// Varyings is what the geometry stages pass to the shading ones.
type Varyings struct {
	Colour accel.Vec4
	UV     accel.Vec2
}

// StageTransform is a vertex stage's by-value parameter.
type StageTransform struct {
	Scale  float32
	Offset accel.Vec2
}

// GeometryVS places a vertex and hands its colour and texture coordinate on.
//
//accel:vertex
func GeometryVS(v accel.Vertex, xf StageTransform, pos accel.Vec3, uv accel.Vec2) (accel.Clip, Varyings) {
	return accel.Clip{pos[0]*xf.Scale + xf.Offset[0], pos[1]*xf.Scale + xf.Offset[1], pos[2], 1},
		Varyings{Colour: accel.Vec4{pos[0], pos[1], pos[2], 1}, UV: uv}
}

// FullScreenVS computes its position from the vertex index alone, which is what
// a vertex-buffer-less pipeline does.
//
//accel:vertex
func FullScreenVS(v accel.Vertex) (accel.Clip, accel.NoVaryings) {
	i := v.VertexIndex()
	x := float32(-1)
	y := float32(-1)
	if i == 1 {
		x = 3
	}
	if i == 2 {
		y = 3
	}
	return accel.Clip{x, y, 0, 1}, accel.NoVaryings{}
}

// Targets is a fragment stage's colour attachments, one field each.
type Targets struct {
	Albedo accel.Vec4
	Normal accel.Vec4
}

// ShadeFS writes two attachments, which is how MRT is expressed.
//
//accel:fragment
func ShadeFS(f accel.Fragment, in Varyings) Targets {
	c := f.Coord()
	return Targets{
		Albedo: in.Colour,
		Normal: accel.Vec4{in.UV[0], in.UV[1], c[2], 1},
	}
}

// HalfTriangleVS covers the lower-left half of the viewport, and covers it from
// the vertex index alone.
//
// Half rather than all of it on purpose: a stage that covers everything cannot
// tell a rasterizer that works from one that ignores its input and writes every
// pixel. The uncovered half is the part of the assertion that has teeth.
//
//accel:vertex
func HalfTriangleVS(v accel.Vertex) (accel.Clip, accel.NoVaryings) {
	x := float32(-1)
	y := float32(-1)
	if v.VertexIndex() == 1 {
		x = 1
	}
	if v.VertexIndex() == 2 {
		y = 1
	}
	return accel.Clip{x, y, 0.5, 1}, accel.NoVaryings{}
}

// Solid is a single colour attachment.
type Solid struct {
	Colour accel.Vec4
}

// SolidFS writes one constant colour, so what a test reads back is decided by
// coverage alone.
//
//accel:fragment
func SolidFS(f accel.Fragment, in accel.NoVaryings) Solid {
	return Solid{Colour: accel.Vec4{0.25, 0.5, 0.75, 1}}
}

// ColourVaryings is what the attribute stages interpolate.
type ColourVaryings struct {
	Tint accel.Vec4
}

// AttributeVS reads its position and colour from vertex buffers.
//
// No by-value parameter, unlike GeometryVS: a stage that declares one cannot be
// drawn yet, so this is the stage the attribute path can be tested through.
//
//accel:vertex
func AttributeVS(v accel.Vertex, pos accel.Vec3, tint accel.Vec4) (accel.Clip, ColourVaryings) {
	return accel.Clip{pos[0], pos[1], pos[2], 1}, ColourVaryings{Tint: tint}
}

// TintFS writes the interpolated colour, so what a test reads back is decided
// by the fetch and the interpolation together.
//
//accel:fragment
func TintFS(f accel.Fragment, in ColourVaryings) Solid {
	return Solid{Colour: in.Tint}
}

// StageTint is a fragment stage's by-value parameter.
//
// A different type from StageTransform on purpose: both stages below take a
// uniform at index 0, so a render path that gave the two stages one shared
// slice would hand this one a StageTransform and the generated adapter would
// assert on the wrong type. The types are the assertion.
type StageTint struct {
	Colour accel.Vec4
}

// ScaledVS scales its position by a by-value parameter.
//
//accel:vertex
func ScaledVS(v accel.Vertex, xf StageTransform, pos accel.Vec3) (accel.Clip, accel.NoVaryings) {
	return accel.Clip{
		pos[0]*xf.Scale + xf.Offset[0],
		pos[1]*xf.Scale + xf.Offset[1],
		pos[2], 1,
	}, accel.NoVaryings{}
}

// TintedFS writes a by-value colour, so what a test reads back names which
// uniform slice reached it.
//
// The varyings come first because specs/032-stage-abi.md identifies them by
// position; the uniform follows.
//
//accel:fragment
func TintedFS(f accel.Fragment, in accel.NoVaryings, tint StageTint) Solid {
	return Solid{Colour: tint.Colour}
}

// TexelVaryings names the texel a fetching stage reads at.
//
// Integer coordinates carried as floats, because a varying is interpolated and
// interpolation is float arithmetic. The stage converts, which is what a real
// one does: nothing in this ABI carries an integer across the rasterizer.
type TexelVaryings struct {
	Texel accel.Vec2
}

// SampledFS fetches a texel and its left neighbour.
//
// The neighbour is the point. At column zero the left fetch is at x = -1, which
// specs/032-stage-abi.md section 5 says returns zero on every backend, so the
// alpha channel this writes is the out-of-range rule made visible in a
// picture. A stage that only ever fetched in range would agree with a backend
// that read whatever was adjacent in memory.
//
//accel:fragment
func SampledFS(f accel.Fragment, in TexelVaryings, src accel.Texture2D) Solid {
	x := kmath.ToI32(in.Texel[0])
	y := kmath.ToI32(in.Texel[1])
	here := accel.Fetch(src, x, y)
	left := accel.Fetch(src, x-1, y)
	return Solid{Colour: accel.Vec4{here[0], here[1], here[2], left[0]}}
}

// DisplacedVS reads its height from a texture rather than from a vertex buffer.
//
// A vertex stage rather than a second fragment one, because the two stages
// declare their texture in different argument positions and reach different
// halves of the emitter: a fragment stage's varyings come first, and a vertex
// stage has attributes it could be confused with.
//
//accel:vertex
func DisplacedVS(v accel.Vertex, height accel.Texture2D) (accel.Clip, TexelVaryings) {
	i := int32(v.VertexIndex())
	h := accel.Fetch(height, i, 0)
	return accel.Clip{h[0], h[1], h[2], 1},
		TexelVaryings{Texel: accel.Vec2{float32(i), 0}}
}

// BlitFS returns the texel under the fragment, which is the stage
// specs/032-stage-abi.md section 5 opens with.
//
// It reads its coordinate from the built-in rather than from a varying, and
// that is what makes it the stage a two-backend comparison can be exact
// through. An interpolated coordinate is a sum of three products of a
// barycentric weight and a varying, and two rasterizers are free to compute the
// weights differently -- so int32() of one would land on a different texel from
// int32() of the other at a pixel or two, and the fetch under test would be
// blamed for the interpolation. Coord() is the pixel centre, x + 0.5, on every
// backend by the ABI, so int32 of it is x exactly.
//
//accel:fragment
func BlitFS(f accel.Fragment, in accel.NoVaryings, src accel.Texture2D) Solid {
	c := f.Coord()
	return Solid{Colour: accel.Fetch(src, kmath.ToI32(c[0]), kmath.ToI32(c[1]))}
}

// DiscardFS covers its half of the viewport and discards the other half.
//
// Half rather than all or none, for HalfTriangleVS's reason: a stage that
// discarded everything cannot tell a backend that honours a discard from one
// that draws nothing, and a stage that discarded nothing cannot tell it from
// one that ignores the call. The kept half is what gives the assertion teeth.
//
// It returns a colour after discarding because there is nothing else it can
// return, and that value is exactly what specs/032-stage-abi.md section 4.2
// says is never read -- on the CPU because the rasterizer skips the write, and
// on Metal because discard_fragment() drops the fragment. Returning the same
// colour on both paths is deliberate: a backend that ignored the discard would
// then write the *same* value it writes where it covers, so the failure is a
// covered half becoming a covered whole rather than a colour change.
//
//accel:fragment
func DiscardFS(f accel.Fragment, in accel.NoVaryings) Solid {
	c := f.Coord()
	if c[0] < 4 {
		f.Discard()
	}
	return Solid{Colour: accel.Vec4{0.25, 0.5, 0.75, 1}}
}

// IndexedVaryings carries an integer alongside a float one.
//
// specs/032-stage-abi.md section 3.1: no backend interpolates an integer, so
// the field is tagged and the tag is what the compiler checks. The float beside
// it is deliberate -- a struct whose every field were flat would not show a
// packer that applied the mask to the wrong slot.
type IndexedVaryings struct {
	Tint accel.Vec4
	ID   int32 `accel:"flat"`
}

// IndexedVS gives each vertex a distinct id, and covers the lower-left half.
//
// Distinct rather than equal, because equal ids interpolate to themselves: a
// backend that ignored the flat tag and interpolated the bit pattern would
// still produce the right answer everywhere, and the test would pass while
// checking nothing.
//
//accel:vertex
func IndexedVS(v accel.Vertex) (accel.Clip, IndexedVaryings) {
	i := v.VertexIndex()
	x := float32(-1)
	y := float32(-1)
	if i == 1 {
		x = 1
	}
	if i == 2 {
		y = 1
	}
	return accel.Clip{x, y, 0.5, 1},
		IndexedVaryings{Tint: accel.Vec4{0, 1, 0, 1}, ID: int32(i)*100 + 7}
}

// IndexedFS writes the integer varying into a channel, so a bit pattern that
// did not survive the flat form is readable as a number.
//
//accel:fragment
func IndexedFS(f accel.Fragment, in IndexedVaryings) Solid {
	return Solid{Colour: accel.Vec4{float32(in.ID), in.Tint[1], in.Tint[2], 1}}
}

// PerspectiveVaryings carries one value twice, interpolated two ways.
//
// The same source in both fields is the point: a pixel where the two disagree
// is proof the qualifiers are different operations, and it needs no second
// implementation of either formula to say so. Where w is constant they agree
// exactly, which is why the triangle this pairs with does not have constant w.
type PerspectiveVaryings struct {
	Smooth accel.Vec2
	Linear accel.Vec2 `accel:"noperspective"`
}

// PerspectiveVS covers the lower-left half from a triangle whose vertices have
// different w.
//
// Different w is the whole fixture. Perspective-correct interpolation divides
// by the interpolated 1/w and screen-linear does not, so the two coincide
// exactly when every w is equal -- and a stage drawn at w = 1 everywhere would
// compare a qualifier against itself.
//
//accel:vertex
func PerspectiveVS(v accel.Vertex) (accel.Clip, PerspectiveVaryings) {
	i := v.VertexIndex()
	pos := accel.Clip{-1, -1, 0, 1}
	if i == 1 {
		pos = accel.Clip{2, -2, 0, 2}
	}
	if i == 2 {
		pos = accel.Clip{-2, 2, 0, 2}
	}
	t := accel.Vec2{float32(i), 1}
	return pos, PerspectiveVaryings{Smooth: t, Linear: t}
}

// PerspectiveFS writes both interpolations of the same value side by side.
//
//accel:fragment
func PerspectiveFS(f accel.Fragment, in PerspectiveVaryings) Solid {
	return Solid{Colour: accel.Vec4{in.Smooth[0], in.Linear[0], in.Smooth[1], 1}}
}
