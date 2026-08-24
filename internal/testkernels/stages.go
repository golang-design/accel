// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import "golang.design/x/accel"

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
