// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import "golang.design/x/accel/internal/kernel"

// Vec2, Vec3 and Vec4 are the vector spellings a kernel signature uses.
//
// Aliases for Go arrays rather than new named types, because the kernel
// language already spells a vector that way: std140 maps [3]float32 to a
// three-component vector consuming twelve bytes aligned to sixteen, and the
// uniform encoder is generated from exactly those array types. A parallel set
// of named vector types would give the compiler two spellings for one thing,
// and the second is the one nobody teaches the layout code about.
type (
	Vec2 = kernel.Vec2
	Vec3 = kernel.Vec3
	Vec4 = kernel.Vec4
)

// Clip is the clip-space position a vertex stage returns, with z in [-w, w].
//
// One convention, presented to every backend: that becomes NDC z in [-1, 1],
// and the backends whose native range is [0, 1] fold the remap into emitted
// code. A caller never adjusts a projection matrix for the backend. The depth
// attachment stores window depth in [0, 1], which is a different range that
// clears and compares use.
type Clip = kernel.Clip

// Vertex carries one vertex invocation's identity, and is a vertex stage's
// first parameter.
//
//	//accel:vertex
//	func Geometry(v accel.Vertex, xf Transforms,
//		pos accel.Vec3, uv accel.Vec2) (accel.Clip, Varyings)
//
// Deliberately not [Thread]: a vertex stage has no workgroup, no barrier, no
// shared memory and no subgroup, so handing it Thread would make three quarters
// of that type's methods a compile-time trap rather than an unavailable one.
type Vertex = kernel.Vertex

// Fragment carries one fragment invocation's window coordinate and facing, and
// is a fragment stage's first parameter.
//
//	//accel:fragment
//	func Shade(f accel.Fragment, in Varyings, mat Material) Targets
//
// The returned struct's fields map, in declaration order, onto the pipeline's
// colour attachments — one field per attachment, which is how MRT is expressed.
//
// Its second parameter is the varyings struct **by position**. A varyings struct
// and a uniform struct are both structs, so nothing else could tell them apart.
type Fragment = kernel.Fragment

// NoVaryings is the empty varyings struct, for a vertex stage that returns only
// a position.
//
// A named empty struct rather than allowing a stage to return one value,
// because the no-varyings case being a different signature shape is how a
// caller ends up writing the two-varying case twice.
type NoVaryings = kernel.NoVaryings

// NewVertexForTest and NewFragmentForTest build a stage receiver directly.
//
// They exist because a stage's receiver has unexported fields — an index a
// caller can set is an index the backend does not own — and the differential
// test that checks a generated lowering against its authored source has to hand
// both the same one. Nothing else should call them: a real invocation's identity
// comes from the rasterizer, which is specs/035-cpu-rasterizer.md's.
//
// Named ForTest rather than hidden behind a build tag, so that the name says
// what it is at every call site rather than only in the file header.
func NewVertexForTest(vertex, instance uint32) Vertex { return kernel.NewVertex(vertex, instance) }

// NewFragmentForTest builds a fragment receiver. See [NewVertexForTest].
func NewFragmentForTest(coord Vec4, front bool) Fragment { return kernel.NewFragment(coord, front) }
