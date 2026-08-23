// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import "golang.design/x/accel/internal/kernel"

// The graphics stage ABI a shader author writes against:
// specs/032-stage-abi.md.
//
// Aliases for the same reason [Thread] is one. The backend that executes stages
// cannot import this package, so the types are declared below it, and an alias
// makes the generated code's accel.Vertex and the runtime's kernel.Vertex one
// type rather than two that have to be converted at the seam.

// Vertex carries one vertex invocation's identity, and is a vertex stage's first
// parameter.
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
// colour attachments. One field per attachment is how MRT is expressed.
type Fragment = kernel.Fragment

// Vec2, Vec3 and Vec4 are the vector spellings a stage signature uses. They are
// aliases for Go arrays, which is already how the kernel language spells a
// vector: the std140 encoder is generated from exactly these types.
type (
	Vec2 = kernel.Vec2
	Vec3 = kernel.Vec3
	Vec4 = kernel.Vec4
)

// Clip is the clip-space position a vertex stage returns, with z in [-w, w].
//
// One convention, presented to every backend. A caller never adjusts a
// projection matrix for the backend, and the depth attachment's [0, 1] window
// range is a different thing that clears and compares use.
type Clip = kernel.Clip

// NoVaryings is the empty varyings struct, for a vertex stage that returns only
// a position.
type NoVaryings = kernel.NoVaryings
