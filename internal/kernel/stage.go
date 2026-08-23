// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel

// The graphics stage ABI of specs/032-stage-abi.md: the two receivers a vertex
// and a fragment stage take, and the vector spellings their signatures use.
//
// These are the authored-side types. What the compiler does with a signature
// carrying them is the front end's subject; what a caller sees is the accel
// package's alias of each.

// Vec2, Vec3 and Vec4 are the vector spellings a stage signature uses.
//
// Aliases for Go arrays rather than new named types, because the kernel language
// already spells a vector that way: std140 maps [3]float32 to a three-component
// vector consuming twelve bytes aligned to sixteen, and the uniform encoder is
// generated from exactly those array types. A parallel set of named vector types
// would give the compiler two spellings for one thing, and the second is the one
// nobody teaches the layout code about.
//
// The names exist for reading. Clip at a signature's return says what the value
// *is*, where [4]float32 says only how wide it is.
type (
	Vec2 = [2]float32
	Vec3 = [3]float32
	Vec4 = [4]float32
)

// Clip is the clip-space position a vertex stage returns.
//
// The convention is one, and it is presented to every backend: z in [-w, w],
// which is NDC z in [-1, 1]. Backends whose native NDC range is [0, 1] -- Metal,
// Vulkan, D3D12 -- have the emitter fold the remap into the position it writes,
// so a caller never adjusts a projection matrix for the backend and never sees
// near geometry vanish for the reason docs/conventions.md describes.
//
// The depth attachment stores window depth in [0, 1]. The two ranges are
// different things and are never the same number in the same place: clears and
// compares are in [0, 1], and a vertex stage emits [-1, 1].
type Clip = Vec4

// Vertex carries one vertex invocation's identity.
//
// It is a vertex stage's first parameter, and it is deliberately not [Thread]: a
// vertex stage has no workgroup, no barrier, no shared memory and no subgroup,
// so handing it Thread would make three quarters of that type's methods a
// compile-time trap rather than an unavailable one.
//
// Its fields are unexported for the same reason Thread's are: an index a caller
// can set is an index the backend does not own, and a stage body cannot
// construct one either, since composite literals of it are outside the subset.
type Vertex struct {
	vertex, instance uint32
}

// NewVertex builds one vertex invocation's identity. It is for the rasterizer
// and the harness that drive generated stages, never for a stage body.
func NewVertex(vertex, instance uint32) Vertex {
	return Vertex{vertex: vertex, instance: instance}
}

// VertexIndex is this invocation's vertex index.
//
// For a non-indexed draw it is firstVertex + i; for an indexed draw it is the
// value read from the index buffer, before the draw's base vertex, which applies
// to the attribute fetch rather than to this number.
//
// It is what makes a vertex-buffer-less pipeline work -- a full-screen triangle
// computes its position from the index alone -- so it is neither optional nor
// capability-gated.
func (v Vertex) VertexIndex() uint32 { return v.vertex }

// InstanceIndex is firstInstance + n.
func (v Vertex) InstanceIndex() uint32 { return v.instance }

// Fragment carries one fragment invocation's position and facing.
//
// A fragment stage's first parameter, and a different type from [Vertex] for the
// same reason Vertex is a different type from Thread: the two stages have
// disjoint built-ins, and one type carrying both would make half of them return
// something meaningless rather than fail to compile.
type Fragment struct {
	coord Vec4
	front bool
}

// NewFragment builds one fragment invocation's identity, for the rasterizer and
// the harness rather than for a stage body.
func NewFragment(coord Vec4, front bool) Fragment {
	return Fragment{coord: coord, front: front}
}

// Coord is the window coordinate: xy is the pixel centre, z is window depth in
// [0, 1], and w is 1/w_clip.
//
// xy is top-origin. specs/005-graphics.md guarantees row 0 is the top row in
// three places -- here, in an on-device texel fetch, and in host readback -- and
// which of the three a backend corrects is the backend's choice, while their
// agreeing is not.
//
// z is window depth, matching what a depth attachment stores and what a depth
// compare uses, which is why docs/conventions.md's recovery to NDC is
// Coord()[2]*2 - 1 and type-checks with both sides in this ABI's own numbers.
func (f Fragment) Coord() Vec4 { return f.coord }

// FrontFacing reports whether this fragment came from a front-facing primitive,
// under the pipeline's declared winding.
//
// Declared, never defaulted: Metal's default winding disagrees with GL's, and
// getting it backwards keeps back faces instead of front faces, so the
// silhouette stays right while every per-pixel attribute comes from the wrong
// surface.
func (f Fragment) FrontFacing() bool { return f.front }

// NoVaryings is the empty varyings struct, for a vertex stage that returns only
// a position.
//
// A named empty struct rather than allowing a stage to return one value, because
// the one-varying case being a different signature shape is how a caller ends up
// writing the two-varying case twice.
type NoVaryings struct{}
