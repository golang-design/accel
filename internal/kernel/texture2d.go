// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel

// Integer texel fetch, and the shader-visible texture binding it reads.
//
// specs/032-stage-abi.md section 5 admits exactly one texture operation: an
// indexed load at an integer coordinate, with no filter, no LOD selection and
// no addressing mode. A filtered sampler is refused there because a CPU oracle
// cannot reproduce a hardware sampler's half-texel addressing, its LOD
// rounding, or its uint8-truncating lerps, so a sampler is a feature the oracle
// cannot check. A fetch has one definition, and a stage that wants filtering
// builds it from fetches.
//
// # Why one element type
//
// [Texture2D] holds four float32 components per texel and [Fetch] returns
// [Vec4], whatever the bound texture's format is. That mirrors what every
// target already does for a float-sampled texture -- MSL's texture2d<float>,
// GLSL's sampler2D, HLSL's Texture2D<float4> -- where the format's decode
// happens in fixed function on the way out and the shader sees four floats.
// Integer-typed textures would need a second binding type and a second
// intrinsic, and there is no consumer for one yet.

// Texture2D is a shader-visible two-dimensional texture binding.
//
// It names one subresource. The mip level and the array layer are the *view's*,
// not the fetch's: specs/045-texture-attachments.md section 2 puts them on
// accel.TextureView so that specs/033-render-api.md section 3.3's feedback
// rejection -- which compares an attachment's subresource with a
// shader-visible one -- has a single shape to compare, and so that the
// comparison is decidable when a pipeline is built rather than when a fragment
// runs.
//
// Its fields are unexported for the reason [Vertex]'s are: a stage body cannot
// construct one, since a composite literal of it would let a stage invent a
// texture the backend never bound.
type Texture2D struct {
	texels        []float32
	width, height int32
}

// NewTexture2D builds a texture binding over four-component texels in row-major
// order, row zero being the top row.
//
// texels is not copied. It is the backend's own storage, which is what makes a
// fetch of a previous pass's output a read of that pass's attachment rather
// than of a snapshot taken at bind time.
//
// A texture whose texel slice is shorter than width*height*4 is refused by
// truncating its extent rather than by panicking mid-fetch: the caller is a
// backend, the failure would be per-fragment, and a fetch of a row that is not
// there returns zero for exactly the reason an out-of-range coordinate does.
func NewTexture2D(width, height int, texels []float32) Texture2D {
	if width < 0 || height < 0 {
		return Texture2D{}
	}
	if n := width * height * 4; len(texels) < n {
		// Keep whole rows only, so that the extent and the storage agree and
		// every in-range coordinate has a texel.
		if width == 0 {
			return Texture2D{}
		}
		height = len(texels) / (width * 4)
	}
	return Texture2D{texels: texels, width: int32(width), height: int32(height)}
}

// Width and Height are the bound subresource's extent in texels.
//
// Host-side accessors for the backend and for tests. A stage body cannot call
// them: querying an extent is a second intrinsic, and specs/032-stage-abi.md
// section 5 admits one.
func (t Texture2D) Width() int  { return int(t.width) }
func (t Texture2D) Height() int { return int(t.height) }

// Fetch reads the texel at (x, y) and returns its four components.
//
// # Out of range returns zero
//
// x < 0, y < 0, x >= width or y >= height all give Vec4{}, on every backend.
// specs/032-stage-abi.md section 5 fixes it there and this is the oracle for it.
//
// Zero rather than a clamp for two reasons. It is the answer a backend can
// guarantee with a bounds test it already has to emit -- Metal leaves an
// out-of-range texture2d::read undefined, so the emitted MSL tests the
// coordinate whatever rule is chosen -- while a clamp would additionally have
// to agree on which edge a corner belongs to. And it is the answer that stays
// correct when a stage builds its own addressing on top: a clamp baked into the
// primitive cannot be undone, whereas a border of zeros is what a stage doing
// its own clamp, wrap or mirror computes *from*.
//
// The coordinates are signed. An unsigned coordinate cannot represent -1, and
// the fetch that reaches -1 is the ordinary one: a neighbourhood read at the
// left or top edge. Making that unrepresentable would move the defect from a
// zero this function returns to a wrap this function cannot see.
func Fetch(t Texture2D, x, y int32) Vec4 {
	if x < 0 || y < 0 || x >= t.width || y >= t.height {
		return Vec4{}
	}
	i := (int(y)*int(t.width) + int(x)) * 4
	return Vec4{t.texels[i], t.texels[i+1], t.texels[i+2], t.texels[i+3]}
}
