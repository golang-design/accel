// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel

// Stage is one compiled graphics stage: what a render pipeline is built from.
//
// A generated file declares one per stage. A caller never constructs one and
// never reads its fields; the address is the whole of the public interface, the
// same way it is for [Kernel].
//
// It is a separate record from Kernel rather than a variant of it, because
// almost nothing is shared: a stage has no workgroup, no bindings in the
// dispatch sense, and no argument set — and the fields it does have, varyings
// and attachments, mean nothing to a compute kernel. One record carrying both
// would be half-empty whichever kind it held. See specs/032-stage-abi.md.
type Stage struct {
	// Name is the authored function's name.
	Name string

	// Kind is which stage this is.
	Kind StageKind

	// Varyings is the name of the struct the two stages exchange. A pipeline
	// checks that its vertex and fragment stages agree on it; the compiler
	// already checked type identity, which is the stronger half, because two
	// structurally identical structs are not interchangeable.
	Varyings string

	// Attributes are a vertex stage's per-vertex inputs, in signature order.
	Attributes []StageAttribute

	// Uniforms are the by-value parameters, in the order the generated adapter
	// indexes them.
	//
	// Recorded for the same reason [Kernel] records its own: a caller supplies
	// a uniform by index, and without the list the render path had nothing to
	// place against and appended values in the order the caller happened to
	// write them -- so two uniforms passed out of order bound to each other's
	// parameters, and a subset shifted every one after it.
	Uniforms []StageUniform

	// Outputs are a fragment stage's colour attachments, one per field of its
	// result struct, in declaration order. One field per attachment is how MRT
	// is expressed.
	Outputs []StageOutput

	// Textures are the shader-visible texture bindings this stage declares, in
	// signature order.
	//
	// A non-empty list is what tells a render pipeline that this stage needs a
	// texture bound, and Index is the slot the pass binds against. A stage that
	// declares one and is drawn with nothing bound there is refused when the
	// graph is built, which is the alternative to an adapter fetching from an
	// empty texture: every fetch would be out of range, the pass would be black,
	// and nothing would fail.
	//
	// See specs/032-stage-abi.md section 5.
	Textures []StageTexture

	// FlatVaryings marks, per float of the flat varyings form, the ones that
	// take the provoking vertex's value rather than being interpolated.
	//
	// Per float rather than per field because that is what a rasterizer indexes:
	// specs/032-stage-abi.md section 3.1 tags a *field*, and a tagged Vec4
	// contributes four flat floats. Nil means every varying interpolates, which
	// is the default and what a stage with no tagged field carries.
	FlatVaryings []bool

	// LinearVaryings marks, per float of the flat varyings form, the ones
	// interpolated linearly in window space rather than perspective-correctly.
	// specs/032-stage-abi.md section 3.1's noperspective row.
	LinearVaryings []bool

	// Discards reports that the body reaches a discard, so a backend does not
	// promise an early depth test this stage cannot have.
	Discards bool

	// RunVertex and RunFragment are the generated adapters a rasterizer calls.
	//
	// One fixed signature per stage kind, because a rasterizer cannot know the
	// authored one: each stage has its own attributes, varyings and uniforms.
	// The generated adapter unpacks the flat form into the typed call and packs
	// the result back, which is the only place that mapping exists — so the
	// rasterizer stays free of the type system and the compiler stays the one
	// thing that knows the layout.
	//
	// Exactly one is set, matching Kind.
	RunVertex   VertexFn
	RunFragment FragmentFn

	// MSL is the generated Metal Shading Language source, and is empty when
	// the stage falls outside the subset that target lowers.
	//
	// Empty is not a fallback: a Metal render pipeline built from a stage with
	// no MSL is an error, for the reason a Metal dispatch of a kernel with no
	// MSL is. A backend that silently ran something else would make the two
	// backends disagree about what a program means.
	MSL string

	// Digest identifies the authored source this was generated from.
	Digest string

	// Generator is the ABI version, checked the way a Kernel's is.
	Generator int
}

// StageKind is which of the two graphics stages a [Stage] is.
type StageKind uint8

const (
	StageVertex StageKind = iota + 1
	StageFragment
)

func (k StageKind) String() string {
	switch k {
	case StageVertex:
		return "vertex stage"
	case StageFragment:
		return "fragment stage"
	}
	return "unknown stage"
}

// StageAttribute is one per-vertex input.
//
// Index is the dense position among the stage's attributes, which is what a
// pipeline's vertex layout binds against — not the parameter position, since the
// receiver and any uniforms are interleaved with them in the signature.
type StageAttribute struct {
	Name       string
	Index      int
	Components int
}

// StageUniform is one by-value parameter a stage declares.
//
// Index is the position in the uniforms slice the generated adapter indexes,
// not the parameter position: the receiver and any attributes are interleaved
// with the uniforms in the authored signature.
type StageUniform struct {
	Name  string
	Type  string
	Index int

	// Size is the encoded std140 block size in bytes, and Encode writes a value
	// into that layout.
	//
	// Carried for the reason [Uniform] carries them: a backend holds the value
	// as an any and needs std140 bytes, and without this it would reflect over
	// the Go struct -- a second layout implementation beside the generated
	// codec, which would disagree with it eventually.
	Size   int
	Encode func(dst []byte, v any) error

	// Decode is Encode's inverse, and it is what lets a stage's by-value
	// parameter come from a *buffer* rather than from pass state.
	//
	// specs/033-render-api.md section 4.1's recorded-offset channel hands a
	// backend std140 bytes; this rasterizer is a Go function that needs the
	// typed value. Generated from the same offsets as Encode, so the two cannot
	// drift.
	Decode func(src []byte) (any, error)
}

// StageTexture is one shader-visible texture a stage declares.
//
// Index is the dense position among the stage's textures, which is what a
// pipeline binds against — not the parameter position, since the receiver, the
// varyings, the attributes and any uniforms are interleaved with them in the
// authored signature. It is the rule [StageAttribute] follows.
type StageTexture struct {
	Name  string
	Index int

	// Reads is whether the body fetches from it, inferred rather than declared.
	// A texture nothing reads is a resource a caller has to bind for no reason,
	// and it is what tells the graph whether a pass depends on the subresource.
	Reads bool
}

// StageOutput is one colour attachment a fragment stage writes.
type StageOutput struct {
	Name  string
	Index int
}

// VertexFn is the flat form of a vertex stage.
//
// attrs is one slice per attribute, already fetched for this vertex, in the
// order Attributes declares. The result is the clip position and the varyings
// flattened into floats — which is what specs/035-cpu-rasterizer.md interpolates
// and why it never has to know about a Go struct.
//
// textures is one per [Stage.Textures] entry, indexed by StageTexture.Index and
// already resolved to the subresource the pass bound: its texels decoded from
// the bound view's format, and its extent the texture's own. A backend supplies
// every one a stage declares, so the adapter indexes rather than tests.
type VertexFn func(v Vertex, uniforms []any, attrs [][]float32, textures []Texture2D) (Clip, []float32)

// FragmentFn is the flat form of a fragment stage.
//
// varyings arrives interpolated, in the same flat order VertexFn produced.
// textures is [VertexFn]'s, indexed by this stage's own dense texture index --
// the two stages count their textures from zero the way they count their
// uniforms. The result is one value per colour attachment, in declaration
// order.
type FragmentFn func(f Fragment, uniforms []any, varyings []float32, textures []Texture2D) [][4]float32
