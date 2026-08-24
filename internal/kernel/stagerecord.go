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
type VertexFn func(v Vertex, uniforms []any, attrs [][]float32) (Clip, []float32)

// FragmentFn is the flat form of a fragment stage.
//
// varyings arrives interpolated, in the same flat order VertexFn produced. The
// result is one value per colour attachment, in declaration order.
type FragmentFn func(f Fragment, uniforms []any, varyings []float32) [][4]float32
