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

	// Outputs are a fragment stage's colour attachments, one per field of its
	// result struct, in declaration order. One field per attachment is how MRT
	// is expressed.
	Outputs []StageOutput

	// Discards reports that the body reaches a discard, so a backend does not
	// promise an early depth test this stage cannot have.
	Discards bool

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

// StageOutput is one colour attachment a fragment stage writes.
type StageOutput struct {
	Name  string
	Index int
}
