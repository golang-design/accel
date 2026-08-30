// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"fmt"
)

// The vertex input layout of specs/033-render-api.md section 2.
//
// # Why this is declared and not inferred
//
// The compiler knows an attribute's type: a vertex stage's by-value array
// parameter is [N]float32 and the generated record carries N. It does not know
// where the bytes are. Byte offsets, strides and step modes are properties of
// the *buffer*, not of the stage — the same stage reads an interleaved buffer
// and three planar ones, and the same buffer feeds stages that read a subset of
// its attributes. Inferring them would either guess a packing or need
// annotations that are a descriptor by another name.
//
// What the compiler does supply is validation. Section 2.2 checks the declared
// formats against the stage's parameters, so the second declaration cannot
// disagree with the first without an error.

// AttrFormat is the type of one vertex attribute in memory.
//
// A stage's attribute parameter is [N]float32, so every format arrives at the
// stage as floats. What differs is the bytes in the buffer and the conversion
// between them, and the normalized integer formats exist because that is how
// meshes are actually stored: a normal, a tangent and a colour are each three
// or four bytes, not sixteen, and a caller who has only the float32 formats
// either quadruples their vertex bandwidth or cannot use the vertex path at
// all.
//
// # The conversions, stated
//
// They are stated rather than left to a backend, for the reason the fill rule
// is: the CPU rasterizer is the oracle, so a conversion with two defensible
// readings would make every comparison against it a tolerance.
//
//	unorm8:   v / 255              snorm8:   max(v / 127, -1)
//	unorm16:  v / 65535            snorm16:  max(v / 32767, -1)
//
// The signed forms clamp at -1 because two's complement has one more negative
// value than positive: -128/127 is below -1, and every target defines the
// result as -1 rather than letting it through. Getting that wrong is a normal
// that is slightly too long on exactly one input value, which is invisible.
//
// # Why two and four components and not one or three
//
// A three-component normalized attribute is not portable: Metal has
// UChar3Normalized and D3D12 does not, so a caller who used one would find the
// same declaration legal on one backend and refused on another. Two and four
// are universal, and four is what a packed normal or colour uses anyway.
type AttrFormat uint8

const (
	AttrInvalid AttrFormat = iota
	AttrFloat32
	AttrFloat32x2
	AttrFloat32x3
	AttrFloat32x4

	// The normalized integer formats. Each converts to [0,1] or [-1,1] on
	// fetch, by the rules stated above.
	AttrUnorm8x2
	AttrUnorm8x4
	AttrSnorm8x2
	AttrSnorm8x4
	AttrUnorm16x2
	AttrUnorm16x4
	AttrSnorm16x2
	AttrSnorm16x4
)

// attrInfo is one format's shape: how many components, how many bytes each, and
// how the bytes become floats.
type attrInfo struct {
	name       string
	components int
	bytes      int  // per component
	signed     bool // two's complement rather than unsigned
	normalized bool
}

var attrTable = map[AttrFormat]attrInfo{
	AttrFloat32:   {"float32", 1, 4, true, false},
	AttrFloat32x2: {"float32x2", 2, 4, true, false},
	AttrFloat32x3: {"float32x3", 3, 4, true, false},
	AttrFloat32x4: {"float32x4", 4, 4, true, false},

	AttrUnorm8x2:  {"unorm8x2", 2, 1, false, true},
	AttrUnorm8x4:  {"unorm8x4", 4, 1, false, true},
	AttrSnorm8x2:  {"snorm8x2", 2, 1, true, true},
	AttrSnorm8x4:  {"snorm8x4", 4, 1, true, true},
	AttrUnorm16x2: {"unorm16x2", 2, 2, false, true},
	AttrUnorm16x4: {"unorm16x4", 4, 2, false, true},
	AttrSnorm16x2: {"snorm16x2", 2, 2, true, true},
	AttrSnorm16x4: {"snorm16x4", 4, 2, true, true},
}

// Components is how many float32 values the format delivers to a stage, and 0
// if it delivers none — which is what makes an unset AttrFormat a refusal
// rather than a single-component fetch.
//
// It is the count the *stage* sees, not the count in memory, and for every
// format here the two are equal: a normalized format converts each component
// and adds none.
func (f AttrFormat) Components() int { return attrTable[f].components }

// Size is the format's size in bytes, which for a normalized format is a
// quarter or a half of what the stage receives.
func (f AttrFormat) Size() int {
	i := attrTable[f]
	return i.components * i.bytes
}

// Normalized reports whether the format converts on fetch.
func (f AttrFormat) Normalized() bool { return attrTable[f].normalized }

func (f AttrFormat) String() string {
	if n := attrTable[f].name; n != "" {
		return n
	}
	return "an unset attribute format"
}

// StepMode is what advances an attribute's index.
type StepMode uint8

const (
	// StepVertex advances per vertex, which is per-vertex geometry.
	StepVertex StepMode = iota

	// StepInstance advances per instance, which is how a per-object transform
	// reaches a stage without a uniform.
	StepInstance
)

func (m StepMode) String() string {
	if m == StepInstance {
		return "per instance"
	}
	return "per vertex"
}

// VertexAttribute is one attribute inside one buffer.
//
// Location is the stage's attribute index, not the parameter position: the
// receiver and any uniforms are interleaved with the attributes in the authored
// signature, so the two numbers differ as soon as a stage takes a uniform.
type VertexAttribute struct {
	Location int
	Format   AttrFormat
	Offset   int
}

// VertexBufferLayout is one bound buffer and what is packed inside it.
//
// Stride is the distance between consecutive elements, which is not the sum of
// the attribute sizes: a caller may pad for alignment, and a buffer that feeds
// two stages carries attributes neither one alone reads.
type VertexBufferLayout struct {
	Stride     int
	StepMode   StepMode
	Attributes []VertexAttribute
}

// checkVertexLayout validates a pipeline's declared layout against its vertex
// stage, and each check is here because its absence is silent.
//
// The two that matter most are the ones a reader would not think to write. A
// format whose component count differs from the parameter's fetches the wrong
// number of floats, so the geometry is subtly deformed rather than absent. And
// locations that are not dense let an attribute read another's data: the stage
// indexes its attributes densely, so a gap shifts every one after it.
func checkVertexLayout(label string, dev string, limit int, vs *Stage, bufs []VertexBufferLayout) error {
	// The ceiling is the *device's*, and it names the device.
	//
	// It used to be mslabi.StageVertexBufferLimit on every device, because on
	// Metal a vertex stage's uniforms and its vertex buffers share one index
	// space and the two are separated by that constant. That is a real ceiling
	// on Metal and no ceiling at all on the CPU oracle, so a caller was refused
	// by one backend's ABI wherever they ran -- which is
	// specs/000-decisions.md's layering rule 3 with the constant standing in
	// for the type.
	//
	// Refused here, where a caller can act on it, rather than at a draw where a
	// uniform would land on top of a vertex buffer and the stage would read
	// geometry as a transform.
	if limit > 0 && len(bufs) > limit {
		return fmt.Errorf("%w: NewRenderPipeline %q: %d vertex buffers and %q reports a "+
			"limit of %d", ErrUnsupported, label, len(bufs), dev, limit)
	}
	declared := map[int]VertexAttribute{}
	for i, b := range bufs {
		if b.Stride <= 0 {
			return fmt.Errorf("accel: NewRenderPipeline %q: vertex buffer %d has a stride "+
				"of %d, and a stride is the distance to the next element", label, i, b.Stride)
		}
		if len(b.Attributes) == 0 {
			return fmt.Errorf("accel: NewRenderPipeline %q: vertex buffer %d declares no "+
				"attributes, so nothing is fetched from it", label, i)
		}
		for _, a := range b.Attributes {
			if a.Format.Components() == 0 {
				return fmt.Errorf("accel: NewRenderPipeline %q: vertex buffer %d attribute "+
					"at location %d has %v", label, i, a.Location, a.Format)
			}
			if a.Offset < 0 || a.Offset+a.Format.Size() > b.Stride {
				return fmt.Errorf("accel: NewRenderPipeline %q: vertex buffer %d attribute "+
					"at location %d is %v at offset %d, which ends at %d in a stride of %d",
					label, i, a.Location, a.Format, a.Offset,
					a.Offset+a.Format.Size(), b.Stride)
			}
			if prev, dup := declared[a.Location]; dup {
				return fmt.Errorf("accel: NewRenderPipeline %q: location %d is declared "+
					"twice, as %v and as %v; one location is one of the stage's attributes",
					label, a.Location, prev.Format, a.Format)
			}
			declared[a.Location] = a
		}
	}

	if len(declared) != len(vs.Attributes) {
		return fmt.Errorf("accel: NewRenderPipeline %q: the layout declares %d attributes "+
			"and %s reads %d; a stage indexes its attributes densely, so a missing one "+
			"shifts every attribute after it", label, len(declared), vs.Name,
			len(vs.Attributes))
	}
	for _, want := range vs.Attributes {
		got, ok := declared[want.Index]
		if !ok {
			return fmt.Errorf("accel: NewRenderPipeline %q: %s reads attribute %q at "+
				"location %d and the layout declares no attribute there", label,
				vs.Name, want.Name, want.Index)
		}
		if got.Format.Components() != want.Components {
			return fmt.Errorf("accel: NewRenderPipeline %q: attribute %q at location %d is "+
				"[%d]float32 in %s and %v in the layout; a fetch of the wrong width "+
				"deforms the geometry rather than losing it", label, want.Name,
				want.Index, want.Components, vs.Name, got.Format)
		}
	}
	return nil
}
