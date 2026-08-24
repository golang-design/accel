// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"fmt"

	"golang.design/x/accel/internal/mslabi"
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
// Only the float32 vector widths exist, because a stage's attribute parameter
// is [N]float32 and nothing else can receive a fetch. Normalized integer
// formats convert on fetch, which is a conversion the CPU rasterizer would have
// to match bit for bit to stay an oracle; they arrive with a spec that states
// the rounding.
type AttrFormat uint8

const (
	AttrInvalid AttrFormat = iota
	AttrFloat32
	AttrFloat32x2
	AttrFloat32x3
	AttrFloat32x4
)

// Components is how many float32 values the format holds, and 0 if it holds
// none — which is what makes an unset AttrFormat a refusal rather than a
// single-component fetch.
func (f AttrFormat) Components() int {
	switch f {
	case AttrFloat32:
		return 1
	case AttrFloat32x2:
		return 2
	case AttrFloat32x3:
		return 3
	case AttrFloat32x4:
		return 4
	}
	return 0
}

// Size is the format's size in bytes.
func (f AttrFormat) Size() int { return f.Components() * 4 }

func (f AttrFormat) String() string {
	if c := f.Components(); c > 1 {
		return fmt.Sprintf("float32x%d", c)
	} else if c == 1 {
		return "float32"
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
func checkVertexLayout(label string, vs *Stage, bufs []VertexBufferLayout) error {
	// The ceiling exists because a Metal vertex stage's uniforms and its vertex
	// buffers share one buffer index space, and the two are separated by a
	// constant both the MSL emitter and the backend follow. Refused here, where
	// a caller can act on it, rather than at a draw where a uniform would land
	// on top of a vertex buffer and the stage would read geometry as a
	// transform.
	if len(bufs) > mslabi.StageVertexBufferLimit {
		return fmt.Errorf("accel: NewRenderPipeline %q: %d vertex buffers, and a stage's "+
			"uniforms begin at index %d on Metal, so %d is the ceiling", label,
			len(bufs), mslabi.StageVertexBufferLimit, mslabi.StageVertexBufferLimit)
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
