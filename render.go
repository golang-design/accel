// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"fmt"

	"golang.design/x/accel/internal/kernel"
)

// The render API of specs/033-render-api.md: what consumes a graphics stage.
//
// Everything here is recorded, never executed at the call. A render pass is a
// node in the graph like any other, and the reason is decision 1: every backend
// records a render pass into a command buffer anyway, so recording costs
// graphics nothing.

// Topology is how vertices group into primitives.
type Topology uint8

const (
	TriangleList Topology = iota
	TriangleStrip

	// LineList, LineStrip and PointList are named so a caller's value maps onto
	// the full enumeration, and are refused. specs/035-cpu-rasterizer.md
	// section 10 leaves their rules open: lines have a diamond-exit rule on some
	// backends and a Bresenham-ish rule on others, and points have a size and a
	// centre convention. Guessing one would put an unstated rule in the oracle
	// every backend is checked against.
	LineList
	LineStrip
	PointList
)

func (t Topology) String() string {
	switch t {
	case TriangleList:
		return "triangle list"
	case TriangleStrip:
		return "triangle strip"
	case LineList:
		return "line list"
	case LineStrip:
		return "line strip"
	case PointList:
		return "point list"
	}
	return fmt.Sprintf("Topology(%d)", uint8(t))
}

// FrontFace is which winding is front-facing.
//
// There is no zero value meaning "the backend's default". Metal's default
// disagrees with GL's, and getting it backwards keeps back faces instead of
// front faces: the silhouette stays right while every per-pixel attribute comes
// from the wrong surface, so it reads as a shading bug. A default would make
// that the easiest thing to write. See docs/conventions.md.
type FrontFace uint8

const (
	// CounterClockwise is front-facing when the vertices wind counter-clockwise
	// in clip space, which is what a caller reasons about. The viewport's y flip
	// reverses the sign of the window-space area, so the implementation's test
	// is not the caller's convention.
	CounterClockwise FrontFace = iota
	Clockwise
)

// CullMode is which faces are discarded.
type CullMode uint8

const (
	CullNone CullMode = iota
	CullFront
	CullBack
)

// CompareFunc is a depth or stencil comparison.
type CompareFunc uint8

const (
	CompareNever CompareFunc = iota
	CompareLess
	CompareEqual
	CompareLessEqual
	CompareGreater
	CompareNotEqual
	CompareGreaterEqual
	CompareAlways
)

// PrimitiveState is the fixed-function state before the fragment stage.
type PrimitiveState struct {
	Topology  Topology
	FrontFace FrontFace
	Cull      CullMode
}

// DepthStencilState is the depth test and write.
//
// Test and Write are separate because read-only depth — test on, write off — is
// a real configuration: it is how a second pass shades exactly the surfaces the
// geometry pass kept.
type DepthStencilState struct {
	Format  Format
	Test    bool
	Write   bool
	Compare CompareFunc
}

// ColorTargetState is one colour attachment's compiled state.
//
// Blend and the write mask are here, on the pipeline, rather than on the
// attachment: a pass holds one set of attachments and many draws with different
// pipelines, so putting them on the attachment would mean rewriting the
// attachment object before every draw. Every backend agrees — Vulkan puts them
// in an array on the pipeline, Metal on the render pipeline's colour attachment
// descriptor, D3D12 in the blend description's render-target array.
type ColorTargetState struct {
	Format Format
	Mask   ColorWriteMask
}

// ColorWriteMask is which channels of an attachment a fragment may write. The
// zero value writes every channel, because a target nobody configured should be
// written rather than silently dropped.
type ColorWriteMask uint8

const (
	WriteRed ColorWriteMask = 1 << iota
	WriteGreen
	WriteBlue
	WriteAlpha
)

// WriteAll is every channel, and is what a zero ColorWriteMask means.
const WriteAll = WriteRed | WriteGreen | WriteBlue | WriteAlpha

func (m ColorWriteMask) resolved() ColorWriteMask {
	if m == 0 {
		return WriteAll
	}
	return m
}

// RenderPipelineDescriptor describes a render pipeline to create.
//
// A pipeline is compiled once, outside the frame loop, and referenced by nodes:
// creating one is expensive on every backend, because it compiles shaders and
// specialises fixed-function state.
type RenderPipelineDescriptor struct {
	Vertex   *Stage
	Fragment *Stage

	Primitive    PrimitiveState
	DepthStencil *DepthStencilState
	Targets      []ColorTargetState

	Label string
}

// Stage is a compiled graphics stage.
//
// An alias for the generated record, the same way [Kernel] is. A caller does not
// construct one: `go generate` emits one per stage and the address is the whole
// of the interface.
type Stage = kernel.Stage

// RenderPipeline is a compiled render pipeline.
type RenderPipeline struct {
	_ noCopy

	dev   *Device
	desc  RenderPipelineDescriptor
	label string
	state resourceState
}

// NewRenderPipeline compiles a render pipeline.
//
// Everything checkable is checked here rather than at draw: the descriptor
// against the stages' generated records, and both against the device's limits.
// A pipeline that survives this is one a pass can only get wrong by pairing it
// with mismatched attachments, which graph build catches.
func (d *Device) NewRenderPipeline(desc RenderPipelineDescriptor) (*RenderPipeline, error) {
	if err := d.state.checkOpen("NewRenderPipeline"); err != nil {
		return nil, err
	}
	label := desc.Label
	if label == "" {
		label = "render pipeline"
	}
	if desc.Vertex == nil {
		return nil, fmt.Errorf("accel: NewRenderPipeline %q: no vertex stage", label)
	}
	if desc.Fragment == nil {
		return nil, fmt.Errorf("accel: NewRenderPipeline %q: no fragment stage", label)
	}
	if desc.Vertex.Kind != kernel.StageVertex {
		return nil, fmt.Errorf("accel: NewRenderPipeline %q: %s is a %v, and the Vertex "+
			"field takes a vertex stage", label, desc.Vertex.Name, desc.Vertex.Kind)
	}
	if desc.Fragment.Kind != kernel.StageFragment {
		return nil, fmt.Errorf("accel: NewRenderPipeline %q: %s is a %v, and the Fragment "+
			"field takes a fragment stage", label, desc.Fragment.Name, desc.Fragment.Kind)
	}

	switch desc.Primitive.Topology {
	case TriangleList, TriangleStrip:
	default:
		return nil, fmt.Errorf("accel: NewRenderPipeline %q: %v is not rasterizable: "+
			"specs/035-cpu-rasterizer.md section 10 leaves its fill rule unstated, and "+
			"guessing one would put an unstated rule in the oracle every backend is "+
			"checked against", label, desc.Primitive.Topology)
	}

	// The count check that matters, and the reason it is here: a field that
	// lands in no attachment writes nowhere, and an attachment with no field
	// holds undefined contents. Neither fails at draw.
	if len(desc.Targets) != len(desc.Fragment.Outputs) {
		return nil, fmt.Errorf("accel: NewRenderPipeline %q: %d colour targets and %s "+
			"writes %d attachments; one struct field per attachment is how MRT is "+
			"expressed, so the two counts are the same number",
			label, len(desc.Targets), desc.Fragment.Name, len(desc.Fragment.Outputs))
	}
	if len(desc.Targets) == 0 {
		return nil, fmt.Errorf("accel: NewRenderPipeline %q: no colour targets", label)
	}
	if max := d.info.Limits.MaxColorAttachments; max > 0 && len(desc.Targets) > max {
		return nil, fmt.Errorf("%w: NewRenderPipeline %q: %d colour targets and %q reports "+
			"a limit of %d", ErrUnsupported, label, len(desc.Targets), d.info.Name, max)
	}
	for i, t := range desc.Targets {
		info := d.FormatInfo(t.Format)
		if info.BytesPerPixel == 0 || info.IsDepth {
			return nil, fmt.Errorf("accel: NewRenderPipeline %q: colour target %d is %v, "+
				"which is not a colour format", label, i, t.Format)
		}
	}
	if ds := desc.DepthStencil; ds != nil {
		info := d.FormatInfo(ds.Format)
		if !info.IsDepth {
			return nil, fmt.Errorf("accel: NewRenderPipeline %q: the depth format is %v, "+
				"which is not a depth format", label, ds.Format)
		}
	}

	// The varyings the two stages exchange must be the same type. Checked by
	// name here because the generated records carry the name the compiler
	// resolved; the compiler itself checked identity, which is the stronger
	// half — two structurally identical structs are not interchangeable.
	if desc.Vertex.Varyings != desc.Fragment.Varyings {
		return nil, fmt.Errorf("accel: NewRenderPipeline %q: %s returns %s varyings and "+
			"%s takes %s; the two stages exchange one type", label,
			desc.Vertex.Name, desc.Vertex.Varyings, desc.Fragment.Name, desc.Fragment.Varyings)
	}

	p := &RenderPipeline{dev: d, desc: desc, label: label}
	p.state.init(label)
	d.countPipelines(1)
	return p, nil
}

// Label is the pipeline's label, which appears in every error about it.
func (p *RenderPipeline) Label() string { return p.label }

// Close releases the pipeline.
func (p *RenderPipeline) Close() error {
	if !p.state.beginClose() {
		return nil
	}
	p.dev.countPipelines(-1)
	return nil
}
