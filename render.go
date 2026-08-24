// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"fmt"

	"golang.design/x/accel/internal/driver"
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

	// VertexBuffers is the vertex input layout. A stage that reads no attribute
	// needs none; one that does needs every attribute declared exactly once.
	VertexBuffers []VertexBufferLayout

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

	// The layout last of the stage-record checks. A caller who swapped in the
	// wrong vertex stage fails both this and the varyings check, and the
	// varyings message names both stages where this one names only the vertex
	// stage -- so it is the more useful of the two to report first.
	if err := checkVertexLayout(label, desc.Vertex, desc.VertexBuffers); err != nil {
		return nil, err
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

// LoadOp and StoreOp are what happens to an attachment at the start and the end
// of a pass. They are [driver.LoadOp] and [driver.StoreOp]: a backend acts on
// the value and cannot import this package, so one definition lives where both
// sides reach it. Two definitions with nothing pinning them together is how
// LoadKeep and LoadDontCare -- which a backend cannot tell apart by their
// effect -- would swap silently.
type (
	LoadOp  = driver.LoadOp
	StoreOp = driver.StoreOp
)

const (
	LoadClear    = driver.LoadClear
	LoadKeep     = driver.LoadKeep
	LoadDontCare = driver.LoadDontCare

	StoreKeep    = driver.StoreKeep
	StoreDiscard = driver.StoreDiscard
)

// ColorAttachment is one colour target of a render pass.
type ColorAttachment struct {
	View  BufferView
	Load  LoadOp
	Clear [4]float32
	Store StoreOp
}

// DepthAttachment is a pass's depth target.
type DepthAttachment struct {
	View  BufferView
	Load  LoadOp
	Clear float32
	Store StoreOp
}

// RenderPassDescriptor describes one render pass.
//
// Attachments are buffer views rather than textures at this milestone, because
// specs/035-cpu-rasterizer.md's reference rasterizer writes float components and
// the texture path needs the format encode/decode 033 leaves to the backend.
// The shape a caller writes does not change when that lands.
type RenderPassDescriptor struct {
	Color []ColorAttachment
	Depth *DepthAttachment

	// Width and Height are the render area, validated against every attachment
	// at build.
	Width, Height int

	Label string
}

// RenderPass records draws into one graph node.
//
// One pass is one node and a draw is not, because the pass is the unit at which
// synchronisation is expressible: Vulkan cannot barrier inside a render pass in
// the general case, tile-based hardware physically cannot — attachment contents
// live in tile memory until the pass ends — and Metal's encoder has the same
// shape. Draw granularity would promise an ordering the hardware cannot provide.
//
// Two rules follow, and both are caller-visible. Draws execute in recorded order
// and the builder never reorders them, because blending is order dependent. And
// the builder inserts no barriers inside a pass, because per-pixel ordering
// between draws is the ROP's job.
type RenderPass struct {
	r    *Recorder
	desc RenderPassDescriptor
	id   NodeID

	pipeline *RenderPipeline

	// vertexUniforms and fragmentUniforms are the by-value parameters set so
	// far, one slice per stage because each stage indexes its own from zero.
	vertexUniforms   []any
	fragmentUniforms []any
	buffers          []BufferView
	draws            []drawCall
	failed           bool
}

// drawCall is one recorded draw.
type drawCall struct {
	pipeline  *RenderPipeline
	vertices  int
	instances int
	first     int
	firstInst int
	vertexBuf []BufferView
	vertexU   []any
	fragmentU []any
}

// Draw is one non-instanced or instanced draw's counts.
//
// Instancing is the instance count and not a separate call, which is why the
// non-instanced case is this one with a count of one: a caller never picks
// between two entry points for the same drawing.
type Draw struct {
	VertexCount   int
	InstanceCount int
	FirstVertex   int
	FirstInstance int
}

// RenderPass begins recording a render pass. The pass becomes one node.
func (r *Recorder) RenderPass(desc RenderPassDescriptor) *RenderPass {
	label := desc.Label
	if label == "" {
		label = "render pass"
	}
	p := &RenderPass{r: r, desc: desc}

	if len(desc.Color) == 0 {
		r.fail("RenderPass %q: no colour attachments", label)
		p.failed = true
	}
	if desc.Width <= 0 || desc.Height <= 0 {
		r.fail("RenderPass %q: the render area is %dx%d, and an area has positive extents",
			label, desc.Width, desc.Height)
		p.failed = true
	}

	// What the pass declares, which is what the builder infers edges from. An
	// attachment loaded Keep is a read as well as a write; one loaded Clear or
	// DontCare is a write only, and DontCare is what makes the
	// read-after-write edge to its previous writer disappear.
	var accesses []access
	declare := func(v BufferView, load LoadOp, what string) {
		mode := AccessWrite
		if load == LoadKeep {
			mode = AccessReadWrite
		}
		a, ok := r.declare("RenderPass "+label+" "+what, v, mode)
		if !ok {
			p.failed = true
			return
		}
		accesses = append(accesses, a)
	}
	for i, c := range desc.Color {
		if c.View.Buffer == nil {
			r.fail("RenderPass %q: colour attachment %d names no resource", label, i)
			p.failed = true
			continue
		}
		declare(c.View, c.Load, fmt.Sprintf("colour %d", i))
	}
	if desc.Depth != nil {
		declare(desc.Depth.View, desc.Depth.Load, "depth")
	}

	p.id = r.node(NodeRenderPass, label, accesses, nil)
	r.state.nodes[p.id].pass = p
	return p
}

// SetPipeline selects the pipeline subsequent draws use.
func (p *RenderPass) SetPipeline(pipe *RenderPipeline) {
	if pipe == nil {
		p.r.fail("RenderPass %q: SetPipeline with no pipeline", p.desc.Label)
		p.failed = true
		return
	}
	p.pipeline = pipe
}

// SetVertexBuffer binds a vertex buffer at one slot.
func (p *RenderPass) SetVertexBuffer(slot int, v BufferView) {
	for len(p.vertexBuffers()) <= slot {
		p.buffers = append(p.buffers, BufferView{})
	}
	p.buffers[slot] = v
}

func (p *RenderPass) vertexBuffers() []BufferView { return p.buffers }

// SetVertexUniform and SetFragmentUniform supply one by-value parameter of the
// stage named, for every draw recorded after the call.
//
// # Why two calls and not one
//
// The two stages are compiled independently, so each indexes its own uniform
// space from zero: a vertex stage's parameter 0 and a fragment stage's
// parameter 0 are different parameters with the same index. One shared slice
// cannot hold both, and a single call taking an index would have to guess which
// stage a value was for. specs/033-render-api.md deviation 1 is what happened
// when it did not: values were placed in the order the caller wrote them, so
// two passed out of order bound to each other's parameters.
//
// # Why pass state and not a draw argument
//
// It matches SetPipeline and SetVertexBuffer, which are the calls a reader
// already knows, and a value shared by several draws is written once. A draw
// captures whatever is set when it is recorded, so a later call does not reach
// back and change an earlier draw.
func (p *RenderPass) SetVertexUniform(index int, v any) {
	p.setUniform(&p.vertexUniforms, "SetVertexUniform", index, v)
}

// SetFragmentUniform supplies one by-value parameter of the fragment stage. See
// [RenderPass.SetVertexUniform].
func (p *RenderPass) SetFragmentUniform(index int, v any) {
	p.setUniform(&p.fragmentUniforms, "SetFragmentUniform", index, v)
}

func (p *RenderPass) setUniform(dst *[]any, what string, index int, v any) {
	if index < 0 {
		p.r.fail("RenderPass %q: %s at index %d", p.desc.Label, what, index)
		p.failed = true
		return
	}
	if v == nil {
		p.r.fail("RenderPass %q: %s at index %d has a nil value", p.desc.Label, what, index)
		p.failed = true
		return
	}
	for len(*dst) <= index {
		*dst = append(*dst, nil)
	}
	(*dst)[index] = v
}

// Draw records one draw.
//
// It executes in the order recorded, and the builder never reorders it: blending
// is order dependent, and so is any reasoning about overdraw.
func (p *RenderPass) Draw(d Draw) {
	if p.failed {
		return
	}
	if p.pipeline == nil {
		p.r.fail("RenderPass %q: a draw with no pipeline; call SetPipeline first",
			p.desc.Label)
		p.failed = true
		return
	}
	if d.VertexCount <= 0 {
		p.r.fail("RenderPass %q: a draw of %d vertices", p.desc.Label, d.VertexCount)
		p.failed = true
		return
	}
	instances := d.InstanceCount
	if instances == 0 {
		// The non-instanced case is the instanced case with a count of one, so
		// an omitted count is one rather than nothing drawn. Go cannot tell an
		// omitted field from an explicit zero, so there is no way to draw zero
		// instances and no need for one: a caller with nothing to draw records
		// no draw.
		instances = 1
	}
	p.draws = append(p.draws, drawCall{
		pipeline: p.pipeline, vertices: d.VertexCount, instances: instances,
		first: d.FirstVertex, firstInst: d.FirstInstance,
		vertexBuf: append([]BufferView(nil), p.buffers...),
		vertexU:   append([]any(nil), p.vertexUniforms...),
		fragmentU: append([]any(nil), p.fragmentUniforms...),
	})
}

// Node is the graph node this pass records into, for a caller who wants to
// name it in a diagnostic.
func (p *RenderPass) Node() NodeID { return p.id }
