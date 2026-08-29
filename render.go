// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"fmt"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/mslabi"
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

	// Blend combines a fragment with what the attachment already holds. The
	// zero value does not blend, which is what makes a target that says nothing
	// about blending replace rather than accumulate.
	Blend BlendState
}

// BlendState is fixed-function attachment read-modify-write.
//
// It is fixed at pipeline creation because D3D12, Vulkan and Metal all take it
// as a compile-time input: changing it per draw would mean a backend that
// silently recompiles mid-frame, which is a notorious stutter source, or one
// that fails at a call the API said was legal.
//
// Colour and alpha have separate factors and operations because premultiplied
// compositing needs them to differ: the usual "over" blend is source alpha on
// colour and one on alpha, and a single pair cannot say that.
type BlendState struct {
	// Enabled is what separates blending from replacing. A zero BlendState with
	// Enabled true would multiply everything by zero, so the flag is not
	// inferred from the factors.
	Enabled bool

	SrcColor, DstColor BlendFactor
	ColorOp            BlendOp

	SrcAlpha, DstAlpha BlendFactor
	AlphaOp            BlendOp
}

// AlphaBlend is the "over" operator, which is what most callers mean by
// blending: source colour weighted by source alpha, over what is already there.
func AlphaBlend() BlendState {
	return BlendState{
		Enabled:  true,
		SrcColor: FactorSrcAlpha, DstColor: FactorOneMinusSrcAlpha, ColorOp: BlendAdd,
		SrcAlpha: FactorOne, DstAlpha: FactorOneMinusSrcAlpha, AlphaOp: BlendAdd,
	}
}

// BlendFactor scales one side of a blend.
type BlendFactor = driver.BlendFactor

// The blend factors. Named for what they scale by, not for where they appear:
// FactorSrcAlpha on the destination side is a legal and useful combination.
const (
	FactorZero             = driver.FactorZero
	FactorOne              = driver.FactorOne
	FactorSrcColor         = driver.FactorSrcColor
	FactorOneMinusSrcColor = driver.FactorOneMinusSrcColor
	FactorSrcAlpha         = driver.FactorSrcAlpha
	FactorOneMinusSrcAlpha = driver.FactorOneMinusSrcAlpha
	FactorDstColor         = driver.FactorDstColor
	FactorOneMinusDstColor = driver.FactorOneMinusDstColor
	FactorDstAlpha         = driver.FactorDstAlpha
	FactorOneMinusDstAlpha = driver.FactorOneMinusDstAlpha
)

// BlendOp combines the two scaled sides.
type BlendOp = driver.BlendOp

const (
	BlendAdd             = driver.BlendAdd
	BlendSubtract        = driver.BlendSubtract
	BlendReverseSubtract = driver.BlendReverseSubtract
	BlendMin             = driver.BlendMin
	BlendMax             = driver.BlendMax
)

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
	// View names the attachment directly, and Slot names it through a
	// rebindable binding point. Exactly one is set, the way exactly one of
	// [Binding]'s two is: a swapchain image cannot be named at record time,
	// because which one the frame gets is decided at acquire.
	//
	// A [TextureView] and not a [BufferView], because the view is what carries
	// the format the pass writes through and the subresource
	// specs/033-render-api.md section 3.3 compares. See
	// [RenderPassDescriptor].
	View TextureView
	Slot Slot

	Load  LoadOp
	Clear [4]float32
	Store StoreOp
}

// DepthAttachment is a pass's depth target.
type DepthAttachment struct {
	View TextureView
	Slot Slot

	Load  LoadOp
	Clear float32
	Store StoreOp
}

// RenderPassDescriptor describes one render pass.
//
// # Attachments are texture views
//
// They were buffer views, and this comment used to say the shape a caller
// writes would not change when textures landed. It did.
// specs/042-surface-completion.md section 5.2 found that eight of the largest
// findings in a review of this surface were consequences of that one decision,
// and specs/045-texture-attachments.md is the change: a buffer view carries a
// dtype, so ColorTargetState.Format reached no backend, sRGB had no owner, the
// V13 check was unimplementable, and 033 section 3.3's feedback rejection had
// no subresource to compare.
//
// A [TextureView] names one subresource of one texture and carries a concrete
// format, and it is the same type a shader-visible binding takes -- which is
// what makes the feedback rule compare one shape rather than two.
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

	// indexBuf and indexFmt are what an indexed draw reads.
	indexBuf BufferView
	indexFmt IndexFormat
	buffers  []BufferView

	// textures are the shader-visible bindings set so far, indexed by slot.
	//
	// One slice for both stages, where the uniforms are two. A uniform is a
	// *value* and each stage indexes its own parameters from zero, so one
	// shared slice would give a fragment stage the vertex stage's parameter 0.
	// A texture is a resource, and a slot names the resource rather than a
	// parameter of either stage: slot n is what a vertex stage's texture n and
	// a fragment stage's texture n both read, which is one binding on every
	// target -- Metal binds the same MTLTexture into both argument tables, and
	// a Vulkan descriptor set is visible to the stages that declare it.
	//
	// The consequence is worth stating: two stages that each declare a texture
	// at index 0 read the same view. That is the G-buffer case rather than a
	// restriction on it, and the alternative -- SetVertexTexture and
	// SetFragmentTexture -- makes a caller bind one resource twice for it.
	textures []TextureView

	draws  []drawCall
	failed bool
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

	// indexed and its buffer are set for an indexed draw. baseVertex applies to
	// the attribute fetch and not to the index the stage sees.
	indexed    bool
	indexBuf   BufferView
	indexFmt   IndexFormat
	baseVertex int

	// indirect and its argument buffer are set for an indirect draw. The
	// counts above are then the build-time maximum rather than the actual.
	indirect       bool
	indirectArgs   BufferView
	indirectAccess int

	// textures is the texture state standing when this draw was recorded,
	// indexed by slot.
	//
	// No access index beside it, where a vertex buffer has one. A buffer view
	// may name a transient or a slot, so build has to reach the access to learn
	// where the bytes ended up; a texture is neither, and its operand is its
	// own allocation whether or not the read was declared -- which it is not
	// for a texture no body fetches.
	textures []TextureView

	// texAccess indexes the node's access list for each stage's textures,
	// dense by the stage's own texture index. Two slices because each stage
	// numbers its textures from zero, the way the uniform slices are two.
	vertexTexAccess   []int
	fragmentTexAccess []int

	// vertexAccess and indexAccess index the node's access list, so build
	// lowers these views the way it lowers an attachment. -1 is unbound.
	vertexAccess []int
	indexAccess  int
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
	declare := func(v TextureView, slot Slot, load LoadOp, what string, elems int, st stage) {
		mode := AccessWrite
		if load == LoadKeep {
			mode = AccessReadWrite
		}
		op := "RenderPass " + label + " " + what
		var (
			a  access
			ok bool
		)
		if slot != 0 {
			a, ok = r.slotAccess(op, slot, 0, elems*F32.Size(), mode)
		} else {
			a, ok = r.declareTexture(op, v, mode)
		}
		if !ok {
			p.failed = true
			return
		}
		a.stage = st
		accesses = append(accesses, a)
	}
	// A clear value with a load op that never clears is silently discarded, and
	// specs/033-render-api.md section 3.1 makes that an error rather than a
	// convention: the caller wrote what they wanted the attachment to start as
	// and got whatever was there.
	//
	// A *zero* clear value is exempt, and there is no way around that: it is
	// also the zero value of the field, so "cleared to black" and "said nothing
	// about clearing" are the same bytes. The refusal fires where the caller's
	// intent is unambiguous, which is the only place it can.
	clearWithoutClear := func(what string, load LoadOp, set bool) {
		if !set || load == LoadClear {
			return
		}
		r.fail("RenderPass %q: %s sets a clear value and loads %v, so the value is "+
			"discarded; use LoadClear, or drop the value "+
			"(specs/033-render-api.md section 3.1)", label, what, load)
		p.failed = true
	}
	for i, c := range desc.Color {
		clearWithoutClear(fmt.Sprintf("colour %d", i), c.Load, c.Clear != [4]float32{})
	}
	if desc.Depth != nil {
		clearWithoutClear("depth", desc.Depth.Load, desc.Depth.Clear != 0)
	}

	for i, c := range desc.Color {
		what := fmt.Sprintf("colour %d", i)
		if (c.View.Texture == nil) == (c.Slot == 0) {
			r.fail("RenderPass %q: %s names %s; an attachment is one resource or one "+
				"slot", label, what, either(c.View.Texture != nil, c.Slot != 0))
			p.failed = true
			continue
		}
		// Colour is written by the blend and output stage, not by the fragment
		// shader: the shader returns a value and the attachment write happens
		// after it, which is the pair of stages Vulkan separates and the reason
		// a barrier against a colour attachment names the output stage.
		declare(c.View, c.Slot, c.Load, what, desc.Width*desc.Height*4, stageColourOutput)
	}
	if dep := desc.Depth; dep != nil {
		if (dep.View.Texture == nil) == (dep.Slot == 0) {
			r.fail("RenderPass %q: the depth attachment names %s; an attachment is one "+
				"resource or one slot", label,
				either(dep.View.Texture != nil, dep.Slot != 0))
			p.failed = true
		} else {
			// Both fragment-test stages, and that is not hedging. Where the
			// depth test runs depends on whether the fragment shader can
			// discard or write depth, which the pass does not know here; a
			// barrier naming only one of the two is wrong for half the
			// pipelines a caller can build, and naming both is what every
			// Vulkan depth barrier does.
			declare(dep.View, dep.Slot, dep.Load, "depth", desc.Width*desc.Height,
				stageEarlyDepth|stageLateDepth)
		}
	}

	p.id = r.node(NodeRenderPass, label, accesses, nil)
	r.state.nodes[p.id].pass = p
	return p
}

// either names what an attachment supplied, for an error that has to say which
// of "neither" and "both" happened.
func either(hasTexture, hasSlot bool) string {
	if hasTexture && hasSlot {
		return "both a resource and a slot"
	}
	return "no resource"
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
//
// The slot is checked rather than trusted, and that is worth a note because it
// was not: a negative slot skipped the grow loop and indexed the slice, which
// took the caller's process down from inside a recording call. Every other
// method on this type reports through the recorder, and a panic is the one
// diagnostic a caller cannot handle, cannot attribute to a slot, and cannot see
// alongside the rest of a build's errors.
func (p *RenderPass) SetVertexBuffer(slot int, v BufferView) {
	if slot < 0 {
		p.r.fail("RenderPass %q: SetVertexBuffer at slot %d; a slot is an index into "+
			"the pipeline's vertex layout and cannot be negative", p.desc.Label, slot)
		p.failed = true
		return
	}
	if slot >= mslabi.StageVertexBufferLimit {
		p.r.fail("RenderPass %q: SetVertexBuffer at slot %d; %d is the ceiling, because "+
			"a stage's uniforms begin there on Metal (specs/032-stage-abi.md section 2.2)",
			p.desc.Label, slot, mslabi.StageVertexBufferLimit)
		p.failed = true
		return
	}
	for len(p.vertexBuffers()) <= slot {
		p.buffers = append(p.buffers, BufferView{})
	}
	p.buffers[slot] = v
}

func (p *RenderPass) vertexBuffers() []BufferView { return p.buffers }

// SetTexture binds a texture a stage fetches from, at one slot.
//
// The view names the subresource, which is the whole of what a stage reads: the
// mip and the layer are the view's rather than the fetch's, so
// specs/032-stage-abi.md section 5's Fetch takes a coordinate and nothing else
// and specs/033-render-api.md section 3.3 has one shape to compare an
// attachment against.
//
// The slot is a stage's dense texture index, counting the textures its
// signature declares and not its parameters -- the rule [StageAttribute]
// follows. Both stages read the same slot space, for the reason
// [RenderPass.textures] gives.
//
// Its two refusals are [RenderPass.SetVertexBuffer]'s, and for that method's
// reason: a negative slot indexed a slice and took the caller's process down
// from inside a recording call, and a panic is the one diagnostic a caller
// cannot handle, cannot attribute to a slot, and cannot see beside the rest of
// a build's errors.
func (p *RenderPass) SetTexture(slot int, v TextureView) {
	if slot < 0 {
		p.r.fail("RenderPass %q: SetTexture at slot %d; a slot is a stage's texture "+
			"index and cannot be negative", p.desc.Label, slot)
		p.failed = true
		return
	}
	if slot >= mslabi.StageTextureLimit {
		p.r.fail("RenderPass %q: SetTexture at slot %d; %d is the ceiling, which is what "+
			"every target this project emits for guarantees a stage "+
			"(specs/032-stage-abi.md section 5)",
			p.desc.Label, slot, mslabi.StageTextureLimit)
		p.failed = true
		return
	}
	for len(p.textures) <= slot {
		p.textures = append(p.textures, TextureView{})
	}
	p.textures[slot] = v
}

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

// IndexFormat is the width of one entry in an index buffer.
type IndexFormat uint8

const (
	// Index16 is the common case and half the bandwidth. It caps a mesh at
	// 65536 vertices, which is why it is not the only option.
	Index16 IndexFormat = iota

	// Index32 addresses any mesh the device can hold.
	Index32
)

func (f IndexFormat) String() string {
	if f == Index32 {
		return "uint32 indices"
	}
	return "uint16 indices"
}

// size is the width of one index in bytes.
func (f IndexFormat) size() int {
	if f == Index32 {
		return 4
	}
	return 2
}

// SetIndexBuffer binds the index buffer subsequent indexed draws read.
//
// The format is given rather than inferred from the buffer's dtype: an index
// buffer is bytes, and a caller packing uint16 indices into a buffer of uint32
// elements is doing something ordinary rather than something wrong.
func (p *RenderPass) SetIndexBuffer(v BufferView, format IndexFormat) {
	if v.Buffer == nil {
		p.r.fail("RenderPass %q: SetIndexBuffer with no buffer", p.desc.Label)
		p.failed = true
		return
	}
	p.indexBuf, p.indexFmt = v, format
}

// DrawIndexed is one indexed draw.
//
// BaseVertex is added to each fetched index before the attribute fetch and is
// not added to the index the stage sees. specs/032-stage-abi.md section 2.1
// declines to expose a base-vertex built-in for that reason: backends disagree
// about whether theirs reports the pre-offset or post-offset value, so the ABI
// exposes only the one a caller can act on.
type DrawIndexed struct {
	IndexCount    int
	InstanceCount int
	FirstIndex    int
	BaseVertex    int
	FirstInstance int
}

// DrawIndexed records one indexed draw. See [RenderPass.Draw].
func (p *RenderPass) DrawIndexed(d DrawIndexed) {
	if p.failed {
		return
	}
	if p.indexBuf.Buffer == nil {
		p.r.fail("RenderPass %q: an indexed draw with no index buffer; call "+
			"SetIndexBuffer first", p.desc.Label)
		p.failed = true
		return
	}
	if d.IndexCount <= 0 {
		p.r.fail("RenderPass %q: an indexed draw of %d indices", p.desc.Label, d.IndexCount)
		p.failed = true
		return
	}
	if d.FirstIndex < 0 || d.BaseVertex < 0 || d.FirstInstance < 0 {
		p.r.fail("RenderPass %q: an indexed draw with a negative first index (%d), base "+
			"vertex (%d) or first instance (%d)", p.desc.Label, d.FirstIndex,
			d.BaseVertex, d.FirstInstance)
		p.failed = true
		return
	}
	p.record(drawCall{
		vertices: d.IndexCount, instances: max(d.InstanceCount, 1),
		first: d.FirstIndex, firstInst: d.FirstInstance,
		baseVertex: d.BaseVertex,
		indexed:    true, indexBuf: p.indexBuf, indexFmt: p.indexFmt,
	})
}

// DrawIndirect records a draw whose counts the device supplies.
//
// # Why a build-time maximum
//
// The same reason indirect dispatch has one: a device-written count sits
// awkwardly with an immutable graph, since without a bound there is nothing to
// validate at build and exceeding a backend's limit is undefined rather than a
// clean error. The node records a maximum, the device supplies the actual, and
// the backend clamps.
//
// **Every build mode clamps.** Correctness does not depend on a flag. What
// [Recorder.CollectRunStats] adds is being told that a clamp happened, which
// costs a readback — so a graph that did not ask is still protected, and what
// it gives up is knowing.
//
// args names four uint32 in a device buffer: vertex count, instance count,
// first vertex, first instance. That is the layout Vulkan, D3D12 and Metal all
// use, so a caller filling it from a compute kernel writes the same four values
// whatever the backend.
func (p *RenderPass) DrawIndirect(args BufferView, bound Draw) {
	if p.failed {
		return
	}
	if p.pipeline == nil {
		p.r.fail("RenderPass %q: an indirect draw with no pipeline; call SetPipeline "+
			"first", p.desc.Label)
		p.failed = true
		return
	}
	if args.Buffer == nil {
		p.r.fail("RenderPass %q: DrawIndirect with no argument buffer", p.desc.Label)
		p.failed = true
		return
	}
	if bound.VertexCount <= 0 {
		p.r.fail("RenderPass %q: an indirect draw with a maximum of %d vertices; the "+
			"maximum is what the device's count is clamped to, so it bounds the draw",
			p.desc.Label, bound.VertexCount)
		p.failed = true
		return
	}
	if need := 4 * 4; args.Count*args.DType.Size() < need {
		p.r.fail("RenderPass %q: the indirect argument buffer holds %d bytes and four "+
			"uint32 arguments need %d", p.desc.Label, args.Count*args.DType.Size(), need)
		p.failed = true
		return
	}
	d := drawCall{
		vertices: bound.VertexCount, instances: max(bound.InstanceCount, 1),
		first: bound.FirstVertex, firstInst: bound.FirstInstance,
		indirect: true, indirectArgs: args,
	}
	d.indirectAccess = p.declareRead(args, "indirect arguments", stageIndirectFetch)
	p.record(d)
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
	p.record(drawCall{
		vertices: d.VertexCount, instances: instances,
		first: d.FirstVertex, firstInst: d.FirstInstance,
	})
}

// record completes a draw with the pass state and appends it.
//
// The state is copied here rather than read at build, so a later SetPipeline or
// SetVertexUniform does not reach back and change a draw already recorded.
func (p *RenderPass) record(d drawCall) {
	d.pipeline = p.pipeline
	d.vertexBuf = append([]BufferView(nil), p.buffers...)
	d.vertexU = append([]any(nil), p.vertexUniforms...)
	d.fragmentU = append([]any(nil), p.fragmentUniforms...)
	d.textures = append([]TextureView(nil), p.textures...)

	// The buffers a draw reads become the pass's declared reads, which is
	// specs/033-render-api.md section 3's table. Declared here rather than at
	// RenderPass, because a draw is what names them and the node exists before
	// any draw is recorded: a pass that did not declare them would run
	// unordered against whatever wrote the vertices.
	// Only the slots this draw's pipeline actually fetches. A buffer bound at a
	// slot the layout does not read is not fetched by the lowering, so
	// declaring a read of it would order this node against whatever wrote it
	// for a fetch that does not happen -- moving barriers and stretching a
	// transient's live range on a binding nobody consumes.
	//
	// Bound-but-unfetched is legitimate rather than a mistake: a caller may
	// bind for the widest pipeline in a pass and draw with a narrower one, and
	// each draw copies the state standing at the time. So the answer is to
	// declare what is read rather than to refuse the binding.
	d.vertexAccess = make([]int, len(d.vertexBuf))
	declared := 0
	if d.pipeline != nil {
		declared = len(d.pipeline.desc.VertexBuffers)
	}
	for i, v := range d.vertexBuf {
		d.vertexAccess[i] = -1
		if v.Buffer != nil && i < declared {
			d.vertexAccess[i] = p.declareRead(v, fmt.Sprintf("vertex buffer %d", i), stageVertexInput)
		}
	}
	d.indexAccess = -1
	if d.indexed {
		d.indexAccess = p.declareRead(d.indexBuf, "index buffer", stageVertexInput)
	}
	p.declareTextureReads(&d)

	p.draws = append(p.draws, d)
}

// declareTextureReads adds one read per texture this draw's stages fetch.
//
// Only the slots a stage *declares*, for the reason only the vertex slots a
// layout reads are declared: a texture bound at a slot no stage names is not
// fetched, so ordering this node against whatever wrote it would move barriers
// and stretch a transient's live range for a read that does not happen. Binding
// for the widest pipeline in a pass and drawing with a narrower one is
// legitimate.
//
// And only a texture the body reads. StageTexture.Reads is inferred from the
// body rather than declared, and a texture nothing fetches is not a
// subresource this pass depends on -- the binding is still required, because
// the shader's argument exists either way, but the graph edge is not.
//
// The stage mask is the stage that declares it, so the barrier
// specs/045-texture-attachments.md section 3 draws -- colour output to a shader
// stage -- names the shader stage that actually reads. A slot both stages
// declare is one access carrying both bits, which is what declareRead's
// de-duplication already does for a view fetched in two roles.
func (p *RenderPass) declareTextureReads(d *drawCall) {
	if d.pipeline == nil {
		return
	}
	for _, s := range []struct {
		stage *Stage
		mask  stage
		what  string
	}{
		{d.pipeline.desc.Vertex, stageVertexShader, "vertex stage texture"},
		{d.pipeline.desc.Fragment, stageFragmentShader, "fragment stage texture"},
	} {
		if s.stage == nil {
			continue
		}
		for _, tx := range s.stage.Textures {
			if !tx.Reads || tx.Index < 0 || tx.Index >= len(d.textures) {
				continue
			}
			v := d.textures[tx.Index]
			if v.Texture == nil {
				continue
			}
			at := p.declareTextureRead(v, fmt.Sprintf("%s %d", s.what, tx.Index), s.mask)
			dst := &d.vertexTexAccess
			if s.mask == stageFragmentShader {
				dst = &d.fragmentTexAccess
			}
			for len(*dst) <= tx.Index {
				*dst = append(*dst, -1)
			}
			(*dst)[tx.Index] = at
		}
	}
}

// declareRead adds one read to the pass's node, once per distinct view, in the
// stage named.
//
// Once, because several draws sharing a vertex buffer are one read of it: a
// duplicate would inflate the hazard count a caller reads, and the builder
// would compute the same barrier twice.
//
// The stage is not part of what makes a read distinct. One range fetched in two
// roles -- an argument buffer a draw also fetches attributes from -- is still
// one read, and the mask is what carries both roles. Splitting on stage instead
// would restore exactly the duplicate this exists to prevent, and would shift
// the returned index, which build consumes positionally.
//
// It returns the index into the node's access list, which is how build turns
// the same view into an operand -- through the same path an attachment takes, so
// a slot or a transient works here for the reason it works there.
func (p *RenderPass) declareRead(v BufferView, what string, st stage) int {
	a, ok := p.r.declare("RenderPass "+p.desc.Label+" "+what, v, AccessRead)
	if !ok {
		p.failed = true
		return -1
	}
	return p.addRead(a, st)
}

// declareTextureRead is [RenderPass.declareRead] for a subresource.
//
// The same body over a different declaration, because what makes a read
// distinct and where it lands in the node's access list are properties of the
// access rather than of the resource kind. Two paths would let a texture read
// and a buffer read de-duplicate by different rules, and build consumes the
// returned index positionally either way.
func (p *RenderPass) declareTextureRead(v TextureView, what string, st stage) int {
	a, ok := p.r.declareTexture("RenderPass "+p.desc.Label+" "+what, v, AccessRead)
	if !ok {
		p.failed = true
		return -1
	}
	return p.addRead(a, st)
}

// addRead places one declared read in the node, once per distinct range.
func (p *RenderPass) addRead(a access, st stage) int {
	a.stage = st
	n := p.r.state.nodes[p.id]
	for i, have := range n.accesses {
		if have.sameRange(a) {
			n.accesses[i].stage |= st
			n.stage |= st
			p.r.state.nodes[p.id] = n
			return i
		}
	}
	n.accesses = append(n.accesses, a)
	n.stage |= st
	p.r.state.nodes[p.id] = n
	// Touched here for the same reason [Recorder.node] touches the accesses it
	// is given: this access arrives after the node exists, so nothing else
	// would put the buffer in the pass's live range.
	p.r.touch(p.id, a)
	return len(n.accesses) - 1
}

// Node is the graph node this pass records into, for a caller who wants to
// name it in a diagnostic.
func (p *RenderPass) Node() NodeID { return p.id }
