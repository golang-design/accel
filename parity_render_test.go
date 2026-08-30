// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"encoding/binary"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/parity"
	"golang.design/x/accel/internal/testkernels"
)

// specs/062-backend-parity.md sections 6.3, 6.4 and 6.5: the fixed-function
// state, enumerated.
//
// Every member of every render enumeration is a place the two backends could
// disagree and still produce a picture. A compare function mapped to the wrong
// Metal constant, a blend factor off by one in an enumeration, a write mask
// numbered blue-first rather than red-first: each of those draws something, and
// what it draws is wrong in a way no single hand-written case would find. The
// enumeration is the test.

const parityW, parityH = 4, 4

// parityClear is the background every case starts from.
//
// Deliberately not black. A case whose draw is culled, depth-failed or fully
// masked out leaves the clear behind, and an all-zero result is one the matrix
// refuses as degenerate -- correctly, since two blank targets agree perfectly.
// A visible clear makes "nothing was drawn" a comparable result rather than an
// absent one.
var parityClear = [4]float32{0.125, 0.25, 0.375, 1}

// fullViewport is a triangle covering the whole target, at the given depth.
func fullViewport(z float32) []float32 {
	return []float32{-1, -1, z, 3, -1, z, -1, 3, z}
}

// parityDraw is one draw inside a parity pass.
type parityDraw struct {
	pipe  func(t *testing.T, d *accel.Device) *accel.RenderPipeline
	verts []float32
	tint  [4]float32
	count int
}

// parityPass is the shape every case in this file has: one RGBA32Float colour
// target, an optional depth attachment, and a list of tinted draws.
type parityPass struct {
	label  string
	load   accel.LoadOp
	prefil bool // fill the target before the pass, which is what LoadKeep keeps
	depth  bool
	draws  []parityDraw
}

// run records the pass and reads the colour target back.
func (p parityPass) run(t *testing.T, d *accel.Device) []byte {
	t.Helper()
	target := colourTarget(t, d, p.label, parityW, parityH)
	if p.prefil {
		prior := make([]float32, parityW*parityH*4)
		for i := range prior {
			prior[i] = float32(i%9) / 8
		}
		fillTarget(t, d, target, prior)
	}

	// The clear value goes only with LoadClear: a pass that sets one and does
	// not clear is refused, because the value would be silently discarded.
	colour := accel.ColorAttachment{View: view(t, target), Load: p.load}
	if p.load == accel.LoadClear {
		colour.Clear = parityClear
	}
	desc := accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{colour},
		Width: parityW, Height: parityH, Label: p.label,
	}
	if p.depth {
		desc.Depth = &accel.DepthAttachment{
			View: view(t, depthTarget(t, d, p.label+" depth", parityW, parityH)),
			Load: accel.LoadClear, Clear: 1, Store: accel.StoreKeep,
		}
	}

	r := d.NewRecorder()
	pass := r.RenderPass(desc)
	for i, dr := range p.draws {
		vb := newBuffer(t, d, "verts", len(dr.verts),
			accel.BufferStorage|accel.BufferCopyDst)
		if err := d.Queue().WriteBuffer(vb, 0, dr.verts); err != nil {
			t.Fatalf("%s draw %d: write verts: %v", p.label, i, err)
		}
		pass.SetPipeline(dr.pipe(t, d))
		pass.SetVertexBuffer(0, whole(t, vb))
		pass.SetVertexUniform(0, testkernels.StageTransform{Scale: 1})
		pass.SetFragmentUniform(0, testkernels.StageTint{Colour: dr.tint})
		pass.Draw(accel.Draw{VertexCount: dr.count})
	}
	submitOne(t, d, r)
	return readTargetBytes(t, d, target)
}

// tintedPipeline compiles ScaledVS/TintedFS with the given fixed-function
// state. Every case in this file differs only in that state, which is what
// makes the comparison about the state rather than about the shader.
func tintedPipeline(desc accel.RenderPipelineDescriptor) func(*testing.T, *accel.Device) *accel.RenderPipeline {
	desc.Vertex = &testkernels.ScaledVSStage
	desc.Fragment = &testkernels.TintedFSStage
	desc.VertexBuffers = []accel.VertexBufferLayout{{
		Stride: 12,
		Attributes: []accel.VertexAttribute{
			{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
		},
	}}
	if desc.Targets == nil {
		desc.Targets = []accel.ColorTargetState{{Format: accel.RGBA32Float}}
	}
	return func(t *testing.T, d *accel.Device) *accel.RenderPipeline {
		t.Helper()
		p, err := d.NewRenderPipeline(desc)
		if err != nil {
			t.Fatalf("pipeline %q: %v", desc.Label, err)
		}
		t.Cleanup(func() { _ = p.Close() })
		return p
	}
}

func renderStateParityCases() []parityCase {
	var cases []parityCase
	cases = append(cases, compareFuncParityCases()...)
	cases = append(cases, rasterStateParityCases()...)
	cases = append(cases, blendParityCases()...)
	cases = append(cases, writeMaskParityCases()...)
	cases = append(cases, attachmentOpParityCases()...)
	cases = append(cases, indexFormatParityCases()...)
	cases = append(cases, attrFormatParityCases()...)
	cases = append(cases, normalizedAttrParityCases()...)
	return cases
}

// compareFuncParityCases seeds the depth buffer at one depth and then draws at
// the same depth through each comparison.
//
// The same depth for both draws is what makes all eight distinguishable: at
// equal depth, Never, Less, Greater and NotEqual reject and Equal, LessEqual,
// GreaterEqual and Always accept, so the eight split four and four and no two
// members can be swapped without the result changing. A seed at a different
// depth would collapse several of them onto the same answer.
func compareFuncParityCases() []parityCase {
	seed := tintedPipeline(accel.RenderPipelineDescriptor{
		Label: "depth seed",
		DepthStencil: &accel.DepthStencilState{
			Format: accel.Depth32Float, Test: true, Write: true,
			Compare: accel.CompareAlways,
		},
	})

	funcs := []struct {
		name string
		fn   accel.CompareFunc
	}{
		{"CompareNever", accel.CompareNever},
		{"CompareLess", accel.CompareLess},
		{"CompareEqual", accel.CompareEqual},
		{"CompareLessEqual", accel.CompareLessEqual},
		{"CompareGreater", accel.CompareGreater},
		{"CompareNotEqual", accel.CompareNotEqual},
		{"CompareGreaterEqual", accel.CompareGreaterEqual},
		{"CompareAlways", accel.CompareAlways},
	}
	cases := make([]parityCase, 0, len(funcs))
	for _, f := range funcs {
		under := tintedPipeline(accel.RenderPipelineDescriptor{
			Label: f.name,
			DepthStencil: &accel.DepthStencilState{
				Format: accel.Depth32Float, Test: true, Write: true, Compare: f.fn,
			},
		})
		cases = append(cases, parityCase{
			name:   "a depth test through " + f.name,
			covers: parity.Covers{"CompareFunc." + f.name},
			run: parityPass{
				label: f.name, load: accel.LoadClear, depth: true,
				draws: []parityDraw{
					{pipe: seed, verts: fullViewport(0.5), tint: [4]float32{1, 0, 0, 1}, count: 3},
					{pipe: under, verts: fullViewport(0.5), tint: [4]float32{0, 1, 0, 1}, count: 3},
				},
			}.run,
		})
	}
	return cases
}

// rasterStateParityCases cover the primitive state: the two windings, the three
// cull modes, and the two admissible topologies.
//
// The geometry winds counter-clockwise in clip space, so which faces survive is
// a function of both fields rather than of one: CullBack keeps it under
// CounterClockwise and discards it under Clockwise, which is the pairing a
// backend that mapped one of the two enumerations backwards would get wrong
// while still producing a picture.
func rasterStateParityCases() []parityCase {
	tri := fullViewport(0.5)
	strip := []float32{-1, -1, 0.5, 1, -1, 0.5, -1, 1, 0.5, 1, 1, 0.5}

	prim := func(label string, t accel.Topology, f accel.FrontFace, c accel.CullMode) func(*testing.T, *accel.Device) *accel.RenderPipeline {
		return tintedPipeline(accel.RenderPipelineDescriptor{
			Label:     label,
			Primitive: accel.PrimitiveState{Topology: t, FrontFace: f, Cull: c},
		})
	}
	// A small un-culled triangle in one corner, drawn after the triangle under
	// test. Three of the five cases below are culled away and leave only the
	// clear, and two blank targets agree perfectly -- so the marker is what
	// separates "the state discarded the triangle on both backends" from "the
	// pass encoded nothing on either". The degeneracy check cannot tell those
	// apart on its own, because a clear is not all zero.
	marker := prim("marker", accel.TriangleList, accel.CounterClockwise, accel.CullNone)
	markerVerts := []float32{-1, -1, 0.5, -0.5, -1, 0.5, -1, -0.5, 0.5}

	one := func(name string, covers parity.Covers, verts []float32, count int,
		pipe func(*testing.T, *accel.Device) *accel.RenderPipeline) parityCase {
		return parityCase{
			name: name, covers: covers,
			run: parityPass{
				label: name, load: accel.LoadClear,
				draws: []parityDraw{
					{pipe: pipe, verts: verts,
						tint: [4]float32{0.75, 0.5, 0.25, 1}, count: count},
					{pipe: marker, verts: markerVerts,
						tint: [4]float32{0, 0.9, 0.9, 1}, count: 3},
				},
			}.run,
		}
	}

	return []parityCase{
		one("a triangle list with no culling",
			parity.Covers{"Topology.TriangleList", "CullMode.CullNone"}, tri, 3,
			prim("list", accel.TriangleList, accel.CounterClockwise, accel.CullNone)),
		one("a triangle strip",
			parity.Covers{"Topology.TriangleStrip"}, strip, 4,
			prim("strip", accel.TriangleStrip, accel.CounterClockwise, accel.CullNone)),
		one("a counter-clockwise front face against back-face culling",
			parity.Covers{"FrontFace.CounterClockwise", "CullMode.CullBack"}, tri, 3,
			prim("ccw back", accel.TriangleList, accel.CounterClockwise, accel.CullBack)),
		one("the same winding declared clockwise, so the triangle is a back face",
			parity.Covers{"FrontFace.Clockwise"}, tri, 3,
			prim("cw back", accel.TriangleList, accel.Clockwise, accel.CullBack)),
		one("front-face culling discarding the same triangle",
			parity.Covers{"CullMode.CullFront"}, tri, 3,
			prim("ccw front", accel.TriangleList, accel.CounterClockwise, accel.CullFront)),
	}
}

// blendParityCases sweep the factors against one operation and the operations
// against one pair of factors.
//
// A sweep rather than the full fifty-way product, per section 7: the product
// costs a pipeline compilation per combination and buys almost nothing over the
// two sweeps, because a factor and an operation are looked up in two
// independent tables and a wrong entry in either shows up in the sweep that
// varies it.
//
// Two draws, and the second is the blended one: a blend needs something in the
// destination to blend against, and blending over the clear would make every
// destination factor read the same background.
func blendParityCases() []parityCase {
	base := tintedPipeline(accel.RenderPipelineDescriptor{Label: "blend base"})
	under := [4]float32{0.5, 0.25, 0.75, 0.5}
	over := [4]float32{0.25, 0.75, 0.5, 0.25}

	blended := func(label string, b accel.BlendState) func(*testing.T, *accel.Device) *accel.RenderPipeline {
		return tintedPipeline(accel.RenderPipelineDescriptor{
			Label:   label,
			Targets: []accel.ColorTargetState{{Format: accel.RGBA32Float, Blend: b}},
		})
	}
	one := func(name string, covers parity.Covers, b accel.BlendState) parityCase {
		return parityCase{
			name: name, covers: covers,
			// The two roundings a blend takes -- one per side of the sum --
			// over values at most one, which is 2u of float32. Nothing is
			// tuned: the factors and the colours here are all in [0, 1].
			ceiling: parity.Ceiling{Abs: 2 * 1.1920929e-7,
				Why: "the two roundings of a blend's weighted sum, at values of at most 1"},
			run: parityPass{
				label: name, load: accel.LoadClear,
				draws: []parityDraw{
					{pipe: base, verts: fullViewport(0.5), tint: under, count: 3},
					{pipe: blended(name, b), verts: fullViewport(0.5), tint: over, count: 3},
				},
			}.run,
		}
	}

	factors := []struct {
		name string
		f    accel.BlendFactor
	}{
		{"FactorZero", accel.FactorZero},
		{"FactorOne", accel.FactorOne},
		{"FactorSrcColor", accel.FactorSrcColor},
		{"FactorOneMinusSrcColor", accel.FactorOneMinusSrcColor},
		{"FactorSrcAlpha", accel.FactorSrcAlpha},
		{"FactorOneMinusSrcAlpha", accel.FactorOneMinusSrcAlpha},
		{"FactorDstColor", accel.FactorDstColor},
		{"FactorOneMinusDstColor", accel.FactorOneMinusDstColor},
		{"FactorDstAlpha", accel.FactorDstAlpha},
		{"FactorOneMinusDstAlpha", accel.FactorOneMinusDstAlpha},
	}
	ops := []struct {
		name string
		op   accel.BlendOp
	}{
		{"BlendAdd", accel.BlendAdd},
		{"BlendSubtract", accel.BlendSubtract},
		{"BlendReverseSubtract", accel.BlendReverseSubtract},
		{"BlendMin", accel.BlendMin},
		{"BlendMax", accel.BlendMax},
	}

	cases := make([]parityCase, 0, len(factors)+len(ops))
	for _, f := range factors {
		// The factor under test scales the source and appears on the
		// destination side of the alpha channel too, so one case exercises it
		// in both positions -- 033 makes that a legal and useful combination,
		// and a backend that mapped the two sides from different tables would
		// only be caught by using both.
		cases = append(cases, one("a blend whose source factor is "+f.name,
			parity.Covers{"BlendFactor." + f.name}, accel.BlendState{
				Enabled:  true,
				SrcColor: f.f, DstColor: accel.FactorOne, ColorOp: accel.BlendAdd,
				SrcAlpha: accel.FactorOne, DstAlpha: f.f, AlphaOp: accel.BlendAdd,
			}))
	}
	for _, o := range ops {
		cases = append(cases, one("a blend combining through "+o.name,
			parity.Covers{"BlendOp." + o.name}, accel.BlendState{
				Enabled:  true,
				SrcColor: accel.FactorSrcAlpha, DstColor: accel.FactorOne, ColorOp: o.op,
				SrcAlpha: accel.FactorOne, DstAlpha: accel.FactorOne, AlphaOp: o.op,
			}))
	}
	return cases
}

// writeMaskParityCases mask one channel at a time.
//
// One channel each rather than one combined case, because the failure this
// finds is a backend numbering its channels from the other end: a mask that let
// red through where the caller asked for alpha writes a picture, and a case
// masking several channels at once could still agree by symmetry. The clear
// supplies four distinct channel values so a swapped pair is visible.
func writeMaskParityCases() []parityCase {
	masks := []struct {
		name string
		m    accel.ColorWriteMask
	}{
		{"WriteRed", accel.WriteRed},
		{"WriteGreen", accel.WriteGreen},
		{"WriteBlue", accel.WriteBlue},
		{"WriteAlpha", accel.WriteAlpha},
	}
	cases := make([]parityCase, 0, len(masks))
	for _, m := range masks {
		pipe := tintedPipeline(accel.RenderPipelineDescriptor{
			Label:   m.name,
			Targets: []accel.ColorTargetState{{Format: accel.RGBA32Float, Mask: m.m}},
		})
		cases = append(cases, parityCase{
			name:   "a draw writing only " + m.name,
			covers: parity.Covers{"ColorWriteMask." + m.name},
			run: parityPass{
				label: m.name, load: accel.LoadClear,
				draws: []parityDraw{{pipe: pipe, verts: fullViewport(0.5),
					tint: [4]float32{0.9, 0.8, 0.7, 0.6}, count: 3}},
			}.run,
		})
	}
	return cases
}

// attachmentOpParityCases cover the load operations and the surviving store
// operation.
//
// LoadDontCare is compared over a full-viewport draw on purpose. The contents
// it leaves where nothing wrote are undefined and the two backends are free to
// differ there; what is not free to differ is what the pass wrote, so the draw
// covers every pixel and the comparison is over written pixels only.
func attachmentOpParityCases() []parityCase {
	pipe := tintedPipeline(accel.RenderPipelineDescriptor{Label: "attachment op"})
	draw := []parityDraw{{pipe: pipe, verts: fullViewport(0.5),
		tint: [4]float32{0.4, 0.6, 0.2, 1}, count: 3}}
	half := []parityDraw{{pipe: pipe,
		verts: []float32{-1, -1, 0.5, 1, -1, 0.5, -1, 1, 0.5},
		tint:  [4]float32{0.4, 0.6, 0.2, 1}, count: 3}}

	return []parityCase{{
		name:   "a pass that clears its attachment",
		covers: parity.Covers{"LoadOp.LoadClear", "StoreOp.StoreKeep"},
		run:    parityPass{label: "clear", load: accel.LoadClear, draws: half}.run,
	}, {
		name:   "a pass that keeps what was already there",
		covers: parity.Covers{"LoadOp.LoadKeep"},
		run: parityPass{label: "keep", load: accel.LoadKeep, prefil: true,
			draws: half}.run,
	}, {
		name:   "a pass that discards the prior contents and covers the target",
		covers: parity.Covers{"LoadOp.LoadDontCare"},
		run: parityPass{label: "dontcare", load: accel.LoadDontCare, prefil: true,
			draws: draw}.run,
	}}
}

// indexFormatParityCases draw the same two triangles through both index widths.
func indexFormatParityCases() []parityCase {
	// Six vertices of position and tint, indexed as two triangles sharing an
	// edge, so an index read at the wrong width names a vertex that exists and
	// produces a deformed picture rather than a refusal.
	verts := []float32{
		-1, -1, 0.5, 1, 0, 0, 1,
		1, -1, 0.5, 0, 1, 0, 1,
		-1, 1, 0.5, 0, 0, 1, 1,
		1, 1, 0.5, 1, 1, 0, 1,
	}
	order := []uint32{0, 1, 2, 1, 3, 2}

	pack16 := func() []float32 {
		out := make([]float32, len(order)/2)
		for i := 0; i < len(order); i += 2 {
			out[i/2] = f32bits(order[i] | order[i+1]<<16)
		}
		return out
	}
	pack32 := func() []float32 {
		out := make([]float32, len(order))
		for i, v := range order {
			out[i] = f32bits(v)
		}
		return out
	}

	one := func(name string, format accel.IndexFormat, packed []float32) parityCase {
		return parityCase{
			name: name,
			covers: parity.Covers{
				"IndexFormat." + indexFormatConstName(format),
				// The layout attributePipeline declares: a three-component
				// position and a four-component colour, both fetched per vertex.
				"AttrFormat.AttrFloat32x3", "AttrFormat.AttrFloat32x4",
			},
			// Interpolated vertex colours: a sum of three products of a weight
			// and a varying, so a few ulps of the largest term, and the two
			// rasterizers may compute the barycentric weights differently.
			ceiling: parity.Ceiling{Abs: 4 * 1.1920929e-7,
				Why: "the barycentric interpolation of a vertex colour at values of at most 1"},
			run: func(t *testing.T, d *accel.Device) []byte {
				t.Helper()
				target := colourTarget(t, d, name, parityW, parityH)
				vb := newBuffer(t, d, "verts", len(verts),
					accel.BufferStorage|accel.BufferCopyDst)
				if err := d.Queue().WriteBuffer(vb, 0, verts); err != nil {
					t.Fatalf("write verts: %v", err)
				}
				ib := newBuffer(t, d, "indices", len(packed),
					accel.BufferStorage|accel.BufferCopyDst)
				if err := d.Queue().WriteBuffer(ib, 0, packed); err != nil {
					t.Fatalf("write indices: %v", err)
				}

				r := d.NewRecorder()
				p := r.RenderPass(accel.RenderPassDescriptor{
					Color: []accel.ColorAttachment{{
						View: view(t, target), Load: accel.LoadClear, Clear: parityClear,
					}},
					Width: parityW, Height: parityH, Label: name,
				})
				p.SetPipeline(attributePipeline(t, d))
				p.SetVertexBuffer(0, whole(t, vb))
				p.SetIndexBuffer(whole(t, ib), format)
				p.DrawIndexed(accel.DrawIndexed{IndexCount: len(order)})
				submitOne(t, d, r)
				return readTargetBytes(t, d, target)
			},
		}
	}
	return []parityCase{
		one("an indexed draw at uint16", accel.Index16, pack16()),
		one("an indexed draw at uint32", accel.Index32, pack32()),
	}
}

func indexFormatConstName(f accel.IndexFormat) string {
	if f == accel.Index32 {
		return "Index32"
	}
	return "Index16"
}

// renderStateParityExclusions are the members no case can compare today.
//
// Each names the code that makes it impossible and what deleting the entry
// waits on. An exclusion is a debt with a stated creditor: when the backend
// gains what the entry names, the entry goes and the gate demands a case.
func renderStateParityExclusions() []parity.Excluded {
	const noStencil = "the Metal backend refuses a stencil state by name " +
		"(internal/metal/render_darwin.go: \"does not lower stencil state\"), so the " +
		"CPU backend's stencil has nothing to be compared against. Delete this entry " +
		"when specs/033-render-api.md section 10.5's Metal half lands"
	const noPrimitive = "refused on both backends: specs/035-cpu-rasterizer.md " +
		"section 10 leaves the rule open, because lines have a diamond-exit rule on " +
		"some backends and a Bresenham-ish one on others. A refusal is not a result " +
		"to compare"

	out := []parity.Excluded{
		{Name: "Topology.LineList", Why: noPrimitive},
		{Name: "Topology.LineStrip", Why: noPrimitive},
		{Name: "Topology.PointList", Why: noPrimitive},
		{Name: "StoreOp.StoreDiscard", Why: "an attachment's contents after a discard " +
			"are undefined by definition, so a comparison would be asserting a value " +
			"neither backend promises. That the write-back is skipped is checked " +
			"elsewhere; what it leaves behind is not a parity claim"},
		{Name: "AttrFormat.AttrInvalid", Why: "the unset value: pipeline creation " +
			"refuses it (vertexlayout.go, Components() == 0). A refusal is not a result"},
		{Name: "AttrFormat.AttrFloat32", Why: "no stage in the corpus reads a " +
			"one-component attribute, so there is nothing to fetch into. Delete this " +
			"entry when a [1]float32 attribute stage is generated"},
	}
	for _, op := range []string{"StencilKeep", "StencilZero", "StencilReplace",
		"StencilIncrementClamp", "StencilDecrementClamp", "StencilInvert",
		"StencilIncrementWrap", "StencilDecrementWrap"} {
		out = append(out, parity.Excluded{Name: "StencilOp." + op, Why: noStencil})
	}
	return out
}

// depthOfTwoFlatTriangles writes two constant depths and reads the depth
// aspect back through a recorded copy.
//
// Constant depths rather than a slanted quad, and that is what makes the
// comparison exact: an interpolated depth is a barycentric sum the two
// rasterizers may compute differently, and this case is about whether the depth
// aspect stores and returns what was written, not about interpolation. Two
// values rather than one, so a target that came back filled with a single
// constant -- a copy that read the wrong subresource, say -- is not mistaken
// for agreement.
func depthOfTwoFlatTriangles(t *testing.T, d *accel.Device) []float32 {
	t.Helper()
	far := tintedPipeline(accel.RenderPipelineDescriptor{
		Label: "depth far",
		DepthStencil: &accel.DepthStencilState{
			Format: accel.Depth32Float, Test: true, Write: true,
			Compare: accel.CompareAlways,
		},
	})
	near := tintedPipeline(accel.RenderPipelineDescriptor{
		Label: "depth near",
		DepthStencil: &accel.DepthStencilState{
			Format: accel.Depth32Float, Test: true, Write: true,
			Compare: accel.CompareLess,
		},
	})

	colour := colourTarget(t, d, "depth colour", parityW, parityH)
	depth := depthTarget(t, d, "depth", parityW, parityH)

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{
			View: view(t, colour), Load: accel.LoadClear, Clear: parityClear,
		}},
		Depth: &accel.DepthAttachment{
			View: view(t, depth), Load: accel.LoadClear, Clear: 1,
			Store: accel.StoreKeep,
		},
		Width: parityW, Height: parityH, Label: "depth aspect",
	})
	for _, dr := range []parityDraw{
		{pipe: far, verts: fullViewport(0.75), tint: [4]float32{1, 0, 0, 1}, count: 3},
		{pipe: near, verts: []float32{-1, -1, 0.25, 1, -1, 0.25, -1, 1, 0.25},
			tint: [4]float32{0, 1, 0, 1}, count: 3},
	} {
		vb := newBuffer(t, d, "verts", len(dr.verts),
			accel.BufferStorage|accel.BufferCopyDst)
		if err := d.Queue().WriteBuffer(vb, 0, dr.verts); err != nil {
			t.Fatalf("write verts: %v", err)
		}
		p.SetPipeline(dr.pipe(t, d))
		p.SetVertexBuffer(0, whole(t, vb))
		p.SetVertexUniform(0, testkernels.StageTransform{Scale: 1})
		p.SetFragmentUniform(0, testkernels.StageTint{Colour: dr.tint})
		p.Draw(accel.Draw{VertexCount: dr.count})
	}
	submitOne(t, d, r)
	return readDepth(t, d, depth)
}

// attrFormatParityCases fetch a two-component attribute, into a pass with two
// colour targets.
//
// Two targets because the stage that reads a two-component attribute is the one
// that writes two of them, and running it covers both at once rather than
// building a second fixture. The second target carries the fetched attribute,
// so a backend that fetched the wrong width is visible as a value rather than
// as an absence.
func attrFormatParityCases() []parityCase {
	// Interleaved position and texture coordinate, five floats per vertex.
	verts := []float32{
		-1, -1, 0.5, 0, 0,
		3, -1, 0.5, 2, 0,
		-1, 3, 0.5, 0, 2,
	}
	return []parityCase{{
		name:   "a two-component attribute into a two-target pass",
		covers: parity.Covers{"AttrFormat.AttrFloat32x2"},
		// Two interpolated varyings, each a barycentric sum of three products,
		// at values of at most two.
		ceiling: parity.Ceiling{Abs: 8 * 1.1920929e-7,
			Why: "the barycentric interpolation of a position and a texture coordinate"},
		run: func(t *testing.T, d *accel.Device) []byte {
			t.Helper()
			pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
				Vertex:   &testkernels.GeometryVSStage,
				Fragment: &testkernels.ShadeFSStage,
				VertexBuffers: []accel.VertexBufferLayout{{
					Stride: 20, StepMode: accel.StepVertex,
					Attributes: []accel.VertexAttribute{
						{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
						{Location: 1, Format: accel.AttrFloat32x2, Offset: 12},
					},
				}},
				Targets: []accel.ColorTargetState{
					{Format: accel.RGBA32Float}, {Format: accel.RGBA32Float},
				},
				Label: "two-component attribute",
			})
			if err != nil {
				t.Fatalf("pipeline: %v", err)
			}
			defer pipe.Close()

			albedo := colourTarget(t, d, "albedo", parityW, parityH)
			normal := colourTarget(t, d, "normal", parityW, parityH)
			vb := newBuffer(t, d, "verts", len(verts),
				accel.BufferStorage|accel.BufferCopyDst)
			if err := d.Queue().WriteBuffer(vb, 0, verts); err != nil {
				t.Fatalf("write verts: %v", err)
			}

			r := d.NewRecorder()
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{
					{View: view(t, albedo), Load: accel.LoadClear, Clear: parityClear},
					{View: view(t, normal), Load: accel.LoadClear, Clear: parityClear},
				},
				Width: parityW, Height: parityH, Label: "two-component attribute",
			})
			p.SetPipeline(pipe)
			p.SetVertexBuffer(0, whole(t, vb))
			p.SetVertexUniform(0, testkernels.StageTransform{Scale: 1})
			p.Draw(accel.Draw{VertexCount: 3})
			submitOne(t, d, r)
			return append(readTargetBytes(t, d, albedo), readTargetBytes(t, d, normal)...)
		},
	}}
}

// normalizedAttrParityCases fetch each normalized integer attribute format.
//
// specs/033-render-api.md states the four conversions rather than leaving them
// to a backend, and a stated conversion is one two backends can disagree about:
// the divisor is exact, so a difference here is a backend using the wrong one
// or reading the wrong width. Every case includes the extreme value -- 0 and
// 255, or -128 and 127 -- because the signed clamp at -1 is one input value
// wide and is invisible in a rendered normal.
func normalizedAttrParityCases() []parityCase {
	type spec struct {
		name  string
		f     accel.AttrFormat
		size  int // bytes the attribute occupies
		bytes func(vertex int) []byte
	}

	// Three vertices, and the per-vertex values walk the range so each case
	// carries its own extremes.
	u8 := func(v int, n int) []byte {
		out := make([]byte, n)
		for i := range out {
			out[i] = []byte{0, 127, 255}[(v+i)%3]
		}
		return out
	}
	s8 := func(v int, n int) []byte {
		out := make([]byte, n)
		for i := range out {
			out[i] = byte(int8([]int{-128, 0, 127}[(v+i)%3]))
		}
		return out
	}
	u16 := func(v int, n int) []byte {
		out := make([]byte, n*2)
		for i := range n {
			binary.LittleEndian.PutUint16(out[i*2:],
				[]uint16{0, 32767, 65535}[(v+i)%3])
		}
		return out
	}
	s16 := func(v int, n int) []byte {
		out := make([]byte, n*2)
		for i := range n {
			binary.LittleEndian.PutUint16(out[i*2:],
				uint16([]int16{-32768, 0, 32767}[(v+i)%3]))
		}
		return out
	}

	specs := []spec{
		{"AttrUnorm8x2", accel.AttrUnorm8x2, 2, func(v int) []byte { return u8(v, 2) }},
		{"AttrUnorm8x4", accel.AttrUnorm8x4, 4, func(v int) []byte { return u8(v, 4) }},
		{"AttrSnorm8x2", accel.AttrSnorm8x2, 2, func(v int) []byte { return s8(v, 2) }},
		{"AttrSnorm8x4", accel.AttrSnorm8x4, 4, func(v int) []byte { return s8(v, 4) }},
		{"AttrUnorm16x2", accel.AttrUnorm16x2, 4, func(v int) []byte { return u16(v, 2) }},
		{"AttrUnorm16x4", accel.AttrUnorm16x4, 8, func(v int) []byte { return u16(v, 4) }},
		{"AttrSnorm16x2", accel.AttrSnorm16x2, 4, func(v int) []byte { return s16(v, 2) }},
		{"AttrSnorm16x4", accel.AttrSnorm16x4, 8, func(v int) []byte { return s16(v, 4) }},
	}

	cases := make([]parityCase, 0, len(specs))
	for _, sp := range specs {
		cases = append(cases, parityCase{
			name:   "a " + sp.name + " attribute fetched and interpolated",
			covers: parity.Covers{"AttrFormat." + sp.name},
			// The conversion itself is one exact division and must agree; what
			// is bounded is the barycentric interpolation of the result, a sum
			// of three products at values of at most one.
			ceiling: parity.Ceiling{Abs: 4 * 1.1920929e-7,
				Why: "the barycentric interpolation of the converted attribute; the conversion's divisor is exact"},
			run: normalizedAttrRun(sp.name, sp.f, sp.size, sp.bytes),
		})
	}
	return cases
}

// normalizedAttrRun draws one triangle whose second attribute is the format
// under test, and reads the interpolated result back.
//
// Two- and four-component formats reach different stages, because a stage's
// attribute parameter is [N]float32 and there is no stage that takes both
// widths. The two-component path writes two attachments, which is what the
// stage that reads a Vec2 attribute happens to do.
func normalizedAttrRun(name string, f accel.AttrFormat, size int,
	at func(int) []byte) func(*testing.T, *accel.Device) []byte {
	// The attribute is padded to a four-byte boundary so the vertex buffer can
	// be written as float32 words, which is the only width the queue takes.
	stride := 12 + (size+3)/4*4
	raw := make([]byte, 3*stride)
	pos := [][3]float32{{-1, -1, 0.5}, {3, -1, 0.5}, {-1, 3, 0.5}}
	for v := range 3 {
		base := v * stride
		for c := range 3 {
			binary.LittleEndian.PutUint32(raw[base+c*4:], math.Float32bits(pos[v][c]))
		}
		copy(raw[base+12:], at(v))
	}
	verts := make([]float32, len(raw)/4)
	for i := range verts {
		verts[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}

	layout := []accel.VertexBufferLayout{{
		Stride: stride, StepMode: accel.StepVertex,
		Attributes: []accel.VertexAttribute{
			{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
			{Location: 1, Format: f, Offset: 12},
		},
	}}

	return func(t *testing.T, d *accel.Device) []byte {
		t.Helper()
		desc := accel.RenderPipelineDescriptor{
			VertexBuffers: layout, Label: "parity " + name,
		}
		targets := 1
		if f.Components() == 4 {
			desc.Vertex, desc.Fragment = &testkernels.AttributeVSStage, &testkernels.TintFSStage
			desc.Targets = []accel.ColorTargetState{{Format: accel.RGBA32Float}}
		} else {
			desc.Vertex, desc.Fragment = &testkernels.GeometryVSStage, &testkernels.ShadeFSStage
			desc.Targets = []accel.ColorTargetState{
				{Format: accel.RGBA32Float}, {Format: accel.RGBA32Float},
			}
			targets = 2
		}
		pipe, err := d.NewRenderPipeline(desc)
		if err != nil {
			t.Fatalf("pipeline for %s: %v", name, err)
		}
		defer pipe.Close()

		colour := make([]*accel.Texture, targets)
		attachments := make([]accel.ColorAttachment, targets)
		for i := range colour {
			colour[i] = colourTarget(t, d, name, parityW, parityH)
			attachments[i] = accel.ColorAttachment{
				View: view(t, colour[i]), Load: accel.LoadClear, Clear: parityClear,
			}
		}
		vb := newBuffer(t, d, "verts", len(verts),
			accel.BufferStorage|accel.BufferCopyDst)
		if err := d.Queue().WriteBuffer(vb, 0, verts); err != nil {
			t.Fatalf("write verts for %s: %v", name, err)
		}

		r := d.NewRecorder()
		p := r.RenderPass(accel.RenderPassDescriptor{
			Color: attachments, Width: parityW, Height: parityH, Label: name,
		})
		p.SetPipeline(pipe)
		p.SetVertexBuffer(0, whole(t, vb))
		// AttributeVS takes no by-value parameter and GeometryVS does, and a
		// uniform set for a stage that declares none is refused rather than
		// ignored.
		if f.Components() != 4 {
			p.SetVertexUniform(0, testkernels.StageTransform{Scale: 1})
		}
		p.Draw(accel.Draw{VertexCount: 3})
		submitOne(t, d, r)

		var out []byte
		for _, tex := range colour {
			out = append(out, readTargetBytes(t, d, tex)...)
		}
		return out
	}
}
