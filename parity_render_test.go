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
	cases := []parityCase{{
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
	cases = append(cases, normalizedAttrParityCases()...)
	return append(cases, stencilOpParityCases()...)
}

// normalizedAttrParityCases compares each normalized integer attribute format
// between the two backends.
//
// specs/042-surface-completion.md section 10.2 states each conversion, and
// these are what hold the two backends to it.
//
// # Bounded, and the bound is not the conversion's
//
// The conversion itself is exact -- a division by a constant, which the two
// backends have nothing to weight differently -- and the value still reaches
// the attachment through an *interpolated varying*. A constant attribute
// interpolates to itself in real arithmetic and to within an ulp in f32, so two
// rasterizers computing barycentric weights differently disagree in the last
// bits. This was found by running the case exactly: one pixel of sixty-four
// differed by one ulp, which is specs/008-numerics.md section 8.1's term rather
// than a conversion error.
//
// # The bytes are chosen where the conversion is interesting
//
// Both ends, and -128 for the signed forms -- the value where the clamp shows.
// Two's complement has one more negative value than positive, so -128/127 is
// -1.007874 and every target defines the result as -1. A fixture of middling
// values would agree with a backend that omitted the clamp.
func normalizedAttrParityCases() []parityCase {
	// The attribute is constant across the three vertices, so interpolation
	// cannot change it: a constant field interpolates to itself under any
	// weights, which leaves the conversion as the only thing under test.
	pos := []float32{-1, -1, 0.5, 3, -1, 0.5, -1, 3, 0.5}

	var cases []parityCase
	for _, c := range []struct {
		name   string
		format accel.AttrFormat
		raw    []byte
	}{
		{"AttrUnorm8x2", accel.AttrUnorm8x2, []byte{0, 255}},
		{"AttrUnorm8x4", accel.AttrUnorm8x4, []byte{0, 255, 128, 1}},
		{"AttrSnorm8x2", accel.AttrSnorm8x2, []byte{0x80, 0x7f}},
		{"AttrSnorm8x4", accel.AttrSnorm8x4, []byte{0x80, 0x81, 0x7f, 0x00}},
		{"AttrUnorm16x2", accel.AttrUnorm16x2, []byte{0x00, 0x00, 0xff, 0xff}},
		{"AttrUnorm16x4", accel.AttrUnorm16x4,
			[]byte{0x00, 0x00, 0xff, 0xff, 0x00, 0x80, 0x01, 0x00}},
		{"AttrSnorm16x2", accel.AttrSnorm16x2, []byte{0x00, 0x80, 0xff, 0x7f}},
		{"AttrSnorm16x4", accel.AttrSnorm16x4,
			[]byte{0x00, 0x80, 0x01, 0x80, 0xff, 0x7f, 0x00, 0x00}},
	} {
		cases = append(cases, normalizedAttrCase(c.name, c.format, c.raw, pos))
	}
	return cases
}

// normalizedAttrCase renders one triangle whose location-1 attribute holds raw.
//
// The stage is chosen by the attribute's width: pipeline creation checks the
// declared format against the parameter's type, so a two-component format needs
// the stage that takes a Vec2 and a four-component one the stage that takes a
// Vec4. A mismatch is refused, correctly -- a fetch of the wrong width deforms
// geometry rather than losing it.
func normalizedAttrCase(name string, f accel.AttrFormat, raw []byte, pos []float32) parityCase {
	two := f.Components() == 2
	stride := 12 + 16
	verts := make([]byte, stride*3)
	for v := range 3 {
		for i := range 3 {
			binary.LittleEndian.PutUint32(verts[v*stride+i*4:], math.Float32bits(pos[v*3+i]))
		}
		copy(verts[v*stride+12:], raw)
	}

	return parityCase{
		name:   "a " + name + " attribute",
		covers: parity.Covers{"AttrFormat." + name},
		// specs/008-numerics.md section 8.1: a triangle's interpolation carries
		// γ(4) + γ(3) + u of the largest vertex value, which is 8u, and a
		// normalized value is at most 1.
		ceiling: parity.Ceiling{Abs: 8 * 1.1920929e-7,
			Why: "the attribute reaches the attachment through an interpolated " +
				"varying; the conversion itself is exact"},
		run: func(t *testing.T, d *accel.Device) []byte {
			t.Helper()
			vs, fs := &testkernels.AttributeVSStage, &testkernels.TintFSStage
			targets := []accel.ColorTargetState{{Format: accel.RGBA32Float}}
			if two {
				vs, fs = &testkernels.GeometryVSStage, &testkernels.ShadeFSStage
				targets = append(targets, accel.ColorTargetState{Format: accel.RGBA32Float})
			}
			pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
				Vertex: vs, Fragment: fs,
				VertexBuffers: []accel.VertexBufferLayout{{
					Stride: stride, StepMode: accel.StepVertex,
					Attributes: []accel.VertexAttribute{
						{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
						{Location: 1, Format: f, Offset: 12},
					},
				}},
				Targets: targets, Label: name,
			})
			if err != nil {
				t.Fatalf("pipeline: %v", err)
			}
			defer pipe.Close()

			vb := newBytes(t, d, "verts", len(verts))
			if err := d.Queue().WriteBuffer(vb, 0, verts); err != nil {
				t.Fatalf("write verts: %v", err)
			}

			first := colourTarget(t, d, "first", parityW, parityH)
			colour := []accel.ColorAttachment{
				{View: view(t, first), Load: accel.LoadClear, Clear: parityClear},
			}
			read := first
			if two {
				// ShadeFS writes the interpolated colour to attachment 0 and
				// the fetched attribute to attachment 1.
				second := colourTarget(t, d, "second", parityW, parityH)
				colour = append(colour, accel.ColorAttachment{
					View: view(t, second), Load: accel.LoadClear, Clear: parityClear,
				})
				read = second
			}

			r := d.NewRecorder()
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: colour, Width: parityW, Height: parityH, Label: name,
			})
			p.SetPipeline(pipe)
			p.SetVertexBuffer(0, whole(t, vb))
			if two {
				p.SetVertexUniform(0, testkernels.StageTransform{Scale: 1})
			}
			p.Draw(accel.Draw{VertexCount: 3})
			submitOne(t, d, r)
			return readTargetBytes(t, d, read)
		},
	}
}

// stencilOpParityCases compares each stencil operation between the backends.
//
// specs/033-render-api.md section 10 built the CPU half and section 10.5's
// Metal half landed with specs/045-texture-attachments.md section 12's planar
// layout, so the whole surface is comparable for the first time.
//
// # The operation is observed through a second pass
//
// A stencil buffer cannot be read back -- Queue.ReadTexture refuses every depth
// format -- so each case marks with the operation under test and then draws a
// full-screen pass that keeps only where the stencil equals the value that
// operation should have left. The coverage of the second pass *is* the stencil
// buffer, and it is a picture the harness already knows how to compare.
func stencilOpParityCases() []parityCase {
	// The marking pass covers the lower-left half, so an operation that wrote
	// everywhere and one that wrote nowhere are different pictures.
	face := func(op accel.StencilOp, compare accel.CompareFunc) accel.StencilFace {
		return accel.StencilFace{
			Compare: compare, ReadMask: 0xff, WriteMask: 0xff,
			Fail: accel.StencilKeep, DepthFail: accel.StencilKeep, Pass: op,
		}
	}
	var cases []parityCase
	for _, c := range []struct {
		name string
		op   accel.StencilOp
		// want is what the buffer holds where the marking pass covered, given a
		// clear of 3 and a reference of 5.
		want uint8
	}{
		{"StencilKeep", accel.StencilKeep, 3},
		{"StencilZero", accel.StencilZero, 0},
		{"StencilReplace", accel.StencilReplace, 5},
		{"StencilIncrementClamp", accel.StencilIncrementClamp, 4},
		{"StencilDecrementClamp", accel.StencilDecrementClamp, 2},
		{"StencilInvert", accel.StencilInvert, 0xfc},
		{"StencilIncrementWrap", accel.StencilIncrementWrap, 4},
		{"StencilDecrementWrap", accel.StencilDecrementWrap, 2},
	} {
		cases = append(cases, stencilOpCase(c.name, c.op, c.want, face))
	}
	return cases
}

func stencilOpCase(name string, op accel.StencilOp, want uint8,
	face func(accel.StencilOp, accel.CompareFunc) accel.StencilFace) parityCase {
	return parityCase{
		name:   "the stencil operation " + name,
		covers: parity.Covers{"StencilOp." + name},
		run: func(t *testing.T, d *accel.Device) []byte {
			t.Helper()
			mark := stencilPipeline(t, d, &testkernels.HalfTriangleVSStage,
				face(op, accel.CompareAlways), true, name+" mark")
			defer mark.Close()
			// Equal against the value the operation should have left, with the
			// reference supplied per pass.
			test := stencilPipeline(t, d, &testkernels.FullScreenVSStage,
				face(accel.StencilKeep, accel.CompareEqual), false, name+" test")
			defer test.Close()

			colour := colourTarget(t, d, "colour", parityW, parityH)
			depth, err := d.NewTexture(accel.TextureDescriptor{
				Format: accel.Depth32FloatStencil8,
				Size:   accel.Extent{Width: parityW, Height: parityH},
				Usage: accel.TextureRenderTarget | accel.TextureCopySrc |
					accel.TextureCopyDst,
				Kind: accel.MemoryDevice, Label: "stencil",
			})
			if err != nil {
				t.Fatalf("depth: %v", err)
			}
			defer depth.Close()
			dv, err := depth.Whole()
			if err != nil {
				t.Fatalf("depth view: %v", err)
			}

			r := d.NewRecorder()
			p1 := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{
					View: view(t, colour), Load: accel.LoadClear, Clear: parityClear,
				}},
				Depth: &accel.DepthAttachment{View: dv, Load: accel.LoadClear, Clear: 1},
				Width: parityW, Height: parityH, Label: "mark",
			})
			p1.SetPipeline(mark)
			p1.SetStencilReference(5)
			p1.Draw(accel.Draw{VertexCount: 3})

			p2 := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{
					View: view(t, colour), Load: accel.LoadKeep,
				}},
				Depth: &accel.DepthAttachment{View: dv, Load: accel.LoadKeep},
				Width: parityW, Height: parityH, Label: "test",
			})
			p2.SetPipeline(test)
			p2.SetStencilReference(want)
			p2.Draw(accel.Draw{VertexCount: 3})

			submitOne(t, d, r)
			return readTargetBytes(t, d, colour)
		},
	}
}

// stencilPipeline is a pipeline whose stencil state is one face repeated.
//
// The clear is 3 rather than 0 so that Keep, Zero and Decrement are three
// different answers rather than two.
func stencilPipeline(t *testing.T, d *accel.Device, vs *accel.Stage,
	f accel.StencilFace, write bool, label string) *accel.RenderPipeline {
	t.Helper()
	p, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   vs,
		Fragment: &testkernels.SolidFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		DepthStencil: &accel.DepthStencilState{
			Format:  accel.Depth32FloatStencil8,
			Stencil: accel.StencilState{Enabled: true, Front: f, Back: f},
		},
		Label: label,
	})
	if err != nil {
		t.Fatalf("%s pipeline: %v", label, err)
	}
	return p
}
