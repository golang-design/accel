// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package testkernels_test

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// notLowered is the Metal backend's honest refusal of a texture attachment.
//
// specs/045-texture-attachments.md made an attachment a texture view and gave
// the Metal path its own slice, so on a Metal device every pass here is
// refused at Build until that lands. The entries skip with the reason rather
// than being deleted, which is what makes the comparison start again on the
// first day it can -- the arrangement TestATextureRoundTripKeepsCallerOrderOnMetal
// already uses, and the reason a convention bug is cheapest to catch in the
// commit that makes it reachable.
const notLowered = "does not lower a texture attachment at specs/045-texture-attachments.md"

func skipUnlessLowered(t *testing.T, err error) {
	t.Helper()
	if err != nil && strings.Contains(err.Error(), notLowered) {
		t.Skipf("owed, not failing: this backend does not lower a texture attachment "+
			"yet, so there is no Metal side to compare against the oracle: %v", err)
	}
}

// colourTexture is a render target the host can read back.
func colourTexture(t *testing.T, d *accel.Device, label string, w, h int) *accel.Texture {
	t.Helper()
	tex, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.RGBA32Float, Size: accel.Extent{Width: w, Height: h},
		Usage: accel.TextureRenderTarget | accel.TextureCopySrc | accel.TextureCopyDst,
		Kind:  accel.MemoryReadback, Label: label,
	})
	if err != nil {
		t.Fatalf("texture %s: %v", label, err)
	}
	t.Cleanup(func() { _ = tex.Close() })
	return tex
}

func wholeOf(t *testing.T, tex *accel.Texture) accel.TextureView {
	t.Helper()
	v, err := tex.Whole()
	if err != nil {
		t.Fatalf("view: %v", err)
	}
	return v
}

func readColourTexture(t *testing.T, d *accel.Device, tex *accel.Texture) []float32 {
	t.Helper()
	sz := tex.Size()
	raw := make([]byte, sz.Width*sz.Height*tex.Format().BytesPerPixel())
	if err := d.Queue().ReadTexture(tex, raw); err != nil {
		t.Fatalf("read texture: %v", err)
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 |
			uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24)
	}
	return out
}

// The differential the MSL stage target could not have without a render path:
// the same graph on the CPU rasterizer and on Metal, compared pixel by pixel.
//
// This is what makes the CPU backend an oracle for graphics rather than the
// only implementation. specs/032-stage-abi.md section 12.1 said the MSL target
// was "emitted and compiled, not differentially verified"; this is the
// verification, and it is the reason one IR with two lowerings is worth the
// arrangement — the two are written independently and must still agree.
//
// Compared within a bound rather than exactly, and the bound is derived rather
// than tuned: interpolation is a sum of three products of a weight and a
// varying, so its error is at most a few ulps of the largest term, and the
// two rasterizers are free to compute the barycentric weights differently.
func TestARenderPassAgreesOnBothBackends(t *testing.T) {
	const w, h = 16, 16
	metal := openMetalDevice(t)
	cpu, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer cpu.Close()

	// Interleaved position and colour, so the vertex layout, the fetch and the
	// interpolation are all in the comparison rather than only coverage.
	verts := []float32{
		-1, -1, 0, 1, 0, 0, 1,
		1, -1, 0, 0, 1, 0, 1,
		-1, 1, 0, 0, 0, 1, 1,
	}

	render := func(t *testing.T, d *accel.Device) []float32 {
		t.Helper()
		q := d.Queue()
		pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
			Vertex:   &testkernels.AttributeVSStage,
			Fragment: &testkernels.TintFSStage,
			VertexBuffers: []accel.VertexBufferLayout{{
				Stride: 28,
				Attributes: []accel.VertexAttribute{
					{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
					{Location: 1, Format: accel.AttrFloat32x4, Offset: 12},
				},
			}},
			Targets: []accel.ColorTargetState{{Format: accel.RGBA32Float}},
			Label:   "differential",
		})
		if err != nil {
			t.Fatalf("pipeline: %v", err)
		}
		defer pipe.Close()

		usage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
		vb, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: len(verts), Usage: usage, Label: "verts",
		})
		if err != nil {
			t.Fatalf("buffer: %v", err)
		}
		defer vb.Close()
		if err := q.WriteBuffer(vb, 0, verts); err != nil {
			t.Fatalf("write: %v", err)
		}
		target := colourTexture(t, d, "colour", w, h)

		vv, err := vb.View(0, vb.Count())
		if err != nil {
			t.Fatalf("view: %v", err)
		}
		tv := wholeOf(t, target)

		r := d.NewRecorder()
		p := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{
				View: tv, Load: accel.LoadClear, Clear: [4]float32{0, 0, 0, 0},
			}},
			Width: w, Height: h, Label: "differential",
		})
		p.SetPipeline(pipe)
		p.SetVertexBuffer(0, vv)
		p.Draw(accel.Draw{VertexCount: 3})

		g, err := r.Build()
		if err != nil {
			skipUnlessLowered(t, err)
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		if err := q.Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		return readColourTexture(t, d, target)
	}

	onCPU := render(t, cpu)
	onMetal := render(t, metal)

	// One float32 rounding of the largest colour component, doubled for the
	// two roundings a weighted sum takes. Nothing is tuned: the colours here
	// are at most 1, so this is 2u.
	const bound = 2 * 1.1920929e-7

	var covered, coveredMetal, differing int
	for i := range onCPU {
		if onCPU[i] != 0 {
			covered++
		}
		if onMetal[i] != 0 {
			coveredMetal++
		}
		if d := math.Abs(float64(onCPU[i] - onMetal[i])); d > bound {
			differing++
			if differing < 4 {
				px := i / 4
				t.Errorf("pixel (%d,%d) channel %d is %v on the CPU and %v on Metal, "+
					"which is %v apart and the derived bound is %v",
					px%w, px/w, i%4, onCPU[i], onMetal[i], d, bound)
			}
		}
	}
	if differing > 0 {
		t.Errorf("%d of %d floats differ by more than %v", differing, len(onCPU), bound)
	}
	// Both must have drawn. Two blank images agree perfectly, and the CPU check
	// alone would not catch a Metal path that encoded nothing -- the comparison
	// would then fail, but only because one side is blank, which is a different
	// bug from the one this test is for.
	if covered == 0 {
		t.Fatal("the CPU backend drew nothing, so agreeing with Metal proves nothing")
	}
	if coveredMetal == 0 {
		t.Fatal("Metal drew nothing")
	}
	t.Logf("%d of %d floats non-zero on the CPU, %d on Metal", covered, len(onCPU),
		coveredMetal)
}

// The differential over the rest of the render surface.
//
// Each case exercises a path the plain triangle does not reach, and every one
// is a place the two backends could disagree in a way that still produces a
// picture: a uniform encoded in the wrong layout, a depth compare mapped to the
// wrong Metal constant, a blend factor off by one in an enumeration, an index
// buffer read at the wrong width. The oracle is the CPU rasterizer, which is
// tested directly against the arithmetic elsewhere.
func TestTheRenderSurfaceAgreesOnBothBackends(t *testing.T) {
	metal := openMetalDevice(t)
	cpu, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer cpu.Close()

	for _, c := range []struct {
		name string
		run  func(t *testing.T, d *accel.Device) []float32
	}{
		{"by-value stage parameters", renderWithUniforms},
		{"depth test and write", renderWithDepth},
		{"alpha blending over two draws", renderBlended},
		{"an indexed draw", renderIndexed},
		{"LoadKeep over prior contents", renderKeeping},
		{"a depth attachment no draw tests", renderUntestedDepth},
		{"two draws sharing one pipeline", renderTwice},
	} {
		t.Run(c.name, func(t *testing.T) {
			onCPU := c.run(t, cpu)
			onMetal := c.run(t, metal)
			compareBackends(t, onCPU, onMetal)
		})
	}
}

// compareBackends asserts the two agree within a derived bound and that neither
// is blank.
func compareBackends(t *testing.T, onCPU, onMetal []float32) {
	t.Helper()
	// Four roundings of a unit-magnitude value: the interpolated varying, the
	// blend's two products and their sum. Nothing tuned.
	const bound = 4 * 1.1920929e-7
	var live, liveMetal, differing int
	for i := range onCPU {
		if onCPU[i] != 0 {
			live++
		}
		if onMetal[i] != 0 {
			liveMetal++
		}
		if d := math.Abs(float64(onCPU[i] - onMetal[i])); d > bound {
			differing++
			if differing < 4 {
				t.Errorf("float %d is %v on the CPU and %v on Metal, %v apart against a "+
					"derived bound of %v", i, onCPU[i], onMetal[i], d, bound)
			}
		}
	}
	if differing > 0 {
		t.Errorf("%d of %d floats differ", differing, len(onCPU))
	}
	if live == 0 || liveMetal == 0 {
		t.Fatalf("one side is blank: %d non-zero on the CPU and %d on Metal; two blank "+
			"images agree perfectly", live, liveMetal)
	}
}

// renderFixture is the shared scaffolding: a pipeline, buffers, one pass, and
// the readback.
type renderFixture struct {
	desc     accel.RenderPipelineDescriptor
	verts    []float32
	indices  []uint32
	depth    bool
	keep     []float32
	dontCare bool
	record   func(p *accel.RenderPass)
	draws    func(p *accel.RenderPass)
}

func runFixture(t *testing.T, d *accel.Device, f renderFixture) []float32 {
	t.Helper()
	const w, h = 16, 16
	q := d.Queue()
	usage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst

	pipe, err := d.NewRenderPipeline(f.desc)
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	newBuf := func(label string, n int) (*accel.Buffer, accel.BufferView) {
		b, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: n, Usage: usage, Label: label,
		})
		if err != nil {
			t.Fatalf("buffer %s: %v", label, err)
		}
		t.Cleanup(func() { _ = b.Close() })
		v, err := b.View(0, n)
		if err != nil {
			t.Fatalf("view %s: %v", label, err)
		}
		return b, v
	}

	_, vv := newBuf("verts", len(f.verts))
	if err := q.WriteBuffer(vv.Buffer, 0, f.verts); err != nil {
		t.Fatalf("write verts: %v", err)
	}
	target := colourTexture(t, d, "colour", w, h)
	tv := wholeOf(t, target)
	var prior accel.BufferView
	if f.keep != nil {
		priorBuf, v := newBuf("prior", w*h*4)
		if err := q.WriteBuffer(priorBuf, 0, f.keep); err != nil {
			t.Fatalf("write prior: %v", err)
		}
		prior = v
	}

	load := accel.LoadClear
	if f.keep != nil {
		load = accel.LoadKeep
	}
	if f.dontCare {
		load = accel.LoadDontCare
	}
	desc := accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{
			View: tv, Load: load, Clear: [4]float32{0, 0, 0, 0},
		}},
		Width: w, Height: h, Label: "differential",
	}
	if f.depth {
		dtex, err := d.NewTexture(accel.TextureDescriptor{
			Format: accel.Depth32Float, Size: accel.Extent{Width: w, Height: h},
			Usage: accel.TextureRenderTarget | accel.TextureCopySrc, Label: "depth",
		})
		if err != nil {
			t.Fatalf("depth texture: %v", err)
		}
		defer dtex.Close()
		desc.Depth = &accel.DepthAttachment{
			View: wholeOf(t, dtex), Load: accel.LoadClear, Clear: 1,
		}
	}

	r := d.NewRecorder()
	// There is no host write to a texture, so a pass that loads Keep is given
	// something to keep by a recorded copy ahead of it.
	if prior.Buffer != nil {
		r.CopyBufferToTexture(target, prior)
	}
	p := r.RenderPass(desc)
	p.SetPipeline(pipe)
	p.SetVertexBuffer(0, vv)
	if f.indices != nil {
		packed := make([]float32, len(f.indices))
		for i, v := range f.indices {
			packed[i] = math.Float32frombits(v)
		}
		_, iv := newBuf("indices", len(packed))
		if err := q.WriteBuffer(iv.Buffer, 0, packed); err != nil {
			t.Fatalf("write indices: %v", err)
		}
		p.SetIndexBuffer(iv, accel.Index32)
	}
	if f.record != nil {
		f.record(p)
	}
	f.draws(p)

	g, err := r.Build()
	if err != nil {
		skipUnlessLowered(t, err)
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	return readColourTexture(t, d, target)
}

// posOnly is the layout for a stage that reads only a position.
func posOnly() []accel.VertexBufferLayout {
	return []accel.VertexBufferLayout{{
		Stride: 12,
		Attributes: []accel.VertexAttribute{
			{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
		},
	}}
}

func renderWithUniforms(t *testing.T, d *accel.Device) []float32 {
	return runFixture(t, d, renderFixture{
		desc: accel.RenderPipelineDescriptor{
			Vertex:        &testkernels.ScaledVSStage,
			Fragment:      &testkernels.TintedFSStage,
			VertexBuffers: posOnly(),
			Targets:       []accel.ColorTargetState{{Format: accel.RGBA32Float}},
			Label:         "uniforms",
		},
		verts: []float32{-1, -1, 0, 1, -1, 0, -1, 1, 0},
		record: func(p *accel.RenderPass) {
			// A scale and an offset, so both fields of the std140 block have to
			// land where the MSL struct expects them. A layout that packed
			// Offset at the wrong byte moves the triangle rather than losing it.
			p.SetVertexUniform(0, testkernels.StageTransform{
				Scale: 0.75, Offset: accel.Vec2{0.25, -0.125},
			})
			p.SetFragmentUniform(0, testkernels.StageTint{
				Colour: accel.Vec4{0.2, 0.4, 0.6, 1},
			})
		},
		draws: func(p *accel.RenderPass) { p.Draw(accel.Draw{VertexCount: 3}) },
	})
}

func renderWithDepth(t *testing.T, d *accel.Device) []float32 {
	return runFixture(t, d, renderFixture{
		desc: accel.RenderPipelineDescriptor{
			Vertex:        &testkernels.AttributeVSStage,
			Fragment:      &testkernels.TintFSStage,
			VertexBuffers: interleavedLayout(),
			DepthStencil: &accel.DepthStencilState{
				Format: accel.Depth32Float, Test: true, Write: true,
				Compare: accel.CompareLess,
			},
			Targets: []accel.ColorTargetState{{Format: accel.RGBA32Float}},
			Label:   "depth",
		},
		depth: true,
		// Two overlapping triangles at different depths. The nearer one must
		// win wherever they overlap, which is what a wrong compare constant
		// inverts.
		verts: []float32{
			-1, -1, 0.5, 1, 0, 0, 1,
			1, -1, 0.5, 1, 0, 0, 1,
			-1, 1, 0.5, 1, 0, 0, 1,

			-1, -1, -0.5, 0, 1, 0, 1,
			1, -1, -0.5, 0, 1, 0, 1,
			1, 1, -0.5, 0, 1, 0, 1,
		},
		draws: func(p *accel.RenderPass) { p.Draw(accel.Draw{VertexCount: 6}) },
	})
}

func renderBlended(t *testing.T, d *accel.Device) []float32 {
	return runFixture(t, d, renderFixture{
		desc: accel.RenderPipelineDescriptor{
			Vertex:        &testkernels.ScaledVSStage,
			Fragment:      &testkernels.TintedFSStage,
			VertexBuffers: posOnly(),
			Targets: []accel.ColorTargetState{{
				Format: accel.RGBA32Float, Blend: accel.AlphaBlend(),
			}},
			Label: "blended",
		},
		verts: []float32{-1, -1, 0, 3, -1, 0, -1, 3, 0},
		draws: func(p *accel.RenderPass) {
			p.SetVertexUniform(0, testkernels.StageTransform{Scale: 1})
			for _, c := range []accel.Vec4{{1, 0, 0, 0.5}, {0, 0, 1, 0.5}} {
				p.SetFragmentUniform(0, testkernels.StageTint{Colour: c})
				p.Draw(accel.Draw{VertexCount: 3})
			}
		},
	})
}

func renderIndexed(t *testing.T, d *accel.Device) []float32 {
	return runFixture(t, d, renderFixture{
		desc: accel.RenderPipelineDescriptor{
			Vertex:        &testkernels.AttributeVSStage,
			Fragment:      &testkernels.TintFSStage,
			VertexBuffers: interleavedLayout(),
			Targets:       []accel.ColorTargetState{{Format: accel.RGBA32Float}},
			Label:         "indexed",
		},
		verts: []float32{
			-1, -1, 0, 1, 0, 0, 1,
			1, -1, 0, 0, 1, 0, 1,
			1, 1, 0, 0, 0, 1, 1,
			-1, 1, 0, 1, 1, 0, 1,
		},
		indices: []uint32{0, 1, 2, 0, 2, 3},
		draws: func(p *accel.RenderPass) {
			p.DrawIndexed(accel.DrawIndexed{IndexCount: 6})
		},
	})
}

func renderKeeping(t *testing.T, d *accel.Device) []float32 {
	const w, h = 16, 16
	prior := make([]float32, w*h*4)
	for i := range prior {
		prior[i] = float32(i%7) / 7
	}
	return runFixture(t, d, renderFixture{
		desc: accel.RenderPipelineDescriptor{
			Vertex:        &testkernels.ScaledVSStage,
			Fragment:      &testkernels.TintedFSStage,
			VertexBuffers: posOnly(),
			Targets:       []accel.ColorTargetState{{Format: accel.RGBA32Float}},
			Label:         "keep",
		},
		// A small triangle, so most of the target is prior contents and the
		// comparison is mostly about what LoadKeep preserved. On Metal that is
		// a blit into the texture before the pass; on the CPU it is nothing at
		// all, which is exactly the asymmetry worth comparing.
		keep:  prior,
		verts: []float32{-1, -1, 0, 1, -1, 0, -1, 1, 0},
		record: func(p *accel.RenderPass) {
			p.SetVertexUniform(0, testkernels.StageTransform{Scale: 0.25})
			p.SetFragmentUniform(0, testkernels.StageTint{Colour: accel.Vec4{1, 1, 1, 1}})
		},
		draws: func(p *accel.RenderPass) { p.Draw(accel.Draw{VertexCount: 3}) },
	})
}

func interleavedLayout() []accel.VertexBufferLayout {
	return []accel.VertexBufferLayout{{
		Stride: 28,
		Attributes: []accel.VertexAttribute{
			{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
			{Location: 1, Format: accel.AttrFloat32x4, Offset: 12},
		},
	}}
}

// A pass with a depth attachment whose pipeline does not test depth.
//
// Metal's default is Always with writes off, and this backend spells that
// rather than leaving the state unset — so the two backends agree when a pass
// carries depth that a draw ignores. Left unset, Metal would inherit whatever
// the previous encoder left, which is a difference that appears only in a
// second pass.
func renderUntestedDepth(t *testing.T, d *accel.Device) []float32 {
	return runFixture(t, d, renderFixture{
		desc: accel.RenderPipelineDescriptor{
			Vertex:        &testkernels.AttributeVSStage,
			Fragment:      &testkernels.TintFSStage,
			VertexBuffers: interleavedLayout(),
			DepthStencil: &accel.DepthStencilState{
				Format: accel.Depth32Float, Test: false, Write: false,
				Compare: accel.CompareLess,
			},
			Targets: []accel.ColorTargetState{{Format: accel.RGBA32Float}},
			Label:   "untested depth",
		},
		depth: true,
		// The second triangle is behind the first and must still win, because
		// nothing is testing depth.
		verts: []float32{
			-1, -1, -0.5, 1, 0, 0, 1,
			1, -1, -0.5, 1, 0, 0, 1,
			-1, 1, -0.5, 1, 0, 0, 1,

			-1, -1, 0.5, 0, 1, 0, 1,
			1, -1, 0.5, 0, 1, 0, 1,
			-1, 1, 0.5, 0, 1, 0, 1,
		},
		draws: func(p *accel.RenderPass) { p.Draw(accel.Draw{VertexCount: 6}) },
	})
}

// Two draws through one pipeline, which is the case the backend's pipeline
// cache exists for: a plan replayed compiles nothing, and two draws of one
// pipeline compile it once.
func renderTwice(t *testing.T, d *accel.Device) []float32 {
	return runFixture(t, d, renderFixture{
		desc: accel.RenderPipelineDescriptor{
			Vertex:        &testkernels.ScaledVSStage,
			Fragment:      &testkernels.TintedFSStage,
			VertexBuffers: posOnly(),
			Targets:       []accel.ColorTargetState{{Format: accel.RGBA32Float}},
			Label:         "shared",
		},
		verts: []float32{-1, -1, 0, 1, -1, 0, -1, 1, 0},
		draws: func(p *accel.RenderPass) {
			for _, c := range []struct {
				scale float32
				tint  accel.Vec4
			}{{1, accel.Vec4{1, 0, 0, 1}}, {0.5, accel.Vec4{0, 1, 0, 1}}} {
				p.SetVertexUniform(0, testkernels.StageTransform{Scale: c.scale})
				p.SetFragmentUniform(0, testkernels.StageTint{Colour: c.tint})
				p.Draw(accel.Draw{VertexCount: 3})
			}
		},
	})
}

// LoadDontCare is where the oracle claim stops, and this is the shape of the
// gap.
//
// specs/033-render-api.md says DontCare leaves an attachment's prior contents
// undefined. Both backends honour that and they honour it differently: the CPU
// framebuffer aliases the caller's buffer, so untouched pixels keep what was
// there, while Metal renders into a fresh texture whose untouched pixels are
// whatever that memory held and blits the whole thing back. Neither is wrong,
// and comparing them there would be comparing two legal answers.
//
// So this asserts what the contract does promise -- the pixels the pass wrote
// agree exactly -- and asserts that the untouched region is *not* required to
// agree, by observing that on this pair it does not. That second half is the
// part worth having: a backend that quietly started preserving contents would
// be making a promise the API does not, and the next backend would then have to
// keep it.
func TestLoadDontCareAgreesOnlyWhereThePassWrote(t *testing.T) {
	metal := openMetalDevice(t)
	cpu, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer cpu.Close()

	onCPU := renderDontCare(t, cpu)
	onMetal := renderDontCare(t, metal)

	tint := [4]float32{1, 1, 1, 1}
	var written, agreed, untouchedDiffer int
	for i := 0; i+3 < len(onCPU); i += 4 {
		c := [4]float32{onCPU[i], onCPU[i+1], onCPU[i+2], onCPU[i+3]}
		m := [4]float32{onMetal[i], onMetal[i+1], onMetal[i+2], onMetal[i+3]}
		if c == tint {
			written++
			if m == tint {
				agreed++
			} else {
				t.Errorf("pixel %d was written by the pass and is %v on the CPU and %v "+
					"on Metal; what a pass writes is defined and must agree", i/4, c, m)
			}
			continue
		}
		if c != m {
			untouchedDiffer++
		}
	}
	if written == 0 {
		t.Fatal("the pass wrote nothing, so agreeing about what it wrote proves nothing")
	}
	if untouchedDiffer == 0 {
		t.Errorf("every untouched pixel matched across the two backends, so one of them " +
			"is preserving contents that specs/033-render-api.md leaves undefined; that " +
			"is a promise the API does not make and the next backend would inherit")
	}
	t.Logf("%d pixels written and agreed, %d untouched pixels differ as the contract "+
		"permits", agreed, untouchedDiffer)
}

func renderDontCare(t *testing.T, d *accel.Device) []float32 {
	const w, h = 16, 16
	prior := make([]float32, w*h*4)
	for i := range prior {
		prior[i] = float32(i%5) / 5
	}
	return runFixture(t, d, renderFixture{
		desc: accel.RenderPipelineDescriptor{
			Vertex:        &testkernels.ScaledVSStage,
			Fragment:      &testkernels.TintedFSStage,
			VertexBuffers: posOnly(),
			Targets:       []accel.ColorTargetState{{Format: accel.RGBA32Float}},
			Label:         "dontcare",
		},
		keep:     prior,
		dontCare: true,
		verts:    []float32{-1, -1, 0, 1, -1, 0, -1, 1, 0},
		record: func(p *accel.RenderPass) {
			p.SetVertexUniform(0, testkernels.StageTransform{Scale: 0.25})
			p.SetFragmentUniform(0, testkernels.StageTint{Colour: accel.Vec4{1, 1, 1, 1}})
		},
		draws: func(p *accel.RenderPass) { p.Draw(accel.Draw{VertexCount: 3}) },
	})
}

// StoreDiscard means the attachment is not written back, and this is where that
// becomes observable.
//
// The field was accepted and read by nothing until this test existed, which is
// the failure mode specs/009-sequencing.md records three times over: a value a
// caller supplies, the API documents, and no backend reads. It manufactures
// confidence — a caller discarding a depth buffer every frame believes they are
// saving a write.
//
// On Metal the saving is real: the blit back is skipped. On the CPU there is
// nothing to skip, because the framebuffer already aliases the caller's buffer.
// So the two differ again, legally, and the assertion is the same shape as the
// DontCare one: what the contract promises is that the contents are undefined,
// and observing them differ is what proves the promise is being used.
func TestStoreDiscardSkipsTheWriteBack(t *testing.T) {
	const w, h = 8, 8
	metal := openMetalDevice(t)

	q := metal.Queue()
	usage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	target := colourTexture(t, metal, "discarded", w, h)
	verts, err := metal.NewBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: 9, Usage: usage, Label: "verts",
	})
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer verts.Close()
	if err := q.WriteBuffer(verts, 0, []float32{-1, -1, 0, 3, -1, 0, -1, 3, 0}); err != nil {
		t.Fatalf("write: %v", err)
	}

	// A recognisable pattern, so what survives says whether anything was
	// written back. It reaches the texture through a recorded copy, because
	// there is no host write to one.
	prior := make([]float32, w*h*4)
	for i := range prior {
		prior[i] = 0.125
	}
	priorBuf, err := metal.NewBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: w * h * 4, Usage: usage, Label: "prior",
	})
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer priorBuf.Close()
	if err := q.WriteBuffer(priorBuf, 0, prior); err != nil {
		t.Fatalf("write prior: %v", err)
	}
	pv, err := priorBuf.View(0, priorBuf.Count())
	if err != nil {
		t.Fatalf("view: %v", err)
	}

	pipe, err := metal.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:        &testkernels.ScaledVSStage,
		Fragment:      &testkernels.TintedFSStage,
		VertexBuffers: posOnly(),
		Targets:       []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		Label:         "discard",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	tv := wholeOf(t, target)
	vv, err := verts.View(0, verts.Count())
	if err != nil {
		t.Fatalf("view: %v", err)
	}

	r := metal.NewRecorder()
	r.CopyBufferToTexture(target, pv)
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{
			View: tv, Load: accel.LoadClear, Clear: [4]float32{1, 0, 0, 1},
			Store: accel.StoreDiscard,
		}},
		Width: w, Height: h, Label: "discard",
	})
	p.SetPipeline(pipe)
	p.SetVertexBuffer(0, vv)
	p.SetVertexUniform(0, testkernels.StageTransform{Scale: 1})
	p.SetFragmentUniform(0, testkernels.StageTint{Colour: accel.Vec4{0, 1, 0, 1}})
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		skipUnlessLowered(t, err)
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	out := readColourTexture(t, metal, target)
	// The target is untouched, so it still holds the pattern written before the
	// pass. Asserted exactly rather than as "not what the draw wrote", because
	// the store action alone already makes the contents undefined -- so a
	// backend that kept the blit would copy garbage, which is also not what the
	// draw wrote and would pass a weaker assertion. Nothing written back is the
	// claim, and an intact image is what that looks like.
	for i, v := range out {
		if v != 0.125 {
			t.Fatalf("float %d is %v and the target held 0.125 before the pass; "+
				"StoreDiscard did not skip the write back", i, v)
		}
	}
}
