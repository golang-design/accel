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
	const w, h = 64, 64
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
	const w, h = 64, 64
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
	const w, h = 64, 64
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
	const w, h = 64, 64
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
	const w, h = 64, 64
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

// Metal renders in the format the attachment declares.
//
// The Metal path hardcoded RGBA32Float for every colour attachment until
// specs/045-texture-attachments.md landed, so a caller who declared RGBA8Unorm
// got a pipeline that compiled, rendered, and read back sixteen bytes per pixel
// where they had asked for four.
//
// What this asserts is the *format*, and only that: a full-viewport clear, no
// geometry. That is deliberate. A first version drew a triangle and compared
// against the CPU oracle, and it failed for both formats with 116 pixels
// differing in a clean edge pattern -- Metal and the reference rasterizer
// disagree about coverage at a triangle edge, which is a fill-rule question
// specs/035-cpu-rasterizer.md section 10 leaves open and which has nothing to
// do with the format mapping this change made. Testing the two together would
// have made a format test that fails for a reason it does not name.
//
// The extent is 4x4 and that is load-bearing. An RGBA32Float row is 64 bytes
// there and an RGBA8Unorm row is 16, both far below the 256-byte alignment a
// device reports for an attachment buffer -- so this is the extent at which
// Metal's write-back used to put every row after the first in the wrong place,
// losing three quarters of the image. A larger extent passes either way, which
// is exactly why the small one is the one to keep.
//
// A full-screen triangle covers every pixel, so no pixel sits on an edge and
// the fill rule cannot enter. The colour is what SolidFS writes, and each of
// its components is exact in f32 and in 8-bit unorm, so a disagreement is a
// format fault rather than a rounding one -- a path still assuming RGBA32Float
// reads four bytes of one f32 as four channels and returns nothing like it.
func TestMetalRendersEachAttachmentFormat(t *testing.T) {
	const w, h = 4, 4
	// What SolidFS writes. The oracle and Metal are compared against each
	// other rather than against these numbers, because 0.25 is not exact in
	// 8-bit unorm -- it quantises to 64/255 -- and asserting the literal would
	// be asserting a rounding rule this test is not about.
	solid := [4]float32{0.25, 0.5, 0.75, 1}

	for _, f := range []accel.Format{accel.RGBA32Float, accel.RGBA8Unorm} {
		t.Run(f.String(), func(t *testing.T) {
			read := func(t *testing.T, d *accel.Device) []float32 {
				t.Helper()
				tex, err := d.NewTexture(accel.TextureDescriptor{
					Format: f, Size: accel.Extent{Width: w, Height: h},
					Usage: accel.TextureRenderTarget | accel.TextureCopySrc |
						accel.TextureCopyDst,
					Kind: accel.MemoryReadback, Label: "target",
				})
				if err != nil {
					t.Fatalf("texture: %v", err)
				}
				defer tex.Close()

				pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
					Vertex:   &testkernels.FullScreenVSStage,
					Fragment: &testkernels.SolidFSStage,
					Targets:  []accel.ColorTargetState{{Format: f}},
					Label:    "format",
				})
				if err != nil {
					t.Fatalf("pipeline: %v", err)
				}
				defer pipe.Close()

				r := d.NewRecorder()
				p := r.RenderPass(accel.RenderPassDescriptor{
					Color: []accel.ColorAttachment{{
						View: wholeOf(t, tex), Load: accel.LoadClear,
						Clear: [4]float32{0, 0, 0, 0},
					}},
					Width: w, Height: h, Label: "format",
				})
				p.SetPipeline(pipe)
				p.Draw(accel.Draw{VertexCount: 3})
				g, err := r.Build()
				if err != nil {
					t.Fatalf("build: %v", err)
				}
				defer g.Close()
				if err := d.Queue().Submit(g).Wait(); err != nil {
					t.Fatalf("submit: %v", err)
				}
				return readAttachment(t, d, tex, f)
			}

			cpu, err := accel.OpenCPU(accel.CPUOptions{})
			if err != nil {
				t.Fatalf("OpenCPU: %v", err)
			}
			defer cpu.Close()

			want := read(t, cpu)
			got := read(t, openMetalDevice(t))
			if len(got) != len(want) {
				t.Fatalf("Metal returned %d floats and the oracle %d; a wrong format is "+
					"a wrong byte count before it is a wrong value", len(got), len(want))
			}
			// Both must equal the clear value, so this cannot pass by the two
			// backends agreeing on something wrong.
			for i := range want {
				if want[i] != solid[i%4] && f == accel.RGBA32Float {
					t.Fatalf("the oracle's element %d is %v, want SolidFS's %v",
						i, want[i], solid[i%4])
				}
				if got[i] != want[i] {
					t.Fatalf("element %d (pixel %d of %d) is %v on Metal and %v on the "+
						"oracle", i, i/4, w*h, got[i], want[i])
				}
			}
		})
	}
}

// readAttachment reads a texture back as normalised floats, whatever its
// format. readColourTexture above decodes f32 only, which is all the
// differential needed while every attachment was RGBA32Float.
func readAttachment(t *testing.T, d *accel.Device, tex *accel.Texture, f accel.Format) []float32 {
	t.Helper()
	sz := tex.Size()
	raw := make([]byte, sz.Width*sz.Height*f.BytesPerPixel())
	if err := d.Queue().ReadTexture(tex, raw); err != nil {
		t.Fatalf("read texture: %v", err)
	}
	switch f {
	case accel.RGBA32Float:
		out := make([]float32, len(raw)/4)
		for i := range out {
			out[i] = math.Float32frombits(uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 |
				uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24)
		}
		return out
	case accel.RGBA8Unorm:
		// v/255 exactly, which is the one obvious definition and the reason
		// specs/045-texture-attachments.md admits this format at all.
		out := make([]float32, len(raw))
		for i, b := range raw {
			out[i] = float32(b) / 255
		}
		return out
	}
	t.Fatalf("this test has no decoder for %v", f)
	return nil
}

// A depth attachment reaches Metal, and the depth test decides the same pixels
// on both backends.
//
// Nothing exercised the depth branch of the Metal backend's attachment
// allocation: the render differential draws one triangle with no depth, so a
// depth attachment on Metal was allocated by code no test reached. Found by
// reading a coverage report rather than by a failure, which is the point of
// having the gate at all.
//
// Two overlapping triangles at different depths, the nearer one drawn second.
// With the depth test on, the nearer wins where they overlap; without it, draw
// order would decide and both backends would still agree -- so the assertion is
// against the *oracle*, and the ordering is what makes the case non-trivial.
func TestADepthAttachmentAgreesOnBothBackends(t *testing.T) {
	const w, h = 64, 64

	render := func(t *testing.T, d *accel.Device) []float32 {
		t.Helper()
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
			DepthStencil: &accel.DepthStencilState{
				Format: accel.Depth32Float, Test: true, Write: true,
				Compare: accel.CompareLess,
			},
			Label: "depth",
		})
		if err != nil {
			t.Fatalf("pipeline: %v", err)
		}
		defer pipe.Close()

		colour, err := d.NewTexture(accel.TextureDescriptor{
			Format: accel.RGBA32Float, Size: accel.Extent{Width: w, Height: h},
			Usage: accel.TextureRenderTarget | accel.TextureCopySrc | accel.TextureCopyDst,
			Kind:  accel.MemoryReadback, Label: "colour",
		})
		if err != nil {
			t.Fatalf("colour: %v", err)
		}
		defer colour.Close()

		depth, err := d.NewTexture(accel.TextureDescriptor{
			Format: accel.Depth32Float, Size: accel.Extent{Width: w, Height: h},
			Usage: accel.TextureRenderTarget | accel.TextureCopySrc | accel.TextureCopyDst,
			Kind:  accel.MemoryDevice, Label: "depth",
		})
		if err != nil {
			t.Fatalf("depth: %v", err)
		}
		defer depth.Close()

		// Far first, near second: the depth test must let the near one through.
		// Position then tint, interleaved: far is red, near is green, so which
		// one won is readable from the pixel rather than inferred.
		far := []float32{
			-1, -1, 0.8, 1, 0, 0, 1,
			1, -1, 0.8, 1, 0, 0, 1,
			-1, 1, 0.8, 1, 0, 0, 1,
		}
		near := []float32{
			-1, -1, 0.2, 0, 1, 0, 1,
			1, -1, 0.2, 0, 1, 0, 1,
			-1, 1, 0.2, 0, 1, 0, 1,
		}
		verts := append(append([]float32{}, far...), near...)
		vb, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: len(verts),
			Usage: accel.BufferVertex | accel.BufferCopyDst, Label: "verts",
		})
		if err != nil {
			t.Fatalf("buffer: %v", err)
		}
		defer vb.Close()
		if err := d.Queue().WriteBuffer(vb, 0, verts); err != nil {
			t.Fatalf("write: %v", err)
		}
		vv, err := vb.View(0, vb.Count())
		if err != nil {
			t.Fatalf("view: %v", err)
		}

		r := d.NewRecorder()
		p := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{
				View: wholeOf(t, colour), Load: accel.LoadClear,
				Clear: [4]float32{0, 0, 0, 1},
			}},
			Depth: &accel.DepthAttachment{
				View: wholeOf(t, depth), Load: accel.LoadClear, Clear: 1,
			},
			Width: w, Height: h, Label: "depth",
		})
		p.SetPipeline(pipe)
		p.SetVertexBuffer(0, vv)
		p.Draw(accel.Draw{VertexCount: 3})
		p.Draw(accel.Draw{VertexCount: 3, FirstVertex: 3})

		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		if err := d.Queue().Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		return readAttachment(t, d, colour, accel.RGBA32Float)
	}

	cpu, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer cpu.Close()

	want := render(t, cpu)
	got := render(t, openMetalDevice(t))
	var drawn int
	for i := range want {
		if want[i] != 0 {
			drawn++
		}
		if got[i] != want[i] {
			t.Fatalf("element %d (pixel %d) is %v on Metal and %v on the oracle",
				i, i/4, got[i], want[i])
		}
	}
	if drawn == 0 {
		t.Fatal("the oracle drew nothing, so the comparison is vacuous")
	}
}

// A stage that fetches a texture must fetch what is bound to it, on Metal as on
// the oracle.
//
// specs/045-texture-attachments.md section 4 owns the binding and
// specs/032-stage-abi.md section 5 owns the fetch. The two halves met only on
// the CPU: build.go filled driver.RenderDraw.VertexTextures/FragmentTextures,
// internal/cpu read them, and internal/metal read them nowhere and refused
// nothing -- so RenderPass.SetTexture was accepted, validated against the
// stage's declared slots, checked for feedback, and dropped. A caller got the
// right picture from the rasterizer and a black one from the GPU.
//
// # Why the assertion is per device rather than across the two
//
// The source pass interpolates, so its image is equal across backends only
// within a bound. The fetch is not: BlitFS reads its own pixel's texel, so the
// second pass must reproduce the first *exactly*, on whichever backend ran it.
// That makes the discriminating comparison a same-device one, and an unbound
// texture fails it as a black image against a gradient rather than as a
// rounding.
//
// # Why 12 wide
//
// 12 texels of RGBA32Float is 192 bytes, and both backends align a copy's rows
// to 256. A width whose tight pitch is already aligned passes at the wrong
// pitch, because there is no padding to get wrong -- and the pitch is the one
// number a buffer-shaped texture makes this backend compute rather than read.
func TestAStageFetchAgreesOnBothBackends(t *testing.T) {
	const w, h = 12, 5
	metal := openMetalDevice(t)
	cpu, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer cpu.Close()

	// A gradient rather than a solid colour: a fetch that lands on the wrong
	// row, or that is dropped for zeros, is a different picture from this one,
	// and is not from a constant.
	verts := []float32{
		-1, -1, 0, 1, 0, 0, 1,
		3, -1, 0, 0, 1, 0, 1,
		-1, 3, 0, 0, 0, 1, 1,
	}

	render := func(t *testing.T, d *accel.Device) (source, fetched []float32) {
		t.Helper()
		q := d.Queue()
		tint, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
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
			Label:   "source",
		})
		if err != nil {
			t.Fatalf("source pipeline: %v", err)
		}
		defer tint.Close()

		blit, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
			Vertex:   &testkernels.FullScreenVSStage,
			Fragment: &testkernels.BlitFSStage,
			Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
			Label:    "fetch",
		})
		if err != nil {
			t.Fatalf("fetch pipeline: %v", err)
		}
		defer blit.Close()

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
		vv, err := vb.View(0, vb.Count())
		if err != nil {
			t.Fatalf("view: %v", err)
		}

		first := sampledColourTexture(t, d, "first", w, h)
		second := colourTexture(t, d, "second", w, h)

		r := d.NewRecorder()
		p1 := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{
				View: wholeOf(t, first), Load: accel.LoadClear,
			}},
			Width: w, Height: h, Label: "source",
		})
		p1.SetPipeline(tint)
		p1.SetVertexBuffer(0, vv)
		p1.Draw(accel.Draw{VertexCount: 3})

		p2 := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{
				View: wholeOf(t, second), Load: accel.LoadClear,
			}},
			Width: w, Height: h, Label: "fetch",
		})
		p2.SetPipeline(blit)
		p2.SetTexture(0, wholeOf(t, first))
		p2.Draw(accel.Draw{VertexCount: 3})

		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		if err := q.Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		return readColourTexture(t, d, first), readColourTexture(t, d, second)
	}

	for _, c := range []struct {
		name string
		dev  *accel.Device
	}{{"cpu", cpu}, {"metal", metal}} {
		t.Run(c.name, func(t *testing.T) {
			source, fetched := render(t, c.dev)
			var nonZero, differing int
			for i := range source {
				if source[i] != 0 {
					nonZero++
				}
				if fetched[i] != source[i] {
					differing++
					if differing < 4 {
						px := i / 4
						t.Errorf("pixel (%d,%d) channel %d is %v in the source pass and "+
							"%v after the fetch; a fetch reproduces its texel exactly",
							px%w, px/w, i%4, source[i], fetched[i])
					}
				}
			}
			if differing > 0 {
				t.Errorf("%d of %d floats differ", differing, len(source))
			}
			// A blank source would make the comparison two black images, which
			// agree perfectly and prove nothing about the binding.
			if nonZero == 0 {
				t.Fatal("the source pass drew nothing, so the fetch has nothing to prove")
			}
		})
	}
}

// sampledColourTexture is colourTexture plus the usage a stage fetch needs.
func sampledColourTexture(t *testing.T, d *accel.Device, label string, w, h int) *accel.Texture {
	t.Helper()
	tex, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.RGBA32Float, Size: accel.Extent{Width: w, Height: h},
		Usage: accel.TextureRenderTarget | accel.TextureSampled |
			accel.TextureCopySrc | accel.TextureCopyDst,
		Kind: accel.MemoryReadback, Label: label,
	})
	if err != nil {
		t.Fatalf("texture %s: %v", label, err)
	}
	t.Cleanup(func() { _ = tex.Close() })
	return tex
}

// A vertex stage's texture is bound too, and it is a different argument table.
//
// specs/032-stage-abi.md section 5 gives each stage its own dense texture
// order, and Metal has one argument table per stage: setVertexTexture: and
// setFragmentTexture: are different calls, so a backend can bind one, drop the
// other, or transpose them. DisplacedVS reads its three clip positions from a
// texture, so an unbound vertex texture is three zero positions -- a degenerate
// triangle covering nothing -- and that is a failure this can see.
//
// # Why the lookup table is painted a column at a time
//
// The obvious source, one gradient triangle, cannot work: an interpolated
// colour along one row is a convex combination of two vertex colours, so every
// texel in that row lies on a line, and three collinear positions are as
// degenerate as three zero ones. Painting each column with a by-value colour
// puts an arbitrary position in each texel instead, and RGBA32Float stores it
// unclamped, so the table holds exactly the full-screen triangle wanted.
//
// # Why coverage rather than colour
//
// SampledFS truncates an interpolated varying to an integer and fetches at it,
// so one ulp of difference in a barycentric weight is a *different texel*
// rather than a rounder colour. Nothing requires two rasterizers to compute
// those weights identically, so comparing colours would assert something no
// spec promises. The covered pixel count is an integer both must agree on, and
// it is what the vertex fetch decides.
func TestAVertexStageFetchAgreesOnBothBackends(t *testing.T) {
	const lutW, lutH = 3, 5
	const w, h = 32, 32
	metal := openMetalDevice(t)
	cpu, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer cpu.Close()

	// A quad in the space ScaledVS maps: x is scaled to one column's width and
	// y by the same factor, so the y extent is pre-divided to survive it.
	quad := []float32{
		-1, -3, 0, 1, -3, 0, 1, 3, 0,
		-1, -3, 0, 1, 3, 0, -1, 3, 0,
	}
	// One column each, and the colour written is the clip position the vertex
	// stage will read back out of that texel.
	columns := []struct {
		centre float32
		pos    accel.Vec4
	}{
		{-2.0 / 3.0, accel.Vec4{-1, -1, 0, 1}},
		{0, accel.Vec4{3, -1, 0, 1}},
		{2.0 / 3.0, accel.Vec4{-1, 3, 0, 1}},
	}

	render := func(t *testing.T, d *accel.Device) []float32 {
		t.Helper()
		q := d.Queue()
		paint, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
			Vertex:   &testkernels.ScaledVSStage,
			Fragment: &testkernels.TintedFSStage,
			VertexBuffers: []accel.VertexBufferLayout{{
				Stride:     12,
				Attributes: []accel.VertexAttribute{{Location: 0, Format: accel.AttrFloat32x3}},
			}},
			Targets: []accel.ColorTargetState{{Format: accel.RGBA32Float}},
			Label:   "paint",
		})
		if err != nil {
			t.Fatalf("paint pipeline: %v", err)
		}
		defer paint.Close()

		// Both stages fetch slot 0, which is what makes a transposed pair of
		// setters visible: the vertex stage would read the fragment's table.
		displaced, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
			Vertex:   &testkernels.DisplacedVSStage,
			Fragment: &testkernels.SampledFSStage,
			Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
			Label:    "displaced",
		})
		if err != nil {
			t.Fatalf("displaced pipeline: %v", err)
		}
		defer displaced.Close()

		usage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
		vb, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: len(quad), Usage: usage, Label: "quad",
		})
		if err != nil {
			t.Fatalf("buffer: %v", err)
		}
		defer vb.Close()
		if err := q.WriteBuffer(vb, 0, quad); err != nil {
			t.Fatalf("write: %v", err)
		}
		vv, err := vb.View(0, vb.Count())
		if err != nil {
			t.Fatalf("view: %v", err)
		}

		lut := sampledColourTexture(t, d, "lut", lutW, lutH)
		out := colourTexture(t, d, "out", w, h)

		r := d.NewRecorder()
		for i, c := range columns {
			load := accel.LoadKeep
			if i == 0 {
				load = accel.LoadClear
			}
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: wholeOf(t, lut), Load: load}},
				Width: lutW, Height: lutH, Label: "paint",
			})
			p.SetPipeline(paint)
			p.SetVertexBuffer(0, vv)
			p.SetVertexUniform(0, testkernels.StageTransform{
				Scale: 1.0 / 3.0, Offset: accel.Vec2{c.centre, 0},
			})
			p.SetFragmentUniform(0, testkernels.StageTint{Colour: c.pos})
			p.Draw(accel.Draw{VertexCount: 6})
		}

		p := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{View: wholeOf(t, out), Load: accel.LoadClear}},
			Width: w, Height: h, Label: "displaced",
		})
		p.SetPipeline(displaced)
		p.SetTexture(0, wholeOf(t, lut))
		p.Draw(accel.Draw{VertexCount: 3})

		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		if err := q.Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		return readColourTexture(t, d, out)
	}

	covered := func(px []float32) int {
		n := 0
		for i := 0; i < len(px); i += 4 {
			if px[i] != 0 || px[i+1] != 0 || px[i+2] != 0 || px[i+3] != 0 {
				n++
			}
		}
		return n
	}

	onCPU := covered(render(t, cpu))
	onMetal := covered(render(t, metal))
	// The positions come from the texture, so nothing drawn means nothing was
	// bound -- and two blank images agree with each other perfectly.
	if onCPU == 0 {
		t.Fatal("the CPU rasterizer covered nothing, so the vertex fetch placed no triangle")
	}
	if onMetal == 0 {
		t.Fatal("Metal covered nothing: the vertex stage read no texture, so its three " +
			"positions were zero and the triangle was degenerate")
	}
	if onCPU != onMetal {
		t.Errorf("the triangle covers %d pixels on the CPU and %d on Metal, from the "+
			"same three positions read out of the same texture", onCPU, onMetal)
	}
	t.Logf("%d of %d pixels covered on both", onCPU, w*h)
}

// A stage texture in a format this backend cannot spell is refused by name.
//
// The staged copy needs a Metal pixel format, and internal/metal maps five.
// R32Float is a format accel offers and that map does not, so a caller who
// binds one has to be told which format is missing rather than handed a
// picture: the same rule metalPixelFormat was written for on the attachment
// path, where hardcoding RGBA32Float had produced sixteen bytes per pixel for
// a caller who asked for four.
//
// The texture is never rendered into, only fetched, because rendering into it
// would be refused first and this is an assertion about the fetch.
func TestAStageTextureFormatMetalCannotSpellIsRefused(t *testing.T) {
	const w, h = 8, 8
	d := openMetalDevice(t)

	pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.FullScreenVSStage,
		Fragment: &testkernels.BlitFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		Label:    "unspellable",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	lut, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.R32Float, Size: accel.Extent{Width: w, Height: h},
		Usage: accel.TextureSampled | accel.TextureCopySrc | accel.TextureCopyDst,
		Kind:  accel.MemoryReadback, Label: "lut",
	})
	if err != nil {
		t.Fatalf("texture: %v", err)
	}
	defer lut.Close()
	out := colourTexture(t, d, "out", w, h)

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: wholeOf(t, out), Load: accel.LoadClear}},
		Width: w, Height: h, Label: "unspellable",
	})
	p.SetPipeline(pipe)
	p.SetTexture(0, wholeOf(t, lut))
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	err = d.Queue().Submit(g).Wait()
	if err == nil {
		t.Fatal("a stage texture in R32Float was accepted; this backend has no pixel " +
			"format for it, so the fetch would have read the wrong bytes")
	}
	// The format and the owning spec, so the next caller to want one is told
	// what is missing rather than left to compare pictures.
	for _, want := range []string{"R32Float", "045-texture-attachments.md"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal does not name %q: %v", want, err)
		}
	}
}

// A discard on Metal drops the same fragments the oracle drops.
//
// specs/032-stage-abi.md section 4.2. The oracle's own assertion is portable
// and lives beside the root package's render tests, because it is a statement
// about the rasterizer; this one is the statement that the MSL lowering agrees,
// which is the only thing that makes discard_fragment() more than emitted text.
//
// Exactly, not within a bound. Every value in the picture is a clear colour or
// a by-value colour, so nothing here is interpolated and there is no rounding
// for the two rasterizers to differ over. What can differ is *which* fragments
// survived, and that is the assertion.
func TestADiscardAgreesOnBothBackends(t *testing.T) {
	const w, h = 8, 4
	metal := openMetalDevice(t)
	cpu, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer cpu.Close()

	onCPU := discardImage(t, cpu, w, h)
	onMetal := discardImage(t, metal, w, h)

	var kept, dropped int
	for i := range onCPU {
		if onMetal[i] != onCPU[i] {
			px := i / 4
			t.Fatalf("pixel (%d,%d) channel %d is %v on Metal and %v on the oracle",
				px%w, px/w, i%4, onMetal[i], onCPU[i])
		}
	}
	// Both outcomes have to be in the picture. Two images where every fragment
	// survived agree perfectly and say nothing about a discard, and so do two
	// where none did.
	for x := range w {
		if onCPU[x*4+1] == 1 {
			dropped++
		} else {
			kept++
		}
	}
	if kept == 0 || dropped == 0 {
		t.Fatalf("%d columns kept and %d discarded; the comparison needs both", kept, dropped)
	}
}

// A flat varying is flat on Metal too, and both backends pick the same vertex.
//
// specs/032-stage-abi.md section 3.1's rule is portable and its *provoking
// vertex* is not: which of a triangle's three vertices supplies a flat value is
// a convention, no spec here fixes it, and a divergence would be invisible in
// any single-backend test. That is what this compares -- the id itself, not
// only that there is one of it.
//
// It is also the first place an integer crosses the stage boundary on a device.
// The flat form carries it as bits through a []float32 on the CPU and as an
// `int` member on Metal, which are two different representations that must
// produce one number.
func TestAFlatVaryingAgreesOnBothBackends(t *testing.T) {
	const w, h = 16, 16
	metal := openMetalDevice(t)
	cpu, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer cpu.Close()

	cpuIDs, cpuCovered := flatIDs(flatImage(t, cpu, w, h), w)
	metalIDs, metalCovered := flatIDs(flatImage(t, metal, w, h), w)

	if cpuCovered == 0 || metalCovered == 0 {
		t.Fatalf("covered %d on the oracle and %d on Metal; two blank images agree",
			cpuCovered, metalCovered)
	}
	if len(metalIDs) != 1 {
		t.Fatalf("Metal produced %d distinct ids over %d covered pixels: %v -- "+
			"the [[flat]] qualifier did not reach the pipeline", len(metalIDs),
			metalCovered, metalIDs)
	}
	for id := range cpuIDs {
		if metalIDs[id] == 0 {
			t.Errorf("the flat id is %v on the oracle and %v on Metal: the two "+
				"backends disagree about which vertex is provoking, which is a "+
				"convention divergence for docs/conventions.md rather than a bug "+
				"in either", id, keysOf(metalIDs))
		}
	}
}

// keysOf names the ids in a histogram, for a diagnostic.
func keysOf(m map[float32]int) []float32 {
	out := make([]float32, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// The two interpolation qualifiers agree with the oracle on Metal.
//
// specs/032-stage-abi.md section 3.1. Both channels at once: red is
// perspective-correct and green is screen-linear over the same value, so a
// backend that applied [[center_no_perspective]] to the wrong member, to both,
// or to neither disagrees with the oracle in a channel this reads.
//
// Within a derived bound rather than exactly, for the reason the plain triangle
// differential gives: the two rasterizers are free to compute barycentric
// weights differently, and section 8 puts interpolated values in the bounded
// half of its split.
func TestTheInterpolationQualifiersAgreeOnBothBackends(t *testing.T) {
	const w, h = 32, 32
	metal := openMetalDevice(t)
	cpu, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer cpu.Close()

	onCPU := perspectiveImage(t, cpu, w, h)
	onMetal := perspectiveImage(t, metal, w, h)

	// The largest interpolated value here is 2, and a weighted sum of three
	// terms takes two roundings, so this is 4u. Nothing is tuned.
	const bound = 4 * 1.1920929e-7

	var covered, differing, qualified int
	for i := 0; i < len(onCPU); i += 4 {
		if onCPU[i+2] == 0 {
			continue
		}
		covered++
		if onCPU[i] != onCPU[i+1] {
			qualified++
		}
		for c := range 3 {
			if d := math.Abs(float64(onCPU[i+c] - onMetal[i+c])); d > bound {
				differing++
				if differing < 4 {
					px := i / 4
					t.Errorf("pixel (%d,%d) channel %d is %v on the oracle and %v on "+
						"Metal, %v apart and the derived bound is %v",
						px%w, px/w, c, onCPU[i+c], onMetal[i+c], d, bound)
				}
			}
		}
	}
	if covered == 0 {
		t.Fatal("the oracle covered nothing, so the comparison is vacuous")
	}
	// Without this the two backends could agree by both ignoring the qualifier.
	if qualified == 0 {
		t.Fatal("the two qualifiers produced the same value at every covered pixel on " +
			"the oracle, so agreeing with Metal says nothing about either")
	}
	if differing > 0 {
		t.Errorf("%d channel values differ by more than %v", differing, bound)
	}
}
