// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"slices"
	"testing"

	"golang.design/x/accel/internal/kernel"
)

// What a bound texture declares on the node, per specs/032-stage-abi.md
// section 5 and specs/045-texture-attachments.md section 3.
//
// In-package for the reason the stage-mask tests are: a declared access and its
// stage are not public values, and the mask is what the barrier between a pass
// that writes an attachment and a pass that fetches it is made of. The stages
// are hand-written here because internal/kernels imports this package.

// texturedStage is a stage record declaring one texture at slot 0.
//
// reads is StageTexture.Reads: a texture the body never fetches is a binding
// the backend still supplies and not a subresource this pass depends on.
//
// No flat adapter, for stageTestVS's reason: every test here stops at the plan,
// so nothing runs one.
func texturedStage(name string, kind kernel.StageKind, slot int, reads bool) Stage {
	s := Stage{
		Name: name, Kind: kind, Varyings: "stageTestVaryings",
		Textures: []kernel.StageTexture{{Name: "src", Index: slot, Reads: reads}},
	}
	if kind == kernel.StageVertex {
		s.Attributes = []kernel.StageAttribute{{Name: "pos", Index: 0, Components: 3}}
		return s
	}
	s.Outputs = []kernel.StageOutput{{Name: "colour", Index: 0}}
	return s
}

// texturePipeline pairs one hand-written stage with the plain other half.
func texturePipeline(t *testing.T, d *Device, vs, fs *Stage) *RenderPipeline {
	t.Helper()
	p, err := d.NewRenderPipeline(RenderPipelineDescriptor{
		Vertex: vs, Fragment: fs,
		VertexBuffers: []VertexBufferLayout{{
			Stride: 12, StepMode: StepVertex,
			Attributes: []VertexAttribute{{Location: 0, Format: AttrFloat32x3, Offset: 0}},
		}},
		Targets: []ColorTargetState{{Format: RGBA32Float}},
		Label:   "texture test",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })
	return p
}

// texturePass records one draw that binds src at slot 0 and returns the pass.
func texturePass(t *testing.T, r *Recorder, d *Device, pipe *RenderPipeline,
	target, src TextureView) *RenderPass {
	t.Helper()
	p := r.RenderPass(RenderPassDescriptor{
		Color: []ColorAttachment{{View: target}},
		Width: 4, Height: 4, Label: "fetching",
	})
	p.SetPipeline(pipe)
	p.SetVertexBuffer(0, stageTestBuffer(t, d, "vb", 9))
	p.SetTexture(0, src)
	p.Draw(Draw{VertexCount: 3})
	return p
}

// A texture a fragment stage fetches is read in the fragment shader stage.
//
// The stage is the point rather than the access: a pass that waits for a texel
// fetch waits at the shader stage that fetches, and naming the pass instead
// names its colour and depth stages too. specs/045-texture-attachments.md
// section 3 draws that edge as colour output on one side and a shader stage on
// the other, and this is the half that says which shader stage.
func TestAFetchedTextureIsReadInItsOwnShaderStage(t *testing.T) {
	d := stageTestDevice(t)
	for _, c := range []struct {
		name string
		kind kernel.StageKind
		want stage
	}{
		{"fragment", kernel.StageFragment, stageFragmentShader},
		{"vertex", kernel.StageVertex, stageVertexShader},
	} {
		t.Run(c.name, func(t *testing.T) {
			vs, fs := stageTestVS, stageTestFS
			s := texturedStage("fetching", c.kind, 0, true)
			if c.kind == kernel.StageVertex {
				vs = s
			} else {
				fs = s
			}
			r := d.NewRecorder()
			target := stageTestTarget(t, d, "target", 4, 4, RGBA32Float)
			src := stageTestTarget(t, d, "src", 4, 4, RGBA32Float)
			p := texturePass(t, r, d, texturePipeline(t, d, &vs, &fs), target, src)

			if got := stageOfTexture(t, r, p.Node(), src); got != c.want {
				t.Errorf("the fetched texture is read in %v, want %v", got, c.want)
			}
		})
	}
}

// A texture both stages declare at one slot is one access naming both stages.
//
// One slot space serves both stages because a texture is a resource rather than
// a per-stage value, so the two declarations are two readers of one
// subresource. Two accesses would inflate the hazard count a caller reads and
// make the builder compute the same barrier twice, which is exactly what
// declareRead's de-duplication exists to prevent for a buffer.
func TestATextureBothStagesFetchIsOneAccessInBothStages(t *testing.T) {
	d := stageTestDevice(t)
	vs := texturedStage("fetchingVS", kernel.StageVertex, 0, true)
	fs := texturedStage("fetchingFS", kernel.StageFragment, 0, true)

	r := d.NewRecorder()
	target := stageTestTarget(t, d, "target", 4, 4, RGBA32Float)
	src := stageTestTarget(t, d, "src", 4, 4, RGBA32Float)
	p := texturePass(t, r, d, texturePipeline(t, d, &vs, &fs), target, src)

	// stageOfTexture fails the test if the node declares more than one access
	// naming the texture, which is the de-duplication half.
	want := stageVertexShader | stageFragmentShader
	if got := stageOfTexture(t, r, p.Node(), src); got != want {
		t.Errorf("the shared texture is read in %v, want %v", got, want)
	}
}

// A texture no stage fetches does not order the graph against whatever wrote it.
//
// Two ways a bound texture is not a dependency, and both are ordinary. A slot
// no stage of this draw declares is the wide-pipeline case
// TestAnUnfetchedVertexBufferDoesNotOrderTheGraph names: a caller may bind for
// the widest pipeline in a pass and draw with a narrower one. And a declared
// texture the body never fetches is a binding the backend still supplies --
// StageTexture.Reads is inferred from the body rather than declared -- but not
// a subresource the pass reads.
//
// Declaring either would move barriers and stretch a transient's live range for
// a fetch that does not happen.
func TestAnUnfetchedTextureDoesNotOrderTheGraph(t *testing.T) {
	for _, c := range []struct {
		name string
		fs   Stage
		slot int
	}{
		{"a slot no stage declares", stageTestFS, 3},
		{"a declared texture the body never reads",
			texturedStage("unread", kernel.StageFragment, 0, false), 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			d := stageTestDevice(t)
			r := d.NewRecorder()
			target := stageTestTarget(t, d, "target", 4, 4, RGBA32Float)

			// Written by an earlier node, so an unwanted read is an edge rather
			// than nothing.
			src := stageTestTarget(t, d, "src", 4, 4, RGBA32Float)
			written := r.RenderPass(RenderPassDescriptor{
				Color: []ColorAttachment{{View: src, Load: LoadClear}},
				Width: 4, Height: 4, Label: "writer",
			})
			written.SetPipeline(stageTestPipeline(t, d))
			written.SetVertexBuffer(0, stageTestBuffer(t, d, "wvb", 9))
			written.Draw(Draw{VertexCount: 3})

			fs := c.fs
			pipe := texturePipeline(t, d, &stageTestVS, &fs)
			p := r.RenderPass(RenderPassDescriptor{
				Color: []ColorAttachment{{View: target}},
				Width: 4, Height: 4, Label: "reader",
			})
			p.SetPipeline(pipe)
			p.SetVertexBuffer(0, stageTestBuffer(t, d, "vb", 9))
			p.SetTexture(c.slot, src)
			p.Draw(Draw{VertexCount: 3})

			g, err := r.Build()
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			defer g.Close()
			for _, succ := range g.Edges()[int(written.Node())] {
				if succ == p.Node() {
					t.Fatalf("the pass that binds a texture nothing fetches depends on "+
						"the pass that wrote it; edges: %v", g.Edges())
				}
			}
		})
	}
}

// A fetched texture written by an earlier pass makes that pass a predecessor.
//
// The accepting half of the rule above, and the edge the whole feature exists
// for: a pass reads what a previous pass drew, and the graph knows.
func TestAFetchedTextureOrdersThePassAgainstItsWriter(t *testing.T) {
	d := stageTestDevice(t)
	fs := texturedStage("fetching", kernel.StageFragment, 0, true)

	r := d.NewRecorder()
	target := stageTestTarget(t, d, "target", 4, 4, RGBA32Float)
	src := stageTestTarget(t, d, "src", 4, 4, RGBA32Float)
	written := r.RenderPass(RenderPassDescriptor{
		Color: []ColorAttachment{{View: src, Load: LoadClear}},
		Width: 4, Height: 4, Label: "writer",
	})
	written.SetPipeline(stageTestPipeline(t, d))
	written.SetVertexBuffer(0, stageTestBuffer(t, d, "wvb", 9))
	written.Draw(Draw{VertexCount: 3})

	p := texturePass(t, r, d, texturePipeline(t, d, &stageTestVS, &fs), target, src)

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	found := false
	for _, succ := range g.Edges()[int(written.Node())] {
		if succ == p.Node() {
			found = true
		}
	}
	if !found {
		t.Fatalf("the pass that fetches the texture does not depend on the pass that "+
			"wrote it, so the two are free to run in either order; edges: %v", g.Edges())
	}
}

// A barrier emitted for a texture hazard names the texture.
//
// The reason is what a caller asking why a graph does not overlap reads, and
// labelOf knew buffers and slots only, so every texture hazard was "an unknown
// resource". The same recording as the test above: the barrier before the
// fetching pass exists for the write to src.
func TestATextureHazardNamesTheTexture(t *testing.T) {
	d := stageTestDevice(t)
	fs := texturedStage("fetching", kernel.StageFragment, 0, true)

	r := d.NewRecorder()
	target := stageTestTarget(t, d, "target", 4, 4, RGBA32Float)
	src := stageTestTarget(t, d, "src", 4, 4, RGBA32Float)
	written := r.RenderPass(RenderPassDescriptor{
		Color: []ColorAttachment{{View: src, Load: LoadClear}},
		Width: 4, Height: 4, Label: "writer",
	})
	written.SetPipeline(stageTestPipeline(t, d))
	written.SetVertexBuffer(0, stageTestBuffer(t, d, "wvb", 9))
	written.Draw(Draw{VertexCount: 3})
	p := texturePass(t, r, d, texturePipeline(t, d, &stageTestVS, &fs), target, src)

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	b := g.barriersBefore[p.Node()]
	if b == nil {
		t.Fatal("no barrier before the pass that fetches what the previous pass wrote")
	}
	var labels []string
	for _, reason := range b.reasons {
		labels = append(labels, reason.label)
	}
	if !slices.Contains(labels, "src") {
		t.Errorf("the barrier's reasons name %q, and none of them is the texture %q",
			labels, "src")
	}
}
