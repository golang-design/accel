// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"fmt"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

func uniformPipeline(t *testing.T, d *accel.Device) *accel.RenderPipeline {
	t.Helper()
	p, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.ScaledVSStage,
		Fragment: &testkernels.TintedFSStage,
		VertexBuffers: []accel.VertexBufferLayout{{
			Stride: 12,
			Attributes: []accel.VertexAttribute{
				{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
			},
		}},
		Targets: []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		Label:   "uniforms",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })
	return p
}

// Both stages take a by-value parameter at index 0, and each gets its own.
//
// The two parameters are different types, which is what makes this an assertion
// rather than a demonstration: a render path holding one shared slice would
// hand the fragment stage a StageTransform, and the generated adapter's type
// assertion would fail. specs/033-render-api.md deviation 1 is the bug this
// replaces, and one shared index space is exactly what it could not support.
//
// The vertex uniform is asserted through geometry and the fragment uniform
// through colour, so neither can pass by accident of the other.
func TestEachStageGetsItsOwnUniforms(t *testing.T) {
	const w, h = 16, 16
	d := openDevice(t)
	q := d.Queue()
	pipe := uniformPipeline(t, d)

	// A triangle covering the lower-left half, scaled by the vertex uniform to
	// half size about the origin. Half of half is a quarter of the viewport.
	pos := []float32{-1, -1, 0, 1, -1, 0, -1, 1, 0}
	vb := newBuffer(t, d, "pos", len(pos), accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(vb, 0, pos); err != nil {
		t.Fatalf("write: %v", err)
	}
	target := colourTarget(t, d, "colour", w, h)

	r := d.NewRecorder()
	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{
			View: view(t, target), Load: accel.LoadClear, Clear: [4]float32{0, 0, 0, 0},
		}},
		Width: w, Height: h, Label: "uniforms",
	})
	pass.SetPipeline(pipe)
	pass.SetVertexBuffer(0, whole(t, vb))
	pass.SetVertexUniform(0, testkernels.StageTransform{Scale: 0.5})
	pass.SetFragmentUniform(0, testkernels.StageTint{Colour: [4]float32{0.2, 0.4, 0.6, 1}})
	pass.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := readTarget(t, d, target)
	at := func(x, y int) [4]float32 {
		i := (y*w + x) * 4
		return [4]float32{got[i], got[i+1], got[i+2], got[i+3]}
	}

	// The fragment uniform: whatever is covered is exactly the tint.
	tint := [4]float32{0.2, 0.4, 0.6, 1}
	var covered int
	for y := range h {
		for x := range w {
			px := at(x, y)
			if px == tint {
				covered++
				continue
			}
			if px != ([4]float32{0, 0, 0, 0}) {
				t.Fatalf("pixel (%d,%d) is %v, which is neither the tint %v nor the "+
					"clear; the fragment stage received something else", x, y, px, tint)
			}
		}
	}

	// The vertex uniform: scaling clip positions by 0.5 halves each side, so
	// the covered area is a quarter of what an unscaled draw covers. Unscaled
	// is 120 of 256 under the top-left rule; a quarter of that area is 28.
	// Asserted as a band rather than a number because the fill rule decides the
	// edge pixels and the scaled triangle's edges do not land on pixel centres.
	if covered < 20 || covered > 40 {
		t.Errorf("%d pixels covered; a triangle scaled to half its size covers about a "+
			"quarter of the 120 an unscaled one does, so the vertex stage did not "+
			"receive Scale 0.5", covered)
	}
}

// Every refusal the uniform channel owns.
func TestRenderUniformRefusals(t *testing.T) {
	for _, c := range []struct {
		name   string
		record func(t *testing.T, d *accel.Device, p *accel.RenderPass, pipe *accel.RenderPipeline)
		says   string
	}{{
		name: "a stage parameter with no value",
		record: func(t *testing.T, d *accel.Device, p *accel.RenderPass, pipe *accel.RenderPipeline) {
			p.SetFragmentUniform(0, testkernels.StageTint{})
		},
		says: `ScaledVS takes "xf" at index 0 and no value was set`,
	}, {
		name: "a value of the wrong type",
		record: func(t *testing.T, d *accel.Device, p *accel.RenderPass, pipe *accel.RenderPipeline) {
			p.SetVertexUniform(0, testkernels.StageTint{})
			p.SetFragmentUniform(0, testkernels.StageTint{})
		},
		says: "ScaledVS takes StageTransform as \"xf\" at index 0 and a StageTint was set",
	}, {
		name: "a nil value",
		record: func(t *testing.T, d *accel.Device, p *accel.RenderPass, pipe *accel.RenderPipeline) {
			p.SetVertexUniform(0, nil)
		},
		says: "SetVertexUniform at index 0 has a nil value",
	}, {
		name: "a negative index",
		record: func(t *testing.T, d *accel.Device, p *accel.RenderPass, pipe *accel.RenderPipeline) {
			p.SetFragmentUniform(-1, testkernels.StageTint{})
		},
		says: "SetFragmentUniform at index -1",
	}} {
		t.Run(c.name, func(t *testing.T) {
			d := openDevice(t)
			q := d.Queue()
			pipe := uniformPipeline(t, d)
			vb := newBuffer(t, d, "pos", 9, accel.BufferStorage|accel.BufferCopyDst)
			if err := q.WriteBuffer(vb, 0, make([]float32, 9)); err != nil {
				t.Fatalf("write: %v", err)
			}
			target := colourTarget(t, d, "colour", 4, 4)

			r := d.NewRecorder()
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, target)}},
				Width: 4, Height: 4, Label: "refused",
			})
			p.SetPipeline(pipe)
			p.SetVertexBuffer(0, whole(t, vb))
			c.record(t, d, p, pipe)
			p.Draw(accel.Draw{VertexCount: 3})

			g, err := r.Build()
			if err == nil {
				_ = g.Close()
				t.Fatalf("Build accepted a recording it should refuse")
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Fatalf("the error should say %q, got %v", c.says, err)
			}
		})
	}
}

// A uniform set for a stage that declares none is a refusal, not a no-op.
//
// The caller believes a value reached the stage. Accepting it silently is how a
// misspelled intent survives to the frame that looks wrong.
func TestAUniformSetForAStageThatTakesNone(t *testing.T) {
	d := openDevice(t)
	target := colourTarget(t, d, "colour", 4, 4)

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: view(t, target)}},
		Width: 4, Height: 4, Label: "spurious",
	})
	p.SetPipeline(solidPipeline(t, d))
	p.SetVertexUniform(0, testkernels.StageTransform{})
	p.Draw(accel.Draw{VertexCount: 3})

	_, err := r.Build()
	if err == nil {
		t.Fatal("a uniform set for a stage that declares none was accepted")
	}
	if !strings.Contains(err.Error(), "HalfTriangleVS declares no by-value parameters") {
		t.Errorf("the error should name the stage, got %v", err)
	}
}

// A draw captures the uniforms set when it is recorded, so a later call does
// not reach back and change it.
//
// Two draws with different tints in one pass. Pass state that was read at build
// rather than captured at record would give both draws the second tint, and the
// first draw's pixels would be wrong in a way no single-draw test can see.
func TestUniformsAreCapturedPerDraw(t *testing.T) {
	const w, h = 8, 8
	d := openDevice(t)
	q := d.Queue()
	pipe := uniformPipeline(t, d)

	// Two triangles: the first covers the lower-left half, the second is
	// scaled to nothing so it cannot overdraw the first.
	pos := []float32{-1, -1, 0, 1, -1, 0, -1, 1, 0}
	vb := newBuffer(t, d, "pos", len(pos), accel.BufferStorage|accel.BufferCopyDst)
	if err := q.WriteBuffer(vb, 0, pos); err != nil {
		t.Fatalf("write: %v", err)
	}
	target := colourTarget(t, d, "colour", w, h)

	first := [4]float32{1, 0, 0, 1}
	second := [4]float32{0, 1, 0, 1}

	r := d.NewRecorder()
	pass := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: view(t, target), Load: accel.LoadClear}},
		Width: w, Height: h, Label: "two draws",
	})
	pass.SetPipeline(pipe)
	pass.SetVertexBuffer(0, whole(t, vb))
	pass.SetVertexUniform(0, testkernels.StageTransform{Scale: 1})
	pass.SetFragmentUniform(0, testkernels.StageTint{Colour: first})
	pass.Draw(accel.Draw{VertexCount: 3})

	// The second draw is scaled to zero, so it covers nothing and the first
	// draw's colour survives.
	pass.SetVertexUniform(0, testkernels.StageTransform{Scale: 0})
	pass.SetFragmentUniform(0, testkernels.StageTint{Colour: second})
	pass.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := q.Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	got := readTarget(t, d, target)
	i := ((h - 1) * w) * 4
	px := [4]float32{got[i], got[i+1], got[i+2], got[i+3]}
	if px != first {
		t.Errorf("pixel (0,%d) is %v, want the first draw's %v; the second draw's "+
			"uniforms reached the first", h-1, px, first)
	}
}

// A slot outside the vertex layout is a diagnostic, not a panic.
//
// It was a panic: a negative slot skipped SetVertexBuffer's grow loop and
// indexed the slice, taking the caller's process down from inside a recording
// call (render.go:588, found by a surface audit and reproduced here). Every
// other method on RenderPass reports through the recorder, and a panic is the
// one diagnostic a caller cannot handle, cannot attribute to a slot, and cannot
// see alongside the rest of a build's errors.
func TestAVertexSlotOutsideTheLayoutIsRefused(t *testing.T) {
	for _, c := range []struct {
		name string
		slot func(limit int) int
		says string
	}{
		{"negative", func(int) int { return -1 }, "cannot be negative"},
		{"at the ceiling", func(limit int) int { return limit }, "reports a limit"},
		{"far past the ceiling", func(int) int { return 4096 }, "reports a limit"},
	} {
		t.Run(c.name, func(t *testing.T) {
			d := openDevice(t)
			// The ceiling is the device's Limits.MaxVertexBuffers, which
			// TestAVertexBufferSlotIsRefusedAgainstTheDevicesLimit pins on
			// both sides; here it only has to be a diagnostic.
			slot := c.slot(d.Info().Limits.MaxVertexBuffers)
			target := colourTarget(t, d, "colour", 4, 4)
			vb := newBuffer(t, d, "vb", 9*4, accel.BufferVertex|accel.BufferCopyDst)

			r := d.NewRecorder()
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, target)}},
				Width: 4, Height: 4, Label: "slots",
			})
			// No recover(): a panic here fails the test by taking it down, which
			// is the behaviour being fixed.
			p.SetVertexBuffer(slot, whole(t, vb))

			_, err := r.Build()
			if err == nil {
				t.Fatalf("slot %d was accepted", slot)
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Fatalf("the refusal should say %q, got %v", c.says, err)
			}
			// And it names the slot, so a caller with several knows which.
			if !strings.Contains(err.Error(), fmt.Sprint(slot)) {
				t.Errorf("the refusal should name slot %d: %v", slot, err)
			}
		})
	}
}

// A vertex buffer bound at a slot the pipeline does not fetch does not order
// the graph against whatever wrote it.
//
// It did. RenderPass.record declared a read on every bound buffer while the
// lowering fetches only the slots the pipeline's layout names, so a buffer
// nobody fetched still reached the node -- moving barriers and stretching a
// transient's live range for a fetch that does not happen.
//
// Binding one is not a mistake, which is why the answer is to declare what is
// read rather than to refuse the binding: a caller may bind for the widest
// pipeline in a pass and draw with a narrower one, and each draw copies the
// state standing at the time.
func TestAnUnfetchedVertexBufferDoesNotOrderTheGraph(t *testing.T) {
	d := openDevice(t)
	target := colourTarget(t, d, "target", 4, 4)
	// Written by an earlier node, so an unwanted read of it is an edge that
	// shows up as a dependency rather than as nothing.
	stray := newBuffer(t, d, "stray", 12, accel.BufferStorage|accel.BufferCopyDst)

	r := d.NewRecorder()
	src := newBuffer(t, d, "src", 12, accel.BufferStorage|accel.BufferCopySrc)
	r.CopyBuffer(whole(t, stray), whole(t, src))

	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: view(t, target)}},
		Width: 4, Height: 4, Label: "unfetched",
	})
	p.SetPipeline(solidPipeline(t, d))
	p.SetVertexBuffer(2, whole(t, stray))
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	// solidPipeline declares no vertex buffers, so nothing fetches slot 2 and
	// the pass must not depend on the copy that wrote it.
	edges := g.Edges()
	for _, succ := range edges[0] {
		if succ == p.Node() {
			t.Fatalf("node 0 (the copy that wrote a buffer nothing fetches) has the "+
				"pass as a successor; edges: %v", edges)
		}
	}
}
