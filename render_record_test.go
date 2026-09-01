// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// solidPipeline is a pipeline nothing refuses, so a test below fails for the
// reason it names rather than for its fixture.
func solidPipeline(t *testing.T, d *accel.Device) *accel.RenderPipeline {
	t.Helper()
	p, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.HalfTriangleVSStage,
		Fragment: &testkernels.SolidFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		Label:    "solid",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })
	return p
}

// Every refusal the recorder and the builder own for a render pass.
//
// A recorder does not return an error per call: it collects them, and Build
// reports them together. Each case here is therefore a Build that must fail,
// and the message is asserted because a caller acts on which of these it is.
func TestRenderPassRefusals(t *testing.T) {
	for _, c := range []struct {
		name   string
		record func(t *testing.T, d *accel.Device, r *accel.Recorder, target *accel.Texture)
		says   string
	}{{
		name: "no colour attachments",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			r.RenderPass(accel.RenderPassDescriptor{Width: 4, Height: 4, Label: "bare"})
		},
		says: "no colour attachments",
	}, {
		name: "an area with no extent",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, b)}}, Label: "flat",
			})
		},
		says: "render area is 0x0",
	}, {
		name: "an attachment naming no resource",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{}}, Width: 4, Height: 4, Label: "empty",
			})
		},
		says: "names no resource",
	}, {
		name: "a pass with no draws",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, b)}},
				Width: 4, Height: 4, Label: "idle",
			})
		},
		says: "records no draws",
	}, {
		name: "SetPipeline with no pipeline",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, b)}},
				Width: 4, Height: 4, Label: "unset",
			})
			p.SetPipeline(nil)
		},
		says: "SetPipeline with no pipeline",
	}, {
		// Both are what a dispatch refuses of a compute pipeline. Accepted, a
		// closed pipeline would be lowered at build, and a foreign one would
		// hand another device's compiled stages to this one's backend.
		name: "SetPipeline with a closed pipeline",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, b)}},
				Width: 4, Height: 4, Label: "stale",
			})
			pipe := solidPipeline(t, d)
			if err := pipe.Close(); err != nil {
				t.Fatalf("close pipeline: %v", err)
			}
			p.SetPipeline(pipe)
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: `SetPipeline "solid": resource is closed`,
	}, {
		name: "SetPipeline with another device's pipeline",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, b)}},
				Width: 4, Height: 4, Label: "foreign",
			})
			p.SetPipeline(solidPipeline(t, openDevice(t)))
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: "belongs to a different device",
	}, {
		name: "a draw before a pipeline",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, b)}},
				Width: 4, Height: 4, Label: "early",
			})
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: "a draw with no pipeline",
	}, {
		// The indexed form took the same recording and then dereferenced the
		// missing pipeline inside Build, which took the process down where the
		// direct and indirect forms reported.
		name: "an indexed draw before a pipeline",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, b)}},
				Width: 4, Height: 4, Label: "early",
			})
			ib := newBuffer(t, d, "indices", 3, accel.BufferStorage)
			p.SetIndexBuffer(whole(t, ib), accel.Index32)
			p.DrawIndexed(accel.DrawIndexed{IndexCount: 3})
		},
		says: "an indexed draw with no pipeline",
	}, {
		name: "a draw of no vertices",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, b)}},
				Width: 4, Height: 4, Label: "nothing",
			})
			p.SetPipeline(solidPipeline(t, d))
			p.Draw(accel.Draw{VertexCount: 0})
		},
		says: "a draw of 0 vertices",
	}, {
		// The indexed form refused a negative first index or instance and the
		// other two forms did not, so a negative first vertex reached the
		// backend as a fetch before the start of the buffer.
		name: "a draw from a negative first vertex",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, b)}},
				Width: 4, Height: 4, Label: "backwards",
			})
			p.SetPipeline(solidPipeline(t, d))
			p.Draw(accel.Draw{VertexCount: 3, FirstVertex: -1})
		},
		says: "negative first vertex (-1)",
	}, {
		name: "an indirect draw bounded from a negative first instance",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, b)}},
				Width: 4, Height: 4, Label: "backwards",
			})
			p.SetPipeline(solidPipeline(t, d))
			args := newBuffer(t, d, "args", 4, accel.BufferStorage|accel.BufferIndirect)
			p.DrawIndirect(whole(t, args), accel.Draw{VertexCount: 3, FirstInstance: -2})
		},
		says: "first instance (-2)",
	}, {
		name: "an attachment too small for the area",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, b)}},
				Width: 64, Height: 64, Label: "cramped",
			})
			p.SetPipeline(solidPipeline(t, d))
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: `is texture "target" at mip 0, which is 4x4, and the render area is 64x64`,
	}, {
		name: "a depth attachment too small for the area",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			small := depthTarget(t, d, "shallow", 2, 2)
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, b)}},
				Depth: &accel.DepthAttachment{View: view(t, small)},
				Width: 4, Height: 4, Label: "shallow",
			})
			p.SetPipeline(depthPipeline(t, d))
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: `depth attachment is texture "shallow" at mip 0, which is 2x2`,
	}, {
		name: "a depth clear outside the window range",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			depth := depthTarget(t, d, "depth", 4, 4)
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, b)}},
				Depth: &accel.DepthAttachment{View: view(t, depth), Clear: -1},
				Width: 4, Height: 4, Label: "deep",
			})
			p.SetPipeline(depthPipeline(t, d))
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: "stored window depth is in [0, 1]",
	}, {
		name: "a pipeline with more targets than the pass has attachments",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			mrt, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
				Vertex:   &testkernels.GeometryVSStage,
				Fragment: &testkernels.ShadeFSStage,
				VertexBuffers: []accel.VertexBufferLayout{{
					Stride: 32,
					Attributes: []accel.VertexAttribute{
						{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
						{Location: 1, Format: accel.AttrFloat32x2, Offset: 12},
					},
				}},
				Targets: []accel.ColorTargetState{
					{Format: accel.RGBA32Float}, {Format: accel.RGBA32Float},
				},
				Label: "mrt",
			})
			if err != nil {
				t.Fatalf("pipeline: %v", err)
			}
			t.Cleanup(func() { _ = mrt.Close() })
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, b)}},
				Width: 4, Height: 4, Label: "single",
			})
			p.SetPipeline(mrt)
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: "has 2 colour targets and the pass has 1 attachments",
	}, {
		name: "a pipeline that tests depth in a pass with none",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{{View: view(t, b)}},
				Width: 4, Height: 4, Label: "flat",
			})
			p.SetPipeline(depthPipeline(t, d))
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: "has a depth state and the pass has no depth attachment",
	}, {
		name: "a stage whose by-value parameter has no value",
		record: func(t *testing.T, d *accel.Device, r *accel.Recorder, b *accel.Texture) {
			pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
				Vertex:   &testkernels.GeometryVSStage,
				Fragment: &testkernels.ShadeFSStage,
				VertexBuffers: []accel.VertexBufferLayout{{
					Stride: 32,
					Attributes: []accel.VertexAttribute{
						{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
						{Location: 1, Format: accel.AttrFloat32x2, Offset: 12},
					},
				}},
				Targets: []accel.ColorTargetState{
					{Format: accel.RGBA32Float}, {Format: accel.RGBA32Float},
				},
				Label: "uniformed",
			})
			if err != nil {
				t.Fatalf("pipeline: %v", err)
			}
			t.Cleanup(func() { _ = pipe.Close() })
			second := colourTarget(t, d, "second", 4, 4)
			p := r.RenderPass(accel.RenderPassDescriptor{
				Color: []accel.ColorAttachment{
					{View: view(t, b)}, {View: view(t, second)},
				},
				Width: 4, Height: 4, Label: "uniformed",
			})
			p.SetPipeline(pipe)
			p.Draw(accel.Draw{VertexCount: 3})
		},
		says: `GeometryVS takes "xf" at index 0 and no value was set`,
	}} {
		t.Run(c.name, func(t *testing.T) {
			d := openDevice(t)
			target := colourTarget(t, d, "target", 4, 4)
			r := d.NewRecorder()
			c.record(t, d, r, target)
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

func depthPipeline(t *testing.T, d *accel.Device) *accel.RenderPipeline {
	t.Helper()
	p, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.HalfTriangleVSStage,
		Fragment: &testkernels.SolidFSStage,
		Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float}},
		DepthStencil: &accel.DepthStencilState{
			Format: accel.Depth32Float, Test: true, Write: true,
			Compare: accel.CompareLess,
		},
		Label: "depth",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })
	return p
}

// A pass reports the node it recorded into, and a vertex buffer bound at a slot
// is held against the draw that follows it.
//
// SetVertexBuffer records and the vertex layout that would fetch through it is
// unbuilt, so a draw whose stage reads an attribute is refused above. What is
// asserted here is that binding one is not itself an error and that the slots
// below the one named are filled rather than left short.
func TestARenderPassNamesItsNodeAndHoldsItsBuffers(t *testing.T) {
	d := openDevice(t)
	target := colourTarget(t, d, "target", 4, 4)
	verts := newBuffer(t, d, "verts", 12, accel.BufferStorage|accel.BufferCopyDst)

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{View: view(t, target)}},
		Width: 4, Height: 4, Label: "noted",
	})
	p.SetPipeline(solidPipeline(t, d))
	p.SetVertexBuffer(2, whole(t, verts))
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if p.Node() != 0 {
		t.Errorf("the only node recorded is %d, want 0", p.Node())
	}
}

// An omitted instance count draws once.
//
// Go cannot tell an omitted field from an explicit zero, so a count of zero is
// one instance and there is no way to draw none. Asserted because the opposite
// reading -- zero means nothing drawn -- is the one a caller brings from an API
// where the count is a separate argument.
func TestAnOmittedInstanceCountDrawsOnce(t *testing.T) {
	const w, h = 4, 4
	d := openDevice(t)
	target := colourTarget(t, d, "target", w, h)

	r := d.NewRecorder()
	p := r.RenderPass(accel.RenderPassDescriptor{
		Color: []accel.ColorAttachment{{
			View: view(t, target), Load: accel.LoadClear, Clear: [4]float32{1, 0, 0, 1},
		}},
		Width: w, Height: h, Label: "once",
	})
	p.SetPipeline(solidPipeline(t, d))
	p.Draw(accel.Draw{VertexCount: 3})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	if got := readTarget(t, d, target)[(1*w+0)*4]; got != 0.25 {
		t.Errorf("pixel (0,1) is %v, so the draw with no instance count drew nothing", got)
	}
}
