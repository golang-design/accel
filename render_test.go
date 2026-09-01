// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"errors"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

func rgba() accel.ColorTargetState {
	return accel.ColorTargetState{Format: accel.RGBA8Unorm}
}

// A render pipeline compiles from two generated stages.
//
// specs/033-render-api.md. The point of the checks below is that they happen
// here rather than at draw: a pipeline that survives creation can only be got
// wrong by pairing it with mismatched attachments, which graph build catches.
func TestRenderPipelineCompiles(t *testing.T) {
	d := openDevice(t)
	p, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.GeometryVSStage,
		Fragment: &testkernels.ShadeFSStage,
		VertexBuffers: []accel.VertexBufferLayout{{
			Stride: 32,
			Attributes: []accel.VertexAttribute{
				{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
				{Location: 1, Format: accel.AttrFloat32x2, Offset: 12},
			},
		}},
		Primitive: accel.PrimitiveState{
			Topology: accel.TriangleList, FrontFace: accel.CounterClockwise,
			Cull: accel.CullBack,
		},
		Targets: []accel.ColorTargetState{rgba(), rgba()},
		Label:   "geometry",
	})
	if err != nil {
		t.Fatalf("NewRenderPipeline: %v", err)
	}
	defer p.Close()
	if p.Label() != "geometry" {
		t.Errorf("label is %q", p.Label())
	}
}

// Every refusal that creation owns, each because its absence is silent.
func TestRenderPipelineRefusals(t *testing.T) {
	d := openDevice(t)
	base := func() accel.RenderPipelineDescriptor {
		return accel.RenderPipelineDescriptor{
			Vertex:   &testkernels.GeometryVSStage,
			Fragment: &testkernels.ShadeFSStage,
			VertexBuffers: []accel.VertexBufferLayout{{
				Stride: 32,
				Attributes: []accel.VertexAttribute{
					{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
					{Location: 1, Format: accel.AttrFloat32x2, Offset: 12},
				},
			}},
			Primitive: accel.PrimitiveState{Topology: accel.TriangleList},
			Targets:   []accel.ColorTargetState{rgba(), rgba()},
			Label:     "p",
		}
	}

	for _, c := range []struct {
		name string
		mut  func(*accel.RenderPipelineDescriptor)
		says string
	}{
		{"no vertex stage", func(d *accel.RenderPipelineDescriptor) { d.Vertex = nil },
			"no vertex stage"},
		{"no fragment stage", func(d *accel.RenderPipelineDescriptor) { d.Fragment = nil },
			"no fragment stage"},
		{"stages swapped", func(d *accel.RenderPipelineDescriptor) {
			d.Vertex, d.Fragment = &testkernels.ShadeFSStage, &testkernels.GeometryVSStage
		}, "takes a vertex stage"},
		{"one target for two attachments", func(d *accel.RenderPipelineDescriptor) {
			d.Targets = []accel.ColorTargetState{rgba()}
		}, "one struct field per attachment"},
		{"no targets", func(d *accel.RenderPipelineDescriptor) { d.Targets = nil },
			"writes 2 attachments"},
		{"a depth format as a colour target", func(d *accel.RenderPipelineDescriptor) {
			d.Targets = []accel.ColorTargetState{{Format: accel.Depth32Float}, rgba()}
		}, "not a colour format"},
		{"a colour format as depth", func(d *accel.RenderPipelineDescriptor) {
			d.DepthStencil = &accel.DepthStencilState{Format: accel.RGBA8Unorm}
		}, "not a depth format"},
		{"a topology with no stated rule", func(d *accel.RenderPipelineDescriptor) {
			d.Primitive.Topology = accel.LineList
		}, "leaves its fill rule unstated"},
		{"stages that disagree about varyings", func(d *accel.RenderPipelineDescriptor) {
			d.Vertex = &testkernels.FullScreenVSStage
		}, "the two stages exchange one type"},
	} {
		t.Run(c.name, func(t *testing.T) {
			desc := base()
			c.mut(&desc)
			p, err := d.NewRenderPipeline(desc)
			if err == nil {
				_ = p.Close()
				t.Fatal("accepted")
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Errorf("the message should say %q, got %v", c.says, err)
			}
		})
	}
}

// A vertex-buffer layout is refused against the *device's* limit, and the
// refusal names the device.
//
// specs/042-surface-completion.md section 5.3. It used to be refused against
// mslabi.StageVertexBufferLimit on every device -- Metal's reservation, the
// index a stage's uniforms begin at -- so a caller on the CPU oracle met a
// ceiling one backend's ABI happens to have. That is 000's layering rule 3 with
// a constant standing in for the type.
func TestAVertexLayoutIsRefusedAgainstTheDevicesLimit(t *testing.T) {
	d := openDevice(t)
	limit := d.Info().Limits.MaxVertexBuffers
	if limit <= 0 {
		t.Fatal("the device reports no vertex-buffer limit, so nothing constrains a layout")
	}

	layout := func(n int) []accel.VertexBufferLayout {
		out := make([]accel.VertexBufferLayout, n)
		for i := range out {
			out[i] = accel.VertexBufferLayout{
				Stride:     12,
				Attributes: []accel.VertexAttribute{{Location: i, Format: accel.AttrFloat32x3}},
			}
		}
		return out
	}
	_, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
		Vertex:   &testkernels.AttributeVSStage,
		Fragment: &testkernels.TintFSStage,
		Targets:  []accel.ColorTargetState{rgba()},
		// One past the limit, which is where an off-by-one shows; a wildly
		// oversized layout is refused by a wrong comparison too.
		VertexBuffers: layout(limit + 1),
		Label:         "too many",
	})
	if err == nil {
		t.Fatal("a layout past the device's vertex-buffer limit was accepted")
	}
	for _, want := range []string{d.Info().Name, "limit"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal does not name %q: %v", want, err)
		}
	}
	if !errors.Is(err, accel.ErrUnsupported) {
		t.Errorf("the refusal is not ErrUnsupported: %v", err)
	}
}

// A vertex buffer slot is refused against the device's limit, not Metal's.
//
// The same rule as the layout check above, at the other place a slot is named:
// SetVertexBuffer refused every slot at or past mslabi.StageVertexBufferLimit
// on every device after Limits.MaxVertexBuffers had replaced that constant for
// the layout, so a device reporting a larger limit accepted a layout its passes
// could not bind. One below the limit is accepted and the limit itself is
// refused naming the device, so an off-by-one shows on either side.
func TestAVertexBufferSlotIsRefusedAgainstTheDevicesLimit(t *testing.T) {
	d := openDevice(t)
	limit := d.Info().Limits.MaxVertexBuffers
	if limit <= 0 {
		t.Fatal("the device reports no vertex-buffer limit, so nothing constrains a slot")
	}
	record := func(slot int) error {
		target := colourTarget(t, d, "target", 4, 4)
		vb := newBuffer(t, d, "vb", 12, accel.BufferStorage)
		r := d.NewRecorder()
		p := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{View: view(t, target), Load: accel.LoadClear}},
			Width: 4, Height: 4, Label: "slots",
		})
		p.SetPipeline(solidPipeline(t, d))
		p.SetVertexBuffer(slot, whole(t, vb))
		p.Draw(accel.Draw{VertexCount: 3})
		g, err := r.Build()
		if err == nil {
			_ = g.Close()
		}
		return err
	}

	if err := record(limit - 1); err != nil {
		t.Errorf("slot %d is below the device's limit of %d and was refused: %v",
			limit-1, limit, err)
	}
	err := record(limit)
	if err == nil {
		t.Fatalf("slot %d is the device's limit and was accepted", limit)
	}
	for _, want := range []string{d.Info().Name, "limit"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal does not name %q: %v", want, err)
		}
	}
}
