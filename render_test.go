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
