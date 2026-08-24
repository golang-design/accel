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

// Every constant in the render surface spells itself, and spells itself
// correctly.
//
// Not decoration. These strings appear in the refusals a caller reads: "%v is
// not rasterizable", "attribute %q is [%d]float32 and %v in the layout", "the
// pipeline %s depth state". A String that names the wrong constant makes every
// one of those messages lie, and the lie is worse than a number because a
// reader believes it and looks in the wrong place.
//
// The unknown-value case is included for each, because that is what a caller
// sees when a value came from arithmetic or an uninitialised struct, and
// "Topology(9)" sends them somewhere useful where an empty string does not.
func TestTheRenderConstantsSpellThemselves(t *testing.T) {
	t.Run("topology", func(t *testing.T) {
		for _, c := range []struct {
			v    accel.Topology
			want string
		}{
			{accel.TriangleList, "triangle list"},
			{accel.TriangleStrip, "triangle strip"},
			{accel.LineList, "line list"},
			{accel.LineStrip, "line strip"},
			{accel.PointList, "point list"},
		} {
			if got := c.v.String(); got != c.want {
				t.Errorf("%d spells itself %q, want %q", uint8(c.v), got, c.want)
			}
		}
		if got := accel.Topology(200).String(); !strings.Contains(got, "200") {
			t.Errorf("an unknown topology spells itself %q and should name its value", got)
		}
	})

	t.Run("attribute formats", func(t *testing.T) {
		for _, c := range []struct {
			v          accel.AttrFormat
			components int
			size       int
			want       string
		}{
			{accel.AttrFloat32, 1, 4, "float32"},
			{accel.AttrFloat32x2, 2, 8, "float32x2"},
			{accel.AttrFloat32x3, 3, 12, "float32x3"},
			{accel.AttrFloat32x4, 4, 16, "float32x4"},
		} {
			if got := c.v.Components(); got != c.components {
				t.Errorf("%v holds %d components, want %d", c.want, got, c.components)
			}
			if got := c.v.Size(); got != c.size {
				t.Errorf("%v is %d bytes, want %d", c.want, got, c.size)
			}
			if got := c.v.String(); got != c.want {
				t.Errorf("a format spells itself %q, want %q", got, c.want)
			}
		}
		// An unset format holds nothing, which is what makes the zero value a
		// refusal rather than a single-component fetch.
		if got := accel.AttrInvalid.Components(); got != 0 {
			t.Errorf("an unset format holds %d components, want 0", got)
		}
		if got := accel.AttrInvalid.Size(); got != 0 {
			t.Errorf("an unset format is %d bytes, want 0", got)
		}
		if got := accel.AttrInvalid.String(); !strings.Contains(got, "unset") {
			t.Errorf("an unset format spells itself %q and should say so", got)
		}
	})

	t.Run("step modes", func(t *testing.T) {
		if got := accel.StepVertex.String(); got != "per vertex" {
			t.Errorf("StepVertex spells itself %q", got)
		}
		if got := accel.StepInstance.String(); got != "per instance" {
			t.Errorf("StepInstance spells itself %q", got)
		}
	})

	t.Run("index formats", func(t *testing.T) {
		if got := accel.Index16.String(); !strings.Contains(got, "uint16") {
			t.Errorf("Index16 spells itself %q", got)
		}
		if got := accel.Index32.String(); !strings.Contains(got, "uint32") {
			t.Errorf("Index32 spells itself %q", got)
		}
	})

	t.Run("native handles", func(t *testing.T) {
		if got := accel.NativeMetalLayer.String(); !strings.Contains(got, "CAMetalLayer") {
			t.Errorf("NativeMetalLayer spells itself %q", got)
		}
		if got := accel.NativeNSView.String(); !strings.Contains(got, "NSView") {
			t.Errorf("NativeNSView spells itself %q", got)
		}
		// The zero value is the one a caller reaches by forgetting the field,
		// so it says there is no handle rather than naming a platform.
		if got := accel.NativeHandleKind(0).String(); !strings.Contains(got, "no native") {
			t.Errorf("the zero handle kind spells itself %q", got)
		}
	})
}

// An omitted colour write mask writes everything, and an explicit empty one
// writes nothing.
//
// The distinction is the whole reason the mask resolves rather than being taken
// as given: a zero value means "not stated", and taking it literally would make
// every pipeline that did not mention a mask draw nothing — which looks like a
// broken rasterizer rather than a defaulting rule.
func TestAnOmittedWriteMaskWritesEverything(t *testing.T) {
	const w, h = 4, 4
	d := openDevice(t)
	q := d.Queue()

	render := func(t *testing.T, mask accel.ColorWriteMask) []float32 {
		pipe, err := d.NewRenderPipeline(accel.RenderPipelineDescriptor{
			Vertex:   &testkernels.HalfTriangleVSStage,
			Fragment: &testkernels.SolidFSStage,
			Targets:  []accel.ColorTargetState{{Format: accel.RGBA32Float, Mask: mask}},
			Label:    "masked",
		})
		if err != nil {
			t.Fatalf("pipeline: %v", err)
		}
		defer pipe.Close()

		target := newBuffer(t, d, "colour", w*h*4,
			accel.BufferStorage|accel.BufferCopySrc|accel.BufferCopyDst)
		r := d.NewRecorder()
		p := r.RenderPass(accel.RenderPassDescriptor{
			Color: []accel.ColorAttachment{{
				View: whole(t, target), Load: accel.LoadClear,
				Clear: [4]float32{0, 0, 0, 0},
			}},
			Width: w, Height: h, Label: "masked",
		})
		p.SetPipeline(pipe)
		p.Draw(accel.Draw{VertexCount: 3})
		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		if err := q.Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		return readback(t, d, target)
	}

	// The bottom-left pixel, which the triangle covers.
	at := (h - 1) * w * 4
	if got := render(t, 0)[at]; got != 0.25 {
		t.Errorf("with no mask stated the red channel is %v, want the drawn 0.25: a zero "+
			"mask was taken as writing nothing", got)
	}
	if got := render(t, accel.WriteGreen)[at]; got != 0 {
		t.Errorf("with only green enabled the red channel is %v, want the clear 0", got)
	}
	if got := render(t, accel.WriteGreen)[at+1]; got != 0.5 {
		t.Errorf("with only green enabled the green channel is %v, want the drawn 0.5", got)
	}
}
