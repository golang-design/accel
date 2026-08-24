// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package cpu

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"

	"golang.design/x/accel/internal/driver"
)

// A pass reads and writes its attachments through the format the plan names.
//
// Asserted here rather than only through a rendered image, because the two
// halves fail differently: a decode that ignored the format reads a byte as a
// float32 bit pattern, and a decode that ignored the pitch reads the padding
// between rows as texels. Both produce an image, and neither produces an error.
func TestAPassDecodesAndStoresThroughTheAttachmentFormat(t *testing.T) {
	const w, h, pitch = 3, 2, 32

	c, err := codecFor(driver.RGBA8Unorm)
	if err != nil {
		t.Fatalf("codec: %v", err)
	}
	raw := make([]byte, pitch*h)
	// Row 1, texel 2, red is 200. Everything else stays zero, so a decode that
	// stepped the wrong stride reads zero here and a full byte somewhere else.
	raw[pitch+2*4] = 200

	n := &resolvedNode{
		render: &driver.RenderPass{
			Label: "formats", Width: w, Height: h,
			Color:       []driver.Operand{{}},
			ColorFormat: []driver.Format{driver.RGBA8Unorm},
			ColorPitch:  []int{pitch},
		},
		colorAttach: [][]byte{raw},
		colorCodec:  []texelCodec{c},
	}

	fb, err := framebufferFor(n)
	if err != nil {
		t.Fatalf("framebuffer: %v", err)
	}
	if len(fb.Color) != 1 {
		t.Fatalf("%d colour targets, want one", len(fb.Color))
	}
	if got, want := fb.Color[0].Pix[(1*w+2)*4], unorm8ToFloat(200); got != want {
		t.Errorf("texel (2,1) decodes to %v, want %v: the pass is not reading the "+
			"attachment through its format and pitch", got, want)
	}

	// Write a value the rasterizer would have written and store it back.
	fb.Color[0].Pix[(0*w+1)*4+1] = 1
	if err := storeAttachments(n, fb); err != nil {
		t.Fatalf("store: %v", err)
	}
	if got := raw[1*4+1]; got != 255 {
		t.Errorf("green at texel (1,0) stored as %d, want 255", got)
	}
	if got := raw[pitch+2*4]; got != 200 {
		t.Errorf("texel (2,1) stored as %d, want the 200 it started with", got)
	}
	// The padding after the last texel of a row belongs to nobody and must
	// stay untouched: on a real texture it is another allocation's business.
	for i := w * 4; i < pitch; i++ {
		if raw[i] != 0 {
			t.Fatalf("byte %d of the row padding is %d, want 0", i, raw[i])
		}
	}
}

// A depth attachment decodes as one component per texel, not four.
func TestAPassDecodesDepthAsOneComponent(t *testing.T) {
	const w, h = 2, 2

	c, err := codecFor(driver.Depth32Float)
	if err != nil {
		t.Fatalf("codec: %v", err)
	}
	raw := make([]byte, w*h*4)
	for i, z := range []float32{0, 0.25, 0.5, 1} {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(z))
	}
	n := &resolvedNode{
		render: &driver.RenderPass{
			Label: "depth", Width: w, Height: h,
			Color:       []driver.Operand{},
			ColorFormat: []driver.Format{},
			ColorPitch:  []int{},
			Depth:       &driver.Operand{}, DepthFormat: driver.Depth32Float, DepthPitch: w * 4,
		},
		depthAttach: raw,
		depthCodec:  c,
	}
	fb, err := framebufferFor(n)
	if err != nil {
		t.Fatalf("framebuffer: %v", err)
	}
	if len(fb.Depth.Z) != w*h {
		t.Fatalf("the depth target holds %d floats, want %d", len(fb.Depth.Z), w*h)
	}
	if fb.Depth.Z[3] != 1 {
		t.Errorf("depth texel 3 decodes to %v, want 1", fb.Depth.Z[3])
	}
	fb.Depth.Z[0] = 0.75
	if err := storeAttachments(n, fb); err != nil {
		t.Fatalf("store: %v", err)
	}
	if got := math.Float32frombits(binary.LittleEndian.Uint32(raw)); got != 0.75 {
		t.Errorf("depth texel 0 stored as %v, want 0.75", got)
	}
}

// An attachment too small for the render area is refused rather than read past.
func TestAPassRefusesAnAttachmentTooSmallForItsArea(t *testing.T) {
	c, err := codecFor(driver.RGBA32Float)
	if err != nil {
		t.Fatalf("codec: %v", err)
	}
	n := &resolvedNode{
		render: &driver.RenderPass{
			Label: "small", Width: 4, Height: 4,
			Color:       []driver.Operand{{}},
			ColorFormat: []driver.Format{driver.RGBA32Float},
			ColorPitch:  []int{64},
		},
		colorAttach: [][]byte{make([]byte, 32)},
		colorCodec:  []texelCodec{c},
	}
	_, err = framebufferFor(n)
	if err == nil {
		t.Fatal("a 32-byte attachment was accepted for a 4x4 RGBA32Float area")
	}
	if !strings.Contains(err.Error(), "colour attachment 0") {
		t.Errorf("%v does not name the attachment", err)
	}
}
