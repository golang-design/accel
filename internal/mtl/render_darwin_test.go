// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl_test

import (
	"math"
	"os"
	"testing"

	"golang.design/x/accel/internal/mtl"
)

// The whole Metal render path, at the level this package owns it: compile two
// stages, build a pipeline state, render into a private texture, blit it into a
// buffer, and read the pixels.
//
// It is here rather than only in internal/metal because every step is a place a
// selector can be wrong in a way that compiles: a struct passed as separate
// arguments, an ownership rule inverted, a descriptor field the validator
// aborts on. Those fail here with a small stack rather than inside a graph.
func TestARenderPassProducesPixels(t *testing.T) {
	devs, err := mtl.Devices()
	if err != nil || len(devs) == 0 {
		if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
			t.Fatalf("this job promises Metal and found no device (err=%v)", err)
		}
		t.Skipf("no Metal device (err=%v)", err)
	}
	d := devs[0]
	defer func() {
		for _, x := range devs {
			x.Close()
		}
	}()

	const src = `#include <metal_stdlib>
using namespace metal;

struct VOut { float4 _pos [[position]]; };

vertex VOut vmain(uint vid [[vertex_id]]) {
    VOut o;
    float x = -1, y = -1;
    if (vid == 1) { x = 3; }
    if (vid == 2) { y = 3; }
    o._pos = float4(x, y, 0, 1);
    return o;
}

struct FOut { float4 c [[color(0)]]; };

fragment FOut fmain(VOut in [[stage_in]]) {
    FOut o;
    o.c = float4(0.25, 0.5, 0.75, 1);
    return o;
}`

	vs, err := d.CompileFunction(src, "vmain")
	if err != nil {
		t.Fatalf("vertex: %v", err)
	}
	defer vs.Close()
	fs, err := d.CompileFunction(src, "fmain")
	if err != nil {
		t.Fatalf("fragment: %v", err)
	}
	defer fs.Close()

	pipe, err := d.NewRenderPipeline(mtl.RenderPipelineSpec{
		Vertex: vs, Fragment: fs,
		ColorFormats: []int{mtl.PixelFormatRGBA32Float},
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	const w, h = 4, 4
	tex, err := d.NewRenderTarget(mtl.PixelFormatRGBA32Float, w, h)
	if err != nil {
		t.Fatalf("target: %v", err)
	}
	defer tex.Close()

	out, err := d.NewBuffer(w*h*16, mtl.StorageShared)
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer out.Close()

	q := d.NewQueue()
	defer q.Close()

	cb := q.Begin()
	enc, err := cb.Render([]mtl.RenderAttachment{{
		Texture: tex, Load: 2, Store: 1, // clear, store
		ClearColor: [4]float64{1, 0, 0, 1},
	}}, nil)
	if err != nil {
		t.Fatalf("encoder: %v", err)
	}
	enc.SetPipeline(pipe)
	enc.Draw(3, 0, 3, 1, 0) // MTLPrimitiveTypeTriangle
	enc.End()

	blit := cb.Blit()
	blit.CopyTextureToBuffer(tex, out, 0)
	blit.End()
	cb.Commit()
	cb.Wait()

	px := out.Bytes()
	if len(px) < 16 {
		t.Fatalf("the readback buffer is %d bytes", len(px))
	}
	got := [4]float32{
		f32(px[0:]), f32(px[4:]), f32(px[8:]), f32(px[12:]),
	}
	want := [4]float32{0.25, 0.5, 0.75, 1}
	if got != want {
		t.Errorf("pixel (0,0) is %v, want the fragment's %v; red is the clear, which "+
			"means the draw produced no coverage", got, want)
	}
}

func f32(b []byte) float32 {
	return math.Float32frombits(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
}
