// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal

import (
	"fmt"
	"sync"
	"time"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/mtl"
)

// The on-screen present path: specs/034-surface-present.md section 7.
//
// # Why presenting is a render pass and not a blit
//
// A drawable's pixel format comes from the short list Core Animation
// composites, and RGBA32Float is not on it, so presenting is a conversion. A
// blit cannot convert between formats.
//
// A *render* pass rather than a compute kernel writing the texture, and the
// reason is the caller's layer rather than preference: writing a texture from a
// compute kernel needs framebufferOnly = NO, which is a property of a layer the
// caller created and 034 section 6 does not obviously put on accel's side of the
// line. A render pass works either way, so accel does not have to touch a flag
// that is arguably not its to touch.
//
// The fragment stage reads the float buffer directly, indexed by its own
// window position. No sampler and no second texture: the pixels are already in
// device memory in the layout this needs.

// presentSource is the built-in stage pair that converts a rendered frame.
//
// Hand-written rather than generated, for the reason mtl.subgroupProbe is: it
// is backend infrastructure rather than a kernel a caller wrote, and putting it
// in the corpus would make the corpus describe accel rather than describe what
// accel runs.
const presentSource = `#include <metal_stdlib>
using namespace metal;

struct _accel_present_out { float4 pos [[position]]; };

// A triangle larger than the viewport, so one primitive covers every pixel and
// no seam can appear along a diagonal.
vertex _accel_present_out _accel_present_vs(uint vid [[vertex_id]]) {
    _accel_present_out o;
    float x = -1, y = -1;
    if (vid == 1) { x = 3; }
    if (vid == 2) { y = 3; }
    o.pos = float4(x, y, 0, 1);
    return o;
}

struct _accel_present_dims { uint width; uint height; };

fragment float4 _accel_present_fs(_accel_present_out in [[stage_in]],
                                  device const float4 *src [[buffer(0)]],
                                  constant _accel_present_dims &dims [[buffer(1)]]) {
    uint x = uint(in.pos.x);
    uint y = uint(in.pos.y);
    if (x >= dims.width || y >= dims.height) { return float4(0); }
    // Clamped, because the render path works in float and a drawable is
    // normalized: a value outside [0, 1] wraps into a plausible colour rather
    // than a visibly wrong one.
    return clamp(src[y * dims.width + x], 0.0, 1.0);
}`

// presentPipeline is the compiled conversion, once per device.
type presentPipeline struct {
	once sync.Once
	err  error
	vs   *mtl.Function
	fs   *mtl.Function
	pipe *mtl.RenderPipeline
}

func (p *presentPipeline) get(d *mtl.Device) (*mtl.RenderPipeline, error) {
	p.once.Do(func() {
		p.vs, p.err = d.CompileFunction(presentSource, "_accel_present_vs")
		if p.err != nil {
			return
		}
		p.fs, p.err = d.CompileFunction(presentSource, "_accel_present_fs")
		if p.err != nil {
			return
		}
		p.pipe, p.err = d.NewRenderPipeline(mtl.RenderPipelineSpec{
			Vertex: p.vs, Fragment: p.fs,
			ColorFormats: []int{mtl.PixelFormatBGRA8Unorm},
		})
	})
	return p.pipe, p.err
}

func (p *presentPipeline) close() {
	if p.pipe != nil {
		p.pipe.Close()
	}
	if p.vs != nil {
		p.vs.Close()
	}
	if p.fs != nil {
		p.fs.Close()
	}
}

// NewPresentTarget wraps a caller's native handle.
//
// Given a CAMetalLayer it adopts it and does not retain it: 034 section 6 puts
// the window and its lifetime on the caller's side, and retaining theirs would
// make accel a co-owner of an object the caller deallocates. Given an NSView it
// makes the layer, which must happen on the main thread.
func (d *device) NewPresentTarget(h driver.NativeHandle, w, height int) (driver.PresentTarget, error) {
	var layer *mtl.MetalLayer
	switch h.Kind {
	case driver.NativeMetalLayer:
		if h.Ptr == 0 {
			return nil, fmt.Errorf("accel: a nil CAMetalLayer")
		}
		layer = mtl.WrapLayer(h.Ptr)
	case driver.NativeNSView:
		// Refused rather than implemented. Reaching an NSView's layer needs
		// AppKit, which cannot be loaded in a process with no display session,
		// so the path could not be tested at all -- and an untestable branch a
		// caller can reach is the shape specs/009-sequencing.md records four
		// times. It also saves the caller nothing: they already have AppKit
		// loaded, because they made the window, and `view.layer` is one line on
		// their side.
		return nil, fmt.Errorf("accel: pass the view's CAMetalLayer rather than the " +
			"NSView: reaching one from the other needs AppKit, which accel cannot load " +
			"in a process with no display, so this path is untestable here and is not " +
			"built (specs/034-surface-present.md section 8)")
	default:
		return nil, fmt.Errorf("accel: the Metal backend presents to %v, and was given %v",
			driver.NativeMetalLayer, h.Kind)
	}
	t := &presentTarget{dev: d, layer: layer}
	if err := t.Configure(w, height); err != nil {
		layer.Close()
		return nil, err
	}
	return t, nil
}

// presentTarget is one CAMetalLayer.
type presentTarget struct {
	dev    *device
	layer  *mtl.MetalLayer
	pipe   presentPipeline
	mu     sync.Mutex
	w, h   int
	closed bool
}

func (t *presentTarget) Configure(w, h int) error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return fmt.Errorf("accel: the present target is closed")
	}
	if err := t.layer.Configure(t.dev.dev, w, h); err != nil {
		return err
	}
	t.w, t.h = w, h
	return nil
}

// Acquire takes the next drawable.
//
// The timeout is honoured by retrying rather than by asking Core Animation to
// wait, because nextDrawable takes no timeout: it blocks or returns nil, and
// which of the two it does is not ours to choose. Polling turns "returns nil"
// into a bounded wait; a nextDrawable that blocks is not interrupted by this
// and never was interruptible.
func (t *presentTarget) Acquire(timeout time.Duration) (driver.PresentImage, error) {
	t.mu.Lock()
	closed, w, h := t.closed, t.w, t.h
	t.mu.Unlock()
	if closed {
		return nil, fmt.Errorf("accel: the present target is closed")
	}

	deadline := time.Now().Add(timeout)
	for {
		if d := t.layer.NextDrawable(); d != nil {
			return &presentImage{target: t, drawable: d, w: w, h: h}, nil
		}
		if !time.Now().Before(deadline) {
			return nil, fmt.Errorf("%w: the layer has no free drawable after %v",
				driver.ErrImageUnavailable, timeout)
		}
		time.Sleep(time.Millisecond)
	}
}

func (t *presentTarget) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return nil
	}
	t.closed = true
	t.pipe.close()
	t.layer.Close()
	return nil
}

// presentImage is one acquired drawable.
type presentImage struct {
	target   *presentTarget
	drawable *mtl.Drawable
	w, h     int
	spent    bool
}

// Present converts the rendered frame into the drawable and shows it.
//
// The present is scheduled on the command buffer that does the conversion, and
// before it is committed. That is the call shape section 7 names: on Metal
// there is no "present this later" call, and the ordering a caller writes as a
// fence is expressed here by which command buffer the request rides on.
func (i *presentImage) Present(src driver.Block, offset int) error {
	if i.spent {
		return fmt.Errorf("accel: this drawable has already been presented or discarded")
	}
	i.spent = true
	defer func() { i.drawable.Release() }()

	buf, err := blockBuffer(src)
	if err != nil {
		return err
	}
	pipe, err := i.target.pipe.get(i.target.dev.dev)
	if err != nil {
		return fmt.Errorf("accel: compiling the present conversion: %w", err)
	}

	// Released on every path out of here. Committing hands the buffer to the
	// queue, which keeps it alive until it completes, so this retain has done
	// its job once Commit returns; one that was never committed has nothing
	// to wait for. Neither path kept a reference before, and a frame loop
	// leaked one command buffer per frame.
	cb := i.target.dev.queue.Begin()
	defer cb.Close()
	enc, err := cb.Render([]mtl.RenderAttachment{{
		Texture: i.drawable.Texture(i.w, i.h),
		Load:    mtl.LoadActionDontCare,
		Store:   mtl.StoreActionStore,
	}}, nil)
	if err != nil {
		return fmt.Errorf("accel: encoding the present pass: %w", err)
	}
	enc.SetPipeline(pipe)
	enc.SetFragmentBuffer(buf, offset, 0)
	enc.SetFragmentBytes(presentDims(i.w, i.h), 1)
	enc.Draw(3, 0, 3, 1, 0)
	enc.End()

	cb.PresentDrawable(i.drawable)
	cb.Commit()
	return nil
}

// Discard hands the drawable back unpresented.
func (i *presentImage) Discard() {
	if i.spent {
		return
	}
	i.spent = true
	i.drawable.Release()
}

// blockBuffer resolves a driver block to the Metal buffer behind it.
//
// Unwrapped first: a Block may be a handle to another, which is how the shared
// transient pool grows without invalidating what points into it.
func blockBuffer(b driver.Block) (*mtl.Buffer, error) {
	m, ok := driver.Unwrap(b).(*block)
	if !ok {
		return nil, fmt.Errorf("accel: the frame names a %T, which is not Metal memory", b)
	}
	return m.buf, nil
}

// presentDims encodes the two uint32 the fragment stage indexes with.
func presentDims(w, h int) []byte {
	out := make([]byte, 8)
	for i, v := range [2]uint32{uint32(w), uint32(h)} {
		out[i*4+0] = byte(v)
		out[i*4+1] = byte(v >> 8)
		out[i*4+2] = byte(v >> 16)
		out[i*4+3] = byte(v >> 24)
	}
	return out
}
