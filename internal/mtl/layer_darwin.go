// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl

import (
	"fmt"
	"sync"

	"github.com/ebitengine/purego/objc"
)

// CAMetalLayer and its drawables: specs/034-surface-present.md section 7.
//
// # Why this is the risky one
//
// The predecessor implemented on-screen present for X11/EGL and Win32/ANGLE and
// never implemented this path, so present worked everywhere except Metal. Four
// things differ here from every other backend:
//
//   - the drawable is owned by the layer, not by us;
//   - nextDrawable can block or return nothing, and both are ordinary;
//   - the drawable must be presented on the command buffer that rendered it,
//     not afterwards, which is a different call shape entirely;
//   - and a drawable held across frames exhausts the pool, whose symptom is a
//     frame loop that stops rather than an error.
//
// # Main-thread affinity
//
// A CAMetalLayer must be created and resized on the main thread. This package
// does not enforce it -- it cannot know which thread it is on in a way Go makes
// meaningful -- and the requirement is documented on every call that reaches
// one, up through the public API.

var (
	layerOnce      sync.Once
	clsMetalLayer  objc.Class
	selSetDevice   objc.SEL
	selSetPixFmt   objc.SEL
	selSetDrawSize objc.SEL
	selDrawSize    objc.SEL
	selNextDraw    objc.SEL
	selDrawTexture objc.SEL
	selPresentDraw objc.SEL
)

func layerSelectors() {
	layerOnce.Do(func() {
		// Looked up on first use, not at package initialization: QuartzCore may
		// not be loaded yet, and a class that resolves to zero answers every
		// message with zero rather than crashing -- so the symptom would be a
		// layer that "could not be created" from a call that never happened.
		clsMetalLayer = objc.GetClass("CAMetalLayer")
		selSetDevice = objc.RegisterName("setDevice:")
		selSetPixFmt = objc.RegisterName("setPixelFormat:")
		selSetDrawSize = objc.RegisterName("setDrawableSize:")
		selDrawSize = objc.RegisterName("drawableSize")
		selNextDraw = objc.RegisterName("nextDrawable")
		selDrawTexture = objc.RegisterName("texture")
		selPresentDraw = objc.RegisterName("presentDrawable:")
	})
}

// PixelFormatBGRA8Unorm is what a CAMetalLayer presents in.
//
// Not a choice: a drawable's format comes from the short list Core Animation
// composites, which does not include the RGBA32Float the render path works in.
// So presenting is a conversion, and the conversion is what the present pass
// does.
const PixelFormatBGRA8Unorm = 80

// Size is a CGSize, passed by value.
type Size2D struct{ W, H float64 }

// MetalLayer is a caller-owned CAMetalLayer.
type MetalLayer struct {
	id    objc.ID
	owned bool
}

// WrapLayer adopts a CAMetalLayer the caller created and attached.
//
// The layer is not retained: specs/034-surface-present.md section 6 puts the
// window and its lifetime on the caller's side of the line, and retaining
// theirs would make accel a co-owner of an object whose deallocation the caller
// controls.
//
// The caller must have created it on the main thread, and must resize it there.
func WrapLayer(ptr uintptr) *MetalLayer {
	layerSelectors()
	return &MetalLayer{id: objc.ID(ptr)}
}

// NewOffscreenLayer makes a CAMetalLayer attached to nothing.
//
// It exists for tests, and what it is for is stated so nobody mistakes it for
// the real path: an unattached layer hands out drawables and presents them, so
// it exercises the drawable *lifetime* -- acquire, render, present on the
// command buffer, release. It does not exercise the bounded pool: measured on
// an M-series machine, an unattached layer hands out eight drawables without
// presenting any, and setting maximumDrawableCount does not bound it. So the
// blocking and unavailable paths of an attached layer are not reachable this
// way, and the claim about them is separate.
func NewOffscreenLayer() (*MetalLayer, error) {
	layerSelectors()
	if clsMetalLayer == 0 {
		return nil, fmt.Errorf("accel/mtl: CAMetalLayer is unavailable")
	}
	l := &MetalLayer{owned: true}
	withPool(func() { l.id = objc.ID(clsMetalLayer).Send(selAlloc).Send(selInit) })
	if l.id == 0 {
		return nil, fmt.Errorf("accel/mtl: could not create a CAMetalLayer")
	}
	return l, nil
}

// Configure points the layer at a device and sizes its drawables.
//
// The drawable size is in pixels and the layer's bounds are in points, so a
// caller on a Retina display passes the backing size. accel reports the
// constraint and does not compute it: the scale factor is the window's, which
// is on the caller's side of the line.
//
// Must be called on the main thread when the layer is attached to a window.
func (l *MetalLayer) Configure(d *Device, w, h int) error {
	if w <= 0 || h <= 0 {
		return fmt.Errorf("accel/mtl: a %dx%d drawable size", w, h)
	}
	withPool(func() {
		l.id.Send(selSetDevice, d.id)
		l.id.Send(selSetPixFmt, uintptr(PixelFormatBGRA8Unorm))
		l.id.Send(selSetDrawSize, Size2D{W: float64(w), H: float64(h)})
	})
	return nil
}

// NextDrawable takes the next image from the layer.
//
// Nil is ordinary rather than a failure: the compositor may hold every image,
// and the caller retries. It can also block, which is why the layer of the API
// above this one takes a timeout.
func (l *MetalLayer) NextDrawable() *Drawable {
	var d *Drawable
	withPool(func() {
		// Retained: nextDrawable is not a new* selector, so the drawable is
		// autoreleased and this pool would take it before it was presented.
		if id := l.id.Send(selNextDraw); id != 0 {
			d = &Drawable{id: retain(id)}
		}
	})
	return d
}

// Pointer is the layer as a native handle, for handing to the layer above.
func (l *MetalLayer) Pointer() uintptr { return uintptr(l.id) }

// Close releases the layer if this package made it.
func (l *MetalLayer) Close() {
	if l.owned {
		withPool(func() { release(l.id) })
	}
	l.id = 0
}

// Drawable is one image from a layer.
type Drawable struct {
	id  objc.ID
	tex *Texture
}

// Texture is the drawable's colour attachment.
//
// Not owned: it belongs to the drawable, which belongs to the layer. Releasing
// it would take a texture the compositor is about to read.
func (d *Drawable) Texture(w, h int) *Texture {
	if d.tex == nil {
		var id objc.ID
		withPool(func() { id = d.id.Send(selDrawTexture) })
		// The layer's format, which Configure set: a drawable is BGRA8Unorm.
		d.tex = &Texture{id: id, width: w, height: h, bpp: bytesPerPixel(PixelFormatBGRA8Unorm)}
	}
	return d.tex
}

// Release hands the drawable back without presenting it.
//
// Every acquired drawable must be released or presented exactly once. One held
// across frames exhausts the layer's pool, and the symptom is a frame loop that
// stops with no error and no stack -- which is why this exists rather than
// leaving an abandoned frame to a finalizer.
func (d *Drawable) Release() {
	if d.id == 0 {
		return
	}
	withPool(func() { release(d.id) })
	d.id, d.tex = 0, nil
}

// PresentDrawable schedules the drawable to be shown when this command buffer
// completes.
//
// On the command buffer that rendered it, and before Commit. That is the call
// shape specs/034-surface-present.md section 7 singles out: presenting
// afterwards is a different API on every other backend and is not available
// here, and the ordering a caller expresses with a fence is expressed here by
// which command buffer the call is made on.
func (cb *CommandBuffer) PresentDrawable(d *Drawable) {
	withPool(func() { cb.id.Send(selPresentDraw, d.id) })
}
