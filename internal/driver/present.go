// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package driver

import (
	"errors"
	"time"
)

// The on-screen present seam of specs/034-surface-present.md section 6.
//
// # Why this is a driver interface and not a call into a backend
//
// accel links backends in, so a backend cannot import accel and accel cannot
// import one: naming internal/mtl from the public package would drag Objective-C
// into every compute binary, which TestRootPackageDoesNotDependOnTheToolchain
// exists to prevent. So the shared vocabulary lives here, and both sides speak
// it.
//
// # Why it is this narrow
//
// 034 section 6 puts the window on the caller's side of the line and everything
// from the swapchain inward on accel's. What crosses the line is a native handle
// going in and pixels coming out, and the two calls below are exactly that.

// ErrNoPresent reports that a backend has no on-screen path.
//
// A distinct error rather than a nil interface, because "this backend cannot
// present" is something a caller acts on -- specs/006-backends.md decision 6:
// absence is reported, not discovered.
var ErrNoPresent = errors.New("accel: this backend has no on-screen present path")

// ErrImageUnavailable reports that no image became available in time.
//
// Distinct from a failure, because it is ordinary: a compositor that has not
// released an image is not an error, and a caller retries rather than gives up.
var ErrImageUnavailable = errors.New("accel: no presentable image is available")

// NativeHandleKind says what a NativeHandle points at.
type NativeHandleKind uint8

const (
	NativeNone NativeHandleKind = iota

	// NativeMetalLayer is a CAMetalLayer*, which the caller created and
	// attached. It must be created and resized on the main thread.
	NativeMetalLayer

	// NativeNSView is an NSView*, from which a backend creates the layer -- and
	// that call must then itself be on the main thread.
	NativeNSView
)

func (k NativeHandleKind) String() string {
	switch k {
	case NativeMetalLayer:
		return "a CAMetalLayer"
	case NativeNSView:
		return "an NSView"
	}
	return "no native handle"
}

// NativeHandle is a platform-tagged pointer to a caller-owned window resource.
//
// Tagged rather than bare, because a backend given the wrong kind of pointer
// sends a message to an object that does not answer it, and the crash names
// neither the caller nor the mistake.
type NativeHandle struct {
	Kind NativeHandleKind
	Ptr  uintptr
}

// Presenter is a device that can put pixels on a screen.
//
// Optional, and a backend that does not implement it reports ErrNoPresent
// rather than the type assertion failing somewhere a caller cannot see.
type Presenter interface {
	// NewPresentTarget wraps a caller's native handle.
	NewPresentTarget(h NativeHandle, width, height int) (PresentTarget, error)
}

// PresentTarget is one on-screen surface.
type PresentTarget interface {
	// Configure resizes the target's images.
	Configure(width, height int) error

	// Acquire hands out the next image to present into.
	//
	// It reports ErrImageUnavailable when none is free within the timeout,
	// which is ordinary rather than a failure: the compositor may hold every
	// image, and a caller retries.
	Acquire(timeout time.Duration) (PresentImage, error)

	Close() error
}

// PresentImage is one acquired image.
//
// Exactly one of Present and Discard is called on it, and after either it is
// spent. Holding one across frames exhausts the pool, and the symptom is a
// frame loop that stops rather than an error -- which is why Discard exists at
// all: a frame the caller abandons has to go back.
type PresentImage interface {
	// Present shows the image, taking its pixels from a block holding
	// width*height*4 float32 in RGBA order.
	//
	// The pixels are a block rather than a texture because
	// specs/033-render-api.md makes an attachment a buffer view; a backend
	// converts to whatever its swapchain format is.
	Present(src Block, offset int) error

	// Discard returns the image unpresented.
	Discard()
}
