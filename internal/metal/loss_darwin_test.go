// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal

import (
	"errors"
	"fmt"
	"testing"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/mtl"
)

// Every dtype's element stride is its size, which is what a binding's generated
// length is computed from.
//
// A wrong stride gives a kernel the wrong len() and it walks off the end of a
// buffer. On a GPU that is undefined rather than a panic, so it would surface
// as a wrong answer somewhere else entirely -- and no corpus kernel binds an i8
// or a bf16, so nothing else here would notice.
//
// specs/001-device-resources.md section 3.2: a storage buffer is a tightly
// packed array of one dtype, so the size is also the stride and there is no
// padding anywhere, ever.
func TestElementStrides(t *testing.T) {
	for _, tc := range []struct {
		dt   kernel.DType
		size int
	}{
		{kernel.F32, 4}, {kernel.I32, 4}, {kernel.U32, 4},
		{kernel.F16, 2}, {kernel.BF16, 2},
		{kernel.I8, 1}, {kernel.U8, 1},
	} {
		if got := elemBytes(tc.dt); got != tc.size {
			t.Errorf("%v is %d bytes, want %d", tc.dt, got, tc.size)
		}
	}
}

// Device loss is derived from what submissions report, and once derived it
// never clears.
//
// A real loss cannot be provoked on a healthy machine, and provoking one would
// take the developer's display with it. What can be tested is the classifier
// and the stickiness, which is where the decisions are: Metal reports loss as
// an error on one command buffer, so this backend has to decide which errors
// mean the device is gone and has to remember the answer.
//
// An internal test, because the classifier is not API. Exporting it for a test
// would put a function in the package's surface that exists only to be called
// from beside it.
func TestDeviceLossClassification(t *testing.T) {
	for _, tc := range []struct {
		code int
		msg  string
		lost bool
	}{
		{mtl.CommandBufferErrorDeviceRemoved, "Device Removed", true},
		{mtl.CommandBufferErrorAccessRevoked, "Access to the GPU was revoked", true},
		// The half that matters more. Each of these is a bug in the work and
		// leaves the device usable, and reporting one as loss would turn a
		// recoverable kernel bug into a device the caller must discard --
		// which specs/001-device-resources.md section 7.4 makes permanent.
		//
		// The timeout is the one that was wrong. Its text says "Caused GPU
		// Hang", the classifier matched that text, and a kernel with an
		// infinite loop made the device unusable for good.
		{mtl.CommandBufferErrorTimeout, "Execution of the command buffer was aborted due " +
			"to an error during execution. Caused GPU Hang Error (IOAF code 3)", false},
		{mtl.CommandBufferErrorPageFault, "Caused GPU Address Fault Error (IOAF code 11)", false},
		{mtl.CommandBufferErrorOutOfMemory, "Insufficient Memory", false},
		{mtl.CommandBufferErrorInvalidResource, "Invalid Resource", false},
		{mtl.CommandBufferErrorNotPermitted, "Not Permitted", false},
		{mtl.CommandBufferErrorInternal, "Internal Error", false},
	} {
		err := &mtl.CommandBufferError{Domain: mtl.CommandBufferErrorDomain, Code: tc.code, Message: tc.msg}
		if got := isDeviceLoss(err); got != tc.lost {
			t.Errorf("code %d %q classified as lost=%v, want %v", tc.code, tc.msg, got, tc.lost)
		}
		// Wrapped, as a fence reports it.
		if got := isDeviceLoss(fmt.Errorf("submit: %w", err)); got != tc.lost {
			t.Errorf("wrapped code %d classified as lost=%v, want %v", tc.code, got, tc.lost)
		}
	}

	// A code from another domain says nothing about the device, and neither
	// does an error that is not a command buffer's -- even one whose text
	// happens to say the words.
	other := &mtl.CommandBufferError{Domain: "NSCocoaErrorDomain",
		Code: mtl.CommandBufferErrorDeviceRemoved, Message: "Device Removed"}
	if isDeviceLoss(other) {
		t.Error("a DeviceRemoved code from another domain was classified as loss")
	}
	if isDeviceLoss(errors.New("Device Removed")) {
		t.Error("a plain error saying Device Removed was classified as loss")
	}
}

// deviceRemoved is the synthetic error a lost device would report.
func deviceRemoved() error {
	return &mtl.CommandBufferError{Domain: mtl.CommandBufferErrorDomain,
		Code: mtl.CommandBufferErrorDeviceRemoved, Message: "Device Removed"}
}

// timedOut is the synthetic error a hung kernel reports.
func timedOut() error {
	return &mtl.CommandBufferError{Domain: mtl.CommandBufferErrorDomain,
		Code: mtl.CommandBufferErrorTimeout, Message: "Caused GPU Hang Error"}
}

// Loss is sticky, and the first report is the one kept.
func TestDeviceLossIsSticky(t *testing.T) {
	d := &device{}
	if d.Lost() != nil {
		t.Fatal("a fresh device reports loss")
	}

	// Neither of these is loss, so neither may make the device unusable.
	d.noteSubmissionError(nil)
	d.noteSubmissionError(timedOut())
	if err := d.Lost(); err != nil {
		t.Fatalf("an ordinary submission failure was treated as device loss: %v", err)
	}

	d.noteSubmissionError(deviceRemoved())
	first := d.Lost()
	if first == nil {
		t.Fatal("a lost device still reports healthy")
	}
	if !errors.Is(first, driver.ErrDeviceLost) {
		t.Errorf("loss must be reported as driver.ErrDeviceLost so a caller can test "+
			"for it: %v", first)
	}

	// A later, different loss does not replace the first. The first is what a
	// caller was told and what explains everything that failed after it.
	d.noteSubmissionError(&mtl.CommandBufferError{Domain: mtl.CommandBufferErrorDomain,
		Code: mtl.CommandBufferErrorAccessRevoked, Message: "Access revoked"})
	if d.Lost() != first {
		t.Error("a second loss replaced the first; the original cause is what explains " +
			"every failure that followed it")
	}
	// And a healthy submission afterwards does not clear it, which is the whole
	// point of terminal: a driver reset that appeared to recover would leave a
	// caller running on resources whose contents are undefined.
	d.noteSubmissionError(nil)
	if d.Lost() == nil {
		t.Error("loss cleared, so it is not terminal")
	}
}
