// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal_test

import (
	"strings"
	"testing"

	"golang.design/x/accel/internal/driver"
)

// The device refuses to close while an executable or an allocation is open.
//
// It released the queue and the pipelines regardless, so an executable still
// open encoded its next submission into a released queue, and a block still
// live was memory whose device had gone. The CPU backend refuses for its
// allocations; this makes Metal refuse for the same, and for executables,
// which is where a submission in flight lives.
func TestTheDeviceRefusesToCloseWhileAnythingIsOpen(t *testing.T) {
	d, err := adapters(t)[0].Open(nil)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	// Registered first, so it runs after every Free the helpers register.
	t.Cleanup(func() {
		if err := d.Close(); err != nil {
			t.Errorf("close at the end: %v", err)
		}
	})

	b, err := d.Alloc(driver.MemoryShared, 64, "live")
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	if err := d.Close(); err == nil {
		t.Fatal("the device closed with a live allocation")
	} else if !strings.Contains(err.Error(), "1 allocations") {
		t.Errorf("the refusal should count the allocation: %v", err)
	}
	b.Free()

	e := spinExecutable(t, d, 1)
	if err := d.Close(); err == nil {
		t.Fatal("the device closed with an open executable")
	} else if !strings.Contains(err.Error(), "1 executables") {
		t.Errorf("the refusal should count the executable: %v", err)
	}
	// In flight: the executable refuses its own close, and the device refuses
	// through it.
	f, err := e.Submit()
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	if err := d.Close(); err == nil {
		t.Fatal("the device closed with a submission in flight")
	}
	if err := f.Wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
	if err := e.Close(); err != nil {
		t.Fatalf("close executable: %v", err)
	}
	// spinExecutable's cleanup frees its blocks after this test body, so they
	// are freed here first; a second Free is not part of the contract, which
	// is why the cleanup closes the executable rather than the blocks.
	if err := d.Close(); err == nil || !strings.Contains(err.Error(), "2 allocations") {
		t.Fatalf("the executable's two input blocks are still live, got %v", err)
	}
}

// A device with nothing open closes, and closes again harmlessly.
func TestTheDeviceClosesOnceEverythingIsReleased(t *testing.T) {
	d, err := adapters(t)[0].Open(nil)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	b, err := d.Alloc(driver.MemoryShared, 64, "live")
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	e := benchExecutable(t, 1, false)
	_ = e // its own device; nothing to do with d
	b.Free()
	if err := d.Close(); err != nil {
		t.Fatalf("close with nothing open: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("closing twice: %v", err)
	}
}
