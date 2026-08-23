// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl_test

import "testing"

// A Metal device opens and answers for itself.
//
// This test fails rather than skips when there is no device, per
// specs/006-backends.md section 7: a job that promises a backend and finds none
// is a failure. A skip here would let the whole Metal backend rot green.
func TestDeviceOpens(t *testing.T) {
	devs := requireDevice(t)
	for _, d := range devs {
		defer d.Close()
		if d.Name() == "" {
			t.Error("a device with no name means -name or -UTF8String is wrong")
		}
		if d.RegistryID() == 0 {
			t.Error("a zero registry id cannot seed a stable adapter token")
		}
		// The ceilings are read through selectors that return different C
		// types -- a struct by value, two integers, two booleans -- and a
		// wrong return type reads a register that holds something else. Zero
		// is what that looks like.
		if d.MaxThreadsPerThreadgroup.Width == 0 || d.MaxThreadsPerThreadgroup.Depth == 0 {
			t.Errorf("maxThreadsPerThreadgroup is %+v: a struct returned by value "+
				"is where an ABI mismatch shows up first", d.MaxThreadsPerThreadgroup)
		}
		if d.MaxThreadgroupMemoryBytes == 0 {
			t.Error("a device reporting no threadgroup memory would fail every cooperative kernel")
		}
		if d.MaxBufferBytes == 0 {
			t.Error("a device reporting no maximum buffer length cannot size a pool")
		}
		t.Logf("%s registry=%#x threads=%+v shared=%d buffer=%d unified=%v lowpower=%v",
			d.Name(), d.RegistryID(), d.MaxThreadsPerThreadgroup,
			d.MaxThreadgroupMemoryBytes, d.MaxBufferBytes, d.UnifiedMemory, d.LowPower)
	}
}
