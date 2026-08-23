// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package tensor_test

import (
	"testing"

	"golang.design/x/accel"
)

// runMetalModel runs the model on Metal, or returns nil when the job did not
// promise a device.
func runMetalModel(t *testing.T, run func(*testing.T, *accel.Device, []uint32) []float32,
	tokens []uint32) []float32 {
	t.Helper()
	for _, info := range accel.Enumerate().Devices {
		if info.Backend != accel.BackendMetal {
			continue
		}
		d, err := accel.OpenDevice(info.ID)
		if err != nil {
			t.Fatalf("OpenDevice: %v", err)
		}
		defer d.Close()
		return run(t, d, tokens)
	}
	return nil
}
