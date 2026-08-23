// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build !darwin

package tensor_test

import (
	"testing"

	"golang.design/x/accel"
)

// runMetalModel returns nil where there is no GPU backend compiled in.
func runMetalModel(*testing.T, func(*testing.T, *accel.Device, []uint32) []float32,
	[]uint32) []float32 {
	return nil
}
