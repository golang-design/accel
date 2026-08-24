// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"strings"
	"testing"

	"golang.design/x/accel"
)

// A nil CAMetalLayer is refused by name rather than dereferenced.
//
// The zero value of NativeHandle has the right Kind and a nil pointer, which is
// what a caller who built the struct and forgot the layer passes. Nothing else
// in the surface path can tell that apart from a real layer, so the check is
// here and the message says which field is missing.
func TestANilLayerIsRefused(t *testing.T) {
	d := openMetal(t)
	_, err := d.NewWindowSurface(accel.NativeHandle{Kind: accel.NativeMetalLayer},
		accel.SurfaceDescriptor{Width: 8, Height: 8})
	if err == nil {
		t.Fatal("a nil CAMetalLayer was accepted")
	}
	if !strings.Contains(err.Error(), "nil CAMetalLayer") {
		t.Errorf("the refusal should name the layer, got %v", err)
	}
}
