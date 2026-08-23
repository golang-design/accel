// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package emit_test

import (
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/emit"
)

func TestMSLShape(t *testing.T) {
	src, err := emit.MSL(corpusKernel(t, "Add"))
	if err != nil {
		t.Fatalf("MSL: %v", err)
	}
	t.Log("\n" + src)
	for _, want := range []string{
		"kernel void Add(",
		"const device float *a [[buffer(0)]]",
		"device float *out [[buffer(2)]]",
		"constant uint *_lens [[buffer(3)]]",
		"[[thread_position_in_grid]]",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("emitted MSL is missing %q", want)
		}
	}
}
