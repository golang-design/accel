// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl_test

import "testing"

// The device's total threadgroup ceiling is what a compiled pipeline reports.
//
// MTLDevice has no query for the total; -maxThreadsPerThreadgroup is per axis.
// The device used to report that per-axis width as the total, which is the
// same number on Apple silicon -- so this test cannot fail on this hardware
// for that reason, and says so. What it does check is that the reported total
// is the pipeline's, is positive, and is not above any axis's limit.
func TestTheTotalThreadgroupCeilingIsThePipelines(t *testing.T) {
	d := open(t)
	p, err := d.Compile(`#include <metal_stdlib>
using namespace metal;
kernel void trivial(device uint *out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  out[gid] = gid;
}`, "trivial")
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer p.Close()

	total := d.MaxTotalThreadsPerThreadgroup()
	if total <= 0 {
		t.Fatalf("the device reports a threadgroup ceiling of %d", total)
	}
	if total != p.MaxTotalThreadsPerThreadgroup {
		t.Errorf("the device reports a ceiling of %d and a trivial pipeline %d",
			total, p.MaxTotalThreadsPerThreadgroup)
	}
	axes := d.MaxThreadsPerThreadgroup
	if uint64(total) > axes.Width*axes.Height*axes.Depth {
		t.Errorf("a ceiling of %d exceeds the product of the axis limits %+v", total, axes)
	}
	t.Logf("total %d, per axis %+v", total, axes)
}
