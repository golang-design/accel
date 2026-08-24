// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package emit_test

import (
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/emit"
)

// The MSL spelling of specs/032-stage-abi.md section 5's texel fetch.
//
// This is asserted as text rather than left to the golden, because the golden
// records whatever was emitted and a wrong guard would be recorded as happily
// as a right one. The Metal differential cannot cover it either: no render pass
// binds a texture yet, so nothing on the device *runs* a fetch. Until it does,
// the guard's shape is what stands between an out-of-range fetch and an
// out-of-bounds read, and it is stated here.
func TestMSLTexelFetchIsBoundsGuarded(t *testing.T) {
	src, err := emit.MSL(corpusKernel(t, "SampledFS"))
	if err != nil {
		t.Fatalf("MSL: %v", err)
	}
	t.Log("\n" + src)

	for _, want := range []string{
		// The binding, in Metal's own texture argument space.
		"texture2d<float> src [[texture(0)]]",
		// The fetch goes through the helper, never straight to read().
		"_accel_fetch2d(src, x, y)",

		// The guard, in two statements that are not interchangeable.
		//
		// get_width returns uint. `x < t.get_width()` with an int x is compared
		// *unsigned* by C's usual arithmetic conversions, so x = -1 becomes
		// 4294967295 and passes every plausible width -- which is exactly the
		// out-of-range read this rule exists to prevent. The sign test comes
		// first, and the magnitude test converts explicitly only once the
		// coordinate is known non-negative.
		"if (x < 0 || y < 0) { return float4(0.0); }",
		"if (uint(x) >= t.get_width() || uint(y) >= t.get_height()) { return float4(0.0); }",
		"return t.read(uint2(uint(x), uint(y)));",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("emitted MSL is missing %q", want)
		}
	}

	// The read must not appear outside the helper: a second, unguarded call
	// site would satisfy every assertion above.
	if n := strings.Count(src, ".read("); n != 1 {
		t.Errorf("the emitted MSL calls read() %d times, want exactly 1 — inside the "+
			"guarded helper and nowhere else", n)
	}
	// An unsigned comparison against a raw int coordinate is the defect this
	// spelling avoids, so its text is refused by name.
	for _, forbidden := range []string{
		"x < t.get_width()",
		"y < t.get_height()",
	} {
		if strings.Contains(src, forbidden) {
			t.Errorf("the emitted MSL contains %q, which C compares unsigned and lets a "+
				"negative coordinate through the guard", forbidden)
		}
	}
}

// A vertex stage declares its texture in the same argument space, and the
// helper is emitted once for it too.
//
// A separate case because the two stage kinds build their parameter lists in
// different functions: a fragment stage's uniforms start at buffer zero and a
// vertex stage's start above its vertex buffers, and the texture space is
// neither.
func TestMSLAVertexStageBindsATexture(t *testing.T) {
	src, err := emit.MSL(corpusKernel(t, "DisplacedVS"))
	if err != nil {
		t.Fatalf("MSL: %v", err)
	}
	for _, want := range []string{
		"texture2d<float> height [[texture(0)]]",
		"static float4 _accel_fetch2d(texture2d<float> t, int x, int y)",
		"_accel_fetch2d(height, i, int(0))",
	} {
		if !strings.Contains(src, want) {
			t.Errorf("emitted MSL is missing %q", want)
		}
	}
}

// A stage that fetches nothing carries no fetch helper.
//
// The prelude is emitted on demand, so a helper that appeared unconditionally
// would be dead text in every shader — and would stop this file's assertions
// from meaning that the body reached a fetch.
func TestMSLNoFetchHelperWithoutAFetch(t *testing.T) {
	src, err := emit.MSL(corpusKernel(t, "TintFS"))
	if err != nil {
		t.Fatalf("MSL: %v", err)
	}
	if strings.Contains(src, "_accel_fetch2d") {
		t.Error("a stage that fetches nothing carries the fetch helper")
	}
}
