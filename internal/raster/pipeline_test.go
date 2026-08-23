// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package raster_test

import (
	"testing"

	"golang.design/x/accel/internal/raster"
)

// tri returns a full-viewport triangle pair's lower half at the given depth.
// A quad would be two triangles; one covering triangle is enough for every test
// here and keeps a failure about the chain rather than about coverage.
func quadAt(z float32) [3]raster.Vertex {
	return [3]raster.Vertex{at(-1, -1, z), at(3, -1, z), at(-1, 3, z)}
}

// constant returns a stage writing the same colour to every attachment.
func constant(c ...[4]float32) func(raster.Fragment) raster.Shaded {
	return func(raster.Fragment) raster.Shaded { return raster.Shaded{Color: c} }
}

func pass(w, h int) raster.PassState {
	return raster.PassState{State: state(w, h)}
}

// Occlusion is draw-order independent: the near triangle wins the overlap
// whichever order the two are drawn in.
//
// This fails without a working depth attachment, which is what makes it worth
// writing. The depths are widely separated so the comparison is exact --
// specs/035-cpu-rasterizer.md section 6 refuses to assert a portable winner when
// two depth intervals overlap, because depth testing picks a discrete surface
// and no numeric tolerance repairs a different one.
func TestOcclusionIsDrawOrderIndependent(t *testing.T) {
	const n = 8
	near, far := [4]float32{1, 0, 0, 1}, [4]float32{0, 1, 0, 1}

	// Clip z = -0.9 is window 0.05 and +0.9 is window 0.95, an interval apart.
	for _, order := range []string{"near first", "far first"} {
		t.Run(order, func(t *testing.T) {
			ps := pass(n, n)
			ps.Depth = raster.DepthState{Test: true, Write: true, Compare: raster.CompareLess}
			fb := &raster.Framebuffer{
				Color: []*raster.ColorTarget{raster.NewColorTarget(n, n, [4]float32{})},
				Depth: raster.NewDepthTarget(n, n, 1, 0),
			}
			draws := []struct {
				z float32
				c [4]float32
			}{{-0.9, near}, {0.9, far}}
			if order == "far first" {
				draws[0], draws[1] = draws[1], draws[0]
			}
			for _, d := range draws {
				raster.Draw(ps, fb, quadAt(d.z), constant(d.c))
			}
			if got := fb.Color[0].At(2, 2); got != near {
				t.Errorf("the overlap holds %v, want the near triangle's %v", got, near)
			}
		})
	}
}

// A discard writes neither colour nor depth.
//
// This is the assertion that separates specs/035-cpu-rasterizer.md section 5's
// ordering from early-Z: with the tests run before the stage, the depth is
// already written by the time the discard happens.
func TestDiscardWritesNeitherColourNorDepth(t *testing.T) {
	const n = 4
	ps := pass(n, n)
	ps.Depth = raster.DepthState{Test: true, Write: true, Compare: raster.CompareLess}

	clear := [4]float32{0.25, 0.5, 0.75, 1}
	fb := &raster.Framebuffer{
		Color: []*raster.ColorTarget{raster.NewColorTarget(n, n, clear)},
		Depth: raster.NewDepthTarget(n, n, 1, 0),
	}
	written := raster.Draw(ps, fb, quadAt(0), func(raster.Fragment) raster.Shaded {
		return raster.Shaded{Discard: true, Color: [][4]float32{{1, 1, 1, 1}}}
	})
	if written != 0 {
		t.Errorf("%d fragments reached an attachment write after discarding", written)
	}
	if got := fb.Color[0].At(1, 1); got != clear {
		t.Errorf("colour is %v after a discard, want the clear value %v", got, clear)
	}
	for i, z := range fb.Depth.Z {
		if z != 1 {
			t.Fatalf("depth[%d] is %g after a discard; the fragment's depth reached the "+
				"buffer, which is what running the depth test before the stage does", i, z)
		}
	}
}

// MRT: a stage writing a different constant to each attachment leaves each
// holding its own value. A single-target test cannot see aliased attachments.
func TestMRTAttachmentsAreDistinct(t *testing.T) {
	const n = 4
	want := [][4]float32{{1, 0, 0, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}}
	fb := &raster.Framebuffer{Color: []*raster.ColorTarget{
		raster.NewColorTarget(n, n, [4]float32{}),
		raster.NewColorTarget(n, n, [4]float32{}),
		raster.NewColorTarget(n, n, [4]float32{}),
	}}
	if raster.Draw(pass(n, n), fb, quadAt(0), constant(want...)) == 0 {
		t.Fatal("nothing was written")
	}
	for i, tgt := range fb.Color {
		if got := tgt.At(1, 1); got != want[i] {
			t.Errorf("attachment %d holds %v, want %v", i, got, want[i])
		}
	}
}

// Read-only depth: test on, write off. It is how a second pass shades exactly
// the surfaces the geometry pass kept, and the two booleans exist separately
// for it.
func TestDepthWriteOffKeepsTheBuffer(t *testing.T) {
	const n = 4
	fb := &raster.Framebuffer{
		Color: []*raster.ColorTarget{raster.NewColorTarget(n, n, [4]float32{})},
		Depth: raster.NewDepthTarget(n, n, 1, 0),
	}
	ps := pass(n, n)
	ps.Depth = raster.DepthState{Test: true, Write: false, Compare: raster.CompareLess}

	red := [4]float32{1, 0, 0, 1}
	if raster.Draw(ps, fb, quadAt(0), constant(red)) == 0 {
		t.Fatal("the draw was rejected by a test it should pass")
	}
	if got := fb.Color[0].At(1, 1); got != red {
		t.Errorf("colour is %v, want %v: the fragment passed the depth test", got, red)
	}
	for i, z := range fb.Depth.Z {
		if z != 1 {
			t.Fatalf("depth[%d] is %g with writes disabled", i, z)
		}
	}
}

// The write mask selects channels rather than gating the whole write.
func TestWriteMaskSelectsChannels(t *testing.T) {
	const n = 4
	clear := [4]float32{0.1, 0.2, 0.3, 0.4}
	tgt := raster.NewColorTarget(n, n, clear)
	fb := &raster.Framebuffer{Color: []*raster.ColorTarget{tgt}}
	ps := pass(n, n)
	ps.Mask = []raster.WriteMask{raster.WriteR | raster.WriteB}

	raster.Draw(ps, fb, quadAt(0), constant([4]float32{1, 1, 1, 1}))
	want := [4]float32{1, clear[1], 1, clear[3]}
	if got := tgt.At(1, 1); got != want {
		t.Errorf("masked write produced %v, want %v", got, want)
	}
}

// Blending, against the equation rather than against a remembered result.
func TestBlendFollowsTheEquation(t *testing.T) {
	const n = 4
	dst := [4]float32{0.2, 0.4, 0.6, 1}
	tgt := raster.NewColorTarget(n, n, dst)
	fb := &raster.Framebuffer{Color: []*raster.ColorTarget{tgt}}
	ps := pass(n, n)
	ps.Blend = []raster.Blend{{
		Enabled:  true,
		SrcColor: raster.FactorSrcAlpha, DstColor: raster.FactorOneMinusSrcAlpha,
		ColorOp:  raster.BlendAdd,
		SrcAlpha: raster.FactorOne, DstAlpha: raster.FactorZero,
		AlphaOp: raster.BlendAdd,
	}}

	src := [4]float32{1, 0, 0, 0.25}
	raster.Draw(ps, fb, quadAt(0), constant(src))

	var want [4]float32
	for c := range 3 {
		want[c] = src[3]*src[c] + (1-src[3])*dst[c]
	}
	want[3] = src[3]
	if got := tgt.At(1, 1); got != want {
		t.Errorf("blended to %v, want the equation's %v", got, want)
	}

	// Order matters, which is the reason draws inside a pass are never
	// reordered: blending the same two fragments the other way round gives a
	// different answer.
	tgt2 := raster.NewColorTarget(n, n, src)
	fb2 := &raster.Framebuffer{Color: []*raster.ColorTarget{tgt2}}
	raster.Draw(ps, fb2, quadAt(0), constant(dst))
	if got := tgt2.At(1, 1); got == want {
		t.Error("blending is order independent here, so this fixture does not show " +
			"why recorded draw order is preserved")
	}
}

// Stencil: a first draw marks the buffer, and a second draw is confined to what
// the first marked. The classic use, and the one that exercises Replace, a
// compare, and the read mask together.
func TestStencilConfinesASecondDraw(t *testing.T) {
	const n = 8
	fb := &raster.Framebuffer{
		Color: []*raster.ColorTarget{raster.NewColorTarget(n, n, [4]float32{})},
		Depth: raster.NewDepthTarget(n, n, 1, 0),
	}

	// Mark the left half with 1.
	mark := pass(n, n)
	mark.Scissor = raster.Rect{X: 0, Y: 0, W: n / 2, H: n}
	mark.Stencil = raster.StencilState{
		Enabled:   true,
		Reference: 1,
		Front: raster.StencilFace{
			Compare: raster.CompareAlways, ReadMask: 0xFF, WriteMask: 0xFF,
			Fail: raster.StencilKeep, DepthFail: raster.StencilKeep,
			Pass: raster.StencilReplace,
		},
	}
	mark.Stencil.Back = mark.Stencil.Front
	raster.Draw(mark, fb, quadAt(0), constant([4]float32{}))

	for y := range n {
		for x := range n {
			want := uint8(0)
			if x < n/2 {
				want = 1
			}
			if got := fb.Depth.Stencil[y*n+x]; got != want {
				t.Fatalf("stencil at (%d,%d) is %d, want %d", x, y, got, want)
			}
		}
	}

	// Now draw everywhere, but only where the stencil equals 1.
	shade := pass(n, n)
	shade.Stencil = raster.StencilState{
		Enabled:   true,
		Reference: 1,
		Front: raster.StencilFace{
			Compare: raster.CompareEqual, ReadMask: 0xFF, WriteMask: 0,
			Fail: raster.StencilKeep, DepthFail: raster.StencilKeep,
			Pass: raster.StencilKeep,
		},
	}
	shade.Stencil.Back = shade.Stencil.Front
	red := [4]float32{1, 0, 0, 1}
	raster.Draw(shade, fb, quadAt(0), constant(red))

	for y := range n {
		for x := range n {
			got := fb.Color[0].At(x, y)
			if x < n/2 {
				if got != red {
					t.Fatalf("(%d,%d) is %v inside the stencil, want %v", x, y, got, red)
				}
			} else if got != ([4]float32{}) {
				t.Fatalf("(%d,%d) is %v outside the stencil, want the clear value",
					x, y, got)
			}
		}
	}
}

// The depth-fail stencil operation, which is the one a naive implementation
// folds into the fail case: it fires when the stencil test passed and the depth
// test did not, and it is what shadow-volume techniques count with.
func TestStencilDepthFailIsItsOwnCase(t *testing.T) {
	const n = 4
	fb := &raster.Framebuffer{
		Color: []*raster.ColorTarget{raster.NewColorTarget(n, n, [4]float32{})},
		Depth: raster.NewDepthTarget(n, n, 0.5, 0),
	}
	ps := pass(n, n)
	ps.Depth = raster.DepthState{Test: true, Write: false, Compare: raster.CompareLess}
	ps.Stencil = raster.StencilState{
		Enabled:   true,
		Reference: 9,
		Front: raster.StencilFace{
			Compare: raster.CompareAlways, ReadMask: 0xFF, WriteMask: 0xFF,
			Fail:      raster.StencilZero,
			DepthFail: raster.StencilIncrementClamp,
			Pass:      raster.StencilReplace,
		},
	}
	ps.Stencil.Back = ps.Stencil.Front

	// Window depth 0.95, behind the cleared 0.5: the stencil test passes and
	// the depth test fails, so DepthFail runs and the colour is not written.
	if written := raster.Draw(ps, fb, quadAt(0.9), constant([4]float32{1, 1, 1, 1})); written != 0 {
		t.Errorf("%d fragments wrote colour after failing the depth test", written)
	}
	if got := fb.Depth.Stencil[0]; got != 1 {
		t.Errorf("stencil is %d, want 1 from one increment; 0 would mean the fail "+
			"operation ran and 9 would mean the pass operation did", got)
	}
}

// The stencil write mask selects bits, which is what lets two techniques share
// one buffer.
func TestStencilWriteMaskSelectsBits(t *testing.T) {
	const n = 2
	fb := &raster.Framebuffer{
		Color: []*raster.ColorTarget{raster.NewColorTarget(n, n, [4]float32{})},
		Depth: raster.NewDepthTarget(n, n, 1, 0xF0),
	}
	ps := pass(n, n)
	ps.Stencil = raster.StencilState{
		Enabled:   true,
		Reference: 0xFF,
		Front: raster.StencilFace{
			Compare: raster.CompareAlways, ReadMask: 0xFF, WriteMask: 0x0F,
			Fail: raster.StencilKeep, DepthFail: raster.StencilKeep,
			Pass: raster.StencilReplace,
		},
	}
	ps.Stencil.Back = ps.Stencil.Front
	raster.Draw(ps, fb, quadAt(0), constant([4]float32{}))

	if got := fb.Depth.Stencil[0]; got != 0xFF {
		t.Errorf("stencil is %#x, want 0xFF: the low nibble replaced and the high one kept",
			got)
	}
	// And with the mask excluding the bits the reference would change, nothing
	// moves at all.
	fb.Depth.Clear(1, 0xF0)
	ps.Stencil.Front.WriteMask = 0
	ps.Stencil.Back.WriteMask = 0
	raster.Draw(ps, fb, quadAt(0), constant([4]float32{}))
	if got := fb.Depth.Stencil[0]; got != 0xF0 {
		t.Errorf("stencil is %#x with a zero write mask, want 0xF0", got)
	}
}

// Draw reports how many fragments reached an attachment write, which is what
// distinguishes "covered nothing" from "covered and every fragment was
// rejected" -- two failures that look identical in the image.
func TestDrawReportsWrittenFragments(t *testing.T) {
	const n = 4
	fb := &raster.Framebuffer{
		Color: []*raster.ColorTarget{raster.NewColorTarget(n, n, [4]float32{})},
		Depth: raster.NewDepthTarget(n, n, 0, 0),
	}
	ps := pass(n, n)
	ps.Depth = raster.DepthState{Test: true, Write: true, Compare: raster.CompareLess}

	// Every fragment is behind the cleared depth of 0, so all are rejected.
	if got := raster.Draw(ps, fb, quadAt(0.5), constant([4]float32{1, 1, 1, 1})); got != 0 {
		t.Errorf("%d fragments written past a depth test none can pass", got)
	}
	// And nothing covered at all reports the same number for a different
	// reason, which is why the count alone is not the whole diagnosis.
	off := [3]raster.Vertex{at(-9, -9, 0), at(-7, -9, 0), at(-8, -7, 0)}
	if got := raster.Draw(pass(n, n), fb, off, constant([4]float32{1, 1, 1, 1})); got != 0 {
		t.Errorf("%d fragments written by an offscreen triangle", got)
	}

	ps.Depth.Compare = raster.CompareGreater
	if got := raster.Draw(ps, fb, quadAt(0.5), constant([4]float32{1, 1, 1, 1})); got != n*n {
		t.Errorf("%d of %d fragments written when every one should pass", got, n*n)
	}
}

// Every compare function, against its own definition. A table this shape is
// where an off-by-one between Less and LessEqual hides.
func TestCompareFunctions(t *testing.T) {
	const n = 2
	for _, tc := range []struct {
		name string
		cmp  raster.Compare
		// want[i] is whether a fragment at depth 0.5 passes against a buffer
		// holding {0.25, 0.5, 0.75}.
		want [3]bool
	}{
		{"never", raster.CompareNever, [3]bool{false, false, false}},
		{"less", raster.CompareLess, [3]bool{false, false, true}},
		{"equal", raster.CompareEqual, [3]bool{false, true, false}},
		{"lessEqual", raster.CompareLessEqual, [3]bool{false, true, true}},
		{"greater", raster.CompareGreater, [3]bool{true, false, false}},
		{"notEqual", raster.CompareNotEqual, [3]bool{true, false, true}},
		{"greaterEqual", raster.CompareGreaterEqual, [3]bool{true, true, false}},
		{"always", raster.CompareAlways, [3]bool{true, true, true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for i, buf := range []float32{0.25, 0.5, 0.75} {
				fb := &raster.Framebuffer{
					Color: []*raster.ColorTarget{raster.NewColorTarget(n, n, [4]float32{})},
					Depth: raster.NewDepthTarget(n, n, buf, 0),
				}
				ps := pass(n, n)
				ps.Depth = raster.DepthState{Test: true, Compare: tc.cmp}
				// Clip z = 0 is window depth 0.5.
				got := raster.Draw(ps, fb, quadAt(0), constant([4]float32{1, 1, 1, 1})) > 0
				if got != tc.want[i] {
					t.Errorf("depth 0.5 against %g: passed=%v, want %v", buf, got, tc.want[i])
				}
			}
		})
	}
}

// Every stencil operation, against its own definition, including the two wrap
// cases that differ from the clamping ones only at the extremes.
func TestStencilOperations(t *testing.T) {
	const n = 1
	for _, tc := range []struct {
		name string
		op   raster.StencilOp
		from uint8
		ref  uint8
		want uint8
	}{
		{"keep", raster.StencilKeep, 5, 9, 5},
		{"zero", raster.StencilZero, 5, 9, 0},
		{"replace", raster.StencilReplace, 5, 9, 9},
		{"incrementClamp", raster.StencilIncrementClamp, 5, 9, 6},
		{"incrementClamp at max", raster.StencilIncrementClamp, 0xFF, 9, 0xFF},
		{"decrementClamp", raster.StencilDecrementClamp, 5, 9, 4},
		{"decrementClamp at zero", raster.StencilDecrementClamp, 0, 9, 0},
		{"invert", raster.StencilInvert, 0x0F, 9, 0xF0},
		{"incrementWrap at max", raster.StencilIncrementWrap, 0xFF, 9, 0},
		{"decrementWrap at zero", raster.StencilDecrementWrap, 0, 9, 0xFF},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fb := &raster.Framebuffer{
				Color: []*raster.ColorTarget{raster.NewColorTarget(n, n, [4]float32{})},
				Depth: raster.NewDepthTarget(n, n, 1, tc.from),
			}
			ps := pass(n, n)
			ps.Stencil = raster.StencilState{
				Enabled:   true,
				Reference: tc.ref,
				Front: raster.StencilFace{
					Compare: raster.CompareAlways, ReadMask: 0xFF, WriteMask: 0xFF,
					Fail: raster.StencilKeep, DepthFail: raster.StencilKeep, Pass: tc.op,
				},
			}
			ps.Stencil.Back = ps.Stencil.Front
			raster.Draw(ps, fb, quadAt(0), constant([4]float32{}))
			if got := fb.Depth.Stencil[0]; got != tc.want {
				t.Errorf("%d -> %d, want %d", tc.from, got, tc.want)
			}
		})
	}
}

// Every blend factor and operation, against the equation.
func TestBlendFactorsAndOperations(t *testing.T) {
	const n = 1
	src := [4]float32{0.5, 0.25, 0.125, 0.75}
	dst := [4]float32{0.125, 0.5, 0.25, 0.375}

	factors := []raster.BlendFactor{
		raster.FactorZero, raster.FactorOne,
		raster.FactorSrcColor, raster.FactorOneMinusSrcColor,
		raster.FactorSrcAlpha, raster.FactorOneMinusSrcAlpha,
		raster.FactorDstColor, raster.FactorOneMinusDstColor,
		raster.FactorDstAlpha, raster.FactorOneMinusDstAlpha,
	}
	// The reference, written from the definition rather than from the
	// implementation's table.
	ref := func(f raster.BlendFactor, c int) float32 {
		switch f {
		case raster.FactorZero:
			return 0
		case raster.FactorOne:
			return 1
		case raster.FactorSrcColor:
			return src[c]
		case raster.FactorOneMinusSrcColor:
			return 1 - src[c]
		case raster.FactorSrcAlpha:
			return src[3]
		case raster.FactorOneMinusSrcAlpha:
			return 1 - src[3]
		case raster.FactorDstColor:
			return dst[c]
		case raster.FactorOneMinusDstColor:
			return 1 - dst[c]
		case raster.FactorDstAlpha:
			return dst[3]
		default:
			return 1 - dst[3]
		}
	}
	apply := func(op raster.BlendOp, s, d float32) float32 {
		switch op {
		case raster.BlendSubtract:
			return s - d
		case raster.BlendReverseSubtract:
			return d - s
		case raster.BlendMin:
			return min(s, d)
		case raster.BlendMax:
			return max(s, d)
		default:
			return s + d
		}
	}
	ops := []raster.BlendOp{
		raster.BlendAdd, raster.BlendSubtract, raster.BlendReverseSubtract,
		raster.BlendMin, raster.BlendMax,
	}

	for _, sf := range factors {
		for _, df := range factors {
			for _, op := range ops {
				tgt := raster.NewColorTarget(n, n, dst)
				fb := &raster.Framebuffer{Color: []*raster.ColorTarget{tgt}}
				ps := pass(n, n)
				ps.Blend = []raster.Blend{{
					Enabled:  true,
					SrcColor: sf, DstColor: df, ColorOp: op,
					SrcAlpha: sf, DstAlpha: df, AlphaOp: op,
				}}
				raster.Draw(ps, fb, quadAt(0), constant(src))

				var want [4]float32
				for c := range 4 {
					want[c] = apply(op, ref(sf, c)*src[c], ref(df, c)*dst[c])
				}
				if got := tgt.At(0, 0); got != want {
					t.Fatalf("src %v dst %v op %v: got %v, want %v", sf, df, op, got, want)
				}
			}
		}
	}
}

// A framebuffer with no depth attachment runs the colour path, which is 035
// section 8's first step: an offscreen triangle, one colour target, no depth.
func TestDrawWithNoDepthAttachment(t *testing.T) {
	const n = 4
	fb := &raster.Framebuffer{Color: []*raster.ColorTarget{
		raster.NewColorTarget(n, n, [4]float32{}),
	}}
	ps := pass(n, n)
	// Depth testing and stencil are configured but there is nothing to test
	// against, which must not reject every fragment or panic.
	ps.Depth = raster.DepthState{Test: true, Write: true, Compare: raster.CompareLess}
	ps.Stencil = raster.StencilState{Enabled: true, Reference: 1}

	red := [4]float32{1, 0, 0, 1}
	if got := raster.Draw(ps, fb, quadAt(0), constant(red)); got != n*n {
		t.Fatalf("%d of %d fragments written with no depth attachment", got, n*n)
	}
	if got := fb.Color[0].At(1, 1); got != red {
		t.Errorf("colour is %v, want %v", got, red)
	}
}

// A stage returning fewer colours than there are attachments leaves the
// remaining ones alone rather than reading past its slice.
func TestFewerStageOutputsThanAttachments(t *testing.T) {
	const n = 2
	clear := [4]float32{0.5, 0.5, 0.5, 1}
	fb := &raster.Framebuffer{Color: []*raster.ColorTarget{
		raster.NewColorTarget(n, n, clear),
		raster.NewColorTarget(n, n, clear),
		nil,
	}}
	red := [4]float32{1, 0, 0, 1}
	raster.Draw(pass(n, n), fb, quadAt(0), constant(red))
	if got := fb.Color[0].At(0, 0); got != red {
		t.Errorf("attachment 0 is %v, want %v", got, red)
	}
	if got := fb.Color[1].At(0, 0); got != clear {
		t.Errorf("attachment 1 is %v, want the untouched %v", got, clear)
	}
}

// Blend and mask are per attachment, so two attachments in one draw take
// different ones.
//
// This is the configuration that decides where the state lives: a pass holds one
// set of attachments and many draws, so blend belongs to the pipeline the draw
// carries rather than to the attachment it writes. specs/033-render-api.md fixes
// it at pipeline creation for that reason, and every backend agrees.
func TestBlendAndMaskArePerAttachment(t *testing.T) {
	const n = 2
	dst := [4]float32{0.5, 0.5, 0.5, 1}
	a := raster.NewColorTarget(n, n, dst)
	b := raster.NewColorTarget(n, n, dst)
	fb := &raster.Framebuffer{Color: []*raster.ColorTarget{a, b}}

	ps := pass(n, n)
	ps.Blend = []raster.Blend{
		{}, // attachment 0 does not blend
		{Enabled: true,
			SrcColor: raster.FactorOne, DstColor: raster.FactorOne, ColorOp: raster.BlendAdd,
			SrcAlpha: raster.FactorOne, DstAlpha: raster.FactorOne, AlphaOp: raster.BlendAdd},
	}
	ps.Mask = []raster.WriteMask{raster.WriteR} // and only attachment 0 is masked

	src := [4]float32{0.25, 0.25, 0.25, 0}
	raster.Draw(ps, fb, quadAt(0), constant(src, src))

	wantA := [4]float32{src[0], dst[1], dst[2], dst[3]}
	if got := a.At(0, 0); got != wantA {
		t.Errorf("attachment 0 is %v, want %v: unblended and red-masked", got, wantA)
	}
	wantB := [4]float32{src[0] + dst[0], src[1] + dst[1], src[2] + dst[2], src[3] + dst[3]}
	if got := b.At(0, 0); got != wantB {
		t.Errorf("attachment 1 is %v, want %v: additively blended and unmasked", got, wantB)
	}
}

// An attachment past the end of either slice takes the default, which is no
// blending and every channel written -- not a target nothing reaches.
func TestAbsentBlendAndMaskEntriesDefault(t *testing.T) {
	const n = 2
	tgt := raster.NewColorTarget(n, n, [4]float32{})
	fb := &raster.Framebuffer{Color: []*raster.ColorTarget{tgt}}
	ps := pass(n, n) // no Blend, no Mask

	red := [4]float32{1, 0, 0, 1}
	raster.Draw(ps, fb, quadAt(0), constant(red))
	if got := tgt.At(0, 0); got != red {
		t.Errorf("with no blend or mask configured the target holds %v, want %v", got, red)
	}
}
