// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal

import (
	"errors"
	"os"
	"strings"
	"testing"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/mtl"
)

// Every value of every enumeration this backend translates.
//
// The translations are written out by name rather than converted numerically,
// and this is what makes that worth doing: a value added to either side is a
// compile error or a failure here rather than a picture that is wrong in a
// plausible way. A numeric conversion would agree with both lists today and
// mistranslate every value after an insertion.
//
// The Metal constants are from MTLRenderCommandEncoder.h and MTLDepthStencil.h,
// spelled here so a reader can check them against the headers rather than infer
// them from the code they are checking.
func TestTheMetalEnumerationsMapByName(t *testing.T) {
	t.Run("compare functions", func(t *testing.T) {
		for _, c := range []struct {
			from uint8
			to   int
			name string
		}{
			{0, 0, "Never"}, {1, 1, "Less"}, {2, 2, "Equal"}, {3, 3, "LessEqual"},
			{4, 4, "Greater"}, {5, 5, "NotEqual"}, {6, 6, "GreaterEqual"},
			{7, 7, "Always"},
		} {
			if got := metalCompare(c.from); got != c.to {
				t.Errorf("%s maps to %d, want MTLCompareFunction%s = %d",
					c.name, got, c.name, c.to)
			}
		}
		// An unknown value is Always, which is the permissive answer and the
		// one that draws rather than silently discarding everything.
		if got := metalCompare(200); got != 7 {
			t.Errorf("an unrecognised compare maps to %d, want Always", got)
		}
	})

	t.Run("blend factors", func(t *testing.T) {
		for _, c := range []struct {
			from driver.BlendFactor
			to   int
			name string
		}{
			{driver.FactorZero, 0, "Zero"},
			{driver.FactorOne, 1, "One"},
			{driver.FactorSrcColor, 2, "SourceColor"},
			{driver.FactorOneMinusSrcColor, 3, "OneMinusSourceColor"},
			{driver.FactorSrcAlpha, 4, "SourceAlpha"},
			{driver.FactorOneMinusSrcAlpha, 5, "OneMinusSourceAlpha"},
			{driver.FactorDstColor, 6, "DestinationColor"},
			{driver.FactorOneMinusDstColor, 7, "OneMinusDestinationColor"},
			{driver.FactorDstAlpha, 8, "DestinationAlpha"},
			{driver.FactorOneMinusDstAlpha, 9, "OneMinusDestinationAlpha"},
		} {
			if got := metalFactor(c.from); got != c.to {
				t.Errorf("%v maps to %d, want MTLBlendFactor%s = %d",
					c.from, got, c.name, c.to)
			}
		}
	})

	t.Run("blend operations", func(t *testing.T) {
		for _, c := range []struct {
			from driver.BlendOp
			to   int
			name string
		}{
			{driver.BlendAdd, 0, "Add"},
			{driver.BlendSubtract, 1, "Subtract"},
			{driver.BlendReverseSubtract, 2, "ReverseSubtract"},
			{driver.BlendMin, 3, "Min"},
			{driver.BlendMax, 4, "Max"},
		} {
			if got := metalBlendOp(c.from); got != c.to {
				t.Errorf("%v maps to %d, want MTLBlendOperation%s = %d",
					c.from, got, c.name, c.to)
			}
		}
	})

	t.Run("primitives", func(t *testing.T) {
		// accel's Topology has TriangleList first and TriangleStrip second;
		// Metal's are 3 and 4.
		if got := metalPrimitive(0); got != 3 {
			t.Errorf("TriangleList maps to %d, want MTLPrimitiveTypeTriangle = 3", got)
		}
		if got := metalPrimitive(1); got != 4 {
			t.Errorf("TriangleStrip maps to %d, want MTLPrimitiveTypeTriangleStrip = 4",
				got)
		}
	})

	t.Run("winding", func(t *testing.T) {
		// The one inverted pair in the whole mapping, and the reason each of
		// these is written out: accel's FrontFace has CounterClockwise first
		// and Metal's MTLWinding has Clockwise first. A numeric conversion here
		// keeps back faces instead of front ones, and the silhouette still
		// looks right.
		if got := metalWinding(0); got != 1 {
			t.Errorf("CounterClockwise maps to %d, want MTLWindingCounterClockwise = 1",
				got)
		}
		if got := metalWinding(1); got != 0 {
			t.Errorf("Clockwise maps to %d, want MTLWindingClockwise = 0", got)
		}
	})

	t.Run("vertex formats", func(t *testing.T) {
		// MTLVertexFormatFloat is 28 and the widths follow, which is the one
		// place this backend relies on an enumeration being contiguous.
		float32x := func(n int) driver.VertexAttribute {
			return driver.VertexAttribute{Components: n, Bytes: 4, Signed: true}
		}
		for n, want := range map[int]int{1: 28, 2: 29, 3: 30, 4: 31} {
			if got := metalVertexFormat(float32x(n)); got != want {
				t.Errorf("%d components maps to %d, want MTLVertexFormatFloat%d = %d",
					n, got, n, want)
			}
		}
		for _, n := range []int{0, 5, -1} {
			if got := metalVertexFormat(float32x(n)); got != 0 {
				t.Errorf("%d components maps to %d, want an invalid format so the "+
					"pipeline refuses rather than fetching something", n, got)
			}
		}

		// The normalized families are *not* contiguous: MTLVertexFormat
		// interleaves the plain and normalized integer families and their
		// widths, so each is written out by name and each is checked.
		for _, c := range []struct {
			attr driver.VertexAttribute
			want int
			name string
		}{
			{driver.VertexAttribute{Components: 2, Bytes: 1, Normalized: true}, 7, "UChar2Normalized"},
			{driver.VertexAttribute{Components: 4, Bytes: 1, Normalized: true}, 9, "UChar4Normalized"},
			{driver.VertexAttribute{Components: 2, Bytes: 1, Signed: true, Normalized: true}, 10, "Char2Normalized"},
			{driver.VertexAttribute{Components: 4, Bytes: 1, Signed: true, Normalized: true}, 12, "Char4Normalized"},
			{driver.VertexAttribute{Components: 2, Bytes: 2, Normalized: true}, 19, "UShort2Normalized"},
			{driver.VertexAttribute{Components: 4, Bytes: 2, Normalized: true}, 21, "UShort4Normalized"},
			{driver.VertexAttribute{Components: 2, Bytes: 2, Signed: true, Normalized: true}, 22, "Short2Normalized"},
			{driver.VertexAttribute{Components: 4, Bytes: 2, Signed: true, Normalized: true}, 24, "Short4Normalized"},
		} {
			if got := metalVertexFormat(c.attr); got != c.want {
				t.Errorf("%s maps to %d, want %d", c.name, got, c.want)
			}
		}
		// A width no format carries is invalid rather than a guess.
		if got := metalVertexFormat(driver.VertexAttribute{
			Components: 3, Bytes: 1, Normalized: true,
		}); got != 0 {
			t.Errorf("a three-wide normalized format maps to %d, and it is not portable "+
				"-- Metal has UChar3Normalized and D3D12 does not", got)
		}
	})

	t.Run("load actions", func(t *testing.T) {
		for _, c := range []struct {
			from driver.LoadOp
			to   int
			name string
		}{
			{driver.LoadDontCare, 0, "DontCare"},
			{driver.LoadKeep, 1, "Load"},
			{driver.LoadClear, 2, "Clear"},
		} {
			if got := metalLoadAction(c.from); got != c.to {
				t.Errorf("%v maps to %d, want MTLLoadAction%s = %d",
					c.from, got, c.name, c.to)
			}
		}
	})
}

// A stage with no MSL is refused rather than run as something else.
//
// specs/032-stage-abi.md section 12.1: empty is not a fallback. A backend that
// silently ran a different lowering would make the two backends disagree about
// what a program means, which is the one thing the oracle arrangement exists to
// prevent.
func TestAStageWithNoMSLIsRefused(t *testing.T) {
	e := &executable{}
	_, err := e.stageFunction(&kernel.Stage{Name: "Bare"})
	if err == nil {
		t.Fatal("a stage carrying no MSL was accepted")
	}
	if !strings.Contains(err.Error(), "carries no MSL") {
		t.Errorf("the error should say the stage has no MSL, got %v", err)
	}
}

// A by-value parameter the stage does not declare, and one whose record carries
// no encoder, are errors rather than a panic inside a driver.
func TestUniformBytesRefusals(t *testing.T) {
	e := &executable{}
	if _, err := e.uniformBytes(&kernel.Stage{Name: "S"}, 0, 1); err == nil {
		t.Error("a parameter index outside the stage's list was accepted")
	}
	if _, err := e.uniformBytes(nil, 0, 1); err == nil {
		t.Error("a nil stage was accepted")
	}
	s := &kernel.Stage{
		Name:     "S",
		Uniforms: []kernel.StageUniform{{Name: "u", Size: 4}},
	}
	if _, err := e.uniformBytes(s, 0, 1); err == nil {
		t.Error("a parameter carrying no encoder was accepted")
	}
}

// The fallbacks each mapping takes for a value it does not know.
//
// They exist because a switch over an enumeration has to end somewhere, and the
// value chosen matters: an unknown blend factor is Zero and an unknown
// operation is Add, which together are "do not blend" rather than an arbitrary
// combination. An unknown load action is DontCare, which is the only one that
// promises nothing.
func TestTheMappingFallbacks(t *testing.T) {
	if got := metalFactor(driver.BlendFactor(200)); got != 0 {
		t.Errorf("an unknown blend factor maps to %d, want Zero", got)
	}
	if got := metalBlendOp(driver.BlendOp(200)); got != 0 {
		t.Errorf("an unknown blend operation maps to %d, want Add", got)
	}
	if got := metalLoadAction(driver.LoadOp(200)); got != 0 {
		t.Errorf("an unknown load action maps to %d, want DontCare", got)
	}
}

// A by-value parameter whose codec refuses the value is an error naming the
// parameter, not a panic inside a command encoder.
func TestUniformBytesReportsACodecFailure(t *testing.T) {
	e := &executable{}
	s := &kernel.Stage{
		Name: "S",
		Uniforms: []kernel.StageUniform{{
			Name: "u", Size: 4,
			Encode: func([]byte, any) error { return errBadValue },
		}},
	}
	if _, err := e.uniformBytes(s, 0, "not the right type"); err == nil {
		t.Error("a codec that refused the value was reported as success")
	}
}

var errBadValue = errors.New("the codec refused this value")

// A stage whose MSL the device compiler rejects is an error naming the stage.
//
// The emitter is a text transform, so this is the failure that appears when it
// widens past what the compiler accepts. Naming the stage is the difference
// between a fixable report and a wall of MSL.
func TestAStageWithBrokenMSLIsReported(t *testing.T) {
	devs, err := mtl.Devices()
	if err != nil || len(devs) == 0 {
		if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
			t.Fatalf("this job promises Metal and found no device (err=%v)", err)
		}
		t.Skipf("no Metal device (err=%v)", err)
	}
	d := devs[0]
	defer func() {
		for _, x := range devs {
			x.Close()
		}
	}()

	e := &executable{dev: &device{dev: d}}
	_, err = e.stageFunction(&kernel.Stage{
		Name: "Broken",
		MSL:  "this is not Metal Shading Language",
	})
	if err == nil {
		t.Fatal("the device compiler accepted text that is not MSL")
	}
	if !strings.Contains(err.Error(), "Broken") {
		t.Errorf("the error should name the stage, got %v", err)
	}
}

// A draw whose by-value parameter cannot be encoded is refused before anything
// is written into the command encoder.
//
// Before, and that is the point: a failure discovered halfway through binding a
// draw's arguments leaves an encoder holding some of them, and the pass that
// follows is neither the one recorded nor cleanly abandoned. Both stages are
// checked, because the fragment loop runs only when the vertex loop finished.
func TestBindStageUniformsFailsBeforeTouchingTheEncoder(t *testing.T) {
	e := &executable{}
	bad := &kernel.Stage{
		Name: "S",
		Uniforms: []kernel.StageUniform{{
			Name: "u", Size: 4,
			Encode: func([]byte, any) error { return errBadValue },
		}},
	}
	empty := &kernel.Stage{Name: "E"}

	// A nil encoder is safe precisely because the failure happens first. If
	// this ever panics, the check has moved after the first write.
	if err := e.bindStageUniforms(nil, driver.RenderDraw{
		Vertex: bad, Fragment: empty, VertexUniforms: []any{1},
	}); err == nil {
		t.Error("a vertex parameter that could not be encoded was accepted")
	}
	if err := e.bindStageUniforms(nil, driver.RenderDraw{
		Vertex: empty, Fragment: bad, FragmentUniforms: []any{1},
	}); err == nil {
		t.Error("a fragment parameter that could not be encoded was accepted")
	}
}
