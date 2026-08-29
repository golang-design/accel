// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package cpu

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"

	"golang.design/x/accel/internal/driver"
)

// unorm8 is v/255 exactly, and the round trip is the identity.
//
// Exactly, because this backend is the oracle and a lost bit is invisible in an
// image and fatal in a comparison.
//
// The round trip alone is a weaker check than it looks: truncating instead of
// rounding survives it here, because float32(b)/255 widened to float64 and
// multiplied by 255 lands at or above b for every b. What catches truncation is
// the halfway case in TestUnorm8ClampsItsInput, and that is why it is written
// out as a value rather than left to this loop.
func TestUnorm8IsExactAndRoundTrips(t *testing.T) {
	if got := unorm8ToFloat(0); got != 0 {
		t.Errorf("unorm8 0 decodes to %v, want 0", got)
	}
	if got := unorm8ToFloat(255); got != 1 {
		t.Errorf("unorm8 255 decodes to %v, want 1", got)
	}
	if got := unorm8ToFloat(51); got != 0.2 {
		t.Errorf("unorm8 51 decodes to %v, want exactly 51/255", got)
	}
	for b := range 256 {
		if got := floatToUnorm8(unorm8ToFloat(byte(b))); got != byte(b) {
			t.Fatalf("unorm8 %d round trips to %d", b, got)
		}
	}
}

// Out-of-range and NaN encode to the ends rather than to an arbitrary byte.
func TestUnorm8ClampsItsInput(t *testing.T) {
	for _, c := range []struct {
		in   float32
		want byte
	}{
		{-1, 0}, {0, 0}, {1, 255}, {2, 255},
		{float32(math.NaN()), 0},
		// Halfway rounds away from zero, which is round-to-nearest as every
		// target API specifies it.
		{0.5/255 + 1e-9, 1},
	} {
		if got := floatToUnorm8(c.in); got != c.want {
			t.Errorf("%v encodes to %d, want %d", c.in, got, c.want)
		}
	}
}

// The sRGB transfer function is IEC 61966-2-1, checked against hand-computed
// values rather than against its own inverse.
//
// Against constants, because an encode and a decode that are both wrong by the
// same factor round trip perfectly: the round trip is a necessary property and
// not a sufficient one.
func TestSRGBMatchesIEC61966(t *testing.T) {
	const eps = 1e-6
	for _, c := range []struct {
		name       string
		srgb, want float32
	}{
		{"black", 0, 0},
		{"white", 1, 1},
		// 0.5^2.4-ish: ((0.5 + 0.055)/1.055)^2.4.
		{"mid", 0.5, 0.2140411},
		// The linear segment, where the curve is c/12.92.
		{"toe", 0.02, 0.02 / 12.92},
	} {
		if got := srgbToLinear(c.srgb); math.Abs(float64(got-c.want)) > eps {
			t.Errorf("srgbToLinear(%v) = %v, want %v (%s)", c.srgb, got, c.want, c.name)
		}
	}
	for _, c := range []struct {
		name         string
		linear, want float32
	}{
		{"black", 0, 0},
		{"white", 1, 1},
		// 1.055 * 0.5^(1/2.4) - 0.055.
		{"mid", 0.5, 0.7353569},
		{"toe", 0.002, 0.002 * 12.92},
	} {
		if got := linearToSRGB(c.linear); math.Abs(float64(got-c.want)) > eps {
			t.Errorf("linearToSRGB(%v) = %v, want %v (%s)", c.linear, got, c.want, c.name)
		}
	}

	// The two thresholds are the same point on the curve. They are written as
	// different constants -- 0.04045 encoded and 0.0031308 linear -- and a
	// transposed pair is a discontinuity at the darkest values a display shows.
	if a, b := srgbToLinear(0.04045), float32(0.0031308); math.Abs(float64(a-b)) > 1e-7 {
		t.Errorf("the encoded threshold 0.04045 decodes to %v and the linear threshold "+
			"is %v; the two segments do not meet", a, b)
	}
}

// Every one of the 256 stored values survives a decode and an encode.
func TestSRGBRoundTripsEveryStoredByte(t *testing.T) {
	for b := range 256 {
		lin := srgbToLinear(unorm8ToFloat(byte(b)))
		if got := floatToUnorm8(linearToSRGB(lin)); got != byte(b) {
			t.Errorf("sRGB byte %d round trips to %d through linear %v", b, got, lin)
		}
	}
}

// The same bytes read through two views of one family are different values.
//
// This is the case specs/045-texture-attachments.md section 2.1 says the view
// format exists for: RGBA8Unorm and RGBA8UnormSRGB describe the same four
// bytes, and only the encoding differs. If they decoded alike the field would
// buy nothing.
func TestTheSRGBAndLinearCodecsDisagreeOnTheSameBytes(t *testing.T) {
	src := []byte{128, 128, 128, 128}
	lin, err := codecFor(driver.RGBA8Unorm)
	if err != nil {
		t.Fatalf("linear codec: %v", err)
	}
	srgb, err := codecFor(driver.RGBA8UnormSRGB)
	if err != nil {
		t.Fatalf("srgb codec: %v", err)
	}
	a, b := lin.decode(src), srgb.decode(src)
	if a[0] == b[0] {
		t.Errorf("both views read %v from the same bytes; an sRGB view that did not "+
			"convert would be a linear one with a different name", a[0])
	}
	if want := float32(128) / 255; a[0] != want {
		t.Errorf("the linear view reads %v, want %v", a[0], want)
	}
	if want := srgbToLinear(float32(128) / 255); b[0] != want {
		t.Errorf("the sRGB view reads %v, want %v", b[0], want)
	}
	// Alpha is a coverage weight and is linear in both.
	if a[3] != b[3] {
		t.Errorf("alpha reads %v linearly and %v through sRGB; alpha does not carry a "+
			"transfer function on any target", a[3], b[3])
	}
}

// A format with fewer than four channels reads its missing ones as (0, 0, 0, 1).
func TestAMissingChannelReadsAsZeroOrOne(t *testing.T) {
	c, err := codecFor(driver.R32Float)
	if err != nil {
		t.Fatalf("codec: %v", err)
	}
	src := make([]byte, 4)
	binary.LittleEndian.PutUint32(src, math.Float32bits(0.25))
	if got := c.decode(src); got != [4]float32{0.25, 0, 0, 1} {
		t.Errorf("R32Float decodes to %v, want {0.25 0 0 1}", got)
	}
}

// Channel order is the whole content of BGRA8Unorm.
func TestBGRAReadsTheSameBytesInTheOtherOrder(t *testing.T) {
	src := []byte{1, 2, 3, 4}
	bgra, err := codecFor(driver.BGRA8Unorm)
	if err != nil {
		t.Fatalf("codec: %v", err)
	}
	got := bgra.decode(src)
	want := [4]float32{
		unorm8ToFloat(3), unorm8ToFloat(2), unorm8ToFloat(1), unorm8ToFloat(4),
	}
	if got != want {
		t.Errorf("BGRA8Unorm decodes %v to %v, want %v", src, got, want)
	}
	out := make([]byte, 4)
	bgra.encode(out, got)
	if string(out) != string(src) {
		t.Errorf("BGRA8Unorm re-encodes to %v, want %v", out, src)
	}
}

// A padded image decodes and re-encodes to the bytes it started with, and the
// padding between rows is left alone.
//
// The pitch is the point: a texture's rows are padded to the device's
// alignment, and an image decoded as though they were tight is sheared.
func TestAnImageRoundTripsThroughItsRowPitch(t *testing.T) {
	const w, h, pitch = 3, 4, 32
	c, err := codecFor(driver.RGBA8Unorm)
	if err != nil {
		t.Fatalf("codec: %v", err)
	}
	src := make([]byte, pitch*h)
	for i := range src {
		src[i] = byte(i * 7 % 251)
	}
	before := append([]byte(nil), src...)

	pix := make([]float32, w*h*4)
	if err := c.decodeImage(pix, src, w, h, pitch); err != nil {
		t.Fatalf("decode: %v", err)
	}
	// Row 2, texel 1, channel 0 is at byte 2*pitch + 1*4.
	if want := unorm8ToFloat(src[2*pitch+4]); pix[(2*w+1)*4] != want {
		t.Errorf("texel (1,2) reads %v, want %v: the decode is not stepping the pitch",
			pix[(2*w+1)*4], want)
	}

	dst := make([]byte, pitch*h)
	copy(dst, src)
	for i := range pix {
		pix[i] = 0
	}
	if err := c.decodeImage(pix, src, w, h, pitch); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if err := c.encodeImage(dst, pix, w, h, pitch); err != nil {
		t.Fatalf("encode: %v", err)
	}
	if string(dst) != string(before) {
		t.Errorf("the round trip changed the image; unorm8 is exact both ways")
	}
}

// Every format a plan can name either decodes or refuses by name.
//
// The refusal is the interesting half: a format with two defensible encodings
// must not silently acquire one, because this backend is what every other
// backend is compared against.
func TestEveryPlanFormatDecodesOrRefusesByName(t *testing.T) {
	for _, f := range driver.Formats {
		c, err := codecFor(f)
		if err == nil {
			if c.bytes <= 0 || c.components <= 0 || c.format != f {
				t.Errorf("%v decodes with %d bytes and %d components", f, c.bytes, c.components)
			}
			continue
		}
		if f != driver.Depth24PlusStencil8 {
			t.Errorf("%v does not decode: %v", f, err)
			continue
		}
		if !strings.Contains(err.Error(), f.String()) {
			t.Errorf("the refusal of %v does not name it: %v", f, err)
		}
	}
	if _, err := codecFor(driver.FormatInvalid); err == nil {
		t.Error("FormatInvalid decodes, so an attachment with no format would be guessed at")
	}
}

// A rectangle that does not fit is refused with both sizes named.
func TestACodecRefusesAnAttachmentTooSmallForTheArea(t *testing.T) {
	c, err := codecFor(driver.RGBA8Unorm)
	if err != nil {
		t.Fatalf("codec: %v", err)
	}
	for _, tc := range []struct {
		name          string
		bytes, floats int
		w, h, pitch   int
		says          string
	}{
		{"short bytes", 16, 64, 4, 4, 16, "needs"},
		{"short floats", 256, 8, 4, 4, 16, "framebuffer"},
		{"pitch under a row", 256, 64, 4, 4, 8, "row pitch"},
		{"empty area", 256, 64, 0, 4, 16, "rectangle"},
	} {
		err := c.decodeImage(make([]float32, tc.floats), make([]byte, tc.bytes),
			tc.w, tc.h, tc.pitch)
		if err == nil {
			t.Errorf("%s: decoded", tc.name)
			continue
		}
		if !strings.Contains(err.Error(), tc.says) {
			t.Errorf("%s: %v does not say %q", tc.name, err, tc.says)
		}
		if err := c.encodeImage(make([]byte, tc.bytes), make([]float32, tc.floats),
			tc.w, tc.h, tc.pitch); err == nil {
			t.Errorf("%s: encoded", tc.name)
		}
	}
}

// The half-float formats go through the same conversion a narrow binding uses.
func TestTheHalfFormatsUseTheSharedFloat16Conversion(t *testing.T) {
	c, err := codecFor(driver.RGBA16Float)
	if err != nil {
		t.Fatalf("codec: %v", err)
	}
	want := [4]float32{0.5, -2, 65504, 0.25}
	dst := make([]byte, 8)
	c.encode(dst, want)
	if got := c.decode(dst); got != want {
		t.Errorf("RGBA16Float round trips %v to %v; each of these is exact in half", want, got)
	}
}

// Depth is one float per texel, not four.
func TestADepthCodecIsOneComponentWide(t *testing.T) {
	c, err := codecFor(driver.Depth32Float)
	if err != nil {
		t.Fatalf("codec: %v", err)
	}
	if c.components != 1 {
		t.Fatalf("Depth32Float has %d components, want 1", c.components)
	}
	const w, h = 2, 2
	z := []float32{0, 0.25, 0.5, 1}
	dst := make([]byte, w*h*4)
	if err := c.encodeImage(dst, z, w, h, w*4); err != nil {
		t.Fatalf("encode: %v", err)
	}
	got := make([]float32, w*h)
	if err := c.decodeImage(got, dst, w, h, w*4); err != nil {
		t.Fatalf("decode: %v", err)
	}
	for i := range z {
		if got[i] != z[i] {
			t.Errorf("depth %d round trips %v to %v", i, z[i], got[i])
		}
	}
}

// The stencil aspect survives the attachment's encoding, at every value a byte
// can hold.
//
// specs/033-render-api.md's stencil state needs a format that carries the
// aspect across passes, and this codec is what carries it. The stencil travels
// as a float because that is the only channel type here, and it is exact:
// every uint8 is a float32 with no rounding, so the round trip is the identity
// rather than a conversion with a domain.
func TestTheStencilAspectRoundTrips(t *testing.T) {
	c, err := codecFor(driver.Depth32FloatStencil8)
	if err != nil {
		t.Fatalf("codec: %v", err)
	}
	if c.bytes != 8 || c.components != 2 {
		t.Fatalf("the codec is %d bytes and %d components, want 8 and 2",
			c.bytes, c.components)
	}
	for _, z := range []float32{0, 0.5, 1, math.SmallestNonzeroFloat32, math.MaxFloat32} {
		for s := range 256 {
			var raw [8]byte
			c.encode(raw[:], [4]float32{z, float32(s), 0, 1})
			got := c.decode(raw[:])
			if got[0] != z || got[1] != float32(s) {
				t.Fatalf("(%v, %d) round-tripped to (%v, %v)", z, s, got[0], got[1])
			}
			// Written zero rather than left alone: an attachment's bytes are
			// compared against the oracle's, and uninitialised padding is a
			// difference nobody caused.
			if raw[5] != 0 || raw[6] != 0 || raw[7] != 0 {
				t.Fatalf("(%v, %d) left the reserved bytes as %v", z, s, raw[5:])
			}
		}
	}
}

// A stencil value outside a byte clamps rather than wrapping.
//
// The oracle has to answer a caller error the same way every time, and a wrap
// depends on the conversion's direction: 256 truncating to 0 and -1 truncating
// to 255 are both defensible and neither is a stencil value anyone meant.
func TestAStencilValueOutsideAByteClamps(t *testing.T) {
	c, err := codecFor(driver.Depth32FloatStencil8)
	if err != nil {
		t.Fatalf("codec: %v", err)
	}
	for _, tc := range []struct {
		in   float32
		want float32
	}{{-1, 0}, {256, 255}, {1e9, 255}, {float32(math.NaN()), 0}} {
		var raw [8]byte
		c.encode(raw[:], [4]float32{0, tc.in, 0, 1})
		if got := c.decode(raw[:])[1]; got != tc.want {
			t.Errorf("stencil %v encoded to %v, want %v", tc.in, got, tc.want)
		}
	}
}
