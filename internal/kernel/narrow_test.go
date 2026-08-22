// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel_test

import (
	"math"
	"testing"

	"golang.design/x/accel/internal/kernel"
)

// TestF16Conversions is spec 008 section 4's table, compared by bits.
//
// By bits, because that is what the table specifies and because a value
// comparison cannot see a negative zero that came back positive or two NaNs
// that differ. Every backend has to produce these exact encodings, so the
// oracle has to as well.
func TestF16Conversions(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   float32
		want uint16
	}{
		{"zero", 0, 0x0000},
		{"negative zero", float32(math.Copysign(0, -1)), 0x8000},
		{"one", 1, 0x3c00},
		{"negative one", -1, 0xbc00},
		{"two", 2, 0x4000},
		{"half", 0.5, 0x3800},
		{"largest finite f16", 65504, 0x7bff},
		{"just past f16's range", 65520, 0x7c00}, // overflows to infinity
		{"huge", 1e30, 0x7c00},
		{"negative huge", -1e30, 0xfc00},
		{"infinity", float32(math.Inf(1)), 0x7c00},
		{"negative infinity", float32(math.Inf(-1)), 0xfc00},
		{"smallest normal f16", 6.103515625e-05, 0x0400},
		{"largest subnormal f16", 6.097555160522461e-05, 0x03ff},
		{"smallest subnormal f16", 5.960464477539063e-08, 0x0001},
		{"underflows to zero", 1e-10, 0x0000},
		{"negative underflow keeps its sign", -1e-10, 0x8000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := kernel.ToFloat16(tc.in).Bits(); got != tc.want {
				t.Errorf("ToFloat16(%v) = %#04x, want %#04x", tc.in, got, tc.want)
			}
		})
	}

	// A NaN becomes the canonical quiet encoding rather than carrying a payload
	// no backend agrees on.
	if got := kernel.ToFloat16(float32(math.NaN())).Bits(); got != 0x7e00 {
		t.Errorf("ToFloat16(NaN) = %#04x, want the canonical 0x7e00", got)
	}
}

// TestF16RoundsToNearestEven is the rule truncation would silently violate.
func TestF16RoundsToNearestEven(t *testing.T) {
	// f16 has 10 mantissa bits, so 1 + 2^-11 is exactly halfway between 1 and
	// the next representable value. Ties go to even, which is 1.
	if got := kernel.ToFloat16(1 + 1.0/2048).Bits(); got != 0x3c00 {
		t.Errorf("a tie at 1 rounded to %#04x, want 0x3c00: ties go to even", got)
	}
	// 1 + 3*2^-11 is halfway between the second and third values, and the even
	// one is the third.
	if got := kernel.ToFloat16(1 + 3.0/2048).Bits(); got != 0x3c02 {
		t.Errorf("a tie above 1 rounded to %#04x, want 0x3c02", got)
	}
	// Just over halfway rounds up regardless of parity.
	if got := kernel.ToFloat16(1 + 1.1/2048).Bits(); got != 0x3c01 {
		t.Errorf("just past a tie rounded to %#04x, want 0x3c01", got)
	}
}

// TestF16WidensExactly checks the direction with no rounding, including the
// subnormal path, which is the one an implementation usually gets wrong.
func TestF16WidensExactly(t *testing.T) {
	for _, tc := range []struct {
		bits uint16
		want float32
	}{
		{0x0000, 0},
		{0x3c00, 1},
		{0xbc00, -1},
		{0x7bff, 65504},
		{0x0400, 6.103515625e-05},
		{0x0001, 5.960464477539063e-08},
		{0x03ff, 6.097555160522461e-05},
	} {
		if got := kernel.Float16FromBits(tc.bits).F32(); got != tc.want {
			t.Errorf("Float16(%#04x).F32() = %v, want %v", tc.bits, got, tc.want)
		}
	}

	if got := kernel.Float16FromBits(0x8000).F32(); !math.Signbit(float64(got)) || got != 0 {
		t.Errorf("negative zero widened to %v, losing its sign", got)
	}
	if got := kernel.Float16FromBits(0x7c00).F32(); !math.IsInf(float64(got), 1) {
		t.Errorf("infinity widened to %v", got)
	}
	if got := kernel.Float16FromBits(0xfc00).F32(); !math.IsInf(float64(got), -1) {
		t.Errorf("negative infinity widened to %v", got)
	}
	if got := kernel.Float16FromBits(0x7e00).F32(); !math.IsNaN(float64(got)) {
		t.Errorf("NaN widened to %v", got)
	}
}

// TestBF16Conversions covers the format whose whole point is f32's range at
// half the width.
func TestBF16Conversions(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   float32
		want uint16
	}{
		{"zero", 0, 0x0000},
		{"negative zero", float32(math.Copysign(0, -1)), 0x8000},
		{"one", 1, 0x3f80},
		{"negative one", -1, 0xbf80},
		{"two", 2, 0x4000},
		// bf16 keeps f32's exponent, so a value that overflows f16 by orders of
		// magnitude is ordinary here. That is the trade the format exists for.
		// float32(1e30) is 0x7149f2ca, whose low half is past the halfway
		// point, so it rounds up. The value here was originally guessed and
		// the test caught it, which is the point of comparing bits.
		{"far past f16's range", 1e30, 0x714a},
		{"infinity", float32(math.Inf(1)), 0x7f80},
		{"negative infinity", float32(math.Inf(-1)), 0xff80},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := kernel.ToBFloat16(tc.in).Bits(); got != tc.want {
				t.Errorf("ToBFloat16(%v) = %#04x, want %#04x", tc.in, got, tc.want)
			}
		})
	}

	if got := kernel.ToBFloat16(float32(math.NaN())).Bits(); got != 0x7fc0 {
		t.Errorf("ToBFloat16(NaN) = %#04x, want the canonical 0x7fc0", got)
	}
}

// TestBF16RoundsRatherThanTruncates is the difference spec 008 section 4 makes
// normative and a naive implementation misses.
//
// Truncating the low 16 bits is round-toward-zero, which biases every value in
// one direction, and a bias that small compounds over a reduction into a result
// nobody can attribute.
func TestBF16RoundsRatherThanTruncates(t *testing.T) {
	// A value whose low 16 bits are just over half: rounding goes up,
	// truncating stays.
	x := math.Float32frombits(0x3f80_8001) // 1 + a bit more than half an ulp
	if got := kernel.ToBFloat16(x).Bits(); got != 0x3f81 {
		t.Errorf("ToBFloat16(%v) = %#04x, want 0x3f81: truncation would give 0x3f80", got, got)
	}

	// An exact tie with an even result stays even.
	tie := math.Float32frombits(0x3f80_8000)
	if got := kernel.ToBFloat16(tie).Bits(); got != 0x3f80 {
		t.Errorf("a tie rounded to %#04x, want the even 0x3f80", got)
	}
	// An exact tie with an odd result rounds up to even.
	tieOdd := math.Float32frombits(0x3f81_8000)
	if got := kernel.ToBFloat16(tieOdd).Bits(); got != 0x3f82 {
		t.Errorf("an odd tie rounded to %#04x, want the even 0x3f82", got)
	}
}

// TestBF16WidensExactly checks that a bf16 is the upper half of an f32, which
// is what makes widening free.
func TestBF16WidensExactly(t *testing.T) {
	for _, bits := range []uint16{0x0000, 0x3f80, 0xbf80, 0x7f80, 0x4049} {
		got := kernel.BFloat16FromBits(bits).F32()
		if want := math.Float32frombits(uint32(bits) << 16); math.Float32bits(got) != math.Float32bits(want) {
			t.Errorf("BFloat16(%#04x).F32() = %v, want %v", bits, got, want)
		}
	}
}

// TestNarrowRoundTrip checks that a value already representable survives.
//
// The property is not that every f32 round-trips, which is impossible at half
// the width. It is that a value which *is* representable comes back unchanged,
// so storing and loading a weight does not walk it toward zero over successive
// passes.
func TestNarrowRoundTrip(t *testing.T) {
	for bits := range uint32(1 << 16) {
		f := kernel.Float16FromBits(uint16(bits))
		wide := f.F32()
		if math.IsNaN(float64(wide)) {
			continue // canonicalized on the way back, by design
		}
		if got := kernel.ToFloat16(wide).Bits(); got != uint16(bits) {
			t.Fatalf("f16 %#04x widened to %v and came back %#04x", bits, wide, got)
		}
	}

	for bits := range uint32(1 << 16) {
		b := kernel.BFloat16FromBits(uint16(bits))
		wide := b.F32()
		if math.IsNaN(float64(wide)) {
			continue
		}
		if got := kernel.ToBFloat16(wide).Bits(); got != uint16(bits) {
			t.Fatalf("bf16 %#04x widened to %v and came back %#04x", bits, wide, got)
		}
	}
}
