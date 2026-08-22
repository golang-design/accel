// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel

import "math"

// Float16 is a 16-bit float, stored.
//
// It is not named F16, because that is the [DType] constant naming the same
// format. A dtype is metadata a descriptor carries and a storage type is what a
// kernel parameter is made of, and both wanted the short name; the constant
// shipped first and appears at every allocation site. See
// specs/004-kernel-authoring.md.
//
// # Why a struct and not a defined integer type
//
// A defined type over uint16 would carry uint16's operators, so `a + b` on two
// narrow values would compile and add their *bit patterns*. That is a
// wrong-answer bug with no diagnostic anywhere. A struct has no operators at
// all, so Go itself forces [Float16.F32] on the way in and [ToFloat16] on the way out,
// and f32 accumulation stops being a convention a kernel author has to
// remember and becomes the only thing that compiles.
//
// Narrow types are storage, not arithmetic. Every backend can store one, by bit
// packing where it has no native type, and native narrow arithmetic is a
// separate capability that spec 002 gates behind CapF16Arithmetic. Making the
// conversion explicit is what lets one kernel run everywhere.
type Float16 struct{ bits uint16 }

// BFloat16 is a 16-bit float with f32's exponent range and less mantissa.
//
// It is the upper half of an f32 plus a rounding adjustment, which is why it
// trades precision for range at the same width. Truncating to the upper half is
// not the conversion: spec 008 section 4 requires round-to-nearest-even, and
// truncation is round-toward-zero wearing a different name.
type BFloat16 struct{ bits uint16 }

// Bits returns the storage bits.
func (f Float16) Bits() uint16 { return f.bits }

// Bits returns the storage bits.
func (b BFloat16) Bits() uint16 { return b.bits }

// Float16FromBits reinterprets storage bits, without converting.
func Float16FromBits(b uint16) Float16 { return Float16{bits: b} }

// BFloat16FromBits reinterprets storage bits, without converting.
func BFloat16FromBits(b uint16) BFloat16 { return BFloat16{bits: b} }

// Canonical quiet NaN encodings. Spec 008 section 4 canonicalizes a narrowing
// NaN rather than propagating its payload, because a payload is not a
// cross-backend guarantee and a test that asserted one would be asserting what
// a particular driver happens to do.
const (
	quietNaNF16  = 0x7e00
	quietNaNBF16 = 0x7fc0
)

// ToFloat16 converts an f32, rounding to nearest with ties to even.
//
// Overflow goes to a signed infinity rather than to the largest finite value,
// and a NaN becomes the canonical quiet encoding. Signed zero survives: a
// negative zero that came back positive would change the sign of a later
// division, which is the kind of difference nobody looks for.
func ToFloat16(x float32) Float16 {
	b := math.Float32bits(x)
	sign := uint16(b >> 16 & 0x8000)
	exp := int32(b>>23&0xff) - 127
	mant := b & 0x7fffff

	switch {
	case b&0x7fffffff > 0x7f800000: // NaN
		return Float16{bits: quietNaNF16}
	case exp == 128: // infinity
		return Float16{bits: sign | 0x7c00}
	case exp > 15: // overflows f16's range
		return Float16{bits: sign | 0x7c00}
	case exp < -24: // rounds to zero, keeping the sign
		return Float16{bits: sign}
	case exp < -14:
		// Subnormal. The mantissa gains its implicit bit and is shifted right by
		// however far the exponent is below the smallest normal, so the rounding
		// point moves with it. Dropping subnormals instead would flush a whole
		// range of small values to zero, which spec 002 makes a reported
		// capability rather than something a conversion decides.
		mant |= 0x800000
		shift := uint32(-exp - 14)
		return Float16{bits: sign | uint16(roundShift(mant, 13+shift))}
	default:
		half := uint32(exp+15)<<10 | roundShift(mant, 13)
		// Rounding the mantissa can carry into the exponent, and if that carries
		// out of the exponent the result is an infinity. Letting it wrap would
		// turn the largest finite f16 into a zero.
		return Float16{bits: sign | uint16(half)}
	}
}

// F32 widens exactly. Widening never rounds, so this is the direction with no
// contract to get wrong beyond the special values.
func (f Float16) F32() float32 {
	sign := uint32(f.bits&0x8000) << 16
	exp := uint32(f.bits >> 10 & 0x1f)
	mant := uint32(f.bits & 0x3ff)

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign) // signed zero
		}
		// Subnormal f16, normal f32: shift the mantissa up until its leading bit
		// falls off, and lower the exponent to match.
		shift := uint32(0)
		for mant&0x400 == 0 {
			mant <<= 1
			shift++
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | (127-15+1-shift)<<23 | mant<<13)
	case 0x1f:
		if mant == 0 {
			return math.Float32frombits(sign | 0x7f800000) // signed infinity
		}
		// A quiet NaN widens with its payload where it fits, which spec 008
		// section 4 permits in this direction because widening cannot lose it.
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	default:
		return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
	}
}

// ToBFloat16 converts an f32, rounding to nearest with ties to even.
//
// The rounding is the whole content of this function. Truncating the low 16
// bits is what a naive implementation does, and it is round-toward-zero, which
// biases every accumulation in one direction. Spec 008 section 4 requires
// round-to-nearest-even, and a bias that small compounds over a reduction.
func ToBFloat16(x float32) BFloat16 {
	b := math.Float32bits(x)
	if b&0x7fffffff > 0x7f800000 { // NaN
		return BFloat16{bits: quietNaNBF16}
	}
	// Round to nearest, ties to even: add half an ulp, plus one more when the
	// value is already odd, then truncate.
	rounded := b + 0x7fff + (b >> 16 & 1)
	return BFloat16{bits: uint16(rounded >> 16)}
}

// F32 widens exactly: a bf16 is the upper half of an f32 by construction.
func (b BFloat16) F32() float32 { return math.Float32frombits(uint32(b.bits) << 16) }

// roundShift shifts right by n, rounding to nearest with ties to even.
func roundShift(v, n uint32) uint32 {
	if n == 0 {
		return v
	}
	if n > 31 {
		return 0
	}
	shifted := v >> n
	half := uint32(1) << (n - 1)
	rest := v & (1<<n - 1)
	if rest > half || (rest == half && shifted&1 == 1) {
		shifted++
	}
	return shifted
}
