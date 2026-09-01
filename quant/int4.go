// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package quant

import (
	"fmt"
	"math"

	"golang.design/x/accel"
)

// Int4Group is how many weights share one scale and one zero point.
//
// A hundred and twenty-eight, and the reason is not [Int8Block]'s. That is 32
// because the tiled GEMM steps K in sixteen -- a tiling choice, made when the
// metadata it implied was invisible. At four bits it is not: halving the
// payload doubles the metadata's share of it, so an fp16 scale per 32 costs
// 12.5% of a 4-bit format where it costs 6.2% of an 8-bit one.
//
// A group of 128 carries *twice* as much -- a scale and a zero -- and still
// costs 6.2%, because it is amortised over four times as many weights. And 128
// is a multiple of sixteen and is exactly the row kernels' width, so the
// argument that fixed 32 permits this rather than forbidding it.
//
// It is also what AWQ and GPTQ publish, which is what lets one representation
// read the ecosystem's checkpoints and give a caller a format to quantize into.
const Int4Group = 128

// Int4Max is the largest code a 4-bit weight takes.
//
// Fifteen, and the range is [0, 15] rather than symmetric about zero: see
// [Int4Quantize] for why asymmetry is the whole design at this width.
const Int4Max = 15

// Int4Quantize packs weights into 4-bit codes with a scale and zero per group.
//
// # Why asymmetric, where int8 is symmetric
//
// [Int8Quantize] centres its range on zero, which is right at eight bits. At
// four there are sixteen codes, and spending them symmetrically means a group
// whose weights all sit near one nonzero value wastes almost all of them. So a
// group carries a scale *and* a zero point:
//
//	s = (max - min) / 15,  z = -min / s,  w ~= (q - z) * s
//
// specs/048-int4.md §3 states what that buys and costs: the error is a *range*
// over 30 where int8's is a *peak* over 254, so a group clustered away from
// zero is represented better by four asymmetric bits than by eight symmetric
// ones, and a group centred on zero is about seventeen times worse.
//
// # The packing is into words, not bytes
//
// Eight weights per uint32, low nibble first: word j holds weights 8j..8j+7
// with weight 8j+n in bits 4n..4n+3. A group of 128 is 16 words, so a group
// boundary is always a word boundary and no weight straddles one.
//
// Words rather than bytes because a kernel cannot do arithmetic on a u8 at all:
// specs/002-compute-model.md makes narrow dtypes *storage*, converted to f32 on
// load, so a shift and a mask on one is outside the subset. Byte packing would
// have needed the caller to reinterpret the slice as words on upload, which
// works only because every supported platform is little-endian -- a dependency
// worth not having when the alternative removes it. It is also what AWQ and
// GPTQ pack into, so the ecosystem's files need no repacking either.
//
// The returned slice is ceil(len(w)/8) words; at a length that is not a
// multiple of eight the last word's high nibbles are zero and read back as
// weights nobody asked for, which is why [Int4Dequantize] takes the count
// rather than deriving it.
func Int4Quantize(w []float32) (packed []uint32, scales, zeros []accel.Float16) {
	groups := (len(w) + Int4Group - 1) / Int4Group
	packed = make([]uint32, (len(w)+7)/8)
	scales = make([]accel.Float16, groups)
	zeros = make([]accel.Float16, groups)

	for g := range groups {
		lo := g * Int4Group
		hi := min(lo+Int4Group, len(w))

		lowest, highest := w[lo], w[lo]
		for _, v := range w[lo:hi] {
			if v < lowest {
				lowest = v
			}
			if v > highest {
				highest = v
			}
		}

		// A group whose weights are all one value, which a pruned or padded
		// matrix really has. The range is zero, so the scale is zero and
		// dividing by it would make every code a NaN. The zero point carries
		// the constant instead, and [Int4Dequantize] reads a zero scale as
		// exactly that.
		if highest == lowest {
			scales[g] = accel.ToFloat16(0)
			zeros[g] = accel.ToFloat16(lowest)
			continue
		}

		s := (highest - lowest) / Int4Max
		// Stored as f16 and read back before quantizing against it, for
		// [Int8Quantize]'s reason: quantizing against the f32 scale and storing
		// a rounded one leaves the kernel reconstructing with a different
		// number than the quantizer used.
		scales[g] = accel.ToFloat16(s)
		s16 := scales[g].F32()
		if s16 == 0 {
			// The scale underflowed f16. Every weight in the group is within
			// f16's smallest normal of every other, so the constant branch
			// above is the honest reconstruction.
			zeros[g] = accel.ToFloat16(lowest)
			continue
		}
		z := -lowest / s16
		zeros[g] = accel.ToFloat16(z)
		z16 := zeros[g].F32()

		for i := lo; i < hi; i++ {
			q := math.RoundToEven(float64(w[i])/float64(s16) + float64(z16))
			// Two roundings feed this, so the clamp catches what neither can.
			// s and z are both stored as f16 and read back, so the weight equal
			// to the group's maximum need not land on 15: it can reach 16,
			// which four bits cannot hold and which wraps to 0 -- turning the
			// largest weight in the group into the smallest. The low end wraps
			// the other way for the same reason.
			if q > Int4Max {
				q = Int4Max
			}
			if q < 0 {
				q = 0
			}
			setNibble(packed, i, uint32(q))
		}
	}
	return packed, scales, zeros
}

// setNibble writes one 4-bit code at index i.
func setNibble(packed []uint32, i int, q uint32) {
	shift := 4 * (i % 8)
	packed[i/8] = packed[i/8]&^(0xF<<shift) | q<<shift
}

// Int4Nibble reads the 4-bit code at index i.
//
// Exported because it is the one piece of this representation a kernel and a
// caller must spell identically, and so the place where two implementations
// disagree without either being obviously wrong.
func Int4Nibble(packed []uint32, i int) uint32 {
	return packed[i/8] >> (4 * (i % 8)) & 0xF
}

// Int4Dequantize reconstructs the weights a packed group represents.
//
// n is the weight count rather than derived from len(packed), because a count
// that is not a multiple of eight leaves the last word's high nibbles holding
// codes nobody wrote.
func Int4Dequantize(packed []uint32, scales, zeros []accel.Float16, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		g := i / Int4Group
		s := scales[g].F32()
		z := zeros[g].F32()
		if s == 0 {
			// The constant group: the zero point carries the value, because a
			// zero scale annihilates the code.
			out[i] = z
			continue
		}
		out[i] = (float32(Int4Nibble(packed, i)) - z) * s
	}
	return out
}

// Int4ErrorBound is the quantization term of a dot product, over the inputs it
// used and the scale and zero point the quantizer stored.
//
// Derived, not observed, and derived from what is *stored* rather than from the
// range: the code is rounded against the f16 scale s and zero z the group
// carries, so rounding the code costs at most s/2 -- but s and z are
// themselves rounded from s* = range/15 and z* = -min/s, and a weight at
// either end of the group can land outside [0, 15] by that rounding and be
// clamped. Clamping is not a half-step error; it is the whole excess. Per
// weight,
//
//	|w - (q-z)s|  <=  s/2 + 15|s* - s| + |z* - z| s
//
// and each rounding is at most half an ulp of the stored f16 plus the f32
// arithmetic that produced the value being rounded. A group stored with a zero
// scale is a constant carried in z, and its only error is z's own rounding.
// Per dot product the terms are weighted by |x_i| and summed.
//
// specs/048-int4.md §3 states the bound as range over thirty, which is the
// first term alone and is what this used to compute. It is exceeded where the
// zero point is large: weights in [1000, 1001] store z = -15000, whose f16
// spacing is 8, so the group is reconstructed with a bias of up to 4s = 0.27
// against a claimed 0.033. That is what this bound covers and that did not.
//
// The stored scale and zero are taken per term rather than per group, for
// [Int8ErrorBound]'s reason: a per-group figure bounds a dot product only
// where the caller says which group each term came from.
//
// **This covers quantization only.** The products are summed in f32, so
// specs/008-numerics.md section 7's reduction bound applies to the sum on top,
// and a caller comparing against an exact reference adds the two.
//
// Panics when len(termScales) or len(termZeros) differs from len(x). A
// mismatch is a caller passing per-group figures where per-term ones are
// required, which is the mistake this signature exists to refuse, and the
// number it would otherwise return is not a bound.
func Int4ErrorBound(x []float32, termScales, termZeros []accel.Float16) float64 {
	if len(termScales) != len(x) || len(termZeros) != len(x) {
		panic(fmt.Sprintf("accel/quant: Int4ErrorBound has %d terms, %d scales and %d "+
			"zeros; it takes one scale and one zero per term, because a per-group figure "+
			"is a bound only where the caller says which group each term came from",
			len(x), len(termScales), len(termZeros)))
	}
	// f32 unit roundoff, for the divisions the quantizer performs before
	// rounding to f16: s* = range/15 is a subtraction and a division, z* =
	// -min/s is a division, each within u of exact relative to its result.
	const u = 1.0 / (1 << 24)
	var sum float64
	for i, xi := range x {
		s := float64(termScales[i].F32())
		z := float64(termZeros[i].F32())
		var e float64
		if s == 0 {
			e = ulp16(termZeros[i]) / 2
		} else {
			ds := ulp16(termScales[i])/2 + 2*u*s
			dz := ulp16(termZeros[i])/2 + u*math.Abs(z)
			e = s/2 + 15*ds + dz*s
		}
		sum += math.Abs(float64(xi)) * e
	}
	return sum
}

// ulp16 is the spacing of f16 values at f, so half of it bounds the rounding
// that produced f. Infinite for a non-finite f: a group whose zero point
// overflowed f16 reconstructs nothing, and a bound on it is not a number.
func ulp16(f accel.Float16) float64 {
	exp := int(f.Bits()>>10) & 0x1f
	switch exp {
	case 0x1f:
		return math.Inf(1)
	case 0:
		return math.Ldexp(1, -24)
	}
	return math.Ldexp(1, exp-25)
}
