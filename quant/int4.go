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
// # The packing
//
// Two weights per byte, low nibble first: byte b holds weight 2b in its low
// nibble and 2b+1 in its high one. A group of 128 is 64 bytes, so a group
// boundary is always a byte boundary and no weight straddles one.
//
// The returned slice is ceil(len(w)/2) bytes; at an odd length the last byte's
// high nibble is zero and reads back as a weight nobody asked for, which is why
// [Int4Dequantize] takes the length rather than deriving it.
func Int4Quantize(w []float32) (packed []uint8, scales, zeros []accel.Float16) {
	groups := (len(w) + Int4Group - 1) / Int4Group
	packed = make([]uint8, (len(w)+1)/2)
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
			setNibble(packed, i, uint8(q))
		}
	}
	return packed, scales, zeros
}

// setNibble writes one 4-bit code at index i.
func setNibble(packed []uint8, i int, q uint8) {
	if i%2 == 0 {
		packed[i/2] = packed[i/2]&0xF0 | q
	} else {
		packed[i/2] = packed[i/2]&0x0F | q<<4
	}
}

// Int4Nibble reads the 4-bit code at index i.
//
// Exported because it is the one piece of this representation a kernel and a
// caller must spell identically, and so the place where two implementations
// disagree without either being obviously wrong.
func Int4Nibble(packed []uint8, i int) uint8 {
	return packed[i/2] >> (4 * (i % 2)) & 0x0F
}

// Int4Dequantize reconstructs the weights a packed group represents.
//
// n is the weight count rather than derived from len(packed), because an odd
// count leaves the last byte's high nibble holding a code nobody wrote.
func Int4Dequantize(packed []uint8, scales, zeros []accel.Float16, n int) []float32 {
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
// used.
//
// specs/048-int4.md §3. Derived, not observed: rounding to nearest gives at
// most half a step, and the step is a group's *range* over fifteen where
// [Int8ErrorBound]'s is a peak over 127.
//
//	|sum (q-z)s x - sum w x|  <=  (1/30) sum |x_i| (max_g(i) - min_g(i))
//
// termRanges is one group range per term, for [Int8ErrorBound]'s reason: a
// per-group figure bounds a dot product only where the caller says which group
// each term came from.
func Int4ErrorBound(x []float32, termRanges []float32) float64 {
	if len(termRanges) != len(x) {
		panic(fmt.Sprintf("accel/quant: Int4ErrorBound has %d terms and %d ranges; "+
			"it takes one range per term, because a per-group range is a bound only "+
			"where the caller says which group each term came from", len(x), len(termRanges)))
	}
	var sum float64
	for i, xi := range x {
		sum += math.Abs(float64(xi)) * float64(termRanges[i]) / 30
	}
	return sum
}
