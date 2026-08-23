// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package quant turns weights into the form a quantized kernel reads.
//
// A quantized weight matrix is two arrays rather than one: the quants, one
// small integer per weight, and the scales, one per block of weights. You get
// both from [Int8], hand them to the device as an I8 buffer and an F16 buffer,
// and a quantized operator reads them together.
//
//	q, s := quant.Int8(weights)
//
// # What it costs you
//
// Accuracy, by a bounded amount. A weight comes back as
// [Int8Block] values times its block's scale, which is within half a scale step
// of what you put in — so the error is proportional to the *largest* weight in
// each block, and a block containing one outlier represents its other 31
// weights worse than a block without one. [Error] computes the bound for a dot
// product so a test can assert against it rather than against a tolerance
// somebody tuned.
//
// # What it saves you
//
// Half the memory of f16 and a quarter of f32, and on a memory-bound decode
// step that is most of the run time.
//
// specs/027-quantization.md has the derivation and the reasoning.
package quant

import (
	"math"

	"golang.design/x/accel"
)

// Int8Block is how many weights share one scale.
//
// Thirty-two, because specs/010-kernel-corpus.md's tiled GEMM steps K in
// sixteen and a multiple of that means no K-step ever straddles a scale
// boundary: a step reads one scale rather than two. The row kernels are 128
// wide, also a multiple.
const Int8Block = 32

// Int8Max is the largest magnitude a quant takes.
//
// 127 rather than 128, so the representable range is symmetric about zero.
// int8 reaches -128, and using it would make -128*s a value with no positive
// counterpart -- which is exactly the special case the error bound in
// specs/027-quantization.md is stated without.
const Int8Max = 127

// Int8 quantizes weights into per-block scaled integers.
//
// The returned quants are one per weight and the scales one per [Int8Block] of
// them, padded when the length is not a multiple: a trailing partial block gets
// its own scale over the weights it has.
func Int8(w []float32) (quants []int8, scales []accel.Float16) {
	blocks := (len(w) + Int8Block - 1) / Int8Block
	quants = make([]int8, len(w))
	scales = make([]accel.Float16, blocks)

	for b := range blocks {
		lo := b * Int8Block
		hi := min(lo+Int8Block, len(w))

		var peak float32
		for _, v := range w[lo:hi] {
			if a := float32(math.Abs(float64(v))); a > peak {
				peak = a
			}
		}

		// A block of all zeros, which a pruned or padded matrix really has.
		// Dividing by a zero scale would make every quant a NaN and every
		// reconstruction one too; zero scale with zero quants reconstructs
		// exactly, because 0*0 is 0.
		if peak == 0 {
			scales[b] = accel.ToFloat16(0)
			continue
		}

		s := peak / Int8Max
		// The scale is stored as f16 and *read back* before quantizing against
		// it. Quantizing against the f32 scale and storing a rounded one would
		// leave the kernel reconstructing with a different number than the
		// quantizer used, which is a second error nobody accounted for.
		scales[b] = accel.ToFloat16(s)
		s16 := scales[b].F32()
		if s16 == 0 {
			// The scale underflowed f16. Every weight in the block is below
			// f16's smallest normal divided by 127, so zero is the honest
			// reconstruction and the alternative is a division by zero.
			continue
		}

		for i := lo; i < hi; i++ {
			q := math.RoundToEven(float64(w[i]) / float64(s16))
			// The clamp is load-bearing. round(w/s) reaches 127 at the peak,
			// and rounding can push a weight one ulp below it to 128 -- which
			// wraps to -128 in an int8 and turns the largest weight in the
			// block into the most negative one.
			if q > Int8Max {
				q = Int8Max
			}
			if q < -Int8Max {
				q = -Int8Max
			}
			quants[i] = int8(q)
		}
	}
	return quants, scales
}

// Dequantize reconstructs the weights a quantized pair represents.
//
// The reference for what a kernel must compute, and the thing to compare
// against when asking how much accuracy a quantization cost.
func Dequantize(quants []int8, scales []accel.Float16) []float32 {
	out := make([]float32, len(quants))
	for i := range quants {
		out[i] = float32(quants[i]) * scales[i/Int8Block].F32()
	}
	return out
}

// Error bounds how far a quantized dot product may sit from the exact one.
//
// # Why a bound rather than a tolerance
//
// specs/008-numerics.md admits no tolerance parameter anywhere, and this is why:
// a tolerance is a number somebody raised until a test passed. This is derived
// from the representation. Rounding to nearest puts each weight within half a
// scale step of its original, a step is the block's largest magnitude over 127,
// and a dot product weights each of those errors by the activation it
// multiplies:
//
//	|Σ qᵢsᵢxᵢ − Σ wᵢxᵢ| ≤ (1/254) Σ |xᵢ| · max|w in block(i)|
//
// It is an absolute bound over the actual inputs, so a test computes it from
// the values it used rather than guessing a relative one.
//
// **This covers quantization only.** The products are summed in f32, so
// specs/008-numerics.md section 7's reduction bound applies to the sum on top,
// and a caller comparing against an exact reference adds the two.
func Error(x []float32, scales []accel.Float16) float64 {
	var sum float64
	for i, xi := range x {
		s := float64(scales[i/Int8Block].F32())
		sum += math.Abs(float64(xi)) * s / 2
	}
	return sum
}
