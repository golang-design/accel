// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import "golang.design/x/accel"

// QuantBlock is how many weights share one scale, and must equal
// quant.Int8Block.
//
// Declared here rather than imported because a kernel is compiled from a
// closed subset that has no imports beyond accel and kmath. The two are checked
// against each other by a test, which is the only thing that can keep them
// equal.
const QuantBlock = 32

// QuantMatMul multiplies f16 activations by int8 quantized weights.
//
//	out[m,n] = Σₖ a[m,k] · (bq[k,n] · bs[(k·N+n)/32])
//
// # Why the products widen to f32 before accumulating
//
// The obvious quantized GEMM accumulates in int32 and scales once at the end,
// which is faster and is what CapI8DotProduct exists for. It is also wrong
// here: the scale varies per block, so an integer accumulator would be summing
// quants that mean different things. Widening each product to f32 keeps every
// term in the same units, and specs/027-quantization.md's error bound is stated
// against exactly this evaluation.
//
// An integer-accumulating variant is possible for the special case of one scale
// per output column, and is a different kernel.
//
// # Why the scale index is computed per element
//
// Because the block is over the *flattened* weight array. specs/027-quantization.md
// fixes 32 as a multiple of the tiled GEMM's K-step so no step straddles a
// boundary, but this variant is the straightforward one: it recomputes the index
// rather than hoisting it, so the correctness of the layout is visible in the
// expression rather than in a loop invariant.
//
//accel:kernel workgroup=64
func QuantMatMul(t accel.Thread, d GEMMDims, a []accel.Float16, bq []int8,
	bs []accel.Float16, out []float32) {

	i := t.GlobalID().X
	if i < d.M*d.N {
		row := i / d.N
		col := i % d.N

		acc := float32(0)
		for k := uint32(0); k < d.K; k++ {
			w := k*d.N + col
			q := float32(bq[w])
			s := bs[w/QuantBlock].F32()
			acc = acc + a[row*d.K+k].F32()*(q*s)
		}
		out[i] = acc
	}
}

// QuantRows gathers rows of a quantized table.
//
// An embedding table is the largest single tensor in a small model and the one
// quantization helps most, because every token reads one row of it and nothing
// else.
//
//accel:kernel workgroup=64
func QuantRows(t accel.Thread, p RowParams, tq []int8, ts []accel.Float16,
	ids []uint32, out []float32) {

	i := t.GlobalID().X
	if i < p.Rows*p.Width {
		r := i / p.Width
		c := i % p.Width
		id := ids[r]
		if id < p.Capacity {
			w := id*p.Width + c
			out[i] = float32(tq[w]) * ts[w/QuantBlock].F32()
		} else {
			out[i] = float32(0)
		}
	}
}

// QuantMatMulF32 is [QuantMatMul] over f32 activations.
//
//	out[m,n] = Σₖ a[m,k] · (bq[k,n] · bs[(k·N+n)/32])
//
// # Why the activation width is the thing that varies
//
// An activation is produced by the graph and a weight is loaded from a file, so
// they answer to different pressures. Every other operator in the tensor layer
// produces f32, and int8 is the width a model reaches for exactly because it is
// too large to hold otherwise. Requiring the two to agree therefore inserted a
// Cast in front of every projection of the configuration least able to afford
// one (accel issue 14).
//
// The body is [QuantMatMul]'s with the activation load already wide. Nothing
// else moves: the products still widen to f32 before accumulating, because the
// scale varies per block and an integer accumulator would be summing quants
// that mean different things, and specs/027-quantization.md's error bound is
// stated against this evaluation. On activations f16 holds exactly the two
// kernels agree bit for bit, which is what the test asserts.
//
//accel:kernel workgroup=64
func QuantMatMulF32(t accel.Thread, d GEMMDims, a []float32, bq []int8,
	bs []accel.Float16, out []float32) {

	i := t.GlobalID().X
	if i < d.M*d.N {
		row := i / d.N
		col := i % d.N

		acc := float32(0)
		for k := uint32(0); k < d.K; k++ {
			w := k*d.N + col
			q := float32(bq[w])
			s := bs[w/QuantBlock].F32()
			acc = acc + a[row*d.K+k]*(q*s)
		}
		out[i] = acc
	}
}
