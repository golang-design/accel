// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import "golang.design/x/accel"

// QuantMatMulInt4 multiplies f32 activations by an asymmetric 4-bit matrix.
//
//	out[m][n] = Σₖ a[m][k] · ((nibble(bq, k·N+n) − z[g]) · s[g]),  g = (k·N+n)/Int4Group
//
// specs/048-int4.md §5. This is the shape a *prefill* has, where
// [QuantMatVecInt4] is the shape a decode step has: several rows of activations
// against one weight matrix, so the matrix is read once for a whole tile of
// tokens rather than once per token.
//
// # The dequantization moves to the tile load
//
// [MatMulTiledF32]'s body, with one change that is the whole reason this is a
// separate kernel rather than a wrapper. The B tile is filled by *unpacking*,
// so each weight is decoded once per tile and then read TileM times out of
// shared memory as an ordinary f32:
//
//	packed word ──▶ nibble ──▶ (code − z)·s ──▶ tileB ──▶ TileM reads
//
// In the matvec there is nothing to amortise, so it decodes in the inner loop.
// Here the inner loop is a plain multiply-add over two shared tiles, identical
// to the f32 kernel, and the arithmetic that makes four bits four bits happens
// once per element instead of once per use.
//
// # The nibble order is the same contract
//
// Weight i lives in word i/8 at nibble i%8, low first. This is the fourth place
// that order is written down — quant.Int4Quantize packs it, quant.Int4Nibble
// reads it, [QuantMatVecInt4] decodes it, and this decodes it again. A variant
// that disagrees round-trips perfectly against itself and produces noise
// against the others, which is why the differential against the matvec is what
// checks it rather than a golden.
//
// # Zero padding past the edge
//
// # Zero padding past the edge
//
// The guards write a zero rather than skipping, exactly as [MatMulTiledF32]
// does, so the accumulation needs no second guard. Two things about that zero
// are worth stating, because one of them is not what it looks like.
//
// It is written rather than left alone so the read below is *defined*. Skipping
// leaves the slot holding the previous round's value, which definition tracking
// reports.
//
// It is a literal f32 zero and not a dequantized code zero, and that choice is
// **not observable today**. Code zero reconstructs to −z·s, a nonzero weight,
// so it would bias the accumulator — except that the A tile is zeroed at the
// same k, and the product of the two is zero whatever this holds. The two pads
// are mutually redundant, and it takes breaking both to change an answer.
// Verified by mutation, which is why the claim is stated this narrowly rather
// than as "the padding prevents a bias": it does not, on its own, and a comment
// saying so would send the next reader looking for a test that cannot exist.
//
// The f32 zero is kept because it is correct without depending on the A tile's
// guard, and the two guards are written a hundred lines apart.
//
//accel:kernel workgroup=16,8
func QuantMatMulInt4(t accel.Thread, d GEMMDims, a []float32, bq []uint32,
	bs []accel.Float16, bz []accel.Float16, out []float32,
	tileA *[128]float32, tileB *[256]float32) {

	lx := t.LocalID().X // 0..15, along N
	ly := t.LocalID().Y // 0..7, along M
	tid := ly*TileN + lx

	row := t.GroupID().Y*TileM + ly
	col := t.GroupID().X*TileN + lx

	acc := float32(0)
	zero := float32(0)

	for k0 := uint32(0); k0 < d.K; k0 += TileK {
		if row < d.M && k0+lx < d.K {
			tileA[tid] = a[row*d.K+k0+lx]
		} else {
			tileA[tid] = zero
		}

		// B's tile is 16 K by 16 N, two elements per invocation, and each is
		// unpacked from its nibble as it lands.
		kk := tid / TileN
		nn := tid % TileN
		if k0+kk < d.K && t.GroupID().X*TileN+nn < d.N {
			w := (k0+kk)*d.N + t.GroupID().X*TileN + nn
			code := bq[w/8] >> (4 * (w % 8)) & 0xF
			g := w / Int4Group
			s := bs[g].F32()
			z := bz[g].F32()
			// A zero scale is a constant group and the zero point carries
			// it; [QuantMatVecInt4] says why, and this is the same reading.
			wv := (float32(code) - z) * s
			if s == 0 {
				wv = z
			}
			tileB[tid] = wv
		} else {
			tileB[tid] = zero
		}
		kk2 := kk + TileM
		if k0+kk2 < d.K && t.GroupID().X*TileN+nn < d.N {
			w := (k0+kk2)*d.N + t.GroupID().X*TileN + nn
			code := bq[w/8] >> (4 * (w % 8)) & 0xF
			g := w / Int4Group
			s2 := bs[g].F32()
			z2 := bz[g].F32()
			wv2 := (float32(code) - z2) * s2
			if s2 == 0 {
				wv2 = z2
			}
			tileB[tid+128] = wv2
		} else {
			tileB[tid+128] = zero
		}

		t.Barrier()

		for k := uint32(0); k < TileK; k++ {
			av := tileA[ly*TileK+k]
			bv := tileB[k*TileN+lx]
			acc = acc + av*bv
		}

		t.Barrier()
	}

	if row < d.M && col < d.N {
		out[row*d.N+col] = acc
	}
}
