// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import "golang.design/x/accel"

// MatVec computes out = a·b where a is a single row.
//
// # Why M=1 gets its own kernel
//
// It is the decode shape: one token against a weight matrix, which is every
// step after the first in an autoregressive model. The tiled GEMM handles it
// correctly and wastes seven eighths of each tile, because [TileM] is eight and
// only one row is live. Spec 010 makes this a separate semantic ID selected
// exactly when M=1 for that reason, and the selection is deterministic rather
// than heuristic so that two runs of the same model dispatch the same kernel.
//
// One workgroup per output column-block, each invocation folding a strided slice
// of K and the tree reducing the partials — the same shape as the other
// reductions here, and the same argument: depth seven rather than K-1.
//
//accel:kernel workgroup=128
func MatVec(t accel.Thread, d GEMMDims, a []accel.Float16, b []accel.Float16,
	out []float32, sh *[128]float32) {

	col := t.GroupID().X
	lid := t.LocalID().X

	acc := float32(0)
	if col < d.N {
		for k := lid; k < d.K; k += RowWidth {
			acc = acc + a[k].F32()*b[k*d.N+col].F32()
		}
	}
	sh[lid] = acc
	t.Barrier()

	for stride := uint32(RowWidth / 2); stride > 0; stride /= 2 {
		if lid < stride {
			sh[lid] = sh[lid] + sh[lid+stride]
		}
		t.Barrier()
	}

	if lid == 0 && col < d.N {
		out[col] = sh[0]
	}
}

// LinearTiled is [MatMulTiled] with a bias added once per output.
//
// # Why the bias is an epilogue rather than a separate kernel
//
// Adding it in a second pass reads and writes the whole output again, which for
// a decode step is more traffic than the matmul itself. Spec 010 gives it a
// distinct stable ID sharing the tile body for exactly that reason: the same
// arithmetic, one fused store.
//
// The bias is broadcast along M, which is what "broadcast-compatible" means
// here: one value per output column, added to every row.
//
//accel:kernel workgroup=16,8
func LinearTiled(t accel.Thread, d GEMMDims, a []accel.Float16, b []accel.Float16,
	bias []float32, out []float32, tileA *[128]accel.Float16, tileB *[256]accel.Float16) {

	lx := t.LocalID().X
	ly := t.LocalID().Y
	tid := ly*TileN + lx

	row := t.GroupID().Y*TileM + ly
	col := t.GroupID().X*TileN + lx

	acc := float32(0)
	zero := accel.ToFloat16(float32(0))

	for k0 := uint32(0); k0 < d.K; k0 += TileK {
		if row < d.M && k0+lx < d.K {
			tileA[tid] = a[row*d.K+k0+lx]
		} else {
			tileA[tid] = zero
		}

		kk := tid / TileN
		nn := tid % TileN
		if k0+kk < d.K && t.GroupID().X*TileN+nn < d.N {
			tileB[tid] = b[(k0+kk)*d.N+t.GroupID().X*TileN+nn]
		} else {
			tileB[tid] = zero
		}
		kk2 := kk + TileM
		if k0+kk2 < d.K && t.GroupID().X*TileN+nn < d.N {
			tileB[tid+128] = b[(k0+kk2)*d.N+t.GroupID().X*TileN+nn]
		} else {
			tileB[tid+128] = zero
		}

		t.Barrier()

		for k := uint32(0); k < TileK; k++ {
			av := tileA[ly*TileK+k].F32()
			bv := tileB[k*TileN+lx].F32()
			acc = acc + av*bv
		}

		t.Barrier()
	}

	// The epilogue. One fused store rather than a second pass over the output.
	if row < d.M && col < d.N {
		out[row*d.N+col] = acc + bias[col]
	}
}
