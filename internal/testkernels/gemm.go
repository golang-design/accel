// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import "golang.design/x/accel"

// GEMMDims are a matmul's shape, passed by value because they are the same for
// every invocation of the dispatch.
type GEMMDims struct {
	M, N, K uint32
}

// The tile spec 010 fixes: a 16-wide output-N tile, eight output rows, and a
// 16-wide K step. That is 128 invocations and, at f16 storage, 768 bytes of
// shared memory — which is the portable footprint, not an arbitrary choice.
const (
	TileN = 16
	TileM = 8
	TileK = 16
)

// MatMulTiled computes out = a·b with f16 storage and f32 accumulation.
//
// # Why f16 in and f32 out
//
// The storage format is what a model ships and the accumulation format is what
// a sum of hundreds of products needs. specs/008-numerics.md section 7 gives a
// dot product the error of its products plus its sum, and accumulating in f16
// would put every partial sum through a ten-bit mantissa. The narrow types
// carry no arithmetic operators at all, so Go itself forces the widening rather
// than leaving it to the author to remember.
//
// # The two barriers, and what each is for
//
// The first orders the tile loads against the reads that follow them: without
// it an invocation reads a tile slot another invocation has not written, which
// definition tracking reports. The second orders this round's reads against the
// next round's writes: without it the next iteration overwrites a tile some
// invocation is still reading, which conflicting-access reporting catches.
//
// They fail differently, and spec 009's M5 criterion asks for exactly that:
// removing either must fail, and through the diagnostic that names its own
// hazard.
//
//accel:kernel workgroup=16,8
func MatMulTiled(t accel.Thread, d GEMMDims, a []accel.Float16, b []accel.Float16,
	out []float32, tileA *[128]accel.Float16, tileB *[256]accel.Float16) {

	lx := t.LocalID().X // 0..15, along N
	ly := t.LocalID().Y // 0..7, along M
	tid := ly*TileN + lx

	row := t.GroupID().Y*TileM + ly
	col := t.GroupID().X*TileN + lx

	acc := float32(0)
	zero := accel.ToFloat16(float32(0))

	for k0 := uint32(0); k0 < d.K; k0 += TileK {
		// A's tile is 8 rows by 16 K, one element per invocation.
		if row < d.M && k0+lx < d.K {
			tileA[tid] = a[row*d.K+k0+lx]
		} else {
			tileA[tid] = zero
		}

		// B's tile is 16 K by 16 N, two elements per invocation, because 256
		// slots are filled by 128 invocations. The guard zeroes what is past
		// the edge, so the accumulation below needs no second guard.
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

		// The tile is loaded; every invocation reads a row of A and a column of
		// B out of it. The zero padding above is what makes this unguarded.
		for k := uint32(0); k < TileK; k++ {
			av := tileA[ly*TileK+k].F32()
			bv := tileB[k*TileN+lx].F32()
			acc = acc + av*bv
		}

		t.Barrier()
	}

	if row < d.M && col < d.N {
		out[row*d.N+col] = acc
	}
}

// MatMulTiledF32 is [MatMulTiled] over f32 operands.
//
// # Why both exist
//
// A consumer reported the cost of having only the f16 form: every other
// operator in the tensor layer is f32, so a transformer built on it casts
// before each projection and back after -- seven casts per layer, each a full
// pass over the activations that exists only to satisfy a dtype check. On a
// 36-layer model that is 252 extra dispatches per forward pass, on a workload
// that is memory-bound.
//
// The f16 form stays and is not the fallback: weights ship narrow, and reading
// them narrow is what halves the largest allocation after the KV cache. What
// changes is that an f32 graph no longer has to become an f16 one and back to
// multiply.
//
// The body is [MatMulTiled]'s with the tiles and the loads widened. The
// accumulator was already f32 in both, which is the whole reason the narrow
// form is safe -- see specs/002-compute-model.md.
//
//accel:kernel workgroup=16,8
func MatMulTiledF32(t accel.Thread, d GEMMDims, a []float32, b []float32,
	out []float32, tileA *[128]float32, tileB *[256]float32) {

	lx := t.LocalID().X // 0..15, along N
	ly := t.LocalID().Y // 0..7, along M
	tid := ly*TileN + lx

	row := t.GroupID().Y*TileM + ly
	col := t.GroupID().X*TileN + lx

	acc := float32(0)
	zero := float32(0)

	for k0 := uint32(0); k0 < d.K; k0 += TileK {
		// A's tile is 8 rows by 16 K, one element per invocation.
		if row < d.M && k0+lx < d.K {
			tileA[tid] = a[row*d.K+k0+lx]
		} else {
			tileA[tid] = zero
		}

		// B's tile is 16 K by 16 N, two elements per invocation, because 256
		// slots are filled by 128 invocations. The guard zeroes what is past
		// the edge, so the accumulation below needs no second guard.
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

		// The tile is loaded; every invocation reads a row of A and a column of
		// B out of it. The zero padding above is what makes this unguarded.
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
