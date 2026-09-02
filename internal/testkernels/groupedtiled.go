// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import "golang.design/x/accel"

// GroupedTiledDims is a tiled grouped product's shape.
//
// [GroupedDims] plus Tokens, and the extra field is not cosmetic: it is what
// keeps this kernel in bounds. See [GroupedMatMul].
type GroupedTiledDims struct {
	// Experts is how many weight matrices there are, so the offsets are
	// Experts+1 long. An expert with no tokens is one of them.
	Experts uint32

	// Tokens is how many rows x actually holds.
	//
	// [GroupedMatVec] needs no such field: its grid is derived from x's row
	// count, so a token index cannot leave the buffer however wrong the counts
	// are. This kernel's row index comes from the *offsets* instead, which are
	// device data, so counts summing past x's rows would walk off the end --
	// and here that is a write, not just a read. Tokens is x.shape[0], which
	// the host does know, and it is the bound the offsets cannot provide.
	Tokens uint32

	// K is the contraction width and N the output width, shared by every
	// expert: they are the same layer, differing only in weights.
	K uint32
	N uint32
}

// GroupedMatMul multiplies each expert's tokens by that expert's matrix, tiled.
//
//	out[t][n] = Σₖ x[t][k] · w[e][k][n]   for every t in expert e's segment
//
// specs/049-grouped-gemm.md §5. This is the shape a *prefill* has, where
// [GroupedMatVec] is the shape a decode step has: a decode routes one token to
// k experts and reads those matrices once, and a prefill has many tokens per
// expert and should read each matrix once per *tile* of them.
//
// # One workgroup per (expert, column tile), not per (token, column)
//
// The unit of work is an expert's whole segment rather than a token, and that
// is what makes tiling possible at all. A token-blocked grid would put rows of
// two different experts in one tile whenever a block straddled a segment
// boundary, and those rows need different weight matrices — so the tile could
// not be shared, which is the only reason to have one.
//
//	grid.X = ceil(N / TileN)     column tiles
//	grid.Y = Experts             one per matrix
//
// Each workgroup then walks its own segment in blocks of TileM. An expert
// nothing routed to has first == last and the loop does not run, which is what
// makes the zero-count case cost a workgroup launch rather than a branch.
//
// # What the tiling buys, stated exactly
//
// Each weight is read once per *token block* rather than once per token, so a
// segment of n tokens reads the matrix ceil(n/TileM) times instead of n times.
// It is not once per expert: the K loop is inside the token-block loop, so a
// new block reloads the tiles. Making it once per expert needs the whole matrix
// resident, which is a different kernel and a different constraint.
//
// # The loop bound is clamped, and that is a memory-safety guard
//
// `last` comes from the offsets, which are device data, and nothing on the host
// can check that they sum to x's row count — specs/046-segmented-extents.md §1
// property 3. Counts summing *past* x's rows would make this read x and write
// out beyond their ends. Clamping to Tokens makes that case a wrong answer
// instead of a stray write, which is the same trade §1 property 3 records for
// the other direction.
//
//accel:uniform offsets
//accel:kernel workgroup=16,8
func GroupedMatMul(t accel.Thread, d GroupedTiledDims, x []float32, w []float32,
	offsets []uint32, out []float32, tileA *[128]float32, tileB *[256]float32) {

	lx := t.LocalID().X // 0..15, along N
	ly := t.LocalID().Y // 0..7, along M
	tid := ly*TileN + lx

	e := t.GroupID().Y
	col := t.GroupID().X*TileN + lx

	first := offsets[e]
	last := offsets[e+1]
	if last > d.Tokens {
		last = d.Tokens
	}
	wBase := e * d.K * d.N

	zero := float32(0)

	for r0 := first; r0 < last; r0 += TileM {
		row := r0 + ly
		acc := float32(0)

		for k0 := uint32(0); k0 < d.K; k0 += TileK {
			if row < last && k0+lx < d.K {
				tileA[tid] = x[row*d.K+k0+lx]
			} else {
				tileA[tid] = zero
			}

			kk := tid / TileN
			nn := tid % TileN
			if k0+kk < d.K && t.GroupID().X*TileN+nn < d.N {
				tileB[tid] = w[wBase+(k0+kk)*d.N+t.GroupID().X*TileN+nn]
			} else {
				tileB[tid] = zero
			}
			kk2 := kk + TileM
			if k0+kk2 < d.K && t.GroupID().X*TileN+nn < d.N {
				tileB[tid+128] = w[wBase+(k0+kk2)*d.N+t.GroupID().X*TileN+nn]
			} else {
				tileB[tid+128] = zero
			}

			t.Barrier()

			for k := uint32(0); k < TileK; k++ {
				acc = acc + tileA[ly*TileK+k]*tileB[k*TileN+lx]
			}

			t.Barrier()
		}

		if row < last && col < d.N {
			out[row*d.N+col] = acc
		}
	}
}
