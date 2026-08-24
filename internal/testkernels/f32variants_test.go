// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"fmt"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
	"golang.design/x/accel/kernelabi"
)

// The f32 GEMM matches the f16 one on operands both can hold exactly.
//
// # Why compare the two rather than compute a reference twice
//
// The f32 kernel is the f16 one with the tiles and the loads widened, so what
// is worth checking is that widening changed nothing else — the guarded tails,
// the tile indexing, the accumulation order. On operands exact in f16 the two
// must agree *bit for bit*, and any difference is a transcription error in the
// widening rather than a numeric one.
//
// The dimensions are the criterion, as in the f16 test: a GEMM at 16x8x16
// exercises none of the guarded tails, and every tail is a place an off-by-one
// produces a plausible matrix.
func TestTheF32GEMMMatchesTheF16OneExactly(t *testing.T) {
	for _, c := range []struct{ m, n, k int }{
		{8, 16, 16},  // exactly one tile
		{1, 1, 1},    // every dimension a partial tile
		{3, 5, 7},    // all three tails, none aligned
		{8, 16, 33},  // a K tail
		{9, 16, 16},  // an M tail
		{8, 17, 16},  // an N tail
		{17, 19, 23}, // all three, larger than one tile each
	} {
		t.Run(fmt.Sprintf("%dx%dx%d", c.m, c.n, c.k), func(t *testing.T) {
			// Small integers over a power of two: exact in f16 and in f32, so
			// the two kernels read the same values and any difference is the
			// widening rather than the storage.
			af, bf := make([]float32, c.m*c.k), make([]float32, c.k*c.n)
			a16, b16 := make([]accel.Float16, len(af)), make([]accel.Float16, len(bf))
			for i := range af {
				af[i] = float32((i%13)-6) / 4
				a16[i] = accel.ToFloat16(af[i])
			}
			for i := range bf {
				bf[i] = float32((i%11)-5) / 2
				b16[i] = accel.ToFloat16(bf[i])
			}

			dims := testkernels.GEMMDims{M: uint32(c.m), N: uint32(c.n), K: uint32(c.k)}
			groups := accel.ID3{
				X: uint32((c.n + testkernels.TileN - 1) / testkernels.TileN),
				Y: uint32((c.m + testkernels.TileM - 1) / testkernels.TileM),
				Z: 1,
			}
			narrow := make([]float32, c.m*c.n)
			wide := make([]float32, c.m*c.n)
			if err := kernel.DispatchCooperative(&testkernels.MatMulTiledKernel, groups,
				kernelabi.Args{Slices: []any{a16, b16, narrow}, Uniforms: []any{dims}}); err != nil {
				t.Fatalf("f16 dispatch: %v", err)
			}
			if err := kernel.DispatchCooperative(&testkernels.MatMulTiledF32Kernel, groups,
				kernelabi.Args{Slices: []any{af, bf, wide}, Uniforms: []any{dims}}); err != nil {
				t.Fatalf("f32 dispatch: %v", err)
			}
			for i := range narrow {
				if narrow[i] != wide[i] {
					t.Fatalf("element %d is %v narrow and %v wide; on operands exact in "+
						"f16 the two kernels must agree bit for bit, so this is a "+
						"transcription error in the widening", i, narrow[i], wide[i])
				}
			}
		})
	}
}

// The f16 attention kernel matches the f32 one on a cache exact in f16.
//
// Same argument: the body is the f32 one with two loads widened, so on values
// f16 holds exactly the two must agree bit for bit. A numeric comparison would
// pass on a kernel that read the wrong element; this does not.
func TestTheF16AttentionMatchesTheF32OneExactly(t *testing.T) {
	for _, c := range []struct{ qHeads, kvHeads, headDim, kvLen uint32 }{
		{2, 1, 8, 3},
		{4, 2, 8, 6},
		{6, 3, 16, 5},
	} {
		t.Run(fmt.Sprintf("q%d_kv%d_d%d_len%d", c.qHeads, c.kvHeads, c.headDim, c.kvLen),
			func(t *testing.T) {
				d := testkernels.AttnDims{
					QHeads: c.qHeads, KVHeads: c.kvHeads, HeadDim: c.headDim,
					Scale: float32(1 / math.Sqrt(float64(c.headDim))),
				}
				lengths := []uint32{c.kvLen}

				q := make([]float32, c.qHeads*c.headDim)
				kf := make([]float32, c.kvLen*c.kvHeads*c.headDim)
				vf := make([]float32, len(kf))
				k16 := make([]accel.Float16, len(kf))
				v16 := make([]accel.Float16, len(vf))
				for i := range q {
					q[i] = float32((int(i)%9)-4) / 8
				}
				for i := range kf {
					kf[i] = float32((i%13)-6) / 4
					vf[i] = float32((i%7)-3) / 2
					k16[i] = accel.ToFloat16(kf[i])
					v16[i] = accel.ToFloat16(vf[i])
				}

				wide := make([]float32, c.qHeads*c.headDim)
				narrow := make([]float32, c.qHeads*c.headDim)
				groups := accel.ID3{X: c.qHeads}
				if err := kernel.DispatchCooperative(&testkernels.AttentionDecodeKernel, groups,
					kernelabi.Args{
						Slices: []any{q, kf, vf, lengths, wide}, Uniforms: []any{d},
					}); err != nil {
					t.Fatalf("f32 dispatch: %v", err)
				}
				if err := kernel.DispatchCooperative(&testkernels.AttentionDecodeF16Kernel, groups,
					kernelabi.Args{
						Slices: []any{q, k16, v16, lengths, narrow}, Uniforms: []any{d},
					}); err != nil {
					t.Fatalf("f16 dispatch: %v", err)
				}
				for i := range wide {
					if wide[i] != narrow[i] {
						t.Fatalf("element %d is %v with an f32 cache and %v with an f16 "+
							"one holding the same values; the two must agree bit for bit",
							i, wide[i], narrow[i])
					}
				}
			})
	}
}
