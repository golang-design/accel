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

// The f16 scatter places the same rows the f32 scatter does, and drops the same
// write.
//
// A scatter performs no arithmetic, so the two widths must agree exactly on
// every element -- the ones written and the ones left alone. The state starts
// at a value no write produces, which is what makes a dropped write visible
// rather than hidden behind a zero, and one id is past the capacity so the
// range check is compared and not merely the addressing.
func TestTheF16ScatterMatchesTheF32OneExactly(t *testing.T) {
	const rows, width, capacity = 3, 4, 8
	p := testkernels.RowParams{Rows: rows, Width: width, Capacity: capacity}

	in16 := make([]accel.Float16, rows*width)
	in32 := make([]float32, len(in16))
	for i := range in16 {
		in16[i] = accel.ToFloat16(float32(i)*0.375 - 5)
		in32[i] = in16[i].F32()
	}
	ids := []uint32{5, 0, capacity + 1} // the last is past the state

	state16 := make([]accel.Float16, capacity*width)
	state32 := make([]float32, len(state16))
	for i := range state16 {
		state16[i] = accel.ToFloat16(-1) // a value no write produces
		state32[i] = state16[i].F32()
	}

	n := rows * width
	groups := accel.ID3{X: uint32((n + 63) / 64)}
	if err := kernel.Dispatch(&testkernels.ScatterRowsKernel, groups,
		kernelabi.Args{Slices: []any{in32, ids, state32}, Uniforms: []any{p}}); err != nil {
		t.Fatalf("f32 dispatch: %v", err)
	}
	if err := kernel.Dispatch(&testkernels.ScatterRowsF16Kernel, groups,
		kernelabi.Args{Slices: []any{in16, ids, state16}, Uniforms: []any{p}}); err != nil {
		t.Fatalf("f16 dispatch: %v", err)
	}

	for i := range state32 {
		if got := state16[i].F32(); got != state32[i] {
			t.Fatalf("element %d is %v in an f32 state and %v in an f16 one holding the "+
				"same values; a scatter does no arithmetic, so the two must agree exactly",
				i, state32[i], got)
		}
	}
	// And the scatter actually moved something, so the comparison above is not
	// two untouched states agreeing.
	moved := 0
	for i := range state16 {
		if state16[i].F32() != -1 {
			moved++
		}
	}
	if moved != 2*width {
		t.Fatalf("%d elements changed; two of the three ids are inside the state, so "+
			"%d should have", moved, 2*width)
	}
}

// The authored form of each new kernel agrees with its generated lowering.
//
// This is specs/012-kernel-pipeline.md's obligation on every kernel, and it is
// what makes the CPU path an oracle rather than a second implementation: the
// authored Go is the reference, the generated Go is what runs, and a
// disagreement is a generator bug rather than a kernel one.
//
// Both were added as variants of kernels that already had this, and a variant
// arriving without it would be the one kernel in the corpus whose lowering
// nothing checks.
func TestTheNewVariantsMatchTheirAuthoredForms(t *testing.T) {
	t.Run("MatMulTiledF32", func(t *testing.T) {
		// Dimensions with all three tails, so the guards are exercised rather
		// than skipped.
		const m, n, k = 9, 19, 23
		a := make([]float32, m*k)
		b := make([]float32, k*n)
		for i := range a {
			a[i] = float32((i%13)-6) / 4
		}
		for i := range b {
			b[i] = float32((i%11)-5) / 2
		}
		d := testkernels.GEMMDims{M: m, N: n, K: k}
		groups := kernel.ID3{
			X: uint32((n + testkernels.TileN - 1) / testkernels.TileN),
			Y: uint32((m + testkernels.TileM - 1) / testkernels.TileM),
			Z: 1,
		}
		size := kernel.ID3{X: testkernels.TileN, Y: testkernels.TileM, Z: 1}

		authored := make([]float32, m*n)
		for gy := range groups.Y {
			for gx := range groups.X {
				var tileA [128]float32
				var tileB [256]float32
				kernel.RunAuthored(size, kernel.ID3{X: gx, Y: gy}, groups,
					size.X*size.Y, func(th kernel.Thread) {
						testkernels.MatMulTiledF32(th, d, a, b, authored, &tileA, &tileB)
					})
			}
		}

		generated := make([]float32, m*n)
		err := kernel.DispatchCooperative(&testkernels.MatMulTiledF32Kernel,
			accel.ID3{X: groups.X, Y: groups.Y, Z: 1},
			kernelabi.Args{Slices: []any{a, b, generated}, Uniforms: []any{d}})
		if err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range authored {
			if authored[i] != generated[i] {
				t.Fatalf("element %d is %v authored and %v generated", i,
					authored[i], generated[i])
			}
		}
	})

	t.Run("AttentionDecodeF16", func(t *testing.T) {
		const qHeads, kvHeads, headDim, kvLen = 4, 2, 8, 5
		d := testkernels.AttnDims{
			QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
			Scale: float32(1 / math.Sqrt(headDim)),
		}
		lengths := []uint32{kvLen}
		q := make([]float32, qHeads*headDim)
		k16 := make([]accel.Float16, kvLen*kvHeads*headDim)
		v16 := make([]accel.Float16, len(k16))
		for i := range q {
			q[i] = float32((i%9)-4) / 8
		}
		for i := range k16 {
			k16[i] = accel.ToFloat16(float32((i%13)-6) / 4)
			v16[i] = accel.ToFloat16(float32((i%7)-3) / 2)
		}

		authored := make([]float32, qHeads*headDim)
		size := kernel.ID3{X: 128, Y: 1, Z: 1}
		groups := kernel.ID3{X: qHeads, Y: 1, Z: 1}
		for g := range groups.X {
			var scores, red [128]float32
			kernel.RunAuthored(size, kernel.ID3{X: g}, groups, 128, func(th kernel.Thread) {
				testkernels.AttentionDecodeF16(th, d, q, k16, v16, lengths, authored,
					&scores, &red)
			})
		}

		generated := make([]float32, qHeads*headDim)
		err := kernel.DispatchCooperative(&testkernels.AttentionDecodeF16Kernel,
			accel.ID3{X: qHeads},
			kernelabi.Args{
				Slices: []any{q, k16, v16, lengths, generated}, Uniforms: []any{d},
			})
		if err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range authored {
			if authored[i] != generated[i] {
				t.Fatalf("element %d is %v authored and %v generated", i,
					authored[i], generated[i])
			}
		}
	})

	t.Run("Pack", func(t *testing.T) {
		src := make([]float32, 24)
		for i := range src {
			src[i] = float32(i) * 1.5
		}
		p := testkernels.PackParams{Rank: 2, Count: 12, Offset: 2}
		p.Extent[0], p.Extent[1] = 4, 3
		p.Stride[0], p.Stride[1] = 1, 6

		authored := make([]float32, 12)
		for i := range uint32(12) {
			testkernels.Pack(kernel.NewThread(
				kernel.ID3{X: i}, kernel.ID3{X: i}, kernel.ID3{},
				kernel.ID3{X: 12}, kernel.ID3{X: 1}), p, src, authored)
		}
		generated := make([]float32, 12)
		err := kernel.Dispatch(&testkernels.PackKernel, accel.ID3{X: 12},
			kernelabi.Args{Slices: []any{src, generated}, Uniforms: []any{p}})
		if err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range authored {
			if authored[i] != generated[i] {
				t.Fatalf("element %d is %v authored and %v generated", i,
					authored[i], generated[i])
			}
		}
	})
}

// The f16 cache past one block. It is the one attention kernel whose element
// width differs from the others, and the loop bound is len(k) divided by the
// head geometry -- so if len() ever reported bytes rather than elements for a
// narrow binding, this kernel alone would walk twice the cache.
//
// Both backends compute that length from the binding's declared dtype: the CPU
// lowering uses Go's len() on a []accel.Float16, and the Metal backend divides
// the view's byte size by elemBytes(F16). This test is what would notice if
// either changed.
func TestAttentionDecodeF16ScoresACacheLongerThanAWorkgroup(t *testing.T) {
	const qHeads, kvHeads, headDim = 4, 2, 32
	const kvLen, capacity = 300, 384

	d := testkernels.AttnDims{
		QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
		Scale: float32(1 / math.Sqrt(headDim)),
	}
	q := make([]float32, qHeads*headDim)
	for i := range q {
		q[i] = float32(math.Sin(float64(i) * 0.31))
	}
	// f32 originals kept beside the narrow ones, so the reference compares
	// against what the cache actually holds rather than against what it was
	// asked to hold. A narrowing that lost a value would otherwise be scored
	// against the unrounded number and read as an error in this kernel.
	k32 := make([]float32, capacity*kvHeads*headDim)
	v32 := make([]float32, len(k32))
	k16 := make([]accel.Float16, len(k32))
	v16 := make([]accel.Float16, len(k32))
	for i := range k32 {
		k16[i] = accel.ToFloat16(float32(math.Cos(float64(i) * 0.021)))
		v16[i] = accel.ToFloat16(float32(math.Sin(float64(i) * 0.017)))
		k32[i] = k16[i].F32()
		v32[i] = v16[i].F32()
	}

	got := make([]float32, qHeads*headDim)
	if err := kernel.DispatchCooperative(&testkernels.AttentionDecodeF16Kernel,
		accel.ID3{X: qHeads},
		kernelabi.Args{
			Slices:   []any{q, k16, v16, []uint32{kvLen}, got},
			Uniforms: []any{d},
		}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	want := composedAttention(d, kvLen, q, k32, v32)

	const u = 1.0 / (1 << 24)
	n := float64(kvLen+headDim+1) + math.Ceil(float64(kvLen)/AttnBlockT)
	gamma := n * u / (1 - n*u)
	maxV := 0.0
	for _, x := range v32 {
		maxV = math.Max(maxV, math.Abs(float64(x)))
	}
	tol := maxV * gamma

	for i := range got {
		if diff := math.Abs(float64(got[i]) - want[i]); diff > tol {
			t.Fatalf("element %d is %v, want %v: off by %g, tolerance %g",
				i, got[i], want[i], diff, tol)
		}
	}
}
