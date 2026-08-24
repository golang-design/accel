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

// The mixed GEMM matches the f16 one on activations f16 holds exactly.
//
// # Why compare the two rather than compute a reference twice
//
// [testkernels.MatMulTiledF32F16] is [testkernels.MatMulTiled] with the
// activation tile widened, so what is worth checking is that widening changed
// nothing else -- the guarded tails, the tile indexing, the accumulation order.
// The f16 kernel already widens both operands before multiplying, so on
// activations f16 holds exactly the two kernels evaluate the same products in
// the same order and must agree *bit for bit*. A difference is a transcription
// error in the widening rather than a numeric one.
//
// The self-comparison in TestAuthoredFormsAgreeWithTheirLowerings cannot catch
// that: an authored form and its lowering agree perfectly on a kernel that
// reads the wrong element of the tile. This is what a wrong element fails.
//
// The dimensions are the criterion: a GEMM at 16x8x16 exercises none of the
// guarded tails, and every tail is a place an off-by-one produces a plausible
// matrix.
func TestTheMixedGEMMMatchesTheF16OneExactly(t *testing.T) {
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
			// the two kernels read the same activations and any difference is
			// the widening rather than the storage.
			af := make([]float32, c.m*c.k)
			a16 := make([]accel.Float16, len(af))
			b16 := make([]accel.Float16, c.k*c.n)
			for i := range af {
				af[i] = float32((i%13)-6) / 4
				a16[i] = accel.ToFloat16(af[i])
			}
			for i := range b16 {
				b16[i] = accel.ToFloat16(float32((i%11)-5) / 2)
			}

			dims := testkernels.GEMMDims{M: uint32(c.m), N: uint32(c.n), K: uint32(c.k)}
			groups := accel.ID3{
				X: uint32((c.n + testkernels.TileN - 1) / testkernels.TileN),
				Y: uint32((c.m + testkernels.TileM - 1) / testkernels.TileM),
				Z: 1,
			}
			narrow := make([]float32, c.m*c.n)
			mixed := make([]float32, c.m*c.n)
			if err := kernel.DispatchCooperative(&testkernels.MatMulTiledKernel, groups,
				kernelabi.Args{Slices: []any{a16, b16, narrow}, Uniforms: []any{dims}}); err != nil {
				t.Fatalf("f16 dispatch: %v", err)
			}
			if err := kernel.DispatchCooperative(&testkernels.MatMulTiledF32F16Kernel, groups,
				kernelabi.Args{Slices: []any{af, b16, mixed}, Uniforms: []any{dims}}); err != nil {
				t.Fatalf("mixed dispatch: %v", err)
			}
			for i := range narrow {
				if narrow[i] != mixed[i] {
					t.Fatalf("element %d is %v narrow and %v mixed; on activations exact "+
						"in f16 the two kernels must agree bit for bit, so this is a "+
						"transcription error in the widening", i, narrow[i], mixed[i])
				}
			}
		})
	}
}

// The f32-activation quantized kernels match their f16 forms exactly.
//
// Same argument as the mixed GEMM's, and the same thing at stake: each is its
// f16 form with the activation load already wide, so on activations f16 holds
// exactly the two evaluate the same products in the same order. Both shapes are
// covered because both are selected -- M=1 is every decode step and larger M is
// every prefill.
func TestTheF32ActivationQuantizedKernelsMatchTheirF16Forms(t *testing.T) {
	// K is a multiple of QuantBlock so the scale plane divides evenly, and N
	// varies so the flattened block boundary falls at a different place in each
	// case: the scale index is computed from k*N+n, so an N that always aligned
	// would leave the interesting indexing untested.
	for _, c := range []struct{ m, n, k int }{
		{1, 8, 32},   // the decode shape
		{1, 5, 64},   // decode, with N off the block
		{4, 8, 32},   // prefill
		{3, 7, 96},   // prefill, all three off the tile
		{1, 16, 256}, // decode with K past one lane's fold, so a lane folds twice
	} {
		t.Run(fmt.Sprintf("%dx%dx%d", c.m, c.n, c.k), func(t *testing.T) {
			af := make([]float32, c.m*c.k)
			a16 := make([]accel.Float16, len(af))
			for i := range af {
				af[i] = float32((i%13)-6) / 4
				a16[i] = accel.ToFloat16(af[i])
			}
			bq := make([]int8, c.k*c.n)
			for i := range bq {
				bq[i] = int8(i%201 - 100)
			}
			bs := make([]accel.Float16, (len(bq)+testkernels.QuantBlock-1)/testkernels.QuantBlock)
			for i := range bs {
				bs[i] = accel.ToFloat16(0.25 + float32(i%3)/8)
			}
			dims := testkernels.GEMMDims{M: uint32(c.m), N: uint32(c.n), K: uint32(c.k)}

			narrow := make([]float32, c.m*c.n)
			wide := make([]float32, c.m*c.n)
			wg := int(testkernels.QuantMatMulKernel.WorkgroupSize.X)
			flat := accel.ID3{X: uint32((c.m*c.n + wg - 1) / wg), Y: 1, Z: 1}
			if err := kernel.Dispatch(&testkernels.QuantMatMulKernel, flat,
				kernelabi.Args{
					Slices: []any{a16, bq, bs, narrow}, Uniforms: []any{dims},
				}); err != nil {
				t.Fatalf("f16 GEMM dispatch: %v", err)
			}
			if err := kernel.Dispatch(&testkernels.QuantMatMulF32Kernel, flat,
				kernelabi.Args{
					Slices: []any{af, bq, bs, wide}, Uniforms: []any{dims},
				}); err != nil {
				t.Fatalf("f32 GEMM dispatch: %v", err)
			}
			for i := range narrow {
				if narrow[i] != wide[i] {
					t.Fatalf("QuantMatMul element %d is %v narrow and %v wide", i,
						narrow[i], wide[i])
				}
			}

			if c.m != 1 {
				return
			}
			// The M=1 pair, whose reduction is a tree rather than a fold. Its
			// numbers differ from the GEMM's above and must not differ from its
			// own f16 form's.
			narrowVec := make([]float32, c.n)
			wideVec := make([]float32, c.n)
			cols := accel.ID3{X: uint32(c.n), Y: 1, Z: 1}
			if err := kernel.DispatchCooperative(&testkernels.QuantMatVecKernel, cols,
				kernelabi.Args{
					Slices: []any{a16, bq, bs, narrowVec}, Uniforms: []any{dims},
				}); err != nil {
				t.Fatalf("f16 matvec dispatch: %v", err)
			}
			if err := kernel.DispatchCooperative(&testkernels.QuantMatVecF32Kernel, cols,
				kernelabi.Args{
					Slices: []any{af, bq, bs, wideVec}, Uniforms: []any{dims},
				}); err != nil {
				t.Fatalf("f32 matvec dispatch: %v", err)
			}
			for i := range narrowVec {
				if narrowVec[i] != wideVec[i] {
					t.Fatalf("QuantMatVec column %d is %v narrow and %v wide", i,
						narrowVec[i], wideVec[i])
				}
			}
		})
	}
}

// The bf16 widening is the f32 value it names, exactly.
//
// # Why this is not the same test as the f16 widening's
//
// bf16 keeps f32's eight-bit exponent, so its range is f32's and the values
// worth checking include ones f16 cannot hold at all -- which is the reason
// this widening is registered and bf16 to f16 is not. And the conversion is a
// shift rather than a rounding, so the reference is not "close to": every input
// has an exact f32 image, including the infinities and NaNs, and the kernel
// must produce it.
func TestTheBF16WideningIsExact(t *testing.T) {
	// A spread of magnitudes, each exactly representable in bf16 because it is
	// built from seven mantissa bits, plus the boundary values a shift has to
	// carry unchanged.
	var in []accel.BFloat16
	var want []float32
	for _, e := range []int{-30, -14, -1, 0, 1, 8, 20, 40} {
		for m := 0; m < 8; m++ {
			v := float32(math.Ldexp(1+float64(m)/8, e))
			in = append(in, accel.ToBFloat16(v), accel.ToBFloat16(-v))
			want = append(want, v, -v)
		}
	}
	// Zero, and a magnitude f16 overflows to infinity on: 65504 is f16's
	// largest finite value, so the last case is one only bf16's exponent can
	// carry. Its expectation is the bf16 value's own f32 image rather than 1e30,
	// because 1e30 has more mantissa than bf16 holds.
	big := accel.ToBFloat16(1e30)
	in = append(in, accel.ToBFloat16(0), big)
	want = append(want, 0, big.F32())

	out := make([]float32, len(in))
	wg := int(testkernels.CastBF16ToF32Kernel.WorkgroupSize.X)
	groups := accel.ID3{X: uint32((len(in) + wg - 1) / wg), Y: 1, Z: 1}
	if err := kernel.Dispatch(&testkernels.CastBF16ToF32Kernel, groups,
		kernelabi.Args{Slices: []any{in, out}}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	for i := range out {
		if out[i] != want[i] {
			t.Fatalf("element %d widened to %v, want exactly %v; bf16 is f32's top "+
				"half, so a widening that is not equality is a shift done wrong",
				i, out[i], want[i])
		}
	}
	// And the values past f16's range really are past it, or the case above
	// proves nothing about why this kernel exists.
	if !math.IsInf(float64(accel.ToFloat16(out[len(out)-1]).F32()), 1) {
		t.Fatalf("the large case is %v, which f16 holds; it has to be one f16 "+
			"overflows or it does not test bf16's exponent", out[len(out)-1])
	}
}
