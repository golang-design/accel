// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
	"golang.design/x/accel/kernelabi"
	"golang.design/x/accel/quant"
)

// The kernel reads the weights the quantizer wrote, and within the bound the
// spec derives.
//
// specs/048-int4.md §6. This is the assertion a round-trip through `quant`
// alone cannot make: a packer and an unpacker that agree with each other about
// the wrong nibble order round-trip perfectly, and only a third implementation
// reading real weights notices. The kernel is that third implementation.
//
// Compared against the *unquantized* product rather than against a dequantized
// one, because the question the bound answers is how far a quantized dot
// product lands from the real one -- not whether two spellings of the same
// approximation agree.
func TestTheInt4KernelReadsWhatTheQuantizerWrote(t *testing.T) {
	const K, N = 256, 4
	rng := rand.New(rand.NewPCG(19, 23))

	// Weights clustered away from zero, which is where asymmetry earns its
	// keep and where a symmetric reading would be visibly worse.
	w := make([]float32, K*N)
	for i := range w {
		w[i] = 0.75 + rng.Float32()*0.5
	}
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(math.Sin(float64(i) * 0.37))
	}

	packed, scales, zeros := quant.Int4Quantize(w)
	out := make([]float32, N)
	if err := kernel.DispatchCooperative(&testkernels.QuantMatVecInt4Kernel,
		accel.ID3{X: N},
		kernelabi.Args{
			Uniforms: []any{testkernels.GEMMDims{K: K, N: N}},
			Slices:   []any{a, packed, scales, zeros, out},
		}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	for n := range N {
		// The exact product, and the group range of each term, which is what
		// the bound is stated over.
		var exact float64
		ranges := make([]float32, K)
		for k := range K {
			idx := k*N + n
			exact += float64(a[k]) * float64(w[idx])
			g := idx / quant.Int4Group
			lo := g * quant.Int4Group
			hi := min(lo+quant.Int4Group, len(w))
			lowest, highest := w[lo], w[lo]
			for _, v := range w[lo:hi] {
				lowest, highest = min(lowest, v), max(highest, v)
			}
			ranges[k] = highest - lowest
		}
		bound := quant.Int4ErrorBound(a, ranges)

		// Plus the accumulation term: the products are summed in f32, so 008
		// section 7's reduction bound applies on top of the quantization one.
		// Composed by addition, which is conservative and deliberate.
		var mag float64
		for k := range K {
			mag += math.Abs(float64(a[k]) * float64(w[k*N+n]))
		}
		bound += mag * 1e-6

		if e := math.Abs(float64(out[n]) - exact); e > bound {
			t.Fatalf("column %d: the quantized product is %v and the exact one %v, "+
				"off by %v where the derived budget is %v", n, out[n], exact, e, bound)
		}
	}
}

// Both kernels read a constant group the way the quantizer wrote it.
//
// quant.Int4Quantize stores a group of one repeated value as a zero scale with
// the value in the zero point, and quant.Int4Dequantize -- the host reference
// -- reads a zero scale as exactly that. The kernels computed (q - z) * s,
// which is zero for every weight of such a group, so a pruned or padded matrix
// multiplied differently on the device than on the host and nothing said so.
//
// The constant is nonzero and the activations are all one, so the group's
// contribution to each column is 128 times the constant: an answer of zero is
// the defect and not a rounding difference. Both kernels are checked because
// each decodes the representation in its own place.
func TestTheInt4KernelsReadAConstantGroupAsTheQuantizerWroteIt(t *testing.T) {
	const K, N, M = 2 * quant.Int4Group, 2, 3
	const c = 0.75
	rng := rand.New(rand.NewPCG(53, 59))

	// Group 0 is the constant; group 1 is ordinary, so the test also says the
	// select does not disturb a group with a real scale.
	w := make([]float32, K*N)
	for i := range w {
		if i < quant.Int4Group {
			w[i] = c
		} else {
			w[i] = rng.Float32()*2 - 1
		}
	}
	packed, scales, zeros := quant.Int4Quantize(w)
	if scales[0].F32() != 0 || zeros[0].F32() != c {
		t.Fatalf("the quantizer stored the constant group as s=%v z=%v; this test is "+
			"about the zero-scale encoding and needs it", scales[0].F32(), zeros[0].F32())
	}
	back := quant.Int4Dequantize(packed, scales, zeros, len(w))

	a := make([]float32, M*K)
	for i := range a {
		a[i] = 1
	}
	// The host reference, over the dequantized weights the kernel is meant to
	// read, in f64 so the only slack is the kernel's own f32 accumulation.
	want := make([]float64, M*N)
	mag := make([]float64, M*N)
	for m := range M {
		for n := range N {
			for k := range K {
				want[m*N+n] += float64(a[m*K+k]) * float64(back[k*N+n])
				mag[m*N+n] += math.Abs(float64(a[m*K+k]) * float64(back[k*N+n]))
			}
		}
	}

	row := make([]float32, N)
	if err := kernel.DispatchCooperative(&testkernels.QuantMatVecInt4Kernel,
		accel.ID3{X: N},
		kernelabi.Args{
			Uniforms: []any{testkernels.GEMMDims{K: K, N: N}},
			Slices:   []any{a[:K], packed, scales, zeros, row},
		}); err != nil {
		t.Fatalf("row dispatch: %v", err)
	}
	tiled := make([]float32, M*N)
	if err := kernel.DispatchCooperative(&testkernels.QuantMatMulInt4Kernel,
		accel.ID3{
			X: (N + testkernels.TileN - 1) / testkernels.TileN,
			Y: (M + testkernels.TileM - 1) / testkernels.TileM,
		},
		kernelabi.Args{
			Uniforms: []any{testkernels.GEMMDims{M: M, K: K, N: N}},
			Slices:   []any{a, packed, scales, zeros, tiled},
		}); err != nil {
		t.Fatalf("tiled dispatch: %v", err)
	}

	// specs/008-numerics.md section 7's sequential bound, gamma(K-1) times the
	// sum of the terms' magnitudes, and nothing for the quantization: the
	// reference is over the dequantized weights. The sequential figure rather
	// than the row kernel's tree depth because it covers both kernels, and the
	// constant group makes the running sum large against the terms -- 96 from
	// 128 terms of 0.75 -- so a bound that looked proportionate for a
	// sign-mixed sum is too small here by an order of magnitude.
	nu := float64(K-1) * math.Ldexp(1, -24)
	gamma := nu / (1 - nu)
	for n := range N {
		if e := math.Abs(float64(row[n]) - want[n]); e > gamma*mag[n] {
			t.Fatalf("row kernel column %d is %v, want %v: the constant group of %v "+
				"contributes %v and the kernel read it as (q-z)*s with s = 0",
				n, row[n], want[n], c, c*quant.Int4Group)
		}
	}
	for i := range tiled {
		if e := math.Abs(float64(tiled[i]) - want[i]); e > gamma*mag[i] {
			t.Fatalf("tiled kernel element %d is %v, want %v: the constant group of %v "+
				"was read as (q-z)*s with s = 0", i, tiled[i], want[i], c)
		}
	}
}

// The kernel's nibble order is the quantizer's.
//
// The sharpest form of the test above, and the one that fails loudly rather
// than by a small margin: two weights in one word chosen so that reading them
// the other way round gives a different product by a wide margin.
func TestTheInt4KernelAgreesAboutWhichNibbleIsWhich(t *testing.T) {
	const K, N = 8, 1

	// One group, whose extremes sit at indices 0 and 1 -- adjacent nibbles of
	// the same word. Swapping them swaps the largest weight with the smallest.
	w := make([]float32, K*N)
	for i := range w {
		w[i] = 1
	}
	w[0], w[1] = 0, 4

	// An activation vector that reads only those two, with different weights,
	// so a swap changes the answer rather than permuting equal terms.
	a := make([]float32, K)
	a[0], a[1] = 1, 10

	packed, scales, zeros := quant.Int4Quantize(w)
	out := make([]float32, N)
	if err := kernel.DispatchCooperative(&testkernels.QuantMatVecInt4Kernel,
		accel.ID3{X: N},
		kernelabi.Args{
			Uniforms: []any{testkernels.GEMMDims{K: K, N: N}},
			Slices:   []any{a, packed, scales, zeros, out},
		}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	// a·w with the correct order is 0*1 + 4*10 = 40; with the two nibbles
	// swapped it is 4*1 + 0*10 = 4. The gap is what makes this loud.
	const correct, swapped = 40.0, 4.0
	got := float64(out[0])
	if math.Abs(got-swapped) < math.Abs(got-correct) {
		t.Fatalf("the product is %v, which is nearer the swapped-nibble answer %v "+
			"than the correct %v: the kernel and quant.Int4Quantize disagree about "+
			"which nibble holds which weight", got, swapped, correct)
	}
	if math.Abs(got-correct) > 1 {
		t.Fatalf("the product is %v and should be near %v", got, correct)
	}
}

// The authored int4 kernel and its generated lowering agree.
//
// specs/010-kernel-corpus.md §6's obligation. On Linux there is no Metal
// differential to cover the authored form, which is how this package's coverage
// gate failed once already this milestone.
func TestTheAuthoredInt4KernelMatchesItsLowering(t *testing.T) {
	const K, N = 128, 2
	rng := rand.New(rand.NewPCG(29, 31))
	w := make([]float32, K*N)
	for i := range w {
		w[i] = rng.Float32()*2 - 1
	}
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(math.Cos(float64(i) * 0.19))
	}
	packed, scales, zeros := quant.Int4Quantize(w)
	dims := testkernels.GEMMDims{K: K, N: N}

	authored := make([]float32, N)
	groups := kernel.ID3{X: N, Y: 1, Z: 1}
	for g := range groups.X {
		var sh [128]float32
		kernel.RunAuthored(&testkernels.QuantMatVecInt4Kernel, kernel.ID3{X: g},
			groups, 128, func(th kernel.Thread) {
				testkernels.QuantMatVecInt4(th, dims, a, packed, scales, zeros,
					authored, &sh)
			})
	}

	generated := make([]float32, N)
	if err := kernel.DispatchCooperative(&testkernels.QuantMatVecInt4Kernel,
		accel.ID3{X: N},
		kernelabi.Args{
			Uniforms: []any{dims},
			Slices:   []any{a, packed, scales, zeros, generated},
		}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	// Within a bound for the reason every f32 comparison of the two forms here
	// carries one: the generated lowering rounds each product explicitly and
	// ordinary Go may fuse a multiply and an add on a target with FMA.
	for i := range authored {
		if math.Abs(float64(authored[i]-generated[i])) > 1e-5 {
			t.Fatalf("column %d: authored %v, generated %v", i, authored[i], generated[i])
		}
	}
}

// The tiled 4-bit kernel agrees with the row kernel, row for row.
//
// specs/048-int4.md §5. The two decode the same representation at different
// points -- the row kernel per use in its inner loop, the tiled one once per
// element as it fills the shared tile -- so this is what says the fourth
// spelling of the nibble order agrees with the third.
//
// A differential rather than a bound: both kernels approximate the same exact
// product, and asking whether they agree with each other is a sharper question
// than asking whether each lands inside a budget wide enough for both. The
// budget is checked against the row kernel above.
func TestTheTiledInt4KernelMatchesTheRowKernel(t *testing.T) {
	const K, N, M = 200, 20, 13
	rng := rand.New(rand.NewPCG(41, 43))

	w := make([]float32, K*N)
	for i := range w {
		w[i] = 0.75 + rng.Float32()*0.5
	}
	packed, scales, zeros := quant.Int4Quantize(w)

	// None of M, K or N is a multiple of its tile, so every edge guard runs.
	// That is coverage rather than a bound: the pads in the two shared tiles are
	// mutually redundant -- the A tile is zeroed at the same k the B tile is --
	// so no single wrong pad value changes an answer here. The kernel's comment
	// records that, having been checked by mutation rather than assumed.
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(math.Sin(float64(i) * 0.29))
	}

	tiled := make([]float32, M*N)
	if err := kernel.DispatchCooperative(&testkernels.QuantMatMulInt4Kernel,
		accel.ID3{
			X: (N + testkernels.TileN - 1) / testkernels.TileN,
			Y: (M + testkernels.TileM - 1) / testkernels.TileM,
		},
		kernelabi.Args{
			Uniforms: []any{testkernels.GEMMDims{M: M, K: K, N: N}},
			Slices:   []any{a, packed, scales, zeros, tiled},
		}); err != nil {
		t.Fatalf("tiled dispatch: %v", err)
	}

	for m := range M {
		row := make([]float32, N)
		if err := kernel.DispatchCooperative(&testkernels.QuantMatVecInt4Kernel,
			accel.ID3{X: N},
			kernelabi.Args{
				Uniforms: []any{testkernels.GEMMDims{K: K, N: N}},
				Slices:   []any{a[m*K : (m+1)*K], packed, scales, zeros, row},
			}); err != nil {
			t.Fatalf("row dispatch: %v", err)
		}
		for n := range N {
			// The two sum in different orders -- the row kernel by a tree over
			// 128 lanes, the tiled one sequentially within each K tile -- so
			// specs/008-numerics.md section 7's reduction bound applies rather
			// than equality.
			var mag float64
			for k := range K {
				mag += math.Abs(float64(a[m*K+k]) * float64(w[k*N+n]))
			}
			if e := math.Abs(float64(tiled[m*N+n] - row[n])); e > mag*1e-6 {
				t.Fatalf("row %d column %d: tiled %v, row kernel %v, off by %v "+
					"where the reduction budget is %v",
					m, n, tiled[m*N+n], row[n], e, mag*1e-6)
			}
		}
	}
}

// The authored tiled 4-bit kernel and its generated lowering agree.
func TestTheAuthoredTiledInt4KernelMatchesItsLowering(t *testing.T) {
	const K, N, M = 128, 18, 9
	rng := rand.New(rand.NewPCG(7, 11))

	w := make([]float32, K*N)
	for i := range w {
		w[i] = 0.75 + rng.Float32()*0.5
	}
	packed, scales, zeros := quant.Int4Quantize(w)
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(math.Cos(float64(i) * 0.31))
	}
	d := testkernels.GEMMDims{M: M, K: K, N: N}

	groups := kernel.ID3{
		X: (N + testkernels.TileN - 1) / testkernels.TileN,
		Y: (M + testkernels.TileM - 1) / testkernels.TileM,
		Z: 1,
	}
	authored := make([]float32, M*N)
	for gy := range groups.Y {
		for gx := range groups.X {
			var tileA [128]float32
			var tileB [256]float32
			kernel.RunAuthored(&testkernels.QuantMatMulInt4Kernel,
				kernel.ID3{X: gx, Y: gy}, groups, 128, func(th kernel.Thread) {
					testkernels.QuantMatMulInt4(th, d, a, packed, scales, zeros,
						authored, &tileA, &tileB)
				})
		}
	}

	generated := make([]float32, M*N)
	if err := kernel.DispatchCooperative(&testkernels.QuantMatMulInt4Kernel,
		accel.ID3{X: groups.X, Y: groups.Y},
		kernelabi.Args{
			Uniforms: []any{d},
			Slices:   []any{a, packed, scales, zeros, generated},
		}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	for i := range authored {
		if math.Abs(float64(authored[i]-generated[i])) > 1e-5 {
			t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
		}
	}
}
