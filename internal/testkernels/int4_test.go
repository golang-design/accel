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
		kernel.RunAuthored(kernel.ID3{X: 128, Y: 1, Z: 1}, kernel.ID3{X: g},
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
