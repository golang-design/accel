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
	"golang.design/x/accel/quant"
)

// The kernel's block size and the quantizer's are the same number.
//
// They cannot import each other: a kernel is compiled from a closed subset with
// no imports beyond accel and kmath, so the constant is declared twice. This is
// the only thing that keeps the two equal, and if they drift the kernel reads
// the wrong scale for every weight past the first block -- producing a matrix
// that is wrong everywhere and plausible nowhere anyone would notice quickly.
func TestTheBlockSizesAgree(t *testing.T) {
	if testkernels.QuantBlock != quant.Int8Block {
		t.Fatalf("the kernel blocks at %d and the quantizer at %d; a kernel cannot import "+
			"the quantizer, so this test is what keeps them equal",
			testkernels.QuantBlock, quant.Int8Block)
	}
}

// The quantized GEMM lands within specs/027-quantization.md's derived budget of
// the exact product over the *original* weights.
//
// Against the originals rather than the dequantized ones, because that is the
// question a caller has: how much did quantizing cost me. Comparing against the
// dequantized weights would only re-check the arithmetic and would pass for a
// representation that threw away everything.
//
// The budget is quantization plus accumulation, added rather than combined in
// quadrature: the two are not independent, and a bound that assumed they were
// would be tighter and wrong.
func TestQuantMatMulMeetsItsBudget(t *testing.T) {
	for _, c := range []struct{ m, n, k int }{
		{1, 8, 64},   // one row: the decode shape
		{4, 16, 128}, // whole blocks everywhere
		{3, 5, 33},   // K past a block boundary by one, N not a divisor
		{2, 7, 96},
	} {
		t.Run(fmt.Sprintf("%dx%dx%d", c.m, c.n, c.k), func(t *testing.T) {
			a := make([]float32, c.m*c.k)
			w := make([]float32, c.k*c.n)
			for i := range a {
				a[i] = float32(math.Sin(float64(i)*0.23)) * 2
			}
			for i := range w {
				w[i] = float32(math.Cos(float64(i)*0.31)) * 1.5
			}

			bq, bs := quant.Int8(w)
			af16 := make([]accel.Float16, len(a))
			for i := range a {
				af16[i] = accel.ToFloat16(a[i])
			}
			out := make([]float32, c.m*c.n)

			dims := testkernels.GEMMDims{M: uint32(c.m), N: uint32(c.n), K: uint32(c.k)}
			wg := int(testkernels.QuantMatMulKernel.WorkgroupSize.X)
			err := kernel.Dispatch(&testkernels.QuantMatMulKernel,
				accel.ID3{X: uint32((c.m*c.n + wg - 1) / wg)},
				accel.KernelArgs{
					Slices: []any{af16, bq, bs, out}, Uniforms: []any{dims},
				})
			if err != nil {
				t.Fatalf("dispatch: %v", err)
			}

			for row := range c.m {
				for col := range c.n {
					// The exact product over the original weights, in f64, and
					// the per-output budget: this element's own activations and
					// its own column's scales.
					var exact, magnitude float64
					x := make([]float32, c.k)
					blockScales := make([]accel.Float16, c.k)
					for k := range c.k {
						av := accel.ToFloat16(a[row*c.k+k]).F32()
						exact += float64(av) * float64(w[k*c.n+col])
						magnitude += math.Abs(float64(av) * float64(w[k*c.n+col]))
						x[k] = av
						blockScales[k] = bs[(k*c.n+col)/quant.Int8Block]
					}
					// quant.Error indexes scales by i/Int8Block, so the
					// per-term scales are laid out to match: one entry per k
					// holding the scale that term's weight actually used.
					var qErr float64
					for k := range c.k {
						qErr += math.Abs(float64(x[k])) * float64(blockScales[k].F32()) / 2
					}
					// The f32 accumulation on top, from 008 section 7: a
					// sequential sum of K terms.
					accErr := gammaBound(c.k-1) * magnitude

					got := float64(out[row*c.n+col])
					if d := math.Abs(got - exact); d > qErr+accErr {
						t.Fatalf("(%d,%d) is %v against an exact %v: off by %v, and the "+
							"budget is %v quantization plus %v accumulation",
							row, col, got, exact, d, qErr, accErr)
					}
				}
			}
		})
	}
}

// gammaBound is specs/008-numerics.md section 7's relative bound for a
// sequential sum of n additions in f32.
func gammaBound(n int) float64 {
	const u = 1.0 / (1 << 24) // f32 unit roundoff
	nu := float64(n) * u
	if nu >= 1 {
		return math.Inf(1)
	}
	return nu / (1 - nu)
}

// The quantized gather returns what the table holds, and answers an
// out-of-range id with zeros.
func TestQuantRowsGathers(t *testing.T) {
	const vocab, width, rows = 8, 64, 3
	table := make([]float32, vocab*width)
	for i := range table {
		table[i] = float32(math.Sin(float64(i)*0.11)) * 4
	}
	tq, ts := quant.Int8(table)
	back := quant.Dequantize(tq, ts)

	ids := []uint32{5, 0, vocab + 3}
	out := make([]float32, rows*width)
	p := testkernels.RowParams{Rows: rows, Width: width, Capacity: vocab}
	wg := int(testkernels.QuantRowsKernel.WorkgroupSize.X)
	err := kernel.Dispatch(&testkernels.QuantRowsKernel,
		accel.ID3{X: uint32((rows*width + wg - 1) / wg)},
		accel.KernelArgs{Slices: []any{tq, ts, ids, out}, Uniforms: []any{p}})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	for r := range rows {
		for c := range width {
			got := out[r*width+c]
			if ids[r] >= vocab {
				if got != 0 {
					t.Fatalf("an out-of-range id produced %v rather than zero", got)
				}
				continue
			}
			// Exactly the dequantized value: a gather converts and copies, so
			// anything but equality is an arithmetic bug rather than a
			// quantization cost.
			if want := back[int(ids[r])*width+c]; got != want {
				t.Fatalf("row %d column %d is %v, want %v", r, c, got, want)
			}
		}
	}
}
