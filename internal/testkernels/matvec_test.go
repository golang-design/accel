// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"fmt"
	"golang.design/x/accel/kernelabi"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/numeq"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
)

// MatVec agrees with the tiled GEMM at M=1, which is the property that makes it
// a *selection* rather than a different operator.
//
// Spec 010 makes it a distinct semantic ID chosen exactly when M=1, and a
// selection is only sound if the two produce the same answer — otherwise the
// model's output would depend on which kernel a dispatcher happened to pick,
// which is the class of bug a deterministic selection rule exists to prevent.
func TestMatVecAgreesWithTheTiledGEMM(t *testing.T) {
	for _, c := range []struct{ n, k int }{
		{1, 1}, {16, 16}, {17, 23}, {64, 128}, {19, 300},
	} {
		t.Run(fmt.Sprintf("N=%d,K=%d", c.n, c.k), func(t *testing.T) {
			a := make([]accel.Float16, c.k)
			b := make([]accel.Float16, c.k*c.n)
			for i := range a {
				a[i] = accel.ToFloat16(float32(math.Sin(float64(i))) * 2)
			}
			for i := range b {
				b[i] = accel.ToFloat16(float32(math.Cos(float64(i))) * 3)
			}
			dims := testkernels.GEMMDims{M: 1, N: uint32(c.n), K: uint32(c.k)}

			viaMatVec := make([]float32, c.n)
			err := kernel.DispatchCooperative(&testkernels.MatVecKernel,
				accel.ID3{X: uint32(c.n)},
				kernelabi.Args{Slices: []any{a, b, viaMatVec}, Uniforms: []any{dims}})
			if err != nil {
				t.Fatalf("matvec: %v", err)
			}

			viaGEMM := make([]float32, c.n)
			err = kernel.DispatchCooperative(&testkernels.MatMulTiledKernel,
				accel.ID3{
					X: uint32((c.n + testkernels.TileN - 1) / testkernels.TileN),
					Y: 1, Z: 1,
				},
				kernelabi.Args{Slices: []any{a, b, viaGEMM}, Uniforms: []any{dims}})
			if err != nil {
				t.Fatalf("gemm: %v", err)
			}

			// Not bit-for-bit: the two sum in different orders, and f32 addition
			// is not associative. Spec 008 section 7 is what says how far apart
			// they may be, and the budget is per output element from its own K
			// and its own sum of magnitudes.
			for i := range viaMatVec {
				terms := make([]float32, c.k)
				for kk := range c.k {
					terms[kk] = a[kk].F32() * b[kk*c.n+i].F32()
				}
				// The looser of the two depths bounds the difference: matvec is
				// a tree and the GEMM folds K in tiles.
				r := numeq.Sum(viaMatVec[i], terms, c.k-1)
				g := numeq.Sum(viaGEMM[i], terms, c.k-1)
				if !r.OK() {
					t.Fatalf("matvec element %d: %v", i, r)
				}
				if !g.OK() {
					t.Fatalf("gemm element %d: %v", i, g)
				}
				if diff := math.Abs(float64(viaMatVec[i] - viaGEMM[i])); diff > r.Budget+g.Budget {
					t.Fatalf("element %d: matvec %v, gemm %v, differing by %g beyond the "+
						"combined budget %g -- a selection between them would make the "+
						"model's output depend on which was picked",
						i, viaMatVec[i], viaGEMM[i], diff, r.Budget+g.Budget)
				}
			}
		})
	}
}

// LinearTiled is the GEMM plus a bias, fused into the store.
//
// Asserted against the GEMM rather than against a reference, because the claim
// spec 010 makes is that it *shares the tile body*: a difference beyond the bias
// would mean the two had drifted apart, which is what a distinct stable ID
// sharing an implementation is meant to prevent.
func TestLinearIsTheGEMMPlusItsBias(t *testing.T) {
	for _, c := range []struct{ m, n, k int }{
		{1, 16, 16}, {9, 18, 20}, {17, 19, 23},
	} {
		t.Run(fmt.Sprintf("%dx%dx%d", c.m, c.n, c.k), func(t *testing.T) {
			a := make([]accel.Float16, c.m*c.k)
			b := make([]accel.Float16, c.k*c.n)
			bias := make([]float32, c.n)
			for i := range a {
				a[i] = accel.ToFloat16(float32(i%7) - 3)
			}
			for i := range b {
				b[i] = accel.ToFloat16(float32(i%5) - 2)
			}
			for i := range bias {
				bias[i] = float32(i%3) - 1
			}
			dims := testkernels.GEMMDims{M: uint32(c.m), N: uint32(c.n), K: uint32(c.k)}
			groups := accel.ID3{
				X: uint32((c.n + testkernels.TileN - 1) / testkernels.TileN),
				Y: uint32((c.m + testkernels.TileM - 1) / testkernels.TileM),
				Z: 1,
			}

			plain := make([]float32, c.m*c.n)
			if err := kernel.DispatchCooperative(&testkernels.MatMulTiledKernel, groups,
				kernelabi.Args{Slices: []any{a, b, plain}, Uniforms: []any{dims}}); err != nil {
				t.Fatalf("gemm: %v", err)
			}
			withBias := make([]float32, c.m*c.n)
			if err := kernel.DispatchCooperative(&testkernels.LinearTiledKernel, groups,
				kernelabi.Args{Slices: []any{a, b, bias, withBias},
					Uniforms: []any{dims}}); err != nil {
				t.Fatalf("linear: %v", err)
			}

			for i := range plain {
				// Bit for bit: the same tile body over the same inputs, and one
				// addition. Anything else means the two implementations have
				// drifted, which is what sharing the body is supposed to stop.
				want := plain[i] + bias[i%c.n]
				if math.Float32bits(withBias[i]) != math.Float32bits(want) {
					t.Fatalf("element %d is %v, want the GEMM's %v plus the bias %v",
						i, withBias[i], plain[i], bias[i%c.n])
				}
			}
		})
	}
}

// The bias is broadcast along M: one value per output column, added to every
// row. A bias indexed by the flat output position would be right for a single
// row and wrong for every model with more than one.
func TestTheBiasIsBroadcastAlongRows(t *testing.T) {
	const m, n, k = 4, 5, 3
	a := make([]accel.Float16, m*k) // all zero, so the output is the bias
	b := make([]accel.Float16, k*n)
	bias := make([]float32, n)
	for i := range bias {
		bias[i] = float32(i) + 1
	}
	out := make([]float32, m*n)

	dims := testkernels.GEMMDims{M: m, N: n, K: k}
	err := kernel.DispatchCooperative(&testkernels.LinearTiledKernel,
		accel.ID3{X: 1, Y: 1, Z: 1},
		kernelabi.Args{Slices: []any{a, b, bias, out}, Uniforms: []any{dims}})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	for row := range m {
		for col := range n {
			if got, want := out[row*n+col], bias[col]; got != want {
				t.Fatalf("row %d column %d is %v, want the column's bias %v",
					row, col, got, want)
			}
		}
	}
}

// The authored halves, spec 004's fifth testing level.
//
// The invocations rendezvous for real, one goroutine each behind a cyclic
// barrier, so neither kernel's shape has to be bent to suit the test. See
// [kernel.RunAuthored] for why the obvious alternative is unsound.
func TestAuthoredMatVecAndLinear(t *testing.T) {
	t.Run("MatVec", func(t *testing.T) {
		const n, k = 3, 300 // K past the workgroup, so the strided fold runs several times
		a := make([]accel.Float16, k)
		b := make([]accel.Float16, k*n)
		for i := range a {
			a[i] = accel.ToFloat16(float32(i%5) - 2)
		}
		for i := range b {
			b[i] = accel.ToFloat16(float32(i%3) - 1)
		}
		dims := testkernels.GEMMDims{M: 1, N: n, K: k}

		authored := make([]float32, n)
		for col := range uint32(n) {
			var sh [128]float32
			kernelabi.Poison(sh[:])
			kernel.RunAuthored(&testkernels.MatVecKernel, kernel.ID3{X: col}, kernel.ID3{X: n},
				testkernels.RowWidth, func(th kernel.Thread) {
					testkernels.MatVec(th, dims, a, b, authored, &sh)
				})
		}

		generated := make([]float32, n)
		if err := kernel.DispatchCooperative(&testkernels.MatVecKernel, accel.ID3{X: n},
			kernelabi.Args{Slices: []any{a, b, generated}, Uniforms: []any{dims}}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range authored {
			if math.Abs(float64(authored[i]-generated[i])) > 1e-4 {
				t.Fatalf("column %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
	})

	t.Run("LinearTiled", func(t *testing.T) {
		const m, n, k = 5, 7, 40 // three K steps, which a whole-function emulation could not do
		a := make([]accel.Float16, m*k)
		b := make([]accel.Float16, k*n)
		bias := make([]float32, n)
		for i := range a {
			a[i] = accel.ToFloat16(float32(i%5) - 2)
		}
		for i := range b {
			b[i] = accel.ToFloat16(float32(i%3) - 1)
		}
		for i := range bias {
			bias[i] = float32(i) * 0.5
		}
		dims := testkernels.GEMMDims{M: m, N: n, K: k}
		groups := kernel.ID3{X: 1, Y: 1, Z: 1}

		authored := make([]float32, m*n)
		var tileA [128]accel.Float16
		var tileB [256]accel.Float16
		kernel.RunAuthored(&testkernels.LinearTiledKernel, kernel.ID3{}, groups, 128, func(th kernel.Thread) {
			testkernels.LinearTiled(th, dims, a, b, bias, authored, &tileA, &tileB)
		})

		generated := make([]float32, m*n)
		if err := kernel.DispatchCooperative(&testkernels.LinearTiledKernel,
			accel.ID3{X: 1, Y: 1, Z: 1},
			kernelabi.Args{Slices: []any{a, b, bias, generated},
				Uniforms: []any{dims}}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range authored {
			if math.Abs(float64(authored[i]-generated[i])) > 1e-4 {
				t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
	})
}
