// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/numeq"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
)

// reference computes a·b at f64 from the values the f16 bits represent.
//
// Written independently of the kernel — a straight triple loop, no tiling, no
// shared memory — because a reference sharing the kernel's structure would
// share its bugs. The widening is exact, so specs/008-numerics.md section 7 adds
// no input-conversion error to the budget.
func reference(m, n, k int, a, b []accel.Float16) []float64 {
	out := make([]float64, m*n)
	for i := range m {
		for j := range n {
			var acc float64
			for kk := range k {
				acc += float64(a[i*k+kk].F32()) * float64(b[kk*n+j].F32())
			}
			out[i*n+j] = acc
		}
	}
	return out
}

// The tiled GEMM matches an independently written higher-precision reference
// under spec 008's per-output dot-product budget, at dimensions that are not
// multiples of any tile dimension.
//
// The dimensions are the criterion. A GEMM at 16x8x16 exactly exercises none of
// the guarded tails, and every one of those tails is a place an off-by-one
// produces a plausible matrix.
func TestTiledGEMMMatchesItsReference(t *testing.T) {
	cases := []struct{ m, n, k int }{
		{8, 16, 16},  // exactly one tile, the easy case
		{1, 1, 1},    // every dimension a partial tile
		{3, 5, 7},    // all three tails, none aligned
		{8, 16, 33},  // a K tail
		{9, 16, 16},  // an M tail
		{8, 17, 16},  // an N tail
		{17, 19, 23}, // all three, larger than one tile each
		{40, 40, 40},
	}
	for _, c := range cases {
		t.Run(fmt.Sprintf("%dx%dx%d", c.m, c.n, c.k), func(t *testing.T) {
			a := make([]accel.Float16, c.m*c.k)
			b := make([]accel.Float16, c.k*c.n)
			for i := range a {
				a[i] = accel.ToFloat16(float32(math.Sin(float64(i))) * 2)
			}
			for i := range b {
				b[i] = accel.ToFloat16(float32(math.Cos(float64(i))) * 3)
			}
			out := make([]float32, c.m*c.n)

			dims := testkernels.GEMMDims{M: uint32(c.m), N: uint32(c.n), K: uint32(c.k)}
			groups := accel.ID3{
				X: uint32((c.n + testkernels.TileN - 1) / testkernels.TileN),
				Y: uint32((c.m + testkernels.TileM - 1) / testkernels.TileM),
				Z: 1,
			}
			err := kernel.DispatchCooperative(&testkernels.MatMulTiledKernel, groups,
				accel.KernelArgs{
					Slices:   []any{a, b, out},
					Uniforms: []any{dims},
				})
			if err != nil {
				t.Fatalf("dispatch: %v", err)
			}

			want := reference(c.m, c.n, c.k, a, b)
			for i := range out {
				// The per-output budget: this element's own K terms and its own
				// sum of magnitudes, which is what spec 008 section 7 requires
				// rather than one budget for the whole matrix.
				row, col := i/c.n, i%c.n
				terms := make([]float32, c.k)
				for kk := range c.k {
					terms[kk] = a[row*c.k+kk].F32() * b[kk*c.n+col].F32()
				}
				r := numeq.Sum(out[i], terms, c.k-1)
				if !r.OK() {
					t.Fatalf("element (%d,%d) of %dx%dx%d: %v (reference %v)",
						row, col, c.m, c.n, c.k, r, want[i])
				}
			}
		})
	}
}

// Each of the GEMM's two barriers is load-bearing, and each fails through the
// diagnostic that names its own hazard.
//
// Spec 009's M5 criterion asks for exactly this, and the *different* failures
// are the content: a test showing only that both are needed would not show that
// the diagnostics distinguish what went wrong. The first barrier orders the
// tile loads against the reads after them, so removing it means reading a slot
// nothing wrote — definition tracking. The second orders this round's reads
// against the next round's writes, so removing it means two invocations
// touching one slot with nothing between them — conflicting access.
//
// The kernels below are hand-written rather than generated, because the corpus
// carries the correct kernel and a deliberately broken one has no business in a
// generated file. What they share with it is the shape the diagnostics see: a
// tile written by every invocation and read across invocations.
func TestBothGEMMBarriersAreLoadBearing(t *testing.T) {
	const lanes = 16

	// tile[lane] is written by lane and read by its neighbour, which is the
	// cross-invocation dependency a tile load creates.
	build := func(firstBarrier, secondBarrier bool) *accel.Kernel {
		return &accel.Kernel{
			Name: "Tile", WorkgroupSize: accel.ID3{X: lanes, Y: 1, Z: 1},
			Generator: accel.KernelABIVersion, Suspensions: 4,
			SharedSizes: []int{lanes},
			Bindings: []accel.KernelBinding{
				{Name: "out", DType: accel.KernelF32, Access: accel.KernelWrite},
			},
			NewShared: func() []any {
				var sh [lanes]float32
				accel.KernelPoison(sh[:])
				return []any{&sh}
			},
			Cooperative: func(th accel.Thread, a accel.KernelArgs, f *accel.KernelFrame) bool {
				sh := kernel.SharedSlice[[lanes]float32](a, 0)
				out := a.Slices[0].([]float32)
				lane := th.LocalID().X
				next := (lane + 1) % lanes

				switch f.Pass {
				case 0: // round one: write my slot
					f.Shared.Write(0, int(lane))
					sh[lane] = float32(lane)
					f.Pass = 1
					if firstBarrier {
						f.Barrier = accel.KernelBarrierID{Index: 0, Pos: "tile.go:1:1"}
						return true
					}
					fallthrough
				case 1: // read my neighbour's
					out[lane] = sh[f.Shared.ReadAt(0, int(next))]
					f.Pass = 2
					if secondBarrier {
						f.Barrier = accel.KernelBarrierID{Index: 1, Pos: "tile.go:2:2"}
						return true
					}
					fallthrough
				case 2: // round two: overwrite my slot
					f.Shared.Write(0, int(lane))
					sh[lane] = float32(lane) * 2
					f.Pass = 3
					f.Barrier = accel.KernelBarrierID{Index: 2, Pos: "tile.go:3:3"}
					return true
				default:
					out[lane] += sh[f.Shared.ReadAt(0, int(next))]
					return false
				}
			},
		}
	}

	run := func(k *accel.Kernel) error {
		return kernel.DispatchCooperative(k, accel.ID3{X: 1},
			accel.KernelArgs{Slices: []any{make([]float32, lanes)}})
	}

	if err := run(build(true, true)); err != nil {
		t.Fatalf("with both barriers this is a correct kernel: %v", err)
	}

	t.Run("without the first", func(t *testing.T) {
		err := run(build(false, true))
		if err == nil {
			t.Fatal("reading a tile slot nothing wrote should be reported")
		}
		if !strings.Contains(err.Error(), "undefined shared memory") {
			t.Errorf("the first barrier's absence should surface as an undefined read, "+
				"since it is what orders the writes against the reads: got %v", err)
		}
	})

	t.Run("without the second", func(t *testing.T) {
		err := run(build(true, false))
		if err == nil {
			t.Fatal("overwriting a tile slot another invocation is reading should be reported")
		}
		if !strings.Contains(err.Error(), "conflicting access") {
			t.Errorf("the second barrier's absence should surface as a conflicting access, "+
				"since it is what orders the reads against the next round's writes: got %v", err)
		}
	})
}

// The authored GEMM, which is spec 004's fifth testing level: the generated
// lowering is what runs, so nothing else calls this function, and a kernel
// nobody executes means whatever the IR made of it.
//
// The invocations rendezvous for real, one goroutine each behind a cyclic
// barrier. An earlier version ran every invocation through the whole function
// twice per K step, which is exact only while the loop runs once -- so it needed
// K bounded under a tile and could not exercise a multi-step K at all. See
// [kernel.RunAuthored].
func TestAuthoredTiledGEMM(t *testing.T) {
	const m, n, k = 9, 18, 40 // tails on all three axes, and three K steps

	a := make([]accel.Float16, m*k)
	b := make([]accel.Float16, k*n)
	for i := range a {
		a[i] = accel.ToFloat16(float32(i%5) - 2)
	}
	for i := range b {
		b[i] = accel.ToFloat16(float32(i%3) - 1)
	}

	dims := testkernels.GEMMDims{M: m, N: n, K: k}
	size := kernel.ID3{X: testkernels.TileN, Y: testkernels.TileM, Z: 1}
	groups := kernel.ID3{
		X: (n + testkernels.TileN - 1) / testkernels.TileN,
		Y: (m + testkernels.TileM - 1) / testkernels.TileM,
		Z: 1,
	}

	authored := make([]float32, m*n)
	for gy := range groups.Y {
		for gx := range groups.X {
			var tileA [128]accel.Float16
			var tileB [256]accel.Float16
			kernel.RunAuthored(size, kernel.ID3{X: gx, Y: gy}, groups, 128,
				func(th kernel.Thread) {
					testkernels.MatMulTiled(th, dims, a, b, authored, &tileA, &tileB)
				})
		}
	}

	generated := make([]float32, m*n)
	err := kernel.DispatchCooperative(&testkernels.MatMulTiledKernel,
		accel.ID3{X: groups.X, Y: groups.Y, Z: 1},
		accel.KernelArgs{Slices: []any{a, b, generated}, Uniforms: []any{dims}})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	want := reference(m, n, k, a, b)
	for i := range generated {
		if math.Abs(float64(generated[i])-want[i]) > 1e-3 {
			t.Fatalf("the generated lowering's element %d is %v, want about %v",
				i, generated[i], want[i])
		}
		if math.Abs(float64(authored[i])-want[i]) > 1e-3 {
			t.Fatalf("the authored kernel's element %d is %v, want about %v: the two "+
				"halves of spec 004's fifth level must agree", i, authored[i], want[i])
		}
	}
}
