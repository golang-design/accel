// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
)

// The corpus differential's case table, and the gate that says what is missing
// from it.
//
// Split out of differential_darwin_test.go by specs/062-backend-parity.md
// section 6.7, and the split is the point. The gate below compares two lists of
// names and needs no device, so behind //go:build darwin it was answering
// "which kernel has never been compared against Metal" on one of the three Tier
// 1 platforms. A kernel added on Linux without a case was invisible there until
// somebody ran the suite on a Mac.
//
// The table is data and needs no device either, so it comes with the gate. What
// stays behind the build tag is the half that opens a GPU.

// direction specs/009-sequencing.md's risk row permits.
//
// **The table is written out rather than derived.** Each kernel's shapes have
// to satisfy relationships no record carries -- a GEMM's M, N and K against its
// three buffers, a row kernel's width against its workgroup -- and a harness
// that guessed them would test whichever kernels its guess happened to fit.

// diffCase is one kernel's inputs.
type diffCase struct {
	kernel *accel.Kernel

	// counts is the element count of each binding, in binding order.
	counts []int

	// uniforms are the by-value parameters, in signature order.
	uniforms []any

	groups accel.WorkgroupCount

	// ulp is how far the two backends may diverge, in ULP, and zero means they
	// must agree bit for bit.
	//
	// A ceiling rather than a tolerance: the number is derived below from
	// specs/008-numerics.md section 6's normative table for whichever bounded
	// primitive the kernel reaches, and never from what a run produced. A
	// kernel reaching none of them keeps zero, which is what stops a tolerance
	// spreading from the kernels that need one to the kernels that do not.
	ulp uint64

	// abs is an absolute ceiling, used where a ULP count is not meaningful.
	// section 6 bounds sin and cos absolutely for that reason: argument
	// reduction dominates, and a ULP count near a zero crossing says nothing.
	abs float64

	// why records where ulp or abs came from, and is printed on failure so a
	// reader does not have to reconstruct the derivation.
	why string

	// seed fills binding b's element i. Nil means the default: a bounded,
	// sign-varying value that is exact in f16 as well as f32, so an f16 binding
	// carries what was intended rather than what rounding produced.
	seed func(b, i int) float32
}

// defaultSeed is exact in f16: a small integer over a power of two.
//
// Exactness matters because three kernels take f16 inputs. A value that rounded
// on the way in would still compare equal between backends -- both read the same
// bits -- but a failure would then be about the test's inputs rather than about
// the kernels, which is the hardest kind to diagnose.
func defaultSeed(b, i int) float32 {
	return float32((i+b*7)%13-6) / 4
}

// subgroupWidth is the width both backends are run at, set from the device.
//
// A package-level variable because the fallback kernel takes it as a binding
// and the case table is built before the device is open. It is written once,
// by the test, before any case runs.
var subgroupWidth = 32

func diffCases() []diffCase {
	const width = 128 // matches the row kernels' workgroup

	return []diffCase{
		{kernel: &kernels.AddKernel, counts: []int{256, 256, 256}, groups: accel.WorkgroupCount{X: 4}},

		// The dispatch shape, specs/052-dispatch-shape.md. This is the case
		// that matters most for the accessors, because the two backends read
		// them from genuinely different places: the CPU asks the Thread the
		// runtime built, and MSL takes NumGroups from a [[threadgroups_per_grid]]
		// attribute while WorkgroupSize is a literal the generator wrote in
		// from the accel:kernel directive. The grid's three axes differ so a
		// transposed attribute lands on a slot this compares.
		//
		// Exact: these are integers, so §3's last assertion is equality and no
		// ULP budget applies.
		{
			kernel:   &kernels.DispatchShapeKernel,
			counts:   []int{9},
			uniforms: []any{kernels.ShapeDims{Stride: 3}},
			groups:   accel.WorkgroupCount{X: 5, Y: 3, Z: 2},
			seed:     func(b, i int) float32 { return 0 },
			why:      "the dispatch shape is three integers per accessor, so this is exact",
		},
		{
			// The three flat indices over a 4x2x2 workgroup and a 3x2x2
			// grid: 16 invocations per group, 12 groups, three values each.
			// The Metal lowering folds the workgroup extent in as literals
			// and reads the group count from the grid, and this is the case
			// that says the two linearizations agree.
			kernel: &kernels.IndexShapeKernel,
			counts: []int{3 * 16 * 12},
			groups: accel.WorkgroupCount{X: 3, Y: 2, Z: 2},
			seed:   func(b, i int) float32 { return 0 },
			why:    "a flat index is an integer, so this is exact",
		},
		{
			// The workgroup-bounded loop with a barrier in it, which on Metal
			// is a real barrier inside a loop whose bound is a literal, and on
			// the CPU is the resumable state machine.
			kernel: &kernels.ShapeBoundedSumKernel,
			counts: []int{24, 24},
			groups: accel.WorkgroupCount{X: 3},
			ulp:    8,
			why:    "a sum of 8 terms, per section 7's reduction bound",
		},

		// The masked barriers, specs/050-barrier-scopes.md. Both backends must
		// agree on the *result*, which is what says the two schedulers
		// implement one memory model -- the CPU's epochs and Metal's
		// threadgroup_barrier at a narrower mem_flags mask. Exact: these are
		// u32 payloads, and any difference is a divergence.
		//
		// The scopes themselves are asserted on the emitted text, in
		// TestEachBarrierScopeLowersToItsOwnMask, because a workgroup's data
		// fits in one threadgroup and a result cannot tell three scopes apart.
		{
			kernel: &kernels.PublishStorageKernel,
			counts: []int{3, 96},
			groups: accel.WorkgroupCount{X: 3},
			seed:   func(b, i int) float32 { return 0 },
			why:    "u32 payloads through a storage buffer, so this is exact",
		},
		{
			kernel: &kernels.PublishSharedKernel,
			counts: []int{64},
			groups: accel.WorkgroupCount{X: 2},
			seed:   func(b, i int) float32 { return 0 },
			why:    "u32 payloads through a shared array, so this is exact",
		},
		{
			// Subgroup scope. The two backends narrow the rendezvous by
			// different means -- the CPU checks arrival per subgroup, Metal
			// emits simdgroup_barrier -- and both run at the device's real lane
			// count here rather than the emulated sweep, which is the point of
			// comparing them at all.
			kernel: &kernels.SubgroupPublishKernel,
			counts: []int{64},
			groups: accel.WorkgroupCount{X: 1},
			seed:   func(b, i int) float32 { return 0 },
			why:    "each lane copies one shared f32, so this is exact",
		},
		{
			// The integer minima and maxima,
			// specs/059-subgroup-reductions.md. Exact: a minimum selects an
			// input rather than computing one, so the two backends must agree
			// bit for bit and no ULP budget applies.
			//
			// This case is also what verifies §5's unverified claim that MSL's
			// simd_min and simd_max carry integer overloads. They do -- the
			// kernel lowers -- and this is what says the overload picked is the
			// right one rather than a silent conversion through float.
			kernel: &kernels.IntReduceKernel,
			counts: []int{64, 64, 64, 64, 64},
			groups: accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b == 0 {
					// The kernel's own shuffled input, spanning both signs so
					// the signed and unsigned answers differ.
					return float32((i*37)%64 - 21)
				}
				return 0
			},
			why: "a minimum and a maximum select an input, so this is exact",
		},
		{
			// The f32 minimum and maximum, specs/059-subgroup-reductions.md
			// §5, with the input IntReduce cannot carry: a NaN, and one that
			// is never lane 0 of a subgroup on either backend (the CPU
			// scheduler's default width is 4, Metal's is 32; lane 5 of every
			// 64 is lane 1 of a 4-wide subgroup and lane 5 of a 32-wide one).
			// kmath's contract is that a NaN in any lane makes the reduction
			// NaN. simd_min and simd_max are fmin and fmax across lanes and
			// drop it, and the CPU scheduler once kept only a lane-0 NaN, so
			// this is the case both lowerings got wrong.
			kernel: &kernels.FloatReduceKernel,
			counts: []int{64, 64, 64},
			groups: accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b != 0 {
					return 0
				}
				if i%64 == 5 {
					return float32(math.NaN())
				}
				return float32((i*37)%64-21) * 0.5
			},
			why: "a minimum and a maximum select an input, so this is exact; a NaN is compared as a NaN",
		},
		{
			// The bitwise family, specs/059-subgroup-reductions.md §6's second
			// slice. Exact for a sharper reason than the minima: each is
			// associative *and* commutative over its whole domain, so no
			// ordering of the lanes can produce a different answer.
			//
			// The seed reproduces the kernel's own pattern -- a shared mask
			// every lane carries plus one private bit -- because a
			// pseudorandom input makes And zero and Or all-ones over any wide
			// subgroup, which a lowering ignoring its input also produces.
			kernel: &kernels.BitReduceKernel,
			counts: []int{64, 64, 64, 64, 64, 64, 64},
			groups: accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b == 0 {
					return float32(int32(0x00F0F00F) | int32(1)<<(uint(i)%8+20))
				}
				return 0
			},
			why: "and, or and xor are associative and commutative, so this is exact",
		},
		{
			// The products, specs/059-subgroup-reductions.md §6's third slice.
			// The f32 one is the only reduction here that is *not* exact:
			// specs/008-numerics.md §7.1 bounds it relatively, and Metal
			// combines in whatever order the hardware scans in.
			//
			// The seed is near one at every lane, which is §7.1's domain
			// rather than a convenience: a 64-lane product of magnitude-4
			// values reaches 2^128 and overflows f32 while every term is
			// ordinary. The two integer outputs stay exact.
			kernel: &kernels.MulReduceKernel,
			counts: []int{64, 64, 64, 64, 64},
			groups: accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				switch b {
				case 0:
					if i%2 == 0 {
						return 1 + float32(1+i%4)/64
					}
					return 1 - float32(1+i%4)/64
				case 1:
					return float32(3 + 2*(i%5))
				}
				return 0
			},
			// A product of 64 terms each within a few percent of one: §7.1's
			// gamma(63) is about 3.8e-6 relative, and the outputs are near one.
			ulp: 64,
			why: "a product of 64 near-unit terms, per section 7.1's relative bound",
		},
		{
			// Subgroups genuinely out of step: the loop's trip count is
			// SubgroupIndex, so at any moment they are at different barriers.
			// Metal executes simdgroup_barrier a different number of times per
			// simdgroup and the CPU checks arrival per subgroup; agreeing here
			// is what says the two implement one rule rather than two.
			kernel: &kernels.SubgroupStaggerKernel,
			counts: []int{64},
			groups: accel.WorkgroupCount{X: 1},
			seed:   func(b, i int) float32 { return 0 },
			why:    "each lane copies one shared f32, so this is exact",
		},
		{kernel: &kernels.ElemAddKernel, counts: []int{256, 256, 256}, groups: accel.WorkgroupCount{X: 4}},
		{kernel: &kernels.ElemMulKernel, counts: []int{256, 256, 256}, groups: accel.WorkgroupCount{X: 4}},
		{kernel: &kernels.ScaleKernel, counts: []int{256, 256}, groups: accel.WorkgroupCount{X: 4}},
		{
			kernel: &kernels.ElemScaleKernel, counts: []int{256, 256},
			uniforms: []any{kernels.ScaleParams{Factor: 2.5}},
			groups:   accel.WorkgroupCount{X: 4},
		},
		{
			// x*sigmoid(x) reaches exp and a division. Section 6 bounds each at
			// 4 and 2.5 ULP from correctly rounded, and two implementations may
			// sit on opposite sides, so their mutual distance is up to twice
			// each: 8 + 5, rounded up to 16 for the surrounding multiply.
			kernel: &kernels.SiLUKernel, counts: []int{256, 256},
			groups: accel.WorkgroupCount{X: 4},
			ulp:    16, why: "exp (4 ULP) and a division (2.5 ULP), doubled for two implementations",
		},
		{
			kernel: &kernels.SwiGLUKernel, counts: []int{256, 256, 256},
			groups: accel.WorkgroupCount{X: 4},
			ulp:    16, why: "SiLU's exp and division, then a multiply",
		},
		{
			// The saturating conversions over their boundaries, on both
			// backends. specs/051-float-to-int.md §2.1: the two lowerings are
			// written separately and mirror each other by hand, which is an
			// argument rather than a test, and this is the test.
			//
			// No ulp: the outputs are integers and the class is Exact, so any
			// difference at all is a divergence. That is the point of comparing
			// here rather than through the graphics stages, which only ever pass
			// in-range coordinates and would agree under either lowering.
			kernel: &kernels.SaturatingConvertKernel,
			counts: []int{16, 16, 16},
			groups: accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b != 0 {
					return 0 // the two outputs start empty
				}
				// Every case the branches distinguish, and the exact limits
				// rather than values near them: a lowering that reads < where
				// the other reads <= differs only at the limit itself.
				switch i {
				case 0:
					return float32(math.NaN())
				case 1:
					return float32(math.Inf(1))
				case 2:
					return float32(math.Inf(-1))
				case 3:
					return 0
				case 4:
					return 1.9
				case 5:
					return -1.9
				case 6:
					return -2147483648 // -2^31, exactly int32's minimum
				case 7:
					return 2147483648 // +2^31, one past int32's maximum
				case 8:
					return 4294967296 // 2^32, one past uint32's maximum
				case 9:
					return 4294967040 // the largest f32 below 2^32
				case 10:
					return -0.5 // negative but truncating to zero
				case 11:
					return 1e30
				case 12:
					return -1e30
				}
				return float32(i) * 1.5
			},
			why: "integer results of an exact class, so they agree or they do not",
		},
		{kernel: &kernels.SegmentSumKernel, counts: []int{256, 8}, groups: accel.WorkgroupCount{X: 1}},
		// Helpers reached only through helpers, in the MSL as well as the Go:
		// the emitted source has to declare halve and putAt before their
		// callers, which is the order Func.Helpers carries.
		{kernel: &kernels.PairAverageKernel, counts: []int{128, 64}, groups: accel.WorkgroupCount{X: 2}},
		{kernel: &kernels.CountAboveKernel, counts: []int{256, 1}, groups: accel.WorkgroupCount{X: 16}},
		{kernel: &kernels.HistogramKernel, counts: []int{256, 4}, groups: accel.WorkgroupCount{X: 4}},
		{kernel: &kernels.CountWorkgroupsKernel, counts: []int{1}, groups: accel.WorkgroupCount{X: 7}},
		{
			// Ten state slots and ten results, which is what the kernel indexes.
			// Its ninth case compare-exchanges against an expected 1, so the
			// seed puts a 1 there and a 0 in the tenth: one swap must happen and
			// one must not, and a seed that made both cases alike would let a
			// compare-exchange that never swaps pass.
			kernel: &kernels.AtomicOpsKernel, counts: []int{10, 10},
			groups: accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b == 0 && i == 8 {
					return 1
				}
				if b == 0 {
					return float32(i%7) + 2
				}
				return 0
			},
		},
		{kernel: &kernels.ExchangeKernel, counts: []int{64, 64}, groups: accel.WorkgroupCount{X: 1}},
		{kernel: &kernels.ReduceLoopKernel, counts: []int{64, 1}, groups: accel.WorkgroupCount{X: 1}},
		{kernel: &kernels.ReduceUnrolledKernel, counts: []int{64, 1}, groups: accel.WorkgroupCount{X: 1}},
		{kernel: &kernels.ReduceSumKernel, counts: []int{512, 4}, groups: accel.WorkgroupCount{X: 4}},
		{kernel: &kernels.SubgroupReduceKernel, counts: []int{64, 1}, groups: accel.WorkgroupCount{X: 1}},
		{
			kernel: &kernels.SubgroupReduceFallbackKernel, counts: []int{64, 1, 1},
			groups: accel.WorkgroupCount{X: 1},
			// The fallback's third binding is the emulated width, and it must
			// be the width the device actually executes at or the two backends
			// are running different reductions. 32 is the Apple silicon SIMD
			// width, asserted by the probe rather than assumed here.
			seed: func(b, i int) float32 {
				if b == 2 {
					return float32(subgroupWidth)
				}
				return defaultSeed(b, i)
			},
		},
		// The shuffles. Bit for bit: every one of them moves a value between
		// lanes without arithmetic, and the combination each lane does
		// afterwards is a sum of dyadic values that both backends round
		// identically, so anything but equality is a lowering difference.
		{kernel: &kernels.SubgroupShuffleMixKernel, counts: []int{64, 64},
			groups: accel.WorkgroupCount{X: 1}},
		{
			kernel: &kernels.SubgroupShuffleMixFallbackKernel, counts: []int{64, 64, 1},
			groups: accel.WorkgroupCount{X: 1},
			// As above: the fallback's width binding has to be the width this
			// device executes at, or the two are shuffling within different
			// subgroups.
			seed: func(b, i int) float32 {
				if b == 2 {
					return float32(subgroupWidth)
				}
				return defaultSeed(b, i)
			},
		},
		// The scans, compared exactly — and the exactness is a property of the
		// inputs rather than of the two implementations agreeing on an order.
		// The oracle scans in ascending lane order and Metal's
		// simd_prefix_inclusive_sum does not say what order it uses, so a
		// prefix sum that rounded would be free to differ. defaultSeed produces
		// quarters in [-1.5, 1.5], so every prefix over 32 lanes is a multiple
		// of 1/4 below 48 in magnitude: exact in f32 whatever order it is
		// summed in, which makes a difference of one bit a difference in what
		// was summed.
		{kernel: &kernels.SubgroupScanKernel, counts: []int{64, 64, 64},
			groups: accel.WorkgroupCount{X: 1}},
		{
			kernel: &kernels.SubgroupScanFallbackKernel, counts: []int{64, 64, 64, 1},
			groups: accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b == 3 {
					return float32(subgroupWidth)
				}
				return defaultSeed(b, i)
			},
		},
		{
			kernel: &kernels.NormalizeKernel, counts: []int{64, 64, 64},
			groups: accel.WorkgroupCount{X: 2},
		},
		{
			kernel: &kernels.RMSNormKernel, counts: []int{4 * width, width, 4 * width},
			uniforms: []any{kernels.RowDims{Rows: 4, Width: width, Eps: 1e-5}},
			groups:   accel.WorkgroupCount{X: 4},
			ulp:      16, why: "rsqrt (4 ULP), doubled for two implementations, then a multiply",
		},
		{
			kernel: &kernels.SoftmaxKernel, counts: []int{4 * width, 4 * width},
			uniforms: []any{kernels.RowDims{Rows: 4, Width: width}},
			groups:   accel.WorkgroupCount{X: 4},
			ulp:      16, why: "exp (4 ULP) and a division (2.5 ULP), doubled, then the reduction",
		},
		{
			kernel: &kernels.GatherRowsKernel, counts: []int{8 * 16, 4, 4 * 16},
			uniforms: []any{kernels.RowParams{Rows: 4, Width: 16, Capacity: 8}},
			groups:   accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b == 1 {
					return float32(i % 8) // row ids, inside the table
				}
				return defaultSeed(b, i)
			},
		},
		{
			// The same gather over an f16 table. A gather does no arithmetic, so
			// what this compares is the two lowerings of the widening load, and
			// the widening is exact -- hence no ULP budget.
			kernel: &kernels.GatherRowsF16Kernel, counts: []int{8 * 16, 4, 4 * 16},
			uniforms: []any{kernels.RowParams{Rows: 4, Width: 16, Capacity: 8}},
			groups:   accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b == 1 {
					return float32(i % 8) // row ids, inside the table
				}
				return defaultSeed(b, i)
			},
		},
		{
			kernel: &kernels.ScatterRowsKernel, counts: []int{4 * 16, 4, 8 * 16},
			uniforms: []any{kernels.RowParams{Rows: 4, Width: 16, Capacity: 8}},
			groups:   accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b == 1 {
					return float32(i % 8)
				}
				return defaultSeed(b, i)
			},
		},
		{
			// The same scatter into an f16 state from f16 rows. A scatter does
			// no arithmetic and this one does not even convert -- the two
			// lowerings move the same sixteen bits -- so nothing here reaches a
			// bounded primitive and the ceiling stays zero.
			kernel: &kernels.ScatterRowsF16Kernel, counts: []int{4 * 16, 4, 8 * 16},
			uniforms: []any{kernels.RowParams{Rows: 4, Width: 16, Capacity: 8}},
			groups:   accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b == 1 {
					return float32(i % 8)
				}
				return defaultSeed(b, i)
			},
		},
		{
			// Positions first, then the buffer it rotates. Four rows at four
			// different positions, which a shared offset could not express and
			// which is the whole point of specs/043-per-row-values.md.
			kernel: &kernels.RoPEKernel, counts: []int{4, 4 * 16},
			uniforms: []any{kernels.RoPEParams{
				Rows: 4, Width: 16, RotaryDim: 8, Base: 10000,
			}},
			// Four positions, none of them row+k for a shared k, so a kernel
			// that had kept the old arithmetic would disagree with itself.
			seed: func(b, i int) float32 {
				if b == 0 {
					return float32(3 + 5*i)
				}
				return defaultSeed(b, i)
			},
			groups: accel.WorkgroupCount{X: 1},
			// Absolute, not ULP: section 6 bounds sin and cos at 2^-20 absolute
			// because argument reduction dominates and a ULP count near a zero
			// crossing is not meaningful. Two implementations give 2^-19, and
			// the rotation combines two of them over inputs bounded by about
			// two, so 4 * 2^-20.
			abs: 4 * math.Ldexp(1, -20),
			why: "sin and cos, bounded absolutely at 2^-20 by section 6",
		},
		{
			kernel: &kernels.TransformKernel, counts: []int{64, 64},
			uniforms: []any{kernels.Params{
				Scale:  1.5,
				Origin: [3]float32{0.25, -0.5, 2},
				Steps:  64,
				Inverse: [4][4]float32{
					{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16},
				},
			}},
			groups: accel.WorkgroupCount{X: 2},
		},
		{
			// specs/014-kernel-uniforms.md section 4's device check for the
			// members whose two extents differ: the CPU reads the Go struct
			// and Metal reads the std140 bytes the codec wrote against the
			// block the emitter declared, so a codec looping the column count
			// for the rows shows here and nowhere host-side.
			kernel: &kernels.MatrixShapesKernel, counts: []int{24},
			uniforms: []any{kernels.MatrixParams{
				Wide:   [2][4]float32{{1, 2, 3, 4}, {5, 6, 7, 8}},
				Tall:   [4][2]float32{{-1, -2}, {-3, -4}, {-5, -6}, {-7, -8}},
				Column: [6]float32{10, 11, 12, 13, 14, 15},
			}},
			groups: accel.WorkgroupCount{X: 1},
		},
		{
			// Tails on all three axes, so every guarded edge of the tiled GEMM
			// runs. A shape that fitted the tile exactly would exercise none of
			// them, and every one is a place an off-by-one produces a plausible
			// matrix.
			kernel: &kernels.MatMulTiledKernel, counts: []int{9 * 23, 23 * 19, 9 * 19},
			uniforms: []any{kernels.GEMMDims{M: 9, N: 19, K: 23}},
			groups: accel.WorkgroupCount{
				X: (19 + kernels.TileN - 1) / kernels.TileN,
				Y: (9 + kernels.TileM - 1) / kernels.TileM,
			},
		},
		{
			// The same shape with f32 operands. Bit-exact for the same reason
			// the f16 one is: both accumulate f32 in the same order, and the
			// tiles differ only in what they hold.
			kernel:   &kernels.MatMulTiledF32Kernel,
			counts:   []int{9 * 23, 23 * 19, 9 * 19},
			uniforms: []any{kernels.GEMMDims{M: 9, N: 19, K: 23}},
			groups: accel.WorkgroupCount{
				X: (19 + kernels.TileN - 1) / kernels.TileN,
				Y: (9 + kernels.TileM - 1) / kernels.TileM,
			},
		},
		{
			// And the mixed one, f32 activations against f16 weights. Bit-exact
			// again: the f16 form already widens its weight load before
			// multiplying, so this kernel differs only in where the widening of
			// the *activation* happens, which is not arithmetic either backend
			// performs differently.
			kernel:   &kernels.MatMulTiledF32F16Kernel,
			counts:   []int{9 * 23, 23 * 19, 9 * 19},
			uniforms: []any{kernels.GEMMDims{M: 9, N: 19, K: 23}},
			groups: accel.WorkgroupCount{
				X: (19 + kernels.TileN - 1) / kernels.TileN,
				Y: (9 + kernels.TileM - 1) / kernels.TileM,
			},
		},
		{
			kernel: &kernels.LinearTiledKernel, counts: []int{9 * 23, 23 * 19, 19, 9 * 19},
			uniforms: []any{kernels.GEMMDims{M: 9, N: 19, K: 23}},
			groups: accel.WorkgroupCount{
				X: (19 + kernels.TileN - 1) / kernels.TileN,
				Y: (9 + kernels.TileM - 1) / kernels.TileM,
			},
		},
		{
			kernel: &kernels.MatVecKernel, counts: []int{1 * 40, 40 * 12, 1 * 12},
			uniforms: []any{kernels.GEMMDims{M: 1, N: 12, K: 40}},
			groups:   accel.WorkgroupCount{X: 1},
		},
		{
			// The mixed and the f32 matrix-vector kernels, the same shape.
			kernel: &kernels.MatVecF32F16Kernel, counts: []int{1 * 40, 40 * 12, 1 * 12},
			uniforms: []any{kernels.GEMMDims{M: 1, N: 12, K: 40}},
			groups:   accel.WorkgroupCount{X: 1},
		},
		{
			kernel: &kernels.MatVecF32Kernel, counts: []int{1 * 40, 40 * 12, 1 * 12},
			uniforms: []any{kernels.GEMMDims{M: 1, N: 12, K: 40}},
			groups:   accel.WorkgroupCount{X: 1},
		},
		{
			// Top-k over a distribution with a deliberate plateau at the
			// boundary: the two backends must keep the same *set*, and a tie
			// rule that differed would keep different entries while keeping the
			// same count.
			kernel: &kernels.TopKMaskKernel, counts: []int{256, 256},
			uniforms: []any{kernels.TopDims{Vocab: 256, K: 12}},
			groups:   accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b != 0 {
					return 0
				}
				// Sixteen entries share the boundary value, so twelve of them
				// are kept and four are not -- which is only well defined with
				// a tie rule both backends share.
				if i%16 == 3 {
					return 7
				}
				return float32(i%11) - 5
			},
		},
		{
			kernel: &kernels.TopPMaskKernel, counts: []int{256, 256},
			uniforms: []any{kernels.TopDims{Vocab: 256, P: 0.6}},
			groups:   accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b != 0 {
					return 0
				}
				// Non-negative, because a nucleus over signed weights is not a
				// nucleus: the mass has to accumulate monotonically.
				return float32(i%13) + 1
			},
		},
		{
			// The same two masks over *two* rows, because the row offset is
			// the only part of them a single-row case cannot see: at group
			// zero the offset is zero and the batched form is byte for byte
			// the unbatched one. Metal launches the second workgroup for real,
			// so this is where a group id the transform dropped shows up.
			//
			// The rows differ in scale, which is what top-p is sensitive to:
			// its threshold is a fraction of its own row's total, so a row
			// reading another row's sum keeps a plausible mask that is the
			// wrong set.
			kernel: &kernels.TopKMaskKernel, counts: []int{512, 512},
			uniforms: []any{kernels.TopDims{Vocab: 256, K: 12}},
			groups:   accel.WorkgroupCount{X: 2},
			seed: func(b, i int) float32 {
				if b != 0 {
					return 0
				}
				v := float32(i%11) - 5
				if i%16 == 3 {
					v = 7
				}
				if i >= 256 {
					v = 8 * (v + 6) // row 1: larger, and shifted positive
				}
				return v
			},
		},
		{
			kernel: &kernels.TopPMaskKernel, counts: []int{512, 512},
			uniforms: []any{kernels.TopDims{Vocab: 256, P: 0.6}},
			groups:   accel.WorkgroupCount{X: 2},
			seed: func(b, i int) float32 {
				if b != 0 {
					return 0
				}
				v := float32(i%13) + 1
				if i >= 256 {
					v = 16 * v
				}
				return v
			},
		},
		{
			// Argmax over a vocabulary with a deliberate plateau at the top.
			// A tie rule that differed between the backends would move the
			// answer, which is the one thing this kernel must not do -- and a
			// distribution of distinct values would compare equal whatever
			// either backend did with ties.
			kernel: &kernels.SampleArgmaxKernel, counts: []int{512, 1},
			uniforms: []any{kernels.SampleDims{Vocab: 512}},
			groups:   accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b != 0 {
					return 0
				}
				// Three equal maxima, spread so the pairs that meet them form
				// at different depths of the reduction tree.
				if i == 41 || i == 200 || i == 380 {
					return 9
				}
				return float32(i%17) - 8
			},
		},
		{
			kernel: &kernels.SampleCategoricalKernel, counts: []int{256, 1, 1},
			uniforms: []any{kernels.SampleDims{Vocab: 256, Rows: 1}},
			groups:   accel.WorkgroupCount{X: 1},
			// A distribution with equal masses, so the boundary the draw lands
			// on is one an in-order walk and a parallel scan would place
			// differently.
			seed: func(b, i int) float32 {
				if b == 1 {
					return 0.5 // the draw
				}
				if b != 0 {
					return 0
				}
				return 1.0 / 256
			},
		},
		{
			// Two rows of the same distribution against two different draws,
			// which is specs/043-per-row-values.md section 8's assertion made
			// cross-backend. The single-row case above cannot see it: at one
			// row the draws binding has one element and the walk's row base is
			// zero, so a kernel that ignored the row entirely would agree with
			// itself.
			//
			// Equal masses again, so the boundary each draw lands on is one an
			// in-order walk and a parallel scan would place differently.
			kernel: &kernels.SampleCategoricalKernel, counts: []int{512, 2, 2},
			uniforms: []any{kernels.SampleDims{Vocab: 256, Rows: 2}},
			groups:   accel.WorkgroupCount{X: 2},
			seed: func(b, i int) float32 {
				if b == 1 {
					// Far apart, so the two rows must land on tokens roughly a
					// hundred indices apart rather than merely on different
					// ones.
					if i == 0 {
						return 0.2
					}
					return 0.8
				}
				if b != 0 {
					return 0
				}
				return 1.0 / 256
			},
		},
		{
			// The quantized GEMM. Exact between backends: both widen each
			// product to f32 and sum in the same order, so quantization changes
			// what is computed and not whether the two agree about it.
			kernel:   &kernels.QuantMatMulKernel,
			counts:   []int{4 * 32, 32 * 8, 32 * 8 / 32, 4 * 8},
			uniforms: []any{kernels.GEMMDims{M: 4, N: 8, K: 32}},
			groups:   accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				// Binding 1 is the i8 quant plane and 2 the f16 scales. The
				// seed feeds writeSeed, which converts per dtype, so the quants
				// need values an int8 holds and the scales need to be non-zero
				// or every product is zero and the comparison proves nothing.
				switch b {
				case 1:
					return float32(i%201) - 100
				case 2:
					return 0.25 + float32(i%3)/8
				}
				return defaultSeed(b, i)
			},
		},
		{
			// The same kernel over f32 activations. The quant and scale planes
			// keep their widths -- a weight is loaded from a file and an
			// activation is produced by the graph -- so only binding 0 changes,
			// and the seed table is the one above.
			kernel:   &kernels.QuantMatMulF32Kernel,
			counts:   []int{4 * 32, 32 * 8, 32 * 8 / 32, 4 * 8},
			uniforms: []any{kernels.GEMMDims{M: 4, N: 8, K: 32}},
			groups:   accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				switch b {
				case 1:
					return float32(i%201) - 100
				case 2:
					return 0.25 + float32(i%3)/8
				}
				return defaultSeed(b, i)
			},
		},
		{
			// The tiled forms, over an output that is not a whole number of
			// tiles in either axis, so the edge guards run on both backends.
			kernel:   &kernels.QuantMatMulTiledKernel,
			counts:   []int{11 * 40, 40 * 21, 40*21/32 + 1, 11 * 21},
			uniforms: []any{kernels.GEMMDims{M: 11, N: 21, K: 40}},
			groups:   accel.WorkgroupCount{X: 2, Y: 2},
			seed: func(b, i int) float32 {
				switch b {
				case 1:
					return float32(i%201) - 100
				case 2:
					return 0.25 + float32(i%3)/8
				}
				return defaultSeed(b, i)
			},
		},
		{
			kernel:   &kernels.QuantMatMulTiledF32Kernel,
			counts:   []int{11 * 40, 40 * 21, 40*21/32 + 1, 11 * 21},
			uniforms: []any{kernels.GEMMDims{M: 11, N: 21, K: 40}},
			groups:   accel.WorkgroupCount{X: 2, Y: 2},
			seed: func(b, i int) float32 {
				switch b {
				case 1:
					return float32(i%201) - 100
				case 2:
					return 0.25 + float32(i%3)/8
				}
				return defaultSeed(b, i)
			},
		},
		{
			// The M=1 quantized selection. It folds K across the lanes and tree
			// reduces where QuantMatMul sums sequentially, so its rounding
			// differs from that kernel's -- but both backends run *this* order,
			// which is what this compares. The tree is the reason for the ULP
			// budget where QuantMatMul needed none.
			kernel:   &kernels.QuantMatVecKernel,
			counts:   []int{32, 32 * 8, 32 * 8 / 32, 8},
			uniforms: []any{kernels.GEMMDims{M: 1, N: 8, K: 32}},
			groups:   accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				switch b {
				case 1:
					return float32(i%201) - 100
				case 2:
					return 0.25 + float32(i%3)/8
				}
				return defaultSeed(b, i)
			},
		},
		{
			// The decode shape with f32 activations, which is where a
			// transformer spends nearly all of its dispatches: M=1 is every
			// step after the first, and the graph feeding it is f32.
			kernel:   &kernels.QuantMatVecF32Kernel,
			counts:   []int{32, 32 * 8, 32 * 8 / 32, 8},
			uniforms: []any{kernels.GEMMDims{M: 1, N: 8, K: 32}},
			groups:   accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				switch b {
				case 1:
					return float32(i%201) - 100
				case 2:
					return 0.25 + float32(i%3)/8
				}
				return defaultSeed(b, i)
			},
		},
		{
			// Pack, which tensor.Contiguous lowers to. Its uniform block holds
			// two eight-element arrays, and std140 gives an array member a
			// sixteen-byte stride whatever its element type -- so the two
			// lowerings must agree about a padding dimension that exists only
			// in the MSL declaration. Nothing else in the corpus has one.
			//
			// A transpose of a 3x4 matrix: the source is read with the strides
			// swapped, so a lowering that ignored a stride would produce the
			// *same* twelve values in the wrong order rather than obviously
			// wrong ones.
			kernel: &kernels.PackKernel,
			counts: []int{12, 12},
			uniforms: []any{kernels.PackParams{
				Rank: 2, Count: 12, Offset: 0,
				Extent: [8]uint32{4, 3},
				Stride: [8]uint32{1, 4},
			}},
			groups: accel.WorkgroupCount{X: 1},
		},
		{
			kernel:   &kernels.QuantRowsKernel,
			counts:   []int{8 * 32, 8 * 32 / 32, 4, 4 * 32},
			uniforms: []any{kernels.RowParams{Rows: 4, Width: 32, Capacity: 8}},
			groups:   accel.WorkgroupCount{X: 2},
			seed: func(b, i int) float32 {
				switch b {
				case 0:
					return float32(i%201) - 100
				case 1:
					return 0.5 + float32(i%2)/4
				case 2:
					return float32(i % 8) // ids inside the table
				}
				return defaultSeed(b, i)
			},
		},
		{
			// f16 to f32 is exact -- every f16 value is an f32 value -- so this
			// must agree bit for bit and would be the first thing to fail if a
			// backend's widening were not a widening.
			kernel: &kernels.CastF16ToF32Kernel, counts: []int{256, 256},
			groups: accel.WorkgroupCount{X: 4},
		},
		{
			// bf16 to f32 is exact for a stronger reason than f16's: bf16 *is*
			// f32's top half, so the widening is a 16-bit shift and a backend
			// that got it wrong would be wrong on every input rather than on
			// the ones near a rounding boundary.
			kernel: &kernels.CastBF16ToF32Kernel, counts: []int{256, 256},
			groups: accel.WorkgroupCount{X: 4},
			// bf16 carries seven mantissa bits, so the default seed's quarters
			// survive it -- but a value with more precision than that would
			// round on upload and the comparison would be about the seed.
			seed: func(b, i int) float32 { return float32((i+b*7)%13-6) / 4 },
		},
		{
			// f32 to f16 rounds, and to nearest-even, which is the only
			// rounding 002 admits. Bit for bit again: the two backends must
			// round the same way, and a backend that truncated instead would
			// differ on half its inputs.
			kernel: &kernels.CastF32ToF16Kernel, counts: []int{256, 256},
			groups: accel.WorkgroupCount{X: 4},
			// Values with bits below f16's precision, so the rounding actually
			// happens: a seed of small integers would be exact in f16 and would
			// compare equal however either backend rounded.
			seed: func(b, i int) float32 { return float32(i)*1.0009765625 - 100 },
		},
		{
			// Four query positions over four cached ones, causally masked, with
			// grouped query heads. Base zero, so the first query sees one
			// position and the last sees four -- which exercises the mask
			// across its whole range rather than at one length.
			kernel: &kernels.AttentionPrefillKernel,
			counts: []int{4 * 2 * 8, 4 * 1 * 8, 4 * 1 * 8, 1, 4 * 2 * 8},
			uniforms: []any{kernels.PrefillDims{
				QHeads: 2, KVHeads: 1, HeadDim: 8, QSeq: 4, Base: 0,
				Scale: float32(1) / float32(math.Sqrt(8)),
			}},
			groups: accel.WorkgroupCount{X: 4 * 2},
			ulp:    32, why: "a softmax over a masked row, per section 8's propagation",
		},
		{
			// The same prefill over an f16 cache. Both backends read the same
			// halves, so this compares the two lowerings of the widening
			// conversion rather than the conversion itself -- and the widening
			// is exact, so the ceiling is the f32 kernel's.
			kernel: &kernels.AttentionPrefillF16Kernel,
			counts: []int{4 * 2 * 8, 4 * 1 * 8, 4 * 1 * 8, 1, 4 * 2 * 8},
			uniforms: []any{kernels.PrefillDims{
				QHeads: 2, KVHeads: 1, HeadDim: 8, QSeq: 4, Base: 0,
				Scale: float32(1) / float32(math.Sqrt(8)),
			}},
			groups: accel.WorkgroupCount{X: 4 * 2},
			ulp:    32, why: "a softmax over a masked row, per section 8's propagation",
		},
		{
			// The same prefill through a page table, with the pages out of
			// order so the two backends must agree about the addressing and
			// not only about the attention.
			kernel: &kernels.AttentionPrefillPagedKernel,
			counts: []int{4 * 2 * 8, 8 * 1 * 8, 8 * 1 * 8, 2, 1, 4 * 2 * 8},
			uniforms: []any{kernels.PagedPrefillDims{
				QHeads: 2, KVHeads: 1, HeadDim: 8, QSeq: 4, Base: 0, Block: 2,
				Scale: float32(1) / float32(math.Sqrt(8)),
			}},
			groups: accel.WorkgroupCount{X: 4 * 2},
			seed: func(b, i int) float32 {
				if b == 3 { // the page table: blocks 3 and 1, out of order
					return float32([]int{3, 1}[i])
				}
				if b == 4 { // the cache length
					return 4
				}
				return defaultSeed(b, i)
			},
			ulp: 32, why: "a softmax over a masked row, per section 8's propagation",
		},
		{
			// The same paged prefill over an f16 cache. The combination issue
			// 25 was filed for, so it is checked on both backends rather than
			// only through the operator: a narrow load lowered differently
			// reads the halves as full floats and answers plausibly.
			kernel: &kernels.AttentionPrefillPagedF16Kernel,
			counts: []int{4 * 2 * 8, 8 * 1 * 8, 8 * 1 * 8, 2, 1, 4 * 2 * 8},
			uniforms: []any{kernels.PagedPrefillDims{
				QHeads: 2, KVHeads: 1, HeadDim: 8, QSeq: 4, Base: 0, Block: 2,
				Scale: float32(1) / float32(math.Sqrt(8)),
			}},
			groups: accel.WorkgroupCount{X: 4 * 2},
			seed: func(b, i int) float32 {
				if b == 3 { // the page table: blocks 3 and 1, out of order
					return float32([]int{3, 1}[i])
				}
				if b == 4 { // the cache length
					return 4
				}
				return defaultSeed(b, i)
			},
			ulp: 32, why: "a softmax over a masked row, per section 8's propagation",
		},
		{
			// A batch of three sequences of different lengths over interleaved
			// pages: the two backends must agree about which sequence reads
			// which blocks and stops where, not merely about the attention.
			kernel: &kernels.AttentionDecodeBatchedKernel,
			counts: []int{3 * 2 * 8, 12 * 4 * 1 * 8, 12 * 4 * 1 * 8, 3 * 3, 3, 3 * 2 * 8},
			uniforms: []any{kernels.BatchedDims{
				Batch: 3, QHeads: 2, KVHeads: 1, HeadDim: 8, Block: 4, MaxPages: 3,
				Scale: float32(1) / float32(math.Sqrt(8)),
			}},
			groups: accel.WorkgroupCount{X: 6},
			seed: func(b, i int) float32 {
				switch b {
				case 3: // page tables, interleaved across the three sequences
					return []float32{0, 3, 6, 1, 0, 0, 4, 7, 0}[i]
				case 4: // lengths, deliberately unequal
					return []float32{9, 3, 6}[i]
				}
				return defaultSeed(b, i)
			},
			ulp: 32, why: "a softmax over each sequence's cache, per section 8's propagation",
		},
		{
			// The paged decode, with pages deliberately out of order so the two
			// backends must agree about the *addressing* and not merely about
			// the attention. An identity page table would compare equal even
			// for a kernel that ignored the table.
			kernel: &kernels.AttentionDecodePagedKernel,
			counts: []int{2 * 8, 8 * 4 * 1 * 8, 8 * 4 * 1 * 8, 2, 1, 2 * 8},
			uniforms: []any{kernels.PagedDims{
				QHeads: 2, KVHeads: 1, HeadDim: 8, Block: 4,
				Scale: float32(1) / float32(math.Sqrt(8)),
			}},
			groups: accel.WorkgroupCount{X: 2},
			seed: func(b, i int) float32 {
				if b == 3 {
					// The page table: logical block 0 lives at physical 5 and
					// logical block 1 at physical 2.
					return []float32{5, 2}[i]
				}
				return defaultSeed(b, i)
			},
			ulp: 32, why: "a softmax over the cache, per section 8's propagation",
		},
		{
			// The paged decode over an f16 cache, where the two memory savings
			// meet. The page table is out of order for the case above's reason,
			// and the ceiling is the f32 kernel's because the widening is
			// exact.
			kernel: &kernels.AttentionDecodePagedF16Kernel,
			counts: []int{2 * 8, 8 * 4 * 1 * 8, 8 * 4 * 1 * 8, 2, 1, 2 * 8},
			uniforms: []any{kernels.PagedDims{
				QHeads: 2, KVHeads: 1, HeadDim: 8, Block: 4,
				Scale: float32(1) / float32(math.Sqrt(8)),
			}},
			groups: accel.WorkgroupCount{X: 2},
			seed: func(b, i int) float32 {
				if b == 3 {
					return []float32{5, 2}[i]
				}
				return defaultSeed(b, i)
			},
			ulp: 32, why: "a softmax over the cache, per section 8's propagation",
		},
		{
			kernel: &kernels.AttentionDecodeKernel,
			counts: []int{2 * 8, 3 * 1 * 8, 3 * 1 * 8, 1, 2 * 8},
			uniforms: []any{kernels.AttnDims{
				QHeads: 2, KVHeads: 1, HeadDim: 8, Scale: float32(1) / float32(math.Sqrt(8)),
			}},
			groups: accel.WorkgroupCount{X: 2},
			ulp:    32, why: "a softmax composed with two dot products, per section 8's propagation",
		},
		{
			// The same shape over an f16 cache. Both backends read the same
			// halves, so this compares the two *lowerings* of the widening
			// conversion rather than the conversion itself -- and the widening
			// is exact, so the ceiling is the f32 kernel's.
			kernel: &kernels.AttentionDecodeF16Kernel,
			counts: []int{2 * 8, 3 * 1 * 8, 3 * 1 * 8, 1, 2 * 8},
			uniforms: []any{kernels.AttnDims{
				QHeads: 2, KVHeads: 1, HeadDim: 8, Scale: float32(1) / float32(math.Sqrt(8)),
			}},
			groups: accel.WorkgroupCount{X: 2},
			ulp:    32, why: "a softmax composed with two dot products, per section 8's propagation",
		},
		{
			// The counting pass. Both backends must reach the same counts, and
			// the reason this is comparable at all is that the accumulation is
			// an integer atomic: an f32 one would be class E, which section 9
			// of the harness spec excludes from bit comparison, and there would
			// be nothing here to compare.
			//
			// The history repeats ids on purpose -- a frequency penalty exists
			// to count repeats -- so several invocations increment one address
			// and the two backends' orders differ.
			kernel: &kernels.PenaltyCountKernel,
			counts: []int{64, 256},
			uniforms: []any{kernels.PenaltyDims{
				Vocab: 256, History: 64, Count: 64,
				Repetition: 1.5, Presence: 0.2, Frequency: 0.05,
			}},
			groups: accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b == 1 {
					return 0 // the counts start empty
				}
				// Ids that repeat, and a few past the vocabulary so the drop
				// rather than clamp rule is compared too.
				if i%17 == 0 {
					return 300
				}
				return float32(i % 23)
			},
			why: "integer counts, exact on both backends",
		},
		{
			// The apply pass, including both branches of the divisive penalty:
			// the seeded logits change sign, so a backend that lowered the
			// branch differently would disagree on the negative half only.
			kernel: &kernels.PenaltyApplyKernel,
			counts: []int{256, 256, 256},
			uniforms: []any{kernels.PenaltyDims{
				Vocab: 256, History: 64, Count: 64,
				Repetition: 1.5, Presence: 0.2, Frequency: 0.05,
			}},
			groups: accel.WorkgroupCount{X: 4},
			seed: func(b, i int) float32 {
				if b == 1 {
					// Counts including zero, so the untouched path is compared
					// alongside the penalised one.
					return float32(i % 3)
				}
				return defaultSeed(b, i)
			},
			why: "one divide, one multiply and two subtracts, all exact per section 2",
		},
		{
			// The signed atomics, whose min and max are the only two of the six
			// that a signed and an unsigned lowering compute differently. The
			// seeded state is negative for those two indices, which is what
			// makes this a comparison of signedness rather than of reachability.
			// A gated delta recurrence: three passes over the state per token,
			// with one workgroup per (sequence, head) walking its own tokens.
			// The state is both read and written, so this compares the scan's
			// result as well as its output.
			kernel: &kernels.LinearAttentionKernel,
			counts: []int{4 * 2 * 6, 4 * 2 * 6, 4 * 2 * 4, 4, 4, 3, 2 * 2 * 4 * 6, 4 * 2 * 4},
			uniforms: []any{kernels.LinearDims{
				Batch: 2, Heads: 2, KeyDim: 6, ValueDim: 4, GateHeads: 1,
			}},
			groups: accel.WorkgroupCount{X: 2 * 2},
			seed: func(b, i int) float32 {
				switch b {
				case 3: // alpha: a decay under one
					return 0.9
				case 4: // beta: a write rate under one
					return 0.5
				case 5: // offsets: sequence 0 gets three tokens, sequence 1 one
					return []float32{0, 3, 4}[i]
				case 7: // the output starts empty
					return 0
				}
				return defaultSeed(b, i) / 4
			},
			ulp: 32, why: "three sums of products per token, per section 8's propagation",
		},
		{
			// The same scan with a gate per head (accel issue 27). The gate
			// index is `tok*GateHeads + h%GateHeads`, so this is the case where
			// the modulo is the identity rather than a constant zero, and the
			// two heads are given different decays so a lowering that dropped
			// the head term disagrees on a value rather than on nothing.
			kernel: &kernels.LinearAttentionKernel,
			counts: []int{4 * 2 * 6, 4 * 2 * 6, 4 * 2 * 4, 4 * 2, 4 * 2, 3, 2 * 2 * 4 * 6, 4 * 2 * 4},
			uniforms: []any{kernels.LinearDims{
				Batch: 2, Heads: 2, KeyDim: 6, ValueDim: 4, GateHeads: 2,
			}},
			groups: accel.WorkgroupCount{X: 2 * 2},
			seed: func(b, i int) float32 {
				switch b {
				case 3: // alpha, one per (token, head): head 1 forgets faster
					return 0.9 - 0.4*float32(i%2)
				case 4: // beta, likewise
					return 0.5 - 0.2*float32(i%2)
				case 5:
					return []float32{0, 3, 4}[i]
				case 7:
					return 0
				}
				return defaultSeed(b, i) / 4
			},
			ulp: 32, why: "three sums of products per token, per section 8's propagation",
		},
		{
			// A 4-bit matvec: nibble extraction, a zero point, and a tree
			// reduction. The two backends must agree on the *unpacking* as well
			// as the arithmetic, and a shift lowered differently would show as
			// a wholly different product rather than as a rounding difference.
			kernel:   &kernels.QuantMatVecInt4Kernel,
			counts:   []int{128, 128 / 8 * 2, 2, 2, 2},
			uniforms: []any{kernels.GEMMDims{K: 128, N: 2}},
			groups:   accel.WorkgroupCount{X: 2},
			seed: func(b, i int) float32 {
				switch b {
				case 1: // the packed codes, as u32 words
					return float32(i%251 + 1)
				case 2: // scales: positive and modest
					return 0.25
				case 3: // zero points
					return 4
				case 4:
					return 0
				}
				return defaultSeed(b, i)
			},
			why: "a sum of products, per section 7's reduction bound",
			ulp: 8,
		},
		{
			// The same matvec with a constant group: quant.Int4Quantize stores
			// one as a zero scale with the value in the zero point, and the
			// kernel selects the zero point when the scale is zero. Group 0
			// takes that path and group 1 the ordinary one, so the select is
			// lowered and compared on both backends rather than only read.
			//
			// Groups are over the flat weight index k*N+n, so K=256 by N=2 is
			// four of them.
			kernel:   &kernels.QuantMatVecInt4Kernel,
			counts:   []int{256, 256 * 2 / 8, 4, 4, 2},
			uniforms: []any{kernels.GEMMDims{K: 256, N: 2}},
			groups:   accel.WorkgroupCount{X: 2},
			seed: func(b, i int) float32 {
				switch b {
				case 1:
					return float32(i%251 + 1)
				case 2: // group 0 is constant, the rest are not
					return []float32{0, 0.25, 0.25, 0.25}[i]
				case 3: // the constant, then ordinary zero points
					return []float32{0.75, 4, 4, 4}[i]
				case 4:
					return 0
				}
				return defaultSeed(b, i)
			},
			why: "a sum of products, per section 7's reduction bound",
			ulp: 8,
		},
		{
			// The tiled 4-bit product: the same unpacking as the matvec above,
			// moved into a shared-tile load, plus two barriers. K is not a
			// multiple of TileK, so the edge guards run on both backends and a
			// tile the two lower differently shows as a wrong element rather
			// than as a rounding difference.
			kernel:   &kernels.QuantMatMulInt4Kernel,
			counts:   []int{3 * 40, 40 * 16 / 8, 8, 8, 3 * 16},
			uniforms: []any{kernels.GEMMDims{M: 3, K: 40, N: 16}},
			groups:   accel.WorkgroupCount{X: 1, Y: 1},
			seed: func(b, i int) float32 {
				switch b {
				case 1: // the packed codes, as u32 words
					return float32(i%251 + 1)
				case 2: // scales: positive and modest
					return 0.25
				case 3: // zero points
					return 4
				case 4: // the output starts empty
					return 0
				}
				return defaultSeed(b, i)
			},
			why: "a sum of products over a shared tile, per section 7's reduction bound",
			ulp: 8,
		},
		{
			// The tiled grouped product: a workgroup per (expert, column tile)
			// walking its own segment through shared tiles. The offsets decide
			// the loop bound, so a backend that disagreed about the segment
			// walk would show as whole rows differing rather than as rounding.
			kernel: &kernels.GroupedMatMulKernel,
			counts: []int{6 * 40, 2 * 40 * 20, 3, 6 * 20},
			uniforms: []any{kernels.GroupedTiledDims{
				Experts: 2, Tokens: 6, K: 40, N: 20,
			}},
			groups: accel.WorkgroupCount{X: 2, Y: 2},
			seed: func(b, i int) float32 {
				switch b {
				case 2: // offsets: expert 0 gets four tokens, expert 1 two
					return []float32{0, 4, 6}[i]
				case 3: // the output starts empty
					return 0
				}
				return defaultSeed(b, i)
			},
			why: "a sum of products over a shared tile, per section 7's reduction bound",
			ulp: 8,
		},
		{
			// A grouped product: a segment lookup choosing which weight matrix
			// a token multiplies against, then the row kernels' reduction. The
			// lookup is integer and must agree exactly; the reduction carries
			// section 7's bound.
			kernel:   &kernels.GroupedMatVecKernel,
			counts:   []int{6 * 32, 3 * 32 * 4, 4, 6 * 4},
			uniforms: []any{kernels.GroupedDims{Experts: 3, K: 32, N: 4}},
			groups:   accel.WorkgroupCount{X: 6 * 4},
			seed: func(b, i int) float32 {
				switch b {
				case 2: // offsets: experts get 2, 0, 4 tokens
					return []float32{0, 2, 2, 6}[i]
				case 3:
					return 0
				}
				return defaultSeed(b, i)
			},
			why: "a sum of products, per section 7's reduction bound",
			ulp: 8,
		},
		{
			// The same ragged step over an f16 cache. Both backends read the
			// same halves, so this compares the two lowerings of the widening
			// rather than the conversion, and the ceiling is the f32 kernel's
			// because the widening is exact.
			kernel: &kernels.AttentionRaggedF16Kernel,
			counts: []int{6 * 2 * 8, 12 * 4 * 1 * 8, 12 * 4 * 1 * 8, 3 * 4, 3, 4, 6 * 2 * 8},
			uniforms: []any{kernels.RaggedDims{
				Batch: 3, QHeads: 2, KVHeads: 1, HeadDim: 8,
				Block: 4, MaxPages: 4, Scale: float32(1) / float32(math.Sqrt(8)),
			}},
			groups: accel.WorkgroupCount{X: 6 * 2},
			seed: func(b, i int) float32 {
				switch b {
				case 2:
					return float32(i%7+1) / 4
				case 3:
					return float32(i)
				case 4:
					return []float32{4, 2, 5}[i]
				case 5:
					return []float32{0, 3, 4, 6}[i]
				}
				return defaultSeed(b, i)
			},
			ulp: 32, why: "a softmax over the cache, per section 8's propagation",
		},
		{
			// The ragged step with a sequence shorter than its count: sequence
			// 0 contributes three tokens over two cached positions, so its
			// first token has no position and writes zero, and the kernel's
			// guard against the wrapping limit runs on both backends. Both
			// cache widths, because each carries the guard.
			kernel: &kernels.AttentionRaggedKernel,
			counts: []int{6 * 2 * 8, 12 * 4 * 1 * 8, 12 * 4 * 1 * 8, 3 * 4, 3, 4, 6 * 2 * 8},
			uniforms: []any{kernels.RaggedDims{
				Batch: 3, QHeads: 2, KVHeads: 1, HeadDim: 8,
				Block: 4, MaxPages: 4, Scale: float32(1) / float32(math.Sqrt(8)),
			}},
			groups: accel.WorkgroupCount{X: 6 * 2},
			seed: func(b, i int) float32 {
				switch b {
				case 2:
					return float32(i%7+1) / 4
				case 3:
					return float32(i)
				case 4: // lengths: sequence 0 holds fewer positions than its count
					return []float32{2, 2, 5}[i]
				case 5: // offsets: counts 3, 1, 2
					return []float32{0, 3, 4, 6}[i]
				}
				return defaultSeed(b, i)
			},
			ulp: 32, why: "a softmax over the cache, per section 8's propagation",
		},
		{
			kernel: &kernels.AttentionRaggedF16Kernel,
			counts: []int{6 * 2 * 8, 12 * 4 * 1 * 8, 12 * 4 * 1 * 8, 3 * 4, 3, 4, 6 * 2 * 8},
			uniforms: []any{kernels.RaggedDims{
				Batch: 3, QHeads: 2, KVHeads: 1, HeadDim: 8,
				Block: 4, MaxPages: 4, Scale: float32(1) / float32(math.Sqrt(8)),
			}},
			groups: accel.WorkgroupCount{X: 6 * 2},
			seed: func(b, i int) float32 {
				switch b {
				case 2:
					return float32(i%7+1) / 4
				case 3:
					return float32(i)
				case 4:
					return []float32{2, 2, 5}[i]
				case 5:
					return []float32{0, 3, 4, 6}[i]
				}
				return defaultSeed(b, i)
			},
			ulp: 32, why: "a softmax over the cache, per section 8's propagation",
		},
		{
			// The exclusive prefix sum. Integer, so the two backends agree
			// exactly, and the seeded counts include a zero so the repeated
			// offset a zero-count row produces is compared too.
			kernel:   &kernels.SegmentOffsetsKernel,
			counts:   []int{4, 5},
			uniforms: []any{kernels.SegmentDims{Rows: 4}},
			groups:   accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b == 1 {
					return 0
				}
				return []float32{3, 0, 4, 1}[i]
			},
			why: "an integer prefix sum, exact on both backends",
		},
		{
			// A ragged step: one sequence contributing several tokens beside
			// sequences contributing one, which is the mixed shape the extent
			// exists for. The offsets are seeded rather than derived, so this
			// compares the attention kernel alone.
			kernel: &kernels.AttentionRaggedKernel,
			counts: []int{6 * 2 * 8, 12 * 4 * 1 * 8, 12 * 4 * 1 * 8, 3 * 4, 3, 4, 6 * 2 * 8},
			uniforms: []any{kernels.RaggedDims{
				Batch: 3, QHeads: 2, KVHeads: 1, HeadDim: 8,
				Block: 4, MaxPages: 4, Scale: float32(1) / float32(math.Sqrt(8)),
			}},
			groups: accel.WorkgroupCount{X: 6 * 2},
			seed: func(b, i int) float32 {
				switch b {
				case 2:
					// V is seeded strictly positive, and that is about the
					// measurement rather than about the kernel. defaultSeed is
					// symmetric about zero, so a softmax-weighted sum of it
					// cancels: the first run of this case produced an output of
					// 8.5e-4 from values near +/-1, where 266 ULP is 2e-5
					// relative and is ordinary f32 softmax noise magnified by
					// the cancellation. Raising the ceiling would have hidden
					// that, and the ceiling is derived from section 8's
					// propagation rather than from what a run produced. A
					// positive V makes the ULP count measure the two lowerings.
					return float32(i%7+1) / 4
				case 3: // the page table: block ids, one row per sequence
					return float32(i)
				case 4: // lengths, one per sequence
					return []float32{4, 2, 5}[i]
				case 5: // offsets: counts 3, 1, 2
					return []float32{0, 3, 4, 6}[i]
				}
				return defaultSeed(b, i)
			},
			ulp: 32, why: "a softmax over the cache, per section 8's propagation",
		},
		{
			kernel: &kernels.AtomicOpsI32Kernel,
			counts: []int{7, 7},
			groups: accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b == 1 {
					return 0
				}
				// state[2] and state[3] negative, so min and max over the
				// kernel's positive operand disagree between the two readings.
				return []float32{10, 10, -5, -5, 10, -1, 7}[i]
			},
			why: "integer atomics, exact on both backends",
		},
		{
			kernel:   &kernels.ElemBiasKernel,
			counts:   []int{256, 256},
			uniforms: []any{kernels.BiasParams{Offset: -3}},
			groups:   accel.WorkgroupCount{X: 4},
			// A signed uniform, which is the case no kernel here had. The two
			// backends must agree on the arithmetic and not only on the bits:
			// int32(-3) and uint32(4294967293) are the same four bytes, so a
			// backend reading the uniform as unsigned round-trips it unchanged
			// and adds the wrong number.
			why: "an integer add, exact on both backends",
		},
		{
			kernel:   &kernels.PenaltyClearKernel,
			counts:   []int{256},
			uniforms: []any{kernels.PenaltyDims{Vocab: 256, History: 64, Count: 64}},
			groups:   accel.WorkgroupCount{X: 4},
			seed:     func(b, i int) float32 { return float32(i % 7) },
			why:      "a store of zero",
		},
	}
}

// Every kernel that lowers to MSL is in a differential case.
//
// The generated Kernels slice is the universe and the table is checked against
// it rather than trusted. A kernel added and never listed looks exactly like
// one that passes, which is the whole reason the corpus is generated: the list
// nobody maintains by hand is the one that cannot go stale.
func TestEveryLoweredKernelIsInADifferentialCase(t *testing.T) {
	listed := map[string]bool{}
	for _, c := range diffCases() {
		listed[c.kernel.Name] = true
	}
	for _, k := range kernels.Kernels {
		if k.MSL != "" && !listed[k.Name] {
			t.Errorf("%s lowers to MSL and is in no differential case, so nothing "+
				"compares its two lowerings", k.Name)
		}
	}
}
