// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package testkernels_test

import (
	"fmt"
	"golang.design/x/accel/kernelabi"
	"math"
	"os"
	"testing"
	"time"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/numeq"
	"golang.design/x/accel/internal/testkernels"
)

// # The corpus differential
//
// specs/022-msl-target.md's central criterion: every kernel in the corpus runs
// on the CPU backend and on Metal, from one generated record, and the two
// agree. The CPU side runs the resumable lowering with a program counter and
// the Metal side runs the authored structure with a real barrier, so a
// disagreement is the transform's and nothing else's.
//
// **Bit for bit, not within a tolerance.** The measured profile
// (specs/022-msl-target.md) has both backends rounding to nearest-even with
// contraction off, which is what specs/008-numerics.md requires before an exact
// comparison may be claimed. A tolerance would hide the failure this exists to
// catch: a contracted multiply-add moves a result by about one part in 2^24,
// and even 1e-6 sails past it.
//
// The one restriction is the domain. Apple GPUs flush a subnormal *result* to
// zero, so the inputs below are scaled to keep every intermediate well inside
// the normal range. That is a narrower domain, not a wider bound, which is the
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
		{kernel: &testkernels.AddKernel, counts: []int{256, 256, 256}, groups: accel.WorkgroupCount{X: 4}},
		{kernel: &testkernels.ElemAddKernel, counts: []int{256, 256, 256}, groups: accel.WorkgroupCount{X: 4}},
		{kernel: &testkernels.ElemMulKernel, counts: []int{256, 256, 256}, groups: accel.WorkgroupCount{X: 4}},
		{kernel: &testkernels.ScaleKernel, counts: []int{256, 256}, groups: accel.WorkgroupCount{X: 4}},
		{
			kernel: &testkernels.ElemScaleKernel, counts: []int{256, 256},
			uniforms: []any{testkernels.ScaleParams{Factor: 2.5}},
			groups:   accel.WorkgroupCount{X: 4},
		},
		{
			// x*sigmoid(x) reaches exp and a division. Section 6 bounds each at
			// 4 and 2.5 ULP from correctly rounded, and two implementations may
			// sit on opposite sides, so their mutual distance is up to twice
			// each: 8 + 5, rounded up to 16 for the surrounding multiply.
			kernel: &testkernels.SiLUKernel, counts: []int{256, 256},
			groups: accel.WorkgroupCount{X: 4},
			ulp:    16, why: "exp (4 ULP) and a division (2.5 ULP), doubled for two implementations",
		},
		{
			kernel: &testkernels.SwiGLUKernel, counts: []int{256, 256, 256},
			groups: accel.WorkgroupCount{X: 4},
			ulp:    16, why: "SiLU's exp and division, then a multiply",
		},
		{kernel: &testkernels.SegmentSumKernel, counts: []int{256, 8}, groups: accel.WorkgroupCount{X: 1}},
		{kernel: &testkernels.CountAboveKernel, counts: []int{256, 1}, groups: accel.WorkgroupCount{X: 16}},
		{kernel: &testkernels.HistogramKernel, counts: []int{256, 4}, groups: accel.WorkgroupCount{X: 4}},
		{kernel: &testkernels.CountWorkgroupsKernel, counts: []int{1}, groups: accel.WorkgroupCount{X: 7}},
		{
			// Ten state slots and ten results, which is what the kernel indexes.
			// Its ninth case compare-exchanges against an expected 1, so the
			// seed puts a 1 there and a 0 in the tenth: one swap must happen and
			// one must not, and a seed that made both cases alike would let a
			// compare-exchange that never swaps pass.
			kernel: &testkernels.AtomicOpsKernel, counts: []int{10, 10},
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
		{kernel: &testkernels.ExchangeKernel, counts: []int{64, 64}, groups: accel.WorkgroupCount{X: 1}},
		{kernel: &testkernels.ReduceLoopKernel, counts: []int{64, 1}, groups: accel.WorkgroupCount{X: 1}},
		{kernel: &testkernels.ReduceUnrolledKernel, counts: []int{64, 1}, groups: accel.WorkgroupCount{X: 1}},
		{kernel: &testkernels.ReduceSumKernel, counts: []int{512, 4}, groups: accel.WorkgroupCount{X: 4}},
		{kernel: &testkernels.SubgroupReduceKernel, counts: []int{64, 1}, groups: accel.WorkgroupCount{X: 1}},
		{
			kernel: &testkernels.SubgroupReduceFallbackKernel, counts: []int{64, 1, 1},
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
		{
			kernel: &testkernels.NormalizeKernel, counts: []int{64, 64, 64},
			groups: accel.WorkgroupCount{X: 2},
		},
		{
			kernel: &testkernels.RMSNormKernel, counts: []int{4 * width, width, 4 * width},
			uniforms: []any{testkernels.RowDims{Rows: 4, Width: width, Eps: 1e-5}},
			groups:   accel.WorkgroupCount{X: 4},
			ulp:      16, why: "rsqrt (4 ULP), doubled for two implementations, then a multiply",
		},
		{
			kernel: &testkernels.SoftmaxKernel, counts: []int{4 * width, 4 * width},
			uniforms: []any{testkernels.RowDims{Rows: 4, Width: width}},
			groups:   accel.WorkgroupCount{X: 4},
			ulp:      16, why: "exp (4 ULP) and a division (2.5 ULP), doubled, then the reduction",
		},
		{
			kernel: &testkernels.GatherRowsKernel, counts: []int{8 * 16, 4, 4 * 16},
			uniforms: []any{testkernels.RowParams{Rows: 4, Width: 16, Capacity: 8}},
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
			kernel: &testkernels.GatherRowsF16Kernel, counts: []int{8 * 16, 4, 4 * 16},
			uniforms: []any{testkernels.RowParams{Rows: 4, Width: 16, Capacity: 8}},
			groups:   accel.WorkgroupCount{X: 1},
			seed: func(b, i int) float32 {
				if b == 1 {
					return float32(i % 8) // row ids, inside the table
				}
				return defaultSeed(b, i)
			},
		},
		{
			kernel: &testkernels.ScatterRowsKernel, counts: []int{4 * 16, 4, 8 * 16},
			uniforms: []any{testkernels.RowParams{Rows: 4, Width: 16, Capacity: 8}},
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
			kernel: &testkernels.RoPEKernel, counts: []int{4, 4 * 16},
			uniforms: []any{testkernels.RoPEParams{
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
			kernel: &testkernels.TransformKernel, counts: []int{64, 64},
			uniforms: []any{testkernels.Params{
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
			// Tails on all three axes, so every guarded edge of the tiled GEMM
			// runs. A shape that fitted the tile exactly would exercise none of
			// them, and every one is a place an off-by-one produces a plausible
			// matrix.
			kernel: &testkernels.MatMulTiledKernel, counts: []int{9 * 23, 23 * 19, 9 * 19},
			uniforms: []any{testkernels.GEMMDims{M: 9, N: 19, K: 23}},
			groups: accel.WorkgroupCount{
				X: (19 + testkernels.TileN - 1) / testkernels.TileN,
				Y: (9 + testkernels.TileM - 1) / testkernels.TileM,
			},
		},
		{
			// The same shape with f32 operands. Bit-exact for the same reason
			// the f16 one is: both accumulate f32 in the same order, and the
			// tiles differ only in what they hold.
			kernel:   &testkernels.MatMulTiledF32Kernel,
			counts:   []int{9 * 23, 23 * 19, 9 * 19},
			uniforms: []any{testkernels.GEMMDims{M: 9, N: 19, K: 23}},
			groups: accel.WorkgroupCount{
				X: (19 + testkernels.TileN - 1) / testkernels.TileN,
				Y: (9 + testkernels.TileM - 1) / testkernels.TileM,
			},
		},
		{
			kernel: &testkernels.LinearTiledKernel, counts: []int{9 * 23, 23 * 19, 19, 9 * 19},
			uniforms: []any{testkernels.GEMMDims{M: 9, N: 19, K: 23}},
			groups: accel.WorkgroupCount{
				X: (19 + testkernels.TileN - 1) / testkernels.TileN,
				Y: (9 + testkernels.TileM - 1) / testkernels.TileM,
			},
		},
		{
			kernel: &testkernels.MatVecKernel, counts: []int{1 * 40, 40 * 12, 1 * 12},
			uniforms: []any{testkernels.GEMMDims{M: 1, N: 12, K: 40}},
			groups:   accel.WorkgroupCount{X: 12},
		},
		{
			// Top-k over a distribution with a deliberate plateau at the
			// boundary: the two backends must keep the same *set*, and a tie
			// rule that differed would keep different entries while keeping the
			// same count.
			kernel: &testkernels.TopKMaskKernel, counts: []int{256, 256},
			uniforms: []any{testkernels.TopDims{Vocab: 256, K: 12}},
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
			kernel: &testkernels.TopPMaskKernel, counts: []int{256, 256},
			uniforms: []any{testkernels.TopDims{Vocab: 256, P: 0.6}},
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
			// Argmax over a vocabulary with a deliberate plateau at the top.
			// A tie rule that differed between the backends would move the
			// answer, which is the one thing this kernel must not do -- and a
			// distribution of distinct values would compare equal whatever
			// either backend did with ties.
			kernel: &testkernels.SampleArgmaxKernel, counts: []int{512, 1},
			uniforms: []any{testkernels.SampleDims{Vocab: 512}},
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
			kernel: &testkernels.SampleCategoricalKernel, counts: []int{256, 1, 1},
			uniforms: []any{testkernels.SampleDims{Vocab: 256, Rows: 1}},
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
			// The quantized GEMM. Exact between backends: both widen each
			// product to f32 and sum in the same order, so quantization changes
			// what is computed and not whether the two agree about it.
			kernel:   &testkernels.QuantMatMulKernel,
			counts:   []int{4 * 32, 32 * 8, 32 * 8 / 32, 4 * 8},
			uniforms: []any{testkernels.GEMMDims{M: 4, N: 8, K: 32}},
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
			// The M=1 quantized selection. It folds K across the lanes and tree
			// reduces where QuantMatMul sums sequentially, so its rounding
			// differs from that kernel's -- but both backends run *this* order,
			// which is what this compares. The tree is the reason for the ULP
			// budget where QuantMatMul needed none.
			kernel:   &testkernels.QuantMatVecKernel,
			counts:   []int{32, 32 * 8, 32 * 8 / 32, 8},
			uniforms: []any{testkernels.GEMMDims{M: 1, N: 8, K: 32}},
			groups:   accel.WorkgroupCount{X: 8},
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
			kernel:   &testkernels.QuantRowsKernel,
			counts:   []int{8 * 32, 8 * 32 / 32, 4, 4 * 32},
			uniforms: []any{testkernels.RowParams{Rows: 4, Width: 32, Capacity: 8}},
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
			kernel: &testkernels.CastF16ToF32Kernel, counts: []int{256, 256},
			groups: accel.WorkgroupCount{X: 4},
		},
		{
			// f32 to f16 rounds, and to nearest-even, which is the only
			// rounding 002 admits. Bit for bit again: the two backends must
			// round the same way, and a backend that truncated instead would
			// differ on half its inputs.
			kernel: &testkernels.CastF32ToF16Kernel, counts: []int{256, 256},
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
			kernel: &testkernels.AttentionPrefillKernel,
			counts: []int{4 * 2 * 8, 4 * 1 * 8, 4 * 1 * 8, 1, 4 * 2 * 8},
			uniforms: []any{testkernels.PrefillDims{
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
			kernel: &testkernels.AttentionPrefillPagedKernel,
			counts: []int{4 * 2 * 8, 8 * 1 * 8, 8 * 1 * 8, 2, 1, 4 * 2 * 8},
			uniforms: []any{testkernels.PagedPrefillDims{
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
			kernel: &testkernels.AttentionDecodeBatchedKernel,
			counts: []int{3 * 2 * 8, 12 * 4 * 1 * 8, 12 * 4 * 1 * 8, 3 * 3, 3, 3 * 2 * 8},
			uniforms: []any{testkernels.BatchedDims{
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
			kernel: &testkernels.AttentionDecodePagedKernel,
			counts: []int{2 * 8, 8 * 4 * 1 * 8, 8 * 4 * 1 * 8, 2, 1, 2 * 8},
			uniforms: []any{testkernels.PagedDims{
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
			kernel: &testkernels.AttentionDecodeKernel,
			counts: []int{2 * 8, 3 * 1 * 8, 3 * 1 * 8, 1, 2 * 8},
			uniforms: []any{testkernels.AttnDims{
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
			kernel: &testkernels.AttentionDecodeF16Kernel,
			counts: []int{2 * 8, 3 * 1 * 8, 3 * 1 * 8, 1, 2 * 8},
			uniforms: []any{testkernels.AttnDims{
				QHeads: 2, KVHeads: 1, HeadDim: 8, Scale: float32(1) / float32(math.Sqrt(8)),
			}},
			groups: accel.WorkgroupCount{X: 2},
			ulp:    32, why: "a softmax composed with two dot products, per section 8's propagation",
		},
	}
}

// Every corpus kernel agrees between the CPU oracle and Metal, bit for bit.
func TestTheCorpusAgreesOnCPUAndMetal(t *testing.T) {

	cases := diffCases()

	// The table is checked against the corpus rather than trusted. A kernel
	// added and never listed here would otherwise look exactly like one that
	// passes, which is the failure the generated Kernels slice exists to make
	// impossible.
	listed := map[string]bool{}
	for _, c := range cases {
		listed[c.kernel.Name] = true
	}
	for _, k := range testkernels.Kernels {
		if k.MSL != "" && !listed[k.Name] {
			t.Errorf("%s lowers to MSL and is in no differential case, so nothing "+
				"compares its two lowerings", k.Name)
		}
	}

	gpu := openMetalDevice(t)

	// The oracle is configured to the device it is checking. The CPU backend
	// emulates subgroups at a width a caller chooses, and its default is 4
	// while this device executes 32 -- so a subgroup reduction over 64 elements
	// would produce different partial sums on each and the differential would
	// be comparing two different computations. specs/006-backends.md section 5
	// makes the width an option for exactly this.
	width := gpu.Info().Limits.MinSubgroupSize
	subgroupWidth = width
	cpu, err := accel.OpenCPU(accel.CPUOptions{SubgroupSize: width})
	if err != nil {
		t.Fatalf("open CPU at subgroup width %d: %v", width, err)
	}
	defer cpu.Close()
	if got := gpu.Info().Limits.MaxSubgroupSize; got != width {
		t.Fatalf("this device reports a subgroup width range of [%d, %d]; the oracle can "+
			"emulate one width, so a varying one needs a different arrangement",
			width, got)
	}

	for _, c := range cases {
		t.Run(c.kernel.Name, func(t *testing.T) {
			want := runCase(t, cpu, c)
			got := runCase(t, gpu, c)
			for b := range want {
				if len(want[b]) != len(got[b]) {
					t.Fatalf("binding %d: %d elements on the CPU and %d on Metal",
						b, len(want[b]), len(got[b]))
				}
				var r numeq.Report
				switch {
				case c.abs > 0:
					r = withinAbs(got[b], want[b], c.abs)
				default:
					r = numeq.WithinULP(got[b], want[b], c.ulp)
				}
				if !r.Equal {
					t.Fatalf("binding %d (%s): %v\n  the ceiling comes from %s\n  both "+
						"lowerings come from one IR, so a disagreement beyond it is the "+
						"transform's", b, c.kernel.Bindings[b].Name, r, ceilingNote(c))
				}
			}
		})
	}
}

// ceilingNote explains where a case's ceiling came from.
func ceilingNote(c diffCase) string {
	if c.why == "" {
		return "no bounded primitive in this kernel, so the two lowerings must agree exactly"
	}
	return c.why
}

// withinAbs compares against an absolute ceiling.
//
// Separate from numeq.WithinULP rather than a mode of it, because the two
// answer different questions and specs/008-numerics.md uses each where the
// other is meaningless: ULP near a zero crossing, and absolute across many
// binades.
func withinAbs(got, want []float32, ceiling float64) numeq.Report {
	r := numeq.Report{Equal: true, FirstDiff: -1, Len: len(got), WantLen: len(want)}
	if len(got) != len(want) {
		r.Equal = false
		return r
	}
	for i := range got {
		g, w := got[i], want[i]
		gNaN, wNaN := g != g, w != w
		bad := gNaN != wNaN
		if !gNaN && !wNaN {
			bad = math.Abs(float64(g)-float64(w)) > ceiling
		}
		if !bad {
			continue
		}
		r.Diffs++
		if r.FirstDiff < 0 {
			r.FirstDiff, r.Equal = i, false
			r.Got = fmt.Sprintf("%v", g)
			r.Want = fmt.Sprintf("%v, differing by %g against a ceiling of %g",
				w, math.Abs(float64(g)-float64(w)), ceiling)
		}
	}
	return r
}

// runCase records one graph, submits it, and reads every written binding back.
//
// Every binding is uploaded, including the outputs: a kernel that writes only
// part of its output leaves the rest at whatever the buffer held, and comparing
// uninitialized memory between two backends is a flake waiting to happen.
func runCase(t *testing.T, d *accel.Device, c diffCase) [][]float32 {
	t.Helper()
	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	if len(c.uniforms) > 0 {
		storage |= accel.BufferUniform
	}
	seed := c.seed
	if seed == nil {
		seed = defaultSeed
	}

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: c.kernel, Label: c.kernel.Name,
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	r := d.NewRecorder()
	bufs := make([]*accel.Buffer, len(c.counts))
	binds := make([]accel.Binding, 0, len(c.counts))
	uniforms := make([]accel.UniformValue, 0, len(c.uniforms))
	for i, u := range c.uniforms {
		uniforms = append(uniforms, accel.UniformValue{Index: i, Value: u})
	}
	for i, n := range c.counts {
		dt := dtypeOf(c.kernel.Bindings[i].DType)
		b, err := d.NewBuffer(accel.BufferDescriptor{
			DType: dt, Count: n, Usage: storage,
			Label: fmt.Sprintf("%s.%s", c.kernel.Name, c.kernel.Bindings[i].Name),
		})
		if err != nil {
			t.Fatalf("buffer %d: %v", i, err)
		}
		defer b.Close()
		bufs[i] = b

		v, err := b.View(0, n)
		if err != nil {
			t.Fatalf("view %d: %v", i, err)
		}
		writeSeed(t, r, v, dt, n, func(k int) float32 { return seed(i, k) })
		binds = append(binds, accel.Binding{Index: i, Buffer: v})
	}

	r.Dispatch(p, binds, uniforms, c.groups)
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	// Everything is read back, not only what the kernel declares it writes: an
	// access mode inferred wrongly would let a kernel write a binding recorded
	// as read-only, and reading only the outputs would never notice.
	out := make([][]float32, len(bufs))
	for i, b := range bufs {
		out[i] = readAsF32(t, d, b, c.kernel.Bindings[i].DType, c.counts[i])
	}
	return out
}

func dtypeOf(d kernelabi.DType) accel.DType {
	switch d {
	case kernelabi.F16:
		return accel.F16
	case kernelabi.U32:
		return accel.U32
	case kernelabi.I32:
		return accel.I32
	case kernelabi.I8:
		return accel.I8
	case kernelabi.U8:
		return accel.U8
	}
	return accel.F32
}

// writeSeed uploads a binding's initial contents through the graph.
func writeSeed(t *testing.T, r *accel.Recorder, v accel.BufferView, dt accel.DType, n int, at func(int) float32) {
	t.Helper()
	switch dt {
	case accel.F16:
		// The host slice for an f16 buffer is []uint16: the API boundary moves
		// bit patterns, and Float16 is a value type whose layout is not part of
		// the contract.
		vals := make([]uint16, n)
		for i := range vals {
			vals[i] = accel.ToFloat16(at(i)).Bits()
		}
		r.UploadToBuffer(v, vals)
	case accel.U32:
		vals := make([]uint32, n)
		for i := range vals {
			vals[i] = uint32(math.Abs(float64(at(i))))
		}
		r.UploadToBuffer(v, vals)
	case accel.I8:
		vals := make([]int8, n)
		for i := range vals {
			vals[i] = int8(at(i))
		}
		r.UploadToBuffer(v, vals)
	case accel.U8:
		vals := make([]uint8, n)
		for i := range vals {
			vals[i] = uint8(math.Abs(float64(at(i))))
		}
		r.UploadToBuffer(v, vals)
	case accel.I32:
		vals := make([]int32, n)
		for i := range vals {
			vals[i] = int32(at(i))
		}
		r.UploadToBuffer(v, vals)
	default:
		vals := make([]float32, n)
		for i := range vals {
			vals[i] = at(i)
		}
		r.UploadToBuffer(v, vals)
	}
}

// readAsF32 reads a binding back in one comparable representation.
//
// Integers are widened rather than compared as integers so one comparison
// covers every dtype. The widening is exact for u32 and i32 values this corpus
// produces, and a count that exceeded 2^24 would be a different bug.
func readAsF32(t *testing.T, d *accel.Device, b *accel.Buffer, dt kernelabi.DType, n int) []float32 {
	t.Helper()
	out := make([]float32, n)
	switch dt {
	case kernelabi.F16:
		raw := make([]uint16, n)
		if err := d.Queue().ReadBuffer(b, 0, raw); err != nil {
			t.Fatalf("readback: %v", err)
		}
		for i, v := range raw {
			out[i] = accel.Float16FromBits(v).F32()
		}
	case kernelabi.U32:
		raw := make([]uint32, n)
		if err := d.Queue().ReadBuffer(b, 0, raw); err != nil {
			t.Fatalf("readback: %v", err)
		}
		for i, v := range raw {
			out[i] = float32(v)
		}
	case kernelabi.I32:
		raw := make([]int32, n)
		if err := d.Queue().ReadBuffer(b, 0, raw); err != nil {
			t.Fatalf("readback: %v", err)
		}
		for i, v := range raw {
			out[i] = float32(v)
		}
	case kernelabi.I8:
		raw := make([]int8, n)
		if err := d.Queue().ReadBuffer(b, 0, raw); err != nil {
			t.Fatalf("readback: %v", err)
		}
		for i, v := range raw {
			out[i] = float32(v)
		}
	case kernelabi.U8:
		raw := make([]uint8, n)
		if err := d.Queue().ReadBuffer(b, 0, raw); err != nil {
			t.Fatalf("readback: %v", err)
		}
		for i, v := range raw {
			out[i] = float32(v)
		}
	default:
		if err := d.Queue().ReadBuffer(b, 0, out); err != nil {
			t.Fatalf("readback: %v", err)
		}
	}
	return out
}

// openMetalDevice opens the enumerated Metal adapter, failing or skipping on
// what the job promised. See .github/workflows/ci-metal.yml.
func openMetalDevice(t *testing.T) *accel.Device {
	t.Helper()
	e := accel.Enumerate()
	for _, info := range e.Devices {
		if info.Backend != accel.BackendMetal {
			continue
		}
		d, err := accel.OpenDevice(info.ID)
		if err != nil {
			t.Fatalf("OpenDevice(%s): %v", info.Name, err)
		}
		t.Cleanup(func() { _ = d.Close() })
		return d
	}
	if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
		t.Fatalf("this job promises Metal and enumerated no adapter; diagnostics: %v",
			e.Diagnostics)
	}
	t.Skipf("no Metal adapter on this machine; diagnostics: %v", e.Diagnostics)
	return nil
}

// The portable tiled GEMM matches an independently written higher-precision
// reference on Metal, at dimensions that are not multiples of any tile
// dimension.
//
// specs/022-msl-target.md names this separately from the corpus differential,
// and the separation is the point. The differential says Metal agrees with the
// CPU; this says Metal is *right*, against a reference written as a straight
// triple loop with no tiling and no shared memory. A reference sharing the
// kernel's structure would share its bugs, and two backends agreeing on a wrong
// answer is exactly what one IR makes possible.
//
// The budget is per output element -- that element's own K terms and its own sum
// of magnitudes -- which is what specs/008-numerics.md section 7 requires rather
// than one budget for the whole matrix.
func TestTheTiledGEMMMatchesItsReferenceOnMetal(t *testing.T) {
	d := openMetalDevice(t)

	for _, c := range []struct{ m, n, k int }{
		{8, 16, 16}, // exactly one tile
		{3, 5, 7},   // all three tails, none aligned
		{9, 19, 23}, // all three, each larger than one tile
		{1, 1, 40},  // a single output over several K steps
	} {
		t.Run(fmt.Sprintf("%dx%dx%d", c.m, c.n, c.k), func(t *testing.T) {
			a := make([]accel.Float16, c.m*c.k)
			b := make([]accel.Float16, c.k*c.n)
			for i := range a {
				a[i] = accel.ToFloat16(float32(math.Sin(float64(i))) * 2)
			}
			for i := range b {
				b[i] = accel.ToFloat16(float32(math.Cos(float64(i))) * 3)
			}

			out := runGEMM(t, d, c.m, c.n, c.k, a, b)
			for i := range out {
				row, col := i/c.n, i%c.n
				terms := make([]float32, c.k)
				for kk := range c.k {
					terms[kk] = a[row*c.k+kk].F32() * b[kk*c.n+col].F32()
				}
				if r := numeq.Sum(out[i], terms, c.k-1); !r.OK() {
					t.Fatalf("element (%d,%d) of %dx%dx%d: %v", row, col, c.m, c.n, c.k, r)
				}
			}
		})
	}
}

func runGEMM(t *testing.T, d *accel.Device, m, n, k int, a, b []accel.Float16) []float32 {
	t.Helper()
	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst | accel.BufferUniform

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.MatMulTiledKernel, Label: "gemm",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	f16 := func(label string, v []accel.Float16) accel.BufferView {
		buf, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F16, Count: len(v), Usage: storage, Label: label,
		})
		if err != nil {
			t.Fatalf("buffer %s: %v", label, err)
		}
		t.Cleanup(func() { _ = buf.Close() })
		view, err := buf.View(0, len(v))
		if err != nil {
			t.Fatalf("view %s: %v", label, err)
		}
		return view
	}

	av, bv := f16("a", a), f16("b", b)
	outBuf, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: m * n, Usage: storage, Label: "out",
	})
	if err != nil {
		t.Fatalf("buffer out: %v", err)
	}
	defer outBuf.Close()
	outView, err := outBuf.View(0, m*n)
	if err != nil {
		t.Fatalf("view out: %v", err)
	}

	bits := func(v []accel.Float16) []uint16 {
		raw := make([]uint16, len(v))
		for i := range v {
			raw[i] = v[i].Bits()
		}
		return raw
	}

	r := d.NewRecorder()
	r.UploadToBuffer(av, bits(a))
	r.UploadToBuffer(bv, bits(b))
	r.UploadToBuffer(outView, make([]float32, m*n))
	r.Dispatch(p, []accel.Binding{
		{Index: 0, Buffer: av},
		{Index: 1, Buffer: bv},
		{Index: 2, Buffer: outView},
	}, []accel.UniformValue{{Index: 0, Value: testkernels.GEMMDims{M: uint32(m), N: uint32(n), K: uint32(k)}}}, accel.WorkgroupCount{
		X: (n + testkernels.TileN - 1) / testkernels.TileN,
		Y: (m + testkernels.TileM - 1) / testkernels.TileM,
	})

	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	out := make([]float32, m*n)
	if err := d.Queue().ReadBuffer(outBuf, 0, out); err != nil {
		t.Fatalf("readback: %v", err)
	}
	return out
}

// Metal reports device time from the GPU's own clock, not from the wall.
//
// The distinction is the whole point of the feature. A wall-clock figure around
// Commit and Wait includes queueing, driver work and whatever else the process
// was doing; a caller measuring throughput needs the time the device spent.
// Metal gives both timestamps on a completed command buffer, so the
// whole-submission figure costs no timestamp pool.
//
// Asserted as a bound rather than a value — device time is not reproducible —
// and the bound is the one that catches a wall-clock substitute: the GPU cannot
// have spent longer on the work than the whole call took.
func TestMetalReportsDeviceTimeFromTheGPUClock(t *testing.T) {
	const n = 65536
	d := openMetalDevice(t)
	q := d.Queue()

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "timed",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	usage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	bufs := make([]accel.BufferView, 3)
	for i, name := range []string{"a", "b", "out"} {
		b, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: n, Usage: usage, Label: name,
		})
		if err != nil {
			t.Fatalf("buffer: %v", err)
		}
		defer b.Close()
		v, err := b.View(0, n)
		if err != nil {
			t.Fatalf("view: %v", err)
		}
		bufs[i] = v
	}

	r := d.NewRecorder()
	r.CollectTimings(true)
	for range 32 {
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: bufs[0]},
			{Index: 1, Buffer: bufs[1]},
			{Index: 2, Buffer: bufs[2]},
		}, nil, p.Workgroups(n))
	}
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	start := time.Now()
	f := q.Submit(g)
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	wall := time.Since(start)

	stats, err := f.Stats()
	if err != nil {
		t.Fatalf("stats: %v", err)
	}
	if stats.Elapsed <= 0 {
		t.Fatalf("Metal reported %v for 32 dispatches over %d elements", stats.Elapsed, n)
	}
	if stats.Elapsed > wall {
		t.Errorf("the device reported %v and the whole call took %v; device time cannot "+
			"exceed wall time, so this is a wall-clock figure taken somewhere wider",
			stats.Elapsed, wall)
	}
}

// A Metal graph that did not ask for timings reports none, and a fence read
// after its executable closes reports none rather than reaching a freed
// command buffer.
//
// The second half is the one worth having: the timing is read from the command
// buffer, which the executable owns, so a caller holding a fence past Close
// would otherwise send a message to a released object.
func TestMetalTimingsAreSilentWhenNotAskedFor(t *testing.T) {
	const n = 1024
	d := openMetalDevice(t)
	q := d.Queue()

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.AddKernel, Label: "untimed",
	})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()

	usage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	bufs := make([]accel.BufferView, 3)
	for i, name := range []string{"a", "b", "out"} {
		b, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.F32, Count: n, Usage: usage, Label: name,
		})
		if err != nil {
			t.Fatalf("buffer: %v", err)
		}
		defer b.Close()
		v, err := b.View(0, n)
		if err != nil {
			t.Fatalf("view: %v", err)
		}
		bufs[i] = v
	}

	r := d.NewRecorder()
	// No CollectTimings.
	r.Dispatch(p, []accel.Binding{
		{Index: 0, Buffer: bufs[0]}, {Index: 1, Buffer: bufs[1]},
		{Index: 2, Buffer: bufs[2]},
	}, nil, p.Workgroups(n))
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}

	f := q.Submit(g)
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	stats, err := f.Stats()
	if err != nil {
		t.Fatalf("stats: %v", err)
	}
	if stats.Elapsed != 0 {
		t.Errorf("a graph that did not ask for timings reported %v", stats.Elapsed)
	}

	// Closing the graph releases the command buffer the timing came from.
	// Reading afterwards must report nothing rather than message a freed
	// object.
	if err := g.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if _, err := f.Stats(); err != nil {
		t.Errorf("reading stats after the graph closed gave %v", err)
	}
}
