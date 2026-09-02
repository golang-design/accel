// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"fmt"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
)

// Matrix multiplication, and the one selection v0 actually makes.
//
// specs/007-tensor-layer.md is explicit that `MatVec` is "the selected M=1
// implementation, not a distinct public semantic operation". A caller writes
// MatMul; which kernel they got is reported by [Plan.Selections]. That is the
// shape every later selection takes -- fused attention against the composed
// graph, a quantized variant against an unquantized one -- so it is worth
// getting right on the one case v0 has.
//
// # Which width pairs multiply
//
// Three, and specs/010-kernel-corpus.md registers a GEMM for each: f16 times
// f16, f32 times f32, and f32 activations times f16 weights. The third is the
// pair a transformer actually has. Its activations are f32 because every other
// operator in this package is -- RMSNorm, Softmax, RoPE, Attention, the
// residual Add -- and its weights are f16 because a four billion parameter
// model is 16 GB in f32, so f32 weights are not a precision choice but the
// choice not to load the model. A rule that the two agree therefore made the
// graph pay for the weight's memory decision, in four casts per layer (accel
// issue 14).
//
// The rule stays exactly right for two operands that are both activations,
// which is why the relaxation is asymmetric rather than "any two widths". An
// activation is produced by the graph and a weight is loaded from a file, and
// `MatMul(b, x, w)` already says which is which by position: the *first*
// operand may be the wider one. A weight wider than the activation it
// multiplies is the memory decision made in the expensive direction, and stays
// refused.

// matShape checks a matrix multiplication's operands and returns M, N, K.
func matShape(b *Builder, op string, x, w *Tensor) (m, n, k int, ok bool) {
	if len(x.shape) != 2 || len(w.shape) != 2 {
		b.fail(2, op, "operands are %v and %v; v0 multiplies two matrices, and a batched "+
			"form needs the leading axes broadcast, which specs/025-tensor-operators.md "+
			"does not do", x.shape, w.shape)
		return 0, 0, 0, false
	}
	if x.shape[1] != w.shape[0] {
		b.fail(2, op, "%v times %v: the contracted axes are %d and %d and must agree",
			x.shape, w.shape, x.shape[1], w.shape[0])
		return 0, 0, 0, false
	}
	if !gemmPair(x.dtype, w.dtype) {
		if x.dtype == accel.F16 && w.dtype == accel.F32 {
			b.fail(2, op, "the activations are %v and the weight is %v; a weight wider "+
				"than the activation it multiplies has no registered GEMM, because that "+
				"is the memory decision made in the expensive direction -- the mixed "+
				"kernel widens the activation, not the weight", x.dtype, w.dtype)
			return 0, 0, 0, false
		}
		b.fail(2, op, "operands are %v and %v; specs/010-kernel-corpus.md registers f16 "+
			"times f16, f32 times f32, and f32 activations times f16 weights, all "+
			"accumulating in f32", x.dtype, w.dtype)
		return 0, 0, 0, false
	}
	return x.shape[0], w.shape[1], x.shape[1], true
}

// gemmPair reports whether a registered GEMM reads this pair of widths.
//
// A table rather than a pair of independent checks, because the pairs are what
// the corpus registers: a rule reading only x's width would admit an f16
// activation against an f32 weight, and a selection reading only x's width
// would then bind an f16 buffer to an f32 binding and produce a plausible
// matrix. [gemmKernel] switches on the same pair for that reason.
func gemmPair(x, w DType) bool {
	switch {
	case x == accel.F16 && w == accel.F16:
		return true
	case x == accel.F32 && w == accel.F32:
		return true
	case x == accel.F32 && w == accel.F16:
		return true
	}
	return false
}

// gemmKernel picks the GEMM the operand *pair* registers, and says why.
//
// The pair and not x's width alone. Binding an f16 weight to the f32 kernel's
// f32 binding is not a compile error at any layer below this one -- it is a
// buffer of the wrong element size read as the right one, which produces a
// matrix rather than a diagnostic.
func gemmKernel(x, w DType, m, n int) (*accel.Kernel, string) {
	tiles := fmt.Sprintf("a %dx%d output in %dx%d tiles", m, n,
		kernels.TileM, kernels.TileN)
	switch {
	case x == accel.F32 && w == accel.F16:
		return &kernels.MatMulTiledF32F16Kernel,
			"the mixed tiled GEMM, f32 activations against f16 weights, over " + tiles
	case x == accel.F32:
		return &kernels.MatMulTiledF32Kernel,
			"the tiled GEMM over f32 operands, " + tiles
	}
	return &kernels.MatMulTiledKernel,
		"the portable tiled GEMM over " + tiles
}

// gemmGrid covers the output in tiles, which is what the tiled kernel expects:
// one workgroup per output tile rather than per element.
func gemmGrid(m, n int) grid {
	return func(*Tensor) accel.WorkgroupCount {
		return accel.WorkgroupCount{
			X: (n + kernels.TileN - 1) / kernels.TileN,
			Y: (m + kernels.TileM - 1) / kernels.TileM,
		}
	}
}

// MatMul multiplies two matrices, accumulating in f32.
//
// The result is f32 whatever the operands are, because that is what
// accumulating in f32 means: rounding the sum back to f16 at the end would
// throw away the precision the accumulation was for, and a caller who wants it
// narrow can convert.
func MatMul(b *Builder, x, w *Tensor) *Tensor {
	if poisoned(x, w) {
		return b.poison()
	}
	m, n, k, ok := matShape(b, "MatMul", x, w)
	if !ok {
		return b.poison()
	}
	dims := kernels.GEMMDims{M: uint32(m), N: uint32(n), K: uint32(k)}

	// The one selection v0 makes. A matrix-vector product has one output row,
	// so a tile eight rows tall would leave seven idle; the M=1 kernel gives
	// each output column a workgroup instead.
	//
	// Only for the f16 pair, because the matrix-vector kernel reads f16 on
	// both operands and a second variant of it would be a kernel added for a
	// shape the tiled one already computes correctly. The f32 and mixed pairs
	// take the tile and Selections says the rows are idle, which is the cost
	// reported rather than hidden. That is a real gap for decode against f16
	// weights, and it is reported rather than closed here: the quantized path
	// has its own M=1 kernel because int8 is the width a large model is in,
	// and a mixed matvec is the same shape of argument at f16.
	if m == 1 && x.dtype == accel.F16 {
		return b.record(node{
			op: "MatMul", inputs: []*Tensor{x, w}, kernel: &kernels.MatVecKernel,
			uniform: func(map[string]ScalarValue) any { return dims },
			grid: func(*Tensor) accel.WorkgroupCount {
				return accel.WorkgroupCount{X: n}
			},
			reason: fmt.Sprintf("M is 1, so the matrix-vector kernel gives each of the %d "+
				"output columns a workgroup", n),
			rejected: []string{fmt.Sprintf("the tiled GEMM: its %d-row tile would leave %d "+
				"rows idle", kernels.TileM, kernels.TileM-1)},
		}, accel.F32, Shape{m, n})
	}
	gemm, why := gemmKernel(x.dtype, w.dtype, m, n)
	rejected := []string{"the matrix-vector kernel: it applies only at M=1"}
	if m == 1 {
		// Reached only when x is f32: the f16 pair took the MatVec branch
		// above. The matrix-vector kernel reads f16 on *both* operands, so
		// neither the f32 nor the mixed pair can use it, and the cost of the
		// tile is reported rather than hidden.
		rejected = []string{fmt.Sprintf("the matrix-vector kernel: it reads f16 on both "+
			"operands, so %d of this tile's %d rows are idle",
			kernels.TileM-1, kernels.TileM)}
	}
	return b.record(node{
		op: "MatMul", inputs: []*Tensor{x, w}, kernel: gemm,
		uniform:  func(map[string]ScalarValue) any { return dims },
		grid:     gemmGrid(m, n),
		reason:   why,
		rejected: rejected,
	}, accel.F32, Shape{m, n})
}

// Linear multiplies and adds a bias in one kernel.
//
// An authored epilogue rather than MatMul followed by Add, which is a selection
// specs/010-kernel-corpus.md registers: the composed form is correct and writes
// the whole result twice. The composed form remains the reference.
func Linear(b *Builder, x, w, bias *Tensor) *Tensor {
	if poisoned(x, w, bias) {
		return b.poison()
	}
	m, n, k, ok := matShape(b, "Linear", x, w)
	if !ok {
		return b.poison()
	}
	// x alone decides, and the message still names both. Of the three pairs
	// [gemmPair] admits, the only one whose activation is f16 is (f16, f16), so
	// an f16 x means an f16 w and testing w as well would be a branch nothing
	// can reach. What changed is the diagnostic: MatMul now admits f32
	// activations against f16 weights and the fused epilogue has no such
	// variant, so a caller can arrive here with a pair that is half right, and
	// a message naming only x would point at the operand that is fine.
	if x.dtype != accel.F16 {
		return b.fail(1, "Linear", "operands are %v and %v and the fused epilogue reads "+
			"f16 on both; the composed form -- MatMul then Add -- takes every pair "+
			"MatMul does and is the reference this kernel is checked against, so it is "+
			"the right answer here rather than a second fused variant", x.dtype, w.dtype)
	}
	if bias.dtype != accel.F32 {
		return b.fail(1, "Linear", "the bias is %v and the epilogue adds in f32", bias.dtype)
	}
	if !bias.shape.Equal(Shape{n}) {
		return b.fail(1, "Linear", "the bias is %v and the output is %d wide; a bias is one "+
			"value per output column", bias.shape, n)
	}
	dims := kernels.GEMMDims{M: uint32(m), N: uint32(n), K: uint32(k)}
	return b.record(node{
		op: "Linear", inputs: []*Tensor{x, w, bias}, kernel: &kernels.LinearTiledKernel,
		uniform:  func(map[string]ScalarValue) any { return dims },
		grid:     gemmGrid(m, n),
		reason:   "the authored epilogue, which adds the bias while the tile is still in registers",
		rejected: []string{"MatMul followed by Add: correct, and writes the whole result twice"},
	}, accel.F32, Shape{m, n})
}
