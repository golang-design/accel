// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"fmt"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
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
// # Why the operands are f16
//
// specs/010-kernel-corpus.md registers a tiled GEMM that reads f16 and
// accumulates in f32, which is what a transformer's weights are. 007 admits f16
// *or* f32 storage; the f32 GEMM is a corpus kernel that does not exist, and
// adding one belongs to 010 rather than to an improvisation here. The refusal
// says which spec owns it.

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
	if x.dtype != accel.F16 || w.dtype != accel.F16 {
		b.fail(2, op, "operands are %v and %v; the registered GEMM reads f16 and "+
			"accumulates in f32, and specs/010-kernel-corpus.md owns an f32 variant",
			x.dtype, w.dtype)
		return 0, 0, 0, false
	}
	return x.shape[0], w.shape[1], x.shape[1], true
}

// gemmGrid covers the output in tiles, which is what the tiled kernel expects:
// one workgroup per output tile rather than per element.
func gemmGrid(m, n int) grid {
	return func(*Tensor) accel.WorkgroupCount {
		return accel.WorkgroupCount{
			X: (n + testkernels.TileN - 1) / testkernels.TileN,
			Y: (m + testkernels.TileM - 1) / testkernels.TileM,
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
	dims := testkernels.GEMMDims{M: uint32(m), N: uint32(n), K: uint32(k)}

	// The one selection v0 makes. A matrix-vector product has one output row,
	// so a tile eight rows tall would leave seven idle; the M=1 kernel gives
	// each output column a workgroup instead.
	if m == 1 {
		return b.record(node{
			op: "MatMul", inputs: []*Tensor{x, w}, kernel: &testkernels.MatVecKernel,
			uniform: func(map[string]ScalarValue) any { return dims },
			grid: func(*Tensor) accel.WorkgroupCount {
				return accel.WorkgroupCount{X: n}
			},
			reason: fmt.Sprintf("M is 1, so the matrix-vector kernel gives each of the %d "+
				"output columns a workgroup", n),
			rejected: []string{fmt.Sprintf("the tiled GEMM: its %d-row tile would leave %d "+
				"rows idle", testkernels.TileM, testkernels.TileM-1)},
		}, accel.F32, Shape{m, n})
	}
	return b.record(node{
		op: "MatMul", inputs: []*Tensor{x, w}, kernel: &testkernels.MatMulTiledKernel,
		uniform: func(map[string]ScalarValue) any { return dims },
		grid:    gemmGrid(m, n),
		reason: fmt.Sprintf("the portable tiled GEMM over a %dx%d output, in %dx%d tiles",
			m, n, testkernels.TileM, testkernels.TileN),
		rejected: []string{"the matrix-vector kernel: it applies only at M=1"},
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
	if bias.dtype != accel.F32 {
		return b.fail(1, "Linear", "the bias is %v and the epilogue adds in f32", bias.dtype)
	}
	if !bias.shape.Equal(Shape{n}) {
		return b.fail(1, "Linear", "the bias is %v and the output is %d wide; a bias is one "+
			"value per output column", bias.shape, n)
	}
	dims := testkernels.GEMMDims{M: uint32(m), N: uint32(n), K: uint32(k)}
	return b.record(node{
		op: "Linear", inputs: []*Tensor{x, w, bias}, kernel: &testkernels.LinearTiledKernel,
		uniform:  func(map[string]ScalarValue) any { return dims },
		grid:     gemmGrid(m, n),
		reason:   "the authored epilogue, which adds the bias while the tile is still in registers",
		rejected: []string{"MatMul followed by Add: correct, and writes the whole result twice"},
	}, accel.F32, Shape{m, n})
}
