// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"fmt"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
	"golang.design/x/accel/quant"
)

// Quantized is a weight stored as quants and scales.
//
// Two tensors rather than one, because that is what the device holds:
// specs/001-device-resources.md types a buffer by dtype and an interleaved
// block struct has no dtype, so a quantized matrix is two planes. Bundling them
// in one value keeps a caller from binding a matrix's quants against another
// matrix's scales, which would compile, run, and produce a matrix of noise.
type Quantized struct {
	// Quants is i8, one per weight, shaped like the matrix.
	Quants *Tensor

	// Scales is f16, one per [quant.Int8Block] weights of the flattened
	// matrix.
	Scales *Tensor
}

// checkQuantized reports why a quantized pair is not usable, or "".
func checkQuantized(q Quantized, what string) string {
	switch {
	case q.Quants == nil || q.Scales == nil:
		return what + " needs both a quant plane and a scale plane"
	case q.Quants.dtype != accel.I8:
		return fmt.Sprintf("%s quants are %v and must be i8", what, q.Quants.dtype)
	case q.Scales.dtype != accel.F16:
		return fmt.Sprintf("%s scales are %v and must be f16", what, q.Scales.dtype)
	}
	// The scale count follows from the quant count, so a mismatch means the two
	// planes describe different matrices -- which is exactly what bundling them
	// is meant to prevent, caught here for the case where a caller built the
	// pair by hand.
	n := q.Quants.shape.Elements()
	want := (n + quant.Int8Block - 1) / quant.Int8Block
	if got := q.Scales.shape.Elements(); got != want {
		return fmt.Sprintf("%s has %d quants, which is %d blocks of %d, and %d scales",
			what, n, want, quant.Int8Block, got)
	}
	return ""
}

// QuantMatMul multiplies f16 or f32 activations by a quantized weight matrix.
//
// A separate operator rather than [MatMul] taking a different argument type,
// because the cost is different and a caller should see which one they wrote.
// specs/027-quantization.md states the error budget, and it is not the
// unquantized one: the result is within a derived distance of what the
// unquantized product would have given, and that distance is proportional to
// the largest weight in each block.
//
// # Why the activation may be either width
//
// Because the weight's width is not the activation's business. int8 is what a
// model reaches for when it is too large to hold otherwise, and every other
// operator in this package produces f32 -- so requiring the activation to be
// f16 put a Cast in front of every projection of the configuration least able
// to afford one (accel issue 14). The quants and scales keep their widths in
// both cases: a weight is loaded from a file and an activation is produced by
// the graph, and only the second is free to be wide.
func QuantMatMul(b *Builder, x *Tensor, w Quantized) *Tensor {
	// A nil plane is checked before the poison test, because poison propagation
	// exists to suppress the *echoes* of an error and a missing argument is not
	// an echo -- it is the error. Letting it through as poison would give the
	// caller silence: no diagnostic, and a graph that fails to compile for a
	// reason nobody stated.
	if w.Quants == nil || w.Scales == nil {
		return b.fail(1, "QuantMatMul", "the weight needs both a quant plane and a scale plane")
	}
	if poisoned(x, w.Quants, w.Scales) {
		return b.poison()
	}
	if why := checkQuantized(w, "the weight"); why != "" {
		return b.fail(1, "QuantMatMul", "%s", why)
	}
	if x.dtype != accel.F16 && x.dtype != accel.F32 {
		return b.fail(1, "QuantMatMul", "activations are %v; specs/010-kernel-corpus.md "+
			"registers f16 and f32 activations against int8 weights", x.dtype)
	}
	if len(x.shape) != 2 || len(w.Quants.shape) != 2 {
		return b.fail(1, "QuantMatMul", "operands are %v and %v; v0 multiplies two matrices",
			x.shape, w.Quants.shape)
	}
	if x.shape[1] != w.Quants.shape[0] {
		return b.fail(1, "QuantMatMul", "%v times %v: the contracted axes are %d and %d",
			x.shape, w.Quants.shape, x.shape[1], w.Quants.shape[0])
	}
	m, k, n := x.shape[0], x.shape[1], w.Quants.shape[1]

	dims := kernels.GEMMDims{M: uint32(m), N: uint32(n), K: uint32(k)}
	integer := "an integer-accumulating variant: it needs one scale per output " +
		"column, and this representation has one per block"

	// The activation's width picks the variant within each shape, and it has to
	// be read here rather than assumed: the two kernels of a pair differ only in
	// binding 0's element type, so an f32 activation sent to the f16 kernel is a
	// buffer of the wrong element size read as the right one.
	wide := x.dtype == accel.F32
	acts := "f16"
	if wide {
		acts = "f32"
	}

	// The same M=1 selection MatMul makes, and it belongs here more than there.
	// A decode step is M=1 and int8 is the width a model reaches for because it
	// is large, so the quantized path is both where this shape is most common
	// and where a missing specialization costs most -- a consumer reported the
	// asymmetry rather than a measurement (accel issue 11). Both widths of it
	// exist for that reason: closing issue 14's refusal on the general kernel
	// alone would leave decode running the unspecialized shape again.
	if m == 1 {
		vec := &kernels.QuantMatVecKernel
		if wide {
			vec = &kernels.QuantMatVecF32Kernel
		}
		return b.record(node{
			op: "QuantMatMul", inputs: []*Tensor{x, w.Quants, w.Scales},
			kernel:  vec,
			uniform: func(map[string]ScalarValue) any { return dims },
			grid: func(*Tensor) accel.WorkgroupCount {
				return accel.WorkgroupCount{X: n}
			},
			reason: fmt.Sprintf("M is 1, so the quantized matrix-vector kernel over %s "+
				"activations gives each of the %d output columns a workgroup and folds "+
				"K across the lanes", acts, n),
			rejected: []string{integer, "the per-element quantized GEMM: it gives each " +
				"output element one invocation, which at M=1 leaves K unparallelized"},
		}, accel.F32, Shape{m, n})
	}

	gemm := &kernels.QuantMatMulKernel
	if wide {
		gemm = &kernels.QuantMatMulF32Kernel
	}
	return b.record(node{
		op: "QuantMatMul", inputs: []*Tensor{x, w.Quants, w.Scales},
		kernel:  gemm,
		uniform: func(map[string]ScalarValue) any { return dims },
		grid:    perElement(int(gemm.WorkgroupSize.X)),
		reason: fmt.Sprintf("the int8 quantized GEMM over %s activations and a %dx%d "+
			"output; each product widens to f32 before accumulating, because the scale "+
			"varies per block", acts, m, n),
		rejected: []string{integer, "the quantized matrix-vector kernel: it applies only at M=1"},
	}, accel.F32, Shape{m, n})
}

// QuantRows gathers rows of a quantized table.
//
// An embedding table is the largest single tensor in a small model and the one
// quantization helps most: every token reads one row of it and nothing else, so
// the whole table sits in memory to serve one row at a time.
func QuantGatherRows(b *Builder, table Quantized, ids *Tensor) *Tensor {
	if table.Quants == nil || table.Scales == nil {
		return b.fail(1, "QuantGatherRows", "the table needs both a quant plane and a scale plane")
	}
	if poisoned(table.Quants, table.Scales, ids) {
		return b.poison()
	}
	if why := checkQuantized(table, "the table"); why != "" {
		return b.fail(1, "QuantGatherRows", "%s", why)
	}
	if ids.dtype != accel.U32 {
		return b.fail(1, "QuantGatherRows", "ids are %v and must be u32", ids.dtype)
	}
	if len(table.Quants.shape) != 2 {
		return b.fail(1, "QuantGatherRows", "the table is %v and must be [vocab, width]",
			table.Quants.shape)
	}
	capacity, width := table.Quants.shape[0], table.Quants.shape[1]
	rows := ids.shape.Elements()

	out := make(Shape, 0, len(ids.shape)+1)
	out = append(out, ids.shape...)
	out = append(out, width)

	return b.record(node{
		op: "QuantGatherRows", inputs: []*Tensor{table.Quants, table.Scales, ids},
		kernel: &kernels.QuantRowsKernel,
		uniform: func(map[string]ScalarValue) any {
			return kernels.RowParams{
				Rows: uint32(rows), Width: uint32(width), Capacity: uint32(capacity),
			}
		},
		grid: perElement(int(kernels.QuantRowsKernel.WorkgroupSize.X)),
		reason: "the quantized gather; an id at or above the table's capacity writes " +
			"zeros, because a GPU cannot report one",
	}, accel.F32, out)
}
