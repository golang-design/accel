// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// The rest of the v0 operator set.
//
// Each maps to a kernel specs/010-kernel-corpus.md registers, and each carries
// the grid that kernel expects: an elementwise kernel wants one invocation per
// element, a row kernel wants one workgroup per row, and a tiled GEMM wants
// tiles of the output. That mapping is the only thing this file knows that the
// operator's contract does not, which is why the grid sits beside the kernel
// rather than in a table somewhere else.

// grid is how a node's workgroup count is computed from its result.
type grid func(out *Tensor) accel.WorkgroupCount

// perElement covers one output element per invocation.
func perElement(wg int) grid {
	return func(out *Tensor) accel.WorkgroupCount {
		n := out.shape.Elements()
		return accel.WorkgroupCount{X: (n + wg - 1) / wg}
	}
}

// perRow gives each row of the result its own workgroup, which is what a
// reduction across a row needs: the invocations in it share the partial sums
// through workgroup memory, and a row split across two workgroups could not.
func perRow(out *Tensor) accel.WorkgroupCount {
	rows := out.shape.Elements() / out.shape[len(out.shape)-1]
	return accel.WorkgroupCount{X: rows}
}

// Rows gathers whole rows of a table by index.
//
// This is an embedding lookup, and the out-of-range rule is the interesting
// part: specs/007-tensor-layer.md makes an out-of-range id a caller error, and
// the kernel writes zeros rather than reading outside the table. Zeros are a
// plausible embedding, so this is the one place where a diagnostic would be
// worth more than a safe answer -- and where the corpus kernel gives the safe
// answer because a GPU has no other option.
func GatherRows(b *Builder, table, ids *Tensor) *Tensor {
	if poisoned(table, ids) {
		return b.poison()
	}
	if table.dtype != accel.F32 {
		return b.fail(1, "GatherRows", "the table is %v and the registered kernel reads f32; "+
			"specs/010-kernel-corpus.md owns the f16 variant", table.dtype)
	}
	if ids.dtype != accel.U32 {
		return b.fail(1, "GatherRows", "ids are %v and must be u32", ids.dtype)
	}
	if len(table.shape) != 2 {
		return b.fail(1, "GatherRows", "the table is %v and must be [vocab, width]", table.shape)
	}
	capacity, width := table.shape[0], table.shape[1]
	rows := ids.shape.Elements()

	out := make(Shape, 0, len(ids.shape)+1)
	out = append(out, ids.shape...)
	out = append(out, width)

	return b.record(node{
		op: "GatherRows", inputs: []*Tensor{table, ids}, kernel: &testkernels.GatherRowsKernel,
		uniform: func(map[string]ScalarValue) any {
			return testkernels.RowParams{
				Rows: uint32(rows), Width: uint32(width), Capacity: uint32(capacity),
			}
		},
		grid: perElement(int(testkernels.GatherRowsKernel.WorkgroupSize.X)),
		reason: "the gather variant; an id at or above the table's capacity writes zeros, " +
			"because a GPU cannot report one",
	}, table.dtype, out)
}

// RMSNorm normalizes each row by its root mean square and scales by a gain.
//
// The mean square and the reciprocal square root are computed in f32 whatever
// the storage, which specs/007-tensor-layer.md requires: a normalization
// accumulated in f16 loses the thing it is measuring.
func RMSNorm(b *Builder, x, gain *Tensor, eps float32) *Tensor {
	if poisoned(x, gain) {
		return b.poison()
	}
	if x.dtype != accel.F32 || gain.dtype != accel.F32 {
		return b.fail(1, "RMSNorm", "x is %v and gain is %v; the registered kernel reads "+
			"f32, and specs/010-kernel-corpus.md owns the f16 variant", x.dtype, gain.dtype)
	}
	if len(x.shape) < 1 {
		return b.fail(1, "RMSNorm", "x has no shape")
	}
	width := x.shape[len(x.shape)-1]
	if !gain.shape.Equal(Shape{width}) {
		return b.fail(1, "RMSNorm", "gain is %v and x's last axis is %d; the gain is one "+
			"value per feature", gain.shape, width)
	}
	if !(eps > 0) {
		return b.fail(1, "RMSNorm", "eps is %v and must be positive and finite: it is what "+
			"keeps a row of zeros from dividing by zero", eps)
	}
	rows := x.shape.Elements() / width

	return b.record(node{
		op: "RMSNorm", inputs: []*Tensor{x, gain}, kernel: &testkernels.RMSNormKernel,
		uniform: func(map[string]ScalarValue) any {
			return testkernels.RowDims{Rows: uint32(rows), Width: uint32(width), Eps: eps}
		},
		grid:   perRow,
		reason: "the cooperative row reduction, one workgroup per row",
	}, x.dtype, x.shape)
}

// SoftmaxOptions carries the attributes softmax takes.
type SoftmaxOptions struct {
	// Axis is normalized; it must be the last, which is the only axis the
	// registered kernel reduces over.
	Axis int
}

// Softmax normalizes each row into a distribution.
//
// The maximum subtraction, the exponentiation, the sum and the division all
// happen in f32. Subtracting the maximum first is not an optimization: without
// it, exp overflows for any row whose largest value exceeds about 88.
func Softmax(b *Builder, x *Tensor, opts SoftmaxOptions) *Tensor {
	if poisoned(x) {
		return b.poison()
	}
	if x.dtype != accel.F32 {
		return b.fail(1, "Softmax", "x is %v and the registered kernel reads f32", x.dtype)
	}
	if len(x.shape) < 1 {
		return b.fail(1, "Softmax", "x has no shape")
	}
	axis, ok := normalizeAxis(opts.Axis, len(x.shape))
	if !ok {
		return b.fail(1, "Softmax", "axis %d is outside a rank-%d tensor",
			opts.Axis, len(x.shape))
	}
	if axis != len(x.shape)-1 {
		return b.fail(1, "Softmax", "axis %d is not the last; the registered kernel reduces "+
			"over the innermost axis, and another one needs a transpose", axis)
	}
	width := x.shape[len(x.shape)-1]
	rows := x.shape.Elements() / width

	return b.record(node{
		op: "Softmax", inputs: []*Tensor{x}, kernel: &testkernels.SoftmaxKernel,
		uniform: func(map[string]ScalarValue) any {
			return testkernels.RowDims{Rows: uint32(rows), Width: uint32(width)}
		},
		grid:     perRow,
		reason:   "the cooperative row reduction, one workgroup per row",
		rejected: []string{"a masked or causal variant: specs/010-kernel-corpus.md has none"},
	}, x.dtype, x.shape)
}

// RoPE applies rotary position embedding in place on a copy.
//
// In place on a *copy*, because the registered kernel rotates its buffer and a
// tensor is an immutable value. So this lowers to a copy into a transient
// followed by a dispatch over it, and the copy is reported: it is real work and
// hiding it would make a rotation look free.
//
// # Why positions are a tensor and the base is a scalar
//
// specs/043-per-row-values.md draws the line: a value every row of a dispatch
// shares is a uniform, and a value that differs per row is device data. The
// frequency base is a property of the model; the position is a property of the
// sequence.
//
// This took a scalar offset and computed `row + offset`, which is exactly right
// for one sequence and exactly wrong for two. In a batched decode the row index
// is the *slot*, so slot 0 rotates at the offset and slot 1 at offset+1 —
// meaning one member of the batch is ever rotated at its own cache length. The
// output stays finite and fluent; long-range coherence degrades in a way that
// reads as "the model is a bit weak" rather than as a bug.
//
// A single sequence binds a one-row positions tensor. That is the same path, not
// a special case.
func RoPE(b *Builder, x *Tensor, rotaryDim int, baseName string, positions *Tensor) *Tensor {
	if poisoned(x, positions) {
		return b.poison()
	}
	if x.dtype != accel.F32 {
		return b.fail(1, "RoPE", "x is %v and the registered kernel reads f32", x.dtype)
	}
	if len(x.shape) < 1 {
		return b.fail(1, "RoPE", "x has no shape")
	}
	width := x.shape[len(x.shape)-1]
	if rotaryDim <= 0 || rotaryDim%2 != 0 || rotaryDim > width {
		return b.fail(1, "RoPE", "rotaryDim is %d against a width of %d; it rotates pairs, "+
			"so it is positive, even, and no wider than the row", rotaryDim, width)
	}
	base, ok := b.scalarKind(baseName)
	if !ok || base != ScalarF32 {
		return b.fail(1, "RoPE", "%q is not a declared f32 scalar; the frequency base varies "+
			"per step and is named for that reason", baseName)
	}
	rows := x.shape.Elements() / width
	if positions.dtype != accel.U32 {
		return b.fail(1, "RoPE", "positions are %v and the kernel reads u32", positions.dtype)
	}
	if got := positions.shape.Elements(); got != rows {
		return b.fail(1, "RoPE", "x has %d rows and positions holds %d; every row rotates "+
			"at its own position, so there is exactly one per row "+
			"(specs/043-per-row-values.md)", rows, got)
	}

	return b.record(node{
		// Positions first, then the buffer rewritten: operand order is the
		// kernel's binding order, and the in-place operand is the last.
		op: "RoPE", inputs: []*Tensor{positions, x}, kernel: &testkernels.RoPEKernel,
		inPlace: true,
		reads:   []string{baseName},
		uniform: func(vals map[string]ScalarValue) any {
			return testkernels.RoPEParams{
				Rows: uint32(rows), Width: uint32(width), RotaryDim: uint32(rotaryDim),
				Base: vals[baseName].F32,
			}
		},
		// One invocation per rotated *pair*, not per element: the kernel bounds
		// itself by Rows*RotaryDim/2 and a grid sized per element would launch
		// invocations with nothing to do.
		grid: func(out *Tensor) accel.WorkgroupCount {
			wg := int(testkernels.RoPEKernel.WorkgroupSize.X)
			pairs := rows * rotaryDim / 2
			return accel.WorkgroupCount{X: (pairs + wg - 1) / wg}
		},
		reason: "the in-place rotation, preceded by a copy because a tensor is a value",
	}, x.dtype, x.shape)
}

// Cast converts between storage formats.
//
// An operator rather than an implicit rule at every boundary, and
// specs/007-tensor-layer.md's reason is the one that matters: a conversion
// costs a pass over the data and changes the numbers, so it is something a
// caller writes rather than something that happens to them. `Add` refusing two
// dtypes and `Cast` existing are the same decision seen from two sides.
//
// f16 to f32 is exact; f32 to f16 rounds to nearest-even and a value outside
// f16's range becomes an infinity rather than a saturated maximum, because a
// silently clamped weight is a plausible weight.
func Cast(b *Builder, x *Tensor, to DType) *Tensor {
	if poisoned(x) {
		return b.poison()
	}
	if x.dtype == to {
		// Not an error and not a dispatch: a conversion to the format a value
		// already has is the identity, and making it a copy would charge a pass
		// over the data for nothing.
		return x
	}
	var k *accel.Kernel
	switch {
	case x.dtype == accel.F32 && to == accel.F16:
		k = &testkernels.CastF32ToF16Kernel
	case x.dtype == accel.F16 && to == accel.F32:
		k = &testkernels.CastF16ToF32Kernel
	default:
		return b.fail(1, "Cast", "%v to %v; specs/010-kernel-corpus.md registers f32 to f16 "+
			"and f16 to f32, and a conversion it does not register is a kernel rather than "+
			"something this layer can compose", x.dtype, to)
	}
	reason := "the exact widening: every f16 value is an f32 value"
	if to == accel.F16 {
		reason = "the narrowing, rounding to nearest even; a value outside f16's range " +
			"becomes an infinity rather than a saturated maximum"
	}
	return b.record(node{
		op: "Cast", inputs: []*Tensor{x}, kernel: k, bcast: true, reason: reason,
	}, to, x.shape)
}
