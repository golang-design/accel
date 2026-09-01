// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// The elementwise family.
//
// # Why these five first
//
// specs/024-tensor-bringup.md picks them as the smallest set that exercises
// every part of the slice: two operands and broadcasting (Add, Mul), a named
// runtime scalar (Scale), a unary with a bounded primitive (SiLU), and a fused
// two-operand form whose semantic definition is a composition (SwiGLU). A larger
// set would prove nothing more about the machinery and a smaller one would leave
// a piece of it unexercised.
//
// # Where the kernels come from
//
// specs/010-kernel-corpus.md's corpus, unchanged. The tensor layer selects
// among kernels; it does not author them, and it does not reach past the device
// layer to run them. Every operator here becomes an ordinary Recorder dispatch.

// elementwiseOperand refuses an operand the lowering cannot bring to the
// result's shape, at the caller's line. It returns nil when the operand is
// fine and the poisoned result otherwise.
//
// The elementwise kernels index their operands together, so an operand that is
// not already the result's shape and layout is materialized into a transient
// first (materialize.go), and the only layout that materializes with copies
// alone is a contiguous run repeated a whole number of times. Anything else --
// an interior axis expanding, a permuted operand -- was refused at Compile,
// where a diagnostic has no source position and the line that wrote the
// broadcast is long returned. This is the lowering's own check, run when the
// operator is recorded so the refusal names the line.
func (b *Builder) elementwiseOperand(skip int, op string, i int, t *Tensor, want Shape) *Tensor {
	if _, ok := broadcastRepeats(t, want); ok {
		return nil
	}
	return b.fail(skip+1, op, "operand %d is %v with strides %v at offset %d against a "+
		"result of %v, which is not a contiguous run repeated a whole number of times -- "+
		"the only broadcast an elementwise operand is materialized into: a vector across "+
		"rows, a row across a batch. Insert Contiguous to pack it",
		i, t.shape, t.strides, t.offset, want)
}

// binary is the shared body of Add and Mul.
func binary(b *Builder, op string, k *accel.Kernel, x, y *Tensor, skip int) *Tensor {
	if poisoned(x, y) {
		return b.poison()
	}
	if x.dtype != y.dtype {
		return b.fail(skip+1, op, "operands are %v and %v; elementwise operators need one "+
			"dtype, and a conversion is an operator rather than an implicit rule",
			x.dtype, y.dtype)
	}
	if !elementwiseDType(x.dtype) {
		return b.fail(skip+1, op, "%v is not an elementwise dtype; v0 computes in f32 over "+
			"f32 or f16 storage", x.dtype)
	}
	shape, ok := broadcastShape(x.shape, y.shape)
	if !ok {
		return b.fail(skip+1, op, "shapes %v and %v do not broadcast: axes are matched from "+
			"the right, and only a size-one axis expands", x.shape, y.shape)
	}
	if bad := b.elementwiseOperand(skip+1, op, 0, x, shape); bad != nil {
		return bad
	}
	if bad := b.elementwiseOperand(skip+1, op, 1, y, shape); bad != nil {
		return bad
	}
	return b.record(node{
		op: op, inputs: []*Tensor{x, y}, kernel: k, bcast: true,
		reason: "the contiguous elementwise variant, which is the only one registered for " +
			"this operator in v0",
	}, x.dtype, shape)
}

// elementwiseDType reports the storage types v0 computes over.
func elementwiseDType(d DType) bool { return d == accel.F32 || d == accel.F16 }

// Add returns x + y, elementwise, with NumPy broadcasting.
func Add(b *Builder, x, y *Tensor) *Tensor {
	return binary(b, "Add", &testkernels.ElemAddKernel, x, y, 1)
}

// Mul returns x * y, elementwise, with NumPy broadcasting.
func Mul(b *Builder, x, y *Tensor) *Tensor {
	return binary(b, "Mul", &testkernels.ElemMulKernel, x, y, 1)
}

// Scale multiplies x by a named runtime f32 scalar.
//
// A named scalar rather than a Go float32, because the value changes every step
// and a compiled plan must not have to be rebuilt for it. An attribute that
// changed a shape, a layout, or which kernel is selected would be different:
// that needs another plan, and specs/007-tensor-layer.md draws the line there.
func Scale(b *Builder, x *Tensor, scalarName string) *Tensor {
	if poisoned(x) {
		return b.poison()
	}
	if !elementwiseDType(x.dtype) {
		return b.fail(1, "Scale", "%v is not an elementwise dtype", x.dtype)
	}
	kind, ok := b.scalarKind(scalarName)
	if !ok {
		return b.fail(1, "Scale", "%q is not a declared scalar; declare it with Scalar so a "+
			"misspelling is an error rather than a value nobody binds", scalarName)
	}
	if kind != ScalarF32 {
		return b.fail(1, "Scale", "%q is declared %v and Scale needs f32", scalarName, kind)
	}
	if bad := b.elementwiseOperand(1, "Scale", 0, x, x.shape); bad != nil {
		return bad
	}
	return b.record(node{
		op: "Scale", inputs: []*Tensor{x}, kernel: &testkernels.ElemScaleKernel, bcast: true,
		reads: []string{scalarName},
		uniform: func(vals map[string]ScalarValue) any {
			return testkernels.ScaleParams{Factor: vals[scalarName].F32}
		},
		reason: "the contiguous elementwise variant, with the factor in a uniform block " +
			"rewritten before each submission",
	}, x.dtype, x.shape)
}

// SiLU returns x * sigmoid(x), elementwise.
//
// Evaluated in f32 whatever the storage dtype, which specs/007-tensor-layer.md
// requires and specs/008-numerics.md explains: an activation evaluated in f16
// loses accuracy where it matters most, near zero.
func SiLU(b *Builder, x *Tensor) *Tensor {
	if poisoned(x) {
		return b.poison()
	}
	if !elementwiseDType(x.dtype) {
		return b.fail(1, "SiLU", "%v is not an elementwise dtype", x.dtype)
	}
	if bad := b.elementwiseOperand(1, "SiLU", 0, x, x.shape); bad != nil {
		return bad
	}
	return b.record(node{
		op: "SiLU", inputs: []*Tensor{x}, kernel: &testkernels.SiLUKernel, bcast: true,
		reason: "the contiguous elementwise variant; exp is bounded by " +
			"specs/008-numerics.md section 6",
	}, x.dtype, x.shape)
}

// SwiGLU returns SiLU(gate) * value, elementwise.
//
// One authored kernel rather than two operators composed, which is a selection
// and not a fusion pass: specs/010-kernel-corpus.md registers it, and the
// composed definition remains the correctness reference. The shapes must be
// equal rather than merely broadcastable, because the fused kernel indexes all
// three operands together.
func SwiGLU(b *Builder, gate, value *Tensor) *Tensor {
	if poisoned(gate, value) {
		return b.poison()
	}
	if gate.dtype != value.dtype {
		return b.fail(1, "SwiGLU", "operands are %v and %v", gate.dtype, value.dtype)
	}
	if !elementwiseDType(gate.dtype) {
		return b.fail(1, "SwiGLU", "%v is not an elementwise dtype", gate.dtype)
	}
	if !gate.shape.Equal(value.shape) {
		return b.fail(1, "SwiGLU", "shapes %v and %v differ; the fused kernel indexes both "+
			"operands together, so broadcasting one would need the composed form",
			gate.shape, value.shape)
	}
	if bad := b.elementwiseOperand(1, "SwiGLU", 0, gate, gate.shape); bad != nil {
		return bad
	}
	if bad := b.elementwiseOperand(1, "SwiGLU", 1, value, gate.shape); bad != nil {
		return bad
	}
	return b.record(node{
		op: "SwiGLU", inputs: []*Tensor{gate, value}, kernel: &testkernels.SwiGLUKernel,
		bcast:    true,
		reason:   "the authored fused kernel, which specs/010-kernel-corpus.md registers",
		rejected: []string{"SiLU followed by Mul: correct, and two dispatches over the same bytes"},
	}, gate.dtype, gate.shape)
}
