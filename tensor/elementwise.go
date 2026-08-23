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
// # Why these four first
//
// specs/024-tensor-bringup.md picks them as the smallest set that exercises
// every part of the slice: two operands and broadcasting (Add, Mul), a unary
// with a bounded primitive (SiLU), and a fused two-operand form whose semantic
// definition is a composition (SwiGLU). A larger set would prove nothing more
// about the machinery and a smaller one would leave a piece of it unexercised.
//
// # Where the kernels come from
//
// specs/010-kernel-corpus.md's corpus, unchanged. The tensor layer selects
// among kernels; it does not author them, and it does not reach past the device
// layer to run them. Every operator here becomes an ordinary Recorder dispatch.

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
	return b.record(node{
		op: op, inputs: []*Tensor{x, y}, kernel: k,
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

// Scale, and the whole named-scalar surface, is not built here. See
// specs/024-tensor-bringup.md's deviation 1.
//
// A by-value uniform travels with a recorded dispatch and the device layer has
// no way to rewrite one between submissions, so a factor that changed every
// step would need a new plan -- exactly what specs/007-tensor-layer.md says a
// runtime scalar must not need. Closing that is a device-layer change with its
// own scope rather than something to improvise here.
//
// Scalar, ScalarDesc and ScalarValue are absent rather than present and
// unusable, because an API a caller can call and cannot use is worse than one
// that is not there: the compiler says nothing, and they find out at runtime.
// specs/025-tensor-operators.md adds the mechanism and the surface together.

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
	return b.record(node{
		op: "SiLU", inputs: []*Tensor{x}, kernel: &testkernels.SiLUKernel,
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
	return b.record(node{
		op: "SwiGLU", inputs: []*Tensor{gate, value}, kernel: &testkernels.SwiGLUKernel,
		reason:   "the authored fused kernel, which specs/010-kernel-corpus.md registers",
		rejected: []string{"SiLU followed by Mul: correct, and two dispatches over the same bytes"},
	}, gate.dtype, gate.shape)
}
