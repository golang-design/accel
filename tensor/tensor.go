// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package tensor is the layer that turns a model into device work.
//
// You describe a computation once as a graph of tensors, compile it into a
// [Plan], and submit that plan with different inputs as often as you like. The
// plan owns its intermediate memory and its device graph; you own the buffers
// you named.
//
// # Why compile once
//
// A transformer runs the same shapes thousands of times. Deciding which kernel
// to use, where the intermediates live, and which barriers are needed is work
// that depends on the shapes and not on the numbers, so it is done once:
//
//	rt, _ := tensor.NewRuntime(dev)
//	b := rt.NewBuilder("mlp")
//	x := tensor.Input(b, tensor.ValueDesc{Name: "x", DType: accel.F32, Shape: tensor.Shape{4, 8}})
//	w := tensor.Weight(b, tensor.ValueDesc{Name: "w", DType: accel.F32, Shape: tensor.Shape{4, 8}})
//	tensor.Output(b, "y", tensor.Mul(b, x, w))
//	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "mlp"})
//
// Then submit it as often as you like, rebinding what changed.
//
// # Errors do not interrupt you
//
// An operator that cannot be built records the problem and returns a poisoned
// tensor, so model code has no error branch per line. A poisoned tensor
// produces poisoned results and no further errors: one mistake gives one
// diagnostic, and [Builder.Compile] returns them all together, each naming the
// operator, the operand, and the line of your code that made it.
//
// # What it does not do
//
// No quantization, no sampling, no automatic plan cache, and no scheduler
// across requests. Those are deliberately above this package. See
// specs/007-tensor-layer.md.
package tensor

import (
	"fmt"
	"strings"

	"golang.design/x/accel"
)

// Shape is a tensor's extent, outermost dimension first.
//
// The last dimension varies fastest, which is row-major and is what every
// kernel in the corpus indexes. All dimensions are positive concrete integers:
// there is no symbolic shape in v0, so inference is arithmetic rather than a
// solver.
type Shape []int

// Elements is the number of values a shape holds.
func (s Shape) Elements() int {
	n := 1
	for _, d := range s {
		n *= d
	}
	return n
}

func (s Shape) String() string {
	var b strings.Builder
	b.WriteByte('[')
	for i, d := range s {
		if i > 0 {
			b.WriteByte(' ')
		}
		fmt.Fprintf(&b, "%d", d)
	}
	b.WriteByte(']')
	return b.String()
}

// Equal reports whether two shapes are the same.
func (s Shape) Equal(t Shape) bool {
	if len(s) != len(t) {
		return false
	}
	for i := range s {
		if s[i] != t[i] {
			return false
		}
	}
	return true
}

// contiguous is the stride vector of a densely packed shape, in elements.
//
// The suffix products, which is what "the last dimension varies fastest"
// means arithmetically: stride[i] is the number of elements one step along
// axis i skips.
func contiguous(s Shape) []int {
	st := make([]int, len(s))
	n := 1
	for i := len(s) - 1; i >= 0; i-- {
		st[i] = n
		n *= s[i]
	}
	return st
}

// DType is a storage element type, shared with the device layer.
type DType = accel.DType

// Tensor is an immutable logical value in one builder's graph.
//
// Immutable because the graph is a DAG of values rather than a sequence of
// mutations: an operator returns a new tensor and never writes into an operand,
// which is what lets the planner decide where intermediates live and which of
// them can share memory.
type Tensor struct {
	b     *Builder
	dtype DType
	shape Shape

	// strides is in elements, and offset is where this view starts. Together
	// they make Reshape, Slice, Permute and Broadcast host-side bookkeeping
	// rather than copies.
	strides []int
	offset  int

	// node is the index of the operator that produced this value, or -1 for an
	// external port.
	node int

	// port names the external buffer this value comes from, when it has one.
	port string

	// poison marks a value that could not be built. It flows through operators
	// without producing further errors, so one mistake yields one diagnostic.
	poison bool
}

// DType reports the storage element type.
func (t *Tensor) DType() DType { return t.dtype }

// Shape reports the extent.
func (t *Tensor) Shape() Shape { return t.shape }

// Contiguous reports whether this tensor's strides are the densely packed ones,
// which is what the corpus kernels index.
func (t *Tensor) contiguousLayout() bool {
	want := contiguous(t.shape)
	if t.offset != 0 || len(want) != len(t.strides) {
		return false
	}
	for i := range want {
		if want[i] != t.strides[i] {
			return false
		}
	}
	return true
}

func (t *Tensor) String() string {
	if t == nil {
		return "<nil tensor>"
	}
	if t.poison {
		return "<invalid tensor>"
	}
	return fmt.Sprintf("%v%v", t.dtype, t.shape)
}

// broadcastShape is NumPy's rule: right-aligned, and a size-one axis expands.
//
// Returned rather than applied, because both operands need the same answer and
// computing it twice invites the two from drifting.
func broadcastShape(a, b Shape) (Shape, bool) {
	n := max(len(a), len(b))
	out := make(Shape, n)
	for i := range n {
		x, y := 1, 1
		if j := len(a) - n + i; j >= 0 {
			x = a[j]
		}
		if j := len(b) - n + i; j >= 0 {
			y = b[j]
		}
		switch {
		case x == y:
			out[i] = x
		case x == 1:
			out[i] = y
		case y == 1:
			out[i] = x
		default:
			return nil, false
		}
	}
	return out, true
}
