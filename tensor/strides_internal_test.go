// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"testing"

	"golang.design/x/accel"
)

// No view produces a negative stride, which is why Contiguous's guard against
// one cannot be reached.
//
// specs/009-sequencing.md's refusal audit left two of Contiguous's refusals
// untested, and both are dominated rather than merely unexercised:
//
//   - "a rank-zero operand has no layout to pack" sits behind an early return
//     for an already-contiguous layout, and a rank-zero tensor has no strides
//     to be non-contiguous with. Both lines are in the same function, which is
//     a claim a reader can check without leaving it.
//
//   - "a negative stride" needs a stride below zero, and nothing makes one.
//
// The second is a claim about the whole package rather than one function, so it
// is asserted here instead of argued. That is the standard the audit's own
// correction set: it once called a family unreachable from a single
// construction path, and views were the path it had missed. So this enumerates
// the operators that write strides -- the ones that permute, restrict, or
// insert -- rather than reasoning from how the tensor was declared.
//
// Internal because a stride is not exported and should not become exported to
// let a test read it.
func TestNoViewProducesANegativeStride(t *testing.T) {
	b := &Builder{}
	x := Input(b, ValueDesc{Name: "x", DType: accel.F32, Shape: Shape{4, 3, 2}})

	for _, c := range []struct {
		name string
		make func() *Tensor
	}{
		{"declared", func() *Tensor { return x }},
		{"transposed", func() *Tensor { return Transpose(b, x, 0, 2) }},
		{"sliced", func() *Tensor { return Slice(b, x, 1, 1, 2) }},
		{"sliced to empty", func() *Tensor { return Slice(b, x, 0, 2, 2) }},
		{"reshaped", func() *Tensor { return Reshape(b, x, Shape{6, 4}) }},
		{"transposed then sliced", func() *Tensor {
			return Slice(b, Transpose(b, x, 0, 1), 2, 0, 1)
		}},
		{"sliced then transposed", func() *Tensor {
			return Transpose(b, Slice(b, x, 0, 1, 3), 1, 2)
		}},
	} {
		v := c.make()
		if b.Err() != nil {
			t.Fatalf("%s: %v", c.name, b.Err())
		}
		for i, s := range v.strides {
			if s < 0 {
				t.Errorf("%s: axis %d has stride %d, so Contiguous's guard "+
					"against a negative stride is reachable after all", c.name, i, s)
			}
		}
	}
}
