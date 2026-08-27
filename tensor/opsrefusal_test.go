// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// The dtype and shape refusals the operator layer makes, each naming what it
// refused.
//
// specs/009-sequencing.md's refusal audit: these exist in code and no test had
// made any of them fire, so nothing said whether the condition was right or the
// message named the right value. They are gathered in one table because they
// are one kind of check -- an operator declining a dtype or a shape it has no
// kernel for -- and reading them together is how a missing one is noticed.
func TestTheOperatorDtypeAndShapeRefusals(t *testing.T) {
	rt := newRuntime(t)

	f32 := func(b *tensor.Builder, n string, dims ...int) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: n, DType: accel.F32, Shape: dims})
	}
	typed := func(b *tensor.Builder, n string, dt accel.DType, dims ...int) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: n, DType: dt, Shape: dims})
	}

	for _, tc := range []struct {
		name  string
		build func(b *tensor.Builder)
		want  string
	}{{
		name:  "a port with no name",
		build: func(b *tensor.Builder) { f32(b, "", 4) },
		want:  "a port needs a name",
	}, {
		name: "a port with no shape",
		build: func(b *tensor.Builder) {
			tensor.Input(b, tensor.ValueDesc{Name: "x", DType: accel.F32})
		},
		want: "a scalar is a shape of [1]",
	}, {
		name:  "an output with no name",
		build: func(b *tensor.Builder) { tensor.Output(b, "", f32(b, "x", 4)) },
		want:  "an output needs a name",
	}, {
		name: "an elementwise op over an integer dtype",
		build: func(b *tensor.Builder) {
			tensor.Add(b, typed(b, "x", accel.U32, 4), typed(b, "y", accel.U32, 4))
		},
		want: "is not an elementwise dtype",
	}, {
		name: "Scale over an integer dtype",
		build: func(b *tensor.Builder) {
			tensor.Scalar(b, tensor.ScalarDesc{Name: "s", Kind: tensor.ScalarF32})
			tensor.Scale(b, typed(b, "x", accel.U32, 4), "s")
		},
		want: "is not an elementwise dtype",
	}, {
		name: "SwiGLU over two different dtypes",
		build: func(b *tensor.Builder) {
			tensor.SwiGLU(b, f32(b, "g", 4), typed(b, "v", accel.F16, 4))
		},
		want: "operands are",
	}, {
		name: "SwiGLU over an integer dtype",
		build: func(b *tensor.Builder) {
			tensor.SwiGLU(b, typed(b, "g", accel.U32, 4), typed(b, "v", accel.U32, 4))
		},
		want: "is not an elementwise dtype",
	}, {
		name: "RoPE over a narrow dtype",
		build: func(b *tensor.Builder) {
			tensor.Scalar(b, tensor.ScalarDesc{Name: "theta", Kind: tensor.ScalarF32})
			pos := typed(b, "pos", accel.U32, 1)
			tensor.RoPE(b, typed(b, "x", accel.F16, 1, 2, 4), 4, "theta", pos)
		},
		want: "the registered kernel reads f32",
	}, {
		name: "a Cast the corpus does not register",
		build: func(b *tensor.Builder) {
			tensor.Cast(b, typed(b, "x", accel.F16, 4), accel.BF16)
		},
		want: "registers f32 to f16",
	}, {
		// Both axes, because the two checks are separate lines carrying the
		// same sentence. One case reaches whichever is second and leaves the
		// first untested, and the message cannot tell them apart -- so the
		// axis number in each expectation is what says which one answered.
		name: "a Transpose first axis outside the tensor",
		build: func(b *tensor.Builder) {
			tensor.Transpose(b, f32(b, "x", 2, 3), 7, 1)
		},
		want: "axis 7 is outside a rank-2 tensor",
	}, {
		name: "a Transpose second axis outside the tensor",
		build: func(b *tensor.Builder) {
			tensor.Transpose(b, f32(b, "x", 2, 3), 0, 5)
		},
		want: "axis 5 is outside a rank-2 tensor",
	}, {
		// Contiguous returns its operand untouched when the layout is already
		// packed, so every refusal below it needs a *strided* operand. A
		// transpose is what makes one.
		name: "packing a dtype the kernel does not move",
		build: func(b *tensor.Builder) {
			tensor.Contiguous(b, tensor.Transpose(b, typed(b, "x", accel.U32, 2, 3), 0, 1))
		},
		want: "not a dtype the packing kernel moves",
	}, {
		name: "packing a strided view with an empty axis",
		build: func(b *tensor.Builder) {
			x := tensor.Transpose(b, f32(b, "x", 2, 3), 0, 1)
			tensor.Contiguous(b, tensor.Slice(b, x, 0, 1, 1))
		},
		want: "needs every extent to compute an index",
	}} {
		t.Run(tc.name, func(t *testing.T) {
			b := rt.NewBuilder("refusal")
			tc.build(b)
			err := b.Err()
			if err == nil {
				t.Fatalf("%s was accepted", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("the refusal should say %q, got %v", tc.want, err)
			}
		})
	}
}
