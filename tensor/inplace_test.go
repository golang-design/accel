// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// An in-place operator whose result nothing names costs one intermediate.
//
// RoPE rewrites what it reads, so the lowering copies its operand into a
// scratch transient and the kernel rewrites that. The scratch *is* the value
// its consumers read. The lowering also allocated the ordinary result transient
// first and then never bound it, so a RoPE feeding another operator was
// planned, aliased, and reported in Memory as two intermediates for one value.
//
// UnaliasedBytes is what the planner would need with no aliasing at all, which
// is the sum of every intermediate's allocation, so it is the figure that sees
// the second one. It is compared against a plan with one ordinary intermediate
// of the same size rather than against an element count, because an
// allocation is rounded to the device's granularity and that rounding is the
// graph's business rather than this test's.
func TestAnInPlaceOperatorCostsOneIntermediate(t *testing.T) {
	const rows, width = 4, 8
	rt := newRuntime(t)

	unaliased := func(label string, mid func(b *tensor.Builder, x *tensor.Tensor) *tensor.Tensor) int {
		t.Helper()
		b := rt.NewBuilder(label)
		x := tensor.Input(b, tensor.ValueDesc{
			Name: "x", DType: accel.F32, Shape: tensor.Shape{rows, width}})
		// The middle value feeds SiLU rather than the output directly: an
		// output is written into the caller's slot and allocates nothing, so
		// a RoPE named as the output would show one intermediate whether or
		// not the second was allocated.
		tensor.Output(b, "y", tensor.SiLU(b, mid(b, x)))
		plan, err := b.Compile(rt, tensor.CompileOptions{Label: label})
		if err != nil {
			t.Fatalf("%s: compile: %v", label, err)
		}
		defer plan.Close()
		return plan.Memory().UnaliasedBytes
	}

	// One ordinary intermediate of the same shape: SiLU's result.
	one := unaliased("silu", func(b *tensor.Builder, x *tensor.Tensor) *tensor.Tensor {
		return tensor.SiLU(b, x)
	})
	if one == 0 {
		t.Fatal("the reference plan reports no intermediate, so the comparison below " +
			"is about nothing")
	}

	got := unaliased("rope", func(b *tensor.Builder, x *tensor.Tensor) *tensor.Tensor {
		tensor.Scalar(b, tensor.ScalarDesc{Name: "theta", Kind: tensor.ScalarF32})
		pos := tensor.Input(b, tensor.ValueDesc{
			Name: "pos", DType: accel.U32, Shape: tensor.Shape{rows}})
		return tensor.RoPE(b, x, width, "theta", pos)
	})
	if got != one {
		t.Fatalf("a RoPE feeding SiLU holds %d bytes of intermediates before aliasing, "+
			"and a SiLU feeding SiLU holds %d: the in-place scratch is the value, and "+
			"a result transient nothing binds should not be allocated beside it",
			got, one)
	}
}
