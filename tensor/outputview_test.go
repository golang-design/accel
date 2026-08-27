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

// outputSum builds x+y and hands the result to decl, which names the ports.
func outputSum(t *testing.T, rt *tensor.Runtime, label string,
	decl func(b *tensor.Builder, sum *tensor.Tensor)) (*tensor.Plan, error) {

	t.Helper()
	b := rt.NewBuilder(label)
	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F32, Shape: tensor.Shape{3, 4}})
	y := tensor.Input(b, tensor.ValueDesc{
		Name: "y", DType: accel.F32, Shape: tensor.Shape{3, 4}})
	decl(b, tensor.Add(b, x, y))
	return b.Compile(rt, tensor.CompileOptions{Label: label})
}

// A reshaped result reaches its output port.
//
// [#26](https://github.com/golang-design/accel/issues/26), and the worst class
// there is: the shape was right, nothing refused, and every number was zero.
//
// The cause was that an output is matched to its producer by *tensor identity*,
// and a view is a different tensor than the node result it aliases. So the view
// named no producer, nothing wrote the port, and the caller read an untouched
// buffer out of a graph that compiled and ran.
//
// Reshape requires a contiguous operand and produces a contiguous result, so
// the reshaped value is the producer's own bytes under a different shape --
// which is why this resolves rather than refuses.
func TestAReshapedResultReachesItsOutput(t *testing.T) {
	rt := newRuntime(t)
	d := rt.Device()

	xs := make([]float32, 12)
	ys := make([]float32, 12)
	for i := range xs {
		xs[i], ys[i] = float32(i), float32(i)*2
	}

	run := func(label string, decl func(*tensor.Builder, *tensor.Tensor)) []float32 {
		t.Helper()
		plan, err := outputSum(t, rt, label, decl)
		if err != nil {
			t.Fatalf("%s compile: %v", label, err)
		}
		defer plan.Close()
		out := f32Buffer(t, d, "out", make([]float32, 12))
		f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
			"x": f32Buffer(t, d, "x", xs), "y": f32Buffer(t, d, "y", ys), "out": out,
		}})
		if err := f.Wait(); err != nil {
			t.Fatalf("%s submit: %v", label, err)
		}
		got := make([]float32, 12)
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("%s readback: %v", label, err)
		}
		return got
	}

	direct := run("direct", func(b *tensor.Builder, sum *tensor.Tensor) {
		tensor.Output(b, "out", sum)
	})
	reshaped := run("reshaped", func(b *tensor.Builder, sum *tensor.Tensor) {
		tensor.Output(b, "out", tensor.Reshape(b, sum, tensor.Shape{2, 6}))
	})

	// Against the direct form, and against zero. Comparing only the two would
	// pass if both were zero, which is the state this test was written from.
	for i := range direct {
		if reshaped[i] != direct[i] {
			t.Fatalf("element %d: reshaped %v, direct %v", i, reshaped[i], direct[i])
		}
	}
	nonzero := false
	for _, v := range direct {
		if v != 0 {
			nonzero = true
		}
	}
	if !nonzero {
		t.Fatal("the direct form is all zeros, so the comparison says nothing")
	}
}

// A strided view cannot back an output port, and says so.
//
// The other half of #26's fix. A slice or a transpose is a different set of
// bytes than its producer wrote, so binding the producer would hand back the
// wrong elements rather than none -- a worse failure than the one being fixed.
// The refusal names Contiguous, which records a pack whose result is a value in
// its own right, so the advice is actionable rather than an apology.
func TestAStridedViewCannotBackAnOutput(t *testing.T) {
	rt := newRuntime(t)

	_, err := outputSum(t, rt, "strided", func(b *tensor.Builder, sum *tensor.Tensor) {
		tensor.Output(b, "out", tensor.Transpose(b, sum, 0, 1))
	})
	if err == nil {
		t.Fatal("a transposed view was accepted as an output")
	}
	for _, want := range []string{"strided or partial view", "Insert Contiguous"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal should say %q, got %v", want, err)
		}
	}

	// The fix the message names works, which is what makes it advice.
	plan, err := outputSum(t, rt, "packed", func(b *tensor.Builder, sum *tensor.Tensor) {
		tensor.Output(b, "out", tensor.Contiguous(b, tensor.Transpose(b, sum, 0, 1)))
	})
	if err != nil {
		t.Fatalf("the fix the refusal names does not compile: %v", err)
	}
	plan.Close()
}

// Several ports naming one value each get it.
//
// Found probing next to #26 rather than reported, and it is the same defect:
// outputs were keyed by tensor, so a second Output of one value replaced the
// first and the earlier port was never written. Silent zeros again.
//
// A dispatch writes one place, so the node writes a transient and every port
// takes a copy. Refusing instead would have left a caller stuck, because
// nothing in this API forces a copy -- Contiguous is a no-op on a value that is
// already contiguous, which is exactly what a node result is.
func TestSeveralPortsMayNameOneValue(t *testing.T) {
	rt := newRuntime(t)
	d := rt.Device()

	plan, err := outputSum(t, rt, "fanout", func(b *tensor.Builder, sum *tensor.Tensor) {
		tensor.Output(b, "a", sum)
		tensor.Output(b, "bb", sum)
		tensor.Output(b, "cc", tensor.Reshape(b, sum, tensor.Shape{2, 6}))
	})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	xs := make([]float32, 12)
	ys := make([]float32, 12)
	for i := range xs {
		xs[i], ys[i] = float32(i+1), float32(i+1)*10
	}
	outs := map[string]accel.BufferView{}
	bufs := map[string]accel.BufferView{
		"x": f32Buffer(t, d, "x", xs), "y": f32Buffer(t, d, "y", ys),
	}
	for _, n := range []string{"a", "bb", "cc"} {
		outs[n] = f32Buffer(t, d, n, make([]float32, 12))
		bufs[n] = outs[n]
	}
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: bufs})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	for _, n := range []string{"a", "bb", "cc"} {
		got := make([]float32, 12)
		if err := d.Queue().ReadBuffer(outs[n].Buffer, 0, got); err != nil {
			t.Fatalf("readback %s: %v", n, err)
		}
		for i := range got {
			if want := xs[i] + ys[i]; got[i] != want {
				t.Fatalf("port %q element %d is %v, want %v", n, i, got[i], want)
			}
		}
	}
}
