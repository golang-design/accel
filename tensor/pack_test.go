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

// A transposed operand packs into the values a transpose describes.
//
// The whole reason this operator exists: a view is free, a kernel that indexes
// contiguously cannot read one, and until now there was no way to convert it —
// specs/025-tensor-operators.md named it in four error messages, one of which
// told a caller to insert it.
//
// Checked element by element against the transpose computed here, so a kernel
// that packed the right *number* of elements in the wrong order fails.
func TestContiguousPacksATransposedView(t *testing.T) {
	const rows, cols = 3, 4
	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("transpose")

	in := tensor.Input(b, tensor.ValueDesc{
		Name: "in", DType: accel.F32, Shape: tensor.Shape{rows, cols},
	})
	tensor.Output(b, "out", tensor.Contiguous(b, tensor.Transpose(b, in, 0, 1)))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "transpose"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	src := make([]float32, rows*cols)
	for i := range src {
		src[i] = float32(i) + 0.5
	}
	out := f32Buffer(t, d, "out", make([]float32, rows*cols))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"in": f32Buffer(t, d, "in", src), "out": out,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, rows*cols)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}

	for r := range rows {
		for c := range cols {
			want := src[r*cols+c]
			if g := got[c*rows+r]; g != want {
				t.Fatalf("packed (%d,%d) is %v, want %v; the gather read the wrong "+
					"element rather than the wrong count", c, r, g, want)
			}
		}
	}
}

// A slice packs the elements the slice names and no others.
//
// The case a transpose does not cover: a slice moves the offset as well as the
// strides, so a kernel ignoring the offset packs the right shape from the wrong
// place — and every element is a plausible value.
func TestContiguousPacksASlice(t *testing.T) {
	const rows, cols = 4, 5
	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("slice")

	in := tensor.Input(b, tensor.ValueDesc{
		Name: "in", DType: accel.F32, Shape: tensor.Shape{rows, cols},
	})
	// Rows one and two, so the view starts at element cols rather than at the
	// origin.
	tensor.Output(b, "out", tensor.Contiguous(b, tensor.Slice(b, in, 0, 1, 3)))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "slice"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	src := make([]float32, rows*cols)
	for i := range src {
		src[i] = float32(i) + 0.25
	}
	out := f32Buffer(t, d, "out", make([]float32, 2*cols))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"in": f32Buffer(t, d, "in", src), "out": out,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, 2*cols)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}

	for r := range 2 {
		for c := range cols {
			want := src[(r+1)*cols+c]
			if g := got[r*cols+c]; g != want {
				t.Fatalf("packed (%d,%d) is %v, want %v; the view's offset was not "+
					"applied", r, c, g, want)
			}
		}
	}
}

// An operand already contiguous is returned unchanged rather than copied.
//
// What makes a defensive call free: a caller wrapping every matmul operand
// should not pay for the ones that did not need it.
func TestContiguousOnAPackedOperandCopiesNothing(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("noop")
	in := tensor.Input(b, tensor.ValueDesc{
		Name: "in", DType: accel.F32, Shape: tensor.Shape{4, 4},
	})
	if got := tensor.Contiguous(b, in); got != in {
		t.Error("a contiguous operand was packed anyway, so a defensive call costs a copy")
	}
}

// A rank beyond what one std140 block carries is refused, naming both numbers.
func TestContiguousRefusesARankBeyondTheBlock(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("deep")
	in := tensor.Input(b, tensor.ValueDesc{
		Name: "in", DType: accel.F32,
		Shape: tensor.Shape{2, 2, 2, 2, 2, 2, 2, 2, 2},
	})
	tensor.Output(b, "out", tensor.Contiguous(b, tensor.Transpose(b, in, 0, 1)))

	if _, err := b.Compile(rt, tensor.CompileOptions{Label: "deep"}); err == nil {
		t.Fatal("a rank-9 operand was accepted")
	} else if !strings.Contains(err.Error(), "rank 9") {
		t.Errorf("the refusal should name the rank, got %v", err)
	}
}
