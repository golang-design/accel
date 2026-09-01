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

// A Broadcast view feeds an elementwise operator, and is materialized as the
// repeated run it is.
//
// Broadcast's doc promised that a broadcast expressible as a repeated
// contiguous run is accepted, and every Broadcast result was refused: the
// lowering decided by rank and required a contiguous layout, and a Broadcast
// result has a zero stride by construction. So Add(x, g) with g of [width]
// worked and Add(x, Broadcast(g, [rows, width])) failed at Compile -- two
// spellings of one read pattern, one of them the spelling the doc recommends.
//
// Three shapes of the run are checked, because the walk that admits them
// treats each differently: a leading axis the operand lacks, a leading axis
// the view expanded from one, and both at once at rank three.
func TestABroadcastViewFeedsAnElementwiseOperator(t *testing.T) {
	const rows, width, batch = 4, 8, 2
	rt := newRuntime(t)
	d := rt.Device()

	xs := make([]float32, batch*rows*width)
	for i := range xs {
		xs[i] = float32(i)
	}
	gs := make([]float32, width)
	for i := range gs {
		gs[i] = float32(i%3) - 1
	}

	for _, tc := range []struct {
		name  string
		shape tensor.Shape // the result's
		g     tensor.Shape // the vector's declared shape
		view  func(b *tensor.Builder, g *tensor.Tensor) *tensor.Tensor
		times string // what the report should count
	}{{
		name: "a row across rows", shape: tensor.Shape{rows, width},
		g: tensor.Shape{width},
		view: func(b *tensor.Builder, g *tensor.Tensor) *tensor.Tensor {
			return tensor.Broadcast(b, g, tensor.Shape{rows, width})
		},
		times: "4 copies",
	}, {
		name: "a size-one row expanded", shape: tensor.Shape{rows, width},
		g: tensor.Shape{1, width},
		view: func(b *tensor.Builder, g *tensor.Tensor) *tensor.Tensor {
			return tensor.Broadcast(b, g, tensor.Shape{rows, width})
		},
		times: "4 copies",
	}, {
		name: "a row across a batch of rows", shape: tensor.Shape{batch, rows, width},
		g: tensor.Shape{1, width},
		view: func(b *tensor.Builder, g *tensor.Tensor) *tensor.Tensor {
			return tensor.Broadcast(b, g, tensor.Shape{batch, rows, width})
		},
		times: "8 copies",
	}} {
		t.Run(tc.name, func(t *testing.T) {
			b := rt.NewBuilder("bcast")
			x := tensor.Input(b, tensor.ValueDesc{Name: "x", DType: accel.F32, Shape: tc.shape})
			g := tensor.Weight(b, tensor.ValueDesc{Name: "g", DType: accel.F32, Shape: tc.g})
			tensor.Output(b, "y", tensor.Add(b, x, tc.view(b, g)))

			plan, err := b.Compile(rt, tensor.CompileOptions{})
			if err != nil {
				t.Fatalf("compile: %v", err)
			}
			defer plan.Close()

			reported := false
			for _, s := range plan.Selections() {
				if s.Kernel == "copy" && strings.Contains(s.Reason, tc.times) {
					reported = true
				}
			}
			if !reported {
				t.Errorf("no copy of %s was reported in %v", tc.times, plan.Selections())
			}

			n := tc.shape.Elements()
			out := f32Buffer(t, d, "y", make([]float32, n))
			f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
				"x": f32Buffer(t, d, "x", xs[:n]), "g": f32Buffer(t, d, "g", gs), "y": out,
			}})
			if err := f.Wait(); err != nil {
				t.Fatalf("submit: %v", err)
			}
			got := make([]float32, n)
			if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
				t.Fatalf("readback: %v", err)
			}
			for i := range got {
				if want := xs[i] + gs[i%width]; got[i] != want {
					t.Fatalf("element %d is %v, want %v: the row was not repeated across "+
						"every row", i, got[i], want)
				}
			}
		})
	}
}

// A broadcast that is not a repeated run is refused when the operator is
// recorded, at the line that wrote it.
//
// It was refused at Compile, whose diagnostic has no source position because
// the call that made the mistake is long returned -- and whose advice was to
// let an elementwise operator materialize it, which is the path that had just
// failed. The check is the lowering's own, run earlier.
func TestABroadcastThatIsNotARunIsRefusedWhereItWasWritten(t *testing.T) {
	const rows, width = 4, 8
	rt := newRuntime(t)

	for _, tc := range []struct {
		name  string
		build func(b *tensor.Builder, x *tensor.Tensor) *tensor.Tensor
	}{{
		// A column across columns: the expanded axis is the inner one, so
		// each element repeats with a stride rather than the row as a run.
		name: "a column across columns",
		build: func(b *tensor.Builder, x *tensor.Tensor) *tensor.Tensor {
			c := tensor.Input(b, tensor.ValueDesc{
				Name: "c", DType: accel.F32, Shape: tensor.Shape{rows, 1}})
			return tensor.Add(b, x, tensor.Broadcast(b, c, tensor.Shape{rows, width}))
		},
	}, {
		// A permuted operand of the result's own shape: no broadcast at
		// all, and not a run either.
		name: "a transposed operand",
		build: func(b *tensor.Builder, x *tensor.Tensor) *tensor.Tensor {
			y := tensor.Input(b, tensor.ValueDesc{
				Name: "y", DType: accel.F32, Shape: tensor.Shape{width, rows}})
			return tensor.Mul(b, x, tensor.Transpose(b, y, 0, 1))
		},
	}, {
		name: "a transposed operand of a unary",
		build: func(b *tensor.Builder, x *tensor.Tensor) *tensor.Tensor {
			return tensor.SiLU(b, tensor.Transpose(b, x, 0, 1))
		},
	}} {
		t.Run(tc.name, func(t *testing.T) {
			b := rt.NewBuilder("refused")
			x := tensor.Input(b, tensor.ValueDesc{
				Name: "x", DType: accel.F32, Shape: tensor.Shape{rows, width}})
			tc.build(b, x)
			err := b.Err()
			if err == nil {
				t.Fatal("recorded without a diagnostic")
			}
			msg := err.Error()
			if !strings.Contains(msg, "broadcast_test.go:") {
				t.Errorf("the refusal does not name the line that wrote it: %v", err)
			}
			if !strings.Contains(msg, "contiguous run repeated") {
				t.Errorf("the refusal does not say which broadcasts it can build: %v", err)
			}
			if !strings.Contains(msg, "Contiguous") {
				t.Errorf("the refusal does not say what to insert: %v", err)
			}
		})
	}
}
