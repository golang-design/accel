// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// A recurrent state is a State indexed by sequence slot, and it shares a graph
// with a KV cache indexed by position.
//
// This is the accepting half of specs/043-per-row-values.md §9's correction,
// and it exists because that correction was published to a consumer as an
// argument rather than as a result. §9 first recorded that State conflates two
// things and that a recurrent state needed a type of its own before any
// recurrent kernel could be built. The correction withdrew that, on the grounds
// that a recurrent state is indexed every step in any batch -- one slot per
// sequence -- and that a hybrid model holding both kinds is therefore two
// States of different shapes rather than a type-level problem.
//
// Both halves of the withdrawal are claims about what compiles and runs, so a
// reading of the shape arithmetic is the wrong kind of evidence for them. This
// is the right kind.
//
// What it asserts, in the order the correction makes them:
//
//  1. a matrix-valued recurrent state declares as an ordinary State;
//  2. writing one sequence's state is ScatterRows at that sequence's slot,
//     which is indexing, and it leaves the other sequences alone;
//  3. a paged KV cache and a recurrent state are live in one graph, in one
//     submission, which is the constraint the correction says dissolves.
func TestARecurrentStateIsAStateIndexedBySequence(t *testing.T) {
	const (
		slots   = 3 // sequences in the batch
		heads   = 2
		keyDim  = 4
		valDim  = 4
		kvHeads = 2
		headDim = 8
		posCap  = 6 // positions the KV cache holds
	)
	// One sequence's recurrent state is everything after the indexed axis:
	// heads x keyDim x valDim. That is what ScatterRows calls a row.
	const width = heads * keyDim * valDim

	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("hybrid")

	// The recurrent state. Its first axis is the sequence slot and it does not
	// grow with context, which is the whole appeal: a 262K-context model has no
	// KV cache for these layers.
	rec := tensor.NewState(b, tensor.StateDesc{
		Name: "recurrent", DType: accel.F32,
		Shape: tensor.Shape{slots, heads, keyDim, valDim},
	})
	// The KV cache of a full-attention layer, in the same graph. Its first axis
	// is the position within a sequence and it does grow.
	kc := tensor.NewState(b, tensor.StateDesc{
		Name: "kcache", DType: accel.F32, Shape: tensor.Shape{posCap, kvHeads, headDim},
	})
	vc := tensor.NewState(b, tensor.StateDesc{
		Name: "vcache", DType: accel.F32, Shape: tensor.Shape{posCap, kvHeads, headDim},
	})

	// Sequence 1's new recurrent state, written at slot 1. The id names a
	// sequence rather than a position, and that is the only difference between
	// this call and the one below it.
	newS := tensor.Input(b, tensor.ValueDesc{
		Name: "news", DType: accel.F32, Shape: tensor.Shape{1, width},
	})
	slot := tensor.Input(b, tensor.ValueDesc{
		Name: "slot", DType: accel.U32, Shape: tensor.Shape{1},
	})
	// The write's result is not an Output: a state is caller-owned storage, so
	// its contents are already the caller's to read from the buffer they bound.
	// Asking the plan to copy it into a second buffer is refused by name, which
	// is the rule this test met on its first run.
	_ = tensor.ScatterRows(b, rec, newS, slot)

	// The full-attention layer's append and read, unchanged from a dense model.
	newK := tensor.Input(b, tensor.ValueDesc{
		Name: "newk", DType: accel.F32, Shape: tensor.Shape{1, kvHeads * headDim},
	})
	newV := tensor.Input(b, tensor.ValueDesc{
		Name: "newv", DType: accel.F32, Shape: tensor.Shape{1, kvHeads * headDim},
	})
	pos := tensor.Input(b, tensor.ValueDesc{
		Name: "pos", DType: accel.U32, Shape: tensor.Shape{1},
	})
	lengths := tensor.Input(b, tensor.ValueDesc{
		Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
	})
	q := tensor.Input(b, tensor.ValueDesc{
		Name: "q", DType: accel.F32, Shape: tensor.Shape{kvHeads, headDim},
	})
	tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
	kc2 := tensor.ScatterRows(b, kc, newK, pos)
	vc2 := tensor.ScatterRows(b, vc, newV, pos)
	tensor.Output(b, "attn", tensor.Attention(b, q, kc2, vc2, tensor.AttentionOptions{
		Lengths: lengths, ScaleName: "scale",
	}))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "hybrid"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	// Slot 0 and slot 2 are filled beforehand with values this submission must
	// not touch, which is what makes the write an index rather than a rewrite.
	before := make([]float32, slots*width)
	for i := range before {
		before[i] = float32(i%17) - 8
	}
	recBuf := f32Buffer(t, d, "recurrent", before)

	written := make([]float32, width)
	for i := range written {
		written[i] = 100 + float32(i)
	}
	qs := make([]float32, kvHeads*headDim)
	ks := make([]float32, kvHeads*headDim)
	vs := make([]float32, kvHeads*headDim)
	for i := range qs {
		qs[i] = float32(math.Sin(float64(i)))
		ks[i] = float32(math.Cos(float64(i)))
		vs[i] = float32(i) / 4
	}

	attnOut := f32Buffer(t, d, "attn", make([]float32, kvHeads*headDim))
	f := plan.Submit(d.Queue(), tensor.Bindings{
		Buffers: map[string]accel.BufferView{
			"recurrent": recBuf,
			"news":      f32Buffer(t, d, "news", written),
			"slot":      u32Buffer(t, d, "slot", []uint32{1}),
			"kcache":    f32Buffer(t, d, "kcache", make([]float32, posCap*kvHeads*headDim)),
			"vcache":    f32Buffer(t, d, "vcache", make([]float32, posCap*kvHeads*headDim)),
			"newk":      f32Buffer(t, d, "newk", ks),
			"newv":      f32Buffer(t, d, "newv", vs),
			"pos":       u32Buffer(t, d, "pos", []uint32{0}),
			"len":       u32Buffer(t, d, "len", []uint32{1}),
			"q":         f32Buffer(t, d, "q", qs),
			"attn":      attnOut,
		},
		Scalars: map[string]tensor.ScalarValue{
			"scale": tensor.F32(float32(1 / math.Sqrt(headDim))),
		},
	})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	// Read back the buffer that was bound, which is where the write landed.
	got := make([]float32, slots*width)
	if err := d.Queue().ReadBuffer(recBuf.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}
	for i := range width {
		if v := got[1*width+i]; v != written[i] {
			t.Fatalf("slot 1 element %d is %v, want %v: the sequence's state did not "+
				"land where its id named", i, v, written[i])
		}
	}
	for _, other := range []int{0, 2} {
		for i := range width {
			if v, want := got[other*width+i], before[other*width+i]; v != want {
				t.Fatalf("slot %d element %d changed from %v to %v: writing one "+
					"sequence's state disturbed another's, so the id is not indexing",
					other, i, want, v)
			}
		}
	}

	// The full-attention half of the same submission produced something, which
	// is the part that says the two kinds coexist rather than merely compile.
	attn := make([]float32, kvHeads*headDim)
	if err := d.Queue().ReadBuffer(attnOut.Buffer, 0, attn); err != nil {
		t.Fatalf("readback: %v", err)
	}
	nonzero := 0
	for _, v := range attn {
		if v != 0 {
			nonzero++
		}
	}
	if nonzero == 0 {
		t.Fatal("the attention output is all zero, so the full-attention layer did " +
			"not run and this says nothing about the two kinds sharing a graph")
	}
}
