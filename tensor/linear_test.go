// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

type linearCase struct {
	batch, heads, keyDim, valueDim int
	counts                         []uint32
}

func (c linearCase) tokens() int {
	n := 0
	for _, v := range c.counts {
		n += int(v)
	}
	return n
}

// linearPlan builds one gated delta layer and returns the plan and its buffers.
type linearPlan struct {
	plan          *tensor.Plan
	dev           *accel.Device
	bufs          map[string]accel.BufferView
	stateBuf, out accel.BufferView
	c             linearCase
}

func buildLinear(t *testing.T, c linearCase) *linearPlan {
	t.Helper()
	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("linear")

	tokens := c.tokens()
	q := tensor.Input(b, tensor.ValueDesc{
		Name: "q", DType: accel.F32, Shape: tensor.Shape{tokens, c.heads, c.keyDim},
	})
	k := tensor.Input(b, tensor.ValueDesc{
		Name: "k", DType: accel.F32, Shape: tensor.Shape{tokens, c.heads, c.keyDim},
	})
	v := tensor.Input(b, tensor.ValueDesc{
		Name: "v", DType: accel.F32, Shape: tensor.Shape{tokens, c.heads, c.valueDim},
	})
	alpha := tensor.Input(b, tensor.ValueDesc{
		Name: "alpha", DType: accel.F32, Shape: tensor.Shape{tokens},
	})
	beta := tensor.Input(b, tensor.ValueDesc{
		Name: "beta", DType: accel.F32, Shape: tensor.Shape{tokens},
	})
	extents := tensor.Input(b, tensor.ValueDesc{
		Name: "extents", DType: accel.U32, Shape: tensor.Shape{c.batch},
	})
	st := tensor.NewState(b, tensor.StateDesc{
		Name: "state", DType: accel.F32,
		Shape: tensor.Shape{c.batch, c.heads, c.valueDim, c.keyDim},
	})

	o, _ := tensor.LinearAttention(b, q, k, v, st, tensor.LinearOptions{
		Alpha: alpha, Beta: beta, QueryExtents: extents,
	})
	tensor.Output(b, "out", o)

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "linear"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	t.Cleanup(func() { plan.Close() })

	stateBuf := f32Buffer(t, d, "state",
		make([]float32, c.batch*c.heads*c.valueDim*c.keyDim))
	out := f32Buffer(t, d, "out", make([]float32, tokens*c.heads*c.valueDim))
	return &linearPlan{
		plan: plan, dev: d, stateBuf: stateBuf, out: out, c: c,
		bufs: map[string]accel.BufferView{"state": stateBuf, "out": out},
	}
}

// step submits one dispatch with the given per-token inputs.
func (p *linearPlan) step(t *testing.T, q, k, v, alpha, beta []float32,
	counts []uint32) []float32 {

	t.Helper()
	bufs := map[string]accel.BufferView{
		"q":       f32Buffer(t, p.dev, "q", q),
		"k":       f32Buffer(t, p.dev, "k", k),
		"v":       f32Buffer(t, p.dev, "v", v),
		"alpha":   f32Buffer(t, p.dev, "alpha", alpha),
		"beta":    f32Buffer(t, p.dev, "beta", beta),
		"extents": u32Buffer(t, p.dev, "extents", counts),
		"state":   p.stateBuf,
		"out":     p.out,
	}
	f := p.plan.Submit(p.dev.Queue(), tensor.Bindings{Buffers: bufs})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, p.c.tokens()*p.c.heads*p.c.valueDim)
	if err := p.dev.Queue().ReadBuffer(p.out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}
	return got
}

func linearInputs(c linearCase, seed int) (q, k, v, alpha, beta []float32) {
	tokens := c.tokens()
	q = make([]float32, tokens*c.heads*c.keyDim)
	k = make([]float32, len(q))
	for i := range q {
		q[i] = float32(math.Sin(float64(seed*7+i)*0.31)) / 4
		k[i] = float32(math.Cos(float64(seed*5+i)*0.17)) / 4
	}
	v = make([]float32, tokens*c.heads*c.valueDim)
	for i := range v {
		v[i] = float32(math.Sin(float64(seed*3+i)*0.23)) * 2
	}
	alpha = make([]float32, tokens)
	beta = make([]float32, tokens)
	for i := range alpha {
		alpha[i], beta[i] = 0.9, 0.5
	}
	return q, k, v, alpha, beta
}

// The state survives from one submission to the next.
//
// specs/047-linear-attention.md §3: within one submission the state could be a
// transient, and across submissions it cannot — a decode step is one submission
// per token, so this is what makes the layer a layer rather than a function.
//
// Two identical steps must differ, because the second starts from what the
// first left. A state that reset between submissions makes them equal, which is
// the failure this catches and which no single-step test can see.
func TestALinearStateSurvivesBetweenSubmissions(t *testing.T) {
	c := linearCase{batch: 2, heads: 2, keyDim: 4, valueDim: 4, counts: []uint32{1, 1}}
	p := buildLinear(t, c)
	q, k, v, alpha, beta := linearInputs(c, 0)

	first := append([]float32(nil), p.step(t, q, k, v, alpha, beta, c.counts)...)
	second := p.step(t, q, k, v, alpha, beta, c.counts)

	same := true
	for i := range first {
		if first[i] != second[i] {
			same = false
		}
	}
	if same {
		t.Fatal("two identical steps produced identical outputs; the second started " +
			"from the state the first left, so they must differ -- an equal result " +
			"means the state reset between submissions and the layer has no memory")
	}

	// The assertion above only means something if an equal result is what a
	// reset state actually produces, so that is checked rather than assumed:
	// zero the buffer between two identical steps and they must match. A first
	// attempt to prove this by breaking the kernel proved nothing instead --
	// the mutation used a form outside the kernel subset, go generate failed,
	// and the test ran against the unmutated lowering and passed.
	zeros := make([]float32, c.batch*c.heads*c.valueDim*c.keyDim)
	if err := p.dev.Queue().WriteBuffer(p.stateBuf.Buffer, 0, zeros); err != nil {
		t.Fatalf("clear: %v", err)
	}
	reset := p.step(t, q, k, v, alpha, beta, c.counts)
	for i := range first {
		if reset[i] != first[i] {
			t.Fatalf("element %d: a step from a zeroed state gave %v where the first "+
				"step from a zeroed state gave %v; the two are the same computation, so "+
				"the comparison above cannot tell a carried state from a reset one",
				i, reset[i], first[i])
		}
	}

	// And the state buffer itself moved, which says the write reached the
	// caller's storage rather than a transient the planner aliased.
	got := make([]float32, c.batch*c.heads*c.valueDim*c.keyDim)
	if err := p.dev.Queue().ReadBuffer(p.stateBuf.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}
	moved := false
	for _, x := range got {
		if x != 0 {
			moved = true
		}
	}
	if !moved {
		t.Fatal("the state buffer is still all zero after two steps")
	}
}

// A sequence contributing nothing keeps its state untouched.
func TestALinearStepLeavesAnEmptySequenceAlone(t *testing.T) {
	c := linearCase{batch: 2, heads: 2, keyDim: 4, valueDim: 4, counts: []uint32{2, 0}}
	p := buildLinear(t, c)
	q, k, v, alpha, beta := linearInputs(c, 1)
	p.step(t, q, k, v, alpha, beta, c.counts)

	got := make([]float32, c.batch*c.heads*c.valueDim*c.keyDim)
	if err := p.dev.Queue().ReadBuffer(p.stateBuf.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}
	slot := c.heads * c.valueDim * c.keyDim
	for i := range slot {
		if got[slot+i] != 0 {
			t.Fatalf("sequence 1 contributed no tokens and its state element %d is %v, "+
				"which started at zero", i, got[slot+i])
		}
	}
	// Sequence 0 did move, or the assertion above is about a kernel that wrote
	// nothing at all.
	moved := false
	for i := range slot {
		if got[i] != 0 {
			moved = true
		}
	}
	if !moved {
		t.Fatal("sequence 0 contributed two tokens and its state is still zero")
	}
}

// The state's last two axes are checked, because transposing them still runs.
//
// specs/047-linear-attention.md §1 derives [valueDim, keyDim] from the
// recurrence rather than asserting it. At valueDim == keyDim a transposed state
// has the right size and the wrong meaning, so the shape is refused by name.
func TestALinearStateShapeIsRefusedWhenTransposed(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("shape")
	const batch, heads, keyDim, valueDim = 1, 1, 6, 4

	q := tensor.Input(b, tensor.ValueDesc{
		Name: "q", DType: accel.F32, Shape: tensor.Shape{1, heads, keyDim},
	})
	k := tensor.Input(b, tensor.ValueDesc{
		Name: "k", DType: accel.F32, Shape: tensor.Shape{1, heads, keyDim},
	})
	v := tensor.Input(b, tensor.ValueDesc{
		Name: "v", DType: accel.F32, Shape: tensor.Shape{1, heads, valueDim},
	})
	// The last two swapped.
	st := tensor.NewState(b, tensor.StateDesc{
		Name: "state", DType: accel.F32,
		Shape: tensor.Shape{batch, heads, keyDim, valueDim},
	})
	o, _ := tensor.LinearAttention(b, q, k, v, st, tensor.LinearOptions{
		Alpha: tensor.Input(b, tensor.ValueDesc{
			Name: "alpha", DType: accel.F32, Shape: tensor.Shape{1},
		}),
		Beta: tensor.Input(b, tensor.ValueDesc{
			Name: "beta", DType: accel.F32, Shape: tensor.Shape{1},
		}),
		QueryExtents: tensor.Input(b, tensor.ValueDesc{
			Name: "extents", DType: accel.U32, Shape: tensor.Shape{batch},
		}),
	})
	tensor.Output(b, "out", o)
	_, err := b.Compile(rt, tensor.CompileOptions{Label: "shape"})
	if err == nil {
		t.Fatal("a state with its last two axes transposed was accepted")
	}
	if !strings.Contains(err.Error(), "valueDim, keyDim") {
		t.Fatalf("refused with %q, which does not say which axis is which", err)
	}
}

// A gate bound per sequence rather than per token is refused.
//
// The mistake QueryExtents makes easy: both are one-dimensional u32-or-f32
// tensors of small extent, and at one token per sequence they are the same
// length. The refusal names which is which.
func TestALinearGateIsPerTokenNotPerSequence(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("gate")
	const batch, heads, keyDim, valueDim, tokens = 2, 1, 4, 4, 3

	q := tensor.Input(b, tensor.ValueDesc{
		Name: "q", DType: accel.F32, Shape: tensor.Shape{tokens, heads, keyDim},
	})
	k := tensor.Input(b, tensor.ValueDesc{
		Name: "k", DType: accel.F32, Shape: tensor.Shape{tokens, heads, keyDim},
	})
	v := tensor.Input(b, tensor.ValueDesc{
		Name: "v", DType: accel.F32, Shape: tensor.Shape{tokens, heads, valueDim},
	})
	st := tensor.NewState(b, tensor.StateDesc{
		Name: "state", DType: accel.F32,
		Shape: tensor.Shape{batch, heads, valueDim, keyDim},
	})
	o, _ := tensor.LinearAttention(b, q, k, v, st, tensor.LinearOptions{
		// One per sequence, where the operator needs one per token.
		Alpha: tensor.Input(b, tensor.ValueDesc{
			Name: "alpha", DType: accel.F32, Shape: tensor.Shape{batch},
		}),
		Beta: tensor.Input(b, tensor.ValueDesc{
			Name: "beta", DType: accel.F32, Shape: tensor.Shape{tokens},
		}),
		QueryExtents: tensor.Input(b, tensor.ValueDesc{
			Name: "extents", DType: accel.U32, Shape: tensor.Shape{batch},
		}),
	})
	tensor.Output(b, "out", o)
	_, err := b.Compile(rt, tensor.CompileOptions{Label: "gate"})
	if err == nil {
		t.Fatal("a gate with one entry per sequence was accepted")
	}
	if !strings.Contains(err.Error(), "per token") {
		t.Fatalf("refused with %q, which does not say a gate is per token", err)
	}
}
