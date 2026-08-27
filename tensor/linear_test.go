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

	// gateHeads is how many gates a token carries, or zero for the rank-1
	// shape where every head shares one. Zero rather than one, so every case
	// written before the per-head layout existed still builds what it built.
	gateHeads int
}

// gateShape is the shape of Alpha and Beta: [tokens] or [tokens, heads].
func (c linearCase) gateShape() tensor.Shape {
	if c.gateHeads == 0 {
		return tensor.Shape{c.tokens()}
	}
	return tensor.Shape{c.tokens(), c.gateHeads}
}

// gates is how many f32 entries that shape holds.
func (c linearCase) gates() int { return c.gateShape().Elements() }

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
		Name: "alpha", DType: accel.F32, Shape: c.gateShape(),
	})
	beta := tensor.Input(b, tensor.ValueDesc{
		Name: "beta", DType: accel.F32, Shape: c.gateShape(),
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
	return &linearPlan{plan: plan, dev: d, stateBuf: stateBuf, out: out, c: c}
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
	alpha = make([]float32, c.gates())
	beta = make([]float32, len(alpha))
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

// A per-head gate reaches the kernel through the operator.
//
// accel issue 27. The state is per head and every term of the recurrence
// carries a head index except the two gates, so a gated delta network -- where
// the projection producing them is 2*num_value_heads wide and the whole point
// is that heads forget at different rates -- was inexpressible. The consumer's
// alternatives were 48 dispatches per layer or a model that is not the model.
//
// # What the two halves check
//
// A `[tokens, heads]` gate holding one value everywhere must equal the
// `[tokens]` gate holding it, which says the wider layout did not become a
// second operator. Then two heads with opposite gates must disagree, which is
// what says the head index is read: a kernel that took alpha[tok] and ignored
// the head passes the first half exactly.
func TestALinearPerHeadGateReachesTheKernel(t *testing.T) {
	const heads = 2
	shared := linearCase{
		batch: 2, heads: heads, keyDim: 4, valueDim: 4, counts: []uint32{2, 1},
	}
	perHead := shared
	perHead.gateHeads = heads
	tokens := shared.tokens()

	q, k, v, alpha, beta := linearInputs(shared, 3)
	sharedOut := buildLinear(t, shared).step(t, q, k, v, alpha, beta, shared.counts)

	// The same gate, written out per head.
	wide, wideBeta := make([]float32, tokens*heads), make([]float32, tokens*heads)
	for tok := range tokens {
		for h := range heads {
			wide[tok*heads+h], wideBeta[tok*heads+h] = alpha[tok], beta[tok]
		}
	}
	wideOut := buildLinear(t, perHead).step(t, q, k, v, wide, wideBeta, shared.counts)
	for i := range sharedOut {
		if sharedOut[i] != wideOut[i] {
			t.Fatalf("output %d is %v with one gate per token and %v with that gate "+
				"copied to every head", i, sharedOut[i], wideOut[i])
		}
	}

	// Now the heads disagree: head 0 keeps its state and writes nothing, head 1
	// forgets everything. A kernel reading one gate for both cannot produce two
	// different answers here.
	for tok := range tokens {
		wide[tok*heads+0], wideBeta[tok*heads+0] = 1, 0
		wide[tok*heads+1], wideBeta[tok*heads+1] = 0, 0.5
	}
	split := buildLinear(t, perHead).step(t, q, k, v, wide, wideBeta, shared.counts)

	// Compare the two heads of one token. They read the same state -- a fresh
	// zero buffer -- and differ only in their gate, so an equal pair says the
	// gate did not reach the kernel per head.
	same := true
	for a := range shared.valueDim {
		h0 := split[(0*heads+0)*shared.valueDim+a]
		h1 := split[(0*heads+1)*shared.valueDim+a]
		if h0 != h1 {
			same = false
			break
		}
	}
	if same {
		t.Fatal("head 0 had alpha=1, beta=0 and head 1 had alpha=0, beta=0.5, and the " +
			"two produced the same output: the kernel read one gate for both heads")
	}
}

// The two gate layouts do not share a compiled plan.
//
// A plan cached under a key that ignores the layout would answer a per-head
// request with a shared-gate plan, which specs/009-sequencing.md records twice
// as the shape of defect a digest misses.
func TestTheGateLayoutReachesThePlanKey(t *testing.T) {
	rt := newRuntime(t)
	id := func(gateHeads int) tensor.Identity {
		c := linearCase{
			batch: 1, heads: 2, keyDim: 4, valueDim: 4, counts: []uint32{2},
			gateHeads: gateHeads,
		}
		b := rt.NewBuilder("gatekey")
		tokens := c.tokens()
		mk := func(name string, s tensor.Shape) *tensor.Tensor {
			return tensor.Input(b, tensor.ValueDesc{Name: name, DType: accel.F32, Shape: s})
		}
		st := tensor.NewState(b, tensor.StateDesc{
			Name: "state", DType: accel.F32,
			Shape: tensor.Shape{c.batch, c.heads, c.valueDim, c.keyDim},
		})
		o, _ := tensor.LinearAttention(b,
			mk("q", tensor.Shape{tokens, c.heads, c.keyDim}),
			mk("k", tensor.Shape{tokens, c.heads, c.keyDim}),
			mk("v", tensor.Shape{tokens, c.heads, c.valueDim}), st,
			tensor.LinearOptions{
				Alpha: mk("alpha", c.gateShape()), Beta: mk("beta", c.gateShape()),
				QueryExtents: tensor.Input(b, tensor.ValueDesc{
					Name: "ext", DType: accel.U32, Shape: tensor.Shape{c.batch},
				}),
			})
		tensor.Output(b, "out", o)
		if err := b.Err(); err != nil {
			t.Fatalf("gateHeads %d does not build: %v", gateHeads, err)
		}
		return b.Identity()
	}
	if id(0) == id(2) {
		t.Fatal("a [tokens] gate and a [tokens, heads] gate hash to one plan key, so a " +
			"cache would serve the first plan compiled for either")
	}
}

// The dtypes and shapes a gated delta layer refuses.
//
// Every field is required, so "accepted and ignored" cannot happen here -- what
// these cover is the caller who supplied all three and got one wrong.
func TestALinearStepRefusesWrongTypesAndShapes(t *testing.T) {
	// More than one head, because the gate rows below are exactly the cases a
	// single head cannot tell apart: at heads == 1 a flattened [tokens*heads]
	// gate *is* [tokens], and two gates of different rank agree on the layout.
	const batch, heads, keyDim, valueDim, tokens = 2, 2, 4, 4, 3

	type parts struct {
		q, k, v, alpha, beta, ext tensor.ValueDesc
		state                     tensor.Shape
		omit                      string
	}
	good := parts{
		q:     tensor.ValueDesc{Name: "q", DType: accel.F32, Shape: tensor.Shape{tokens, heads, keyDim}},
		k:     tensor.ValueDesc{Name: "k", DType: accel.F32, Shape: tensor.Shape{tokens, heads, keyDim}},
		v:     tensor.ValueDesc{Name: "v", DType: accel.F32, Shape: tensor.Shape{tokens, heads, valueDim}},
		alpha: tensor.ValueDesc{Name: "alpha", DType: accel.F32, Shape: tensor.Shape{tokens}},
		beta:  tensor.ValueDesc{Name: "beta", DType: accel.F32, Shape: tensor.Shape{tokens}},
		ext:   tensor.ValueDesc{Name: "ext", DType: accel.U32, Shape: tensor.Shape{batch}},
		state: tensor.Shape{batch, heads, valueDim, keyDim},
	}
	for _, c := range []struct {
		name string
		mut  func(*parts)
		want string
	}{
		{"no extents", func(p *parts) { p.omit = "ext" }, "all"},
		{"f16 queries", func(p *parts) { p.q.DType = accel.F16 }, "reads f32"},
		{"extents that are not u32", func(p *parts) { p.ext.DType = accel.F32 }, "reads u32"},
		{"k of a different width", func(p *parts) {
			p.k.Shape = tensor.Shape{tokens, heads, keyDim + 2}
		}, "share a width"},
		{"v with a different token count", func(p *parts) {
			p.v.Shape = tensor.Shape{tokens + 1, heads, valueDim}
		}, "same tokens"},
		{"q that is not [tokens, heads, dim]", func(p *parts) {
			p.q.Shape = tensor.Shape{tokens, keyDim}
			p.k.Shape = tensor.Shape{tokens, keyDim}
			p.v.Shape = tensor.Shape{tokens, valueDim}
		}, "end to end"},

		// The per-head layout's own refusals (accel issue 27). The first is why
		// the check is on rank and not on Elements: this holds exactly as many
		// floats as a [tokens, heads] gate and says nothing about which axis is
		// which, so accepting it would make a caller's transposed gate run.
		{"a flattened per-head gate", func(p *parts) {
			p.alpha.Shape = tensor.Shape{tokens * heads}
			p.beta.Shape = tensor.Shape{tokens * heads}
		}, "per token"},
		{"a gate whose second axis is not the head count", func(p *parts) {
			p.alpha.Shape = tensor.Shape{tokens, heads + 1}
			p.beta.Shape = tensor.Shape{tokens, heads + 1}
		}, "head count"},
		{"a gate of rank three", func(p *parts) {
			p.alpha.Shape = tensor.Shape{tokens, heads, 1}
			p.beta.Shape = tensor.Shape{tokens, heads, 1}
		}, "per token"},
		{"two gates of different layouts", func(p *parts) {
			p.alpha.Shape = tensor.Shape{tokens, heads}
		}, "share a layout"},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := good
			c.mut(&p)
			rt := newRuntime(t)
			b := rt.NewBuilder("refusal")
			mk := func(d tensor.ValueDesc) *tensor.Tensor {
				if p.omit == d.Name {
					return nil
				}
				return tensor.Input(b, d)
			}
			st := tensor.NewState(b, tensor.StateDesc{
				Name: "state", DType: accel.F32, Shape: p.state,
			})
			o, _ := tensor.LinearAttention(b, mk(p.q), mk(p.k), mk(p.v), st,
				tensor.LinearOptions{
					Alpha: mk(p.alpha), Beta: mk(p.beta), QueryExtents: mk(p.ext),
				})
			tensor.Output(b, "out", o)
			_, err := b.Compile(rt, tensor.CompileOptions{Label: "refusal"})
			if err == nil {
				t.Fatal("accepted")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Fatalf("refused with %q, which does not mention %q", err, c.want)
			}
		})
	}
}
