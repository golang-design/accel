// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"fmt"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// LinearOptions configures a gated delta layer.
//
// specs/047-linear-attention.md. No field is a scalar, which is
// specs/043-per-row-values.md §2: a gate differs per row and an extent differs
// per sequence.
type LinearOptions struct {
	// Alpha and Beta are f32 gates, in the flat token order q is in. Alpha
	// decays the state and Beta writes into it: the recurrence is
	// S <- alpha*S + beta*k*(v - S k)^T.
	//
	// The **rank** says which row the gate is per, and both are real:
	//
	//	[tokens]         one gate per token, shared by every head
	//	[tokens, heads]  one gate per head, which is what a gated delta
	//	                 network publishes -- the projection producing them
	//	                 is 2*num_value_heads wide, and heads forgetting at
	//	                 different rates is the point of the gate
	//
	// Element count does not decide it: a [tokens*heads] gate written as rank
	// 1 holds the right number of floats and says nothing, so it is refused.
	// Alpha and Beta share a layout, because every model that produces one
	// produces the other from the same projection (accel issue 27).
	Alpha *Tensor
	Beta  *Tensor

	// QueryExtents is a u32 tensor holding, per sequence, how many tokens that
	// sequence contributes to this step. The same segmented extent
	// [AttentionOptions.QueryExtents] takes, and for the same reason: a decode
	// step is one token per sequence, a prefill is many, and a mixed step is
	// both at once.
	//
	// A count of zero is legal. specs/046-segmented-extents.md §1.
	QueryExtents *Tensor
}

// gateReason says which gate layout a plan was built for, so the two are
// distinguishable in a Plan's reasons and not only in its digest.
func gateReason(gateHeads int) string {
	if gateHeads == 1 {
		return "one gate per token, shared by every head"
	}
	return fmt.Sprintf("a gate per token per head, %d of them", gateHeads)
}

// LinearAttention steps a gated delta recurrence and returns this step's output.
//
// # What it is, and why it is not Attention
//
// A linear-attention layer carries a **matrix per sequence per head** rather
// than a cache per position:
//
//	S_t = S_{t-1}(alpha_t I - beta_t k_t k_t^T) + beta_t v_t k_t^T,   o_t = S_t q_t
//
// The state does not grow with context, which is the entire appeal — a
// 262K-context model has no KV cache for these layers — and it is also why
// nothing here takes Lengths or Pages: there are no positions to address.
//
// # The state is an ordinary State
//
// [tensor.State] with shape [slots, heads, valueDim, keyDim]: its leading axis
// is the sequence slot rather than a position, which specs/043-per-row-values.md
// §9's correction established. So a hybrid model — three of these layers for
// every softmax one — holds two States of different shapes in one graph and
// needs nothing else.
//
// # The recurrence is sequential, and that is visible in the cost
//
// Token t needs the state token t-1 left behind, so this is a scan rather than
// a batch of independent rows. The parallelism is over sequences and heads.
// specs/047-linear-attention.md §4 records the chunked parallel form as
// deliberately not built: it is what makes the layer fast rather than
// expressible, and this kernel is the reference it would be checked against.
// # It returns two things, because it writes two
//
// The step's output *and* the next version of the state. A recurrence reads the
// state and writes it in one kernel, so unlike a KV cache -- where ScatterRows
// writes and Attention reads, as two nodes -- there is no way to separate them.
// Returning the version is what lets a later operator say which contents it
// meant, and specs/007-tensor-layer.md makes that distinction meaningful rather
// than decorative: an operator handed the version from *before* this step reads
// what was there before it.
func LinearAttention(b *Builder, q, k, v *Tensor, s *State, o LinearOptions) (*Tensor, *State) {
	fail := func(format string, args ...any) (*Tensor, *State) {
		b.fail(2, "LinearAttention", format, args...)
		return b.poison(), &State{b: b, poison: true, producer: -1}
	}
	if poisoned(q, k, v) || s == nil || s.poison {
		return b.poison(), &State{b: b, poison: true, producer: -1}
	}
	if o.QueryExtents == nil || o.Alpha == nil || o.Beta == nil {
		return fail("QueryExtents, Alpha and Beta are all " +
			"required: the extent says which tokens belong to which sequence, and the " +
			"two gates are per token (specs/047-linear-attention.md)")
	}
	for _, c := range []struct {
		name string
		t    *Tensor
		want DType
	}{
		{"q", q, accel.F32}, {"k", k, accel.F32}, {"v", v, accel.F32},
		{"Alpha", o.Alpha, accel.F32}, {"Beta", o.Beta, accel.F32},
		{"QueryExtents", o.QueryExtents, accel.U32},
	} {
		if c.t.dtype != c.want {
			return fail("%s is %v and this kernel reads %v",
				c.name, c.t.dtype, c.want)
		}
	}
	if len(q.shape) != 3 || len(k.shape) != 3 || len(v.shape) != 3 {
		return fail("q is %v, k is %v and v is %v; each is "+
			"[tokens, heads, dim] with the tokens of every sequence end to end",
			q.shape, k.shape, v.shape)
	}
	tokens, heads, keyDim := q.shape[0], q.shape[1], q.shape[2]
	if !k.shape.Equal(q.shape) {
		return fail("q is %v and k is %v; they contract against "+
			"the same axis of the state, so they share a width", q.shape, k.shape)
	}
	if v.shape[0] != tokens || v.shape[1] != heads {
		return fail("q is %v and v is %v; they are the same "+
			"tokens and the same heads and differ only in width", q.shape, v.shape)
	}
	valueDim := v.shape[2]

	batch := o.QueryExtents.shape.Elements()
	if batch == 0 {
		return fail("QueryExtents is empty; it is one count per " +
			"sequence and a step has at least one sequence")
	}
	// The gate's **rank** says which layout it is, not its element count. A
	// [tokens*heads] gate flattened to rank 1 holds the right number of floats
	// and means nothing, so it stays refused; only a written second axis says
	// the caller meant one gate per head.
	gateHeads := 0
	for _, g := range []struct {
		name string
		t    *Tensor
	}{{"Alpha", o.Alpha}, {"Beta", o.Beta}} {
		var want int
		switch len(g.t.shape) {
		case 1:
			want = 1
		case 2:
			want = g.t.shape[1]
			if want != heads && want != 1 {
				return fail("%s is %v and this step has %d head(s); "+
					"the second axis of a gate is 1 or the head count",
					g.name, g.t.shape, heads)
			}
		default:
			want = -1
		}
		if want < 0 || g.t.shape[0] != tokens {
			return fail("%s is %v and this step has %d token(s) "+
				"of %d head(s). A gate is per token, shaped [tokens]; or per token "+
				"and head, shaped [tokens, heads], which is what a gated delta network "+
				"publishes because heads forget at different rates. The per-sequence "+
				"value is QueryExtents (accel issue 27)",
				g.name, g.t.shape, tokens, heads)
		}
		if gateHeads == 0 {
			gateHeads = want
		} else if gateHeads != want {
			return fail("Alpha is %v and Beta is %v; the two gates "+
				"share a layout, because every model that produces one produces the "+
				"other from the same projection (specs/047-linear-attention.md section 6)",
				o.Alpha.shape, o.Beta.shape)
		}
	}

	// The state's shape is what says which axis is which, so it is checked
	// rather than assumed: [slots, heads, valueDim, keyDim]. Transposing the
	// last two is the mistake specs/047-linear-attention.md §1 derives the
	// dimensions to prevent, and at valueDim == keyDim it would still run.
	want := Shape{batch, heads, valueDim, keyDim}
	if !s.shape.Equal(want) {
		return fail("the state is %v and this step needs %v: "+
			"[slots, heads, valueDim, keyDim], where valueDim is v's width and keyDim "+
			"is q's. Transposing the last two runs when they are equal and is wrong "+
			"when they are not (specs/047-linear-attention.md section 1)",
			s.shape, want)
	}
	if s.desc.DType != accel.F32 {
		return fail("the state is %v and this kernel reads f32",
			s.desc.DType)
	}
	if stale(b, s) {
		return fail("%s", staleMessage(b, s))
	}

	offsets := b.segmentOffsets(o.QueryExtents, batch)
	if poisoned(offsets) {
		return b.poison(), &State{b: b, poison: true, producer: -1}
	}

	// The state is both read and written, so the node writes into its port and
	// the output is the step's o. Two things the node cannot express at once,
	// which is why the state is a port rather than an operand: a tensor is an
	// immutable value and this recurrence is not.
	out := b.record(node{
		op:     "LinearAttention",
		inputs: []*Tensor{q, k, v, o.Alpha, o.Beta, offsets, readState(b, s)},
		kernel: &testkernels.LinearAttentionKernel,
		uniform: func(map[string]ScalarValue) any {
			return testkernels.LinearDims{
				Batch: uint32(batch), Heads: uint32(heads),
				KeyDim: uint32(keyDim), ValueDim: uint32(valueDim),
				GateHeads: uint32(gateHeads),
			}
		},
		grid: func(*Tensor) accel.WorkgroupCount {
			// One workgroup per (sequence, head). Not per token: the tokens of
			// one sequence are not independent, which is the whole difference
			// between this and softmax attention.
			return accel.WorkgroupCount{X: batch * heads}
		},
		// The gate layout is read through the uniform, so it is recorded here
		// too. Alpha's shape already differs between the two layouts and the
		// digest covers operand shapes, but a value a kernel reads and the key
		// infers is the shape of defect specs/009-sequencing.md records twice.
		attrs: []uint64{uint64(gateHeads)},
		reason: fmt.Sprintf("the gated delta scan: %d sequences of %d heads, each "+
			"walking its own tokens with a [%d, %d] state and %s",
			batch, heads, valueDim, keyDim, gateReason(gateHeads)),
		rejected: []string{"a chunked parallel form: it reassociates the recurrence, " +
			"which is faster and is a different summation order, so it needs its own " +
			"numeric bound derived against this one (specs/047-linear-attention.md section 4)"},
	}, accel.F32, Shape{tokens, heads, valueDim})

	// The state advanced, so the version does. Mirrors what a write into a
	// port does elsewhere; the difference is that this node's *output* is the
	// step's o rather than the state, so the version is advanced beside it
	// rather than derived from it.
	next := *s
	next.version = s.version + 1
	next.producer = out.node
	if b.stateVersion == nil {
		b.stateVersion = map[window]int{}
	}
	b.stateVersion[s.window()] = next.version
	return out, &next
}
