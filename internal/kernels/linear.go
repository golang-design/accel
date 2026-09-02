// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import "golang.design/x/accel"

// LinearDims is a gated delta layer's shape.
type LinearDims struct {
	// Batch is how many sequences contribute to this step, and Heads how many
	// key/value head pairs each carries.
	Batch uint32
	Heads uint32

	// KeyDim is the width of k and q; ValueDim the width of v and o. The state
	// is [ValueDim, KeyDim] per head, which specs/047-linear-attention.md §1
	// derives rather than asserts.
	KeyDim   uint32
	ValueDim uint32

	// GateHeads is how many gates a token carries: 1 when every head shares
	// one, Heads when each head has its own. It is a count rather than a flag
	// because it is also the stride between one token's gates and the next.
	//
	// Both are real. A per-head decay is what a gated delta network publishes
	// -- the projection producing it is 2*num_value_heads wide -- and a shared
	// gate is what a model with one decay per token needs. Which one a step is
	// doing is the rank of alpha (accel issue 27).
	//
	// Zero is not a value. Like Heads it is a divisor, so a LinearDims left
	// zero here is not a shape this kernel has a reading of.
	GateHeads uint32
}

// LinearAttention steps a gated delta recurrence over a segmented extent.
//
// specs/047-linear-attention.md. Each sequence carries a matrix state per head
// and walks its own tokens in order:
//
//	u = S k
//	S = alpha*S + beta * k (v-u)^T
//	o = S q
//
// # Why a scan and not a batch
//
// Token t needs the state token t-1 left behind, so the tokens of one sequence
// are not independent and cannot be spread across workgroups. The parallelism
// is over sequences and heads instead, and each workgroup walks its own tokens.
// That is the one structural difference from softmax attention, where every
// query is independent and the tokens are exactly what gets parallelised.
//
// # A decode and a prefill are this same kernel
//
// The tokens arrive as specs/046-segmented-extents.md's segmented extent, so a
// sequence contributing one token is a decode step, a sequence contributing T
// is a prefill, and a step with both is a mixed one. Nothing here tests which:
// the loop runs offsets[seq] to offsets[seq+1] whatever that range is.
//
// # Why S is read and written per token
//
// One head's state at the widths a real model uses is 128x128 floats, which is
// 64 KiB and past what a workgroup can hold. So it stays in the caller's buffer
// and each token reads it, updates it, and reads it again -- which is why §1's
// arithmetic is three passes over KeyDim*ValueDim rather than one.
//
//accel:kernel workgroup=128
func LinearAttention(t accel.Thread, d LinearDims, q []float32, k []float32,
	v []float32, alpha []float32, beta []float32, offsets []uint32,
	state []float32, out []float32) {

	group := t.GroupID().X
	lane := t.LocalID().X

	// One workgroup per (sequence, head). Sequence-major, so one sequence's
	// heads are adjacent and walk the same token range.
	seq := group / d.Heads
	h := group % d.Heads

	// Lane a owns row a of the state: KeyDim floats. A lane past ValueDim has
	// no row and does nothing, which is what lets the workgroup be a fixed
	// width while ValueDim is not.
	a := lane

	// Where this sequence's state lives, and where its tokens are.
	sBase := (seq*d.Heads + h) * d.ValueDim * d.KeyDim
	first := offsets[seq]
	last := offsets[seq+1]

	// Which of this token's gates this head reads. GateHeads is 1 or Heads, so
	// the modulo pins every head to gate 0 in the first case and is the
	// identity in the second -- one index, no branch, and no separate kernel
	// for a shared gate.
	gh := h % d.GateHeads

	for tok := first; tok < last; tok++ {
		if a >= d.ValueDim {
			continue
		}
		row := sBase + a*d.KeyDim
		qBase := (tok*d.Heads + h) * d.KeyDim
		vBase := (tok*d.Heads + h) * d.ValueDim
		gate := tok*d.GateHeads + gh

		// Pass one: u = S k, this lane's element of it. Every lane reads the
		// whole of k and its own row of S, so nothing is shared and there is
		// no barrier in the recurrence at all.
		u := float32(0)
		for b := uint32(0); b < d.KeyDim; b++ {
			u = u + state[row+b]*k[qBase+b]
		}

		// Pass two: the rank-one update. Written as beta*k[b]*(v-u) rather than
		// as a subtraction of two products, because that is §1's expansion and
		// the two spellings round differently -- the reference is written from
		// the same expression for that reason.
		g := beta[gate] * (v[vBase+a] - u)
		al := alpha[gate]
		for b := uint32(0); b < d.KeyDim; b++ {
			state[row+b] = al*state[row+b] + g*k[qBase+b]
		}

		// Pass three: o = S q, over the state this token just left behind.
		o := float32(0)
		for b := uint32(0); b < d.KeyDim; b++ {
			o = o + state[row+b]*q[qBase+b]
		}
		out[vBase+a] = o
	}
}
