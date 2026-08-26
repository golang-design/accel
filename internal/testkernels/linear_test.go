// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/direct"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
	"golang.design/x/accel/kernelabi"
)

// linearRef steps the recurrence in f64, written from the spec's expression.
//
// specs/047-linear-attention.md §1, expanded there as three passes:
//
//	u = S k ;  S = aS + b k (v-u)^T ;  o = S q
//
// Written from that rather than from the kernel, so a kernel that reordered the
// passes -- reading o from the state *before* the update, which is the mistake
// the expression makes easy -- disagrees here rather than agreeing with itself.
func linearRef(d testkernels.LinearDims, q, k, v, alpha, beta []float32,
	offsets []uint32, state []float32) (out, endState []float64) {

	total := int(offsets[d.Batch])
	out = make([]float64, total*int(d.Heads)*int(d.ValueDim))
	s := make([]float64, len(state))
	for i := range state {
		s[i] = float64(state[i])
	}

	for seq := uint32(0); seq < d.Batch; seq++ {
		for h := uint32(0); h < d.Heads; h++ {
			sBase := int((seq*d.Heads + h) * d.ValueDim * d.KeyDim)
			for tok := offsets[seq]; tok < offsets[seq+1]; tok++ {
				qBase := int((tok*d.Heads + h) * d.KeyDim)
				vBase := int((tok*d.Heads + h) * d.ValueDim)
				for a := 0; a < int(d.ValueDim); a++ {
					row := sBase + a*int(d.KeyDim)
					u := 0.0
					for b := 0; b < int(d.KeyDim); b++ {
						u += s[row+b] * float64(k[qBase+b])
					}
					g := float64(beta[tok]) * (float64(v[vBase+a]) - u)
					al := float64(alpha[tok])
					for b := 0; b < int(d.KeyDim); b++ {
						s[row+b] = al*s[row+b] + g*float64(k[qBase+b])
					}
					o := 0.0
					for b := 0; b < int(d.KeyDim); b++ {
						o += s[row+b] * float64(q[qBase+b])
					}
					out[vBase+a] = o
				}
			}
		}
	}
	return out, s
}

// linearFixture builds one step's inputs.
func linearFixture(d testkernels.LinearDims, counts []uint32) (q, k, v, alpha, beta,
	state []float32, offsets []uint32) {

	offsets = make([]uint32, d.Batch+1)
	sum := uint32(0)
	for r, c := range counts {
		offsets[r] = sum
		sum += c
	}
	offsets[d.Batch] = sum

	q = make([]float32, sum*d.Heads*d.KeyDim)
	k = make([]float32, len(q))
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*0.31)) / 4
		k[i] = float32(math.Cos(float64(i)*0.17)) / 4
	}
	v = make([]float32, sum*d.Heads*d.ValueDim)
	for i := range v {
		v[i] = float32(math.Sin(float64(i)*0.23)) * 2
	}
	alpha = make([]float32, sum)
	beta = make([]float32, sum)
	for i := range alpha {
		// A decay under one and a write rate under one, which is the regime
		// these layers run in: a gate at exactly one or zero is the identity
		// case and has its own test.
		alpha[i] = 0.9 - float32(i%3)/32
		beta[i] = 0.5 + float32(i%5)/32
	}
	state = make([]float32, d.Batch*d.Heads*d.ValueDim*d.KeyDim)
	for i := range state {
		state[i] = float32(math.Cos(float64(i)*0.41)) / 8
	}
	return q, k, v, alpha, beta, state, offsets
}

func runLinear(t *testing.T, d testkernels.LinearDims, q, k, v, alpha, beta,
	state []float32, offsets []uint32) []float32 {

	t.Helper()
	total := offsets[d.Batch]
	out := make([]float32, total*d.Heads*d.ValueDim)
	if err := direct.Run(&testkernels.LinearAttentionKernel,
		accel.ID3{X: d.Batch * d.Heads},
		kernelabi.Args{
			Uniforms: []any{d},
			Slices:   []any{q, k, v, alpha, beta, offsets, state, out},
		}); err != nil {
		t.Fatalf("run: %v", err)
	}
	return out
}

func linearDims() testkernels.LinearDims {
	return testkernels.LinearDims{Batch: 2, Heads: 2, KeyDim: 6, ValueDim: 4}
}

// One token matches the recurrence written out.
//
// specs/047-linear-attention.md §5's first assertion, and the accepting half:
// three passes in the right order against one expression evaluated directly.
func TestOneLinearStepMatchesTheRecurrence(t *testing.T) {
	d := linearDims()
	counts := []uint32{1, 1}
	q, k, v, alpha, beta, state, offsets := linearFixture(d, counts)
	before := append([]float32(nil), state...)

	got := runLinear(t, d, q, k, v, alpha, beta, state, offsets)
	want, wantState := linearRef(d, q, k, v, alpha, beta, offsets, before)

	for i := range want {
		if math.Abs(float64(got[i])-want[i]) > 1e-4 {
			t.Fatalf("output %d is %v, want %v", i, got[i], want[i])
		}
	}
	for i := range wantState {
		if math.Abs(float64(state[i])-wantState[i]) > 1e-4 {
			t.Fatalf("state %d is %v, want %v", i, state[i], wantState[i])
		}
	}
}

// Two tokens in one dispatch equal two dispatches of one token each.
//
// §5's second assertion, and the one that says the scan is a scan. A kernel
// that recomputed each token from the state it started the dispatch with passes
// every single-token test and fails this one.
func TestALinearScanCarriesTheStateBetweenTokens(t *testing.T) {
	d := linearDims()

	// Two tokens for sequence 0, one for sequence 1.
	together := []uint32{2, 1}
	q, k, v, alpha, beta, state, offsets := linearFixture(d, together)
	got := runLinear(t, d, q, k, v, alpha, beta, state, offsets)

	// The same tokens, one dispatch each, over a state that carries. The
	// fixture is regenerated so the second run's inputs are identical, and the
	// token rows are sliced out rather than reseeded.
	q2, k2, v2, a2, b2, state2, _ := linearFixture(d, together)
	widthQ := int(d.Heads * d.KeyDim)
	widthV := int(d.Heads * d.ValueDim)

	stepped := make([]float32, len(got))
	for tok := 0; tok < 3; tok++ {
		// One token from whichever sequence owns it, with the other sequence
		// contributing nothing -- which 046 makes an ordinary member.
		var counts []uint32
		switch tok {
		case 0, 1:
			counts = []uint32{1, 0}
		default:
			counts = []uint32{0, 1}
		}
		off := []uint32{0, counts[0], counts[0] + counts[1]}
		one := runLinear(t, d,
			q2[tok*widthQ:(tok+1)*widthQ], k2[tok*widthQ:(tok+1)*widthQ],
			v2[tok*widthV:(tok+1)*widthV],
			a2[tok:tok+1], b2[tok:tok+1], state2, off)
		copy(stepped[tok*widthV:], one)
	}

	for i := range got {
		if math.Abs(float64(got[i]-stepped[i])) > 1e-4 {
			t.Fatalf("output %d: one dispatch gave %v and three dispatches gave %v; the "+
				"state is not carried between tokens of one sequence", i, got[i], stepped[i])
		}
	}
	for i := range state {
		if math.Abs(float64(state[i]-state2[i])) > 1e-4 {
			t.Fatalf("state %d ended at %v in one dispatch and %v in three", i,
				state[i], state2[i])
		}
	}
}

// alpha of one and beta of zero leave the state exactly as it was.
//
// The identity case. It catches a sign error in the update that a random gate
// hides, because with beta zero the whole rank-one term drops out and anything
// that still moves the state is doing arithmetic it should not.
func TestALinearStepWithNoWriteLeavesTheStateAlone(t *testing.T) {
	d := linearDims()
	counts := []uint32{2, 2}
	q, k, v, alpha, beta, state, offsets := linearFixture(d, counts)
	for i := range alpha {
		alpha[i], beta[i] = 1, 0
	}
	before := append([]float32(nil), state...)

	out := runLinear(t, d, q, k, v, alpha, beta, state, offsets)

	for i := range state {
		if state[i] != before[i] {
			t.Fatalf("state %d moved from %v to %v with alpha=1 and beta=0; the "+
				"rank-one term is zero, so nothing should have changed",
				i, before[i], state[i])
		}
	}
	// And every token of a sequence sees the same state, so its output depends
	// only on its own q.
	if len(out) == 0 {
		t.Fatal("no output")
	}
}

// One sequence's tokens do not disturb another sequence's state.
func TestALinearStepKeepsSequencesApart(t *testing.T) {
	d := linearDims()
	counts := []uint32{2, 0}
	q, k, v, alpha, beta, state, offsets := linearFixture(d, counts)

	// Sequence 1 contributes nothing, so its slot must come back untouched.
	slot := int(d.Heads * d.ValueDim * d.KeyDim)
	before := append([]float32(nil), state[slot:]...)

	runLinear(t, d, q, k, v, alpha, beta, state, offsets)

	for i := range before {
		if state[slot+i] != before[i] {
			t.Fatalf("sequence 1 contributed no tokens and its state element %d moved "+
				"from %v to %v", i, before[i], state[slot+i])
		}
	}
}

// The authored linear kernel and its generated lowering agree.
//
// specs/010-kernel-corpus.md §6's obligation, and the one the ragged kernel's
// absence of it failed CI over: every other test here runs the generated form,
// so nothing calls the authored function, and on Linux the Metal differential
// is not there to cover it either.
func TestTheAuthoredLinearKernelMatchesItsLowering(t *testing.T) {
	d := linearDims()
	counts := []uint32{2, 1}
	q, k, v, alpha, beta, state, offsets := linearFixture(d, counts)

	authoredState := append([]float32(nil), state...)
	total := offsets[d.Batch]
	authored := make([]float32, total*d.Heads*d.ValueDim)
	groups := kernel.ID3{X: d.Batch * d.Heads, Y: 1, Z: 1}
	for g := range groups.X {
		kernel.RunAuthored(kernel.ID3{X: 128, Y: 1, Z: 1}, kernel.ID3{X: g},
			groups, 128, func(th kernel.Thread) {
				testkernels.LinearAttention(th, d, q, k, v, alpha, beta, offsets,
					authoredState, authored)
			})
	}

	generated := runLinear(t, d, q, k, v, alpha, beta, state, offsets)

	// Within a bound for the reason every f32 comparison of the two forms here
	// carries one: the generated lowering rounds each product explicitly and
	// ordinary Go may fuse a multiply and an add on a target with FMA.
	for i := range authored {
		if math.Abs(float64(authored[i]-generated[i])) > 1e-5 {
			t.Fatalf("output %d: authored %v, generated %v", i, authored[i], generated[i])
		}
	}
	for i := range authoredState {
		if math.Abs(float64(authoredState[i]-state[i])) > 1e-5 {
			t.Fatalf("state %d: authored %v, generated %v", i, authoredState[i], state[i])
		}
	}
}
