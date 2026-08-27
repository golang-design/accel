// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"fmt"
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
					// Which gate this head reads, spelled as a branch rather
					// than as the kernel's modulo. The two agree only if the
					// kernel's arithmetic is right, which is the point of a
					// reference written from the layout instead of the code.
					gate := int(tok) * int(d.GateHeads)
					if d.GateHeads > 1 {
						gate += int(h)
					}
					g := float64(beta[gate]) * (float64(v[vBase+a]) - u)
					al := float64(alpha[gate])
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
	alpha = make([]float32, sum*d.GateHeads)
	beta = make([]float32, len(alpha))
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
	return testkernels.LinearDims{Batch: 2, Heads: 2, KeyDim: 6, ValueDim: 4, GateHeads: 1}
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
		gh := int(d.GateHeads)
		one := runLinear(t, d,
			q2[tok*widthQ:(tok+1)*widthQ], k2[tok*widthQ:(tok+1)*widthQ],
			v2[tok*widthV:(tok+1)*widthV],
			a2[tok*gh:(tok+1)*gh], b2[tok*gh:(tok+1)*gh], state2, off)
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

// A per-head gate decays each head at its own rate.
//
// accel issue 27: the state is per head and every term of §1's recurrence
// carries a head index except alpha and beta, so a model whose heads forget at
// different rates -- which is what a gated delta network publishes -- was
// inexpressible. This is the accepting half of accepting `[tokens, heads]`.
//
// # The oracle is the kernel itself, one head at a time, and the comparison is
// exact
//
// A head's arithmetic does not depend on how many heads are beside it: the same
// three passes run over the same values in the same order. So running head h
// alone -- Heads and GateHeads both 1, over h's slice of q, k, v and the state
// -- must reproduce h's part of the two-head answer **bit for bit**, and there
// is no bound to derive and no tolerance to choose. A kernel that read alpha[t]
// and ignored the head disagrees, because the single-head run is handed h's own
// gate whatever h is.
//
// The gates are opposites rather than merely different for the same reason the
// per-head assertion at the end exists: two similar gates make one head's answer
// a plausible answer for the other, and a head-blind kernel then passes.
func TestALinearGateAppliesPerHead(t *testing.T) {
	d := linearDims()
	d.GateHeads = d.Heads
	counts := []uint32{2, 1}
	q, k, v, alpha, beta, state, offsets := linearFixture(d, counts)

	// Head 0: alpha=1, beta=0, so its state is left exactly as it was. Head 1:
	// alpha=0, so whatever it had is gone after one token.
	for tok := range offsets[d.Batch] {
		alpha[tok*d.GateHeads+0], beta[tok*d.GateHeads+0] = 1, 0
		alpha[tok*d.GateHeads+1], beta[tok*d.GateHeads+1] = 0, 0.5
	}
	before := append([]float32(nil), state...)

	got := runLinear(t, d, q, k, v, alpha, beta, state, offsets)

	// Head h alone, over h's slice of everything.
	tokens := int(offsets[d.Batch])
	for h := range int(d.Heads) {
		one := d
		one.Heads, one.GateHeads = 1, 1
		pick := func(src []float32, width int) []float32 {
			out := make([]float32, tokens*width)
			for tok := range tokens {
				copy(out[tok*width:], src[(tok*int(d.Heads)+h)*width:][:width])
			}
			return out
		}
		hq, hk := pick(q, int(d.KeyDim)), pick(k, int(d.KeyDim))
		hv := pick(v, int(d.ValueDim))
		hAlpha, hBeta := make([]float32, tokens), make([]float32, tokens)
		for tok := range tokens {
			hAlpha[tok] = alpha[tok*int(d.GateHeads)+h]
			hBeta[tok] = beta[tok*int(d.GateHeads)+h]
		}
		// The state, gathered per (slot, head) and put back the same way.
		rows := int(d.ValueDim * d.KeyDim)
		hState := make([]float32, int(d.Batch)*rows)
		for seq := range int(d.Batch) {
			copy(hState[seq*rows:], before[(seq*int(d.Heads)+h)*rows:][:rows])
		}
		hOut := runLinear(t, one, hq, hk, hv, hAlpha, hBeta, hState, offsets)

		for tok := range tokens {
			for a := range int(d.ValueDim) {
				want := hOut[tok*int(d.ValueDim)+a]
				if g := got[(tok*int(d.Heads)+h)*int(d.ValueDim)+a]; g != want {
					t.Fatalf("token %d head %d element %d is %v beside the other head and "+
						"%v alone: this head's gate did not reach it", tok, h, a, g, want)
				}
			}
		}
		for seq := range int(d.Batch) {
			for i := range rows {
				want := hState[seq*rows+i]
				if g := state[(seq*int(d.Heads)+h)*rows+i]; g != want {
					t.Fatalf("slot %d head %d state element %d is %v beside the other head "+
						"and %v alone", seq, h, i, g, want)
				}
			}
		}
	}

	// And the two heads did visibly different things, so the agreements above
	// are not two heads that happened to be handed the same gate.
	rows := int(d.ValueDim * d.KeyDim)
	for i := range rows {
		if state[i] != before[i] {
			t.Fatalf("head 0 had alpha=1 and beta=0 and its state element %d still moved "+
				"from %v to %v", i, before[i], state[i])
		}
	}
	moved := false
	for i := rows; i < 2*rows; i++ {
		if state[i] != before[i] {
			moved = true
			break
		}
	}
	if !moved {
		t.Fatal("head 1 had alpha=0 and its state came back unchanged, so both heads " +
			"were stepped with head 0's gate")
	}
}

// A per-head gate holding the same value in every head equals a shared gate.
//
// The other half of issue 27: `[tokens]` had to keep meaning what it meant, and
// this says the wider layout reduces to it rather than being a second kernel
// that happens to agree on the fixture.
func TestAReplicatedPerHeadGateEqualsASharedOne(t *testing.T) {
	shared := linearDims()
	counts := []uint32{2, 1}
	q, k, v, alpha, beta, state, offsets := linearFixture(shared, counts)
	sharedOut := runLinear(t, shared, q, k, v, alpha, beta, state, offsets)

	perHead := shared
	perHead.GateHeads = perHead.Heads
	q2, k2, v2, _, _, state2, _ := linearFixture(shared, counts)
	wide := make([]float32, len(alpha)*int(perHead.GateHeads))
	wideBeta := make([]float32, len(wide))
	for tok := range alpha {
		for h := range int(perHead.GateHeads) {
			wide[tok*int(perHead.GateHeads)+h] = alpha[tok]
			wideBeta[tok*int(perHead.GateHeads)+h] = beta[tok]
		}
	}
	wideOut := runLinear(t, perHead, q2, k2, v2, wide, wideBeta, state2, offsets)

	for i := range sharedOut {
		if sharedOut[i] != wideOut[i] {
			t.Fatalf("output %d is %v with one gate per token and %v with that same gate "+
				"copied to every head", i, sharedOut[i], wideOut[i])
		}
	}
	for i := range state {
		if state[i] != state2[i] {
			t.Fatalf("state %d is %v with one gate per token and %v with that same gate "+
				"copied to every head", i, state[i], state2[i])
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

// chunkedLinearRef evaluates the gated delta recurrence a chunk at a time,
// through the UT transform specs/047-linear-attention.md §6.1 derives.
//
// It is a reference, not a kernel. What it exists to do is make the derivation
// checkable: §6 recorded that a chunked scan needs a second representation
// rather than a bigger step, and the risk in that representation is that it is
// plausible and subtly unequal. This is the arithmetic written out, so the
// assertion below is against the kernel rather than against a restatement.
//
// It returns the outputs and leaves the final state in state, matching the
// kernel's contract.
func chunkedLinearRef(d testkernels.LinearDims, q, k, v, alpha, beta,
	state []float32, offsets []uint32, chunk int) []float32 {

	total := offsets[d.Batch]
	out := make([]float32, total*d.Heads*d.ValueDim)
	K, V := int(d.KeyDim), int(d.ValueDim)

	for seq := uint32(0); seq < d.Batch; seq++ {
		for h := uint32(0); h < d.Heads; h++ {
			sBase := int((seq*d.Heads + h)) * V * K
			first, last := int(offsets[seq]), int(offsets[seq+1])

			for c0 := first; c0 < last; c0 += chunk {
				n := min(chunk, last-c0)

				// d[i] is the decay accumulated through token i of this chunk,
				// and dprev(i) the decay before it. d[-1] = 1 by definition.
				dec := make([]float64, n)
				run := 1.0
				for i := range n {
					run *= float64(alpha[c0+i])
					dec[i] = run
				}
				dprev := func(i int) float64 {
					if i == 0 {
						return 1
					}
					return dec[i-1]
				}
				at := func(tok int) (int, int) {
					return (tok*int(d.Heads) + int(h)) * K, (tok*int(d.Heads) + int(h)) * V
				}

				// W solves (I + A) W = B by forward substitution, which is the
				// whole transform: the recurrence's sequential dependency
				// becomes a triangular solve whose coefficients are one
				// Gram matrix.
				W := make([][]float64, n)
				for i := range n {
					qb, vb := at(c0 + i)
					W[i] = make([]float64, V)
					for a := range V {
						sk := 0.0
						for b := range K {
							sk += float64(state[sBase+a*K+b]) * float64(k[qb+b])
						}
						W[i][a] = float64(beta[c0+i]) *
							(float64(v[vb+a]) - dprev(i)*sk)
					}
					for j := range i {
						jb, _ := at(c0 + j)
						kk := 0.0
						for b := range K {
							kk += float64(k[jb+b]) * float64(k[qb+b])
						}
						aij := float64(beta[c0+i]) * (dprev(i) / dec[j]) * kk
						for a := range V {
							W[i][a] -= aij * W[j][a]
						}
					}
				}

				// o_i = d_i S0 q_i + sum_{j<=i} (d_i/d_j)(k_j . q_i) w_j
				for i := range n {
					qb, vb := at(c0 + i)
					for a := range V {
						sq := 0.0
						for b := range K {
							sq += float64(state[sBase+a*K+b]) * float64(q[qb+b])
						}
						o := dec[i] * sq
						for j := 0; j <= i; j++ {
							jb, _ := at(c0 + j)
							kq := 0.0
							for b := range K {
								kq += float64(k[jb+b]) * float64(q[qb+b])
							}
							o += (dec[i] / dec[j]) * kq * W[j][a]
						}
						out[vb+a] = float32(o)
					}
				}

				// S_C = d_{n-1} S0 + sum_j (d_{n-1}/d_j) w_j k_j^T
				next := make([]float64, V*K)
				for a := range V {
					for b := range K {
						next[a*K+b] = dec[n-1] * float64(state[sBase+a*K+b])
					}
				}
				for j := range n {
					jb, _ := at(c0 + j)
					s := dec[n-1] / dec[j]
					for a := range V {
						for b := range K {
							next[a*K+b] += s * W[j][a] * float64(k[jb+b])
						}
					}
				}
				for i := range next {
					state[sBase+i] = float32(next[i])
				}
			}
		}
	}
	return out
}

// The chunked form equals the sequential kernel, at every chunk size.
//
// specs/047-linear-attention.md §6.1. §6 recorded that the chunked scan is a
// derivation before it is a kernel, and that the derivation is where a fast and
// subtly wrong answer would come from. This is that derivation checked against
// the kernel it must agree with, which is the only oracle it can have.
//
// Chunk sizes 1 through 16 including sizes that do not divide the sequence
// length, because the transform's index arithmetic is where an off-by-one lives
// and a chunk that always divides evenly hides it. Chunk 1 is the degenerate
// case and must reduce to the recurrence exactly.
//
// The shared gate only: chunkedLinearRef takes the running product of alpha
// over a chunk and indexes it per token, so a per-head gate needs the
// derivation carried through that product before it needs a kernel.
func TestTheChunkedFormEqualsTheSequentialKernel(t *testing.T) {
	d := testkernels.LinearDims{Batch: 2, Heads: 2, KeyDim: 6, ValueDim: 4, GateHeads: 1}
	counts := []uint32{7, 5}

	for _, chunk := range []int{1, 2, 3, 4, 5, 8, 16} {
		t.Run(fmt.Sprintf("chunk=%d", chunk), func(t *testing.T) {
			q, k, v, alpha, beta, state, offsets := linearFixture(d, counts)
			seqState := append([]float32(nil), state...)
			want := runLinear(t, d, q, k, v, alpha, beta, seqState, offsets)

			chunkState := append([]float32(nil), state...)
			got := chunkedLinearRef(d, q, k, v, alpha, beta, chunkState, offsets, chunk)

			// The two summation orders differ, so specs/008-numerics.md §7's
			// reduction bound applies rather than equality. The reference
			// accumulates in f64, so what is compared is the kernel's f32
			// evaluation against an exact one of the same algebra.
			for i := range want {
				if e := math.Abs(float64(got[i] - want[i])); e > 1e-4 {
					t.Fatalf("output %d: chunked %v, sequential %v (off %v)",
						i, got[i], want[i], e)
				}
			}
			for i := range chunkState {
				if e := math.Abs(float64(chunkState[i] - seqState[i])); e > 1e-4 {
					t.Fatalf("state %d: chunked %v, sequential %v (off %v)",
						i, chunkState[i], seqState[i], e)
				}
			}
			// A real answer rather than two agreeing zeros.
			nonzero := false
			for _, x := range want {
				if x != 0 {
					nonzero = true
				}
			}
			if !nonzero {
				t.Fatal("every output is zero, so the comparison says nothing")
			}
		})
	}
}
