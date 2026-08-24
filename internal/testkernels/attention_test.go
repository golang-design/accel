// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"golang.design/x/accel/kernelabi"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
)

// composedAttention is spec 010's mandatory path: score, softmax, weight.
//
// Written at f64 and without any of the fused kernel's structure — no shared
// memory, no tree, no lane mapping — because it is the *definition* the fused
// kernel is an optional selection against. A composed path sharing the fused
// one's structure could not tell whether the fusion was right, only that it was
// consistent with itself.
func composedAttention(d testkernels.AttnDims, kvLen uint32, q, k, v []float32) []float64 {
	out := make([]float64, d.QHeads*d.HeadDim)
	group := d.QHeads / d.KVHeads

	for h := range d.QHeads {
		kvHead := h / group

		// MatMul: the scores.
		scores := make([]float64, kvLen)
		for j := range kvLen {
			var acc float64
			for i := range d.HeadDim {
				acc += float64(q[h*d.HeadDim+i]) *
					float64(k[j*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i])
			}
			scores[j] = acc * float64(d.Scale)
		}

		// Softmax, with the maximum subtracted for the same reason the kernel
		// subtracts it.
		m := math.Inf(-1)
		for _, s := range scores {
			m = math.Max(m, s)
		}
		var total float64
		for j := range scores {
			scores[j] = math.Exp(scores[j] - m)
			total += scores[j]
		}

		// MatMul: the weighted sum of V.
		for i := range d.HeadDim {
			var acc float64
			for j := range kvLen {
				acc += scores[j] *
					float64(v[j*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i])
			}
			out[h*d.HeadDim+i] = acc / total
		}
	}
	return out
}

// The fused kernel agrees with the composed definition.
//
// That agreement is what makes it a *selection*: spec 010 says the composed
// path remains mandatory and the fused one is optional, so a device or shape
// that cannot take the fused path falls back — and a fallback is only a
// fallback if the two compute the same thing.
func TestFusedAttentionAgreesWithTheComposedPath(t *testing.T) {
	cases := []struct {
		qHeads, kvHeads, headDim, kvLen uint32
	}{
		{1, 1, 8, 1},     // the smallest legal shape
		{4, 4, 64, 17},   // one KV head per query head
		{8, 2, 32, 40},   // grouped query attention: four query heads per KV head
		{2, 1, 128, 128}, // the maximum head dimension and the longest cache
		{6, 3, 8, 3},
	}
	for _, c := range cases {
		t.Run(fmt.Sprintf("q%d_kv%d_d%d_len%d", c.qHeads, c.kvHeads, c.headDim, c.kvLen),
			func(t *testing.T) {
				d := testkernels.AttnDims{
					QHeads: c.qHeads, KVHeads: c.kvHeads, HeadDim: c.headDim,
					Scale: float32(1 / math.Sqrt(float64(c.headDim))),
				}
				lengths := []uint32{c.kvLen}
				q := ramp32(int(c.qHeads*c.headDim), 0.7)
				k := ramp32(int(c.kvLen*c.kvHeads*c.headDim), 1.1)
				v := ramp32(int(c.kvLen*c.kvHeads*c.headDim), 0.3)
				out := make([]float32, c.qHeads*c.headDim)

				err := kernel.DispatchCooperative(&testkernels.AttentionDecodeKernel,
					accel.ID3{X: c.qHeads},
					kernelabi.Args{Slices: []any{q, k, v, lengths, out}, Uniforms: []any{d}})
				if err != nil {
					t.Fatalf("dispatch: %v", err)
				}

				want := composedAttention(d, lengths[0], q, k, v)
				for i := range out {
					// NaN before the tolerance, because NaN > 1e-4 is false and
					// a difference test alone would pass on it.
					if math.IsNaN(float64(out[i])) {
						t.Fatalf("head %d dim %d is NaN", i/int(c.headDim), i%int(c.headDim))
					}
					if diff := math.Abs(float64(out[i]) - want[i]); diff > 1e-4 {
						t.Fatalf("head %d dim %d is %v, want about %v (differing by %g)",
							i/int(c.headDim), i%int(c.headDim), out[i], want[i], diff)
					}
				}
			})
	}
}

func ramp32(n int, step float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(float64(i) * step))
	}
	return out
}

// Grouped-query attention: several query heads share one KV head, and which
// they share is contiguous by construction.
//
// Getting the grouping wrong produces a plausible tensor — every head attends
// to *something* — so this asserts it directly: with distinct values per KV
// head, each query head's output has to come from its own group's.
func TestGroupedQueryHeadsShareTheRightKVHead(t *testing.T) {
	const qHeads, kvHeads, headDim, kvLen = 6, 3, 8, 4
	d := testkernels.AttnDims{
		QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim, Scale: 1,
	}
	lengths := []uint32{kvLen}

	// Every query is identical, and each KV head's V is a distinct constant, so
	// each output is exactly its group's constant whatever the scores are.
	q := make([]float32, qHeads*headDim)
	for i := range q {
		q[i] = 1
	}
	k := make([]float32, kvLen*kvHeads*headDim)
	v := make([]float32, kvLen*kvHeads*headDim)
	for j := range uint32(kvLen) {
		for h := range uint32(kvHeads) {
			for i := range uint32(headDim) {
				v[j*kvHeads*headDim+h*headDim+i] = float32(h) + 1
			}
		}
	}
	out := make([]float32, qHeads*headDim)

	err := kernel.DispatchCooperative(&testkernels.AttentionDecodeKernel,
		accel.ID3{X: qHeads},
		kernelabi.Args{Slices: []any{q, k, v, lengths, out}, Uniforms: []any{d}})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	group := uint32(qHeads / kvHeads)
	for h := range uint32(qHeads) {
		want := float32(h/group) + 1
		for i := range uint32(headDim) {
			if got := out[h*headDim+i]; math.Abs(float64(got-want)) > 1e-5 {
				t.Fatalf("query head %d dim %d is %v, want its KV head %d's %v",
					h, i, got, h/group, want)
			}
		}
	}
}

// A lane past the cache contributes negative infinity to the maximum, not zero.
//
// Zero would win over a row of genuinely negative scores and shift every
// exponent, which changes the answer rather than only the intermediate. It is
// the case a test with positive scores never sees.
func TestLanesPastTheCacheDoNotSkewTheMaximum(t *testing.T) {
	const qHeads, kvHeads, headDim, kvLen = 1, 1, 8, 3
	d := testkernels.AttnDims{
		QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim, Scale: 1,
	}
	lengths := []uint32{kvLen}

	// Scores are all strongly negative: q and k point in opposite directions.
	q := make([]float32, headDim)
	k := make([]float32, kvLen*headDim)
	v := make([]float32, kvLen*headDim)
	for i := range q {
		q[i] = 4
	}
	for i := range k {
		k[i] = -4
	}
	for j := range kvLen {
		for i := range headDim {
			v[j*headDim+i] = float32(j) + 1
		}
	}
	out := make([]float32, headDim)

	err := kernel.DispatchCooperative(&testkernels.AttentionDecodeKernel,
		accel.ID3{X: qHeads},
		kernelabi.Args{Slices: []any{q, k, v, lengths, out}, Uniforms: []any{d}})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	want := composedAttention(d, lengths[0], q, k, v)
	for i := range out {
		// NaN first, and separately. A comparison against a tolerance cannot
		// see it — NaN > 1e-4 is false — so a test written only as a difference
		// passes on exactly the failure this case exists to catch: a maximum
		// shifted to zero underflows every exponential, and zero over zero is
		// NaN.
		if math.IsNaN(float64(out[i])) || math.IsInf(float64(out[i]), 0) {
			t.Fatalf("dim %d is %v: an inactive lane contributing zero to the maximum "+
				"wins over these negative scores, every exponential underflows, and the "+
				"quotient is zero over zero", i, out[i])
		}
		if diff := math.Abs(float64(out[i]) - want[i]); diff > 1e-4 {
			t.Fatalf("dim %d is %v, want %v: an inactive lane contributing zero to the "+
				"maximum would shift every exponent", i, out[i], want[i])
		}
	}
}

// The authored kernel, spec 004's fifth testing level.
//
// The invocations rendezvous for real, one goroutine each behind a cyclic
// barrier, because this kernel's reductions are destructive: they overwrite the
// shared array they reduce, so running every invocation through the whole
// function once per barrier reduces the second pass's own output. That
// workaround passes by luck on some shapes and produces NaN on others, which is
// how it was found.
func TestAuthoredAttentionDecode(t *testing.T) {
	const qHeads, kvHeads, headDim, kvLen = 4, 2, 32, 20
	d := testkernels.AttnDims{
		QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim, Scale: float32(1 / math.Sqrt(headDim)),
	}
	lengths := []uint32{kvLen}
	q := ramp32(qHeads*headDim, 0.9)
	k := ramp32(kvLen*kvHeads*headDim, 0.4)
	v := ramp32(kvLen*kvHeads*headDim, 1.3)

	authored := make([]float32, qHeads*headDim)
	size := kernel.ID3{X: 128, Y: 1, Z: 1}
	for h := range uint32(qHeads) {
		var scores, red [128]float32
		kernelabi.Poison(scores[:])
		kernelabi.Poison(red[:])
		kernel.RunAuthored(size, kernel.ID3{X: h}, kernel.ID3{X: qHeads}, 128,
			func(th kernel.Thread) {
				testkernels.AttentionDecode(th, d, q, k, v, lengths, authored, &scores, &red)
			})
	}

	generated := make([]float32, qHeads*headDim)
	err := kernel.DispatchCooperative(&testkernels.AttentionDecodeKernel,
		accel.ID3{X: qHeads},
		kernelabi.Args{Slices: []any{q, k, v, lengths, generated}, Uniforms: []any{d}})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	want := composedAttention(d, lengths[0], q, k, v)
	for i := range generated {
		if math.IsNaN(float64(generated[i])) || math.IsNaN(float64(authored[i])) {
			t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
		}
		if math.Abs(float64(generated[i])-want[i]) > 1e-4 {
			t.Fatalf("the generated lowering's element %d is %v, want about %v",
				i, generated[i], want[i])
		}
		if math.Abs(float64(authored[i])-want[i]) > 1e-4 {
			t.Fatalf("the authored kernel's element %d is %v, want about %v: the two "+
				"halves of spec 004's fifth level must agree", i, authored[i], want[i])
		}
	}
}

// The record carries what a backend needs at pipeline creation: two shared
// arrays and the suspension count.
func TestTheAttentionKernelDeclaresItsShape(t *testing.T) {
	k := &testkernels.AttentionDecodeKernel
	if len(k.SharedSizes) != 2 {
		t.Fatalf("it declares %v shared arrays, want two: the scores and the "+
			"reduction scratch. The block loop added no third one -- the output "+
			"accumulator is a local, because each lane owns one element of the "+
			"row and no other lane reads it", k.SharedSizes)
	}
	for i, n := range k.SharedSizes {
		if n != testkernels.AttnBlock {
			t.Errorf("shared array %d has extent %d, want %d", i, n, testkernels.AttnBlock)
		}
	}
	// Two trees and the barriers around them, counted in the source: a barrier
	// inside a loop counts once however many rounds it runs.
	//
	// Six, and one of them is the block loop's: a barrier at the top of the
	// body, separating a pass's writes to the shared arrays from the previous
	// pass's reads of them. That is the hazard a single-pass kernel did not
	// have and the one nothing else would report, since the CPU backend's
	// rendezvous check finds an invocation that fails to arrive and this would
	// be a race between arrivals.
	if got := k.Suspensions; got != 6 {
		t.Errorf("it has %d suspension points, want 6", got)
	}
}

// A cache longer than one workgroup, which is the whole point of the block
// loop: before it, `Attention` refused any cache past 128 positions and no
// model was servable (accel issue 8).
//
// # The tolerance is derived, not chosen
//
// The reference accumulates in float64 and the kernel in float32. Each output
// element is a sum of kvLen products followed by one division, so the rounding
// is bounded by the usual γₙ = nu/(1-nu) for n = kvLen + headDim + 1 terms at
// u = 2⁻²⁴. The running softmax adds one multiply per block to each carried
// term, which is at most ceil(kvLen/AttnBlock) more roundings. The scores are a
// convex combination, so the result is bounded by max|v| and the absolute error
// by that times γ.
func TestAttentionDecodeScoresACacheLongerThanAWorkgroup(t *testing.T) {
	for _, kvLen := range []uint32{129, 300, 512} {
		t.Run(fmt.Sprint(kvLen), func(t *testing.T) {
			const qHeads, kvHeads, headDim = 4, 2, 32
			// Capacity is a whole number of blocks and longer than kvLen, so
			// the last block is partly masked and one block is entirely past
			// the end -- both of the cases the mask has to get right.
			capacity := ((kvLen+AttnBlockT-1)/AttnBlockT + 1) * AttnBlockT

			d := testkernels.AttnDims{
				QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
				Scale: 1 / float32(math.Sqrt(headDim)),
			}
			rng := rand.New(rand.NewPCG(9, 4))
			fill := func(n int) []float32 {
				s := make([]float32, n)
				for i := range s {
					s[i] = float32(rng.NormFloat64())
				}
				return s
			}
			q := fill(qHeads * headDim)
			k := fill(int(capacity) * kvHeads * headDim)
			v := fill(int(capacity) * kvHeads * headDim)
			lengths := []uint32{kvLen}

			got := make([]float32, qHeads*headDim)
			if err := kernel.DispatchCooperative(&testkernels.AttentionDecodeKernel,
				accel.ID3{X: qHeads},
				kernelabi.Args{
					Slices:   []any{q, k, v, lengths, got},
					Uniforms: []any{d},
				}); err != nil {
				t.Fatalf("dispatch: %v", err)
			}

			want := composedAttention(d, kvLen, q, k, v)

			// The derived bound, evaluated for this case.
			const u = 1.0 / (1 << 24)
			n := float64(kvLen+headDim+1) + math.Ceil(float64(kvLen)/AttnBlockT)
			gamma := n * u / (1 - n*u)
			maxV := 0.0
			for _, x := range v {
				maxV = math.Max(maxV, math.Abs(float64(x)))
			}
			tol := maxV * gamma

			for i := range got {
				if diff := math.Abs(float64(got[i]) - want[i]); diff > tol {
					t.Fatalf("element %d is %v, want %v: off by %g, and the derived "+
						"bound for %d positions is %g", i, got[i], want[i], diff, kvLen, tol)
				}
			}
		})
	}
}

// AttnBlockT is [testkernels.AttnBlock] as an untyped constant, so the
// arithmetic above reads as arithmetic.
const AttnBlockT = testkernels.AttnBlock
