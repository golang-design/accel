// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"golang.design/x/accel/kernelabi"
	"golang.design/x/accel/kmath"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/kernels"
)

// composedAttention is spec 010's mandatory path: score, softmax, weight.
//
// Written at f64 and without any of the fused kernel's structure — no shared
// memory, no tree, no lane mapping — because it is the *definition* the fused
// kernel is an optional selection against. A composed path sharing the fused
// one's structure could not tell whether the fusion was right, only that it was
// consistent with itself.
func composedAttention(d kernels.AttnDims, kvLen uint32, q, k, v []float32) []float64 {
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
				d := kernels.AttnDims{
					QHeads: c.qHeads, KVHeads: c.kvHeads, HeadDim: c.headDim,
					Scale: float32(1 / math.Sqrt(float64(c.headDim))),
				}
				lengths := []uint32{c.kvLen}
				q := ramp32(int(c.qHeads*c.headDim), 0.7)
				k := ramp32(int(c.kvLen*c.kvHeads*c.headDim), 1.1)
				v := ramp32(int(c.kvLen*c.kvHeads*c.headDim), 0.3)
				out := make([]float32, c.qHeads*c.headDim)

				err := kernel.DispatchCooperative(&kernels.AttentionDecodeKernel,
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
	d := kernels.AttnDims{
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

	err := kernel.DispatchCooperative(&kernels.AttentionDecodeKernel,
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
	d := kernels.AttnDims{
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

	err := kernel.DispatchCooperative(&kernels.AttentionDecodeKernel,
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
	d := kernels.AttnDims{
		QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim, Scale: float32(1 / math.Sqrt(headDim)),
	}
	lengths := []uint32{kvLen}
	q := ramp32(qHeads*headDim, 0.9)
	k := ramp32(kvLen*kvHeads*headDim, 0.4)
	v := ramp32(kvLen*kvHeads*headDim, 1.3)

	authored := make([]float32, qHeads*headDim)
	for h := range uint32(qHeads) {
		var scores, red [128]float32
		kernelabi.Poison(scores[:])
		kernelabi.Poison(red[:])
		kernel.RunAuthored(&kernels.AttentionDecodeKernel, kernel.ID3{X: h}, kernel.ID3{X: qHeads}, 128,
			func(th kernel.Thread) {
				kernels.AttentionDecode(th, d, q, k, v, lengths, authored, &scores, &red)
			})
	}

	generated := make([]float32, qHeads*headDim)
	err := kernel.DispatchCooperative(&kernels.AttentionDecodeKernel,
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
	k := &kernels.AttentionDecodeKernel
	if len(k.SharedSizes) != 2 {
		t.Fatalf("it declares %v shared arrays, want two: the scores and the "+
			"reduction scratch. The block loop added no third one -- the output "+
			"accumulator is a local, because each lane owns one element of the "+
			"row and no other lane reads it", k.SharedSizes)
	}
	for i, n := range k.SharedSizes {
		if n != kernels.AttnBlock {
			t.Errorf("shared array %d has extent %d, want %d", i, n, kernels.AttnBlock)
		}
	}
	// Two trees and the barriers around them, counted in the source: a barrier
	// inside a loop counts once however many rounds it runs.
	//
	// Eight since 2026-09-05: the six barriers, one of them the block loop's
	// (a barrier at the top of the body, separating a pass's writes to the
	// shared arrays from the previous pass's reads of them), plus the two
	// states a subgroup rendezvous costs -- the per-position dot product is
	// a subgroup sum, and the lowering suspends once to contribute a lane's
	// value and once to read the combined one back.
	if got := k.Suspensions; got != 8 {
		t.Errorf("it has %d suspension points, want 8", got)
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

			d := kernels.AttnDims{
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
			if err := kernel.DispatchCooperative(&kernels.AttentionDecodeKernel,
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

// AttnBlockT is [kernels.AttnBlock] as an untyped constant, so the
// arithmetic above reads as arithmetic.
const AttnBlockT = kernels.AttnBlock

// specs/044-unbounded-context.md's outcome table claims the block loop is
// bit-identical to the single-pass form at one block, "by construction rather
// than by measurement". This is that claim, asserted rather than reasoned.
//
// The reference reproduces the kernel's arithmetic in float32 in the kernel's
// order, tree reductions included, and the comparison is == with no tolerance.
// Reproducing the reduction *order* is the whole point: floating-point addition
// is not associative, so a reference that summed left to right would need a
// tolerance and would prove nothing about this.
func TestOneBlockIsExactlyTheSinglePassForm(t *testing.T) {
	const qHeads, kvHeads, headDim = 4, 2, 32
	for _, kvLen := range []uint32{1, 63, 128} {
		t.Run(fmt.Sprint(kvLen), func(t *testing.T) {
			d := kernels.AttnDims{
				QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
				Scale: 1 / float32(math.Sqrt(headDim)),
			}
			rng := rand.New(rand.NewPCG(21, 5))
			fill := func(n int) []float32 {
				s := make([]float32, n)
				for i := range s {
					s[i] = float32(rng.NormFloat64())
				}
				return s
			}
			q := fill(qHeads * headDim)
			k := fill(AttnBlockT * kvHeads * headDim)
			v := fill(AttnBlockT * kvHeads * headDim)

			got := make([]float32, qHeads*headDim)
			if err := kernel.DispatchCooperative(&kernels.AttentionDecodeKernel,
				accel.ID3{X: qHeads},
				kernelabi.Args{
					Slices:   []any{q, k, v, []uint32{kvLen}, got},
					Uniforms: []any{d},
				}); err != nil {
				t.Fatalf("dispatch: %v", err)
			}

			for h := uint32(0); h < qHeads; h++ {
				kvHead := h / (qHeads / kvHeads)

				// Score, exactly as the kernel does: one lane per position,
				// accumulating the dot product in float32 in index order.
				var red [AttnBlockT]float32
				for lane := uint32(0); lane < AttnBlockT; lane++ {
					s := float32(-3.4e38)
					if lane < kvLen {
						dot := float32(0)
						for i := uint32(0); i < headDim; i++ {
							// float32() on the product and on the sum, which is
							// what the generated kernel emits and what stops the
							// compiler fusing them into an FMA. Without it this
							// reference is a *more* accurate computation and
							// therefore a different one.
							dot = float32(dot + float32(q[h*headDim+i]*
								k[lane*kvHeads*headDim+kvHead*headDim+i]))
						}
						s = dot * d.Scale
					}
					red[lane] = s
				}
				scores := red

				// The tree maximum, stride by stride.
				for stride := uint32(AttnBlockT / 2); stride > 0; stride /= 2 {
					for lane := uint32(0); lane < stride; lane++ {
						red[lane] = max(red[lane], red[lane+stride])
					}
				}
				best := red[0]

				// The shifted exponentials and the tree sum.
				for lane := uint32(0); lane < AttnBlockT; lane++ {
					e := float32(0)
					if lane < kvLen {
						e = kmath.Exp(scores[lane] - best)
					}
					scores[lane] = e
					red[lane] = e
				}
				for stride := uint32(AttnBlockT / 2); stride > 0; stride /= 2 {
					for lane := uint32(0); lane < stride; lane++ {
						red[lane] = float32(red[lane] + red[lane+stride])
					}
				}
				total := red[0]

				for lane := uint32(0); lane < headDim; lane++ {
					o := float32(0)
					for j := uint32(0); j < AttnBlockT; j++ {
						if j < kvLen {
							o = float32(o + float32(scores[j]*
								v[j*kvHeads*headDim+kvHead*headDim+lane]))
						}
					}
					want := o / total
					if g := got[h*headDim+lane]; g != want {
						t.Fatalf("head %d element %d is %v, want exactly %v: at one "+
							"block the running softmax reduces to the single-pass "+
							"form, so this is equality and not a tolerance",
							h, lane, g, want)
					}
				}
			}
		})
	}
}

// A length past the cache's own extent is clamped to it rather than read past
// the end. The contiguous companion to
// TestABatchedLengthPastItsPageTableIsTruncated.
//
// The loop bound limits base and not base+lane, so the lanes of the last block
// reach AttnBlock-1 positions beyond it. Before the clamp this read off the end
// of the cache binding, which the CPU backend reports as an out-of-range index
// and a device would not report at all.
func TestALengthPastTheCacheIsClamped(t *testing.T) {
	const qHeads, kvHeads, headDim, capacity = 2, 1, 16, 192
	d := kernels.AttnDims{
		QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
		Scale: 1 / float32(math.Sqrt(headDim)),
	}
	rng := rand.New(rand.NewPCG(31, 17))
	fill := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64())
		}
		return s
	}
	q := fill(qHeads * headDim)
	k := fill(capacity * kvHeads * headDim)
	v := fill(capacity * kvHeads * headDim)

	run := func(kvLen uint32) []float32 {
		out := make([]float32, qHeads*headDim)
		if err := kernel.DispatchCooperative(&kernels.AttentionDecodeKernel,
			accel.ID3{X: qHeads},
			kernelabi.Args{
				Slices: []any{q, k, v, []uint32{kvLen}, out}, Uniforms: []any{d},
			}); err != nil {
			t.Fatalf("dispatch at length %d: %v", kvLen, err)
		}
		return out
	}

	// capacity+40 lands inside the last block, so an unclamped length reads
	// past the binding rather than merely covering it.
	got, want := run(capacity+40), run(capacity)
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("element %d is %v with a length of %d and %v with a length of "+
				"%d: a length past the cache attends over the cache",
				i, got[i], capacity+40, want[i], capacity)
		}
	}
}
