// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"fmt"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
)

// The prefill attention matches an independently written higher-precision
// reference, and it masks causally.
//
// The reference is a straight quadruple loop in f64 with no tiling, no shared
// memory and no reduction tree, so it shares none of the kernel's structure.
// specs/010-kernel-corpus.md requires that: a reference built the same way
// would share the kernel's bugs.
//
// **Masking is the property worth attacking.** A prefill that let a token
// attend to its own future is not a slower answer, it is a different model --
// and it would still produce plausible numbers, sum to one, and pass every
// shape check. So the test asserts the mask twice: once by comparing against a
// reference that masks, and once by changing a value the mask should hide and
// checking that nothing moves.
func TestAttentionPrefillMatchesItsReference(t *testing.T) {
	for _, c := range []struct{ qHeads, kvHeads, headDim, qSeq, base int }{
		{2, 1, 8, 4, 0},  // a fresh cache: base zero, KVLen equals QSeq
		{4, 2, 8, 5, 0},  // grouped query heads
		{2, 2, 16, 3, 2}, // extending a cache: the first query sees three
		{1, 1, 8, 7, 0},  // one head, a longer sequence
	} {
		name := fmt.Sprintf("q%d_kv%d_d%d_s%d_b%d", c.qHeads, c.kvHeads, c.headDim, c.qSeq, c.base)
		t.Run(name, func(t *testing.T) {
			kvLen := c.base + c.qSeq
			dims := testkernels.PrefillDims{
				QHeads: uint32(c.qHeads), KVHeads: uint32(c.kvHeads),
				HeadDim: uint32(c.headDim), QSeq: uint32(c.qSeq),
				KVLen: uint32(kvLen), Base: uint32(c.base),
				Scale: float32(1 / math.Sqrt(float64(c.headDim))),
			}
			q := make([]float32, c.qSeq*c.qHeads*c.headDim)
			k := make([]float32, kvLen*c.kvHeads*c.headDim)
			v := make([]float32, kvLen*c.kvHeads*c.headDim)
			for i := range q {
				q[i] = float32(math.Sin(float64(i) * 0.31))
			}
			for i := range k {
				k[i] = float32(math.Cos(float64(i) * 0.17))
				v[i] = float32(i%9)/4 - 1
			}
			out := make([]float32, len(q))

			run := func() {
				for i := range out {
					out[i] = 0
				}
				err := kernel.DispatchCooperative(&testkernels.AttentionPrefillKernel,
					accel.ID3{X: uint32(c.qSeq * c.qHeads)},
					accel.KernelArgs{
						Slices: []any{q, k, v, out}, Uniforms: []any{dims},
					})
				if err != nil {
					t.Fatalf("dispatch: %v", err)
				}
			}
			run()

			want := prefillReference(q, k, v, c.qHeads, c.kvHeads, c.headDim,
				c.qSeq, kvLen, c.base, float64(dims.Scale))
			for i := range out {
				if math.Abs(float64(out[i])-want[i]) > 1e-4*(1+math.Abs(want[i])) {
					t.Fatalf("element %d is %v, want about %v", i, out[i], want[i])
				}
			}

			// The mask, attacked directly: change a value only a *future*
			// position could see, and every output must be unchanged. The first
			// query position sees positions 0..base, so position base+1 is
			// hidden from it and visible to the next one.
			if c.qSeq < 2 {
				return
			}
			before := append([]float32(nil), out...)
			hidden := c.base + 1
			for i := range c.kvHeads * c.headDim {
				v[hidden*c.kvHeads*c.headDim+i] += 100
			}
			run()

			row := c.qHeads * c.headDim // one query position's output
			for i := range row {
				if out[i] != before[i] {
					t.Fatalf("query position 0 changed when cached position %d changed, and "+
						"the mask should hide it: %v became %v", hidden, before[i], out[i])
				}
			}
			// And a later position, which *can* see it, must have changed --
			// otherwise the test above passes because nothing reads V at all.
			moved := false
			for i := row; i < len(out); i++ {
				if out[i] != before[i] {
					moved = true
					break
				}
			}
			if !moved {
				t.Fatal("no query position changed when a visible cached value changed, so " +
					"the mask test proves nothing")
			}
		})
	}
}

// prefillReference is causal attention written from its definition.
func prefillReference(q, k, v []float32, qHeads, kvHeads, headDim, qSeq, kvLen, base int,
	scale float64) []float64 {

	out := make([]float64, qSeq*qHeads*headDim)
	group := qHeads / kvHeads
	for s := range qSeq {
		for h := range qHeads {
			kvHead := h / group
			limit := base + s
			var scores []float64
			var best = math.Inf(-1)
			for j := 0; j < kvLen; j++ {
				if j > limit {
					scores = append(scores, math.Inf(-1))
					continue
				}
				var acc float64
				for i := range headDim {
					acc += float64(q[(s*qHeads+h)*headDim+i]) *
						float64(k[j*kvHeads*headDim+kvHead*headDim+i])
				}
				acc *= scale
				scores = append(scores, acc)
				best = math.Max(best, acc)
			}
			var sum float64
			for i := range scores {
				if math.IsInf(scores[i], -1) {
					scores[i] = 0
					continue
				}
				scores[i] = math.Exp(scores[i] - best)
				sum += scores[i]
			}
			for i := range headDim {
				var acc float64
				for j := range scores {
					acc += scores[j] / sum * float64(v[j*kvHeads*headDim+kvHead*headDim+i])
				}
				out[(s*qHeads+h)*headDim+i] = acc
			}
		}
	}
	return out
}

// A prefill of N tokens equals N decode steps over the same cache.
//
// specs/009-sequencing.md's M7 parity criterion, at the kernel level: the two
// paths compute the same function and the whole point of having both is that
// one is faster. If they disagreed, a model would produce different text
// depending on whether it had been prompted or generated -- the least
// debuggable failure this project could ship.
//
// Equal within the softmax's budget rather than bit for bit: the two kernels
// reduce over different numbers of lanes, so the additions happen in a
// different order, which specs/008-numerics.md section 7 bounds rather than
// forbids.
func TestPrefillEqualsIncrementalDecode(t *testing.T) {
	const qHeads, kvHeads, headDim, n = 4, 2, 8, 6
	scale := float32(1 / math.Sqrt(headDim))

	k := make([]float32, n*kvHeads*headDim)
	v := make([]float32, n*kvHeads*headDim)
	q := make([]float32, n*qHeads*headDim)
	for i := range k {
		k[i] = float32(math.Cos(float64(i) * 0.23))
		v[i] = float32(math.Sin(float64(i)*0.19)) * 2
	}
	for i := range q {
		q[i] = float32(math.Sin(float64(i) * 0.41))
	}

	// The prefill: all n tokens at once, causally masked.
	prefill := make([]float32, len(q))
	err := kernel.DispatchCooperative(&testkernels.AttentionPrefillKernel,
		accel.ID3{X: n * qHeads},
		accel.KernelArgs{
			Slices: []any{q, k, v, prefill},
			Uniforms: []any{testkernels.PrefillDims{
				QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
				QSeq: n, KVLen: n, Base: 0, Scale: scale,
			}},
		})
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}

	// The decode path: one token at a time, over a cache that grows.
	decode := make([]float32, len(q))
	for step := range n {
		step1 := make([]float32, qHeads*headDim)
		err := kernel.DispatchCooperative(&testkernels.AttentionDecodeKernel,
			accel.ID3{X: qHeads},
			accel.KernelArgs{
				Slices: []any{
					q[step*qHeads*headDim : (step+1)*qHeads*headDim],
					k[:(step+1)*kvHeads*headDim],
					v[:(step+1)*kvHeads*headDim],
					step1,
				},
				Uniforms: []any{testkernels.AttnDims{
					QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
					KVLen: uint32(step + 1), Scale: scale,
				}},
			})
		if err != nil {
			t.Fatalf("decode step %d: %v", step, err)
		}
		copy(decode[step*qHeads*headDim:], step1)
	}

	for i := range prefill {
		if d := math.Abs(float64(prefill[i] - decode[i])); d > 1e-5*(1+math.Abs(float64(decode[i]))) {
			t.Fatalf("element %d: prefill gives %v and incremental decode gives %v; the two "+
				"paths must compute the same function, or a model produces different text "+
				"depending on whether it was prompted or generated",
				i, prefill[i], decode[i])
		}
	}
}

// The authored kernels and their generated lowerings agree.
//
// specs/004-kernel-authoring.md's fifth testing level. The generated lowering
// is what runs, so nothing else calls the authored function -- and a function
// nobody executes means whatever the IR made of it. These run both and compare.
//
// The cooperative one rendezvouses for real, one goroutine per invocation
// behind a cyclic barrier, because the alternative -- running each invocation
// through the whole function -- is exact only while every loop runs once.
func TestAuthoredFormsAgreeWithTheirLowerings(t *testing.T) {
	t.Run("CastF32ToF16", func(t *testing.T) {
		const n = 128
		in := make([]float32, n)
		for i := range in {
			// Values that need rounding, so the comparison is about the
			// conversion rather than about values exact in both formats.
			in[i] = float32(i)*1.0009765625 - 60
		}
		authored := make([]accel.Float16, n)
		for i := range in {
			testkernels.CastF32ToF16(flatThread(i, n), in, authored)
		}
		generated := make([]accel.Float16, n)
		if err := kernel.Dispatch(&testkernels.CastF32ToF16Kernel, accel.ID3{X: 2},
			accel.KernelArgs{Slices: []any{in, generated}}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range generated {
			if authored[i].Bits() != generated[i].Bits() {
				t.Fatalf("element %d: authored %#04x, generated %#04x",
					i, authored[i].Bits(), generated[i].Bits())
			}
		}
	})

	t.Run("CastF16ToF32", func(t *testing.T) {
		const n = 128
		in := make([]accel.Float16, n)
		for i := range in {
			in[i] = accel.ToFloat16(float32(i)*0.375 - 20)
		}
		authored := make([]float32, n)
		for i := range in {
			testkernels.CastF16ToF32(flatThread(i, n), in, authored)
		}
		generated := make([]float32, n)
		if err := kernel.Dispatch(&testkernels.CastF16ToF32Kernel, accel.ID3{X: 2},
			accel.KernelArgs{Slices: []any{in, generated}}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range generated {
			// Exact: widening loses nothing, so anything but equality is a bug
			// rather than a rounding difference.
			if authored[i] != generated[i] {
				t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
	})

	t.Run("AttentionPrefill", func(t *testing.T) {
		const qHeads, kvHeads, headDim, qSeq = 2, 1, 8, 4
		dims := testkernels.PrefillDims{
			QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
			QSeq: qSeq, KVLen: qSeq, Base: 0,
			Scale: float32(1) / float32(math.Sqrt(headDim)),
		}
		q := make([]float32, qSeq*qHeads*headDim)
		k := make([]float32, qSeq*kvHeads*headDim)
		v := make([]float32, qSeq*kvHeads*headDim)
		for i := range q {
			q[i] = float32(math.Sin(float64(i) * 0.29))
		}
		for i := range k {
			k[i] = float32(math.Cos(float64(i) * 0.13))
			v[i] = float32(i%5) - 2
		}

		authored := make([]float32, len(q))
		groups := kernel.ID3{X: qSeq * qHeads, Y: 1, Z: 1}
		size := kernel.ID3{X: 128, Y: 1, Z: 1}
		for g := range groups.X {
			var scores, red [128]float32
			kernel.RunAuthored(size, kernel.ID3{X: g}, groups, 128, func(th kernel.Thread) {
				testkernels.AttentionPrefill(th, dims, q, k, v, authored, &scores, &red)
			})
		}

		generated := make([]float32, len(q))
		if err := kernel.DispatchCooperative(&testkernels.AttentionPrefillKernel,
			accel.ID3{X: groups.X},
			accel.KernelArgs{
				Slices: []any{q, k, v, generated}, Uniforms: []any{dims},
			}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range generated {
			if math.Abs(float64(authored[i]-generated[i])) > 1e-5 {
				t.Fatalf("element %d: authored %v, generated %v; both halves of "+
					"specs/004-kernel-authoring.md's fifth level must agree",
					i, authored[i], generated[i])
			}
		}
	})
}

// flatThread is one invocation of a flat kernel, at global index i.
//
// Written here rather than exported, because a flat kernel's authored form is
// called only by a test like this one: the generated lowering is what runs.
func flatThread(i, n int) kernel.Thread {
	const wg = 64
	return kernel.NewThread(
		kernel.ID3{X: uint32(i), Y: 0, Z: 0},
		kernel.ID3{X: uint32(i % wg), Y: 0, Z: 0},
		kernel.ID3{X: uint32(i / wg), Y: 0, Z: 0},
		kernel.ID3{X: wg, Y: 1, Z: 1},
		kernel.ID3{X: uint32((n + wg - 1) / wg), Y: 1, Z: 1},
	)
}
