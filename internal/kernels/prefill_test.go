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

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/kernels"
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
			dims := kernels.PrefillDims{
				QHeads: uint32(c.qHeads), KVHeads: uint32(c.kvHeads),
				HeadDim: uint32(c.headDim), QSeq: uint32(c.qSeq),
				Base:  uint32(c.base),
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
			// A prefill's cache holds its own tokens plus whatever preceded
			// them, which is what Base counts.
			lengths := []uint32{uint32(c.base + c.qSeq)}

			run := func() {
				for i := range out {
					out[i] = 0
				}
				err := kernel.DispatchCooperative(&kernels.AttentionPrefillKernel,
					accel.ID3{X: uint32(c.qSeq * c.qHeads)},
					kernelabi.Args{
						Slices: []any{q, k, v, lengths, out}, Uniforms: []any{dims},
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
	err := kernel.DispatchCooperative(&kernels.AttentionPrefillKernel,
		accel.ID3{X: n * qHeads},
		kernelabi.Args{
			Slices: []any{q, k, v, []uint32{n}, prefill},
			Uniforms: []any{kernels.PrefillDims{
				QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
				QSeq: n, Base: 0, Scale: scale,
			}},
		})
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}

	// The decode path: one token at a time, over a cache that grows.
	decode := make([]float32, len(q))
	for step := range n {
		step1 := make([]float32, qHeads*headDim)
		err := kernel.DispatchCooperative(&kernels.AttentionDecodeKernel,
			accel.ID3{X: qHeads},
			kernelabi.Args{
				Slices: []any{
					q[step*qHeads*headDim : (step+1)*qHeads*headDim],
					k[:(step+1)*kvHeads*headDim],
					v[:(step+1)*kvHeads*headDim],
					[]uint32{uint32(step + 1)},
					step1,
				},
				Uniforms: []any{kernels.AttnDims{
					QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
					Scale: scale,
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
			kernels.CastF32ToF16(flatThread(i, n), in, authored)
		}
		generated := make([]accel.Float16, n)
		if err := kernel.Dispatch(&kernels.CastF32ToF16Kernel, accel.ID3{X: 2},
			kernelabi.Args{Slices: []any{in, generated}}); err != nil {
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
			kernels.CastF16ToF32(flatThread(i, n), in, authored)
		}
		generated := make([]float32, n)
		if err := kernel.Dispatch(&kernels.CastF16ToF32Kernel, accel.ID3{X: 2},
			kernelabi.Args{Slices: []any{in, generated}}); err != nil {
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

	t.Run("CastBF16ToF32", func(t *testing.T) {
		const n = 128
		in := make([]accel.BFloat16, n)
		for i := range in {
			// bf16 keeps f32's eight-bit exponent, so the inputs range over
			// magnitudes f16 could not hold at all -- which is the reason this
			// widening is the one registered and bf16 to f16 is not.
			in[i] = accel.ToBFloat16(float32(i)*0.375 - 20)
		}
		authored := make([]float32, n)
		for i := range in {
			kernels.CastBF16ToF32(flatThread(i, n), in, authored)
		}
		generated := make([]float32, n)
		if err := kernel.Dispatch(&kernels.CastBF16ToF32Kernel, accel.ID3{X: 2},
			kernelabi.Args{Slices: []any{in, generated}}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range generated {
			// Exact, and more strongly than f16's: bf16 is f32's top half, so
			// the conversion is a shift and every input is a witness.
			if authored[i] != generated[i] {
				t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
	})

	t.Run("QuantMatMul", func(t *testing.T) {
		const m, k, n = 2, 32, 4
		a := make([]accel.Float16, m*k)
		bq := make([]int8, k*n)
		bs := make([]accel.Float16, (k*n+31)/32)
		for i := range a {
			a[i] = accel.ToFloat16(float32(i%7) - 3)
		}
		for i := range bq {
			bq[i] = int8(i%61) - 30
		}
		for i := range bs {
			bs[i] = accel.ToFloat16(0.25 + float32(i)/16)
		}
		dims := kernels.GEMMDims{M: m, N: n, K: k}

		authored := make([]float32, m*n)
		for i := range authored {
			kernels.QuantMatMul(flatThread(i, m*n), dims, a, bq, bs, authored)
		}
		generated := make([]float32, m*n)
		if err := kernel.Dispatch(&kernels.QuantMatMulKernel, accel.ID3{X: 1},
			kernelabi.Args{
				Slices: []any{a, bq, bs, generated}, Uniforms: []any{dims},
			}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range generated {
			if authored[i] != generated[i] {
				t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
	})

	t.Run("QuantMatMulF32", func(t *testing.T) {
		const m, k, n = 2, 32, 4
		a := make([]float32, m*k)
		bq := make([]int8, k*n)
		bs := make([]accel.Float16, (k*n+31)/32)
		for i := range a {
			a[i] = float32(i%7) - 3
		}
		for i := range bq {
			bq[i] = int8(i%61) - 30
		}
		for i := range bs {
			bs[i] = accel.ToFloat16(0.25 + float32(i)/16)
		}
		dims := kernels.GEMMDims{M: m, N: n, K: k}

		authored := make([]float32, m*n)
		for i := range authored {
			kernels.QuantMatMulF32(flatThread(i, m*n), dims, a, bq, bs, authored)
		}
		generated := make([]float32, m*n)
		if err := kernel.Dispatch(&kernels.QuantMatMulF32Kernel, accel.ID3{X: 1},
			kernelabi.Args{
				Slices: []any{a, bq, bs, generated}, Uniforms: []any{dims},
			}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range generated {
			if authored[i] != generated[i] {
				t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
	})

	t.Run("QuantRows", func(t *testing.T) {
		const vocab, width, rows = 4, 32, 2
		tq := make([]int8, vocab*width)
		ts := make([]accel.Float16, vocab*width/32)
		for i := range tq {
			tq[i] = int8(i%101) - 50
		}
		for i := range ts {
			ts[i] = accel.ToFloat16(0.5 + float32(i)/8)
		}
		ids := []uint32{2, 0}
		p := kernels.RowParams{Rows: rows, Width: width, Capacity: vocab}

		authored := make([]float32, rows*width)
		for i := range authored {
			kernels.QuantRows(flatThread(i, rows*width), p, tq, ts, ids, authored)
		}
		generated := make([]float32, rows*width)
		if err := kernel.Dispatch(&kernels.QuantRowsKernel, accel.ID3{X: 1},
			kernelabi.Args{
				Slices: []any{tq, ts, ids, generated}, Uniforms: []any{p},
			}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range generated {
			if authored[i] != generated[i] {
				t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
	})

	t.Run("SampleCategorical", func(t *testing.T) {
		probs := []float32{0.1, 0.2, 0.3, 0.4}
		draws := []float32{0.35}
		d := kernels.SampleDims{Vocab: uint32(len(probs)), Rows: 1}

		authored := make([]uint32, 1)
		kernels.SampleCategorical(flatThread(0, 1), d, probs, draws, authored)
		generated := make([]uint32, 1)
		if err := kernel.Dispatch(&kernels.SampleCategoricalKernel, accel.ID3{X: 1},
			kernelabi.Args{
				Slices: []any{probs, draws, generated}, Uniforms: []any{d},
			}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		if authored[0] != generated[0] {
			t.Fatalf("authored %d, generated %d", authored[0], generated[0])
		}
	})

	t.Run("SampleArgmax", func(t *testing.T) {
		logits := make([]float32, 300)
		for i := range logits {
			logits[i] = float32(math.Sin(float64(i) * 0.21))
		}
		// A plateau, so the authored and generated forms have to agree about
		// the tie rule and not merely about the maximum.
		logits[19] = 5
		logits[240] = 5
		d := kernels.SampleDims{Vocab: uint32(len(logits))}

		authored := make([]uint32, 1)
		var best [128]float32
		var at [128]uint32
		kernel.RunAuthored(&kernels.SampleArgmaxKernel, kernel.ID3{}, kernel.ID3{X: 1, Y: 1, Z: 1}, 128,
			func(th kernel.Thread) {
				kernels.SampleArgmax(th, d, logits, authored, &best, &at)
			})

		generated := make([]uint32, 1)
		if err := kernel.DispatchCooperative(&kernels.SampleArgmaxKernel,
			accel.ID3{X: 1},
			kernelabi.Args{Slices: []any{logits, generated}, Uniforms: []any{d}}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		if authored[0] != generated[0] {
			t.Fatalf("authored %d, generated %d", authored[0], generated[0])
		}
		if authored[0] != 19 {
			t.Fatalf("both agree on %d, and the lowest of the two equal maxima is 19",
				authored[0])
		}
	})

	t.Run("TopKMask", func(t *testing.T) {
		w := []float32{0.1, 0.5, 0.5, 0.2, 0.4, 0.5}
		d := kernels.TopDims{Vocab: uint32(len(w)), K: 3}
		authored := make([]float32, len(w))
		var best [128]float32
		var at [128]uint32
		kernel.RunAuthored(&kernels.TopKMaskKernel, kernel.ID3{},
			kernel.ID3{X: 1, Y: 1, Z: 1}, 128, func(th kernel.Thread) {
				kernels.TopKMask(th, d, w, authored, &best, &at)
			})
		generated := make([]float32, len(w))
		if err := kernel.DispatchCooperative(&kernels.TopKMaskKernel, accel.ID3{X: 1},
			kernelabi.Args{Slices: []any{w, generated}, Uniforms: []any{d}}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range generated {
			if authored[i] != generated[i] {
				t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
		// Three tied maxima and k of three, so both forms have to agree about
		// the tie rule rather than only about which values are largest.
		n := 0
		for _, v := range generated {
			if v != 0 {
				n++
			}
		}
		if n != 3 {
			t.Fatalf("kept %d entries for a k of 3", n)
		}
	})

	t.Run("TopPMask", func(t *testing.T) {
		w := []float32{0.1, 0.2, 0.3, 0.4}
		d := kernels.TopDims{Vocab: uint32(len(w)), P: 0.75}
		authored := make([]float32, len(w))
		var best [128]float32
		var at [128]uint32
		kernel.RunAuthored(&kernels.TopPMaskKernel, kernel.ID3{},
			kernel.ID3{X: 1, Y: 1, Z: 1}, 128, func(th kernel.Thread) {
				kernels.TopPMask(th, d, w, authored, &best, &at)
			})
		generated := make([]float32, len(w))
		if err := kernel.DispatchCooperative(&kernels.TopPMaskKernel, accel.ID3{X: 1},
			kernelabi.Args{Slices: []any{w, generated}, Uniforms: []any{d}}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range generated {
			if authored[i] != generated[i] {
				t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
	})

	t.Run("AttentionDecodeBatched", func(t *testing.T) {
		const batch, qHeads, kvHeads, headDim, block, maxPages, blocks = 2, 2, 1, 8, 4, 2, 8
		d := kernels.BatchedDims{
			Batch: batch, QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
			Block: block, MaxPages: maxPages,
			Scale: float32(1) / float32(math.Sqrt(headDim)),
		}
		q := make([]float32, batch*qHeads*headDim)
		pk := make([]float32, blocks*block*kvHeads*headDim)
		pv := make([]float32, blocks*block*kvHeads*headDim)
		for i := range q {
			q[i] = float32(math.Sin(float64(i) * 0.23))
		}
		for i := range pk {
			pk[i] = float32(math.Cos(float64(i) * 0.19))
			pv[i] = float32(i%5) - 2
		}
		pages := []uint32{5, 2, 1, 6}
		lengths := []uint32{6, 3}

		authored := make([]float32, batch*qHeads*headDim)
		for g := range uint32(batch * qHeads) {
			var scores, red [128]float32
			kernel.RunAuthored(&kernels.AttentionDecodeBatchedKernel, kernel.ID3{X: g},
				kernel.ID3{X: batch * qHeads, Y: 1, Z: 1}, 128, func(th kernel.Thread) {
					kernels.AttentionDecodeBatched(th, d, q, pk, pv, pages, lengths,
						authored, &scores, &red)
				})
		}
		generated := make([]float32, batch*qHeads*headDim)
		if err := kernel.DispatchCooperative(&kernels.AttentionDecodeBatchedKernel,
			accel.ID3{X: batch * qHeads},
			kernelabi.Args{
				Slices:   []any{q, pk, pv, pages, lengths, generated},
				Uniforms: []any{d},
			}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range generated {
			if math.Abs(float64(authored[i]-generated[i])) > 1e-5 {
				t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
	})

	t.Run("AttentionDecodePaged", func(t *testing.T) {
		const qHeads, kvHeads, headDim, block, kvLen, blocks = 2, 1, 8, 4, 6, 8
		d := kernels.PagedDims{
			QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
			Block: block, Scale: float32(1) / float32(math.Sqrt(headDim)),
		}
		q := make([]float32, qHeads*headDim)
		pk := make([]float32, blocks*block*kvHeads*headDim)
		pv := make([]float32, blocks*block*kvHeads*headDim)
		for i := range q {
			q[i] = float32(math.Sin(float64(i) * 0.3))
		}
		for i := range pk {
			pk[i] = float32(math.Cos(float64(i) * 0.11))
			pv[i] = float32(i%7) - 3
		}
		// Out of order, so the two forms agree about the addressing.
		pages := []uint32{5, 2}
		lengths := []uint32{kvLen}

		authored := make([]float32, qHeads*headDim)
		for g := range uint32(qHeads) {
			var scores, red [128]float32
			kernel.RunAuthored(&kernels.AttentionDecodePagedKernel, kernel.ID3{X: g},
				kernel.ID3{X: qHeads, Y: 1, Z: 1}, 128, func(th kernel.Thread) {
					kernels.AttentionDecodePaged(th, d, q, pk, pv, pages, lengths,
						authored, &scores, &red)
				})
		}
		generated := make([]float32, qHeads*headDim)
		if err := kernel.DispatchCooperative(&kernels.AttentionDecodePagedKernel,
			accel.ID3{X: qHeads},
			kernelabi.Args{
				Slices: []any{q, pk, pv, pages, lengths, generated}, Uniforms: []any{d},
			}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range generated {
			if math.Abs(float64(authored[i]-generated[i])) > 1e-5 {
				t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
	})

	t.Run("AttentionPrefill", func(t *testing.T) {
		const qHeads, kvHeads, headDim, qSeq = 2, 1, 8, 4
		dims := kernels.PrefillDims{
			QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
			QSeq: qSeq, Base: 0,
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

		// A fresh cache, so its length equals the query sequence.
		lengths := []uint32{qSeq}
		authored := make([]float32, len(q))
		groups := kernel.ID3{X: qSeq * qHeads, Y: 1, Z: 1}
		for g := range groups.X {
			var scores, red [128]float32
			kernel.RunAuthored(&kernels.AttentionPrefillKernel, kernel.ID3{X: g}, groups, 128, func(th kernel.Thread) {
				kernels.AttentionPrefill(th, dims, q, k, v, lengths, authored,
					&scores, &red)
			})
		}

		generated := make([]float32, len(q))
		if err := kernel.DispatchCooperative(&kernels.AttentionPrefillKernel,
			accel.ID3{X: groups.X},
			kernelabi.Args{
				Slices: []any{q, k, v, lengths, generated}, Uniforms: []any{dims},
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

	t.Run("AttentionPrefillPaged", func(t *testing.T) {
		const qHeads, kvHeads, headDim, block, qSeq = 2, 1, 8, 2, 4
		const poolBlocks = 6
		dims := kernels.PagedPrefillDims{
			QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim, QSeq: qSeq,
			Base: 0, Block: block, Scale: float32(1) / float32(math.Sqrt(headDim)),
		}
		q := make([]float32, qSeq*qHeads*headDim)
		pk := make([]float32, poolBlocks*block*kvHeads*headDim)
		pv := make([]float32, len(pk))
		for i := range q {
			q[i] = float32(math.Sin(float64(i) * 0.37))
		}
		for i := range pk {
			pk[i] = float32(math.Cos(float64(i) * 0.13))
			pv[i] = float32(i%9) - 4
		}
		pages := []uint32{4, 1} // out of order, so the addressing is compared
		lengths := []uint32{qSeq}

		authored := make([]float32, len(q))
		groups := kernel.ID3{X: qSeq * qHeads, Y: 1, Z: 1}
		for g := range groups.X {
			var scores, red [128]float32
			kernel.RunAuthored(&kernels.AttentionPrefillPagedKernel, kernel.ID3{X: g},
				groups, 128, func(th kernel.Thread) {
					kernels.AttentionPrefillPaged(th, dims, q, pk, pv, pages,
						lengths, authored, &scores, &red)
				})
		}
		generated := make([]float32, len(q))
		if err := kernel.DispatchCooperative(&kernels.AttentionPrefillPagedKernel,
			accel.ID3{X: groups.X},
			kernelabi.Args{
				Slices:   []any{q, pk, pv, pages, lengths, generated},
				Uniforms: []any{dims},
			}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range generated {
			if math.Abs(float64(authored[i]-generated[i])) > 1e-5 {
				t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
	})

	t.Run("ScatterRowsF16", func(t *testing.T) {
		const capacity, width, rows = 8, 4, 3
		p := kernels.RowParams{Rows: rows, Width: width, Capacity: capacity}
		in := make([]accel.Float16, rows*width)
		for i := range in {
			in[i] = accel.ToFloat16(float32(i)*0.375 - 5)
		}
		// The last id is past the state, so the two forms have to agree about
		// the range check and not merely about the addressing.
		ids := []uint32{5, 0, capacity + 1}

		authored := make([]accel.Float16, capacity*width)
		generated := make([]accel.Float16, capacity*width)
		// A state a scatter does not cover keeps what it held, so both copies
		// start from the same non-zero contents: a dropped write is then
		// visible as the old value rather than hidden by a zero.
		for i := range authored {
			authored[i] = accel.ToFloat16(float32(i) - 20)
			generated[i] = authored[i]
		}

		n := rows * width
		for i := range n {
			kernels.ScatterRowsF16(flatThread(i, n), p, in, ids, authored)
		}
		if err := kernel.Dispatch(&kernels.ScatterRowsF16Kernel, accel.ID3{X: 1},
			kernelabi.Args{
				Slices: []any{in, ids, generated}, Uniforms: []any{p},
			}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range generated {
			if authored[i].Bits() != generated[i].Bits() {
				t.Fatalf("element %d: authored %#04x, generated %#04x",
					i, authored[i].Bits(), generated[i].Bits())
			}
		}
	})

	t.Run("AttentionPrefillF16", func(t *testing.T) {
		const qHeads, kvHeads, headDim, qSeq = 2, 1, 8, 4
		dims := kernels.PrefillDims{
			QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
			QSeq: qSeq, Base: 0,
			Scale: float32(1) / float32(math.Sqrt(headDim)),
		}
		q := make([]float32, qSeq*qHeads*headDim)
		k := make([]accel.Float16, qSeq*kvHeads*headDim)
		v := make([]accel.Float16, qSeq*kvHeads*headDim)
		for i := range q {
			q[i] = float32(math.Sin(float64(i) * 0.29))
		}
		for i := range k {
			k[i] = accel.ToFloat16(float32(math.Cos(float64(i) * 0.13)))
			v[i] = accel.ToFloat16(float32(i%5) - 2)
		}

		lengths := []uint32{qSeq}
		authored := make([]float32, len(q))
		groups := kernel.ID3{X: qSeq * qHeads, Y: 1, Z: 1}
		for g := range groups.X {
			var scores, red [128]float32
			kernel.RunAuthored(&kernels.AttentionPrefillF16Kernel, kernel.ID3{X: g},
				groups, 128, func(th kernel.Thread) {
					kernels.AttentionPrefillF16(th, dims, q, k, v, lengths, authored,
						&scores, &red)
				})
		}

		generated := make([]float32, len(q))
		if err := kernel.DispatchCooperative(&kernels.AttentionPrefillF16Kernel,
			accel.ID3{X: groups.X},
			kernelabi.Args{
				Slices: []any{q, k, v, lengths, generated}, Uniforms: []any{dims},
			}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		// The f32 prefill's ceiling above, unchanged. The two forms differ by
		// the product roundings a contraction saves: the authored Go may fuse a
		// multiply-add and the generated Go's explicit f32 rounding points
		// forbid it (specs/008-numerics.md section 5). The widening adds
		// nothing to that -- TestTheF16PrefillMatchesTheF32OneExactly compares
		// the two storage widths bit for bit.
		for i := range generated {
			if math.Abs(float64(authored[i]-generated[i])) > 1e-5 {
				t.Fatalf("element %d: authored %v, generated %v; both halves of "+
					"specs/004-kernel-authoring.md's fifth level must agree",
					i, authored[i], generated[i])
			}
		}
	})

	t.Run("AttentionDecodePagedF16", func(t *testing.T) {
		const qHeads, kvHeads, headDim, block, kvLen, blocks = 2, 1, 8, 4, 6, 8
		d := kernels.PagedDims{
			QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
			Block: block, Scale: float32(1) / float32(math.Sqrt(headDim)),
		}
		q := make([]float32, qHeads*headDim)
		pk := make([]accel.Float16, blocks*block*kvHeads*headDim)
		pv := make([]accel.Float16, len(pk))
		for i := range q {
			q[i] = float32(math.Sin(float64(i) * 0.3))
		}
		for i := range pk {
			pk[i] = accel.ToFloat16(float32(math.Cos(float64(i) * 0.11)))
			pv[i] = accel.ToFloat16(float32(i%7) - 3)
		}
		// Out of order, so the two forms agree about the addressing.
		pages := []uint32{5, 2}
		lengths := []uint32{kvLen}

		authored := make([]float32, qHeads*headDim)
		for g := range uint32(qHeads) {
			var scores, red [128]float32
			kernel.RunAuthored(&kernels.AttentionDecodePagedF16Kernel, kernel.ID3{X: g},
				kernel.ID3{X: qHeads, Y: 1, Z: 1}, 128, func(th kernel.Thread) {
					kernels.AttentionDecodePagedF16(th, d, q, pk, pv, pages, lengths,
						authored, &scores, &red)
				})
		}
		generated := make([]float32, qHeads*headDim)
		if err := kernel.DispatchCooperative(&kernels.AttentionDecodePagedF16Kernel,
			accel.ID3{X: qHeads},
			kernelabi.Args{
				Slices: []any{q, pk, pv, pages, lengths, generated}, Uniforms: []any{d},
			}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		// The paged f32 decode's ceiling above, unchanged, for the reason given
		// in the prefill case: a contraction the authored form may take and the
		// generated one may not.
		for i := range generated {
			if math.Abs(float64(authored[i]-generated[i])) > 1e-5 {
				t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
	})

	t.Run("GatherRowsF16", func(t *testing.T) {
		const vocab, width, rows = 8, 4, 3
		p := kernels.RowParams{Rows: rows, Width: width, Capacity: vocab}
		table := make([]accel.Float16, vocab*width)
		for i := range table {
			table[i] = accel.ToFloat16(float32(i)*0.375 - 5)
		}
		ids := []uint32{5, 0, vocab + 1} // the last is out of range
		n := rows * width

		authored := make([]float32, n)
		for i := range authored {
			kernels.GatherRowsF16(flatThread(i, n), p, table, ids, authored)
		}
		generated := make([]float32, n)
		if err := kernel.Dispatch(&kernels.GatherRowsF16Kernel, accel.ID3{X: 1},
			kernelabi.Args{
				Slices: []any{table, ids, generated}, Uniforms: []any{p},
			}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range generated {
			if authored[i] != generated[i] {
				t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
	})

	t.Run("QuantMatVec", func(t *testing.T) {
		const n, k = 8, 32
		d := kernels.GEMMDims{M: 1, N: n, K: k}
		a := make([]accel.Float16, k)
		bq := make([]int8, k*n)
		bs := make([]accel.Float16, k*n/kernels.QuantBlock)
		for i := range a {
			a[i] = accel.ToFloat16(float32(math.Sin(float64(i) * 0.21)))
		}
		for i := range bq {
			bq[i] = int8(i%201 - 100)
		}
		for i := range bs {
			bs[i] = accel.ToFloat16(0.25 + float32(i%3)/8)
		}

		// One workgroup per block of MatVecCols columns, which for n = 8 is
		// one group; the lanes are the kernel's 32x4 grid.
		authored := make([]float32, n)
		groups := uint32((n + kernels.MatVecCols - 1) / kernels.MatVecCols)
		for g := range groups {
			var sh [512]float32
			kernel.RunAuthored(&kernels.QuantMatVecKernel, kernel.ID3{X: g},
				kernel.ID3{X: groups, Y: 1, Z: 1}, 128, func(th kernel.Thread) {
					kernels.QuantMatVec(th, d, a, bq, bs, authored, &sh)
				})
		}
		generated := make([]float32, n)
		if err := kernel.DispatchCooperative(&kernels.QuantMatVecKernel,
			accel.ID3{X: (n + kernels.MatVecCols - 1) / kernels.MatVecCols},
			kernelabi.Args{
				Slices: []any{a, bq, bs, generated}, Uniforms: []any{d},
			}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range generated {
			if authored[i] != generated[i] {
				t.Fatalf("column %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
	})

	t.Run("QuantMatVecF32", func(t *testing.T) {
		const n, k = 8, 32
		d := kernels.GEMMDims{M: 1, N: n, K: k}
		a := make([]float32, k)
		bq := make([]int8, k*n)
		bs := make([]accel.Float16, k*n/kernels.QuantBlock)
		// Activations with few significant bits, so that a[k]*(q*s) is exact
		// in f32: q is an int8, s one of three dyadic scales, and the product
		// then rounds once at the add whether or not the authored form fused
		// the multiply into it. The generated lowering wraps every operation
		// in float32() and never fuses; Go on arm64 may fuse the authored
		// form, and with several products per lane the two would differ by
		// an ULP on activations like sin(i) -- a property of fusion, not of
		// the lowering, which is what this test is about.
		for i := range a {
			a[i] = float32(i%7) - 3
		}
		for i := range bq {
			bq[i] = int8(i%201 - 100)
		}
		for i := range bs {
			bs[i] = accel.ToFloat16(0.25 + float32(i%3)/8)
		}

		authored := make([]float32, n)
		groups := uint32((n + kernels.MatVecCols - 1) / kernels.MatVecCols)
		for g := range groups {
			var sh [512]float32
			kernel.RunAuthored(&kernels.QuantMatVecF32Kernel, kernel.ID3{X: g},
				kernel.ID3{X: groups, Y: 1, Z: 1}, 128, func(th kernel.Thread) {
					kernels.QuantMatVecF32(th, d, a, bq, bs, authored, &sh)
				})
		}
		generated := make([]float32, n)
		if err := kernel.DispatchCooperative(&kernels.QuantMatVecF32Kernel,
			accel.ID3{X: (n + kernels.MatVecCols - 1) / kernels.MatVecCols},
			kernelabi.Args{
				Slices: []any{a, bq, bs, generated}, Uniforms: []any{d},
			}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range generated {
			if authored[i] != generated[i] {
				t.Fatalf("column %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
	})

	t.Run("MatMulTiledF32F16", func(t *testing.T) {
		// All three tails, so every guarded edge of the mixed tile runs. A
		// shape that fitted the tile exactly would exercise none of them.
		const m, n, k = 9, 19, 23
		a := make([]float32, m*k)
		b := make([]accel.Float16, k*n)
		for i := range a {
			a[i] = float32((i%13)-6) / 4
		}
		for i := range b {
			b[i] = accel.ToFloat16(float32((i%11)-5) / 2)
		}
		d := kernels.GEMMDims{M: m, N: n, K: k}
		groups := kernel.ID3{
			X: uint32((n + kernels.TileN - 1) / kernels.TileN),
			Y: uint32((m + kernels.TileM - 1) / kernels.TileM),
			Z: 1,
		}

		authored := make([]float32, m*n)
		for gy := range groups.Y {
			for gx := range groups.X {
				var tileA [128]float32
				var tileB [256]accel.Float16
				kernel.RunAuthored(&kernels.MatMulTiledF32F16Kernel, kernel.ID3{X: gx, Y: gy}, groups,
					kernels.TileN*kernels.TileM, func(th kernel.Thread) {
						kernels.MatMulTiledF32F16(th, d, a, b, authored, &tileA, &tileB)
					})
			}
		}
		generated := make([]float32, m*n)
		if err := kernel.DispatchCooperative(&kernels.MatMulTiledF32F16Kernel,
			accel.ID3{X: groups.X, Y: groups.Y, Z: 1},
			kernelabi.Args{Slices: []any{a, b, generated}, Uniforms: []any{d}}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range authored {
			if authored[i] != generated[i] {
				t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
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

// A prefill whose sequence is longer than one workgroup, which the 128-position
// cap made unreachable (accel issue 8): a prefill is how a prompt enters the
// cache, so the cap bounded the prompt as well as the cache.
//
// It also exercises the one thing this kernel does that the decode kernels do
// not: the loop bound is the causal limit, so the first query position scores
// one block and the last scores every block the sequence needs. A bound that
// ignored the limit would still be correct here -- the mask would discard the
// extra work -- so the test that it is *tighter* is the suspension count and
// the reasoning in the kernel, not this. This tests the answer.
func TestPrefillScoresASequenceLongerThanAWorkgroup(t *testing.T) {
	const qHeads, kvHeads, headDim = 4, 2, 32
	for _, qSeq := range []int{129, 300} {
		t.Run(fmt.Sprint(qSeq), func(t *testing.T) {
			const base = 0
			kvLen := qSeq
			scale := 1 / math.Sqrt(headDim)
			dims := kernels.PrefillDims{
				QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
				QSeq: uint32(qSeq), Base: base, Scale: float32(scale),
			}
			rng := rand.New(rand.NewPCG(3, 12))
			fill := func(n int) []float32 {
				s := make([]float32, n)
				for i := range s {
					s[i] = float32(rng.NormFloat64())
				}
				return s
			}
			q := fill(qSeq * qHeads * headDim)
			k := fill(kvLen * kvHeads * headDim)
			v := fill(kvLen * kvHeads * headDim)
			lengths := []uint32{uint32(kvLen)}

			got := make([]float32, len(q))
			if err := kernel.DispatchCooperative(&kernels.AttentionPrefillKernel,
				accel.ID3{X: uint32(qSeq * qHeads)},
				kernelabi.Args{
					Slices: []any{q, k, v, lengths, got}, Uniforms: []any{dims},
				}); err != nil {
				t.Fatalf("dispatch: %v", err)
			}

			want := prefillReference(q, k, v, qHeads, kvHeads, headDim, qSeq, kvLen,
				base, scale)

			// The bound derived in
			// TestAttentionDecodeScoresACacheLongerThanAWorkgroup, evaluated at
			// the longest row: the last query position sees every cached one.
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
					t.Fatalf("element %d is %v, want %v: off by %g, tolerance %g",
						i, got[i], want[i], diff, tol)
				}
			}
		})
	}
}

// A paged prefill produces what the contiguous one does over the same logical
// positions, and its sequence is longer than one block.
//
// accel issue 10: this kernel did not exist, and `Attention` accepted a page
// table on a prefill and ignored it. A paged decode is only useful over blocks
// a paged prefill wrote, so its absence made cross-request prefix sharing
// inexpressible.
//
// The pool is four times the sequence and the table is scattered, so a kernel
// that walked the pool in order reads the wrong blocks rather than the right
// ones in a different order -- which is exactly what the ignored-Pages bug did.
func TestPagedPrefillMatchesTheContiguousOne(t *testing.T) {
	const qHeads, kvHeads, headDim, block = 4, 2, 16, 8
	const qSeq, base = 300, 0
	const kvLen = base + qSeq
	const pageCount = (kvLen + block - 1) / block
	const poolBlocks = 4 * pageCount

	rng := rand.New(rand.NewPCG(77, 5))
	fill := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32(rng.NormFloat64())
		}
		return s
	}
	q := fill(qSeq * qHeads * headDim)
	pk := fill(poolBlocks * block * kvHeads * headDim)
	pv := fill(poolBlocks * block * kvHeads * headDim)
	pages := make([]uint32, pageCount)
	for i, p := range rng.Perm(poolBlocks)[:pageCount] {
		pages[i] = uint32(p)
	}
	lengths := []uint32{kvLen}
	scale := float32(1 / math.Sqrt(headDim))

	paged := make([]float32, len(q))
	if err := kernel.DispatchCooperative(&kernels.AttentionPrefillPagedKernel,
		accel.ID3{X: qSeq * qHeads},
		kernelabi.Args{
			Slices: []any{q, pk, pv, pages, lengths, paged},
			Uniforms: []any{kernels.PagedPrefillDims{
				QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim, QSeq: qSeq,
				Base: base, Block: block, Scale: scale,
			}},
		}); err != nil {
		t.Fatalf("paged dispatch: %v", err)
	}

	// The same positions gathered into a contiguous cache, through the
	// contiguous kernel. Exactly equal, not within a budget: paging is an
	// addressing change, so the two read the same values in the same order and
	// compute the same sums.
	gk := make([]float32, kvLen*kvHeads*headDim)
	gv := make([]float32, kvLen*kvHeads*headDim)
	w := kvHeads * headDim
	for j := range kvLen {
		phys := int(pages[j/block])*block + j%block
		copy(gk[j*w:(j+1)*w], pk[phys*w:(phys+1)*w])
		copy(gv[j*w:(j+1)*w], pv[phys*w:(phys+1)*w])
	}
	contiguous := make([]float32, len(q))
	if err := kernel.DispatchCooperative(&kernels.AttentionPrefillKernel,
		accel.ID3{X: qSeq * qHeads},
		kernelabi.Args{
			Slices: []any{q, gk, gv, lengths, contiguous},
			Uniforms: []any{kernels.PrefillDims{
				QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim, QSeq: qSeq,
				Base: base, Scale: scale,
			}},
		}); err != nil {
		t.Fatalf("contiguous dispatch: %v", err)
	}

	for i := range paged {
		if paged[i] != contiguous[i] {
			t.Fatalf("element %d is %v paged and %v contiguous; a page table is an "+
				"addressing, so the two answers are the same answer", i, paged[i],
				contiguous[i])
		}
	}
}
