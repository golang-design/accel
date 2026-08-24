// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import (
	"golang.design/x/accel"
	"golang.design/x/accel/kmath"
)

// AttnDims is a decode-step attention's shape.
type AttnDims struct {
	// QHeads and KVHeads are the query and key/value head counts. QHeads must
	// be a multiple of KVHeads: several query heads share one KV head, which is
	// what grouped-query attention is and what makes a decode cache small
	// enough to keep.
	QHeads  uint32
	KVHeads uint32

	// HeadDim is divisible by eight and at most 128, per spec 010.
	HeadDim uint32

	// Scale multiplies the scores, conventionally 1/sqrt(headDim). It is a
	// parameter rather than computed, because a model may ship a different one
	// and computing it here would silently override what the weights expect.
	Scale float32
}

// AttnBlock is how many cached positions one pass of the loop scores.
//
// The workgroup size, because each lane scores one position of the block. It
// bounds the *shared arrays*, not the cache: the cache is walked a block at a
// time and the running softmax carries the state between passes, so a cache of
// any length is scored by the same kernel.
//
// It was AttnMaxKV, a cap on the whole cache, until a consumer reported that
// 128 positions serve no model (accel issue 8).
const AttnBlock = 128

// AttentionDecode is one decode step of attention, fused.
//
//	out[h] = Σⱼ softmax(qₕ·kⱼ · scale)ⱼ · vⱼ
//
// # Why fused rather than composed
//
// The composed path is MatMul → Softmax → MatMul, and it writes the whole score
// matrix to memory between the first two. For a decode step that matrix is one
// row per head and the traffic dominates the arithmetic. Fusing keeps the scores
// in shared memory, which is the entire reason this kernel exists.
//
// It is not mandatory in principle. Spec 010 makes the composed path the
// definition and this an optional selection -- but the composed path is
// expressible only when kvHeads is 1, because it needs a batched matrix
// multiply per head and specs/025-tensor-operators.md multiplies two matrices.
// specs/007-tensor-layer.md is corrected to say so.
//
// # How a cache longer than a workgroup is scored
//
// A block at a time, carrying a running softmax. Let the cache be blocks
// B₀..Bₙ of AttnBlock positions. After block Bₜ the kernel holds
//
//	m  = max over the positions seen so far
//	ℓ  = Σ exp(sⱼ - m)          over the same positions
//	o  = Σ exp(sⱼ - m)·vⱼ        over the same positions
//
// and out = o/ℓ once every block is in. Admitting block Bₜ₊₁ with maximum m'
// needs the old terms put on the new scale, which is one multiply each:
//
//	mₙₑᵥ = max(m, m')
//	α    = exp(m - mₙₑᵥ)
//	ℓ    ← α·ℓ + Σ exp(sⱼ - mₙₑᵥ)
//	o    ← α·o + Σ exp(sⱼ - mₙₑᵥ)·vⱼ
//
// One block is the closed form: m starts at -3.4e38, so α is exp(-∞) = 0, the
// empty accumulators are multiplied away, and ℓ and o are exactly the single-pass
// kernel's sum and weighted sum, term for term and in the same order. That is
// deliberate -- every test written against the 128-position form keeps its exact
// numbers, so this is a longer reach rather than a different answer.
//
// # The loop bound is a length, not a load
//
// specs/002-compute-model.md section 3.3 makes every load non-uniform, so a
// loop bounded by lengths[0] cannot hold a barrier and this loop has fourteen.
// The bound is therefore the *binding's* extent, which len() reports and
// section 3.3 seeds as workgroup-uniform: it is fixed when the node is
// recorded, and no aliased write can change it the way one can change a
// buffer's contents. Positions past kvLen are masked per lane and cost the
// barriers of an empty block, which is the price of the uniform bound.
//
// One workgroup per query head.
//
//accel:kernel workgroup=128
func AttentionDecode(t accel.Thread, d AttnDims, q []float32, k []float32, v []float32,
	lengths []uint32, out []float32, scores *[AttnBlock]float32, red *[AttnBlock]float32,
	acc *[AttnBlock]float32) {
	h := t.GroupID().X
	// One sequence, so its length is the first entry. The length is a binding
	// rather than a uniform because specs/043-per-row-values.md makes a
	// per-sequence value device data -- and a single sequence is the same path
	// as a batch of one rather than a different one.
	kvLen := lengths[0]
	lane := t.LocalID().X

	// Several query heads share one KV head. Integer division rather than a
	// lookup: the grouping is contiguous by construction, which is what lets a
	// cache hold one entry per KV head rather than per query head.
	group := d.QHeads / d.KVHeads
	kvHead := h / group

	// The cache's capacity, from the binding rather than from a load. See the
	// note above on why the bound cannot be kvLen.
	capacity := uint32(len(k)) / (d.KVHeads * d.HeadDim)

	// The running softmax. m and l are per-lane copies of one value: every lane
	// computes them from the same reduced quantities, so they agree without
	// being shared, and a shared copy would need its own barrier to publish.
	m := float32(-3.4e38)
	l := float32(0)
	if lane < d.HeadDim {
		acc[lane] = 0
	}
	t.Barrier()

	for base := uint32(0); base < capacity; base += AttnBlock {
		pos := base + lane

		// Score this lane's position. Out of range contributes negative
		// infinity to the maximum rather than zero: a zero would win over a row
		// of genuinely negative scores and shift every exponent, which changes
		// the answer rather than only the intermediate.
		s := float32(-3.4e38)
		if pos < kvLen {
			phys := pos*d.KVHeads*d.HeadDim + kvHead*d.HeadDim
			dot := float32(0)
			for i := uint32(0); i < d.HeadDim; i++ {
				dot = dot + q[h*d.HeadDim+i]*k[phys+i]
			}
			s = dot * d.Scale
		}

		// The shared arrays are loop-carried: the previous pass reads scores in
		// its weighted sum and reads red[0] for its total, and both are about to
		// be overwritten. Nothing else orders those reads against these writes
		// -- the CPU backend's rendezvous check finds an invocation that fails
		// to *arrive*, and this would be a race between arrivals rather than a
		// missing one, so it would not be caught there or by a differential
		// that happens to schedule the passes apart.
		t.Barrier()
		scores[lane] = s
		red[lane] = s
		t.Barrier()

		for stride := uint32(AttnBlock / 2); stride > 0; stride /= 2 {
			if lane < stride {
				red[lane] = kmath.Max(red[lane], red[lane+stride])
			}
			t.Barrier()
		}
		blockMax := red[0]

		// Put the carried terms on the new scale. On the first block m is
		// -3.4e38 and alpha is zero, which multiplies away accumulators that
		// are already zero; on a block entirely past kvLen blockMax is also
		// -3.4e38, alpha is one, and nothing changes.
		next := kmath.Max(m, blockMax)
		alpha := kmath.Exp(m - next)

		// A lane past the cache contributes zero, which for a *sum* is the
		// identity and therefore correct where it was not for the maximum.
		e := float32(0)
		if pos < kvLen {
			e = kmath.Exp(s - next)
		}
		t.Barrier()
		scores[lane] = e
		red[lane] = e
		t.Barrier()

		for stride := uint32(AttnBlock / 2); stride > 0; stride /= 2 {
			if lane < stride {
				red[lane] = red[lane] + red[lane+stride]
			}
			t.Barrier()
		}
		l = alpha*l + red[0]
		m = next

		// The weighted sum of V, parallel over the head's dimensions rather
		// than over the cache: the probabilities are in shared memory now, so
		// every lane can read all of them and each owns one output element.
		// Each lane owns acc[lane] across every pass, so the accumulator needs
		// no barrier of its own.
		if lane < d.HeadDim {
			a := alpha * acc[lane]
			for j := uint32(0); j < AttnBlock; j++ {
				if base+j < kvLen {
					a = a + scores[j]*v[(base+j)*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
				}
			}
			acc[lane] = a
		}
	}

	if lane < d.HeadDim {
		out[h*d.HeadDim+lane] = acc[lane] / l
	}
}

// AttentionDecodeF16 is [AttentionDecode] over an f16 cache.
//
// # Why a narrow cache is defensible here and not in general
//
// specs/002-compute-model.md's rule is the right default: a narrow type is
// storage that converts on load, and accumulating a long dot product in f16
// loses accuracy badly. That argument is about the **accumulator**, and K and V
// are *operands*.
//
// softmax(qKᵀ/√d)·V accumulates in f32 whatever the operands are stored as --
// which is exactly what MatMulTiled already does when it reads f16 and
// accumulates f32. This applies the same trade to the one buffer where it is
// worth the most: the KV cache is the largest allocation in a serving process
// after the weights, and the only one that scales with both concurrency and
// context. A consumer reported an f32 cache at 9.66 GB for a single 32k
// sequence, larger than the int8 weights of the model it serves.
//
// The body is [AttentionDecode]'s with two loads widened. It is a separate
// kernel rather than a dtype parameter because specs/004-kernel-authoring.md
// keeps generics out of the subset, and a variant is what
// specs/010-kernel-corpus.md registers.
//
//accel:kernel workgroup=128
func AttentionDecodeF16(t accel.Thread, d AttnDims, q []float32, k []accel.Float16,
	v []accel.Float16, lengths []uint32,
	out []float32, scores *[AttnBlock]float32, red *[AttnBlock]float32,
	acc *[AttnBlock]float32) {
	h := t.GroupID().X
	// One sequence, so its length is the first entry. The length is a binding
	// rather than a uniform because specs/043-per-row-values.md makes a
	// per-sequence value device data -- and a single sequence is the same path
	// as a batch of one rather than a different one.
	kvLen := lengths[0]
	lane := t.LocalID().X

	// Several query heads share one KV head. Integer division rather than a
	// lookup: the grouping is contiguous by construction, which is what lets a
	// cache hold one entry per KV head rather than per query head.
	group := d.QHeads / d.KVHeads
	kvHead := h / group

	// The cache's capacity, from the binding rather than from a load. See the
	// note above on why the bound cannot be kvLen.
	capacity := uint32(len(k)) / (d.KVHeads * d.HeadDim)

	// The running softmax. m and l are per-lane copies of one value: every lane
	// computes them from the same reduced quantities, so they agree without
	// being shared, and a shared copy would need its own barrier to publish.
	m := float32(-3.4e38)
	l := float32(0)
	if lane < d.HeadDim {
		acc[lane] = 0
	}
	t.Barrier()

	for base := uint32(0); base < capacity; base += AttnBlock {
		pos := base + lane

		// Score this lane's position. Out of range contributes negative
		// infinity to the maximum rather than zero: a zero would win over a row
		// of genuinely negative scores and shift every exponent, which changes
		// the answer rather than only the intermediate.
		s := float32(-3.4e38)
		if pos < kvLen {
			phys := pos*d.KVHeads*d.HeadDim + kvHead*d.HeadDim
			dot := float32(0)
			for i := uint32(0); i < d.HeadDim; i++ {
				dot = dot + q[h*d.HeadDim+i]*k[phys+i].F32()
			}
			s = dot * d.Scale
		}

		// The shared arrays are loop-carried: the previous pass reads scores in
		// its weighted sum and reads red[0] for its total, and both are about to
		// be overwritten. Nothing else orders those reads against these writes
		// -- the CPU backend's rendezvous check finds an invocation that fails
		// to *arrive*, and this would be a race between arrivals rather than a
		// missing one, so it would not be caught there or by a differential
		// that happens to schedule the passes apart.
		t.Barrier()
		scores[lane] = s
		red[lane] = s
		t.Barrier()

		for stride := uint32(AttnBlock / 2); stride > 0; stride /= 2 {
			if lane < stride {
				red[lane] = kmath.Max(red[lane], red[lane+stride])
			}
			t.Barrier()
		}
		blockMax := red[0]

		// Put the carried terms on the new scale. On the first block m is
		// -3.4e38 and alpha is zero, which multiplies away accumulators that
		// are already zero; on a block entirely past kvLen blockMax is also
		// -3.4e38, alpha is one, and nothing changes.
		next := kmath.Max(m, blockMax)
		alpha := kmath.Exp(m - next)

		// A lane past the cache contributes zero, which for a *sum* is the
		// identity and therefore correct where it was not for the maximum.
		e := float32(0)
		if pos < kvLen {
			e = kmath.Exp(s - next)
		}
		t.Barrier()
		scores[lane] = e
		red[lane] = e
		t.Barrier()

		for stride := uint32(AttnBlock / 2); stride > 0; stride /= 2 {
			if lane < stride {
				red[lane] = red[lane] + red[lane+stride]
			}
			t.Barrier()
		}
		l = alpha*l + red[0]
		m = next

		// The weighted sum of V, parallel over the head's dimensions rather
		// than over the cache: the probabilities are in shared memory now, so
		// every lane can read all of them and each owns one output element.
		// Each lane owns acc[lane] across every pass, so the accumulator needs
		// no barrier of its own.
		if lane < d.HeadDim {
			a := alpha * acc[lane]
			for j := uint32(0); j < AttnBlock; j++ {
				if base+j < kvLen {
					a = a + scores[j]*v[(base+j)*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane].F32()
				}
			}
			acc[lane] = a
		}
	}

	if lane < d.HeadDim {
		out[h*d.HeadDim+lane] = acc[lane] / l
	}
}
