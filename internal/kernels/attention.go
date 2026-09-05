// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

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
// # The loop bound is the length, declared uniform
//
// specs/002-compute-model.md section 3.3 makes every load non-uniform, so
// until specs/063-uniform-loads.md a loop bounded by lengths[0] could not
// hold a barrier, and this loop has fourteen: the bound was the binding's
// extent, and every position past the length cost the barriers of an empty
// block. At a 4096-position capacity and a 256-position sequence that was
// 533 µs per layer against 198 µs, measured on an M2. The lengths table is
// the host's routing data, no invocation of the dispatch writes it, and
// //accel:uniform says so; the loop stops at the length.
//
// One workgroup per query head.
//
//accel:uniform lengths
//accel:requires subgroup_arithmetic, subgroup_basic
//accel:kernel workgroup=128
func AttentionDecode(t accel.Thread, d AttnDims, q []float32, k []float32, v []float32,
	lengths []uint32, out []float32, scores *[AttnBlock]float32, red *[AttnBlock]float32) {
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

	// The length is clamped to what the binding can reach. A caller's length is
	// device data and nothing above has checked it against the cache -- and the
	// loop bound below limits `base`, not `base+lane`, so an unclamped length
	// past the reach is scored by the lanes of the last block rather than
	// stopped by the loop. For the paged kernels that means reading the *next*
	// sequence's page-table row, which is the failure
	// specs/040-batch-scheduler.md names; for the contiguous ones it is a read
	// past the end of the cache.
	//
	// Clamping truncates: the answer attends over a prefix. That is wrong, and
	// it is the wrong that can be bounded here -- the kernel cannot tell a
	// length that is too large from one that is right.
	if kvLen > capacity {
		kvLen = capacity
	}

	// The running softmax. m and l are per-lane copies of one value: every lane
	// computes them from the same reduced quantities, so they agree without
	// being shared, and a shared copy would need its own barrier to publish.
	m := float32(-3.4e38)
	l := float32(0)

	// The output accumulator is a local, not a shared array: each lane owns
	// exactly one element of the row and no other lane reads it, so there is
	// nothing to publish and no barrier to pay. The resumable lowering carries
	// a local across a suspension point, which is what makes this available
	// inside a loop that holds barriers.
	o := float32(0)

	for base := uint32(0); base < kvLen; base += AttnBlock {
		pos := base + lane

		// Score this lane's position. Out of range contributes negative
		// infinity to the maximum rather than zero: a zero would win over a row
		// of genuinely negative scores and shift every exponent, which changes
		// the answer rather than only the intermediate.
		// The shared arrays are loop-carried: the previous pass reads scores in
		// its weighted sum and reads red[0] for its total, and both are about to
		// be overwritten. Nothing else orders those reads against these writes
		// -- the CPU backend's rendezvous check finds an invocation that fails
		// to *arrive*, and this would be a race between arrivals rather than a
		// missing one, so it would not be caught there or by a differential
		// that happens to schedule the passes apart.
		t.Barrier()

		// The block's scores, with the lanes along the head dimension rather
		// than along the positions. Each subgroup takes the positions
		// congruent to its index; within it, lane l holds the products at
		// l, l+w, l+2w, ... of the row and the subgroup sum is the dot. For
		// one position the subgroup's lanes read one contiguous row segment,
		// where a lane per position read rows KVHeads*HeadDim floats apart:
		// the decode step's K reads were the least coalesced access the
		// kernel made, and this is what moved the kernel off 10 GB/s. The
		// lane count is the device's, so the loop is written against
		// SubgroupSize rather than a literal and the CPU oracle's width of
		// four runs the same code as Metal's thirty-two.
		sgw := t.SubgroupSize()
		sgl := t.SubgroupLane()
		nsg := AttnBlock / sgw
		// Four positions per round, their loads issued before any of the four
		// sums: a round's loads are independent of the round before, and a
		// subgroup that waited on one position's sum before loading the next
		// paid the memory latency once per position. 128 is a multiple of
		// 4*nsg for every subgroup width the model admits.
		for j := t.SubgroupIndex(); j < AttnBlock; j += 4 * nsg {
			jpos0 := base + j + 0*nsg
			part0 := float32(0)
			if jpos0 < kvLen {
				phys0 := jpos0*d.KVHeads*d.HeadDim + kvHead*d.HeadDim
				// Four independent partials over the row: one chain of dependent
				// adds through a load each was the phase's latency, since a lane
				// waited on each load before issuing the next.
				pa0 := float32(0)
				pb0 := float32(0)
				pc0 := float32(0)
				pd0 := float32(0)
				i := sgl
				for i+3*sgw < d.HeadDim {
					pa0 = pa0 + q[h*d.HeadDim+i]*k[phys0+i]
					pb0 = pb0 + q[h*d.HeadDim+i+sgw]*k[phys0+i+sgw]
					pc0 = pc0 + q[h*d.HeadDim+i+2*sgw]*k[phys0+i+2*sgw]
					pd0 = pd0 + q[h*d.HeadDim+i+3*sgw]*k[phys0+i+3*sgw]
					i = i + 4*sgw
				}
				for i < d.HeadDim {
					pa0 = pa0 + q[h*d.HeadDim+i]*k[phys0+i]
					i = i + sgw
				}
				part0 = (pa0 + pb0) + (pc0 + pd0)
			}
			jpos1 := base + j + 1*nsg
			part1 := float32(0)
			if jpos1 < kvLen {
				phys1 := jpos1*d.KVHeads*d.HeadDim + kvHead*d.HeadDim
				// Four independent partials over the row: one chain of dependent
				// adds through a load each was the phase's latency, since a lane
				// waited on each load before issuing the next.
				pa1 := float32(0)
				pb1 := float32(0)
				pc1 := float32(0)
				pd1 := float32(0)
				i := sgl
				for i+3*sgw < d.HeadDim {
					pa1 = pa1 + q[h*d.HeadDim+i]*k[phys1+i]
					pb1 = pb1 + q[h*d.HeadDim+i+sgw]*k[phys1+i+sgw]
					pc1 = pc1 + q[h*d.HeadDim+i+2*sgw]*k[phys1+i+2*sgw]
					pd1 = pd1 + q[h*d.HeadDim+i+3*sgw]*k[phys1+i+3*sgw]
					i = i + 4*sgw
				}
				for i < d.HeadDim {
					pa1 = pa1 + q[h*d.HeadDim+i]*k[phys1+i]
					i = i + sgw
				}
				part1 = (pa1 + pb1) + (pc1 + pd1)
			}
			jpos2 := base + j + 2*nsg
			part2 := float32(0)
			if jpos2 < kvLen {
				phys2 := jpos2*d.KVHeads*d.HeadDim + kvHead*d.HeadDim
				// Four independent partials over the row: one chain of dependent
				// adds through a load each was the phase's latency, since a lane
				// waited on each load before issuing the next.
				pa2 := float32(0)
				pb2 := float32(0)
				pc2 := float32(0)
				pd2 := float32(0)
				i := sgl
				for i+3*sgw < d.HeadDim {
					pa2 = pa2 + q[h*d.HeadDim+i]*k[phys2+i]
					pb2 = pb2 + q[h*d.HeadDim+i+sgw]*k[phys2+i+sgw]
					pc2 = pc2 + q[h*d.HeadDim+i+2*sgw]*k[phys2+i+2*sgw]
					pd2 = pd2 + q[h*d.HeadDim+i+3*sgw]*k[phys2+i+3*sgw]
					i = i + 4*sgw
				}
				for i < d.HeadDim {
					pa2 = pa2 + q[h*d.HeadDim+i]*k[phys2+i]
					i = i + sgw
				}
				part2 = (pa2 + pb2) + (pc2 + pd2)
			}
			jpos3 := base + j + 3*nsg
			part3 := float32(0)
			if jpos3 < kvLen {
				phys3 := jpos3*d.KVHeads*d.HeadDim + kvHead*d.HeadDim
				// Four independent partials over the row: one chain of dependent
				// adds through a load each was the phase's latency, since a lane
				// waited on each load before issuing the next.
				pa3 := float32(0)
				pb3 := float32(0)
				pc3 := float32(0)
				pd3 := float32(0)
				i := sgl
				for i+3*sgw < d.HeadDim {
					pa3 = pa3 + q[h*d.HeadDim+i]*k[phys3+i]
					pb3 = pb3 + q[h*d.HeadDim+i+sgw]*k[phys3+i+sgw]
					pc3 = pc3 + q[h*d.HeadDim+i+2*sgw]*k[phys3+i+2*sgw]
					pd3 = pd3 + q[h*d.HeadDim+i+3*sgw]*k[phys3+i+3*sgw]
					i = i + 4*sgw
				}
				for i < d.HeadDim {
					pa3 = pa3 + q[h*d.HeadDim+i]*k[phys3+i]
					i = i + sgw
				}
				part3 = (pa3 + pb3) + (pc3 + pd3)
			}
			dot0 := t.SubgroupAddF32(part0)
			dot1 := t.SubgroupAddF32(part1)
			dot2 := t.SubgroupAddF32(part2)
			dot3 := t.SubgroupAddF32(part3)
			if sgl == 0 {
				sj0 := float32(-3.4e38)
				if jpos0 < kvLen {
					sj0 = dot0 * d.Scale
				}
				scores[j+0*nsg] = sj0
				sj1 := float32(-3.4e38)
				if jpos1 < kvLen {
					sj1 = dot1 * d.Scale
				}
				scores[j+1*nsg] = sj1
				sj2 := float32(-3.4e38)
				if jpos2 < kvLen {
					sj2 = dot2 * d.Scale
				}
				scores[j+2*nsg] = sj2
				sj3 := float32(-3.4e38)
				if jpos3 < kvLen {
					sj3 = dot3 * d.Scale
				}
				scores[j+3*nsg] = sj3
			}
		}
		t.Barrier()
		s := scores[lane]

		// The block's maximum and the block's mass: a subgroup reduction each,
		// one partial per subgroup into red, one barrier, and every lane folds
		// the partials. The shared-memory trees they replace cost seven
		// barriers apiece, and with one workgroup per head on a device that
		// wants many, the barriers were the block's latency rather than its
		// loads: the time per decode was flat from 8 to 64 workgroups.
		sgMax := t.SubgroupMaxF32(s)
		if sgl == 0 {
			red[t.SubgroupIndex()] = sgMax
		}
		t.Barrier()
		blockMax := red[0]
		for g := uint32(1); g < nsg; g++ {
			blockMax = kmath.Max(blockMax, red[g])
		}

		next := kmath.Max(m, blockMax)
		alpha := kmath.Exp(m - next)

		e := float32(0)
		if pos < kvLen {
			e = kmath.Exp(s - next)
		}
		sgSum := t.SubgroupAddF32(e)
		// Every lane has read red for the maximum before any lane writes the
		// masses into it; scores likewise was read into s by every lane
		// before the weights overwrite it.
		t.Barrier()
		scores[lane] = e
		if sgl == 0 {
			red[t.SubgroupIndex()] = sgSum
		}
		t.Barrier()
		blockSum := red[0]
		for g := uint32(1); g < nsg; g++ {
			blockSum = blockSum + red[g]
		}
		l = alpha*l + blockSum
		m = next

		// The weighted sum of V, parallel over the head's dimensions rather
		// than over the cache: the probabilities are in shared memory now, so
		// every lane can read all of them and each owns one output element.
		//
		// The bound on j is a range check and not a mask. A position past the
		// cache already has a probability of zero, so dropping it changes no
		// arithmetic -- it reads V past its end, which the CPU backend reports
		// as an out-of-range index and a device would not report at all.
		if lane < d.HeadDim {
			o = alpha * o
			// Four positions per step into four independent accumulators, so
			// four loads are in flight where one dependent chain of adds through
			// a load each let one be. The masked positions past the length are
			// zero-weight and skipped by the guard as before.
			o0 := float32(0)
			o1 := float32(0)
			o2 := float32(0)
			o3 := float32(0)
			for j := uint32(0); j < AttnBlock; j += 4 {
				jj0 := base + j + 0
				if jj0 < kvLen {
					o0 = o0 + scores[j+0]*v[(jj0)*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
				}
				jj1 := base + j + 1
				if jj1 < kvLen {
					o1 = o1 + scores[j+1]*v[(jj1)*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
				}
				jj2 := base + j + 2
				if jj2 < kvLen {
					o2 = o2 + scores[j+2]*v[(jj2)*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
				}
				jj3 := base + j + 3
				if jj3 < kvLen {
					o3 = o3 + scores[j+3]*v[(jj3)*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
				}
			}
			o = o + ((o0 + o1) + (o2 + o3))
		}
	}

	if lane < d.HeadDim {
		out[h*d.HeadDim+lane] = o / l
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
//accel:uniform lengths
//accel:requires subgroup_arithmetic, subgroup_basic
//accel:kernel workgroup=128
func AttentionDecodeF16(t accel.Thread, d AttnDims, q []float32, k []accel.Float16,
	v []accel.Float16, lengths []uint32,
	out []float32, scores *[AttnBlock]float32, red *[AttnBlock]float32) {
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
	// Clamped for the reason the f32 form clamps: a length past the cache's
	// reach would read another sequence's blocks.
	if kvLen > capacity {
		kvLen = capacity
	}

	// The running softmax. m and l are per-lane copies of one value: every lane
	// computes them from the same reduced quantities, so they agree without
	// being shared, and a shared copy would need its own barrier to publish.
	m := float32(-3.4e38)
	l := float32(0)

	// The output accumulator is a local, not a shared array: each lane owns
	// exactly one element of the row and no other lane reads it, so there is
	// nothing to publish and no barrier to pay. The resumable lowering carries
	// a local across a suspension point, which is what makes this available
	// inside a loop that holds barriers.
	o := float32(0)

	for base := uint32(0); base < kvLen; base += AttnBlock {
		pos := base + lane

		// Score this lane's position. Out of range contributes negative
		// infinity to the maximum rather than zero: a zero would win over a row
		// of genuinely negative scores and shift every exponent, which changes
		// the answer rather than only the intermediate.
		// The shared arrays are loop-carried: the previous pass reads scores in
		// its weighted sum and reads red[0] for its total, and both are about to
		// be overwritten. Nothing else orders those reads against these writes
		// -- the CPU backend's rendezvous check finds an invocation that fails
		// to *arrive*, and this would be a race between arrivals rather than a
		// missing one, so it would not be caught there or by a differential
		// that happens to schedule the passes apart.
		t.Barrier()

		// The block's scores, with the lanes along the head dimension rather
		// than along the positions. Each subgroup takes the positions
		// congruent to its index; within it, lane l holds the products at
		// l, l+w, l+2w, ... of the row and the subgroup sum is the dot. For
		// one position the subgroup's lanes read one contiguous row segment,
		// where a lane per position read rows KVHeads*HeadDim floats apart:
		// the decode step's K reads were the least coalesced access the
		// kernel made, and this is what moved the kernel off 10 GB/s. The
		// lane count is the device's, so the loop is written against
		// SubgroupSize rather than a literal and the CPU oracle's width of
		// four runs the same code as Metal's thirty-two.
		sgw := t.SubgroupSize()
		sgl := t.SubgroupLane()
		nsg := AttnBlock / sgw
		// Four positions per round, their loads issued before any of the four
		// sums: a round's loads are independent of the round before, and a
		// subgroup that waited on one position's sum before loading the next
		// paid the memory latency once per position. 128 is a multiple of
		// 4*nsg for every subgroup width the model admits.
		for j := t.SubgroupIndex(); j < AttnBlock; j += 4 * nsg {
			jpos0 := base + j + 0*nsg
			part0 := float32(0)
			if jpos0 < kvLen {
				phys0 := jpos0*d.KVHeads*d.HeadDim + kvHead*d.HeadDim
				// Four independent partials over the row: one chain of dependent
				// adds through a load each was the phase's latency, since a lane
				// waited on each load before issuing the next.
				pa0 := float32(0)
				pb0 := float32(0)
				pc0 := float32(0)
				pd0 := float32(0)
				i := sgl
				for i+3*sgw < d.HeadDim {
					pa0 = pa0 + q[h*d.HeadDim+i]*k[phys0+i].F32()
					pb0 = pb0 + q[h*d.HeadDim+i+sgw]*k[phys0+i+sgw].F32()
					pc0 = pc0 + q[h*d.HeadDim+i+2*sgw]*k[phys0+i+2*sgw].F32()
					pd0 = pd0 + q[h*d.HeadDim+i+3*sgw]*k[phys0+i+3*sgw].F32()
					i = i + 4*sgw
				}
				for i < d.HeadDim {
					pa0 = pa0 + q[h*d.HeadDim+i]*k[phys0+i].F32()
					i = i + sgw
				}
				part0 = (pa0 + pb0) + (pc0 + pd0)
			}
			jpos1 := base + j + 1*nsg
			part1 := float32(0)
			if jpos1 < kvLen {
				phys1 := jpos1*d.KVHeads*d.HeadDim + kvHead*d.HeadDim
				// Four independent partials over the row: one chain of dependent
				// adds through a load each was the phase's latency, since a lane
				// waited on each load before issuing the next.
				pa1 := float32(0)
				pb1 := float32(0)
				pc1 := float32(0)
				pd1 := float32(0)
				i := sgl
				for i+3*sgw < d.HeadDim {
					pa1 = pa1 + q[h*d.HeadDim+i]*k[phys1+i].F32()
					pb1 = pb1 + q[h*d.HeadDim+i+sgw]*k[phys1+i+sgw].F32()
					pc1 = pc1 + q[h*d.HeadDim+i+2*sgw]*k[phys1+i+2*sgw].F32()
					pd1 = pd1 + q[h*d.HeadDim+i+3*sgw]*k[phys1+i+3*sgw].F32()
					i = i + 4*sgw
				}
				for i < d.HeadDim {
					pa1 = pa1 + q[h*d.HeadDim+i]*k[phys1+i].F32()
					i = i + sgw
				}
				part1 = (pa1 + pb1) + (pc1 + pd1)
			}
			jpos2 := base + j + 2*nsg
			part2 := float32(0)
			if jpos2 < kvLen {
				phys2 := jpos2*d.KVHeads*d.HeadDim + kvHead*d.HeadDim
				// Four independent partials over the row: one chain of dependent
				// adds through a load each was the phase's latency, since a lane
				// waited on each load before issuing the next.
				pa2 := float32(0)
				pb2 := float32(0)
				pc2 := float32(0)
				pd2 := float32(0)
				i := sgl
				for i+3*sgw < d.HeadDim {
					pa2 = pa2 + q[h*d.HeadDim+i]*k[phys2+i].F32()
					pb2 = pb2 + q[h*d.HeadDim+i+sgw]*k[phys2+i+sgw].F32()
					pc2 = pc2 + q[h*d.HeadDim+i+2*sgw]*k[phys2+i+2*sgw].F32()
					pd2 = pd2 + q[h*d.HeadDim+i+3*sgw]*k[phys2+i+3*sgw].F32()
					i = i + 4*sgw
				}
				for i < d.HeadDim {
					pa2 = pa2 + q[h*d.HeadDim+i]*k[phys2+i].F32()
					i = i + sgw
				}
				part2 = (pa2 + pb2) + (pc2 + pd2)
			}
			jpos3 := base + j + 3*nsg
			part3 := float32(0)
			if jpos3 < kvLen {
				phys3 := jpos3*d.KVHeads*d.HeadDim + kvHead*d.HeadDim
				// Four independent partials over the row: one chain of dependent
				// adds through a load each was the phase's latency, since a lane
				// waited on each load before issuing the next.
				pa3 := float32(0)
				pb3 := float32(0)
				pc3 := float32(0)
				pd3 := float32(0)
				i := sgl
				for i+3*sgw < d.HeadDim {
					pa3 = pa3 + q[h*d.HeadDim+i]*k[phys3+i].F32()
					pb3 = pb3 + q[h*d.HeadDim+i+sgw]*k[phys3+i+sgw].F32()
					pc3 = pc3 + q[h*d.HeadDim+i+2*sgw]*k[phys3+i+2*sgw].F32()
					pd3 = pd3 + q[h*d.HeadDim+i+3*sgw]*k[phys3+i+3*sgw].F32()
					i = i + 4*sgw
				}
				for i < d.HeadDim {
					pa3 = pa3 + q[h*d.HeadDim+i]*k[phys3+i].F32()
					i = i + sgw
				}
				part3 = (pa3 + pb3) + (pc3 + pd3)
			}
			dot0 := t.SubgroupAddF32(part0)
			dot1 := t.SubgroupAddF32(part1)
			dot2 := t.SubgroupAddF32(part2)
			dot3 := t.SubgroupAddF32(part3)
			if sgl == 0 {
				sj0 := float32(-3.4e38)
				if jpos0 < kvLen {
					sj0 = dot0 * d.Scale
				}
				scores[j+0*nsg] = sj0
				sj1 := float32(-3.4e38)
				if jpos1 < kvLen {
					sj1 = dot1 * d.Scale
				}
				scores[j+1*nsg] = sj1
				sj2 := float32(-3.4e38)
				if jpos2 < kvLen {
					sj2 = dot2 * d.Scale
				}
				scores[j+2*nsg] = sj2
				sj3 := float32(-3.4e38)
				if jpos3 < kvLen {
					sj3 = dot3 * d.Scale
				}
				scores[j+3*nsg] = sj3
			}
		}
		t.Barrier()
		s := scores[lane]

		// The block's maximum and the block's mass: a subgroup reduction each,
		// one partial per subgroup into red, one barrier, and every lane folds
		// the partials. The shared-memory trees they replace cost seven
		// barriers apiece, and with one workgroup per head on a device that
		// wants many, the barriers were the block's latency rather than its
		// loads: the time per decode was flat from 8 to 64 workgroups.
		sgMax := t.SubgroupMaxF32(s)
		if sgl == 0 {
			red[t.SubgroupIndex()] = sgMax
		}
		t.Barrier()
		blockMax := red[0]
		for g := uint32(1); g < nsg; g++ {
			blockMax = kmath.Max(blockMax, red[g])
		}

		next := kmath.Max(m, blockMax)
		alpha := kmath.Exp(m - next)

		e := float32(0)
		if pos < kvLen {
			e = kmath.Exp(s - next)
		}
		sgSum := t.SubgroupAddF32(e)
		// Every lane has read red for the maximum before any lane writes the
		// masses into it; scores likewise was read into s by every lane
		// before the weights overwrite it.
		t.Barrier()
		scores[lane] = e
		if sgl == 0 {
			red[t.SubgroupIndex()] = sgSum
		}
		t.Barrier()
		blockSum := red[0]
		for g := uint32(1); g < nsg; g++ {
			blockSum = blockSum + red[g]
		}
		l = alpha*l + blockSum
		m = next

		// The weighted sum of V, parallel over the head's dimensions rather
		// than over the cache: the probabilities are in shared memory now, so
		// every lane can read all of them and each owns one output element.
		//
		// The bound on j is a range check and not a mask. A position past the
		// cache already has a probability of zero, so dropping it changes no
		// arithmetic -- it reads V past its end, which the CPU backend reports
		// as an out-of-range index and a device would not report at all.
		if lane < d.HeadDim {
			o = alpha * o
			// Four positions per step into four independent accumulators, so
			// four loads are in flight where one dependent chain of adds through
			// a load each let one be. The masked positions past the length are
			// zero-weight and skipped by the guard as before.
			o0 := float32(0)
			o1 := float32(0)
			o2 := float32(0)
			o3 := float32(0)
			for j := uint32(0); j < AttnBlock; j += 4 {
				jj0 := base + j + 0
				if jj0 < kvLen {
					o0 = o0 + scores[j+0]*v[(jj0)*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane].F32()
				}
				jj1 := base + j + 1
				if jj1 < kvLen {
					o1 = o1 + scores[j+1]*v[(jj1)*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane].F32()
				}
				jj2 := base + j + 2
				if jj2 < kvLen {
					o2 = o2 + scores[j+2]*v[(jj2)*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane].F32()
				}
				jj3 := base + j + 3
				if jj3 < kvLen {
					o3 = o3 + scores[j+3]*v[(jj3)*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane].F32()
				}
			}
			o = o + ((o0 + o1) + (o2 + o3))
		}
	}

	if lane < d.HeadDim {
		out[h*d.HeadDim+lane] = o / l
	}
}
