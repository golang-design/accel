// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import (
	"golang.design/x/accel"
	"golang.design/x/accel/kmath"
)

// PagedDims is a paged decode step's shape.
//
// The same as [AttnDims] plus the block size, because that is the only thing
// the addressing needs that the contiguous form does not.
type PagedDims struct {
	QHeads  uint32
	KVHeads uint32
	HeadDim uint32

	// Block is how many positions one physical block holds.
	Block uint32

	Scale float32
}

// AttentionDecodePaged is one decode step over a cache addressed through a page
// table.
//
//	physical(j) = pages[j / Block] * Block + (j mod Block)
//
// # What changed from the contiguous form, and what did not
//
// One indirection, and nothing else. The scoring, the shifted-exponential
// softmax, the weighted sum, and the treatment of lanes past the cache are all
// [AttentionDecode]'s, unchanged -- because a paged cache is a different
// *addressing* rather than a different attention, and specs/030-paged-kv.md
// makes that the property to preserve.
//
// The page table is a binding rather than a uniform. It varies per sequence and
// per step, and a uniform would make every sequence its own plan, which is what
// paging exists to avoid.
//
//accel:uniform lengths
//accel:requires subgroup_arithmetic, subgroup_basic
//accel:kernel workgroup=128
func AttentionDecodePaged(t accel.Thread, d PagedDims, q []float32, k []float32,
	v []float32, pages []uint32, lengths []uint32, out []float32,
	scores *[AttnBlock]float32, red *[AttnBlock]float32) {
	h := t.GroupID().X
	// One sequence, so its length is the first entry. See AttentionDecode.
	kvLen := lengths[0]
	lane := t.LocalID().X

	group := d.QHeads / d.KVHeads
	kvHead := h / group
	// How many positions the page table can address. From the table's extent
	// rather than from the pool's: the pool holds every sequence's blocks and
	// is sized for total concurrency, so looping over it would walk other
	// sequences' caches. specs/002-compute-model.md section 3.3 makes both
	// len() and a uniform field workgroup-uniform, which is what lets the loop
	// below hold a barrier -- see [AttentionDecode] for why kvLen cannot.
	capacity := uint32(len(pages)) * d.Block

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

	// The running softmax, carried across blocks. See [AttentionDecode] for the
	// recurrence and for why one block reproduces the single-pass form exactly.
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

		// Each lane scores one cached position, reached through its page.
		// The shared arrays are loop-carried, so this pass's writes have to be
		// ordered against the previous pass's reads of them. See
		// [AttentionDecode].
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
				phys0 := pages[jpos0/d.Block]*d.Block + jpos0%d.Block
				// Four independent partials over the row: one chain of dependent
				// adds through a load each was the phase's latency, since a lane
				// waited on each load before issuing the next.
				pa0 := float32(0)
				pb0 := float32(0)
				pc0 := float32(0)
				pd0 := float32(0)
				i := sgl
				for i+3*sgw < d.HeadDim {
					pa0 = pa0 + q[h*d.HeadDim+i]*k[phys0*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i]
					pb0 = pb0 + q[h*d.HeadDim+i+sgw]*k[phys0*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+sgw]
					pc0 = pc0 + q[h*d.HeadDim+i+2*sgw]*k[phys0*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+2*sgw]
					pd0 = pd0 + q[h*d.HeadDim+i+3*sgw]*k[phys0*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+3*sgw]
					i = i + 4*sgw
				}
				for i < d.HeadDim {
					pa0 = pa0 + q[h*d.HeadDim+i]*k[phys0*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i]
					i = i + sgw
				}
				part0 = (pa0 + pb0) + (pc0 + pd0)
			}
			jpos1 := base + j + 1*nsg
			part1 := float32(0)
			if jpos1 < kvLen {
				phys1 := pages[jpos1/d.Block]*d.Block + jpos1%d.Block
				// Four independent partials over the row: one chain of dependent
				// adds through a load each was the phase's latency, since a lane
				// waited on each load before issuing the next.
				pa1 := float32(0)
				pb1 := float32(0)
				pc1 := float32(0)
				pd1 := float32(0)
				i := sgl
				for i+3*sgw < d.HeadDim {
					pa1 = pa1 + q[h*d.HeadDim+i]*k[phys1*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i]
					pb1 = pb1 + q[h*d.HeadDim+i+sgw]*k[phys1*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+sgw]
					pc1 = pc1 + q[h*d.HeadDim+i+2*sgw]*k[phys1*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+2*sgw]
					pd1 = pd1 + q[h*d.HeadDim+i+3*sgw]*k[phys1*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+3*sgw]
					i = i + 4*sgw
				}
				for i < d.HeadDim {
					pa1 = pa1 + q[h*d.HeadDim+i]*k[phys1*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i]
					i = i + sgw
				}
				part1 = (pa1 + pb1) + (pc1 + pd1)
			}
			jpos2 := base + j + 2*nsg
			part2 := float32(0)
			if jpos2 < kvLen {
				phys2 := pages[jpos2/d.Block]*d.Block + jpos2%d.Block
				// Four independent partials over the row: one chain of dependent
				// adds through a load each was the phase's latency, since a lane
				// waited on each load before issuing the next.
				pa2 := float32(0)
				pb2 := float32(0)
				pc2 := float32(0)
				pd2 := float32(0)
				i := sgl
				for i+3*sgw < d.HeadDim {
					pa2 = pa2 + q[h*d.HeadDim+i]*k[phys2*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i]
					pb2 = pb2 + q[h*d.HeadDim+i+sgw]*k[phys2*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+sgw]
					pc2 = pc2 + q[h*d.HeadDim+i+2*sgw]*k[phys2*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+2*sgw]
					pd2 = pd2 + q[h*d.HeadDim+i+3*sgw]*k[phys2*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+3*sgw]
					i = i + 4*sgw
				}
				for i < d.HeadDim {
					pa2 = pa2 + q[h*d.HeadDim+i]*k[phys2*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i]
					i = i + sgw
				}
				part2 = (pa2 + pb2) + (pc2 + pd2)
			}
			jpos3 := base + j + 3*nsg
			part3 := float32(0)
			if jpos3 < kvLen {
				phys3 := pages[jpos3/d.Block]*d.Block + jpos3%d.Block
				// Four independent partials over the row: one chain of dependent
				// adds through a load each was the phase's latency, since a lane
				// waited on each load before issuing the next.
				pa3 := float32(0)
				pb3 := float32(0)
				pc3 := float32(0)
				pd3 := float32(0)
				i := sgl
				for i+3*sgw < d.HeadDim {
					pa3 = pa3 + q[h*d.HeadDim+i]*k[phys3*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i]
					pb3 = pb3 + q[h*d.HeadDim+i+sgw]*k[phys3*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+sgw]
					pc3 = pc3 + q[h*d.HeadDim+i+2*sgw]*k[phys3*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+2*sgw]
					pd3 = pd3 + q[h*d.HeadDim+i+3*sgw]*k[phys3*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+3*sgw]
					i = i + 4*sgw
				}
				for i < d.HeadDim {
					pa3 = pa3 + q[h*d.HeadDim+i]*k[phys3*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i]
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

		// The weighted sum of V, parallel over the head's dimensions. Each lane
		// owns one output element and walks the block's positions, so the page
		// lookup happens per position here rather than per lane.
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
					phys0 := pages[(jj0)/d.Block]*d.Block + (jj0)%d.Block
					o0 = o0 + scores[j+0]*v[phys0*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
				}
				jj1 := base + j + 1
				if jj1 < kvLen {
					phys1 := pages[(jj1)/d.Block]*d.Block + (jj1)%d.Block
					o1 = o1 + scores[j+1]*v[phys1*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
				}
				jj2 := base + j + 2
				if jj2 < kvLen {
					phys2 := pages[(jj2)/d.Block]*d.Block + (jj2)%d.Block
					o2 = o2 + scores[j+2]*v[phys2*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
				}
				jj3 := base + j + 3
				if jj3 < kvLen {
					phys3 := pages[(jj3)/d.Block]*d.Block + (jj3)%d.Block
					o3 = o3 + scores[j+3]*v[phys3*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
				}
			}
			o = o + ((o0 + o1) + (o2 + o3))
		}
	}

	if lane < d.HeadDim {
		out[h*d.HeadDim+lane] = o / l
	}
}

// AttentionDecodePagedF16 is [AttentionDecodePaged] over an f16 cache.
//
// The body is [AttentionDecodePaged]'s with the two cache loads widened, and
// [AttentionDecodeF16] carries the argument for why K and V may be narrow where
// an accumulator may not: they are operands, and the accumulation stays f32.
//
// # Why paging and a narrow cache had to meet in one kernel
//
// They are two halves of one memory argument and neither reaches the other by
// itself: paging removes the fragmentation, and a narrow element halves what is
// left. Only the contiguous decode read a narrow cache, so a caller had to pick
// one saving or the other -- which is the compounding accel issue 13 reports.
//
// The loop bound stays the page table's reach rather than the pool's, for
// [AttentionDecodePaged]'s reason. The element width does not enter it: the
// bound counts positions the table addresses, so it is the same expression at
// either storage width.
//
//accel:uniform lengths
//accel:requires subgroup_arithmetic, subgroup_basic
//accel:kernel workgroup=128
func AttentionDecodePagedF16(t accel.Thread, d PagedDims, q []float32, k []accel.Float16,
	v []accel.Float16, pages []uint32, lengths []uint32, out []float32,
	scores *[AttnBlock]float32, red *[AttnBlock]float32) {
	h := t.GroupID().X
	// One sequence, so its length is the first entry. See AttentionDecode.
	kvLen := lengths[0]
	lane := t.LocalID().X

	group := d.QHeads / d.KVHeads
	kvHead := h / group
	// How many positions the page table can address. From the table's extent
	// rather than from the pool's: the pool holds every sequence's blocks and
	// is sized for total concurrency, so looping over it would walk other
	// sequences' caches. specs/002-compute-model.md section 3.3 makes both
	// len() and a uniform field workgroup-uniform, which is what lets the loop
	// below hold a barrier -- see [AttentionDecode] for why kvLen cannot.
	capacity := uint32(len(pages)) * d.Block

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

	// The running softmax, carried across blocks. See [AttentionDecode] for the
	// recurrence and for why one block reproduces the single-pass form exactly.
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

		// Each lane scores one cached position, reached through its page.
		// The shared arrays are loop-carried, so this pass's writes have to be
		// ordered against the previous pass's reads of them. See
		// [AttentionDecode].
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
				phys0 := pages[jpos0/d.Block]*d.Block + jpos0%d.Block
				// Four independent partials over the row: one chain of dependent
				// adds through a load each was the phase's latency, since a lane
				// waited on each load before issuing the next.
				pa0 := float32(0)
				pb0 := float32(0)
				pc0 := float32(0)
				pd0 := float32(0)
				i := sgl
				for i+3*sgw < d.HeadDim {
					pa0 = pa0 + q[h*d.HeadDim+i]*k[phys0*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i].F32()
					pb0 = pb0 + q[h*d.HeadDim+i+sgw]*k[phys0*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+sgw].F32()
					pc0 = pc0 + q[h*d.HeadDim+i+2*sgw]*k[phys0*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+2*sgw].F32()
					pd0 = pd0 + q[h*d.HeadDim+i+3*sgw]*k[phys0*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+3*sgw].F32()
					i = i + 4*sgw
				}
				for i < d.HeadDim {
					pa0 = pa0 + q[h*d.HeadDim+i]*k[phys0*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i].F32()
					i = i + sgw
				}
				part0 = (pa0 + pb0) + (pc0 + pd0)
			}
			jpos1 := base + j + 1*nsg
			part1 := float32(0)
			if jpos1 < kvLen {
				phys1 := pages[jpos1/d.Block]*d.Block + jpos1%d.Block
				// Four independent partials over the row: one chain of dependent
				// adds through a load each was the phase's latency, since a lane
				// waited on each load before issuing the next.
				pa1 := float32(0)
				pb1 := float32(0)
				pc1 := float32(0)
				pd1 := float32(0)
				i := sgl
				for i+3*sgw < d.HeadDim {
					pa1 = pa1 + q[h*d.HeadDim+i]*k[phys1*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i].F32()
					pb1 = pb1 + q[h*d.HeadDim+i+sgw]*k[phys1*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+sgw].F32()
					pc1 = pc1 + q[h*d.HeadDim+i+2*sgw]*k[phys1*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+2*sgw].F32()
					pd1 = pd1 + q[h*d.HeadDim+i+3*sgw]*k[phys1*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+3*sgw].F32()
					i = i + 4*sgw
				}
				for i < d.HeadDim {
					pa1 = pa1 + q[h*d.HeadDim+i]*k[phys1*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i].F32()
					i = i + sgw
				}
				part1 = (pa1 + pb1) + (pc1 + pd1)
			}
			jpos2 := base + j + 2*nsg
			part2 := float32(0)
			if jpos2 < kvLen {
				phys2 := pages[jpos2/d.Block]*d.Block + jpos2%d.Block
				// Four independent partials over the row: one chain of dependent
				// adds through a load each was the phase's latency, since a lane
				// waited on each load before issuing the next.
				pa2 := float32(0)
				pb2 := float32(0)
				pc2 := float32(0)
				pd2 := float32(0)
				i := sgl
				for i+3*sgw < d.HeadDim {
					pa2 = pa2 + q[h*d.HeadDim+i]*k[phys2*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i].F32()
					pb2 = pb2 + q[h*d.HeadDim+i+sgw]*k[phys2*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+sgw].F32()
					pc2 = pc2 + q[h*d.HeadDim+i+2*sgw]*k[phys2*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+2*sgw].F32()
					pd2 = pd2 + q[h*d.HeadDim+i+3*sgw]*k[phys2*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+3*sgw].F32()
					i = i + 4*sgw
				}
				for i < d.HeadDim {
					pa2 = pa2 + q[h*d.HeadDim+i]*k[phys2*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i].F32()
					i = i + sgw
				}
				part2 = (pa2 + pb2) + (pc2 + pd2)
			}
			jpos3 := base + j + 3*nsg
			part3 := float32(0)
			if jpos3 < kvLen {
				phys3 := pages[jpos3/d.Block]*d.Block + jpos3%d.Block
				// Four independent partials over the row: one chain of dependent
				// adds through a load each was the phase's latency, since a lane
				// waited on each load before issuing the next.
				pa3 := float32(0)
				pb3 := float32(0)
				pc3 := float32(0)
				pd3 := float32(0)
				i := sgl
				for i+3*sgw < d.HeadDim {
					pa3 = pa3 + q[h*d.HeadDim+i]*k[phys3*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i].F32()
					pb3 = pb3 + q[h*d.HeadDim+i+sgw]*k[phys3*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+sgw].F32()
					pc3 = pc3 + q[h*d.HeadDim+i+2*sgw]*k[phys3*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+2*sgw].F32()
					pd3 = pd3 + q[h*d.HeadDim+i+3*sgw]*k[phys3*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i+3*sgw].F32()
					i = i + 4*sgw
				}
				for i < d.HeadDim {
					pa3 = pa3 + q[h*d.HeadDim+i]*k[phys3*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+i].F32()
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

		// The weighted sum of V, parallel over the head's dimensions. Each lane
		// owns one output element and walks the block's positions, so the page
		// lookup happens per position here rather than per lane.
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
					phys0 := pages[(jj0)/d.Block]*d.Block + (jj0)%d.Block
					o0 = o0 + scores[j+0]*v[phys0*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane].F32()
				}
				jj1 := base + j + 1
				if jj1 < kvLen {
					phys1 := pages[(jj1)/d.Block]*d.Block + (jj1)%d.Block
					o1 = o1 + scores[j+1]*v[phys1*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane].F32()
				}
				jj2 := base + j + 2
				if jj2 < kvLen {
					phys2 := pages[(jj2)/d.Block]*d.Block + (jj2)%d.Block
					o2 = o2 + scores[j+2]*v[phys2*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane].F32()
				}
				jj3 := base + j + 3
				if jj3 < kvLen {
					phys3 := pages[(jj3)/d.Block]*d.Block + (jj3)%d.Block
					o3 = o3 + scores[j+3]*v[phys3*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane].F32()
				}
			}
			o = o + ((o0 + o1) + (o2 + o3))
		}
	}

	if lane < d.HeadDim {
		out[h*d.HeadDim+lane] = o / l
	}
}
