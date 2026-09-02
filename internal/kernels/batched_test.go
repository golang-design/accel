// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels_test

import (
	"golang.design/x/accel/kernelabi"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/kernels"
)

// A batch of sequences produces exactly what each sequence produces alone.
//
// Exactly, for the same reason paging does: batching changes which workgroup
// does the work, not the arithmetic. Any difference is a sequence reading the
// wrong page table, the wrong length, or the wrong query.
//
// The sequences deliberately have **different lengths and interleaved pages**.
// A batched kernel that padded to a common length would read past a short
// sequence's end; one that used a single page table would read another
// sequence's blocks; one that indexed the query by workgroup rather than by
// (sequence, head) would scramble the outputs. Equal-length sequences with
// identity pages would catch none of the three.
func TestBatchedDecodeMatchesOneAtATime(t *testing.T) {
	const (
		batch    = 3
		qHeads   = 2
		kvHeads  = 1
		headDim  = 8
		block    = 4
		maxPages = 3
		blocks   = 12
	)
	scale := float32(1) / float32(math.Sqrt(headDim))

	lengths := []uint32{9, 3, 6}
	// Interleaved, so adjacent physical blocks belong to different sequences.
	pages := []uint32{
		0, 3, 6, // sequence 0
		1, 0, 0, // sequence 1 (one block; the rest is never read)
		4, 7, 0, // sequence 2
	}

	physK := make([]float32, blocks*block*kvHeads*headDim)
	physV := make([]float32, blocks*block*kvHeads*headDim)
	for i := range physK {
		physK[i] = float32(math.Sin(float64(i) * 0.13))
		physV[i] = float32(math.Cos(float64(i) * 0.21))
	}
	q := make([]float32, batch*qHeads*headDim)
	for i := range q {
		q[i] = float32(math.Sin(float64(i) * 0.37))
	}

	batched := make([]float32, batch*qHeads*headDim)
	err := kernel.DispatchCooperative(&kernels.AttentionDecodeBatchedKernel,
		accel.ID3{X: batch * qHeads},
		kernelabi.Args{
			Slices: []any{q, physK, physV, pages, lengths, batched},
			Uniforms: []any{kernels.BatchedDims{
				Batch: batch, QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
				Block: block, MaxPages: maxPages, Scale: scale,
			}},
		})
	if err != nil {
		t.Fatalf("batched: %v", err)
	}

	// Each sequence alone, through the unbatched paged kernel.
	for s := range batch {
		alone := make([]float32, qHeads*headDim)
		err := kernel.DispatchCooperative(&kernels.AttentionDecodePagedKernel,
			accel.ID3{X: qHeads},
			kernelabi.Args{
				Slices: []any{
					q[s*qHeads*headDim : (s+1)*qHeads*headDim],
					physK, physV,
					pages[s*maxPages : (s+1)*maxPages],
					lengths[s : s+1],
					alone,
				},
				Uniforms: []any{kernels.PagedDims{
					QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
					Block: block, Scale: scale,
				}},
			})
		if err != nil {
			t.Fatalf("sequence %d: %v", s, err)
		}
		for i := range alone {
			got := batched[s*qHeads*headDim+i]
			if got != alone[i] {
				t.Fatalf("sequence %d element %d is %v batched and %v alone; batching "+
					"changes which workgroup does the work, not the arithmetic",
					s, i, got, alone[i])
			}
		}
	}
}

// A batch of one is the unbatched kernel, which is the degenerate case a
// scheduler hits whenever only one request is in flight.
func TestBatchOfOne(t *testing.T) {
	const qHeads, kvHeads, headDim, block, maxPages, blocks = 2, 1, 8, 4, 2, 6
	scale := float32(1) / float32(math.Sqrt(headDim))

	physK := make([]float32, blocks*block*kvHeads*headDim)
	physV := make([]float32, blocks*block*kvHeads*headDim)
	for i := range physK {
		physK[i] = float32(i%11) - 5
		physV[i] = float32(i%7) - 3
	}
	q := make([]float32, qHeads*headDim)
	for i := range q {
		q[i] = float32(math.Sin(float64(i) * 0.4))
	}
	pages := []uint32{3, 1}

	one := make([]float32, qHeads*headDim)
	if err := kernel.DispatchCooperative(&kernels.AttentionDecodeBatchedKernel,
		accel.ID3{X: qHeads},
		kernelabi.Args{
			Slices: []any{q, physK, physV, pages, []uint32{5}, one},
			Uniforms: []any{kernels.BatchedDims{
				Batch: 1, QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
				Block: block, MaxPages: maxPages, Scale: scale,
			}},
		}); err != nil {
		t.Fatalf("batched: %v", err)
	}

	unbatched := make([]float32, qHeads*headDim)
	if err := kernel.DispatchCooperative(&kernels.AttentionDecodePagedKernel,
		accel.ID3{X: qHeads},
		kernelabi.Args{
			Slices: []any{q, physK, physV, pages, []uint32{5}, unbatched},
			Uniforms: []any{kernels.PagedDims{
				QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
				Block: block, Scale: scale,
			}},
		}); err != nil {
		t.Fatalf("paged: %v", err)
	}
	for i := range one {
		if one[i] != unbatched[i] {
			t.Fatalf("element %d is %v in a batch of one and %v unbatched",
				i, one[i], unbatched[i])
		}
	}
}

// A length past the page table's reach is truncated, not read out of the
// neighbouring sequence's row.
//
// This is the claim specs/040-batch-scheduler.md and
// specs/044-unbounded-context.md both make and neither could check before. The
// batched kernel reads pages[seq*MaxPages + pos/Block], so before the block
// loop a length above MaxPages*Block indexed into slot seq+1's row -- another
// conversation's physical blocks -- and ran off the buffer for the last slot.
// The loop now stops at MaxPages*Block, so the answer covers a prefix of the
// sequence instead.
//
// A silently short answer is still wrong, which is why admission owes the
// check. What this test fixes is which of the two wrongs happens, and it is the
// difference between a truncated answer and another user's data.
func TestABatchedLengthPastItsPageTableIsTruncated(t *testing.T) {
	const batch, qHeads, kvHeads, headDim = 2, 2, 1, 8
	const block, maxPages = 4, 2
	const reach = maxPages * block // 8 positions per sequence

	d := kernels.BatchedDims{
		Batch: batch, QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
		Block: block, MaxPages: maxPages,
		Scale: float32(1 / math.Sqrt(headDim)),
	}
	// Sequence 0's blocks and sequence 1's are disjoint and hold values far
	// apart, so reading across the row boundary is visible in the answer rather
	// than merely different.
	const poolBlocks = 8
	q := make([]float32, batch*qHeads*headDim)
	pk := make([]float32, poolBlocks*block*kvHeads*headDim)
	pv := make([]float32, poolBlocks*block*kvHeads*headDim)
	for i := range q {
		q[i] = float32(math.Sin(float64(i) * 0.4))
	}
	for i := range pk {
		pk[i] = float32(math.Cos(float64(i) * 0.13))
		pv[i] = float32(i / (block * kvHeads * headDim)) // the block index
	}
	pages := []uint32{0, 1, 4, 5} // seq 0 -> blocks 0,1; seq 1 -> blocks 4,5

	// Sequence 0 asks for more than its table can reach; sequence 1 does not.
	over := []uint32{reach + 5, reach}
	exact := []uint32{reach, reach}

	run := func(lengths []uint32) []float32 {
		out := make([]float32, batch*qHeads*headDim)
		if err := kernel.DispatchCooperative(&kernels.AttentionDecodeBatchedKernel,
			accel.ID3{X: batch * qHeads},
			kernelabi.Args{
				Slices:   []any{q, pk, pv, pages, lengths, out},
				Uniforms: []any{d},
			}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		return out
	}

	got, want := run(over), run(exact)
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("element %d is %v with a length of %d and %v with a length of "+
				"%d: a length past the page table's reach must attend over the "+
				"reach, which is what stops it reaching the next sequence's row",
				i, got[i], over[0], want[i], exact[0])
		}
	}

	// And the truncated answer is a real attention over the eight reachable
	// positions, not zeros or a NaN from an empty softmax.
	for i, x := range got {
		if math.IsNaN(float64(x)) {
			t.Fatalf("element %d is NaN", i)
		}
	}
	if got[0] == 0 && got[1] == 0 {
		t.Fatal("the truncated answer is all zeros, so nothing was attended over")
	}
}
