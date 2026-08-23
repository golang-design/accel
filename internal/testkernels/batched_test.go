// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
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
	err := kernel.DispatchCooperative(&testkernels.AttentionDecodeBatchedKernel,
		accel.ID3{X: batch * qHeads},
		accel.KernelArgs{
			Slices: []any{q, physK, physV, pages, lengths, batched},
			Uniforms: []any{testkernels.BatchedDims{
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
		err := kernel.DispatchCooperative(&testkernels.AttentionDecodePagedKernel,
			accel.ID3{X: qHeads},
			accel.KernelArgs{
				Slices: []any{
					q[s*qHeads*headDim : (s+1)*qHeads*headDim],
					physK, physV,
					pages[s*maxPages : (s+1)*maxPages],
					alone,
				},
				Uniforms: []any{testkernels.PagedDims{
					QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
					KVLen: lengths[s], Block: block, Scale: scale,
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
	if err := kernel.DispatchCooperative(&testkernels.AttentionDecodeBatchedKernel,
		accel.ID3{X: qHeads},
		accel.KernelArgs{
			Slices: []any{q, physK, physV, pages, []uint32{5}, one},
			Uniforms: []any{testkernels.BatchedDims{
				Batch: 1, QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
				Block: block, MaxPages: maxPages, Scale: scale,
			}},
		}); err != nil {
		t.Fatalf("batched: %v", err)
	}

	unbatched := make([]float32, qHeads*headDim)
	if err := kernel.DispatchCooperative(&testkernels.AttentionDecodePagedKernel,
		accel.ID3{X: qHeads},
		accel.KernelArgs{
			Slices: []any{q, physK, physV, pages, unbatched},
			Uniforms: []any{testkernels.PagedDims{
				QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
				KVLen: 5, Block: block, Scale: scale,
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
