// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"golang.design/x/accel/kernelabi"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
)

// A paged decode step produces exactly what the contiguous one does over the
// same logical positions.
//
// Exactly, not within a budget: paging is an addressing change and the
// arithmetic is identical, so any difference at all is a bug in the indexing.
//
// The out-of-order case is the one that matters. A paged kernel that ignored
// its page table and read position j at j would pass the identity-mapping case
// and fail this one, and identity mapping is what a first test naturally uses.
func TestPagedDecodeMatchesContiguous(t *testing.T) {
	const (
		qHeads  = 4
		kvHeads = 2
		headDim = 8
		block   = 4
		kvLen   = 10 // not a multiple of the block, so the last page is partial
	)
	scale := float32(1) / float32(math.Sqrt(headDim))

	q := make([]float32, qHeads*headDim)
	for i := range q {
		q[i] = float32(math.Sin(float64(i) * 0.31))
	}
	// The logical cache: what a contiguous kernel would read.
	logicalK := make([]float32, kvLen*kvHeads*headDim)
	logicalV := make([]float32, kvLen*kvHeads*headDim)
	for i := range logicalK {
		logicalK[i] = float32(math.Cos(float64(i) * 0.17))
		logicalV[i] = float32(i%9) - 4
	}

	contiguous := make([]float32, qHeads*headDim)
	err := kernel.DispatchCooperative(&testkernels.AttentionDecodeKernel,
		accel.ID3{X: qHeads},
		kernelabi.Args{
			Slices: []any{q, logicalK, logicalV, []uint32{kvLen}, contiguous},
			Uniforms: []any{testkernels.AttnDims{
				QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
				Scale: scale,
			}},
		})
	if err != nil {
		t.Fatalf("contiguous: %v", err)
	}

	for _, tc := range []struct {
		name  string
		pages []uint32
	}{
		{"identity mapping", []uint32{0, 1, 2}},
		// Reversed, interleaved, and sparse: each is a mapping a kernel that
		// ignored the page table would get wrong, and the first is the one a
		// naive implementation passes.
		{"reversed", []uint32{5, 4, 3}},
		{"scattered", []uint32{7, 2, 5}},
		{"one block reused nowhere near its logical place", []uint32{9, 0, 4}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// A physical cache big enough for any of the mappings, with the
			// logical contents copied to where the pages say they live.
			const blocks = 12
			physK := make([]float32, blocks*block*kvHeads*headDim)
			physV := make([]float32, blocks*block*kvHeads*headDim)
			for j := range kvLen {
				phys := int(tc.pages[j/block])*block + j%block
				copy(physK[phys*kvHeads*headDim:], logicalK[j*kvHeads*headDim:(j+1)*kvHeads*headDim])
				copy(physV[phys*kvHeads*headDim:], logicalV[j*kvHeads*headDim:(j+1)*kvHeads*headDim])
			}

			paged := make([]float32, qHeads*headDim)
			err := kernel.DispatchCooperative(&testkernels.AttentionDecodePagedKernel,
				accel.ID3{X: qHeads},
				kernelabi.Args{
					Slices: []any{q, physK, physV, tc.pages, []uint32{kvLen}, paged},
					Uniforms: []any{testkernels.PagedDims{
						QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
						Block: block, Scale: scale,
					}},
				})
			if err != nil {
				t.Fatalf("paged: %v", err)
			}
			for i := range contiguous {
				if paged[i] != contiguous[i] {
					t.Fatalf("element %d is %v paged and %v contiguous; paging is an "+
						"addressing change, so the arithmetic is identical and any "+
						"difference is an indexing bug", i, paged[i], contiguous[i])
				}
			}
		})
	}
}

// Two sequences over one pool do not see each other's positions.
//
// The property paging exists for, and the one a shared pool can break: if the
// page tables overlapped, or the kernel read past a sequence's length, one
// conversation would read another's tokens -- which surfaces as a model
// answering from the wrong context and is close to undebuggable from the output.
func TestTwoSequencesShareAPoolWithoutSeeingEachOther(t *testing.T) {
	const (
		qHeads  = 2
		kvHeads = 1
		headDim = 8
		block   = 4
		blocks  = 8
	)
	scale := float32(1) / float32(math.Sqrt(headDim))

	physK := make([]float32, blocks*block*kvHeads*headDim)
	physV := make([]float32, blocks*block*kvHeads*headDim)

	// Interleaved page tables: A holds 0 and 2, B holds 1 and 3. Adjacent
	// physical blocks belong to different sequences, so reading one position
	// past a sequence's end lands in the other's data.
	seqs := []struct {
		name  string
		pages []uint32
		n     int
		fill  float32
	}{
		{"A", []uint32{0, 2}, 6, 1},
		{"B", []uint32{1, 3}, 5, -1},
	}
	for _, s := range seqs {
		for j := range s.n {
			phys := int(s.pages[j/block])*block + j%block
			for i := range kvHeads * headDim {
				physK[phys*kvHeads*headDim+i] = s.fill * float32(j+1)
				physV[phys*kvHeads*headDim+i] = s.fill * float32(j+1)
			}
		}
	}

	q := make([]float32, qHeads*headDim)
	for i := range q {
		q[i] = 0.1
	}

	for _, s := range seqs {
		out := make([]float32, qHeads*headDim)
		err := kernel.DispatchCooperative(&testkernels.AttentionDecodePagedKernel,
			accel.ID3{X: qHeads},
			kernelabi.Args{
				Slices: []any{q, physK, physV, s.pages, []uint32{uint32(s.n)}, out},
				Uniforms: []any{testkernels.PagedDims{
					QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim,
					Block: block, Scale: scale,
				}},
			})
		if err != nil {
			t.Fatalf("sequence %s: %v", s.name, err)
		}
		// Every value this sequence wrote has its own sign, so an output with
		// the wrong sign anywhere means it read the other sequence's blocks.
		for i, v := range out {
			if s.fill > 0 && v <= 0 {
				t.Fatalf("sequence A element %d is %v, and every value it cached is "+
					"positive: it read B's blocks", i, v)
			}
			if s.fill < 0 && v >= 0 {
				t.Fatalf("sequence B element %d is %v, and every value it cached is "+
					"negative: it read A's blocks", i, v)
			}
		}
	}
}

// A paged cache longer than one workgroup. The companion to
// TestAttentionDecodeScoresACacheLongerThanAWorkgroup, and the case where the
// loop bound is the one thing a paged kernel must not get from the pool: the
// pool holds every sequence's blocks, so its extent is total concurrency rather
// than this sequence's reach. The bound is the page table's extent times the
// block size.
//
// The pool here is deliberately far larger than the sequence, and its unpaged
// blocks hold values that would swamp the answer if they were ever read.
func TestPagedDecodeScoresACacheLongerThanAWorkgroup(t *testing.T) {
	const qHeads, kvHeads, headDim, block = 4, 2, 32, 8
	const kvLen = 300
	const pageCount = (kvLen + block - 1) / block // 38 pages -> 304 positions
	const poolBlocks = 4 * pageCount              // the pool is much bigger

	d := testkernels.PagedDims{
		QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim, Block: block,
		Scale: 1 / float32(math.Sqrt(headDim)),
	}
	rng := rand.New(rand.NewPCG(11, 7))
	q := make([]float32, qHeads*headDim)
	for i := range q {
		q[i] = float32(rng.NormFloat64())
	}
	pk := make([]float32, poolBlocks*block*kvHeads*headDim)
	pv := make([]float32, poolBlocks*block*kvHeads*headDim)
	for i := range pk {
		pk[i] = float32(rng.NormFloat64())
		pv[i] = float32(rng.NormFloat64())
	}
	// Scattered and out of order, so a kernel that walked the pool linearly
	// would read the wrong blocks rather than the right ones in the wrong
	// order.
	pages := make([]uint32, pageCount)
	perm := rng.Perm(poolBlocks)
	for i := range pages {
		pages[i] = uint32(perm[i])
	}
	lengths := []uint32{kvLen}

	got := make([]float32, qHeads*headDim)
	if err := kernel.DispatchCooperative(&testkernels.AttentionDecodePagedKernel,
		accel.ID3{X: qHeads},
		kernelabi.Args{
			Slices: []any{q, pk, pv, pages, lengths, got}, Uniforms: []any{d},
		}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	// The reference reads the same positions through the same table, gathered
	// into a contiguous cache: paging is an addressing, so the answer must be
	// the contiguous one.
	gk := make([]float32, kvLen*kvHeads*headDim)
	gv := make([]float32, kvLen*kvHeads*headDim)
	for j := 0; j < kvLen; j++ {
		phys := int(pages[j/block])*block + j%block
		copy(gk[j*kvHeads*headDim:(j+1)*kvHeads*headDim],
			pk[phys*kvHeads*headDim:(phys+1)*kvHeads*headDim])
		copy(gv[j*kvHeads*headDim:(j+1)*kvHeads*headDim],
			pv[phys*kvHeads*headDim:(phys+1)*kvHeads*headDim])
	}
	want := composedAttention(testkernels.AttnDims{
		QHeads: qHeads, KVHeads: kvHeads, HeadDim: headDim, Scale: d.Scale,
	}, kvLen, q, gk, gv)

	// The bound derived in TestAttentionDecodeScoresACacheLongerThanAWorkgroup.
	const u = 1.0 / (1 << 24)
	n := float64(kvLen+headDim+1) + math.Ceil(float64(kvLen)/AttnBlockT)
	gamma := n * u / (1 - n*u)
	maxV := 0.0
	for _, x := range gv {
		maxV = math.Max(maxV, math.Abs(float64(x)))
	}
	tol := maxV * gamma

	for i := range got {
		if diff := math.Abs(float64(got[i]) - want[i]); diff > tol {
			t.Fatalf("element %d is %v, want %v: off by %g, tolerance %g",
				i, got[i], want[i], diff, tol)
		}
	}
}
