// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"fmt"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
)

// GroupedMatVec multiplies each token by the weight matrix its segment names.
//
//	out[t][n] = Σₖ x[t][k] · w[e(t)][k][n]
//
// # What it is for
//
// A mixture-of-experts layer: E weight matrices and a router that sends each
// token to a few of them. The naive form — run every expert and mask — is
// expressible with [MatMul] today and does E/k times the work, which inverts
// the reason such a layer exists.
//
// specs/049-grouped-gemm.md. The shape is
// [046](046-segmented-extents.md)'s segmented extent with the row being an
// expert rather than a sequence, so this operator adds no concept: counts is
// one count per expert, and the offsets are derived here as they are there.
//
// # Tokens arrive ordered by expert
//
// x is [Σ counts, K], the tokens of expert 0 then expert 1 and so on. Producing
// that order from a routing table is a sort of a few thousand small integers,
// which is cheaper on the host than the dispatch it precedes — the same
// argument specs/046-segmented-extents.md §1.1 makes for deriving the offsets.
//
// # An expert with no tokens is ordinary
//
// Its count is zero and it contributes nothing. That is not an edge case here:
// with top-2-of-8 routing, six experts get nothing on every single token.
//
// # Rows of x past the total are padding
//
// The counts may sum to fewer tokens than x holds. Those rows routed nowhere:
// they read no weights and their output is zero. The sum is device data, so
// this is enforced in the kernel rather than refused here --
// specs/046-segmented-extents.md §1 property 3.
func GroupedMatVec(b *Builder, x, w, counts *Tensor) *Tensor {
	if counts == nil {
		return b.fail(1, "GroupedMatVec", "counts is nil; it is one token count per "+
			"expert and is what says which rows belong to which weight matrix "+
			"(specs/049-grouped-gemm.md)")
	}
	if poisoned(x, w, counts) {
		return b.poison()
	}
	for _, c := range []struct {
		name string
		t    *Tensor
		want DType
	}{{"x", x, accel.F32}, {"w", w, accel.F32}, {"counts", counts, accel.U32}} {
		if c.t.dtype != c.want {
			return b.fail(1, "GroupedMatVec", "%s is %v and this kernel reads %v",
				c.name, c.t.dtype, c.want)
		}
	}
	if len(x.shape) != 2 {
		return b.fail(1, "GroupedMatVec", "x is %v; it is [tokens, K] with the tokens "+
			"of every expert end to end", x.shape)
	}
	if len(w.shape) != 3 {
		return b.fail(1, "GroupedMatVec", "w is %v; it is [experts, K, N], one matrix "+
			"per expert", w.shape)
	}
	tokens, k := x.shape[0], x.shape[1]
	experts, wk, n := w.shape[0], w.shape[1], w.shape[2]
	if wk != k {
		return b.fail(1, "GroupedMatVec", "x is %v and w is %v; every expert contracts "+
			"against the same width, so w's second axis is x's second", x.shape, w.shape)
	}
	if got := counts.shape.Elements(); got != experts {
		return b.fail(1, "GroupedMatVec", "counts holds %d entries and w has %d "+
			"experts; it is one token count per expert", got, experts)
	}
	if experts == 0 {
		return b.fail(1, "GroupedMatVec", "w declares no experts")
	}

	offsets := b.segmentOffsets(counts, experts)
	if poisoned(offsets) {
		return b.poison()
	}

	return b.record(node{
		op: "GroupedMatVec", inputs: []*Tensor{x, w, offsets},
		kernel: &kernels.GroupedMatVecKernel,
		uniform: func(map[string]ScalarValue) any {
			return kernels.GroupedDims{
				Experts: uint32(experts), K: uint32(k), N: uint32(n),
			}
		},
		grid: func(*Tensor) accel.WorkgroupCount {
			// One workgroup per (token, column), each reducing over K. Not one
			// per expert: the tokens of one expert are independent, so there is
			// nothing to be gained by keeping them together and a matrix to be
			// re-read if they are split.
			return accel.WorkgroupCount{X: tokens * n}
		},
		reason: fmt.Sprintf("the grouped row kernel: %d tokens over %d experts, each "+
			"looking up which of the %d matrices its segment names", tokens, experts, experts),
		rejected: []string{"running every expert and masking: expressible with MatMul " +
			"today, and it does experts-over-k times the work, which is what a " +
			"mixture of experts exists not to do"},
	}, accel.F32, Shape{tokens, n})
}

// GroupedMatMul multiplies each expert's tokens by that expert's matrix.
//
//	out[t][n] = Σₖ x[t][k] · w[e(t)][k][n]
//
// # Why a separate operator from [GroupedMatVec]
//
// The shapes a decode step and a prefill have. A decode routes one token to a
// few experts and reads those matrices once, so a workgroup per (token, column)
// is right. A prefill has many tokens per expert, and reading an expert's matrix
// once per token wastes the bandwidth a mixture-of-experts layer is built to
// save — so this puts a workgroup on an (expert, column tile) and walks that
// expert's whole segment through shared tiles.
//
// Each weight is read once per block of [kernels.TileM] tokens rather than
// once per token. specs/049-grouped-gemm.md §5.
//
// # It takes the same inputs
//
// x is [Σ counts, K] ordered by expert, w is [E, K, N], counts is one per
// expert — [GroupedMatVec]'s arguments exactly, so switching between the two is
// a one-word edit and not a re-plumbing.
func GroupedMatMul(b *Builder, x, w, counts *Tensor) *Tensor {
	if counts == nil {
		return b.fail(1, "GroupedMatMul", "counts is nil; it is one token count per "+
			"expert and is what says which rows belong to which weight matrix "+
			"(specs/049-grouped-gemm.md)")
	}
	if poisoned(x, w, counts) {
		return b.poison()
	}
	for _, c := range []struct {
		name string
		t    *Tensor
		want DType
	}{{"x", x, accel.F32}, {"w", w, accel.F32}, {"counts", counts, accel.U32}} {
		if c.t.dtype != c.want {
			return b.fail(1, "GroupedMatMul", "%s is %v and this kernel reads %v",
				c.name, c.t.dtype, c.want)
		}
	}
	if len(x.shape) != 2 {
		return b.fail(1, "GroupedMatMul", "x is %v; it is [tokens, K] with the tokens "+
			"of every expert end to end", x.shape)
	}
	if len(w.shape) != 3 {
		return b.fail(1, "GroupedMatMul", "w is %v; it is [experts, K, N], one matrix "+
			"per expert", w.shape)
	}
	tokens, k := x.shape[0], x.shape[1]
	experts, wk, n := w.shape[0], w.shape[1], w.shape[2]
	if wk != k {
		return b.fail(1, "GroupedMatMul", "x is %v and w is %v; every expert contracts "+
			"against the same width, so w's second axis is x's second", x.shape, w.shape)
	}
	if got := counts.shape.Elements(); got != experts {
		return b.fail(1, "GroupedMatMul", "counts holds %d entries and w has %d "+
			"experts; it is one token count per expert", got, experts)
	}
	if experts == 0 {
		return b.fail(1, "GroupedMatMul", "w declares no experts")
	}

	offsets := b.segmentOffsets(counts, experts)
	if poisoned(offsets) {
		return b.poison()
	}

	return b.record(node{
		op: "GroupedMatMul", inputs: []*Tensor{x, w, offsets},
		kernel: &kernels.GroupedMatMulKernel,
		uniform: func(map[string]ScalarValue) any {
			// Tokens is x's row count, and it is the bound the offsets cannot
			// give: they are device data, so nothing here can check they sum to
			// it. specs/049-grouped-gemm.md §5.
			return kernels.GroupedTiledDims{
				Experts: uint32(experts), Tokens: uint32(tokens),
				K: uint32(k), N: uint32(n),
			}
		},
		grid: func(*Tensor) accel.WorkgroupCount {
			// One workgroup per (expert, column tile). Not per token: a token
			// block spanning two experts would need two weight matrices in one
			// tile, which is the tile's whole reason for existing.
			return accel.WorkgroupCount{
				X: (n + kernels.TileN - 1) / kernels.TileN,
				Y: experts,
			}
		},
		reason: fmt.Sprintf("the tiled grouped kernel: %d experts over %d columns, "+
			"each walking its own segment of %d tokens in blocks of %d",
			experts, n, tokens, kernels.TileM),
		rejected: []string{"the grouped row kernel: it reduces one token per " +
			"workgroup, so an expert's matrix is re-read once per token rather " +
			"than once per block"},
	}, accel.F32, Shape{tokens, n})
}
