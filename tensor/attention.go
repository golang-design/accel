// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"fmt"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// AttentionOptions carries what attention needs beyond its operands.
type AttentionOptions struct {
	// Lengths is a u32 tensor holding, per sequence, how much of that
	// sequence's cache holds real tokens.
	//
	// A tensor rather than a scalar, and specs/043-per-row-values.md draws the
	// line: a value every row of a dispatch shares is a uniform, and a value
	// that differs per row is device data. Cache lengths are genuinely
	// independent across a batch -- that is what continuous batching *is* --
	// so no scalar can express them, and no scheduling arrangement makes one
	// correct.
	//
	// One sequence binds a one-element tensor. That is the same path, not a
	// special case.
	Lengths *Tensor

	// Pages is an optional u32 page table: entry [s][i] is the physical block
	// holding sequence s's i-th logical block.
	//
	// Nil means the cache is contiguous, which is the same thing with an
	// identity table and a block size of one. It is nil-able rather than
	// required because the indirection is a real cost in the innermost loop of
	// decode, and a contiguous cache should not pay it -- the operator selects
	// the kernel and reports which in [Plan.Selections].
	//
	// Paging is not a second kind of cache. A State addressed through a page
	// table is the same State; what differs is the binding.
	Pages *Tensor

	// Block is how many positions one physical block holds, and is required
	// when Pages is set.
	Block int

	// ScaleName is a declared f32 scalar, conventionally 1/sqrt(headDim). Named
	// rather than computed here so a caller can use a different convention
	// without a different plan.
	ScaleName string

	// BaseName is a declared u32 scalar, required for a prefill: the position
	// of its first query token within the cache. It decides what the causal
	// mask hides, so a prefill that extends an existing cache masks correctly
	// rather than letting its first token see nothing.
	//
	// Still a scalar because a prefill is one sequence: specs/040-batch-scheduler.md
	// owns batched prefill, and until it exists there is no row for this to
	// differ across.
	BaseName string
}

// Attention scores a query against a cached key/value pair.
//
// # Why the cache is State and the query is a Tensor
//
// The query is this step's; the cache is every step's. That asymmetry is the
// whole shape of decoding, and expressing it in the types means a caller cannot
// accidentally write the query or read a stale cache: a State carries a version
// and a Tensor does not.
//
// # Fusion is a selection
//
// specs/007-tensor-layer.md is explicit that fused attention is "runtime kernel
// selection, not a device capability", and that the composed definition -- score
// MatMul, Softmax, value MatMul -- is the correctness reference. v0 selects the
// fused decode kernel when the shapes fit it and reports that it did; it does
// not yet fall back to the composed graph, and says so rather than pretending
// the choice was made.
func Attention(b *Builder, q *Tensor, k, v *State, opts AttentionOptions) *Tensor {
	if poisoned(q) || k == nil || v == nil || k.poison || v.poison {
		return b.poison()
	}
	if q.dtype != accel.F32 {
		return b.fail(1, "Attention", "q is %v and the registered kernels read an f32 "+
			"query; it is one row and narrowing it saves nothing worth a variant", q.dtype)
	}
	// The cache may be f16, and specs/043-per-row-values.md §5 says why that is
	// defensible where narrow accumulation is not: K and V are *operands*, and
	// the score accumulates in f32 whatever they are stored as -- which is the
	// trade MatMul already makes. It halves the largest allocation a serving
	// process has after the weights, and the only one that scales with both
	// concurrency and context.
	if k.desc.DType != v.desc.DType {
		return b.fail(1, "Attention", "the key cache is %v and the value cache is %v; one "+
			"kernel reads both, so they are the same dtype", k.desc.DType, v.desc.DType)
	}
	cacheDType := k.desc.DType
	if cacheDType != accel.F32 && cacheDType != accel.F16 {
		return b.fail(1, "Attention", "the cache is %v; the registered kernels read f32 or "+
			"f16", cacheDType)
	}
	// [qHeads, headDim] is one token and [qSeq, qHeads, headDim] is a prefill.
	// Which one a caller wrote decides which kernel runs, and Selections says
	// so: a rank is not a hint, it is the shape of the computation.
	qSeq := 1
	prefill := false
	switch len(q.shape) {
	case 2:
	case 3:
		qSeq, prefill = q.shape[0], true
		q = &Tensor{
			b: b, dtype: q.dtype, shape: Shape{q.shape[1], q.shape[2]},
			strides: contiguous(Shape{q.shape[1], q.shape[2]}), node: q.node, port: q.port,
		}
	default:
		return b.fail(1, "Attention", "q is %v; it is [qHeads, headDim] for one token or "+
			"[qSeq, qHeads, headDim] for a prefill", q.shape)
	}
	if len(k.shape) != 3 || !k.shape.Equal(v.shape) {
		return b.fail(1, "Attention", "the key cache is %v and the value cache is %v; both "+
			"are [capacity, kvHeads, headDim] and equal", k.shape, v.shape)
	}
	qHeads, headDim := q.shape[0], q.shape[1]
	kvHeads := k.shape[1]
	if k.shape[2] != headDim {
		return b.fail(1, "Attention", "q's head is %d wide and the cache's is %d",
			headDim, k.shape[2])
	}
	if kvHeads == 0 || qHeads%kvHeads != 0 {
		return b.fail(1, "Attention", "%d query heads over %d key/value heads; several "+
			"query heads share one cache entry, so the first is a multiple of the second",
			qHeads, kvHeads)
	}
	if opts.Lengths == nil || poisoned(opts.Lengths) {
		return b.fail(1, "Attention", "Lengths is required: it holds how much of each "+
			"sequence's cache is real, one entry per sequence "+
			"(specs/043-per-row-values.md). A single sequence binds one element")
	}
	if opts.Lengths.dtype != accel.U32 {
		return b.fail(1, "Attention", "Lengths is %v and a cache length is u32",
			opts.Lengths.dtype)
	}
	if kind, ok := b.scalarKind(opts.ScaleName); !ok || kind != ScalarF32 {
		return b.fail(1, "Attention", "%q is not a declared f32 scalar", opts.ScaleName)
	}
	if stale(b, k) {
		return b.fail(1, "Attention", "the key cache is %s", staleMessage(b, k))
	}
	if stale(b, v) {
		return b.fail(1, "Attention", "the value cache is %s", staleMessage(b, v))
	}
	if k.offset != 0 || v.offset != 0 {
		return b.fail(1, "Attention", "a layer view binds a range of a resource, which a "+
			"slot cannot express yet; use one state per layer")
	}

	// The kernel gives each query head a workgroup and each lane one cached
	// position, so the capacity it can score is bounded by the workgroup.
	lanes := int(testkernels.AttentionDecodeKernel.WorkgroupSize.X)
	if k.shape[0] > lanes {
		return b.fail(1, "Attention", "the cache holds %d positions and the decode kernel "+
			"scores one per lane over %d; a longer cache needs the looping variant, which "+
			"specs/010-kernel-corpus.md does not register", k.shape[0], lanes)
	}

	out := Shape{qHeads, headDim}
	if prefill && cacheDType != accel.F32 {
		return b.fail(1, "Attention", "the cache is %v and only the decode kernel reads a "+
			"narrow cache; specs/010-kernel-corpus.md owns the prefill variant. A prefill "+
			"over an f16 cache is refused rather than run against the f32 kernel, which "+
			"would read every second entry", cacheDType)
	}
	if prefill {
		out = Shape{qSeq, qHeads, headDim}
		if opts.BaseName == "" {
			return b.fail(1, "Attention", "a prefill needs BaseName: the position of its "+
				"first query within the cache, which decides what the causal mask hides")
		}
		if kind, ok := b.scalarKind(opts.BaseName); !ok || kind != ScalarU32 {
			return b.fail(1, "Attention", "%q is not a declared u32 scalar", opts.BaseName)
		}
		return b.record(node{
			op:     "Attention",
			inputs: []*Tensor{q, readState(b, k), readState(b, v), opts.Lengths},
			kernel: &testkernels.AttentionPrefillKernel,
			reads:  []string{opts.ScaleName, opts.BaseName},
			uniform: func(vals map[string]ScalarValue) any {
				return testkernels.PrefillDims{
					QHeads: uint32(qHeads), KVHeads: uint32(kvHeads),
					HeadDim: uint32(headDim), QSeq: uint32(qSeq),
					Base:  vals[opts.BaseName].U32,
					Scale: vals[opts.ScaleName].F32,
				}
			},
			grid: func(*Tensor) accel.WorkgroupCount {
				return accel.WorkgroupCount{X: qSeq * qHeads}
			},
			reason: fmt.Sprintf("the causal prefill kernel: one workgroup per query "+
				"position and head, %d of them", qSeq*qHeads),
			rejected: []string{"the decode kernel: it takes one query token"},
		}, accel.F32, out)
	}

	// Which decode kernel runs. The caller writes one shape and the choice is
	// reported in Plan.Selections, which is how every other operator here
	// behaves: a variant is a selection, not a second API.
	decode := &testkernels.AttentionDecodeKernel
	decodeWhy := "the fused decode kernel"
	inputs := []*Tensor{q, readState(b, k), readState(b, v), opts.Lengths}
	rejected := []string{"the causal prefill kernel: it takes a sequence of query tokens"}

	switch {
	case opts.Pages != nil:
		if opts.Pages.dtype != accel.U32 {
			return b.fail(1, "Attention", "Pages is %v and a page table is u32",
				opts.Pages.dtype)
		}
		if opts.Block <= 0 {
			return b.fail(1, "Attention", "Pages is set and Block is %d; a page table "+
				"addresses blocks, so how many positions one holds is required",
				opts.Block)
		}
		if cacheDType != accel.F32 {
			return b.fail(1, "Attention", "the cache is %v and only the contiguous decode "+
				"kernel reads a narrow cache; specs/010-kernel-corpus.md owns the paged "+
				"f16 variant", cacheDType)
		}
		decode = &testkernels.AttentionDecodePagedKernel
		decodeWhy = fmt.Sprintf("the paged decode kernel: blocks of %d, addressed through "+
			"a page table", opts.Block)
		// Pages before lengths, which is the kernel's binding order.
		inputs = []*Tensor{q, readState(b, k), readState(b, v), opts.Pages, opts.Lengths}
		rejected = append(rejected,
			"the contiguous decode kernel: a page table was supplied")

	case cacheDType == accel.F16:
		decode = &testkernels.AttentionDecodeF16Kernel
		decodeWhy = "the fused decode kernel over an f16 cache, accumulating f32"
	}

	return b.record(node{
		op:     "Attention",
		inputs: inputs,
		kernel: decode,
		reads:  []string{opts.ScaleName},
		uniform: func(vals map[string]ScalarValue) any {
			if opts.Pages != nil {
				return testkernels.PagedDims{
					QHeads: uint32(qHeads), KVHeads: uint32(kvHeads),
					HeadDim: uint32(headDim), Block: uint32(opts.Block),
					Scale: vals[opts.ScaleName].F32,
				}
			}
			return testkernels.AttnDims{
				QHeads: uint32(qHeads), KVHeads: uint32(kvHeads),
				HeadDim: uint32(headDim),
				Scale:   vals[opts.ScaleName].F32,
			}
		},
		grid: func(*Tensor) accel.WorkgroupCount {
			return accel.WorkgroupCount{X: qHeads}
		},
		reason: fmt.Sprintf("%s: one workgroup per query head over %d cached positions",
			decodeWhy, k.shape[0]),
		rejected: rejected,
	}, accel.F32, out)
}
