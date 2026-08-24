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
// # Fusion is not a selection, because there is nothing to select between
//
// specs/007-tensor-layer.md said fused attention is "runtime kernel selection,
// not a device capability", with the composed definition -- score MatMul,
// Softmax, value MatMul -- as both the correctness reference and the fallback.
// The reference half holds and the fallback half does not: several query heads
// share one key/value head, so the composed form needs a matrix multiply per
// head, and specs/025-tensor-operators.md multiplies two matrices with no
// leading axes broadcast. The composition exists only at kvHeads == 1, which no
// model this serves uses. 007 is corrected; the fused kernels are the only
// path, and specs/044-unbounded-context.md is why they can be.
//
// The composed reference still runs, in the corpus tests, over the shapes it
// can express. That is what keeps it a reference.
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
	// q's rank says which computation this is, and Selections says which kernel
	// ran: a rank is not a hint, it is the shape of the computation.
	//
	//	[qHeads, headDim]                  one sequence, one token   -- a decode
	//	[qSeq, qHeads, headDim]            one sequence, many tokens -- a prefill
	//	[batch, qSeq, qHeads, headDim]     several sequences         -- batched
	//
	// The batch axis is rank 4 rather than rank 3 because rank 3 already means
	// a prefill, and a consumer reported what that cost: everything *around*
	// attention was already batched -- Lengths is per sequence, Pages is
	// [s][i], RoPE takes per-row positions, the samplers draw per row -- and
	// the shape was the one thing that was not, so a batched decode was read as
	// a prefill and refused for a missing BaseName (accel issue 12).
	//
	// qSeq is carried in the batched form rather than dropped so that a batched
	// prefill is this shape with qSeq > 1 when specs/040-batch-scheduler.md
	// builds it, rather than a fifth rank.
	qSeq, batch := 1, 1
	prefill, batched := false, false
	flatten := func(from int) {
		shape := Shape{q.shape[from], q.shape[from+1]}
		q = &Tensor{
			b: b, dtype: q.dtype, shape: shape,
			strides: contiguous(shape), node: q.node, port: q.port, win: q.win,
		}
	}
	switch len(q.shape) {
	case 2:
	case 3:
		qSeq, prefill = q.shape[0], true
		flatten(1)
	case 4:
		batch, qSeq, batched = q.shape[0], q.shape[1], true
		flatten(2)
	default:
		return b.fail(1, "Attention", "q is %v; it is [qHeads, headDim] for one token, "+
			"[qSeq, qHeads, headDim] for a prefill, or [batch, qSeq, qHeads, headDim] "+
			"for several sequences stepping together", q.shape)
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

	// The page table is checked before the shape decides which kernel runs,
	// because a check that lives inside one branch is a check the other branch
	// does not have. This one lived inside the decode selection, so a prefill
	// reached neither it nor the table: Pages was accepted, never bound, and
	// the cache was read contiguously -- a plausible wrong answer rather than a
	// diagnostic, which a consumer measured at 0.74 absolute against a reversed
	// table (accel issue 10).
	if opts.Pages != nil {
		if opts.Pages.dtype != accel.U32 {
			return b.fail(1, "Attention", "Pages is %v and a page table is u32",
				opts.Pages.dtype)
		}
		if opts.Block <= 0 {
			return b.fail(1, "Attention", "Pages is set and Block is %d; a page table "+
				"addresses blocks, so how many positions one holds is required",
				opts.Block)
		}
	}

	// One length per sequence, which is one row unless a batch axis says
	// otherwise. Unchecked until now, and a prefill quietly read row zero and
	// ignored the rest -- a value the caller supplied that reached nothing,
	// which is the shape of defect this operator has had six of.
	if got := opts.Lengths.shape.Elements(); got != batch {
		return b.fail(1, "Attention", "Lengths holds %d entries and this step has %d "+
			"sequence(s); it is one length per sequence", got, batch)
	}

	if batched {
		// One kernel is registered for a batch and it reads a page table.
		// That is not an accident of the corpus: sequences that step together
		// have different lengths and cannot share a contiguous cache without
		// padding every one of them to the longest, which is the allocation
		// paging exists to avoid (specs/030-paged-kv.md).
		if opts.Pages == nil {
			return b.fail(1, "Attention", "a batch of %d sequences needs Pages: they have "+
				"different lengths, so a contiguous cache would pad every sequence to the "+
				"longest. specs/010-kernel-corpus.md registers the batched decode over a "+
				"page table and no contiguous variant", batch)
		}
		if qSeq != 1 {
			return b.fail(1, "Attention", "q is %v: a batched step takes one token per "+
				"sequence, and a batched *prefill* is specs/040-batch-scheduler.md's. "+
				"The shape has room for it -- qSeq is this axis -- and no kernel does",
				Shape{batch, qSeq, qHeads, headDim})
		}
		if cacheDType != accel.F32 {
			return b.fail(1, "Attention", "the cache is %v and the batched decode kernel "+
				"reads f32; specs/010-kernel-corpus.md owns the narrow variant",
				cacheDType)
		}
		if len(opts.Pages.shape) != 2 || opts.Pages.shape[0] != batch {
			return b.fail(1, "Attention", "Pages is %v and the batch is %d; a batched page "+
				"table is [batch, maxPages], one row of block ids per sequence",
				opts.Pages.shape, batch)
		}
		maxPages := opts.Pages.shape[1]
		return b.record(node{
			op: "Attention",
			// Pages before lengths, which is the kernel's binding order.
			inputs: []*Tensor{
				q, readState(b, k), readState(b, v), opts.Pages, opts.Lengths,
			},
			kernel: &testkernels.AttentionDecodeBatchedKernel,
			reads:  []string{opts.ScaleName},
			uniform: func(vals map[string]ScalarValue) any {
				return testkernels.BatchedDims{
					Batch: uint32(batch), QHeads: uint32(qHeads),
					KVHeads: uint32(kvHeads), HeadDim: uint32(headDim),
					Block: uint32(opts.Block), MaxPages: uint32(maxPages),
					Scale: vals[opts.ScaleName].F32,
				}
			},
			grid: func(*Tensor) accel.WorkgroupCount {
				// Sequence-major, so one sequence's heads are adjacent and read
				// the same page-table row.
				return accel.WorkgroupCount{X: batch * qHeads}
			},
			reason: fmt.Sprintf("the batched paged decode kernel: %d sequences of %d heads "+
				"step together over blocks of %d, one workgroup per (sequence, head)",
				batch, qHeads, opts.Block),
			rejected: []string{
				"the single-sequence decode kernels: they take one sequence per dispatch, " +
					"so a batch would be one submission each and read every weight once " +
					"per sequence",
			},
		}, accel.F32, Shape{batch, qHeads, headDim})
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
		// Paging is not a second kind of prefill. The causal bound, the running
		// softmax and the masking are the same; what differs is the binding, so
		// this is a selection and Selections reports it -- the shape every
		// other choice here takes.
		prefillKernel := &testkernels.AttentionPrefillKernel
		prefillInputs := []*Tensor{q, readState(b, k), readState(b, v), opts.Lengths}
		prefillWhy := fmt.Sprintf("the causal prefill kernel: one workgroup per query "+
			"position and head, %d of them", qSeq*qHeads)
		prefillRejected := []string{"the decode kernel: it takes one query token"}
		prefillDims := func(vals map[string]ScalarValue) any {
			return testkernels.PrefillDims{
				QHeads: uint32(qHeads), KVHeads: uint32(kvHeads),
				HeadDim: uint32(headDim), QSeq: uint32(qSeq),
				Base:  vals[opts.BaseName].U32,
				Scale: vals[opts.ScaleName].F32,
			}
		}
		if opts.Pages != nil {
			prefillKernel = &testkernels.AttentionPrefillPagedKernel
			// Pages before lengths, which is the kernel's binding order.
			prefillInputs = []*Tensor{
				q, readState(b, k), readState(b, v), opts.Pages, opts.Lengths,
			}
			prefillWhy = fmt.Sprintf("the paged causal prefill kernel: blocks of %d "+
				"addressed through a page table, one workgroup per query position and "+
				"head, %d of them", opts.Block, qSeq*qHeads)
			prefillRejected = append(prefillRejected,
				"the contiguous prefill kernel: a page table was supplied, and it would "+
					"read the pool in order")
			prefillDims = func(vals map[string]ScalarValue) any {
				return testkernels.PagedPrefillDims{
					QHeads: uint32(qHeads), KVHeads: uint32(kvHeads),
					HeadDim: uint32(headDim), QSeq: uint32(qSeq),
					Base:  vals[opts.BaseName].U32,
					Block: uint32(opts.Block),
					Scale: vals[opts.ScaleName].F32,
				}
			}
		}
		return b.record(node{
			op:      "Attention",
			inputs:  prefillInputs,
			kernel:  prefillKernel,
			reads:   []string{opts.ScaleName, opts.BaseName},
			uniform: prefillDims,
			grid: func(*Tensor) accel.WorkgroupCount {
				return accel.WorkgroupCount{X: qSeq * qHeads}
			},
			reason:   prefillWhy,
			rejected: prefillRejected,
		}, accel.F32, out)
	}

	// A base position places a causal mask over a sequence of queries, and a
	// decode step has one query token and no mask. Accepting it here would be
	// the same defect as Pages on a prefill, one field over: a value the caller
	// set, that reaches nothing, with no way to tell.
	if opts.BaseName != "" {
		return b.fail(1, "Attention", "BaseName is set on a decode step, which has one "+
			"query token and therefore no causal mask to place; it belongs to a prefill. "+
			"A decode attends over the whole cache its Lengths names")
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
