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
	// CurrentLengthName is a declared u32 scalar: how much of the cache holds
	// real tokens. It is a scalar rather than a shape because it changes every
	// decode step and the plan must not.
	CurrentLengthName string

	// ScaleName is a declared f32 scalar, conventionally 1/sqrt(headDim). Named
	// rather than computed here so a caller can use a different convention
	// without a different plan.
	ScaleName string

	// BaseName is a declared u32 scalar, required for a prefill: the position
	// of its first query token within the cache. It decides what the causal
	// mask hides, so a prefill that extends an existing cache masks correctly
	// rather than letting its first token see nothing.
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
	if _, ok := b.scalarKind(opts.CurrentLengthName); !ok {
		return b.fail(1, "Attention", "%q is not a declared scalar; the cache's current "+
			"length changes every step and is named for that reason", opts.CurrentLengthName)
	}
	if kind, _ := b.scalarKind(opts.CurrentLengthName); kind != ScalarU32 {
		return b.fail(1, "Attention", "%q is declared %v and a length is u32",
			opts.CurrentLengthName, kind)
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
			op: "Attention", inputs: []*Tensor{q, readState(b, k), readState(b, v)},
			kernel: &testkernels.AttentionPrefillKernel,
			reads:  []string{opts.CurrentLengthName, opts.ScaleName, opts.BaseName},
			uniform: func(vals map[string]ScalarValue) any {
				return testkernels.PrefillDims{
					QHeads: uint32(qHeads), KVHeads: uint32(kvHeads),
					HeadDim: uint32(headDim), QSeq: uint32(qSeq),
					KVLen: vals[opts.CurrentLengthName].U32,
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

	// The narrow variant when the cache is narrow. A separate kernel rather
	// than a dtype parameter, because specs/004-kernel-authoring.md keeps
	// generics out of the subset and a variant is what
	// specs/010-kernel-corpus.md registers.
	decode := &testkernels.AttentionDecodeKernel
	decodeWhy := "the fused decode kernel"
	if cacheDType == accel.F16 {
		decode = &testkernels.AttentionDecodeF16Kernel
		decodeWhy = "the fused decode kernel over an f16 cache, accumulating f32"
	}

	return b.record(node{
		op: "Attention", inputs: []*Tensor{q, readState(b, k), readState(b, v)},
		kernel: decode,
		reads:  []string{opts.CurrentLengthName, opts.ScaleName},
		uniform: func(vals map[string]ScalarValue) any {
			return testkernels.AttnDims{
				QHeads: uint32(qHeads), KVHeads: uint32(kvHeads),
				HeadDim: uint32(headDim),
				KVLen:   vals[opts.CurrentLengthName].U32,
				Scale:   vals[opts.ScaleName].F32,
			}
		},
		grid: func(*Tensor) accel.WorkgroupCount {
			return accel.WorkgroupCount{X: qHeads}
		},
		reason: fmt.Sprintf("%s: one workgroup per query head over %d cached positions",
			decodeWhy, k.shape[0]),
		rejected: []string{"the causal prefill kernel: it takes a sequence of query tokens"},
	}, accel.F32, out)
}
