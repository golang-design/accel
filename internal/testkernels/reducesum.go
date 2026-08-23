// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import "golang.design/x/accel"

// ReduceSumWidth is the workgroup width spec 010 fixes for reduce_sum.
const ReduceSumWidth = 128

// ReduceSum sums a buffer of arbitrary length into out[0].
//
// # Why a tree and not a loop
//
// specs/008-numerics.md section 7 gives a sequential sum of K terms the error
// factor γ(K-1) and a balanced pairwise tree of depth d the factor γ(d). For
// 128 terms that is γ(127) against γ(7), so the tree is not merely faster: it
// is *more accurate*, by a factor of eighteen in the bound. That is the reason
// spec 010 specifies this shape rather than the obvious loop.
//
// # Why the guarded strided load
//
// The length is arbitrary and the workgroup is fixed, so each invocation first
// folds its own strided slice sequentially and the tree runs over the 128
// partial sums. The guard is what admits a length that is not a multiple of the
// width, which is where an off-by-one in a reduction hides — and is exactly
// what spec 009's done criterion asks to be tested.
//
// The maximum addition depth is therefore ⌈n/128⌉-1 for the fold plus 7 for the
// tree, and the test computes the budget from that rather than from the term
// count: specs/008-numerics.md is explicit that the harness does not infer
// "tree" from a name.
//
//accel:kernel workgroup=128
func ReduceSum(t accel.Thread, in []float32, out []float32, sh *[128]float32) {
	lid := t.LocalID().X
	n := uint32(len(in))

	// Each invocation folds its strided slice. Strided rather than blocked so
	// that consecutive invocations read consecutive elements, which is the
	// access pattern every target coalesces.
	acc := float32(0)
	for i := lid; i < n; i += ReduceSumWidth {
		acc = acc + in[i]
	}
	sh[lid] = acc
	t.Barrier()

	// The tree. Each round halves the live width, and the barrier after it is
	// what makes the next round's reads see this round's writes.
	for stride := uint32(ReduceSumWidth / 2); stride > 0; stride /= 2 {
		if lid < stride {
			sh[lid] = sh[lid] + sh[lid+stride]
		}
		t.Barrier()
	}

	if lid == 0 {
		out[0] = sh[0]
	}
}
