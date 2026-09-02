// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import "golang.design/x/accel"

// Exchange has every invocation publish its own value and then read its
// neighbour's, which is the smallest program a barrier is genuinely necessary
// for.
//
// The neighbour matters. A kernel where one invocation publishes and the rest
// read would still produce the right answer if the invocations ran one after
// another, because the publisher happens to go first — so it cannot tell a real
// rendezvous from a barrier lowered as a no-op. Here invocation i reads slot
// i+1, which under sequential execution has not been written yet and holds
// poison. The wrong lowering produces NaNs rather than the right answer slowly.
//
// Everything the transform must do appears here once: shared memory, a write
// before the barrier, a read after it, and locals that have to survive the
// suspension.
//
//accel:kernel workgroup=64
func Exchange(t accel.Thread, in []float32, out []float32, sh *[64]float32) {
	lid := t.LocalID().X
	gid := t.GlobalID().X

	sh[lid] = in[gid]
	t.Barrier()

	next := lid + 1
	if next == 64 {
		next = 0
	}
	if gid < uint32(len(out)) {
		out[gid] = sh[next]
	}
}

// ReduceLoop sums a workgroup's values with a halving stride, one barrier per
// round, which is the shape a tree reduction has and the reason the state
// machine has to resume inside a loop.
//
// Its result is checked against [ReduceUnrolled] rather than against a golden
// of the generated shape: the two compute the same thing by construction, one
// through the loop split and one through the top-level split that already
// worked, so a disagreement is the new state numbering's and nothing else's.
//
//accel:kernel workgroup=64
func ReduceLoop(t accel.Thread, in []float32, out []float32, sh *[64]float32) {
	lid := t.LocalID().X
	gid := t.GlobalID().X

	sh[lid] = in[gid]
	t.Barrier()

	for stride := uint32(32); stride > 0; stride /= 2 {
		if lid < stride {
			sh[lid] = sh[lid] + sh[lid+stride]
		}
		t.Barrier()
	}

	if lid == 0 {
		out[t.GroupID().X] = sh[0]
	}
}

// ReduceUnrolled is the same reduction with its barriers at the top level,
// which the split handled before loops did.
//
// It exists to be the oracle for [ReduceLoop]. Six rounds written out is not
// how anyone would write a reduction, and that is the point: it uses only the
// transform that was already checked, so the comparison isolates what is new.
//
//accel:kernel workgroup=64
func ReduceUnrolled(t accel.Thread, in []float32, out []float32, sh *[64]float32) {
	lid := t.LocalID().X
	gid := t.GlobalID().X

	sh[lid] = in[gid]
	t.Barrier()

	if lid < 32 {
		sh[lid] = sh[lid] + sh[lid+32]
	}
	t.Barrier()
	if lid < 16 {
		sh[lid] = sh[lid] + sh[lid+16]
	}
	t.Barrier()
	if lid < 8 {
		sh[lid] = sh[lid] + sh[lid+8]
	}
	t.Barrier()
	if lid < 4 {
		sh[lid] = sh[lid] + sh[lid+4]
	}
	t.Barrier()
	if lid < 2 {
		sh[lid] = sh[lid] + sh[lid+2]
	}
	t.Barrier()
	if lid < 1 {
		sh[lid] = sh[lid] + sh[lid+1]
	}
	t.Barrier()

	if lid == 0 {
		out[t.GroupID().X] = sh[0]
	}
}
