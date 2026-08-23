// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

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
