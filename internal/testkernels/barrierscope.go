// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import "golang.design/x/accel"

// PublishStorage has one lane write a buffer and every lane read what it wrote,
// with a storage-scope barrier between.
//
// specs/050-barrier-scopes.md §3's accepting half. Lane 0 writes
// `scratch[group]` and every lane of the workgroup then reads it, which is
// exactly the handoff [002](002-compute-model.md) §2.5 says `BarrierStorage`
// orders and `BarrierShared` does not.
//
// # Why the payload goes through a buffer and not shared memory
//
// A shared array is the class `BarrierShared` already orders, so a kernel that
// published through one would pass at every scope and say nothing. The
// distinction this kernel exists for is a *storage* write made visible, so the
// value crosses a `[]uint32` binding.
//
// It writes what it read rather than a constant, so a lane that raced and saw
// the initial contents reports a different number instead of the same one.
//
//accel:kernel workgroup=32
func PublishStorage(t accel.Thread, scratch []uint32, out []uint32) {
	g := t.GroupID().X
	lane := t.LocalID().X

	// One lane publishes a value derived from the workgroup, so a lane reading
	// a neighbour's slot reports that neighbour's number.
	if lane == 0 {
		scratch[g] = g*1000 + 7
	}
	t.BarrierStorage()

	out[g*t.WorkgroupSize().X+lane] = scratch[g]
}

// PublishShared is [PublishStorage]'s legal counterpart: the payload is a
// shared array, so the cheaper barrier is the correct one.
//
// It exists so `BarrierShared` has an accepting half of its own. Asserting only
// that it emits a narrower `mem_flags` mask would leave a lowering that emitted
// the right text and rendezvoused nobody indistinguishable from a correct one.
//
//accel:kernel workgroup=32
func PublishShared(t accel.Thread, out []uint32, sh *[32]uint32) {
	g := t.GroupID().X
	lane := t.LocalID().X

	sh[lane] = g*1000 + lane
	t.BarrierShared()

	// Every lane reads the lane below it, so a missing rendezvous shows as a
	// value from before the write rather than as a value from the right slot.
	out[g*t.WorkgroupSize().X+lane] = sh[(lane+1)%t.WorkgroupSize().X]
}
