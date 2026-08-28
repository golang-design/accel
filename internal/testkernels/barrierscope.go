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

// SubgroupPublish has one lane of each subgroup write a shared slot and every
// lane of that subgroup read it, with a subgroup-scope barrier between.
//
// specs/050-barrier-scopes.md §3's fourth assertion and 002 §5.3's legal shape.
// The barrier sits under `if t.SubgroupIndex() < ...`-style control in the test
// that exercises rejection; here it is at the top level, so what this kernel
// says is that the narrower rendezvous *works*: each subgroup's lanes see their
// own subgroup's write.
//
// # Why each subgroup writes its own slot
//
// A single shared slot written by one lane and read by the whole workgroup
// would need a workgroup barrier, and would pass here at any scope on the CPU
// because the scheduler advances invocations one at a time within an epoch. One
// slot per subgroup is the shape a subgroup barrier actually orders, and it is
// what makes the emulated size sweep meaningful: at size 1 every lane is its own
// subgroup and reads what it wrote, at 64 there is one subgroup, and the answer
// is the same either way only if the indexing is right.
//
//accel:kernel workgroup=64
//accel:requires subgroup_basic
func SubgroupPublish(t accel.Thread, out []float32, sh *[64]float32) {
	lane := t.SubgroupLane()
	sid := t.SubgroupIndex()

	// The lowest lane of each subgroup publishes into that subgroup's slot.
	if lane == 0 {
		sh[sid] = float32(sid) + 1
	}
	t.SubgroupBarrier()

	// LocalID().X rather than LocalIndex(): the workgroup is one-dimensional so
	// they are the same number, and LocalIndex is outside the MSL subset --
	// which would leave this kernel with no Metal half and nothing to compare.
	out[t.LocalID().X] = sh[sid]
}

// SubgroupStagger runs a subgroup barrier a different number of times per
// subgroup, which is the shape only a subgroup-scope rendezvous permits.
//
// specs/002-compute-model.md §5.3 and §12. The loop's trip count is
// `SubgroupIndex`, which is subgroup-uniform: every lane of one subgroup runs
// the loop the same number of times and different subgroups run it different
// numbers of times. Around a workgroup barrier that is illegal and the compiler
// refuses it; around this one it is legal, and it is the case that distinguishes
// a per-subgroup arrival check from a workgroup-wide one.
//
// # What it computes, and why that is checkable
//
// Each lane copies its own subgroup index through shared memory. The value is
// deliberately independent of how many times the loop ran, because what is
// under test is that the kernel *completes* -- a workgroup-wide arrival check
// reports the staggered subgroups as a non-uniform arrival and the dispatch
// fails with a diagnostic rather than a wrong number.
//
//accel:kernel workgroup=64
//accel:requires subgroup_basic
func SubgroupStagger(t accel.Thread, out []float32, sh *[64]float32) {
	sid := t.SubgroupIndex()
	lane := t.LocalID().X
	sh[lane] = float32(sid) + 1

	// Subgroup 0 does not enter, subgroup 1 goes round once, and so on.
	for i := uint32(0); i < sid; i++ {
		t.SubgroupBarrier()
	}

	out[lane] = sh[lane]
}
