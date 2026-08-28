// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import "golang.design/x/accel"

// IntReduce writes each subgroup's integer minimum and maximum, per lane.
//
// specs/059-subgroup-reductions.md §6's first slice. The four operations run in
// one kernel because they share a shape and differ in exactly the two ways a
// transposition confuses: which comparison, and which type.
//
// # Why every lane writes, and why the inputs are not monotone
//
// A reduction is broadcast to every active lane, so writing per lane is what
// says the broadcast happened rather than only the combining. And the inputs
// are shuffled by a multiply rather than being the lane index: with a monotone
// input the minimum is always lane 0's and the maximum always the last lane's,
// so a kernel that returned a fixed lane's value instead of reducing would
// pass.
//
// The u32 inputs are the i32 ones reinterpreted through a bias, so a lowering
// that read the wrong carrier -- the failure §2's field-per-type exists to
// prevent -- produces a different number rather than the same one.
//
//accel:kernel workgroup=64
//accel:requires subgroup_arithmetic
func IntReduce(t accel.Thread, in []int32, minI []int32, maxI []int32,
	minU []uint32, maxU []uint32) {

	i := t.LocalID().X
	v := in[i]

	// Each rendezvous is assigned to its own local, which the cooperative
	// lowering requires: it suspends at each, and a call inside a larger
	// expression would have to resume part-way through evaluating it.
	lo := t.SubgroupMinI32(v)
	hi := t.SubgroupMaxI32(v)

	// The same values as unsigned. A negative i32 becomes a large u32, so the
	// unsigned minimum and maximum land on different lanes than the signed
	// ones -- which is what makes the two pairs distinguishable rather than
	// two spellings of one answer.
	u := uint32(v)
	loU := t.SubgroupMinU32(u)
	hiU := t.SubgroupMaxU32(u)

	minI[i] = lo
	maxI[i] = hi
	minU[i] = loU
	maxU[i] = hiU
}
