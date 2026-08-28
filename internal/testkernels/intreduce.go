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

// BitReduce writes each subgroup's And, Or and Xor, per lane, over both integer
// types.
//
// specs/059-subgroup-reductions.md §6's second slice. Six operations in one
// kernel for the reason [IntReduce] holds four: they share a shape and differ
// in the two ways a transposition confuses.
//
// # Why the input is a bit pattern rather than a counter
//
// And over a run of consecutive integers is almost always zero and Or is almost
// always all-ones, so both are satisfied by a kernel that ignores its input.
// The pattern below gives each lane a different sparse set of bits, so the
// three answers differ from each other and from any constant.
//
//accel:kernel workgroup=64
//accel:requires subgroup_arithmetic
func BitReduce(t accel.Thread, in []int32, andI []int32, orI []int32,
	xorI []int32, andU []uint32, orU []uint32, xorU []uint32) {

	i := t.LocalID().X
	v := in[i]
	u := uint32(v)

	a := t.SubgroupAndI32(v)
	o := t.SubgroupOrI32(v)
	x := t.SubgroupXorI32(v)
	au := t.SubgroupAndU32(u)
	ou := t.SubgroupOrU32(u)
	xu := t.SubgroupXorU32(u)

	andI[i] = a
	orI[i] = o
	xorI[i] = x
	andU[i] = au
	orU[i] = ou
	xorU[i] = xu
}
