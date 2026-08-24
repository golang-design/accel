// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import "golang.design/x/accel"

// SubgroupReduce sums each subgroup's values and has one lane per subgroup
// write the total.
//
// It is the kernel spec 009's subgroup sweep runs at sizes 1, 4, 32 and 64.
// Each size exercises something different: at 1 every operation degenerates to
// the identity and a kernel assuming it has neighbours breaks; at 4 a workgroup
// spans many subgroups so the boundary is crossed repeatedly; 32 and 64 are
// what NVIDIA, Apple and AMD actually have.
//
// The result is checked against [SubgroupReduceFallback], which computes the
// same thing with no subgroup operation at all — so a disagreement is the
// subgroup path's rather than the kernel's.
//
//accel:kernel workgroup=64
//accel:requires subgroup_arithmetic, subgroup_basic
func SubgroupReduce(t accel.Thread, in []float32, out []float32) {
	gid := t.GlobalID().X
	v := float32(0)
	if gid < uint32(len(in)) {
		v = in[gid]
	}

	total := t.SubgroupAddF32(v)

	// One lane per subgroup publishes, which is what Elect is for: hardware
	// guarantees only that exactly one lane is elected, and accel pins it to
	// the lowest so the output does not depend on the device.
	elected := t.SubgroupElect()
	if elected {
		sid := t.SubgroupIndex()
		if sid < uint32(len(out)) {
			out[sid] = total
		}
	}
}

// SubgroupReduceFallback is the same reduction with no subgroup operation,
// which is what a device without them runs.
//
// It exists to be the oracle for [SubgroupReduce]: spec 009 asks for the
// subgroup paths and their fallbacks to agree, and a fallback that shared any
// of the subgroup path's machinery could not show that.
//
//accel:kernel workgroup=64
func SubgroupReduceFallback(t accel.Thread, in []float32, out []float32, width []uint32) {
	lid := t.LocalID().X
	gid := t.GlobalID().X
	w := width[0]

	// The lane's own subgroup, computed the way spec 002 section 5.1 says the
	// oracle maps them, and summed by hand.
	sid := lid / w
	lane := lid % w
	if lane == 0 {
		acc := float32(0)
		for l := uint32(0); l < w; l++ {
			idx := gid - lane + l
			if idx < uint32(len(in)) {
				acc = acc + in[idx]
			}
		}
		if sid < uint32(len(out)) {
			out[sid] = acc
		}
	}
}

// SubgroupShuffleMix reads five lanes five ways and combines what it got.
//
// One kernel rather than five, because the differential's value is in what a
// swapped emit-table entry does to it: each read is scaled by a different power
// of two, so a shuffle emitted as a shuffle-up, or an up emitted as a down,
// changes the output of most lanes rather than of none.
//
// # Why every call is unconditional and every use is guarded
//
// specs/002-compute-model.md section 5.2 leaves a read of a lane outside the
// subgroup undefined, and the lanes at each end of a shuffle up or down always
// read one. Guarding the *call* is not available: a subgroup operation inside a
// conditional is refused by the cooperative lowering, which has no way to
// resume inside a branch. So the calls sit in uniform control flow where every
// lane reaches them, and the result of a read that went out of range is
// discarded rather than used. That is the shape a portable kernel has, which is
// why the corpus carries it.
//
// The reversal and the butterfly are the reason the subgroup size is read
// rather than assumed: a kernel that wrote 31 here would be wrong on the next
// device, which is what section 5.1 says the whole capability is about.
//
//accel:kernel workgroup=64
//accel:requires subgroup_basic, subgroup_shuffle
func SubgroupShuffleMix(t accel.Thread, in []float32, out []float32) {
	gid := t.GlobalID().X
	lane := t.SubgroupLane()
	w := t.SubgroupSize()

	v := float32(0)
	if gid < uint32(len(in)) {
		v = in[gid]
	}

	// Every lane of a full subgroup is active here, so the reversal and the
	// broadcast are defined for all of them.
	rev := t.SubgroupShuffleF32(v, w-1-lane)
	head := t.SubgroupBroadcastF32(v, 0)
	partner := t.SubgroupShuffleXorF32(v, 1)
	up := t.SubgroupShuffleUpF32(v, 1)
	down := t.SubgroupShuffleDownF32(v, 1)

	r := rev + 4*head
	// The butterfly partner is out of range for the top lane of a subgroup
	// whose width is not a power of two, and for the only lane of a subgroup of
	// one.
	if lane^1 < w {
		r = r + 2*partner
	}
	if lane > 0 {
		r = r + 8*up
	}
	if lane+1 < w {
		r = r + 16*down
	}
	if gid < uint32(len(out)) {
		out[gid] = r
	}
}

// SubgroupShuffleMixFallback computes the same function with no subgroup
// operation, reading each neighbour out of the buffer instead.
//
// It is the oracle for [SubgroupShuffleMix] in the same sense that
// [SubgroupReduceFallback] is for [SubgroupReduce]: the two share no machinery,
// so an agreement between them is evidence about the shuffles rather than about
// one implementation compared with itself. The width arrives as a binding
// because a kernel with no subgroups has none to ask.
//
//accel:kernel workgroup=64
func SubgroupShuffleMixFallback(t accel.Thread, in []float32, out []float32, width []uint32) {
	gid := t.GlobalID().X
	lid := t.LocalID().X
	w := width[0]
	lane := lid % w
	// The first invocation of this lane's subgroup, under the mapping spec 002
	// section 5.1 says the oracle uses.
	base := gid - lane
	n := uint32(len(in))

	revIdx := base + w - 1 - lane
	rev := float32(0)
	if revIdx < n {
		rev = in[revIdx]
	}
	head := float32(0)
	if base < n {
		head = in[base]
	}

	r := rev + 4*head

	partner := lane ^ 1
	if partner < w && base+partner < n {
		r = r + 2*in[base+partner]
	}
	if lane > 0 && base+lane-1 < n {
		r = r + 8*in[base+lane-1]
	}
	if lane+1 < w && base+lane+1 < n {
		r = r + 16*in[base+lane+1]
	}
	if gid < uint32(len(out)) {
		out[gid] = r
	}
}

// SubgroupScan writes both prefix sums of its subgroup: the inclusive one, and
// the exclusive one beside it.
//
// Both, in one kernel, because the pair is where the interesting rule lives.
// The two differ only at the lane itself, so a lowering that emitted one for
// the other produces a result that is right for lane 0 of an exclusive scan and
// wrong everywhere else — visible here as two buffers that disagree in a way no
// single-scan kernel would show.
//
//accel:kernel workgroup=64
//accel:requires subgroup_arithmetic
func SubgroupScan(t accel.Thread, in []float32, incl []float32, excl []float32) {
	gid := t.GlobalID().X

	v := float32(0)
	if gid < uint32(len(in)) {
		v = in[gid]
	}

	inclusive := t.SubgroupInclusiveAddF32(v)
	exclusive := t.SubgroupExclusiveAddF32(v)

	if gid < uint32(len(incl)) {
		incl[gid] = inclusive
	}
	if gid < uint32(len(excl)) {
		excl[gid] = exclusive
	}
}

// SubgroupScanFallback computes the same two prefix sums with no subgroup
// operation, by summing the buffer.
//
// # It sums in the order the scan is specified to
//
// Ascending lane, with the first term taken rather than added to a zero
// accumulator. f32 addition is not associative, so a fallback that summed in
// any other order would disagree in the last bit and the disagreement would be
// the test's rather than the kernel's — and taking the first term rather than
// adding it is what keeps a negative zero negative.
//
//accel:kernel workgroup=64
func SubgroupScanFallback(t accel.Thread, in []float32, incl []float32, excl []float32, width []uint32) {
	gid := t.GlobalID().X
	lid := t.LocalID().X
	w := width[0]
	lane := lid % w
	base := gid - lane
	n := uint32(len(in))

	inclusive := float32(0)
	for l := uint32(0); l <= lane; l++ {
		x := float32(0)
		if base+l < n {
			x = in[base+l]
		}
		if l == 0 {
			inclusive = x
		} else {
			inclusive = inclusive + x
		}
	}

	// The lowest lane's exclusive prefix is empty, and an empty sum is +0.
	exclusive := float32(0)
	for l := uint32(0); l < lane; l++ {
		x := float32(0)
		if base+l < n {
			x = in[base+l]
		}
		if l == 0 {
			exclusive = x
		} else {
			exclusive = exclusive + x
		}
	}

	if gid < uint32(len(incl)) {
		incl[gid] = inclusive
	}
	if gid < uint32(len(excl)) {
		excl[gid] = exclusive
	}
}
