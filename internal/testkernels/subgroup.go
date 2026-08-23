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
	elected := t.Elect()
	if elected {
		sid := t.SubgroupID()
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
