// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import "golang.design/x/accel"

// ShapeDims tells the kernel what the host thinks the dispatch is.
//
// It exists so the test can compare the two: a kernel that read its shape from
// the uniform would agree with itself, and what specs/052-dispatch-shape.md
// claims is that the *thread's* answer is the recorded one.
type ShapeDims struct {
	// Stride is how many u32 slots one axis's triple occupies in out, which is
	// 3 so a transposed lowering lands in a neighbouring slot rather than
	// overwriting its own.
	Stride uint32
}

// DispatchShape writes each accessor's three axes to out.
//
// specs/052-dispatch-shape.md §3. One invocation writes, because every value
// here is uniform across the dispatch and nine writes racing on the same nine
// slots would say nothing about which invocation produced them. The guard is
// spelled on GlobalID's three axes rather than GlobalIndex, which is outside
// the MSL subset -- and it has to be, since the point is to compare both
// backends.
//
// The layout is WorkgroupSize, NumGroups, GlobalSize, each x-then-y-then-z. A
// lowering that transposed an axis writes a different number into a slot the
// test names, rather than a number the test would have to infer.
//
//accel:kernel workgroup=4,2,1
func DispatchShape(t accel.Thread, d ShapeDims, out []uint32) {
	g := t.GlobalID()
	if g.X != 0 || g.Y != 0 || g.Z != 0 {
		return
	}
	ws := t.WorkgroupSize()
	ng := t.NumGroups()
	gs := t.GlobalSize()

	out[0*d.Stride+0] = ws.X
	out[0*d.Stride+1] = ws.Y
	out[0*d.Stride+2] = ws.Z

	out[1*d.Stride+0] = ng.X
	out[1*d.Stride+1] = ng.Y
	out[1*d.Stride+2] = ng.Z

	out[2*d.Stride+0] = gs.X
	out[2*d.Stride+1] = gs.Y
	out[2*d.Stride+2] = gs.Z
}

// ShapeBoundedSum sums a workgroup's slice of in, with the workgroup extent as
// the loop bound and a barrier inside the loop.
//
// This is §3's third assertion and the reason the accessor is worth having over
// a uniform field. A barrier must be reached by every invocation of a
// workgroup, so a loop containing one has to have a bound the compiler can see
// is the same for all of them. `t.WorkgroupSize().X` is a compile-time constant
// -- on Metal it lowers to a literal -- and a uniform field is not, so a kernel
// spelled this way compiles and the same kernel with `d.Width` does not.
//
//accel:kernel workgroup=8
func ShapeBoundedSum(t accel.Thread, in []float32, out []float32, sh *[8]float32) {
	lane := t.LocalID().X
	base := t.GroupID().X * t.WorkgroupSize().X

	total := float32(0)
	// Bounded by the workgroup extent, with a barrier in the body.
	for i := uint32(0); i < t.WorkgroupSize().X; i++ {
		sh[lane] = in[base+lane]
		t.Barrier()
		total += sh[i]
		t.Barrier()
	}
	out[base+lane] = total
}
