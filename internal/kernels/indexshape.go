// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import "golang.design/x/accel"

// IndexShape writes each invocation's three flat indices, at its global index.
//
// specs/002-compute-model.md §1.3's GlobalIndex, LocalIndex and GroupIndex
// are the x-fastest linearizations of the three ids, and they had a CPU
// definition and no MSL lowering: every corpus kernel that needed a flat index
// spelled it from GlobalID().X, so the three accessors were public API that
// could not run on Metal. The workgroup is three-dimensional and the grid is
// dispatched over three axes so a linearization with the axes in the wrong
// order, or with the workgroup extent for the group count, lands each value in
// a different slot.
//
//accel:kernel workgroup=4,2,2
func IndexShape(t accel.Thread, out []uint32) {
	g := t.GlobalIndex()
	i := 3 * g
	if i+2 < uint32(len(out)) {
		out[i] = g
		out[i+1] = t.LocalIndex()
		out[i+2] = t.GroupIndex()
	}
}
