// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import "golang.design/x/accel"

// FloatReduce writes each subgroup's f32 minimum and maximum, per lane.
//
// specs/059-subgroup-reductions.md §5. It exists for the input [IntReduce]
// cannot carry: a NaN. kmath.Min and kmath.Max propagate one, the CPU
// scheduler's reductions fold them, and MSL's simd_min and simd_max are fmin
// and fmax, which drop a NaN in favour of the other lane. Without this kernel
// the f32 subgroup minimum and maximum were reachable from the public Thread
// and from no corpus kernel, so the two lowerings were never compared.
//
// Every lane writes, for [IntReduce]'s reason: the broadcast is part of the
// contract.
//
//accel:kernel workgroup=64
//accel:requires subgroup_arithmetic
func FloatReduce(t accel.Thread, in []float32, minF []float32, maxF []float32) {
	i := t.LocalID().X
	v := in[i]

	// One rendezvous per local, which the cooperative lowering requires.
	lo := t.SubgroupMinF32(v)
	hi := t.SubgroupMaxF32(v)

	minF[i] = lo
	maxF[i] = hi
}
