// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import "golang.design/x/accel"

// Histogram counts how many inputs fall in each of four buckets, using an
// atomic increment per element.
//
// It is the smallest kernel that is wrong without atomics: every invocation
// updates a bucket some other invocation may be updating, and the read-modify-
// write is exactly what an atomic makes indivisible. A version using ordinary
// `+= 1` computes a smaller number on any backend that runs invocations at
// once, and the right number on one that does not — which is the failure this
// corpus exists to make visible.
//
//accel:kernel workgroup=64
func Histogram(t accel.Thread, in []float32, counts []uint32) {
	i := t.GlobalID().X
	if i < uint32(len(in)) {
		v := in[i]
		bucket := uint32(0)
		if v >= 0.75 {
			bucket = 3
		} else if v >= 0.5 {
			bucket = 2
		} else if v >= 0.25 {
			bucket = 1
		}
		accel.AddU32(counts, bucket, 1)
	}
}

// AtomicOps exercises the rest of the integer operation set and stores each
// previous value, which is what every one of them returns.
//
// The previous value rather than the new one is the part worth a corpus kernel:
// after an add the new value is derivable and after a min it is not, so a
// caller that assumed otherwise would be right half the time.
//
//accel:kernel workgroup=1
func AtomicOps(t accel.Thread, state []uint32, prev []uint32) {
	prev[0] = accel.AddU32(state, 0, 7)
	prev[1] = accel.SubU32(state, 1, 3)
	prev[2] = accel.MinU32(state, 2, 5)
	prev[3] = accel.MaxU32(state, 3, 5)
	prev[4] = accel.AndU32(state, 4, 0x0F)
	prev[5] = accel.OrU32(state, 5, 0xF0)
	prev[6] = accel.XorU32(state, 6, 0xFF)
	prev[7] = accel.ExchangeU32(state, 7, 42)
	prev[8] = accel.CompareExchangeU32(state, 8, 1, 99) // matches, so it stores
	prev[9] = accel.CompareExchangeU32(state, 9, 1, 99) // does not match
}
