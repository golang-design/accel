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

// CountWorkgroups increments a counter once per workgroup, which makes the
// number of workgroups that actually ran observable.
//
// It exists because most kernels hide their dispatch size behind a bounds
// check: run twice as many workgroups over the same buffer and the extra ones
// write nothing, so the count cannot be told from the output. Spec 003 requires
// an indirect count to be clamped in every build mode, and a test of that needs
// a kernel whose output says how many workgroups there were.
//
//accel:kernel workgroup=64
func CountWorkgroups(t accel.Thread, counts []uint32) {
	if t.LocalID().X == 0 {
		accel.AddU32(counts, 0, 1)
	}
}

// AtomicOpsI32 exercises the signed integer atomics, which nothing did.
//
// [AtomicOps] covers the unsigned set and every one of those is reached by a
// corpus kernel. None of the signed ones was: `accel.AddI32`, `SubI32`,
// `MinI32`, `MaxI32`, `ExchangeI32` and `CompareExchangeI32` were exported for
// kernel authors, lowered by an emitter path nobody ran, and spelled in MSL by
// a mapping nobody compiled. Found by reading coverage on the public surface —
// all six sat at 0.0%.
//
// # The operands are negative, and only two of the six can tell
//
// specs/010-kernel-corpus.md records the rule this kernel was built to satisfy:
// a test for a representation must use an operation the representation changes.
// Add, sub, exchange and compare-exchange are bit-identical whether the operand
// is read signed or unsigned — two's complement makes them the same machine
// operation — so they prove the lowering exists and say nothing about its sign.
//
// **Min and max are the two that differ.** `MinI32(-5, 3)` is -5; read
// unsigned, -5 is 4294967291 and the minimum is 3. So the negative operands
// below are what make this kernel a check on signedness rather than only on
// reachability, and the test asserts those two hardest.
//
//accel:kernel workgroup=1
func AtomicOpsI32(t accel.Thread, state []int32, prev []int32) {
	prev[0] = accel.AddI32(state, 0, -7)
	prev[1] = accel.SubI32(state, 1, -3)
	// The discriminating pair: state[2] and state[3] hold negative values and
	// the operand is positive, so a signed comparison keeps the negative one
	// and an unsigned comparison keeps the other.
	prev[2] = accel.MinI32(state, 2, 3)
	prev[3] = accel.MaxI32(state, 3, 3)
	prev[4] = accel.ExchangeI32(state, 4, -42)
	prev[5] = accel.CompareExchangeI32(state, 5, -1, -99) // matches, so it stores
	prev[6] = accel.CompareExchangeI32(state, 6, -1, -99) // does not match
}
