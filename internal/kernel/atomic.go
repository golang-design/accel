// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel

// Atomic operations.
//
// # Why free functions taking a buffer and an index
//
// Not a pointer into a buffer, because GLSL cannot form one:
// `atomicAdd(buf.data[i], v)` names the base and the index separately and there
// is no intermediate value to pass around. A Go API taking `*uint32` would be
// the pleasanter Go and would not lower. See specs/002-compute-model.md
// section 4.1.
//
// A shared array reaches them as `tile[:]`, which is the one place the subset
// admits a slice expression on a shared parameter. That keeps one name per
// operation rather than a parallel shared-memory family, and lowers exactly:
// the two GLSL forms differ only in the base.
//
// # Why every one returns the previous value
//
// It is what every target's instruction returns, and it is what a caller needs
// for the operations where the result is not recoverable from the arguments:
// after `AddU32` the new value is old+v, but after `MinU32` the old value is not
// derivable from the new one. Returning it uniformly means no caller has to
// remember which is which.
//
// # Why the CPU backend can implement them without synchronisation
//
// The workgroup scheduler advances one invocation at a time, so within a
// workgroup there is no concurrency for an atomic to protect against. That is
// not a shortcut that hides bugs: what an atomic protects against on a GPU is
// two lanes colliding, and the diagnostics in specs/019-cooperative-diagnostics.md
// catch a *missing* atomic by seeing two invocations touch one location
// unordered. The atomicity itself is free here; the checking is not.
//
// Between *workgroups* there is concurrency, because the CPU backend runs the
// grid on a worker pool. These functions are still plain reads and writes, and
// that is safe for one reason only: a kernel that reaches any of them is not
// order-independent, so its whole grid runs on one worker. See
// [Kernel.OrderIndependent], which is what the compiler infers and what
// [DispatchWith] gates on.
//
// Making them synchronised instead would be the wrong repair, and not only for
// the cost. Every one of them returns the value the location held before the
// operation, so a kernel that stores that return has a result that depends on
// which workgroup went first -- a real atomic would make the *counter* right
// and leave the *answer* unreproducible, which is the one thing this backend
// exists not to do.

// AddU32 adds v to b[i] and returns the previous value. It wraps modulo 2^32.
func AddU32(b []uint32, i uint32, v uint32) uint32 {
	old := b[i]
	b[i] = old + v
	return old
}

// AddI32 adds v to b[i] and returns the previous value. It wraps, two's
// complement.
func AddI32(b []int32, i uint32, v int32) int32 {
	old := b[i]
	b[i] = old + v
	return old
}

// SubU32 subtracts v from b[i] and returns the previous value.
func SubU32(b []uint32, i uint32, v uint32) uint32 {
	old := b[i]
	b[i] = old - v
	return old
}

// SubI32 subtracts v from b[i] and returns the previous value.
func SubI32(b []int32, i uint32, v int32) int32 {
	old := b[i]
	b[i] = old - v
	return old
}

// MinU32 stores the smaller of b[i] and v, comparing as uint32, and returns the
// previous value.
func MinU32(b []uint32, i uint32, v uint32) uint32 {
	old := b[i]
	if v < old {
		b[i] = v
	}
	return old
}

// MinI32 stores the smaller of b[i] and v, comparing as int32.
func MinI32(b []int32, i uint32, v int32) int32 {
	old := b[i]
	if v < old {
		b[i] = v
	}
	return old
}

// MaxU32 stores the larger of b[i] and v, comparing as uint32.
func MaxU32(b []uint32, i uint32, v uint32) uint32 {
	old := b[i]
	if v > old {
		b[i] = v
	}
	return old
}

// MaxI32 stores the larger of b[i] and v, comparing as int32.
func MaxI32(b []int32, i uint32, v int32) int32 {
	old := b[i]
	if v > old {
		b[i] = v
	}
	return old
}

// AndU32 stores b[i] & v and returns the previous value.
func AndU32(b []uint32, i uint32, v uint32) uint32 {
	old := b[i]
	b[i] = old & v
	return old
}

// OrU32 stores b[i] | v and returns the previous value.
func OrU32(b []uint32, i uint32, v uint32) uint32 {
	old := b[i]
	b[i] = old | v
	return old
}

// XorU32 stores b[i] ^ v and returns the previous value.
func XorU32(b []uint32, i uint32, v uint32) uint32 {
	old := b[i]
	b[i] = old ^ v
	return old
}

// ExchangeU32 stores v unconditionally and returns the previous value.
func ExchangeU32(b []uint32, i uint32, v uint32) uint32 {
	old := b[i]
	b[i] = v
	return old
}

// ExchangeI32 stores v unconditionally and returns the previous value.
func ExchangeI32(b []int32, i uint32, v int32) int32 {
	old := b[i]
	b[i] = v
	return old
}

// CompareExchangeU32 stores v if b[i] equals cmp, and returns the previous
// value either way. Success is `returned == cmp`.
//
// It is **strong**: it fails only when the observed value differs from cmp,
// never spuriously. Every target's compare-exchange is strong, so promising
// weak would be inventing a hazard for callers to loop around.
func CompareExchangeU32(b []uint32, i, cmp, v uint32) uint32 {
	old := b[i]
	if old == cmp {
		b[i] = v
	}
	return old
}

// CompareExchangeI32 is [CompareExchangeU32] for signed values, and equally
// strong.
func CompareExchangeI32(b []int32, i uint32, cmp, v int32) int32 {
	old := b[i]
	if old == cmp {
		b[i] = v
	}
	return old
}

// AddF32 adds v to b[i] and returns the previous value.
//
// It is a **capability**, not a baseline: several targets lack it, and a kernel
// using it is refused on a device that does. It also makes a reduction
// non-deterministic, because the hardware picks the accumulation order and f32
// addition is not associative — so a test asserting an exact total for a float
// reduction is wrong even where the same test is right for integers. See
// specs/002-compute-model.md section 4.5.
func AddF32(b []float32, i uint32, v float32) float32 {
	old := b[i]
	b[i] = old + v
	return old
}
