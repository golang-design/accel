// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import "golang.design/x/accel/internal/kernel"

// The generated-kernel ABI.
//
// Generated code lives in the package whose kernels it was generated from,
// which is the caller's package, so every type it names has to be reachable
// from outside this module. That is why these are here rather than only in
// internal/kernel: an internal path is importable by this module alone, and a
// generated file that named one would compile here and nowhere else.
//
// They are aliases rather than wrappers so that the record a generated file
// builds and the record a backend consumes are one type. See
// specs/012-kernel-pipeline.md.
type (
	// KernelBinding is one resource a kernel's signature declares, with the
	// access inferred from its body.
	KernelBinding = kernel.Binding

	// KernelArgs carries the host slices a generated kernel runs over.
	KernelArgs = kernel.Args

	// KernelDType is a binding's element type as generated code names it.
	KernelDType = kernel.DType

	// KernelAccess is how a kernel touches a binding.
	KernelAccess = kernel.Access

	// KernelUniform is one by-value parameter a kernel declares.
	KernelUniform = kernel.Uniform

	// KernelSharedTracker records what a workgroup did to its shared memory, so
	// a read of something nothing wrote is reported rather than plausible. It
	// is nil in strict mode, where every call on it is a no-op.
	KernelSharedTracker = kernel.SharedTracker

	// KernelSubgroupOp identifies which subgroup rendezvous a lane is suspended
	// at. A subgroup operation needs every lane's value at the point of the
	// call, so it suspends exactly as a barrier does.
	KernelSubgroupOp = kernel.SubgroupOp

	// KernelMask is a subgroup ballot, one bit per lane.
	KernelMask = kernel.Mask

	// KernelBarrierID identifies one suspension point, with the source position
	// a report needs to name the line rather than only the index.
	KernelBarrierID = kernel.BarrierID

	// KernelFrame is one invocation's saved state between suspension points in
	// a cooperative kernel. The scheduler owns it and the generated lowering
	// decides what it holds.
	KernelFrame = kernel.Frame
)

// Binding element types, as generated code names them. They mirror the public
// [DType] values and are checked against them by test.
const (
	KernelF32  = kernel.F32
	KernelF16  = kernel.F16
	KernelBF16 = kernel.BF16
	KernelI32  = kernel.I32
	KernelU32  = kernel.U32
	KernelI8   = kernel.I8
	KernelU8   = kernel.U8
)

// Inferred accesses, as generated code names them.
const (
	KernelRead  = kernel.Read
	KernelWrite = kernel.Write
)

// KernelShared recovers one workgroup-shared array from an argument set, as a
// pointer, because a workgroup shares one copy.
func KernelShared[T any](a KernelArgs, i int) *T { return kernel.SharedSlice[T](a, i) }

// KernelPoison fills workgroup-shared storage with a pattern no sensible
// kernel computes, so a read before a write is loud rather than plausible.
func KernelPoison[T any](s []T) { kernel.Poison(s) }

// Atomic operations, as a kernel author writes them.
//
// Free functions taking a buffer and an index rather than a pointer into one,
// because GLSL cannot form such a pointer: see specs/002-compute-model.md
// section 4.1. Each returns the previous value, which is what every target's
// instruction returns and what a caller needs for the operations whose old
// value is not recoverable from the new one.
//
// A shared array is passed as tile[:], the one place the subset admits a slice
// expression on a shared parameter.
//
// They are functions rather than variables bound to the internal ones, because
// the kernel compiler resolves an intrinsic by the object go/types produced: a
// variable is not a func, and the table would not find it.
func AddU32(b []uint32, i uint32, v uint32) uint32 { return kernel.AddU32(b, i, v) }
func AddI32(b []int32, i uint32, v int32) int32    { return kernel.AddI32(b, i, v) }
func SubU32(b []uint32, i uint32, v uint32) uint32 { return kernel.SubU32(b, i, v) }
func SubI32(b []int32, i uint32, v int32) int32    { return kernel.SubI32(b, i, v) }
func MinU32(b []uint32, i uint32, v uint32) uint32 { return kernel.MinU32(b, i, v) }
func MinI32(b []int32, i uint32, v int32) int32    { return kernel.MinI32(b, i, v) }
func MaxU32(b []uint32, i uint32, v uint32) uint32 { return kernel.MaxU32(b, i, v) }
func MaxI32(b []int32, i uint32, v int32) int32    { return kernel.MaxI32(b, i, v) }
func AndU32(b []uint32, i uint32, v uint32) uint32 { return kernel.AndU32(b, i, v) }
func OrU32(b []uint32, i uint32, v uint32) uint32  { return kernel.OrU32(b, i, v) }
func XorU32(b []uint32, i uint32, v uint32) uint32 { return kernel.XorU32(b, i, v) }

// ExchangeU32 and ExchangeI32 store unconditionally.
func ExchangeU32(b []uint32, i uint32, v uint32) uint32 { return kernel.ExchangeU32(b, i, v) }
func ExchangeI32(b []int32, i uint32, v int32) int32    { return kernel.ExchangeI32(b, i, v) }

// CompareExchangeU32 and CompareExchangeI32 are **strong**: they fail only when
// the observed value differs from cmp, never spuriously. Every target's
// compare-exchange is strong, so promising weak would invent a hazard for
// callers to loop around. Success is `returned == cmp`.
func CompareExchangeU32(b []uint32, i, cmp, v uint32) uint32 {
	return kernel.CompareExchangeU32(b, i, cmp, v)
}

func CompareExchangeI32(b []int32, i uint32, cmp, v int32) int32 {
	return kernel.CompareExchangeI32(b, i, cmp, v)
}

// AddF32 is a capability rather than a baseline, and it makes a reduction
// non-deterministic: the hardware picks the accumulation order and f32 addition
// is not associative, so a test asserting an exact total for a float reduction
// is wrong even where the same test is right for integers. See
// [CapAtomicFloatAddStorage].
func AddF32(b []float32, i uint32, v float32) float32 { return kernel.AddF32(b, i, v) }

// The subgroup rendezvous a generated lowering names.
const (
	KernelSubNone              = kernel.SubNone
	KernelSubAddF32            = kernel.SubAddF32
	KernelSubMinF32            = kernel.SubMinF32
	KernelSubMaxF32            = kernel.SubMaxF32
	KernelSubBroadcastFirstF32 = kernel.SubBroadcastFirstF32
	KernelSubElect             = kernel.SubElect
	KernelSubAny               = kernel.SubAny
	KernelSubAll               = kernel.SubAll
	KernelSubBallot            = kernel.SubBallot
)

// KernelABIVersion is the contract between a generated kernel and this runtime.
// A generated file records it, and a mismatch refuses to run rather than
// running differently.
const KernelABIVersion = kernel.ABIVersion

// Narrow storage types.
//
// A kernel parameter of `[]accel.Float16` is a binding of 16-bit floats. The
// types carry no arithmetic operators at all, which is not an omission: it is
// what forces `f.F32()` on the way in and [ToFloat16] on the way out, so f32
// accumulation is the only thing that compiles rather than a convention an
// author has to remember. Native narrow arithmetic is a separate capability
// (spec 002's CapF16Arithmetic) and arrives as explicit intrinsics later.
//
// They are not named F16 and BF16 because those are the [DType] constants for
// the same formats. A dtype is metadata a descriptor carries and a storage type
// is what a parameter is made of; both wanted the short name and the constants
// shipped first.
type (
	Float16  = kernel.Float16
	BFloat16 = kernel.BFloat16
)

// ToFloat16 converts an f32 to 16-bit storage, rounding to nearest with ties to
// even and overflowing to a signed infinity. A NaN becomes the canonical quiet
// encoding, per specs/008-numerics.md section 4.
func ToFloat16(x float32) Float16 { return kernel.ToFloat16(x) }

// ToBFloat16 converts an f32 to bf16 storage, rounding to nearest with ties to
// even. Truncating the low bits instead would be round-toward-zero, which
// biases every value in one direction and compounds over a reduction.
func ToBFloat16(x float32) BFloat16 { return kernel.ToBFloat16(x) }

// Float16FromBits and BFloat16FromBits reinterpret storage bits without
// converting, for a caller who already has encoded data.
func Float16FromBits(b uint16) Float16   { return kernel.Float16FromBits(b) }
func BFloat16FromBits(b uint16) BFloat16 { return kernel.BFloat16FromBits(b) }

// KernelSlice recovers one bound argument for a generated entry point.
//
// It is a function rather than an alias because Go has no function aliases. The
// bound argument set has already been checked by [Kernel.Bind], so a mismatch
// here is a generator bug and not a caller's, and it panics rather than
// returning an error a generated caller would have to handle at every binding.
func KernelSlice[T any](a KernelArgs, i int) []T { return kernel.Slice[T](a, i) }

// KernelUniformValue recovers one by-value parameter for a generated entry
// point. Like [KernelSlice] it panics rather than returning an error, because
// [Kernel.Bind] has already checked the argument set's shape and a generated
// caller would be handling an impossible error at every parameter.
func KernelUniformValue[T any](a KernelArgs, i int) T { return kernel.UniformValue[T](a, i) }
