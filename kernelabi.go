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
