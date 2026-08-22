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

// KernelSlice recovers one bound argument for a generated entry point.
//
// It is a function rather than an alias because Go has no function aliases. The
// bound argument set has already been checked by [Kernel.Bind], so a mismatch
// here is a generator bug and not a caller's, and it panics rather than
// returning an error a generated caller would have to handle at every binding.
func KernelSlice[T any](a KernelArgs, i int) []T { return kernel.Slice[T](a, i) }
