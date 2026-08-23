// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package kernelabi is the contract between a generated kernel and the runtime
// that executes it.
//
// # You almost certainly do not need this
//
// Nothing here is written by hand. `cmd/accel-kernel` emits the code that names
// these, into `accel_kernels.go` in your own package, and the only thing you do
// with the result is take its address:
//
//	dev.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: &kernels.ScaleKernel})
//
// What you *do* write — [accel.Thread], the atomics, `accel.Float16` and the
// rest of the kernel language — lives in the root package, where a kernel
// author can find it.
//
// # Why it is a separate package
//
// These thirty-odd names used to be in the root package, prefixed `Kernel` to
// keep them apart, where they outnumbered the ones a caller actually uses and
// filled the pkg.go.dev index a newcomer reads first. Generated code has to
// name them from outside the module, so unexporting was not available; moving
// them was. See specs/036-documentation.md's freeze record, section 5.3.
//
// The types are aliases rather than wrappers, so the record a generated file
// builds and the record a backend consumes are one type. See
// specs/012-kernel-pipeline.md.
package kernelabi

import (
	"fmt"

	"golang.design/x/accel/internal/kernel"
)

// Kernel is one compiled kernel: its entry points, its bindings, and the
// metadata a backend needs to run it.
//
// A generated file declares one of these per kernel as a package-level
// variable. A caller never constructs one and never reads its fields; the
// address is the whole of the public interface.
type Kernel = kernel.Kernel

type (
	// Binding is one resource a kernel's signature declares, with the access
	// inferred from its body.
	Binding = kernel.Binding

	// Args carries the host slices a generated kernel runs over.
	Args = kernel.Args

	// DType is a binding's element type as generated code names it.
	DType = kernel.DType

	// Access is how a kernel touches a binding.
	Access = kernel.Access

	// Uniform is one by-value parameter a kernel declares.
	Uniform = kernel.Uniform

	// SharedTracker records what a workgroup did to its shared memory, so a
	// read of something nothing wrote is reported rather than plausible. It is
	// nil in strict mode, where every call on it is a no-op.
	SharedTracker = kernel.SharedTracker

	// SubgroupOp identifies which subgroup rendezvous a lane is suspended at. A
	// subgroup operation needs every lane's value at the point of the call, so
	// it suspends exactly as a barrier does.
	SubgroupOp = kernel.SubgroupOp

	// Mask is a subgroup ballot, one bit per lane.
	Mask = kernel.Mask

	// BarrierID identifies one suspension point, with the source position a
	// report needs to name the line rather than only the index.
	BarrierID = kernel.BarrierID

	// Frame is one invocation's saved state between suspension points in a
	// cooperative kernel. The scheduler owns it and the generated lowering
	// decides what it holds.
	Frame = kernel.Frame

	// ID3 is a three-component index.
	ID3 = kernel.ID3
)

// Binding element types, as generated code names them. They mirror the public
// accel.DType values and are checked against them by test.
const (
	F32  = kernel.F32
	F16  = kernel.F16
	BF16 = kernel.BF16
	I32  = kernel.I32
	U32  = kernel.U32
	I8   = kernel.I8
	U8   = kernel.U8
)

// Inferred accesses, as generated code names them.
const (
	Read  = kernel.Read
	Write = kernel.Write
)

// The subgroup rendezvous a generated lowering names.
const (
	SubNone              = kernel.SubNone
	SubAddF32            = kernel.SubAddF32
	SubMinF32            = kernel.SubMinF32
	SubMaxF32            = kernel.SubMaxF32
	SubBroadcastFirstF32 = kernel.SubBroadcastFirstF32
	SubElect             = kernel.SubElect
	SubAny               = kernel.SubAny
	SubAll               = kernel.SubAll
	SubBallot            = kernel.SubBallot
)

// Version is the contract between a generated kernel and this runtime. A
// generated file records it, and a mismatch refuses to run rather than running
// differently.
const Version = kernel.ABIVersion

// Shared recovers one workgroup-shared array from an argument set, as a
// pointer, because a workgroup shares one copy.
func Shared[T any](a Args, i int) *T { return kernel.SharedSlice[T](a, i) }

// Poison fills workgroup-shared storage with a pattern no sensible kernel
// computes, so a read before a write is loud rather than plausible.
func Poison[T any](s []T) { kernel.Poison(s) }

// Slice recovers one bound argument for a generated entry point.
//
// It is a function rather than an alias because Go has no function aliases. The
// bound argument set has already been checked, so a mismatch here is a
// generator bug and not a caller's, and it panics rather than returning an
// error a generated caller would have to handle at every binding.
func Slice[T any](a Args, i int) []T { return kernel.Slice[T](a, i) }

// UniformValue recovers one by-value parameter for a generated entry point.
// Like [Slice] it panics rather than returning an error, because the argument
// set's shape has already been checked and a generated caller would be handling
// an impossible error at every parameter.
func UniformValue[T any](a Args, i int) T { return kernel.UniformValue[T](a, i) }

// EncodeUniform is the generated bridge from an untyped uniform value to its
// std140 codec.
//
// It exists so a generated [Uniform.Encode] is one line that names the codec,
// rather than a type switch repeated per uniform type. The type assertion is
// checked and reported: a backend passing the wrong value gets an error naming
// both types, where a bare assertion would panic inside a driver with no way
// for the caller to see which uniform was wrong.
func EncodeUniform[T any](dst []byte, v any, encode func([]byte, T) error) error {
	t, ok := v.(T)
	if !ok {
		return fmt.Errorf("accel: uniform is %T, and its codec takes %T", v, t)
	}
	return encode(dst, t)
}
