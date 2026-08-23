// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import "golang.design/x/accel/internal/kernel"

// The kernel language a caller writes: atomics and the narrow storage types.
//
// What *generated* code names is not here. It moved to
// golang.design/x/accel/kernelabi, where it does not fill the index a newcomer
// reads — about thirty names that outnumbered the ones a caller actually uses.
// See specs/036-documentation.md's freeze record, section 5.3.

// Atomic operations, as a kernel author writes them.
//
// Every one returns the value the location held *before* the operation, which
// is what makes an atomic add usable as a ticket dispenser rather than only as
// a counter. See specs/020-cooperative-atomics.md.
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

// ExchangeU32 and ExchangeI32 store a value and return the previous one.
func ExchangeU32(b []uint32, i uint32, v uint32) uint32 { return kernel.ExchangeU32(b, i, v) }
func ExchangeI32(b []int32, i uint32, v int32) int32    { return kernel.ExchangeI32(b, i, v) }

// CompareExchangeU32 stores v only if the location holds cmp, and returns what
// it held either way — so a caller learns whether the swap happened by
// comparing the result with cmp rather than by a second read that could race.
func CompareExchangeU32(b []uint32, i, cmp, v uint32) uint32 {
	return kernel.CompareExchangeU32(b, i, cmp, v)
}

func CompareExchangeI32(b []int32, i uint32, cmp, v int32) int32 {
	return kernel.CompareExchangeI32(b, i, cmp, v)
}

// AddF32 is a capability, not a guarantee: see [CapAtomicFloatAddStorage]. A
// device without it refuses the kernel at pipeline creation rather than
// producing a wrong sum.
func AddF32(b []float32, i uint32, v float32) float32 { return kernel.AddF32(b, i, v) }

// Narrow storage types.
//
// A kernel parameter of []accel.Float16 is a binding of 16-bit floats. The
// types carry no arithmetic operators at all, which is not an omission: it is
// what forces f.F32() on the way in and [ToFloat16] on the way out, so f32
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
