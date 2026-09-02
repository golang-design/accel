// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import "golang.design/x/accel"

// Conversions between the storage formats, as kernels.
//
// # Why a kernel and not a host loop
//
// A value produced on the device and needed in another format has to be
// converted somewhere. On the host means a readback, a loop, and an upload,
// which is three synchronisation points inside what should be one graph. So the
// conversion is a kernel, and the tensor layer's Cast operator lowers to it.
//
// # Why these are exact in one direction and not the other
//
// f16 to f32 is exact: every f16 value is an f32 value, so the conversion adds
// no error and specs/008-numerics.md admits a bit-for-bit comparison. f32 to
// f16 rounds, and rounds to nearest-even, which is the only rounding this
// project's compute model admits (008 section 4). A value outside f16's range
// becomes an infinity rather than a saturated maximum, because a silently
// clamped weight is a plausible weight.

// CastF32ToF16 narrows f32 storage to f16, rounding to nearest-even.
//
//accel:kernel workgroup=64
func CastF32ToF16(t accel.Thread, in []float32, out []accel.Float16) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		out[i] = accel.ToFloat16(in[i])
	}
}

// CastF16ToF32 widens f16 storage to f32, exactly.
//
//accel:kernel workgroup=64
func CastF16ToF32(t accel.Thread, in []accel.Float16, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		out[i] = in[i].F32()
	}
}

// CastBF16ToF32 widens bf16 storage to f32, exactly.
//
// # Why bf16 only widens
//
// A checkpoint ships bf16 -- Qwen3 does -- and nothing in the corpus reads it,
// so a loader had to convert on the host. Converting to f32 is a 16-bit shift
// and loses nothing, because bf16 is f32's top half: the same eight-bit
// exponent, a seven-bit mantissa, and the low sixteen bits zero. Converting to
// f16 instead is the one lossy step in this pipeline, since f16 carries a
// five-bit exponent and a bf16 value can be outside its range entirely.
//
// So the widening is registered and the narrowing is not. A caller who wants
// f16 writes the two casts and sees where the error enters, which is the same
// argument specs/007-tensor-layer.md makes for Cast existing at all.
//
//accel:kernel workgroup=64
func CastBF16ToF32(t accel.Thread, in []accel.BFloat16, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		out[i] = in[i].F32()
	}
}
