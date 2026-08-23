// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

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
