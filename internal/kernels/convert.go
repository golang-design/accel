// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import (
	"golang.design/x/accel"
	"golang.design/x/accel/kmath"
)

// SaturatingConvert converts each input to an integer both ways.
//
// specs/051-float-to-int.md §2.1. The spec's §3 claims CPU and Metal agree bit
// for bit across the boundary set, and until this kernel existed nothing
// asserted it: the boundaries were checked on the CPU alone, and the two
// backends met only through three graphics stages that pass in-range
// coordinates.
//
// # Why a kernel rather than a wider unit test
//
// The two lowerings are written separately -- kmath.ToI32 in Go, _accel_to_i32
// in the MSL prelude -- and mirror each other line for line by hand. That is an
// argument, not a test. A divergence would be a compare that reads `<` where the
// other reads `<=`, or an integer literal that rounds differently as a float
// constant, and neither shows up anywhere except in the two answers.
//
// Both destinations in one kernel because they share the input: a seed chosen to
// be interesting for the signed conversion is interesting for the unsigned one
// at the same index, and comparing them side by side is what makes a sign error
// visible rather than merely a wrong number.
//
//accel:kernel workgroup=64
func SaturatingConvert(t accel.Thread, in []float32, outI []int32, outU []uint32) {
	i := t.GlobalID().X
	if i < uint32(len(in)) {
		outI[i] = kmath.ToI32(in[i])
		outU[i] = kmath.ToU32(in[i])
	}
}
