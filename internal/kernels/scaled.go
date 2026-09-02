// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import "golang.design/x/accel"

// Params is a uniform block.
//
// It is deliberately the shape spec 001 section 3.3 uses as its worked example,
// because the offsets are not what a reader guesses: Steps lands inside the
// sixteen-byte slot Origin's three components only half fill. A caller writes
// this struct with no padding field and never spells an offset.
type Params struct {
	Scale   float32
	Origin  [3]float32
	Steps   uint32
	Inverse [4][4]float32
}

// Transform applies a uniform's scale and offset to every element.
//
// The loop bound comes from the uniform rather than from len, which is the
// reason spec 001 calls the uniform path required rather than convenient: a
// storage-buffer substitute would make the bound appear non-uniform to the
// barrier analysis at M4.
//
//accel:kernel workgroup=32
func Transform(t accel.Thread, p Params, in []float32, out []float32) {
	i := t.GlobalID().X
	if i >= p.Steps || i >= uint32(len(out)) {
		return
	}

	acc := in[i]*p.Scale + p.Origin[0]
	for c := uint32(0); c < 4; c++ {
		acc += p.Inverse[c][0] * p.Origin[1]
	}
	out[i] = acc + p.Origin[2]
}
