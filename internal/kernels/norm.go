// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import (
	"golang.design/x/accel"
	"golang.design/x/accel/kmath"
)

// RowDims is a row-wise kernel's shape: how many rows and how wide.
type RowDims struct {
	Rows  uint32
	Width uint32

	// Eps is added under the square root, which is what keeps a row of zeros
	// from dividing by zero. It is a parameter rather than a constant because
	// models ship with different values and a mismatch changes the output in a
	// way no test of the kernel alone would catch.
	Eps float32
}

// RowWidth is the workgroup width the row-wise kernels use.
const RowWidth = 128

// RMSNorm scales each row by the reciprocal of its root-mean-square.
//
//	y = x / sqrt(mean(x²) + ε) · w
//
// # Why the sum of squares is f32 whatever the storage is
//
// A row is hundreds or thousands of elements and the sum of their squares is
// what the reciprocal is taken of, so an f16 accumulator would put every partial
// through a ten-bit mantissa before the value that matters is computed. Spec 010
// says f32 sum of squares and rsqrt; the narrow storage types carry no
// arithmetic operators, so Go forces the widening rather than leaving it to be
// remembered.
//
// One workgroup per row, with each invocation folding a strided slice and the
// tree reducing the partials — the same shape as [ReduceSum], and for the same
// reason: a tree of depth seven against a sequential sum of a hundred and
// twenty-seven.
//
//accel:kernel workgroup=128
func RMSNorm(t accel.Thread, d RowDims, x []float32, w []float32, out []float32,
	sh *[128]float32) {

	row := t.GroupID().X
	lid := t.LocalID().X
	base := row * d.Width

	// Each invocation's partial sum of squares over its strided slice.
	acc := float32(0)
	for i := lid; i < d.Width; i += RowWidth {
		v := x[base+i]
		acc = acc + v*v
	}
	sh[lid] = acc
	t.Barrier()

	for stride := uint32(RowWidth / 2); stride > 0; stride /= 2 {
		if lid < stride {
			sh[lid] = sh[lid] + sh[lid+stride]
		}
		t.Barrier()
	}

	// Every invocation reads the total rather than one computing and
	// broadcasting it: the read is free after the barrier and a broadcast would
	// be another rendezvous.
	mean := sh[0] / float32(d.Width)
	scale := kmath.RSqrt(mean + d.Eps)

	for i := lid; i < d.Width; i += RowWidth {
		out[base+i] = x[base+i] * scale * w[i]
	}
}

// Softmax normalizes each row into a distribution.
//
//	y = exp(x - max(x)) / Σ exp(x - max(x))
//
// # Why the maximum is subtracted
//
// exp overflows f32 at about 88, and a logit of 100 is ordinary. Subtracting the
// row's maximum makes the largest exponent zero and every other negative, so no
// term overflows and the result is unchanged — exp(a-m)/Σexp(b-m) is
// algebraically exp(a)/Σexp(b). This is the whole reason a softmax kernel is not
// two lines, and skipping it produces Inf/Inf, which is NaN, on real inputs.
//
// Two reductions and therefore two tree passes: the maximum, then the sum.
//
//accel:kernel workgroup=128
func Softmax(t accel.Thread, d RowDims, x []float32, out []float32, sh *[128]float32) {
	row := t.GroupID().X
	lid := t.LocalID().X
	base := row * d.Width

	// The maximum. Seeded from the invocation's first element rather than from
	// a sentinel, so a row of large negative values does not come back as the
	// sentinel.
	m := x[base+lid%d.Width]
	for i := lid; i < d.Width; i += RowWidth {
		v := x[base+i]
		m = kmath.Max(m, v)
	}
	sh[lid] = m
	t.Barrier()
	for stride := uint32(RowWidth / 2); stride > 0; stride /= 2 {
		if lid < stride {
			sh[lid] = kmath.Max(sh[lid], sh[lid+stride])
		}
		t.Barrier()
	}
	rowMax := sh[0]
	t.Barrier()

	// The sum of the shifted exponentials.
	acc := float32(0)
	for i := lid; i < d.Width; i += RowWidth {
		acc = acc + kmath.Exp(x[base+i]-rowMax)
	}
	sh[lid] = acc
	t.Barrier()
	for stride := uint32(RowWidth / 2); stride > 0; stride /= 2 {
		if lid < stride {
			sh[lid] = sh[lid] + sh[lid+stride]
		}
		t.Barrier()
	}
	total := sh[0]

	for i := lid; i < d.Width; i += RowWidth {
		out[base+i] = kmath.Exp(x[base+i]-rowMax) / total
	}
}
