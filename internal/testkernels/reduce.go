// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import (
	"golang.design/x/accel"
	"golang.design/x/accel/kmath"
)

// clampIndex keeps an index inside a binding.
//
// It is a helper so that the compiler's helper path has something real to
// lower: a helper that takes a scalar, returns one, and is called from more
// than one kernel.
//
//accel:helper
func clampIndex(i uint32, n uint32) uint32 {
	if n == 0 {
		return 0
	}
	if i >= n {
		return n - 1
	}
	return i
}

// accumulate sums a slice of a binding.
//
// It is the helper that takes a resource, so that a binding's access is
// inferred through a call rather than only from the body that names it.
//
//accel:helper
func accumulate(in []float32, from uint32, to uint32) float32 {
	total := float32(0)
	for i := from; i < to; i++ {
		total += in[i]
	}
	return total
}

// SegmentSum sums a fixed-width segment of in into each element of out.
//
// It exists to exercise what spec 013 adds: a three-clause loop, a helper that
// returns a value, a helper that reads a binding, and the access inference that
// has to see through both. Its arithmetic is deliberately a sum rather than a
// product, so the generated lowering's explicit rounding points are visible in a
// differential test rather than collapsing to equality.
//
//accel:kernel workgroup=32
func SegmentSum(t accel.Thread, in []float32, out []float32) {
	i := t.GlobalID().X
	n := uint32(len(out))
	if i >= n {
		return
	}

	width := uint32(4)
	from := clampIndex(i*width, uint32(len(in)))
	to := clampIndex(from+width, uint32(len(in)))

	out[i] = accumulate(in, from, to)
}

// CountAbove counts how many elements exceed a threshold, using every loop form
// the subset admits so that each one has a lowering somebody runs.
//
//accel:kernel workgroup=16
func CountAbove(t accel.Thread, in []float32, out []int32) {
	i := t.GlobalID().X
	if i >= uint32(len(out)) {
		return
	}

	count := int32(0)
	limit := uint32(len(in))

	// Three-clause, with a continue.
	for j := uint32(0); j < limit; j++ {
		if in[j] <= 0 {
			continue
		}
		count++
	}

	// Condition-only.
	j := uint32(0)
	for j < limit {
		j++
	}

	// Infinite, with a break.
	for {
		if j >= limit {
			break
		}
		j++
	}

	out[i] = count
}

// Normalize scales each element by the reciprocal square root of a magnitude,
// which is the shape an RMS normalization takes.
//
// It exists to exercise what the rest of spec 013 adds: narrow storage on the
// way in, a bounded scalar intrinsic, and the explicit conversions Go forces
// because accel.Float16 carries no arithmetic operators at all.
//
//accel:kernel workgroup=32
func Normalize(t accel.Thread, in []accel.Float16, out []float32, scratch []float32) {
	i := t.GlobalID().X
	if i >= uint32(len(out)) {
		return
	}

	// The conversion is not a convention, it is the only thing that compiles:
	// a narrow value has no arithmetic operators, so it has to widen first.
	x := in[i].F32()

	magnitude := kmath.Sqrt(kmath.Abs(x) + 1)
	scratch[i] = magnitude
	out[i] = x * kmath.RSqrt(magnitude*magnitude)
}
