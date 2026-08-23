// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package numeq compares results against references.
//
// It deliberately has no function that takes an absolute or relative tolerance.
// A tolerance parameter is a number somebody tuned until the test passed, and
// once one exists every later failure gets fixed by raising it. Every
// comparison here instead states the *reason* the values may differ, and at M1
// there is exactly one such reason: none. Bytes moved through the device come
// back unchanged, so the comparison is exact.
//
// Spec 008's derived-bound forms arrive with the milestones that produce
// inexact results. They will be new functions, each naming its budget, rather
// than a tolerance argument added to these.
//
// A failure reports the first differing index, because "the arrays differ" does
// not tell anyone where to look, and a dump of two long arrays does not either.
package numeq

import (
	"fmt"
	"math"
	"strings"
)

// Report describes how two sequences differ.
type Report struct {
	// Equal is the answer; everything else explains a false.
	Equal bool

	// FirstDiff is the index of the first differing element, or -1.
	FirstDiff int

	// Got and Want are the values at FirstDiff, formatted for the element type.
	Got, Want string

	// Diffs is how many elements differ in total, which distinguishes one
	// corrupted element from a wholly wrong result.
	Diffs int

	// Len is the shared length, or the two lengths when they differ.
	Len, WantLen int
}

func (r Report) String() string {
	if r.Equal {
		return "equal"
	}
	if r.Len != r.WantLen {
		return fmt.Sprintf("length %d, want %d", r.Len, r.WantLen)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "first difference at index %d: got %s, want %s", r.FirstDiff, r.Got, r.Want)
	if r.Diffs > 1 {
		fmt.Fprintf(&b, " (%d of %d elements differ)", r.Diffs, r.Len)
	}
	return b.String()
}

// Exact compares two sequences element for element.
//
// It is the right comparison for everything M1 produces. A transfer moves bytes
// and never converts, so a byte that comes back different is a defect and not a
// rounding difference, and comparing it under any tolerance would hide exactly
// the class of bug the transfer path can have.
//
// The element type must be comparable rather than floating-point-aware on
// purpose: comparing float32 with == makes a NaN compare unequal to itself,
// which is wrong for a bit-preservation check. Use [ExactBits] for floats.
func Exact[T comparable](got, want []T) Report {
	r := Report{FirstDiff: -1, Len: len(got), WantLen: len(want)}
	if len(got) != len(want) {
		return r
	}
	for i := range got {
		if got[i] != want[i] {
			r.Diffs++
			if r.FirstDiff < 0 {
				r.FirstDiff = i
				r.Got, r.Want = fmt.Sprint(got[i]), fmt.Sprint(want[i])
			}
		}
	}
	r.Equal = r.Diffs == 0
	return r
}

// ExactBits compares two float sequences by their IEEE encodings.
//
// This is the comparison spec 001 section 6.1's bit-pattern guarantee needs. A
// value-wise comparison passes when a backend turns a negative zero into a
// positive one and fails when both sides hold the same NaN, so it is wrong in
// both directions for a path that promises to preserve bits.
func ExactBits[T float32 | float64](got, want []T, bits func(T) uint64) Report {
	r := Report{FirstDiff: -1, Len: len(got), WantLen: len(want)}
	if len(got) != len(want) {
		return r
	}
	for i := range got {
		g, w := bits(got[i]), bits(want[i])
		if g != w {
			r.Diffs++
			if r.FirstDiff < 0 {
				r.FirstDiff = i
				r.Got = fmt.Sprintf("%v (%#016x)", got[i], g)
				r.Want = fmt.Sprintf("%v (%#016x)", want[i], w)
			}
		}
	}
	r.Equal = r.Diffs == 0
	return r
}

// ULPDistance is the ordered-bit distance between two f32 values.
//
// Ordered-bit, which is what makes it meaningful across a binade boundary: the
// two neighbours of a power of two are different absolute sizes, and a
// difference measured in absolute terms would count them differently depending
// on which side of the boundary the value fell. specs/008-numerics.md section 6
// defines the sqrt ceiling in exactly these terms and for exactly this reason.
//
// The sign bit is folded so that -0 and +0 are one step apart from each other
// and zero steps from themselves, rather than 2^32 apart as a raw bit
// comparison would make them.
func ULPDistance(a, b float32) uint64 {
	ordered := func(v float32) int64 {
		bits := math.Float32bits(v)
		if bits&0x80000000 != 0 {
			// Negative floats descend as their bit patterns ascend, so the sign
			// bit is stripped and the magnitude negated, which makes the whole
			// range monotone and puts the two zeros in the same place.
			//
			// The sign is tested on the *unsigned* pattern. Converting to a
			// signed integer first and testing for negative does not work:
			// widening uint32 to int64 never produces a negative, so -0 came
			// out 2^31 steps from +0.
			return -int64(bits & 0x7FFFFFFF)
		}
		return int64(bits)
	}
	d := ordered(a) - ordered(b)
	if d < 0 {
		d = -d
	}
	return uint64(d)
}

// WithinULP compares two f32 slices against a ceiling in ULP.
//
// A ceiling rather than a tolerance, and the distinction is
// specs/008-numerics.md's: the number comes from that spec's normative table
// for the primitive involved, not from what a run happened to produce. A
// backend whose worst case exceeds its ceiling changes its lowering.
//
// NaN is checked first and separately, because every ordinary comparison
// against a NaN is false: a NaN where a number was expected would otherwise be
// reported as within any ceiling at all.
func WithinULP(got, want []float32, ceiling uint64) Report {
	r := Report{Equal: true, FirstDiff: -1, Len: len(got), WantLen: len(want)}
	if len(got) != len(want) {
		r.Equal = false
		return r
	}
	worst := uint64(0)
	for i := range got {
		g, w := got[i], want[i]
		gNaN, wNaN := g != g, w != w
		var bad bool
		var d uint64
		switch {
		case gNaN || wNaN:
			// A NaN is checked first and separately, because every ordinary
			// comparison against one is false: a NaN where a number was
			// expected would otherwise be reported as within any ceiling at
			// all.
			bad = gNaN != wNaN
		default:
			d = ULPDistance(g, w)
			bad = d > ceiling
			if d > worst {
				worst = d
			}
		}
		if !bad {
			continue
		}
		r.Diffs++
		if r.FirstDiff < 0 {
			r.FirstDiff = i
			r.Equal = false
			r.Got = fmt.Sprintf("%v (%#08x)", g, math.Float32bits(g))
			r.Want = fmt.Sprintf("%v (%#08x), %d ULP away, ceiling %d",
				w, math.Float32bits(w), d, ceiling)
		}
	}
	return r
}
