// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package numeq

import (
	"fmt"
	"math"
)

// The derived bound of specs/008-numerics.md section 8.1, for the
// perspective-correct interpolation specs/035-cpu-rasterizer.md performs.
//
// It is a budget and not a tolerance for the reason [SumBudget] is: the number
// comes from counting the roundings the formula performs, so exceeding it means
// the interpolation is wrong rather than that the number was optimistic.

// InterpolationBudget is the absolute error budget for one perspective-correct
// interpolated attribute over a primitive with the given per-vertex values.
//
//	|Δa| ≤ (γ(K+1) + γ(K) + u) · maxᵢ|aᵢ|
//
// The count comes from the longest rounding path through
//
//	a = Σ λᵢ(aᵢ/wᵢ) / Σ (λᵢ/wᵢ)
//
// a numerator term taking a divide and a multiply and the sum taking K-1
// additions, a denominator term taking one divide, and one more rounding for
// the final quotient.
//
// It scales with the largest vertex value rather than with a sum of magnitudes,
// which is the one place this differs from [SumBudget]: every barycentric weight
// is non-negative and every w is positive, so no cancellation is possible and
// the result is a convex combination of the vertex values.
//
// This bound holds for a *fixed* set of barycentrics. Two implementations that
// compute them differently evaluate the attribute at slightly different points,
// and that term is section 8.1's Lipschitz term over the attribute's variation
// across the primitive -- a separate budget, not a larger version of this one.
// Using this function for a cross-backend comparison would be comparing under a
// bound for an error the comparison does not contain.
func InterpolationBudget(vertex []float32) (float64, bool) {
	k := len(vertex)
	if k < 1 {
		return 0, false
	}
	num, ok := Gamma(k + 1)
	if !ok {
		return 0, false
	}
	den, ok := Gamma(k)
	if !ok {
		return 0, false
	}
	var peak float64
	for _, v := range vertex {
		if a := math.Abs(float64(v)); a > peak {
			peak = a
		}
	}
	return (num + den + unitRoundoff) * peak, true
}

// InterpReport is how an interpolated attribute compared against its reference.
type InterpReport struct {
	// FirstDiff is the index of the first element outside the budget, or -1.
	FirstDiff int

	// Got and Want are the values at FirstDiff, and Error is their distance.
	Got, Want, Error float64

	// Budget is what section 8.1 allows, and Peak is the maxᵢ|aᵢ| it scales
	// with. Both are reported so a failure says why the budget is the size it
	// is rather than only that it was exceeded.
	Budget, Peak float64

	// Diffs is how many elements are outside the budget, which distinguishes
	// one bad fragment from a wrong formula.
	Diffs int

	// Len and WantLen are the two lengths, which differ only on a caller error.
	Len, WantLen int

	// Defined is false when no bound exists, which is a refusal rather than a
	// failure: see [Gamma].
	Defined bool
}

func (r InterpReport) OK() bool {
	return r.Defined && r.Len == r.WantLen && r.Diffs == 0
}

func (r InterpReport) String() string {
	if !r.Defined {
		return "no f32 interpolation bound exists for that vertex count"
	}
	if r.Len != r.WantLen {
		return fmt.Sprintf("length %d, want %d", r.Len, r.WantLen)
	}
	if r.OK() {
		return fmt.Sprintf("within the %g budget over %d values (peak vertex value %g)",
			r.Budget, r.Len, r.Peak)
	}
	return fmt.Sprintf("first difference outside the budget at index %d: got %v, want %v, "+
		"error %g against a %g budget from a peak vertex value of %g (%d of %d outside)",
		r.FirstDiff, r.Got, r.Want, r.Error, r.Budget, r.Peak, r.Diffs, r.Len)
}

// WithinInterpolation compares interpolated values against a higher-precision
// reference under [InterpolationBudget].
//
// vertex is the primitive's per-vertex values for this attribute, which is what
// the budget derives from -- not a tolerance, and not something a caller tunes:
// changing it means claiming the primitive had different vertices.
//
// A NaN on either side is a failure whatever the budget says, because every
// ordinary comparison against a NaN is false and one would otherwise be reported
// as within any budget at all. This is the same rule [WithinULP] applies and for
// the same reason.
func WithinInterpolation(got, want []float32, vertex []float32) InterpReport {
	r := InterpReport{FirstDiff: -1, Len: len(got), WantLen: len(want)}
	budget, ok := InterpolationBudget(vertex)
	if !ok {
		return r
	}
	r.Defined, r.Budget = true, budget
	for _, v := range vertex {
		if a := math.Abs(float64(v)); a > r.Peak {
			r.Peak = a
		}
	}
	if len(got) != len(want) {
		return r
	}
	for i := range got {
		g, w := float64(got[i]), float64(want[i])
		gNaN, wNaN := g != g, w != w
		var err float64
		bad := gNaN || wNaN
		if !bad {
			err = math.Abs(g - w)
			bad = err > budget
		}
		if !bad {
			continue
		}
		r.Diffs++
		if r.FirstDiff < 0 {
			r.FirstDiff, r.Got, r.Want, r.Error = i, g, w, err
		}
	}
	return r
}
