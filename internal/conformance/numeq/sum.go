// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package numeq

import (
	"fmt"
	"math"
)

// The derived bounds of specs/008-numerics.md section 7.
//
// These are the "new functions, each naming its budget" the package doc
// promised rather than a tolerance argument added to [Exact]. The distinction
// is the whole design: a tolerance is a number somebody tuned until the test
// passed, and a budget is derived from the operation's shape, so a failure
// means the operation is wrong rather than that the number was optimistic.

// Unit roundoff for f32: half an ulp at 1, which is 2^-24.
const unitRoundoff = 1.0 / (1 << 24)

// Gamma is the classical error factor for n roundings:
//
//	γ(n) = n·u / (1 - n·u)
//
// It is defined only while n·u < 1, and the second return says whether it is.
// Past that point the denominator goes to zero or negative and the "bound" it
// produces is meaningless — an accumulation of that many terms has no useful
// forward bound at f32, which is a fact about the arithmetic rather than a
// limitation here. See specs/008-numerics.md section 7.
func Gamma(n int) (float64, bool) {
	if n < 0 {
		return 0, false
	}
	nu := float64(n) * unitRoundoff
	if nu >= 1 {
		return 0, false
	}
	return nu / (1 - nu), true
}

// SumBudget is the absolute error budget for summing terms.
//
// depth is the maximum number of additions on any path from an input to the
// result: K-1 for a sequential loop, and ⌈log2 K⌉ for a balanced pairwise tree.
// That difference is why a workgroup tree reduction is *more* accurate than a
// sequential loop rather than less, and it is why the depth is a parameter here
// rather than inferred from the term count — specs/008-numerics.md is explicit
// that the harness does not infer "tree" from a name.
//
// magnitude is Σ|xᵢ|, which is what the bound scales: cancellation between
// large terms is exactly the case where a relative bound on the result would be
// wrong.
func SumBudget(depth int, magnitude float64) (float64, bool) {
	g, ok := Gamma(depth)
	if !ok {
		return 0, false
	}
	return g * magnitude, true
}

// SumReport is how a computed sum compared against its reference.
type SumReport struct {
	Got, Want float64

	// Error is |got - want| and Budget is what section 7 allows it to be.
	Error, Budget float64

	// Depth and Magnitude are the budget's inputs, reported so a failure says
	// why the budget is the size it is rather than only that it was exceeded.
	Depth     int
	Magnitude float64

	// Defined is false when no bound exists at all, which is a refusal rather
	// than a failure: see [Gamma].
	Defined bool
}

func (r SumReport) OK() bool { return r.Defined && r.Error <= r.Budget }

func (r SumReport) String() string {
	if !r.Defined {
		return fmt.Sprintf("no f32 error bound exists for an addition depth of %d: "+
			"depth times the unit roundoff reaches 1, so the classical bound's "+
			"denominator vanishes", r.Depth)
	}
	if r.OK() {
		return fmt.Sprintf("got %v, want %v, error %g within the %g budget "+
			"(depth %d, magnitude %g)", r.Got, r.Want, r.Error, r.Budget, r.Depth, r.Magnitude)
	}
	return fmt.Sprintf("got %v, want %v: the error %g exceeds the %g budget for an "+
		"addition depth of %d over terms totalling %g in magnitude",
		r.Got, r.Want, r.Error, r.Budget, r.Depth, r.Magnitude)
}

// Sum compares a computed f32 sum against a higher-precision reference of the
// same terms.
//
// The reference is computed here from the terms rather than taken as a
// parameter, so a caller cannot accidentally supply a reference computed at f32
// — which would make the comparison a tautology, since both sides would carry
// the same rounding.
func Sum(got float32, terms []float32, depth int) SumReport {
	// The reference: exact widening of each stored value, summed at f64. Spec
	// 008 section 7 requires the widening to be exact so no input-conversion
	// error is added to the budget.
	var want, magnitude float64
	for _, t := range terms {
		want += float64(t)
		magnitude += math.Abs(float64(t))
	}

	budget, ok := SumBudget(depth, magnitude)
	return SumReport{
		Got: float64(got), Want: want,
		Error: math.Abs(float64(got) - want), Budget: budget,
		Depth: depth, Magnitude: magnitude, Defined: ok,
	}
}

// TreeDepth is the maximum addition depth of a balanced pairwise reduction over
// n terms, which is ⌈log2 n⌉.
func TreeDepth(n int) int {
	d := 0
	for 1<<d < n {
		d++
	}
	return d
}

// ProductBudget is the absolute error budget for multiplying terms.
//
// # Why it takes the product and SumBudget takes a sum of magnitudes
//
// A product's error is *relative* and a sum's is not, which is the whole
// difference and is specs/008-numerics.md §7.1. Each rounding scales the
// running product by (1+δ), so the errors compose: the computed value is
// p·∏(1+δᵢ) and its deviation from p is bounded by γ(depth)·|p| directly. A sum
// admits no such factorisation, so its budget carries Σ|xᵢ| — a quantity that
// can be arbitrarily larger than the sum it bounds, which is exactly the
// cancellation case.
//
// product is the exact product, not the computed one. Passing the f32 result
// would make the budget scale with the error it is bounding.
func ProductBudget(depth int, product float64) (float64, bool) {
	g, ok := Gamma(depth)
	if !ok {
		return 0, false
	}
	return g * math.Abs(product), true
}

// Product compares a computed f32 product against a higher-precision reference
// of the same terms.
//
// Like [Sum], the reference is computed here rather than taken as a parameter,
// so a caller cannot supply one computed at f32 and make the comparison a
// tautology.
//
// # The domain, which is not a bound
//
// §7.1: a product of K bounded terms is the largest raised to the Kth, so a
// subgroup of 64 lanes holding values of magnitude 4 reaches 2^128 and
// overflows f32 while every term and the true result are ordinary. This reports
// `Defined: false` when the exact product leaves f32's range, because the bound
// above assumes no intermediate overflow — a caller whose inputs do that is
// outside the domain rather than failing the comparison.
func Product(got float32, terms []float32, depth int) ProductReport {
	exact := 1.0
	for _, t := range terms {
		exact *= float64(t)
	}
	r := ProductReport{
		Got: float64(got), Want: exact, Depth: depth, Product: exact,
		Error: math.Abs(float64(got) - exact),
	}
	if math.Abs(exact) > math.MaxFloat32 {
		return r
	}
	r.Budget, r.Defined = ProductBudget(depth, exact)
	return r
}

// ProductReport is how a computed product compared against its reference.
type ProductReport struct {
	Got, Want float64

	// Error is |got - want| and Budget is what §7.1 allows it to be.
	Error, Budget float64

	// Depth is the maximum number of multiplications on any path, and Product
	// is the exact product the budget scales. Both are reported so a failure
	// says why the budget is the size it is.
	Depth   int
	Product float64

	// Defined is false when no bound exists: either the depth is past what
	// γ admits, or the exact product leaves f32's range.
	Defined bool
}

func (r ProductReport) OK() bool { return r.Defined && r.Error <= r.Budget }

func (r ProductReport) String() string {
	if !r.Defined && math.Abs(r.Product) > math.MaxFloat32 {
		return fmt.Sprintf("the exact product is %g, which is outside f32's range: "+
			"specs/008-numerics.md §7.1's bound assumes no intermediate overflow, so "+
			"these inputs are outside the domain rather than failing", r.Product)
	}
	if !r.Defined {
		return fmt.Sprintf("no f32 error bound exists for a multiplication depth of %d: "+
			"depth times the unit roundoff reaches 1, so the classical bound's "+
			"denominator vanishes", r.Depth)
	}
	if r.OK() {
		return fmt.Sprintf("got %v, want %v, error %g within the %g budget "+
			"(depth %d, product %g)", r.Got, r.Want, r.Error, r.Budget, r.Depth, r.Product)
	}
	return fmt.Sprintf("got %v, want %v: the error %g exceeds the %g budget for a "+
		"multiplication depth of %d over a product of %g",
		r.Got, r.Want, r.Error, r.Budget, r.Depth, r.Product)
}
