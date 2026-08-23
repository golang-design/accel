// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package numeq_test

import (
	"math"
	"testing"

	"golang.design/x/accel/internal/conformance/numeq"
)

// The budget is derived from the rounding count, so it is checked against the
// closed form rather than against a number somebody recorded.
func TestInterpolationBudgetIsTheDerivedOne(t *testing.T) {
	const u = 1.0 / (1 << 24)
	gamma := func(n int) float64 { return float64(n) * u / (1 - float64(n)*u) }

	for _, tc := range []struct {
		name   string
		vertex []float32
	}{
		{"triangle", []float32{1, 2, 3}},
		{"one negative dominates", []float32{0.5, -4, 1}},
		{"line", []float32{10, 20}},
		{"all zero", []float32{0, 0, 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			k := len(tc.vertex)
			var peak float64
			for _, v := range tc.vertex {
				if a := math.Abs(float64(v)); a > peak {
					peak = a
				}
			}
			want := (gamma(k+1) + gamma(k) + u) * peak

			got, ok := numeq.InterpolationBudget(tc.vertex)
			if !ok {
				t.Fatal("no budget for an ordinary primitive")
			}
			if got != want {
				t.Errorf("budget %g, want the derived %g", got, want)
			}
		})
	}

	// A triangle of unit-scale values is about 8u, which is the number
	// specs/008-numerics.md section 8.1 states. Checked so a change to the
	// rounding count shows up as a changed magnitude and not only as a changed
	// formula both sides of this test share.
	got, _ := numeq.InterpolationBudget([]float32{1, 1, 1})
	if got < 7*u || got > 9*u {
		t.Errorf("a unit triangle's budget is %g, and 008 section 8.1 says about %g", got, 8*u)
	}
}

// No vertices is a refusal, not a budget of zero. A zero budget would make
// every comparison against it fail while looking like an ordinary strict test.
func TestInterpolationBudgetRefusesWhatItCannotBound(t *testing.T) {
	if _, ok := numeq.InterpolationBudget(nil); ok {
		t.Error("a budget was produced for a primitive with no vertices")
	}
	// And past the point γ has no denominator, which is a fact about f32
	// arithmetic rather than a limit here.
	if _, ok := numeq.InterpolationBudget(make([]float32, 1<<25)); ok {
		t.Error("a budget was produced past the point the classical bound exists")
	}
}

// The comparison itself: inside the budget passes, outside fails, and the
// failure says why the budget is the size it is.
func TestWithinInterpolation(t *testing.T) {
	vertex := []float32{0, 8, 4}
	budget, _ := numeq.InterpolationBudget(vertex)

	// Values a hair inside the budget. float32 has to be able to represent the
	// perturbation, so it is applied at a magnitude where the budget exceeds an
	// ulp.
	want := []float32{4, 5, 6}
	inside := make([]float32, len(want))
	for i, w := range want {
		inside[i] = float32(float64(w) + budget*0.5)
	}
	if r := numeq.WithinInterpolation(inside, want, vertex); !r.OK() {
		t.Errorf("a value inside the budget was rejected: %v", r)
	}

	outside := []float32{4, float32(5 + budget*4), 6}
	r := numeq.WithinInterpolation(outside, want, vertex)
	if r.OK() {
		t.Fatal("a value four times the budget away was accepted")
	}
	if r.FirstDiff != 1 {
		t.Errorf("first difference at %d, want 1", r.FirstDiff)
	}
	if r.Peak != 8 {
		t.Errorf("the report's peak vertex value is %g, want 8", r.Peak)
	}
	if r.Diffs != 1 {
		t.Errorf("%d values outside the budget, want 1", r.Diffs)
	}
}

// A NaN is outside every budget. Without the explicit check it would be inside
// all of them, because every ordinary comparison against a NaN is false.
func TestWithinInterpolationRejectsNaN(t *testing.T) {
	vertex := []float32{1, 1, 1}
	nan := float32(math.NaN())
	if r := numeq.WithinInterpolation([]float32{nan}, []float32{1}, vertex); r.OK() {
		t.Error("a NaN was reported as within the budget")
	}
	if r := numeq.WithinInterpolation([]float32{1}, []float32{nan}, vertex); r.OK() {
		t.Error("a NaN reference was reported as matched")
	}
	// Two NaNs are still a failure: this compares numbers, and a kernel that
	// produced a NaN where the reference did is a separate question with its
	// own comparison.
	if r := numeq.WithinInterpolation([]float32{nan}, []float32{nan}, vertex); r.OK() {
		t.Error("two NaNs compared equal under a numeric budget")
	}
}

// A length mismatch is reported as such rather than as a value difference,
// because the two have completely different causes.
func TestWithinInterpolationReportsLength(t *testing.T) {
	r := numeq.WithinInterpolation([]float32{1, 2}, []float32{1}, []float32{1, 1, 1})
	if r.OK() {
		t.Fatal("mismatched lengths compared equal")
	}
	if r.Len != 2 || r.WantLen != 1 {
		t.Errorf("report says %d and %d, want 2 and 1", r.Len, r.WantLen)
	}
	if got := r.String(); got != "length 2, want 1" {
		t.Errorf("the message is %q", got)
	}
}

// An undefined budget refuses rather than failing, and says so.
func TestInterpReportRefusalReadsAsOne(t *testing.T) {
	r := numeq.WithinInterpolation([]float32{1}, []float32{1}, nil)
	if r.OK() {
		t.Fatal("a comparison with no budget reported success")
	}
	if r.Defined {
		t.Error("the report claims a budget it does not have")
	}
	if got := r.String(); got == "" {
		t.Error("a refusal with no message")
	}
}

// The passing report says what it checked, because a test that prints only
// "ok" on success gives a reader nothing when a neighbouring one fails.
func TestInterpReportDescribesASuccess(t *testing.T) {
	r := numeq.WithinInterpolation([]float32{1}, []float32{1}, []float32{1, 1, 1})
	if !r.OK() {
		t.Fatalf("identical values outside the budget: %v", r)
	}
	if got := r.String(); got == "" {
		t.Error("a success with no message")
	}
}
