// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package numeq_test

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel/internal/conformance/numeq"
)

// f32 is the arithmetic the budget is a bound for: every intermediate rounds.
func f32(v float64) float32 { return float32(v) }

// Each rule's sensitivity is what its derivative says, checked by moving one
// input by its own error and watching the result move by at most the budget.
//
// specs/008-numerics.md section 8's rule is
// local + Σ sensitivity·inputBudget, so a sensitivity that is too small is a
// budget a correct implementation can fail. This walks each operator with a
// deliberately widened input and requires the true displacement to stay inside.
func TestEachRuleBoundsItsOwnDerivative(t *testing.T) {
	for _, c := range []struct {
		name string
		at   float64
		of   func(numeq.Value) numeq.Value
		f    func(float64) float64
	}{
		{"sqrt", 4, numeq.Sqrt, math.Sqrt},
		{"exp", 1.5, numeq.Exp, math.Exp},
		{"log", 3, numeq.Log, math.Log},
		{"tanh", 0.7, numeq.Tanh, math.Tanh},
	} {
		t.Run(c.name, func(t *testing.T) {
			x := numeq.Input(f32(c.at))
			v := c.of(x)
			if !v.Defined {
				t.Fatalf("%s at %v is undefined: %s", c.name, c.at, v)
			}
			// The worst case the budget claims to cover: the input sitting at
			// either end of its own interval.
			for _, at := range []float64{c.at - x.Err, c.at + x.Err} {
				moved := c.f(at)
				if d := math.Abs(moved - v.Ref); d > v.Err {
					t.Errorf("moving the input to %v moves %s by %v, and the budget is %v: %s",
						at, c.name, d, v.Err, v)
				}
			}
		})
	}
}

// A budget that is not conservative is worthless, and a budget that is
// enormous is not a budget. Each composed rule is checked against the f32
// arithmetic it bounds, and required to be tight enough that a drift several
// orders larger fails.
func TestTheComposedBudgetsAreTightAgainstF32(t *testing.T) {
	xs := []float32{1.25, -0.5, 2, 0.125, -1.75, 3.5, 0.75, -2.25}
	vals := make([]numeq.Value, len(xs))
	for i, x := range xs {
		vals[i] = numeq.Input(x)
	}
	depth := numeq.TreeDepth(len(xs))

	t.Run("dot", func(t *testing.T) {
		v := numeq.Dot(vals, vals, depth)
		var got float32
		for _, x := range xs {
			got += x * x
		}
		if !v.OK(got) {
			t.Errorf("the f32 dot product is %v and the budget is %s", got, v)
		}
		// A drift of one part in ten thousand is orders above the budget, so
		// the bound is not a tolerance that would accept anything.
		if v.OK(got * 1.0001) {
			t.Errorf("a drift of 1e-4 fits the budget %s, which is not a bound", v)
		}
	})

	t.Run("rmsnorm", func(t *testing.T) {
		const eps = 1e-6
		w := numeq.Input(1.5)
		var ss float32
		for _, x := range xs {
			ss += x * x
		}
		scale := float32(math.Sqrt(float64(ss/float32(len(xs))) + eps))
		for i := range xs {
			v := numeq.RMSNormValue(vals, w, i, depth, eps)
			got := xs[i] * 1.5 / scale
			if !v.OK(got) {
				t.Errorf("output %d is %v and the budget is %s", i, got, v)
			}
			if v.OK(got*1.001) && got != 0 {
				t.Errorf("output %d accepts a 1e-3 drift: %s", i, v)
			}
		}
	})

	t.Run("softmax", func(t *testing.T) {
		var m float32
		for _, x := range xs {
			m = max(m, x)
		}
		ex := make([]float32, len(xs))
		var sum float32
		for i, x := range xs {
			ex[i] = float32(math.Exp(float64(x - m)))
			sum += ex[i]
		}
		for i := range xs {
			v := numeq.SoftmaxValue(vals, i, depth)
			got := ex[i] / sum
			if !v.OK(got) {
				t.Errorf("output %d is %v and the budget is %s", i, got, v)
			}
			if v.OK(got * 1.01) {
				t.Errorf("output %d accepts a 1e-2 drift: %s", i, v)
			}
		}
	})
}

// A softmax's outputs sum to one, and the composed budget says by how much.
//
// The property is what a caller checks first, and it is a composition of eight
// budgets rather than a tolerance. Included because it is the case where a
// per-output bound and the whole distribution's bound differ.
func TestASoftmaxSumsToOneWithinItsComposedBudget(t *testing.T) {
	xs := []float32{3, 1, -2, 0.5}
	vals := make([]numeq.Value, len(xs))
	for i, x := range xs {
		vals[i] = numeq.Input(x)
	}
	depth := numeq.TreeDepth(len(xs))

	var m float32
	for _, x := range xs {
		m = max(m, x)
	}
	ex := make([]float32, len(xs))
	var sum, total float32
	for i, x := range xs {
		ex[i] = float32(math.Exp(float64(x - m)))
		sum += ex[i]
	}
	budget := 0.0
	for i := range xs {
		v := numeq.SoftmaxValue(vals, i, depth)
		if !v.Defined {
			t.Fatalf("output %d is undefined: %s", i, v)
		}
		budget += v.Err
		total += ex[i] / sum
	}
	if d := math.Abs(float64(total) - 1); d > budget {
		t.Errorf("the outputs sum to %v, which is %v from one, and the composed budget "+
			"is %v", total, d, budget)
	}
}

// Section 8's refusals: where a sensitivity has no finite bound, the harness
// says so rather than returning a large number.
//
// A budget of 1e30 is a comparison that passes, which is the opposite of what a
// derived bound is for.
func TestASingularityIsRefusedRatherThanBounded(t *testing.T) {
	for _, c := range []struct {
		name string
		of   func() numeq.Value
		want string
	}{
		{
			name: "a divisor spanning zero",
			of:   func() numeq.Value { return numeq.Div(numeq.Input(1), numeq.Input(0)) },
			want: "contains zero",
		},
		{
			name: "a divisor whose interval reaches zero",
			of: func() numeq.Value {
				// Not zero itself: its own input rounding puts the interval
				// across zero, which is the case a check against == 0 misses.
				return numeq.Div(numeq.Input(1), numeq.Sub(numeq.Input(2), numeq.Input(2)))
			},
			want: "contains zero",
		},
		{"sqrt at zero", func() numeq.Value { return numeq.Sqrt(numeq.Input(0)) }, "unbounded"},
		{"sqrt of a negative", func() numeq.Value { return numeq.Sqrt(numeq.Input(-1)) }, "unbounded"},
		{"log at zero", func() numeq.Value { return numeq.Log(numeq.Input(0)) }, "zero or below"},
		{"exp overflowing f64", func() numeq.Value { return numeq.Exp(numeq.Input(1000)) }, "overflows"},
	} {
		t.Run(c.name, func(t *testing.T) {
			v := c.of()
			if v.Defined {
				t.Fatalf("accepted with budget %v: %s", v.Err, v)
			}
			if !strings.Contains(v.String(), c.want) {
				t.Errorf("the refusal does not say %q: %s", c.want, v)
			}
		})
	}
}

// A refusal travels forward. One undefined input makes the whole expression
// undefined rather than a number that looks fine.
func TestARefusalPropagatesThroughTheExpression(t *testing.T) {
	bad := numeq.Log(numeq.Input(0))
	v := numeq.Add(numeq.Mul(bad, numeq.Input(2)), numeq.Input(1))
	if v.Defined {
		t.Fatalf("an expression over an undefined value is defined, with budget %v", v.Err)
	}
	if !strings.Contains(v.String(), "undefined") {
		t.Errorf("the trace does not say what went wrong: %s", v)
	}
}

// The trace names the operator that spent the budget, which is the whole reason
// section 8 composes rather than tolerates.
func TestTheTraceNamesTheOperatorThatSpentTheBudget(t *testing.T) {
	// A tiny input through exp against a large one through add: the exp term
	// dominates because its sensitivity is its own value.
	big := numeq.Exp(numeq.Input(20))
	small := numeq.Input(1)
	v := numeq.Add(big, small)
	if !v.Defined {
		t.Fatalf("undefined: %s", v)
	}
	s := v.String()
	if !strings.Contains(s, "exp") {
		t.Errorf("the trace does not name exp, which contributed the most: %s", s)
	}
	if !strings.Contains(s, "<-") {
		t.Errorf("the trace does not mark the largest term: %s", s)
	}
	// And the mark is on the exp term rather than on whichever came first.
	marked := ""
	for _, part := range strings.Split(s[strings.Index(s, "(")+1:], ", ") {
		if strings.Contains(part, "<-") {
			marked = part
		}
	}
	if !strings.HasPrefix(marked, "exp ") {
		t.Errorf("the marked term is %q, want the exp one: %s", marked, s)
	}
}

// Cancellation is where a relative bound is wrong, and the composed budget
// carries the magnitudes rather than the result.
//
// Two large terms summing to a small one: the result is near zero and the
// budget is not, because the error came from the terms.
func TestCancellationKeepsTheTermsMagnitude(t *testing.T) {
	a := numeq.Input(1 << 20)
	b := numeq.Input(-(1 << 20))
	v := numeq.SumValues([]numeq.Value{a, b}, 1)
	if !v.Defined {
		t.Fatalf("undefined: %s", v)
	}
	if v.Ref != 0 {
		t.Fatalf("the reference is %v, want 0", v.Ref)
	}
	// A relative bound on a zero result is zero, which would be a claim that
	// two f32 values a million apart cancel exactly.
	if v.Err <= 0 {
		t.Errorf("the budget is %v: a cancelling sum's error comes from its terms, "+
			"not from its result", v.Err)
	}
}
