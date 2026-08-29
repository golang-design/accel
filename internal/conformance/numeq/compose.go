// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package numeq

import (
	"fmt"
	"math"
	"strings"
)

// Forward absolute-error propagation, specs/008-numerics.md section 8.
//
// Section 7 bounds one operation. This composes them: for y = f(x₁…xₙ) the
// budget is
//
//	local primitive error + Σᵢ sensitivityᵢ · inputBudgetᵢ
//
// where sensitivityᵢ is a conservative bound on |∂f/∂xᵢ| over the interval
// referenceᵢ ± inputBudgetᵢ. That is the whole rule, and everything below is
// one operator's version of it.
//
// # Why this rather than a relative tolerance
//
// A relative tolerance on a composed result is a number nobody derived. It
// cannot say which operator spent the error, it grows silently as a model gains
// layers, and it is right about a cancelling sum by accident. A composed budget
// carries a *trace*, so a failed comparison names the operator rather than the
// model.
//
// # Refusal is a result
//
// Where an input interval crosses a singularity — a divisor spanning zero, a
// logarithm at or below it — there is no finite sensitivity and the harness
// says so instead of returning a large number. Section 8 is explicit that the
// test must then narrow its domain or add a proved rule, and a budget of 1e30
// would be a comparison that passes rather than a refusal.

// A Value is a reference value and a bound on the absolute error around it.
//
// Ref is computed in float64, which is the "higher-precision reference" section
// 8 asks for: 29 more bits of significand than f32, so its own rounding is
// below the budget it is being compared against by a factor of 2^29.
type Value struct {
	Ref float64

	// Err is the absolute error bound: the computed f32 result lies within
	// Ref ± Err.
	Err float64

	// Defined is false once any step had no finite sensitivity. It travels
	// forward, so one refusal anywhere makes the whole expression a refusal
	// rather than a number that looks fine.
	Defined bool

	// Op names what produced this value and Reason says why it is undefined,
	// which is what makes a trace attributable.
	Op     string
	Reason string

	// terms is each input's contribution to Err, for the trace. It is the
	// per-operator attribution section 8 requires and it is why the budget is
	// built as a graph rather than accumulated into one float.
	terms []term
}

type term struct {
	op   string
	sens float64
	err  float64
}

// Constant is an exactly representable input: a literal, a shape, a count.
func Constant(v float64) Value {
	return Value{Ref: v, Defined: true, Op: "constant"}
}

// Input is a value that entered the computation as an f32, so it already
// carries half an ulp of its own magnitude.
//
// Not zero error. A test that fed exact float64 references into a budget and
// compared against an f32 computation would be charging the computation for a
// rounding it did not perform, and would under-budget by exactly the amount the
// inputs were already wrong by.
func Input(v float32) Value {
	return Value{
		Ref: float64(v), Err: math.Abs(float64(v)) * unitRoundoff,
		Defined: true, Op: "input",
	}
}

// Undefined is a value no rule could bound, naming why.
func Undefined(op, reason string) Value {
	return Value{Op: op, Reason: reason}
}

// OK reports whether a computed f32 lies inside this value's budget.
func (v Value) OK(got float32) bool {
	if !v.Defined {
		return false
	}
	d := math.Abs(float64(got) - v.Ref)
	return d <= v.Err
}

// String is the budget trace: the reference, the bound, and what each operator
// contributed to it.
//
// Sorted by contribution rather than by construction order, because the
// question a failure asks is "which operator spent the budget" and the answer
// is the largest term.
func (v Value) String() string {
	if !v.Defined {
		return fmt.Sprintf("%s: undefined, %s", v.Op, v.Reason)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s: ref %v ± %v", v.Op, v.Ref, v.Err)
	if len(v.terms) == 0 {
		return b.String()
	}
	worst := v.terms[0]
	for _, t := range v.terms[1:] {
		if t.sens*t.err > worst.sens*worst.err {
			worst = t
		}
	}
	fmt.Fprintf(&b, " (")
	for i, t := range v.terms {
		if i > 0 {
			fmt.Fprintf(&b, ", ")
		}
		mark := ""
		if t == worst && len(v.terms) > 1 {
			mark = " <-"
		}
		fmt.Fprintf(&b, "%s %v×%v=%v%s", t.op, t.sens, t.err, t.sens*t.err, mark)
	}
	fmt.Fprintf(&b, ")")
	return b.String()
}

// lo and hi are the interval a value could actually hold.
func (v Value) lo() float64 { return v.Ref - v.Err }
func (v Value) hi() float64 { return v.Ref + v.Err }

// absMax is the largest magnitude in the interval, which is the conservative
// side for a sensitivity that grows with the argument.
func (v Value) absMax() float64 { return math.Max(math.Abs(v.lo()), math.Abs(v.hi())) }

// spans reports whether the interval contains zero, which is where a division
// and every logarithmic or reciprocal rule loses its sensitivity.
func (v Value) spans() bool { return v.lo() <= 0 && v.hi() >= 0 }

// compose applies section 8's rule for one operator.
//
// ref is the exact result of the operator on the inputs' references; local is
// the operator's own rounding, which for a single f32 operation is u·|ref|; and
// each sensitivity is a conservative bound on |∂f/∂xᵢ| over the inputs'
// intervals.
func compose(op string, ref, local float64, sens []float64, in []Value) Value {
	out := Value{Ref: ref, Err: local, Defined: true, Op: op}
	out.terms = append(out.terms, term{op: "rounding", sens: 1, err: local})
	for i, x := range in {
		if !x.Defined {
			return Value{Op: op, Reason: fmt.Sprintf("input %d (%s) is undefined: %s",
				i, x.Op, x.Reason)}
		}
		out.Err += sens[i] * x.Err
		out.terms = append(out.terms, term{op: x.Op, sens: sens[i], err: x.Err})
	}
	if math.IsNaN(out.Err) || math.IsInf(out.Err, 0) {
		return Undefined(op, "the propagated bound is not finite")
	}
	return out
}

// roundoff is one f32 operation's own error at a result magnitude.
func roundoff(ref float64) float64 { return math.Abs(ref) * unitRoundoff }

// Add is x + y. Both sensitivities are 1, which is why a sum of values with
// opposite signs has a bound much larger than its result -- the cancellation
// section 7 refuses to hide behind a relative bound.
func Add(x, y Value) Value {
	ref := x.Ref + y.Ref
	return compose("add", ref, roundoff(ref), []float64{1, 1}, []Value{x, y})
}

// Sub is x - y.
func Sub(x, y Value) Value {
	ref := x.Ref - y.Ref
	return compose("sub", ref, roundoff(ref), []float64{1, 1}, []Value{x, y})
}

// Mul is x·y, with ∂/∂x = y and ∂/∂y = x taken at the interval's widest.
func Mul(x, y Value) Value {
	ref := x.Ref * y.Ref
	return compose("mul", ref, roundoff(ref),
		[]float64{y.absMax(), x.absMax()}, []Value{x, y})
}

// Div is x/y, and it refuses a divisor whose interval contains zero.
//
// Not because the division would trap -- it would return an infinity -- but
// because ∂/∂y = -x/y² has no finite bound there, so any number this returned
// would be a claim the arithmetic does not support. Section 8's answer is to
// narrow the domain, and that is a decision for the test rather than for a
// default here.
func Div(x, y Value) Value {
	if y.spans() {
		return Undefined("div", fmt.Sprintf(
			"the divisor's interval [%v, %v] contains zero, so ∂/∂y = -x/y² is unbounded",
			y.lo(), y.hi()))
	}
	ref := x.Ref / y.Ref
	// |1/y| is largest at the interval end nearest zero.
	yMin := math.Min(math.Abs(y.lo()), math.Abs(y.hi()))
	return compose("div", ref, roundoff(ref),
		[]float64{1 / yMin, x.absMax() / (yMin * yMin)}, []Value{x, y})
}

// Sqrt refuses an argument whose interval reaches zero, where ∂/∂x = 1/(2√x)
// is unbounded, and one that goes negative, where the function is not real.
func Sqrt(x Value) Value {
	if x.lo() <= 0 {
		return Undefined("sqrt", fmt.Sprintf(
			"the argument's interval [%v, %v] reaches zero or below, where 1/(2√x) "+
				"is unbounded", x.lo(), x.hi()))
	}
	ref := math.Sqrt(x.Ref)
	return compose("sqrt", ref, roundoff(ref),
		[]float64{1 / (2 * math.Sqrt(x.lo()))}, []Value{x})
}

// Exp's sensitivity is itself, so it is taken at the interval's top.
//
// This is the operator that makes a wide input interval expensive, and that is
// the arithmetic rather than a pessimism here: exp genuinely amplifies an input
// error by its own value.
func Exp(x Value) Value {
	ref := math.Exp(x.Ref)
	top := math.Exp(x.hi())
	if math.IsInf(top, 0) {
		return Undefined("exp", fmt.Sprintf(
			"exp overflows over the argument's interval [%v, %v]", x.lo(), x.hi()))
	}
	return compose("exp", ref, roundoff(ref), []float64{top}, []Value{x})
}

// Log refuses an argument reaching zero, where 1/x is unbounded.
func Log(x Value) Value {
	if x.lo() <= 0 {
		return Undefined("log", fmt.Sprintf(
			"the argument's interval [%v, %v] reaches zero or below", x.lo(), x.hi()))
	}
	ref := math.Log(x.Ref)
	return compose("log", ref, roundoff(ref), []float64{1 / x.lo()}, []Value{x})
}

// Tanh's derivative is 1 - tanh², which is at most 1 everywhere, so its
// sensitivity needs no interval at all.
func Tanh(x Value) Value {
	ref := math.Tanh(x.Ref)
	return compose("tanh", ref, roundoff(ref), []float64{1}, []Value{x})
}

// SumValues adds terms with section 7's tree budget rather than pairwise
// [Add], which would charge γ(depth) per addition instead of once.
//
// depth is the additions on the longest path, as [SumBudget] takes it: K-1 for
// a loop and ⌈log2 K⌉ for a balanced tree.
func SumValues(xs []Value, depth int) Value {
	if len(xs) == 0 {
		return Constant(0)
	}
	ref, magnitude := 0.0, 0.0
	sens := make([]float64, len(xs))
	for i, x := range xs {
		if !x.Defined {
			return Undefined("sum", fmt.Sprintf("term %d (%s) is undefined: %s",
				i, x.Op, x.Reason))
		}
		ref += x.Ref
		magnitude += math.Abs(x.Ref)
		sens[i] = 1
	}
	local, ok := SumBudget(depth, magnitude)
	if !ok {
		return Undefined("sum", fmt.Sprintf(
			"γ(%d) is undefined at f32: an accumulation that deep has no useful "+
				"forward bound", depth))
	}
	return compose("sum", ref, local, sens, xs)
}

// Dot is section 7's dot product: the products, then the sum.
//
// Written as one operator rather than as [Mul] into [SumValues] because the
// products are exact-then-rounded once each and the sum's γ applies to their
// magnitudes -- composing it out of the pieces would charge the products'
// roundings twice.
func Dot(a, b []Value, depth int) Value {
	if len(a) != len(b) {
		return Undefined("dot", fmt.Sprintf("%d terms against %d", len(a), len(b)))
	}
	prods := make([]Value, len(a))
	for i := range a {
		prods[i] = Mul(a[i], b[i])
	}
	v := SumValues(prods, depth)
	if v.Defined {
		v.Op = "dot"
	}
	return v
}

// MaxValues is the largest reference, carrying the widest input error.
//
// The maximum is exact -- it selects rather than computes -- so it adds no
// rounding of its own. What it cannot do is pick correctly when two candidates'
// intervals overlap, and it does not pretend to: the error carried forward is
// the widest of the overlapping ones, which is the conservative answer.
func MaxValues(xs []Value) Value {
	if len(xs) == 0 {
		return Undefined("max", "no terms")
	}
	best := xs[0]
	err := 0.0
	for i, x := range xs {
		if !x.Defined {
			return Undefined("max", fmt.Sprintf("term %d (%s) is undefined: %s",
				i, x.Op, x.Reason))
		}
		if x.Ref > best.Ref {
			best = x
		}
		if x.Err > err {
			err = x.Err
		}
	}
	return Value{
		Ref: best.Ref, Err: err, Defined: true, Op: "max",
		terms: []term{{op: "widest input", sens: 1, err: err}},
	}
}

// RMSNormValue is the composed budget for one output of RMSNorm.
//
//	y_i = x_i · w_i / sqrt(mean(x²) + eps)
//
// specs/008-numerics.md section 8 requires the reduction and the transcendental
// to be composed rather than hidden in one tolerance, so the intermediates are
// built here out of the same operators a caller would use: the sum of squares
// is a [Dot] of x with itself, the mean is a [Div] by the count, and the scale
// is a [Div] by a [Sqrt].
//
// depth is the reduction's, as [SumBudget] takes it.
//
// eps is what keeps the sqrt away from zero, and it is why this composes at all
// for an input that is entirely zero -- without it the sqrt's interval reaches
// zero and section 8's refusal fires, which is the correct answer to a
// normalization with no norm.
func RMSNormValue(x []Value, w Value, i, depth int, eps float64) Value {
	if i < 0 || i >= len(x) {
		return Undefined("rmsnorm", fmt.Sprintf("output %d of %d", i, len(x)))
	}
	ss := Dot(x, x, depth)
	mean := Div(ss, Constant(float64(len(x))))
	scale := Sqrt(Add(mean, Constant(eps)))
	v := Div(Mul(x[i], w), scale)
	if v.Defined {
		v.Op = "rmsnorm"
	}
	return v
}

// SoftmaxValue is the composed budget for one output of a softmax.
//
//	y_i = exp(x_i - m) / Σ_j exp(x_j - m),   m = max_j x_j
//
// The max subtraction is not an optimization to be elided: it is what keeps the
// exponentials inside f32's range, and composing the budget without it makes
// [Exp] refuse on any input above about 88. That is the shape of the real
// kernel, so it is the shape of the budget.
func SoftmaxValue(x []Value, i, depth int) Value {
	if i < 0 || i >= len(x) {
		return Undefined("softmax", fmt.Sprintf("output %d of %d", i, len(x)))
	}
	m := MaxValues(x)
	ex := make([]Value, len(x))
	for j := range x {
		ex[j] = Exp(Sub(x[j], m))
	}
	v := Div(ex[i], SumValues(ex, depth))
	if v.Defined {
		v.Op = "softmax"
	}
	return v
}
