// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"golang.design/x/accel/kernelabi"
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/numeq"
	"golang.design/x/accel/internal/conformance/probe"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
)

// reduceDepth is the kernel's maximum addition depth for n terms: the strided
// fold each invocation does, plus the tree over the 128 partial sums.
//
// Computed rather than inferred from the term count, because spec 008 section 7
// is explicit that the harness does not infer "tree" from a name — the shape is
// the kernel's to declare and the budget follows from it.
func reduceDepth(n int) int {
	const w = testkernels.ReduceSumWidth
	perLane := (n + w - 1) / w
	fold := perLane - 1
	if fold < 0 {
		fold = 0
	}
	return fold + numeq.TreeDepth(w)
}

// reduce_sum matches its higher-precision reference under spec 008, at lengths
// that are **not** multiples of the workgroup size.
//
// That is spec 009's done criterion, and the lengths are the point: the partial
// final round is where an off-by-one in a reduction hides, and a test at 128
// and 256 would never see it.
func TestReduceSumMatchesItsReference(t *testing.T) {
	if p := probe.CPU(); !p.ExactAvailable() {
		t.Logf("this machine's profile is %v; the budget below is the same either "+
			"way, but a failure would need reading against it", p)
	}

	lengths := []int{
		1, 2, 3,
		127, 128, 129, // side by side around the width
		200, 255, 257,
		1000, 1023, 1025,
		4097, // several strided rounds, with a partial one
	}
	for _, n := range lengths {
		t.Run(lengthName(n), func(t *testing.T) {
			in := make([]float32, n)
			for i := range in {
				// Mixed magnitudes and signs, so the sum has cancellation: a
				// budget scaled by the sum of magnitudes is exactly what such a
				// case needs, and one scaled by the result would be wrong.
				in[i] = float32(math.Sin(float64(i)) * math.Pow(1.3, float64(i%17)))
			}
			out := make([]float32, 1)

			err := kernel.DispatchCooperative(&testkernels.ReduceSumKernel,
				accel.ID3{X: 1}, kernelabi.Args{Slices: []any{in, out}})
			if err != nil {
				t.Fatalf("dispatch: %v", err)
			}

			r := numeq.Sum(out[0], in, reduceDepth(n))
			if !r.OK() {
				t.Fatalf("length %d: %v", n, r)
			}
		})
	}
}

func lengthName(n int) string {
	switch {
	case n%testkernels.ReduceSumWidth == 0:
		return "exactly-a-multiple-of-the-width"
	default:
		return "not-a-multiple"
	}
}

// A tree is more accurate than a sequential loop, which is why spec 010
// specifies this shape. Stated as a comparison of the budgets, because the
// claim is about the bound rather than about a particular input.
func TestTheTreeBudgetIsSmallerThanTheSequentialOne(t *testing.T) {
	const terms = 128
	tree, ok := numeq.Gamma(numeq.TreeDepth(terms))
	if !ok {
		t.Fatal("a tree over 128 terms has a bound")
	}
	sequential, ok := numeq.Gamma(terms - 1)
	if !ok {
		t.Fatal("a sequential sum of 128 terms has a bound")
	}
	if tree >= sequential {
		t.Fatalf("the tree's factor %g is not below the sequential %g, so the shape "+
			"spec 010 specifies buys nothing", tree, sequential)
	}
	// Concretely: seven roundings against a hundred and twenty-seven.
	if ratio := sequential / tree; ratio < 15 {
		t.Errorf("the tree bound is only %gx tighter, want about 18x", ratio)
	}
}

// The budget scales with the sum of magnitudes rather than with the result,
// which is what makes it right under cancellation.
func TestTheBudgetScalesWithMagnitudeNotResult(t *testing.T) {
	// Two large terms that nearly cancel: the result is tiny and the rounding
	// error is not.
	terms := []float32{1e7, -1e7 + 1}
	r := numeq.Sum(1, terms, 1)
	if r.Magnitude < 1e7 {
		t.Errorf("the magnitude is %g, want about 2e7: it is the sum of absolute "+
			"values, which is what a cancelling sum needs", r.Magnitude)
	}
	if r.Budget <= 0 {
		t.Error("a cancelling sum has a non-zero budget")
	}
	// A budget scaled by the result would be about 1e-7 here, which no
	// implementation could meet.
	if r.Budget < math.Abs(r.Want)*1e-6 {
		t.Error("the budget looks like it was scaled by the result")
	}
}

// Past the point where depth times the unit roundoff reaches one, no bound
// exists, and that is a refusal rather than a pass.
func TestNoBoundExistsForAnUnboundedDepth(t *testing.T) {
	if _, ok := numeq.Gamma(1 << 24); ok {
		t.Error("2^24 roundings at f32 have no classical forward bound")
	}
	if _, ok := numeq.Gamma(-1); ok {
		t.Error("a negative depth is not a depth")
	}
	r := numeq.Sum(0, []float32{1}, 1<<25)
	if r.OK() {
		t.Error("a comparison with no bound must not report OK")
	}
	if got := r.String(); !strings.Contains(got, "no f32 error bound") {
		t.Errorf("the report should say a bound does not exist, got %q", got)
	}
}

// The tree depth is ceil(log2 n), asserted at the boundaries where an
// off-by-one would hide.
func TestTreeDepthIsCeilLog2(t *testing.T) {
	for _, c := range []struct{ n, want int }{
		{1, 0}, {2, 1}, {3, 2}, {4, 2}, {5, 3},
		{127, 7}, {128, 7}, {129, 8}, {1024, 10},
	} {
		if got := numeq.TreeDepth(c.n); got != c.want {
			t.Errorf("TreeDepth(%d) = %d, want %d", c.n, got, c.want)
		}
	}
}
