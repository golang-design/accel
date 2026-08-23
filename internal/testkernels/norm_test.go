// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"fmt"
	"golang.design/x/accel/kernelabi"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/numeq"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
)

// runRow dispatches a row-wise kernel, one workgroup per row.
func runRow(t *testing.T, k *accel.Kernel, d testkernels.RowDims, slices []any) {
	t.Helper()
	err := kernel.DispatchCooperative(k, accel.ID3{X: d.Rows},
		kernelabi.Args{Slices: slices, Uniforms: []any{d}})
	if err != nil {
		t.Fatalf("dispatch %s: %v", k.Name, err)
	}
}

// RMSNorm matches a higher-precision reference under spec 008's budget, at
// widths that are not multiples of the workgroup.
//
// The reference is a straight loop at f64, written without the tree or the
// strided fold, so it shares neither the kernel's structure nor its bugs.
func TestRMSNormMatchesItsReference(t *testing.T) {
	for _, width := range []int{1, 63, 128, 129, 500, 1024} {
		t.Run(fmt.Sprintf("width=%d", width), func(t *testing.T) {
			const rows = 3
			d := testkernels.RowDims{Rows: rows, Width: uint32(width), Eps: 1e-5}

			x := make([]float32, rows*width)
			w := make([]float32, width)
			for i := range x {
				x[i] = float32(math.Sin(float64(i)) * 3)
			}
			for i := range w {
				w[i] = float32(math.Cos(float64(i))) + 1.5
			}
			out := make([]float32, rows*width)
			runRow(t, &testkernels.RMSNormKernel, d, []any{x, w, out})

			for r := range rows {
				var sumsq, magnitude float64
				for i := range width {
					v := float64(x[r*width+i])
					sumsq += v * v
					magnitude += v * v
				}
				scale := 1 / math.Sqrt(sumsq/float64(width)+float64(d.Eps))

				// The budget: the sum of squares carries the reduction's error,
				// and rsqrt is a bounded primitive. Composed by spec 008
				// section 8's forward propagation rather than guessed.
				depth := numeq.TreeDepth(min(width, 128)) + (width+127)/128 - 1
				sumBudget, ok := numeq.SumBudget(depth, magnitude)
				if !ok {
					t.Fatalf("no bound for depth %d", depth)
				}
				// A relative error in the sum of squares becomes half of one in
				// the reciprocal square root, plus rsqrt's own ceiling.
				rel := sumBudget/sumsq/2 + 1e-6

				for i := range width {
					want := float64(x[r*width+i]) * scale * float64(w[i])
					got := float64(out[r*width+i])
					if math.Abs(got-want) > math.Abs(want)*rel+1e-6 {
						t.Fatalf("row %d element %d is %v, want about %v (relative "+
							"budget %g)", r, i, got, want, rel)
					}
				}
			}
		})
	}
}

// Softmax matches its reference, and every row sums to one.
func TestSoftmaxMatchesItsReference(t *testing.T) {
	for _, width := range []int{1, 63, 128, 129, 500} {
		t.Run(fmt.Sprintf("width=%d", width), func(t *testing.T) {
			const rows = 3
			d := testkernels.RowDims{Rows: rows, Width: uint32(width)}

			x := make([]float32, rows*width)
			for i := range x {
				x[i] = float32(math.Sin(float64(i)) * 4)
			}
			out := make([]float32, rows*width)
			runRow(t, &testkernels.SoftmaxKernel, d, []any{x, out})

			for r := range rows {
				m := math.Inf(-1)
				for i := range width {
					m = math.Max(m, float64(x[r*width+i]))
				}
				var total float64
				for i := range width {
					total += math.Exp(float64(x[r*width+i]) - m)
				}
				var sum float64
				for i := range width {
					want := math.Exp(float64(x[r*width+i])-m) / total
					got := float64(out[r*width+i])
					if math.Abs(got-want) > 1e-5 {
						t.Fatalf("row %d element %d is %v, want %v", r, i, got, want)
					}
					sum += got
				}
				// The property that makes it a distribution, checked separately
				// from the element comparison: a kernel could match element by
				// element under a loose bound and still not sum to one.
				if math.Abs(sum-1) > 1e-4 {
					t.Errorf("row %d sums to %v, want 1", r, sum)
				}
			}
		})
	}
}

// The maximum subtraction is what keeps softmax finite on real logits.
//
// exp overflows f32 at about 88, and a logit of 100 is ordinary. Without the
// subtraction every term is Inf and the result is Inf/Inf, which is NaN — so
// this is not an accuracy refinement, it is the difference between a
// distribution and no answer at all.
func TestSoftmaxSurvivesLargeLogits(t *testing.T) {
	const width = 64
	d := testkernels.RowDims{Rows: 1, Width: width}
	x := make([]float32, width)
	for i := range x {
		x[i] = 100 + float32(i) // every one past exp's f32 range
	}
	out := make([]float32, width)
	runRow(t, &testkernels.SoftmaxKernel, d, []any{x, out})

	var sum float64
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("element %d is %v: without subtracting the row maximum every "+
				"exponential overflows and the quotient is NaN", i, v)
		}
		sum += float64(v)
	}
	if math.Abs(sum-1) > 1e-4 {
		t.Errorf("the row sums to %v, want 1", sum)
	}
	// The largest logit takes the largest share, which says the shift did not
	// change which element is which.
	for i := range out[:width-1] {
		if out[i] > out[i+1] {
			t.Fatalf("element %d is larger than %d, and the input is increasing", i, i+1)
		}
	}
}

// A row of zeros normalizes to zeros rather than dividing by zero, which is
// what epsilon is for.
func TestRMSNormHandlesAZeroRow(t *testing.T) {
	const width = 64
	d := testkernels.RowDims{Rows: 1, Width: width, Eps: 1e-5}
	x := make([]float32, width)
	w := make([]float32, width)
	for i := range w {
		w[i] = 1
	}
	out := make([]float32, width)
	runRow(t, &testkernels.RMSNormKernel, d, []any{x, w, out})

	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("element %d is %v: epsilon is what keeps a row of zeros from "+
				"dividing by zero", i, v)
		}
		if v != 0 {
			t.Errorf("element %d is %v, want 0", i, v)
		}
	}
}

// The authored halves, which is spec 004's fifth testing level.
//
// The invocations rendezvous for real, one goroutine each behind a cyclic
// barrier. Running every invocation through the whole function once per barrier
// is unsound for these kernels -- a tree overwrites the array it reduces, so a
// second pass reduces its own output -- and an earlier version of this test did
// exactly that and passed by luck. See [kernel.RunAuthored].
func TestAuthoredRowKernels(t *testing.T) {
	const width = 100 // not a multiple of the workgroup, so the tail is folded
	size := kernel.ID3{X: testkernels.RowWidth, Y: 1, Z: 1}

	drive := func(run func(th kernel.Thread, sh *[128]float32)) {
		var sh [128]float32
		kernelabi.Poison(sh[:])
		kernel.RunAuthored(size, kernel.ID3{}, kernel.ID3{X: 1}, testkernels.RowWidth,
			func(th kernel.Thread) { run(th, &sh) })
	}

	t.Run("RMSNorm", func(t *testing.T) {
		d := testkernels.RowDims{Rows: 1, Width: width, Eps: 1e-5}
		x := make([]float32, width)
		w := make([]float32, width)
		for i := range x {
			x[i] = float32(i%9) - 4
			w[i] = 1
		}
		authored := make([]float32, width)
		drive(func(th kernel.Thread, sh *[128]float32) {
			testkernels.RMSNorm(th, d, x, w, authored, sh)
		})

		generated := make([]float32, width)
		runRow(t, &testkernels.RMSNormKernel, d, []any{x, w, generated})
		for i := range authored {
			if math.Abs(float64(authored[i]-generated[i])) > 1e-5 {
				t.Fatalf("element %d: authored %v, generated %v",
					i, authored[i], generated[i])
			}
		}
	})

	t.Run("Softmax", func(t *testing.T) {
		d := testkernels.RowDims{Rows: 1, Width: width}
		x := make([]float32, width)
		for i := range x {
			x[i] = float32(i%13) - 6
		}
		authored := make([]float32, width)
		drive(func(th kernel.Thread, sh *[128]float32) {
			testkernels.Softmax(th, d, x, authored, sh)
		})

		generated := make([]float32, width)
		runRow(t, &testkernels.SoftmaxKernel, d, []any{x, generated})
		for i := range authored {
			if math.Abs(float64(authored[i]-generated[i])) > 1e-5 {
				t.Fatalf("element %d: authored %v, generated %v",
					i, authored[i], generated[i])
			}
		}
	})
}

// Both kernels' records carry the shared extent a backend needs at pipeline
// creation, and the number of suspension points in the source.
//
// In the *source*, not per execution: a barrier inside a loop is one suspension
// point however many times the loop runs, because the count is a property of
// the generated state machine rather than of a dispatch. That is why the
// scheduler's epoch bound cannot be derived from it — a barrier in a
// thousand-round loop needs a thousand epochs and still counts once here.
func TestTheRowKernelsDeclareTheirShape(t *testing.T) {
	for _, c := range []struct {
		k          *accel.Kernel
		suspends   int
		sharedSize int
	}{
		// The load barrier, and the one inside the tree loop.
		{&testkernels.RMSNormKernel, 2, testkernels.RowWidth},
		// Two trees: loads and tree for the maximum, the barrier separating the
		// passes, then loads and tree for the sum.
		{&testkernels.SoftmaxKernel, 5, testkernels.RowWidth},
	} {
		if got := c.k.Suspensions; got != c.suspends {
			t.Errorf("%s has %d suspension points, want %d", c.k.Name, got, c.suspends)
		}
		if len(c.k.SharedSizes) != 1 || c.k.SharedSizes[0] != c.sharedSize {
			t.Errorf("%s declares shared %v, want one array of %d",
				c.k.Name, c.k.SharedSizes, c.sharedSize)
		}
	}
}
