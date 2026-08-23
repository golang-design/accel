// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package quant_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"
)

// Every weight comes back within half a scale step, which is the whole claim
// the representation makes.
//
// Asserted per weight rather than in aggregate: an average error inside the
// bound would hide a single weight far outside it, and one badly represented
// weight in a projection matrix is exactly the failure that shows up as a model
// producing nonsense for one token in a thousand.
func TestEveryWeightIsWithinHalfAStep(t *testing.T) {
	for _, tc := range []struct {
		name string
		w    []float32
	}{{
		name: "a smooth distribution",
		w: func() []float32 {
			w := make([]float32, 200)
			for i := range w {
				w[i] = float32(math.Sin(float64(i)*0.31)) * 3
			}
			return w
		}(),
	}, {
		name: "one outlier per block",
		// The case the representation is worst at, and the reason the bound is
		// stated per block: an outlier sets the scale for its 31 neighbours, so
		// they are represented coarsely.
		w: func() []float32 {
			w := make([]float32, 128)
			for i := range w {
				w[i] = 0.01
				if i%quant.Int8Block == 0 {
					w[i] = 100
				}
			}
			return w
		}(),
	}, {
		name: "a length that is not a whole number of blocks",
		w:    []float32{1, -2, 3, -4, 5},
	}, {
		name: "all zeros",
		w:    make([]float32, 64),
	}, {
		name: "one block of zeros beside a block of values",
		w: func() []float32 {
			w := make([]float32, 64)
			for i := 32; i < 64; i++ {
				w[i] = float32(i)
			}
			return w
		}(),
	}, {
		name: "negatives only",
		w: func() []float32 {
			w := make([]float32, 40)
			for i := range w {
				w[i] = -float32(i) - 1
			}
			return w
		}(),
	}} {
		t.Run(tc.name, func(t *testing.T) {
			q, s := quant.Int8(tc.w)
			if len(q) != len(tc.w) {
				t.Fatalf("%d quants for %d weights", len(q), len(tc.w))
			}
			want := (len(tc.w) + quant.Int8Block - 1) / quant.Int8Block
			if len(s) != want {
				t.Fatalf("%d scales for %d weights, want %d", len(s), len(tc.w), want)
			}

			back := quant.Dequantize(q, s)
			for i := range tc.w {
				step := float64(s[i/quant.Int8Block].F32())
				got := math.Abs(float64(back[i] - tc.w[i]))
				// Half a step, with a hair of slack for the f16 scale itself
				// being rounded: the bound is about the quantization, and the
				// scale's own representation error is a separate term.
				if got > step/2*(1+1e-3) {
					t.Fatalf("weight %d was %v and came back %v: off by %v, and half a "+
						"step is %v", i, tc.w[i], back[i], got, step/2)
				}
				if q[i] > quant.Int8Max || q[i] < -quant.Int8Max {
					t.Fatalf("quant %d is %d, outside ±%d", i, q[i], quant.Int8Max)
				}
			}
		})
	}
}

// The clamp is load-bearing, and this is the input that proves it.
//
// A weight one ulp below the block's peak can round to 128, which wraps to -128
// in an int8 -- turning the largest weight in the block into the most negative
// one. Without the clamp this reconstructs with the wrong sign, and the error
// is twice the peak rather than half a step.
func TestTheClampCatchesTheWrapAround(t *testing.T) {
	// A block whose values sit right at the boundary: the peak, and values a
	// hair below it, so w/s lands near 127 from both sides.
	w := make([]float32, quant.Int8Block)
	peak := float32(1.0)
	for i := range w {
		w[i] = math.Float32frombits(math.Float32bits(peak) - uint32(i))
	}
	q, s := quant.Int8(w)
	for i := range q {
		if q[i] < 0 {
			t.Fatalf("weight %d is positive (%v) and quantized to %d, which is the sign "+
				"flip a missing clamp produces", i, w[i], q[i])
		}
	}
	back := quant.Dequantize(q, s)
	for i := range w {
		if back[i] < 0 {
			t.Fatalf("weight %d reconstructed as %v from a positive %v", i, back[i], w[i])
		}
	}
}

// The bound is computed from the inputs rather than tuned, and it holds.
//
// Checked against the exact dot product over the *original* weights, which is
// what the bound is about: how far a quantized product sits from the one a
// caller would have got unquantized.
func TestTheDotProductBoundHolds(t *testing.T) {
	const k = 256
	w := make([]float32, k)
	x := make([]float32, k)
	for i := range w {
		w[i] = float32(math.Sin(float64(i)*0.17)) * 2
		x[i] = float32(math.Cos(float64(i)*0.29)) * 3
	}
	q, s := quant.Int8(w)
	back := quant.Dequantize(q, s)

	var exact, quantized, magnitude float64
	for i := range w {
		exact += float64(w[i]) * float64(x[i])
		quantized += float64(back[i]) * float64(x[i])
		magnitude += math.Abs(float64(w[i]) * float64(x[i]))
	}
	bound := quant.Error(x, s)
	if got := math.Abs(quantized - exact); got > bound {
		t.Fatalf("the quantized dot product is %v from the exact one, and the derived "+
			"bound is %v", got, bound)
	}

	// And the bound is not vacuous. The denominator is the sum of term
	// *magnitudes*, not the dot product itself: these terms largely cancel, so
	// the result is small while the error the bound describes is not
	// proportional to it. Comparing an absolute error bound against a
	// cancelling sum would call a perfectly good bound vacuous -- which is what
	// the first version of this assertion did.
	//
	// Against the magnitude the ratio is what the representation promises:
	// roughly max|w| / (254·mean|w|), a fraction of a percent for a distribution
	// without a dominant outlier.
	if ratio := bound / magnitude; ratio > 0.02 {
		t.Errorf("the bound is %.2f%% of the computation's magnitude, which is far above "+
			"the ~1/254 per term the representation promises", ratio*100)
	} else {
		t.Logf("the bound is %.3f%% of the computation's magnitude", ratio*100)
	}
}

// A scale that underflows f16 reconstructs as zero rather than as NaN.
//
// The weights are below f16's smallest normal divided by 127, so no non-zero
// scale represents them. Zero is the honest answer; the alternative is a
// division by zero inside the quantizer.
func TestScalesThatUnderflowF16(t *testing.T) {
	w := make([]float32, quant.Int8Block)
	for i := range w {
		w[i] = 1e-12
	}
	q, s := quant.Int8(w)
	if s[0].F32() != 0 {
		t.Skipf("f16 represents %v as %v, so this input no longer underflows",
			w[0]/quant.Int8Max, s[0].F32())
	}
	for i := range q {
		if q[i] != 0 {
			t.Errorf("quant %d is %d against a zero scale", i, q[i])
		}
	}
	for i, v := range quant.Dequantize(q, s) {
		if v != 0 || math.IsNaN(float64(v)) {
			t.Errorf("weight %d reconstructed as %v, want zero", i, v)
		}
	}
}

// Dequantize is the inverse the kernels have to match, so it is exact about
// what it does: quant times scale, in f32, with the f16 scale widened.
func TestDequantizeIsTheReference(t *testing.T) {
	q := []int8{1, -2, 127, -127}
	s := []accel.Float16{accel.ToFloat16(0.5)}
	got := quant.Dequantize(q, s)
	want := []float32{0.5, -1, 63.5, -63.5}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("element %d is %v, want %v", i, got[i], want[i])
		}
	}
}
