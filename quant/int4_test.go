// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package quant_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"
)

// A group of one repeated value reconstructs exactly.
//
// specs/048-int4.md §6's first assertion. max == min makes the range zero, so a
// naive quantizer divides by zero and every code becomes NaN. A pruned or
// padded matrix really has such a group, and the honest reconstruction is the
// constant rather than zero.
func TestAConstantInt4GroupReconstructsExactly(t *testing.T) {
	for _, c := range []float32{0, 1, -2.5, 0.125} {
		w := make([]float32, quant.Int4Group)
		for i := range w {
			w[i] = c
		}
		packed, scales, zeros := quant.Int4Quantize(w)
		got := quant.Int4Dequantize(packed, scales, zeros, len(w))
		for i, v := range got {
			if v != c {
				t.Fatalf("a group of %v reconstructed element %d as %v", c, i, v)
			}
		}
	}
}

// The reconstruction is inside the bound the package derives.
//
// Per weight, against Int4ErrorBound over a single unit term, not a tolerance:
// a test that measured what a run produced would pass for a quantizer that got
// worse. The spans include two far from zero, where the stored zero point is
// large enough that its f16 rounding is the dominant error and the range over
// thirty specs/048-int4.md §3 states is exceeded -- which is why the bound is
// over the stored scale and zero rather than over the range.
func TestInt4ReconstructionIsInsideItsDerivedBound(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	for _, span := range []struct {
		name   string
		lo, hi float32
	}{
		{"centred on zero", -1, 1},
		{"clustered away from zero", 0.9, 1.0},
		{"all negative", -3, -2.75},
		{"asymmetric about zero", -0.25, 4},
		{"near a thousand", 1000, 1001},
		{"negative and far from zero", -2000, -1999},
	} {
		t.Run(span.name, func(t *testing.T) {
			w := make([]float32, 3*quant.Int4Group+17)
			for i := range w {
				w[i] = span.lo + rng.Float32()*(span.hi-span.lo)
			}
			// The ends of every group are present, so the clamp the bound
			// accounts for is reached rather than only allowed for.
			for g := 0; g*quant.Int4Group < len(w); g++ {
				w[g*quant.Int4Group] = span.lo
				w[min(g*quant.Int4Group+1, len(w)-1)] = span.hi
			}
			packed, scales, zeros := quant.Int4Quantize(w)
			got := quant.Int4Dequantize(packed, scales, zeros, len(w))

			one := []float32{1}
			for i := range w {
				g := i / quant.Int4Group
				// The code is rounded against the stored f16 scale and zero,
				// so rounding it costs half a step exactly. What is left is
				// the rounding of those two, which lands on the weights at
				// the ends of the group as a clamp, and the bound carries it.
				bound := quant.Int4ErrorBound(one, scales[g:g+1], zeros[g:g+1])
				if e := math.Abs(float64(w[i] - got[i])); e > bound {
					t.Fatalf("weight %d is off by %v against a bound of %v (scale %v, "+
						"zero %v)", i, e, bound, scales[g].F32(), zeros[g].F32())
				}
			}
		})
	}
}

// The dot-product bound holds where the zero point's rounding dominates, and
// the range-over-thirty figure it replaced does not.
//
// Weights in [1000, 1001] store z = -15000 for a z* of -15003.7: the f16
// spacing there is 8. The codes are rounded against the stored z, so a weight
// in the middle of the group is still within half a step -- but the group's
// largest weight computes to code 18.7, is clamped to 15, and reconstructs
// 0.25 low against a claimed 0.033. The activations are one on each group's
// two ends and zero elsewhere, so the product is exactly the clamped terms and
// the excess does not average away over the terms that were fine. A typical
// group is checked beside it, so the bound is seen to hold where the old one
// was right as well.
func TestInt4ErrorBoundHoldsWhereTheZeroPointRounds(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 17))
	for _, span := range []struct {
		name          string
		lo, hi        float32
		exceedsOldOne bool
	}{
		{"near a thousand", 1000, 1001, true},
		{"a typical group", 0.9, 1.0, false},
	} {
		t.Run(span.name, func(t *testing.T) {
			w := make([]float32, 2*quant.Int4Group+17)
			for i := range w {
				w[i] = span.lo + rng.Float32()*(span.hi-span.lo)
			}
			x := make([]float32, len(w))
			for g := 0; g*quant.Int4Group < len(w); g++ {
				lo, hi := g*quant.Int4Group, min(g*quant.Int4Group+1, len(w)-1)
				w[lo], w[hi] = span.lo, span.hi
				x[lo], x[hi] = 1, 1
			}
			packed, scales, zeros := quant.Int4Quantize(w)
			back := quant.Int4Dequantize(packed, scales, zeros, len(w))

			var exact, quantized, old float64
			termScales := make([]accel.Float16, len(x))
			termZeros := make([]accel.Float16, len(x))
			for i := range w {
				exact += float64(w[i]) * float64(x[i])
				quantized += float64(back[i]) * float64(x[i])
				old += math.Abs(float64(x[i])) * float64(span.hi-span.lo) / 30
				termScales[i] = scales[i/quant.Int4Group]
				termZeros[i] = zeros[i/quant.Int4Group]
			}
			bound := quant.Int4ErrorBound(x, termScales, termZeros)
			err := math.Abs(quantized - exact)
			if err > bound {
				t.Fatalf("the quantized product is %v from the exact one, and the derived "+
					"bound is %v", err, bound)
			}
			if span.exceedsOldOne && err <= old {
				t.Fatalf("the product is off by %v, inside the range-over-thirty figure "+
					"of %v; this span was chosen to exceed it, so the fixture does not "+
					"reach the rounding the bound exists to cover", err, old)
			}
			// And not vacuous: no weight is off by more than its group's
			// range, so a bound past that would be saying nothing.
			var ceiling float64
			for i := range x {
				ceiling += math.Abs(float64(x[i])) * float64(span.hi-span.lo)
			}
			if bound > ceiling {
				t.Fatalf("the bound %v exceeds the sum of every term's range %v", bound, ceiling)
			}
		})
	}
}

// Four asymmetric bits beat eight symmetric ones on a group clustered away from
// zero.
//
// specs/048-int4.md §3's table asserted rather than described. It is the whole
// argument for asymmetry, and it fails for a symmetric 4-bit form -- which
// would spend its codes on a range these weights never visit.
func TestInt4BeatsInt8OnAGroupClusteredAwayFromZero(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 5))
	w := make([]float32, quant.Int4Group)
	for i := range w {
		w[i] = 0.9 + rng.Float32()*0.1
	}

	p4, s4, z4 := quant.Int4Quantize(w)
	got4 := quant.Int4Dequantize(p4, s4, z4, len(w))

	q8, s8 := quant.Int8Quantize(w)
	got8 := quant.Int8Dequantize(q8, s8)

	var worst4, worst8 float64
	for i := range w {
		worst4 = max(worst4, math.Abs(float64(w[i]-got4[i])))
		worst8 = max(worst8, math.Abs(float64(w[i]-got8[i])))
	}
	if worst4 >= worst8 {
		t.Fatalf("on weights in [0.9, 1.0] the 4-bit form is off by %v and the 8-bit "+
			"form by %v; asymmetry is supposed to win here, because the symmetric "+
			"form spends its codes on a range these weights never visit",
			worst4, worst8)
	}
	t.Logf("worst error: int4 asymmetric %.3e, int8 symmetric %.3e", worst4, worst8)
}

// And it loses on a group centred on zero, which is the cost of the halving.
//
// Stated as a test rather than a caveat, so the tradeoff is a fact this package
// asserts about itself rather than a sentence in a document.
func TestInt4LosesToInt8OnAGroupCentredOnZero(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 17))
	w := make([]float32, quant.Int4Group)
	for i := range w {
		w[i] = -1 + rng.Float32()*2
	}

	p4, s4, z4 := quant.Int4Quantize(w)
	got4 := quant.Int4Dequantize(p4, s4, z4, len(w))
	q8, s8 := quant.Int8Quantize(w)
	got8 := quant.Int8Dequantize(q8, s8)

	var worst4, worst8 float64
	for i := range w {
		worst4 = max(worst4, math.Abs(float64(w[i]-got4[i])))
		worst8 = max(worst8, math.Abs(float64(w[i]-got8[i])))
	}
	if worst4 <= worst8 {
		t.Fatalf("on weights in [-1, 1] the 4-bit form is off by %v and the 8-bit by "+
			"%v; the 4-bit form is expected to be worse here and this test exists so "+
			"that stops being a claim in a document", worst4, worst8)
	}
}

// The packing round-trips at a length that is not a multiple of the word.
//
// The last word holds one weight and seven nibbles nobody wrote, which is why
// Int4Dequantize takes the count rather than deriving it from the word length.
func TestInt4PackingRoundTripsAtAnOddLength(t *testing.T) {
	w := make([]float32, quant.Int4Group+1)
	for i := range w {
		w[i] = float32(i%16) / 4
	}
	packed, scales, zeros := quant.Int4Quantize(w)
	if want := (len(w) + 7) / 8; len(packed) != want {
		t.Fatalf("%d weights packed into %d words, want %d", len(w), len(packed), want)
	}
	got := quant.Int4Dequantize(packed, scales, zeros, len(w))
	if len(got) != len(w) {
		t.Fatalf("reconstructed %d weights from %d", len(got), len(w))
	}
	// The last weight is the one the partly-filled word holds, and it is the
	// one an off-by-one in the packing loses.
	//
	// The bound is derived rather than chosen: specs/048-int4.md §3 states a
	// group's error as its range over 30, and the last weight is alone in its
	// own group, so its range is zero and it reconstructs *exactly*. A tuned
	// 0.2 stood here and was three orders of magnitude looser than the truth —
	// it would have passed a packing that lost the weight entirely and
	// substituted a neighbour.
	last := len(w) - 1
	if got[last] != w[last] {
		t.Fatalf("the last weight of an odd-length run is %v and reconstructed as %v; "+
			"alone in its group its range is zero, so it is exact",
			w[last], got[last])
	}
}

// The nibbles are where the quantizer put them.
//
// Asserted through Int4Nibble rather than by unpacking with a second copy of
// the shift: a packer and an unpacker that agree with each other about the
// wrong order round-trip perfectly, which is exactly the failure a kernel finds
// and a round-trip test does not.
func TestInt4PacksLowNibbleFirst(t *testing.T) {
	// Two weights whose codes must differ: the group's minimum takes 0 and its
	// maximum takes 15.
	w := make([]float32, quant.Int4Group)
	for i := range w {
		w[i] = 0.5
	}
	w[0] = 0  // the minimum -> code 0
	w[1] = 10 // the maximum -> code 15
	packed, _, _ := quant.Int4Quantize(w)

	if got := quant.Int4Nibble(packed, 0); got != 0 {
		t.Errorf("weight 0 is the group minimum and its code is %d, want 0", got)
	}
	if got := quant.Int4Nibble(packed, 1); got != quant.Int4Max {
		t.Errorf("weight 1 is the group maximum and its code is %d, want %d",
			got, quant.Int4Max)
	}
	// And word 0's low two nibbles are 0xF0: weight 0 in bits 0-3, weight 1 in
	// bits 4-7.
	if packed[0]&0xFF != 0xF0 {
		t.Fatalf("word 0's low byte is %#02x, want 0xF0 -- weight 0 belongs in the "+
			"lowest nibble and weight 1 in the next, and a kernel reading them the "+
			"other way round gets a plausible matrix", packed[0]&0xFF)
	}
}

// The bound refuses a term count that does not match, which its doc states as
// the contract: a per-group figure is not a per-term one, and the number a
// mismatched call would return is not a bound.
func TestInt4ErrorBoundNeedsOneScaleAndZeroPerTerm(t *testing.T) {
	three := []accel.Float16{accel.ToFloat16(1), accel.ToFloat16(1), accel.ToFloat16(1)}
	two := three[:2]
	for _, c := range []struct {
		name          string
		scales, zeros []accel.Float16
	}{
		{"two scales for three terms", two, three},
		{"two zeros for three terms", three, two},
	} {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("a bound over three terms was computed from " + c.name)
				}
			}()
			quant.Int4ErrorBound([]float32{1, 2, 3}, c.scales, c.zeros)
		})
	}
}
