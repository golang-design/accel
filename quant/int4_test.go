// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package quant_test

import (
	"math"
	"math/rand/v2"
	"testing"

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

// The reconstruction is inside the bound the spec derives.
//
// Against (max-min)/30 per weight, not a tolerance: specs/048-int4.md §3
// derives it from rounding to nearest, and a test that measured what a run
// produced would pass for a quantizer that got worse.
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
	} {
		t.Run(span.name, func(t *testing.T) {
			w := make([]float32, 3*quant.Int4Group+17)
			for i := range w {
				w[i] = span.lo + rng.Float32()*(span.hi-span.lo)
			}
			packed, scales, zeros := quant.Int4Quantize(w)
			got := quant.Int4Dequantize(packed, scales, zeros, len(w))

			for g := 0; g*quant.Int4Group < len(w); g++ {
				lo := g * quant.Int4Group
				hi := min(lo+quant.Int4Group, len(w))
				lowest, highest := w[lo], w[lo]
				for _, v := range w[lo:hi] {
					lowest, highest = min(lowest, v), max(highest, v)
				}
				// Half a step, exactly, with no slack for the f16 rounding of
				// the scale and the zero point -- and that is not luck. The
				// quantizer stores both as f16 and reads them *back* before
				// quantizing against them, so the reconstruction uses the same
				// two numbers the quantizer used and the only error left is
				// rounding the code. A quantizer that used the f32 scale and
				// stored a rounded one would need slack here, which is the
				// second error Int8Quantize's comment says nobody accounted
				// for. This assertion is what would notice.
				bound := float64(highest-lowest) / 30
				for i := lo; i < hi; i++ {
					if e := math.Abs(float64(w[i] - got[i])); e > bound {
						t.Fatalf("weight %d in [%v, %v] is off by %v, and the bound "+
							"over that range is %v", i, lowest, highest, e, bound)
					}
				}
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

// The bound refuses a term count that does not match.
func TestInt4ErrorBoundNeedsOneRangePerTerm(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("a bound over three terms with two ranges was computed; a per-group " +
				"range bounds a dot product only where the caller says which group " +
				"each term came from")
		}
	}()
	quant.Int4ErrorBound([]float32{1, 2, 3}, []float32{1, 2})
}
