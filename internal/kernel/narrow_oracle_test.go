// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel_test

import (
	"math"
	"sort"
	"testing"

	"golang.design/x/accel/internal/kernel"
)

// ToFloat16 returns the nearest representable half, checked against an oracle
// that enumerates every one of them.
//
// # Why an oracle and not a table of cases
//
// A spot check is what this had, and it passed for years while the conversion
// halved the magnitude of every value in a band just below every other power of
// two. `ToFloat16(-1.9996898)` returned `-1`.
//
// The cause was one character: the rounded mantissa can carry out of ten bits,
// and the code OR-ed that carry into the exponent instead of adding it. Where
// the exponent's low bit was already set the OR did nothing, so the value kept
// its exponent with a zero mantissa — exactly half what it should have been.
// The comment above the line described the carry correctly; the code did not
// perform it.
//
// A case table cannot find that, because the band is narrow and nobody thinks
// to write a case inside it. Enumerating all 65536 halves and picking the
// nearest is a reference that does not depend on anyone guessing which input is
// interesting.
func TestToFloat16IsNearestAgainstAnOracle(t *testing.T) {
	// Sign is carried, never chosen: a conversion preserves it even for zero,
	// so the oracle only considers candidates of the same sign. Without that,
	// +0 and -0 are equidistant from 0 and the tie rule picks arbitrarily.
	negative := func(x float32) bool {
		return math.Signbit(float64(x))
	}
	// The finite halves of each sign, ascending by magnitude. Built once
	// rather than rescanned per input: the oracle used to be a linear sweep of
	// all 65536 bit patterns inside a loop over ~66000 inputs, which is 4.3
	// billion comparisons -- 127 seconds ordinarily and past the 10-minute
	// test timeout under -race, where it failed the whole package.
	//
	// **The candidate set is what makes this a change of cost and not of
	// meaning.** The magnitude of a half is monotone in its low 15 bits, so
	// the same candidates in the same order are searched either way; only the
	// search is. The result is compared against the same tie rule.
	byMagnitude := func(neg bool) []uint16 {
		var out []uint16
		for bits := range 1 << 15 {
			b := uint16(bits)
			if neg {
				b |= 0x8000
			}
			v := float64(kernel.Float16FromBits(b).F32())
			if math.IsNaN(v) || math.IsInf(v, 0) {
				continue
			}
			out = append(out, b)
		}
		return out
	}
	pos, neg := byMagnitude(false), byMagnitude(true)

	nearest := func(x float32) uint16 {
		cand := pos
		if negative(x) {
			cand = neg
		}
		target := math.Abs(float64(x))
		mag := func(b uint16) float64 {
			return math.Abs(float64(kernel.Float16FromBits(b).F32()))
		}
		// The first candidate whose magnitude is at least the target. Its
		// neighbour below is the only other possibility, so the answer is one
		// of two rather than one of 32768.
		i := sort.Search(len(cand), func(i int) bool { return mag(cand[i]) >= target })

		best, bestErr, found := uint16(0), math.Inf(1), false
		for _, j := range []int{i - 1, i} {
			if j < 0 || j >= len(cand) {
				continue
			}
			b := cand[j]
			e := math.Abs(mag(b) - target)
			// Ties to even, which is what IEEE round-to-nearest does and what
			// makes the answer unique.
			if !found || e < bestErr || (e == bestErr && b&1 == 0) {
				best, bestErr, found = b, e, true
			}
		}
		return best
	}

	var checked int
	for _, x := range sampleValues() {
		got := kernel.ToFloat16(x).Bits()
		want := nearest(x)
		if got != want {
			t.Fatalf("ToFloat16(%v) = 0x%04x (%v), want 0x%04x (%v)",
				x, got, kernel.Float16FromBits(got).F32(),
				want, kernel.Float16FromBits(want).F32())
		}
		checked++
	}
	if checked < 200 {
		t.Fatalf("only %d values were checked", checked)
	}
}

// sampleValues covers the ordinary range plus the bands where rounding carries
// out of the mantissa, which is where the bug lived.
func sampleValues() []float32 {
	var out []float32
	// Every f16 value, exactly — a conversion must be the identity on these.
	for bits := range 1 << 16 {
		h := kernel.Float16FromBits(uint16(bits))
		v := h.F32()
		if !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
			out = append(out, v)
		}
	}
	// And just below every power of two in range, where the mantissa rounds up
	// out of ten bits and has to carry. This is the band that was wrong.
	for e := -14; e <= 15; e++ {
		p := float32(math.Ldexp(1, e))
		for _, d := range []float64{1e-4, 1e-5, 1e-6, 2e-4, 5e-4} {
			out = append(out, p*float32(1-d), -p*float32(1-d))
		}
	}
	// A spread of ordinary values, so the common path is covered too.
	for i := range 400 {
		out = append(out, float32(math.Sin(float64(i)*0.11))*float32(int(1)<<(i%12)))
	}
	return out
}
