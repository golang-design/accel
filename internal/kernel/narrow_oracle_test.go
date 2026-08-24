// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel_test

import (
	"math"
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
	nearest := func(x float32) uint16 {
		var best uint16
		bestErr := math.Inf(1)
		found := false
		for bits := range 1 << 16 {
			b := uint16(bits)
			if (b&0x8000 != 0) != negative(x) {
				continue
			}
			v := float64(kernel.Float16FromBits(b).F32())
			if math.IsNaN(v) || math.IsInf(v, 0) {
				continue
			}
			e := math.Abs(v - float64(x))
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
