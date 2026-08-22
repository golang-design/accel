// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package numeq_test

import (
	"math"
	"testing"

	"golang.design/x/accel/internal/conformance/numeq"
)

// FuzzExactAgreesWithItself is a comparison function, so the thing to check is
// that it is a comparison: reflexive, symmetric, and never reporting equal for
// sequences that differ.
//
// A comparison that is wrong is worse than no comparison. Every numeric claim
// in this project is asserted through this package, so a false "equal" turns
// the whole suite into a set of tests that pass regardless.
func FuzzExactAgreesWithItself(f *testing.F) {
	f.Add([]byte{1, 2, 3}, []byte{1, 2, 3})
	f.Add([]byte{1, 2, 3}, []byte{1, 2, 4})
	f.Add([]byte{}, []byte{})
	f.Add([]byte{0}, []byte{})

	f.Fuzz(func(t *testing.T, a, b []byte) {
		if r := numeq.Exact(a, a); !r.Equal {
			t.Fatalf("a sequence is not equal to itself: %v", r)
		}

		forward := numeq.Exact(a, b)
		backward := numeq.Exact(b, a)
		if forward.Equal != backward.Equal {
			t.Fatalf("comparison is not symmetric: %v then %v", forward, backward)
		}

		if forward.Equal {
			if len(a) != len(b) {
				t.Fatalf("equal at different lengths %d and %d", len(a), len(b))
			}
			for i := range a {
				if a[i] != b[i] {
					t.Fatalf("reported equal while element %d differs: %d and %d", i, a[i], b[i])
				}
			}
			return
		}

		// It reported a difference, so it must point at one.
		if len(a) == len(b) {
			if forward.FirstDiff < 0 || forward.FirstDiff >= len(a) {
				t.Fatalf("FirstDiff is %d for sequences of length %d", forward.FirstDiff, len(a))
			}
			if a[forward.FirstDiff] == b[forward.FirstDiff] {
				t.Fatalf("FirstDiff %d names elements that match", forward.FirstDiff)
			}
			// It is the *first* difference, so everything before it matches.
			for i := range forward.FirstDiff {
				if a[i] != b[i] {
					t.Fatalf("element %d differs before the reported first difference at %d",
						i, forward.FirstDiff)
				}
			}
		}
		if forward.String() == "" {
			t.Fatal("a difference with no explanation")
		}
	})
}

// FuzzExactBitsSeparatesWhatValuesCannot is why the float comparison exists: a
// value-wise comparison calls two NaNs unequal and negative zero equal to
// positive zero, and both are wrong for a path that promises to preserve bits.
func FuzzExactBitsSeparatesWhatValuesCannot(f *testing.F) {
	f.Add(uint32(0), uint32(0x80000000))
	f.Add(uint32(0x7fc00000), uint32(0x7fc00000))
	f.Add(uint32(0x3f800000), uint32(0x3f800000))

	bits := func(x float32) uint64 { return uint64(math.Float32bits(x)) }

	f.Fuzz(func(t *testing.T, x, y uint32) {
		a := []float32{math.Float32frombits(x)}
		b := []float32{math.Float32frombits(y)}

		r := numeq.ExactBits(a, b, bits)
		if got, want := r.Equal, x == y; got != want {
			t.Fatalf("comparing %#08x and %#08x reported equal=%v, want %v", x, y, got, want)
		}
		// Reflexive even for a NaN, which a value comparison is not.
		if r := numeq.ExactBits(a, a, bits); !r.Equal {
			t.Fatalf("%#08x is not equal to itself under a bit comparison", x)
		}
	})
}
