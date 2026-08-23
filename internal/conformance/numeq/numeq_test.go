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

func TestExact(t *testing.T) {
	if r := numeq.Exact([]uint32{1, 2, 3}, []uint32{1, 2, 3}); !r.Equal {
		t.Errorf("identical sequences reported %v", r)
	}

	r := numeq.Exact([]uint32{1, 9, 3, 8}, []uint32{1, 2, 3, 4})
	if r.Equal {
		t.Fatal("differing sequences reported equal")
	}
	if r.FirstDiff != 1 {
		t.Errorf("FirstDiff = %d, want 1", r.FirstDiff)
	}
	if r.Got != "9" || r.Want != "2" {
		t.Errorf("reported got %q want %q at the first difference", r.Got, r.Want)
	}
	if r.Diffs != 2 {
		t.Errorf("Diffs = %d, want 2: one corrupted element reads differently from a wholly wrong result", r.Diffs)
	}
	msg := r.String()
	for _, want := range []string{"index 1", "got 9", "want 2", "2 of 4"} {
		if !strings.Contains(msg, want) {
			t.Errorf("message %q does not carry %q", msg, want)
		}
	}
}

func TestExactReportsLengthFirst(t *testing.T) {
	r := numeq.Exact([]byte{1, 2}, []byte{1, 2, 3})
	if r.Equal {
		t.Fatal("sequences of different lengths reported equal")
	}
	if r.FirstDiff != -1 {
		t.Errorf("FirstDiff = %d, want -1: there is no index to point at", r.FirstDiff)
	}
	if !strings.Contains(r.String(), "length 2, want 3") {
		t.Errorf("message %q does not report the length mismatch", r.String())
	}
}

// TestExactBits is why a value-wise comparison is not enough for the
// bit-preservation guarantee: it is wrong in both directions for floats.
func TestExactBits(t *testing.T) {
	bits := func(f float32) uint64 { return uint64(math.Float32bits(f)) }

	nan := float32(math.NaN())
	if r := numeq.ExactBits([]float32{nan}, []float32{nan}, bits); !r.Equal {
		t.Error("the same NaN compared unequal to itself, which a bit comparison must not do")
	}

	negZero := float32(math.Copysign(0, -1))
	r := numeq.ExactBits([]float32{negZero}, []float32{0}, bits)
	if r.Equal {
		t.Error("negative zero compared equal to positive zero, which is the conversion a backend could do silently")
	}
	if !strings.Contains(r.String(), "0x") {
		t.Errorf("message %q does not show the encodings", r.String())
	}

	if r := numeq.ExactBits([]float32{1}, []float32{1, 2}, bits); r.Equal || r.FirstDiff != -1 {
		t.Errorf("length mismatch reported %v", r)
	}
	if r := numeq.ExactBits([]float32{1, 2}, []float32{1, 2}, bits); !r.Equal {
		t.Errorf("identical floats reported %v", r)
	}
	if got := (numeq.Report{Equal: true}).String(); got != "equal" {
		t.Errorf("an equal report says %q", got)
	}
}

// The ULP distance is ordered-bit, which is what makes it meaningful across a
// binade boundary and around zero.
func TestULPDistance(t *testing.T) {
	next := func(v float32, n int32) float32 {
		return math.Float32frombits(uint32(int32(math.Float32bits(v)) + n))
	}
	cases := []struct {
		name string
		a, b float32
		want uint64
	}{
		{"identical", 1, 1, 0},
		{"one step", 1, next(1, 1), 1},
		// Below one the steps are half the size, so the neighbour on that side
		// is a different absolute distance and the same ordered distance. A
		// comparison in absolute terms would count these two cases differently.
		{"one step below a binade boundary", 1, next(1, -1), 1},
		{"across zero", next(0, 1), -next(0, 1), 2},
		{"the two zeros", 0, float32(math.Copysign(0, -1)), 0},
		{"a large gap", 1, 2, 1 << 23},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := numeq.ULPDistance(c.a, c.b); got != c.want {
				t.Errorf("ULPDistance(%v, %v) = %d, want %d", c.a, c.b, got, c.want)
			}
			if got := numeq.ULPDistance(c.b, c.a); got != c.want {
				t.Errorf("ULPDistance is not symmetric at %v, %v", c.a, c.b)
			}
		})
	}
}

// WithinULP accepts up to its ceiling, refuses past it, and never lets a NaN
// through.
func TestWithinULP(t *testing.T) {
	next := func(v float32, n int32) float32 {
		return math.Float32frombits(uint32(int32(math.Float32bits(v)) + n))
	}
	want := []float32{1, 2, 3}

	if r := numeq.WithinULP([]float32{1, next(2, 4), 3}, want, 4); !r.Equal {
		t.Errorf("four steps should be within a ceiling of four: %v", r)
	}
	if r := numeq.WithinULP([]float32{1, next(2, 5), 3}, want, 4); r.Equal {
		t.Error("five steps should exceed a ceiling of four")
	}
	// The case a tolerance comparison silently passes: NaN > anything is false,
	// so a NaN is inside every ceiling unless it is checked first.
	nan := float32(math.NaN())
	if r := numeq.WithinULP([]float32{1, nan, 3}, want, 1<<62); r.Equal {
		t.Error("a NaN passed a ceiling, so it is not being checked before the distance")
	}
	if r := numeq.WithinULP([]float32{1, nan, 3}, []float32{1, nan, 3}, 0); !r.Equal {
		t.Errorf("two NaNs in the same place agree: %v", r)
	}
	if r := numeq.WithinULP([]float32{1, 2}, want, 4); r.Equal {
		t.Error("slices of different lengths do not agree")
	}
}
