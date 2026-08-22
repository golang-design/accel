// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kmath_test

import (
	"math"
	"testing"

	"golang.design/x/accel/kmath"
)

// TestExactValues covers the results every target has to agree on.
//
// These bodies are the reference the generated lowerings are compared against,
// so a mistake here is a mistake in what "correct" means rather than in one
// backend. Testing them against hand-computed values rather than against
// math.Sqrt is the point: comparing a wrapper to what it wraps proves nothing.
func TestExactValues(t *testing.T) {
	for _, tc := range []struct {
		name string
		got  float32
		want float32
	}{
		{"Sqrt(0)", kmath.Sqrt(0), 0},
		{"Sqrt(1)", kmath.Sqrt(1), 1},
		{"Sqrt(4)", kmath.Sqrt(4), 2},
		{"Sqrt(0.25)", kmath.Sqrt(0.25), 0.5},
		{"RSqrt(1)", kmath.RSqrt(1), 1},
		{"RSqrt(4)", kmath.RSqrt(4), 0.5},
		{"Exp(0)", kmath.Exp(0), 1},
		{"Log(1)", kmath.Log(1), 0},
		{"Sin(0)", kmath.Sin(0), 0},
		{"Cos(0)", kmath.Cos(0), 1},
		{"Tanh(0)", kmath.Tanh(0), 0},
		{"Abs(-3)", kmath.Abs(-3), 3},
		{"Abs(3)", kmath.Abs(3), 3},
		{"Min(1,2)", kmath.Min(1, 2), 1},
		{"Min(2,1)", kmath.Min(2, 1), 1},
		{"Max(1,2)", kmath.Max(1, 2), 2},
		{"Max(2,1)", kmath.Max(2, 1), 2},
	} {
		if tc.got != tc.want {
			t.Errorf("%s = %v, want %v", tc.name, tc.got, tc.want)
		}
	}
}

// TestAbsIsExact checks the claim that puts Abs in the exact class: it clears a
// sign bit and touches nothing else, so it has no error to bound.
func TestAbsIsExact(t *testing.T) {
	for _, x := range []float32{0, 1, -1, 1e-30, -1e30, math.SmallestNonzeroFloat32, math.MaxFloat32} {
		got := kmath.Abs(x)
		if want := math.Float32bits(x) &^ (1 << 31); math.Float32bits(got) != want {
			t.Errorf("Abs(%v) = %#08x, want %#08x: it clears a sign bit and nothing else",
				x, math.Float32bits(got), want)
		}
	}
	// Negative zero becomes positive zero, which is the sign bit clearing and
	// not a special case.
	if got := kmath.Abs(float32(math.Copysign(0, -1))); math.Signbit(float64(got)) {
		t.Error("Abs(-0) kept its sign")
	}
}

// TestMinMaxPropagateNaN is the contract that makes these intrinsics rather
// than an if.
//
// An if-based implementation returns whichever operand the comparison happened
// to favour, because every comparison with a NaN is false. That is a silent
// wrong answer, and it is why the behaviour is stated rather than inherited.
func TestMinMaxPropagateNaN(t *testing.T) {
	nan := float32(math.NaN())
	for _, tc := range []struct {
		name string
		got  float32
	}{
		{"Min(NaN, 1)", kmath.Min(nan, 1)},
		{"Min(1, NaN)", kmath.Min(1, nan)},
		{"Max(NaN, 1)", kmath.Max(nan, 1)},
		{"Max(1, NaN)", kmath.Max(1, nan)},
	} {
		if !math.IsNaN(float64(tc.got)) {
			t.Errorf("%s = %v, want NaN: a comparison with NaN is false, so an if-based "+
				"implementation would return an operand instead", tc.name, tc.got)
		}
	}
}

// TestSpecialValues covers the results spec 008 section 4 names, since a kernel
// reaching one of these has to get the same answer on every backend.
func TestSpecialValues(t *testing.T) {
	inf, ninf := float32(math.Inf(1)), float32(math.Inf(-1))

	if got := kmath.Sqrt(inf); !math.IsInf(float64(got), 1) {
		t.Errorf("Sqrt(+Inf) = %v", got)
	}
	if got := kmath.Sqrt(-1); !math.IsNaN(float64(got)) {
		t.Errorf("Sqrt(-1) = %v, want NaN", got)
	}
	if got := kmath.Log(0); !math.IsInf(float64(got), -1) {
		t.Errorf("Log(0) = %v, want -Inf", got)
	}
	if got := kmath.Log(-1); !math.IsNaN(float64(got)) {
		t.Errorf("Log(-1) = %v, want NaN", got)
	}
	if got := kmath.Exp(ninf); got != 0 {
		t.Errorf("Exp(-Inf) = %v, want 0", got)
	}
	if got := kmath.Exp(inf); !math.IsInf(float64(got), 1) {
		t.Errorf("Exp(+Inf) = %v", got)
	}
	if got := kmath.RSqrt(0); !math.IsInf(float64(got), 1) {
		t.Errorf("RSqrt(0) = %v, want +Inf", got)
	}
	if got := kmath.Tanh(inf); got != 1 {
		t.Errorf("Tanh(+Inf) = %v, want 1", got)
	}
	if got := kmath.Tanh(ninf); got != -1 {
		t.Errorf("Tanh(-Inf) = %v, want -1", got)
	}
}

// TestIdentities checks each function against a relationship it must satisfy,
// which is evidence independent of any particular library's implementation.
func TestIdentities(t *testing.T) {
	const eps = 1e-5

	near := func(name string, got, want float32) {
		t.Helper()
		if d := math.Abs(float64(got - want)); d > eps {
			t.Errorf("%s: %v against %v, differing by %v", name, got, want, d)
		}
	}

	for _, x := range []float32{0.1, 0.5, 1, 2, 7, 100} {
		near("Sqrt squared", kmath.Sqrt(x)*kmath.Sqrt(x), x)
		near("RSqrt times Sqrt", kmath.RSqrt(x)*kmath.Sqrt(x), 1)
		near("sin^2 + cos^2", kmath.Sin(x)*kmath.Sin(x)+kmath.Cos(x)*kmath.Cos(x), 1)
		near("Tanh is odd", kmath.Tanh(-x), -kmath.Tanh(x))
	}

	// Log of Exp holds only where Exp does not overflow. An f32 exponential
	// saturates a little past 88, which is a property of the width rather than
	// of the implementation, and a kernel that exceeds it gets an infinity on
	// every backend rather than a wrong number on one.
	for _, x := range []float32{0.1, 1, 10, 80, 88} {
		near("Log of Exp", kmath.Log(kmath.Exp(x)), x)
	}
	if got := kmath.Exp(89); !math.IsInf(float64(got), 1) {
		t.Errorf("Exp(89) = %v; an f32 exponential saturates just past 88", got)
	}

	// Tanh is bounded by one, and reaches it exactly in f32 well before its
	// mathematical limit: the values either side of 1 are further apart than
	// tanh's remaining distance to it.
	for _, x := range []float32{0.1, 1, 2, 7} {
		if kmath.Abs(kmath.Tanh(x)) >= 1 {
			t.Errorf("Tanh(%v) = %v, which is not inside (-1, 1)", x, kmath.Tanh(x))
		}
	}
	if got := kmath.Tanh(100); got != 1 {
		t.Errorf("Tanh(100) = %v, want exactly 1: an f32 cannot hold a value nearer", got)
	}
}

// BenchmarkKMath records what each intrinsic costs on the reference path, which
// is what a CPU-backend kernel pays per call.
func BenchmarkKMath(b *testing.B) {
	for _, tc := range []struct {
		name string
		fn   func(float32) float32
	}{
		{"Sqrt", kmath.Sqrt},
		{"RSqrt", kmath.RSqrt},
		{"Exp", kmath.Exp},
		{"Log", kmath.Log},
		{"Sin", kmath.Sin},
		{"Tanh", kmath.Tanh},
		{"Abs", kmath.Abs},
	} {
		b.Run(tc.name, func(b *testing.B) {
			x := float32(1.5)
			b.ReportAllocs()
			for b.Loop() {
				x = tc.fn(1.5)
			}
			_ = x
		})
	}
}

// FuzzMinMaxAreTotal checks that Min and Max agree with each other and with
// ordering for every pair of inputs, including the ones a comparison cannot
// order.
func FuzzMinMaxAreTotal(f *testing.F) {
	f.Add(uint32(0), uint32(0x80000000))
	f.Add(uint32(0x7fc00000), uint32(0x3f800000))
	f.Add(uint32(0x7f800000), uint32(0xff800000))

	f.Fuzz(func(t *testing.T, a, b uint32) {
		x, y := math.Float32frombits(a), math.Float32frombits(b)
		lo, hi := kmath.Min(x, y), kmath.Max(x, y)

		if math.IsNaN(float64(x)) || math.IsNaN(float64(y)) {
			if !math.IsNaN(float64(lo)) || !math.IsNaN(float64(hi)) {
				t.Fatalf("Min/Max of (%v, %v) gave %v and %v, want NaN from both", x, y, lo, hi)
			}
			return
		}
		if lo > hi {
			t.Fatalf("Min(%v, %v) = %v exceeds Max = %v", x, y, lo, hi)
		}
		// Both results have to be one of the inputs: neither invents a value.
		if lo != x && lo != y {
			t.Fatalf("Min(%v, %v) = %v, which is neither operand", x, y, lo)
		}
		if hi != x && hi != y {
			t.Fatalf("Max(%v, %v) = %v, which is neither operand", x, y, hi)
		}
		// Swapping the operands cannot change the answer.
		if kmath.Min(y, x) != lo || kmath.Max(y, x) != hi {
			t.Fatalf("Min and Max are not symmetric for (%v, %v)", x, y)
		}
	})
}
