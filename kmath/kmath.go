// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package kmath is the scalar math a kernel may call.
//
// # Why a package and not a set of methods
//
// These are free functions rather than methods on a thread because they do not
// depend on the invocation. Keeping them in their own package is what lets the
// compiler's intrinsic table key on (import path, name) and reject a
// same-named function from anywhere else: the predecessor's table was keyed by
// bare name, so any user function called Sqrt lowered to the GPU builtin, which
// errors nowhere and computes something else.
//
// # Why the set is this small
//
// Every entry here has a normative per-operation domain and error ceiling in
// specs/008-numerics.md section 6. An operation with no bound is rejected
// rather than admitted with a tolerance somebody tuned until a test passed: a
// tolerance that was tuned is a tolerance that will be tuned again.
//
// Division is deliberately absent from the exact class for the same reason.
// SPIR-V specifies OpFDiv at 2.5 ULP rather than correctly rounded, and Metal's
// default floating-point mode may compute x/y as x * (1/y), so a division is
// bounded rather than exact even where addition is not.
//
// # These bodies are not what runs
//
// A kernel calling [Sqrt] is lowered to its target's own square root, and on
// the CPU backend to a generated call. The Go implementations here exist so an
// authored kernel package type-checks and so spec 004's fifth testing level can
// call the authored function directly. They are the reference, not the
// executable.
package kmath

import "math"

// Sqrt returns the square root of x.
//
// Correctly rounded on every target that has an IEEE square root, which is all
// of them: it is the one transcendental-adjacent operation the hardware is
// required to get exactly right.
func Sqrt(x float32) float32 { return float32(math.Sqrt(float64(x))) }

// RSqrt returns the reciprocal square root of x.
//
// It is a separate intrinsic rather than 1/Sqrt(x) because every target has a
// dedicated instruction for it that is faster and less accurate than the
// division would be, and pretending otherwise would either lose the speed or
// misstate the error.
func RSqrt(x float32) float32 { return float32(1 / math.Sqrt(float64(x))) }

// Exp returns e**x.
func Exp(x float32) float32 { return float32(math.Exp(float64(x))) }

// Log returns the natural logarithm of x.
func Log(x float32) float32 { return float32(math.Log(float64(x))) }

// Sin returns the sine of x, in radians.
func Sin(x float32) float32 { return float32(math.Sin(float64(x))) }

// Cos returns the cosine of x, in radians.
func Cos(x float32) float32 { return float32(math.Cos(float64(x))) }

// Tanh returns the hyperbolic tangent of x.
//
// It is admitted because an activation function needs it and because every
// target has one. Its bound is wider than the others', which spec 008 states
// rather than leaving to be discovered.
func Tanh(x float32) float32 { return float32(math.Tanh(float64(x))) }

// Abs returns the absolute value of x.
//
// Exact everywhere: it clears a sign bit and touches nothing else, so it is in
// the exact class rather than the bounded one.
func Abs(x float32) float32 { return float32(math.Abs(float64(x))) }

// Min and Max return the smaller and larger of two values.
//
// They are intrinsics rather than an if, because every target has a dedicated
// instruction and because their NaN behaviour is a contract rather than a
// consequence: these propagate a NaN, which an if-based implementation would
// not.
func Min(x, y float32) float32 {
	if math.IsNaN(float64(x)) || math.IsNaN(float64(y)) {
		return float32(math.NaN())
	}
	if x < y {
		return x
	}
	return y
}

// Max returns the larger of two values, propagating a NaN.
func Max(x, y float32) float32 {
	if math.IsNaN(float64(x)) || math.IsNaN(float64(y)) {
		return float32(math.NaN())
	}
	if x > y {
		return x
	}
	return y
}

// ToI32 converts a float to an int32, saturating, with a NaN becoming zero.
//
// specs/051-float-to-int.md. Go's `int32(x)` is undefined for a value int32
// cannot hold, and so are MSL's and SPIR-V's, so this is the spelling a kernel
// gets and the bare conversion is not the same thing.
//
// # The bounds are exact, and that is why they are written as floats
//
// -2³¹ is representable in f32 exactly, so `x <= -2147483648` is a clean test
// and the equal case is the answer. +2³¹ is representable too and int32's
// maximum is one less, so the upper test is against 2147483648 and returns
// MaxInt32 — testing against 2147483647 would compare against 2147483648 after
// rounding and admit a value that does not fit.
func ToI32(x float32) int32 {
	switch {
	case x != x: // NaN, and the only value not equal to itself
		return 0
	case x <= -2147483648:
		return math.MinInt32
	case x >= 2147483648:
		return math.MaxInt32
	}
	return int32(x)
}

// ToU32 converts a float to a uint32, saturating, with a NaN becoming zero.
//
// [ToI32]'s contract with a lower bound of zero. Negative values clamp to zero
// rather than wrapping, which is what the bare conversion would do on some
// targets and is the difference this function exists for.
func ToU32(x float32) uint32 {
	switch {
	case x != x:
		return 0
	case x <= 0:
		return 0
	case x >= 4294967296:
		return math.MaxUint32
	}
	return uint32(x)
}
