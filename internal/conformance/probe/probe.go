// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package probe measures what a backend's arithmetic actually does.
//
// # Why measure rather than assume
//
// specs/008-numerics.md defines exactness as a property of
// (class, domain, backend profile), not of the Go operator spelling. Every one
// of its six conditions for exact f32 arithmetic is a statement about the
// machine: that rounding is to nearest-even, that contraction is off, that
// subnormals survive. A test asserting an exact result without knowing them is
// asserting what the developer's laptop happens to do.
//
// specs/020-cooperative-atomics.md puts this first in its milestone for that
// reason: a reduction's budget is derived from what a probe found, and asserting
// first and measuring afterwards produces a test that passes locally and fails
// on CI — whose usual remedy is widening the tolerance until it passes
// everywhere, which is asserting nothing.
//
// # What it does not do
//
// It does not make anything exact. It reports what is available so that a test
// can decline a comparison it cannot justify, which is what
// specs/008-numerics.md means by numeq.Exact refusing.
package probe

import (
	"fmt"
	"math"
	"runtime"
	"strings"
)

// Profile is what one backend's f32 arithmetic was measured to do.
type Profile struct {
	// Arch and Backend say whose arithmetic this is, since the answer is a
	// property of the pair.
	Arch    string
	Backend string

	// RoundToNearestEven reports that the four basic operations round the way
	// IEEE 754 specifies. Condition 6 of specs/008-numerics.md section 3.
	RoundToNearestEven bool

	// ContractionOff reports that a multiply followed by an add was not fused.
	// Condition 5.
	ContractionOff bool

	// SubnormalsPreserved reports that a subnormal survives arithmetic rather
	// than being flushed to zero. Condition 2's precondition.
	SubnormalsPreserved bool

	// InfNaNProduced reports that overflow yields an infinity and 0/0 a NaN,
	// rather than being undefined.
	InfNaNProduced bool
}

// ExactAvailable reports whether class-A f32 arithmetic may be compared
// exactly on this profile, over the finite-normal domain.
//
// It is the conjunction specs/008-numerics.md section 3 lists, minus the two
// conditions that belong to a particular comparison rather than to the machine:
// whether the inputs are finite and whether the result is normal.
func (p Profile) ExactAvailable() bool {
	return p.RoundToNearestEven && p.ContractionOff
}

func (p Profile) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s/%s:", p.Backend, p.Arch)
	for _, c := range []struct {
		name string
		on   bool
	}{
		{"round-to-nearest-even", p.RoundToNearestEven},
		{"contraction-off", p.ContractionOff},
		{"subnormals", p.SubnormalsPreserved},
		{"inf/nan", p.InfNaNProduced},
	} {
		if c.on {
			fmt.Fprintf(&b, " %s", c.name)
		} else {
			fmt.Fprintf(&b, " no-%s", c.name)
		}
	}
	return b.String()
}

// Ops is the arithmetic a probe measures.
//
// It is a parameter rather than the operators written inline, so a test can
// supply arithmetic that rounds the other way and check that the probe reports
// it. Without that, the probe's failure branches are unreachable on any machine
// this code runs on -- and a detector nobody has seen detect anything is a
// detector nobody should believe.
type Ops struct {
	Add, Sub, Mul, Div func(a, b float32) float32
	Ldexp              func(m float32, e int) float32

	// MulAdd evaluates a*b+c the way this backend's lowering emits it, as one
	// unit rather than as a multiply whose result is handed back and fed to an
	// add.
	//
	// # Why this cannot be composed from Mul and Add
	//
	// Contraction is a decision a compiler makes *within* one expression. On the
	// CPU each of these is a Go function call and the composition happens in
	// this package, so Mul followed by Add can never fuse -- which is fine,
	// because that is also what the generated lowering does. On a GPU each call
	// is a separate kernel and a separate compilation, so the composition
	// cannot fuse there either, however the device is configured. A probe built
	// that way reports contraction off on a backend that contracts every kernel
	// it compiles, which is a confident wrong answer of exactly the kind
	// specs/008-numerics.md exists to prevent.
	//
	// So the probe asks the backend to evaluate the whole expression. What comes
	// back says whether the code this backend *generates* fuses.
	MulAdd func(a, b, c float32) float32
}

// GoOps is this machine's own arithmetic, reached through functions the
// compiler cannot fold.
//
// The indirection is not ceremony. Go evaluates constant expressions in exact
// arithmetic and rounds once at the conversion, so a probe written with
// constants measures the constant folder and reports the same answer on
// hardware that rounds differently.
func GoOps() Ops {
	return Ops{Add: add32, Sub: sub32, Mul: mul32, Div: div32, Ldexp: ldexp32, MulAdd: muladd32}
}

// CPU measures this machine's Go arithmetic, which is what the CPU backend's
// generated lowerings run on.
func CPU() Profile { return Measure(runtime.GOARCH, "cpu", GoOps()) }

// Measure runs the probes against a given arithmetic.
func Measure(arch, backend string, ops Ops) Profile {
	return Profile{
		Arch:                arch,
		Backend:             backend,
		RoundToNearestEven:  roundsToNearestEven(ops),
		ContractionOff:      contractionOff(ops),
		SubnormalsPreserved: subnormalsPreserved(ops),
		InfNaNProduced:      infNaNProduced(ops),
	}
}

// roundsToNearestEven checks the tie-breaking rule on each of the four
// operations.
//
// # Two things this has to avoid
//
// **Ties are the whole question.** Every rounding mode agrees on a value that
// is not exactly between two representable ones, so a probe using ordinary
// inputs measures nothing. These inputs each land exactly halfway, where
// round-to-nearest-even picks the neighbour with an even mantissa and
// round-half-away picks the larger.
//
// **The inputs must be runtime values, not constants.** Go evaluates constant
// expressions in exact arithmetic and rounds once at the conversion, so a probe
// written with constants measures the compiler rather than the machine — and
// would report the same answer on hardware that rounds differently. Hence the
// package-level variables: the compiler cannot fold what it cannot see the
// value of.
func roundsToNearestEven(ops Ops) bool {
	// f32 spacing at 2^24 is 2, so the odd integers there are exact ties.
	two24 := ops.Ldexp(1, 24)

	// 2^24 + 1 is halfway between 2^24 and 2^24+2. The first has an even
	// mantissa, so nearest-even keeps it and half-away would give 2^24+2.
	if ops.Add(two24, one) != two24 {
		return false
	}
	// 2^24 + 3 is halfway between 2^24+2 (odd mantissa) and 2^24+4 (even), so
	// nearest-even rounds *up* here. A machine that always rounded a tie down
	// would pass the case above and fail this one.
	if ops.Add(two24, three) != ops.Add(two24, four) {
		return false
	}
	// Subtraction lands on the same boundary from the other side.
	if ops.Sub(ops.Add(two24, two), one) != two24 {
		return false
	}
	// Multiplication and division are not probed separately. Constructing a
	// product that lands on a tie *and* discriminates between the modes takes
	// operands that are themselves awkward to write down, and the rounding mode
	// is one FPU setting rather than four: a machine rounding sums to nearest
	// even and products some other way is not a machine this code can be made
	// to work on anyway. The two cases above already discriminate in both
	// directions, which is the evidence that matters.
	return true
}

// The probe's inputs and operations, kept out of the constant folder's reach.
//
// Package-level variables and no-inline functions: a compiler that can see both
// operands folds the expression at build time in exact arithmetic, which is
// precisely the measurement this must not make.
var (
	one   = float32(1)
	two   = float32(2)
	three = float32(3)
	four  = float32(4)
)

//go:noinline
func add32(a, b float32) float32 { return a + b }

//go:noinline
func sub32(a, b float32) float32 { return a - b }

//go:noinline
func mul32(a, b float32) float32 { return a * b }

//go:noinline
func ldexp32(m float32, e int) float32 { return float32(math.Ldexp(float64(m), e)) }

// muladd32 is a*b+c as the generated CPU lowering writes it.
//
// The intermediate conversion is not decoration. Go permits x*y+z to be fused
// into one FMA, and on arm64 it is -- so this without the conversion would
// measure the Go compiler's licence rather than the lowering's behaviour. The
// generated code names the rounding for exactly this reason
// (specs/008-numerics.md's rounding points), and this mirrors it, so the probe
// answers a question about the code that actually runs.
//
//go:noinline
func muladd32(a, b, c float32) float32 { return float32(a*b) + c }

// contractionOff checks that a multiply followed by an add was not fused into
// one correctly rounded operation.
//
// The inputs are chosen so the two differ: the product's rounding error is what
// the add would otherwise absorb. If they agree, either the machine fused it or
// this pair no longer distinguishes the two, and the honest answer to both is
// that contraction cannot be shown to be off.
func contractionOff(ops Ops) bool {
	// x = 1 + 2^-12, so x*x is exactly 1 + 2^-11 + 2^-24.
	//
	// The choice of exponent is the whole probe. At 1 an f32 ulp is 2^-23, so
	// the 2^-24 term is exactly half an ulp -- a tie, which nearest-even
	// resolves by dropping it, since 1 + 2^-11 has the even mantissa. So a
	// rounded product loses that term and the following subtraction cannot
	// recover it, while a fused multiply-add keeps it: at 2^-11 an ulp is
	// 2^-34, so 2^-11 + 2^-24 is exactly representable.
	//
	// An earlier version used 1 + 2^-23, whose square's low term is far below
	// the ulp rather than exactly half of one. Both evaluations dropped it, the
	// probe saw no difference, and it reported contraction *on* for a machine
	// that does not contract. A probe whose inputs do not distinguish the two
	// cases measures nothing and says so confidently, which is worse than not
	// probing.
	x := ops.Add(one, ops.Ldexp(1, -12))
	fused := float32(float64(x)*float64(x) - 1)
	return ops.MulAdd(x, x, -1) != fused
}

// subnormalsPreserved checks that a subnormal survives rather than being
// flushed to zero.
func subnormalsPreserved(ops Ops) bool {
	smallest := ops.Ldexp(1, -149) // the smallest positive f32
	if smallest == 0 {
		return false
	}
	// And that arithmetic *producing* one does not flush it either, which is
	// the case a flush-to-zero machine differs on: it can represent a subnormal
	// and still refuse to produce one.
	return ops.Mul(ops.Ldexp(1, -148), half) == smallest
}

var half = float32(0.5)

// infNaNProduced checks that overflow yields an infinity and 0/0 a NaN, rather
// than being undefined.
func infNaNProduced(ops Ops) bool {
	big := float32(math.MaxFloat32)
	if !math.IsInf(float64(ops.Mul(big, two)), 1) {
		return false
	}
	nan := ops.Div(zero, zero)
	return nan != nan
}

var zero = float32(0)

//go:noinline
func div32(a, b float32) float32 { return a / b }
