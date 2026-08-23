// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package probe_test

import (
	"math"
	"runtime"
	"strings"
	"testing"

	"golang.design/x/accel/internal/conformance/probe"
)

// The probe reports what this machine actually does, and every Go platform
// accel targets does all four.
//
// Asserting the expected answer rather than only that the probe ran: a probe
// that reported "nothing is available" everywhere would pass a
// self-consistency check and quietly disable every exact comparison downstream,
// which is the failure that looks like caution.
func TestTheCPUProfileIsWhatGoGuarantees(t *testing.T) {
	p := probe.CPU()
	if p.Arch != runtime.GOARCH {
		t.Errorf("Arch is %q, want %q", p.Arch, runtime.GOARCH)
	}
	if p.Backend != "cpu" {
		t.Errorf("Backend is %q", p.Backend)
	}
	for _, c := range []struct {
		name string
		got  bool
		why  string
	}{
		{"RoundToNearestEven", p.RoundToNearestEven,
			"the Go spec requires IEEE 754 arithmetic"},
		{"ContractionOff", p.ContractionOff,
			"Go permits fusing only where it does not change the result, and the " +
				"generated lowering emits explicit rounding points"},
		{"SubnormalsPreserved", p.SubnormalsPreserved,
			"Go does not flush to zero"},
		{"InfNaNProduced", p.InfNaNProduced,
			"Go produces infinities and NaNs rather than leaving them undefined"},
	} {
		if !c.got {
			t.Errorf("%s is false on %s, and %s", c.name, runtime.GOARCH, c.why)
		}
	}
	if !p.ExactAvailable() {
		t.Error("exact f32 comparison should be available on the CPU backend")
	}
}

// The tie cases discriminate in both directions, which is what makes the answer
// "nearest-even" rather than "always rounds one way on a tie".
//
// Checked here against hand-computed values so the probe's own reasoning is
// visible: if these change, the probe is measuring something other than what it
// claims.
func TestTheTieCasesAreRealTies(t *testing.T) {
	two24 := float32(math.Ldexp(1, 24))
	spacing := float32(math.Ldexp(1, 24)+2) - two24
	if spacing != 2 {
		t.Fatalf("f32 spacing at 2^24 is %v, want 2: the probe's ties are computed "+
			"from that spacing", spacing)
	}
	// 2^24+1 sits between 2^24 (even mantissa) and 2^24+2 (odd), so nearest-even
	// rounds down and half-away would round up.
	if got := add(two24, 1); got != two24 {
		t.Errorf("2^24+1 rounded to %v, want %v (round-half-away gives %v)",
			got, two24, two24+2)
	}
	// 2^24+3 sits between 2^24+2 (odd) and 2^24+4 (even), so nearest-even
	// rounds up. A machine that always rounded a tie down passes the case above
	// and fails this one.
	if got := add(two24, 3); got != two24+4 {
		t.Errorf("2^24+3 rounded to %v, want %v", got, two24+4)
	}
}

//go:noinline
func add(a, b float32) float32 { return a + b }

// The contraction probe's inputs distinguish a fused evaluation from an
// unfused one, which an earlier version's did not.
//
// This is the check that version would have failed: its low term was far below
// the ulp rather than exactly half of one, so both evaluations dropped it, the
// probe saw no difference, and it reported contraction on for a machine that
// does not contract.
func TestTheContractionProbeInputsDistinguish(t *testing.T) {
	x := float64(1) + math.Ldexp(1, -12)
	exact := x*x - 1                     // 2^-11 + 2^-24
	rounded := float64(float32(x*x)) - 1 // the product rounded to f32 first
	if exact == rounded {
		t.Fatal("the probe's inputs do not distinguish a fused evaluation from an " +
			"unfused one, so the probe measures nothing")
	}
	if want := math.Ldexp(1, -11) + math.Ldexp(1, -24); exact != want {
		t.Errorf("the exact product minus one is %v, want %v", exact, want)
	}
	if want := math.Ldexp(1, -11); rounded != want {
		t.Errorf("the rounded product minus one is %v, want %v", rounded, want)
	}
}

// ExactAvailable is the conjunction spec 008 section 3 lists, minus the two
// conditions that belong to a comparison rather than to the machine.
func TestExactAvailableNeedsBothMachineConditions(t *testing.T) {
	for _, c := range []struct {
		p    probe.Profile
		want bool
	}{
		{probe.Profile{RoundToNearestEven: true, ContractionOff: true}, true},
		{probe.Profile{RoundToNearestEven: true}, false},
		{probe.Profile{ContractionOff: true}, false},
		{probe.Profile{}, false},
		// Subnormals and inf/nan are reported and do not gate exactness: they
		// bear on which *domain* a comparison is valid over, which is the
		// caller's to decide per case.
		{probe.Profile{RoundToNearestEven: true, ContractionOff: true,
			SubnormalsPreserved: false}, true},
	} {
		if got := c.p.ExactAvailable(); got != c.want {
			t.Errorf("%+v: ExactAvailable is %v, want %v", c.p, got, c.want)
		}
	}
}

func TestAProfileDescribesItself(t *testing.T) {
	full := probe.Profile{Arch: "arm64", Backend: "cpu",
		RoundToNearestEven: true, ContractionOff: true,
		SubnormalsPreserved: true, InfNaNProduced: true}
	got := full.String()
	for _, want := range []string{"cpu/arm64", "round-to-nearest-even", "contraction-off"} {
		if !strings.Contains(got, want) {
			t.Errorf("%q should contain %q", got, want)
		}
	}
	// An absent property is named as absent rather than omitted, so a reader
	// of a profile can tell "not measured" from "measured false".
	none := probe.Profile{Arch: "x", Backend: "y"}
	if !strings.Contains(none.String(), "no-round-to-nearest-even") {
		t.Errorf("an absent property should be spelled out, got %q", none.String())
	}
}

// The probe reports the same thing every run, since a measurement that varies
// is one no test can be written against.
func TestTheProbeIsDeterministic(t *testing.T) {
	first := probe.CPU().String()
	for range 50 {
		if got := probe.CPU().String(); got != first {
			t.Fatalf("the probe reported %q and then %q", first, got)
		}
	}
}

// The probe detects arithmetic that differs, which is the thing it exists for
// and which measuring only this machine can never show.
//
// Each case substitutes one operation with a version behaving the way some
// other target might, and asserts the probe notices. Without these the failure
// branches are unreachable on any machine this code runs on, and a detector
// nobody has seen detect anything is a detector nobody should believe.
func TestTheProbeDetectsDifferentArithmetic(t *testing.T) {
	cases := []struct {
		name  string
		ops   func(probe.Ops) probe.Ops
		check func(probe.Profile) bool
		says  string
	}{{
		name: "round-half-away-from-zero",
		ops: func(o probe.Ops) probe.Ops {
			o.Add = func(a, b float32) float32 { return roundHalfAway(float64(a) + float64(b)) }
			return o
		},
		check: func(p probe.Profile) bool { return !p.RoundToNearestEven },
		says:  "a machine rounding ties away from zero",
	}, {
		name: "round-half-toward-zero",
		ops: func(o probe.Ops) probe.Ops {
			o.Add = func(a, b float32) float32 { return roundHalfToward(float64(a) + float64(b)) }
			return o
		},
		check: func(p probe.Profile) bool { return !p.RoundToNearestEven },
		says:  "a machine always rounding ties down, which passes the first tie case",
	}, {
		name: "subtraction rounding differently",
		ops: func(o probe.Ops) probe.Ops {
			o.Sub = func(a, b float32) float32 { return roundHalfAway(float64(a) - float64(b)) }
			return o
		},
		check: func(p probe.Profile) bool { return !p.RoundToNearestEven },
		says:  "a machine whose subtraction rounds differently from its addition",
	}, {
		name: "a fused multiply-add",
		ops: func(o probe.Ops) probe.Ops {
			// Contraction: the multiply keeps full precision and the following
			// subtraction rounds once, which is what fusing does.
			var pending float64
			o.Mul = func(a, b float32) float32 {
				pending = float64(a) * float64(b)
				return float32(pending)
			}
			o.Sub = func(a, b float32) float32 { return float32(pending - float64(b)) }
			return o
		},
		check: func(p probe.Profile) bool { return !p.ContractionOff },
		says:  "a machine that fuses a multiply into a following add",
	}, {
		name: "flush-to-zero",
		ops: func(o probe.Ops) probe.Ops {
			flush := func(v float32) float32 {
				if v != 0 && math.Abs(float64(v)) < math.SmallestNonzeroFloat32*(1<<23) {
					return 0
				}
				return v
			}
			o.Mul = func(a, b float32) float32 { return flush(a * b) }
			o.Ldexp = func(m float32, e int) float32 {
				return flush(float32(math.Ldexp(float64(m), e)))
			}
			return o
		},
		check: func(p probe.Profile) bool { return !p.SubnormalsPreserved },
		says:  "a machine flushing subnormals to zero",
	}, {
		name: "saturating overflow",
		ops: func(o probe.Ops) probe.Ops {
			o.Mul = func(a, b float32) float32 {
				v := float64(a) * float64(b)
				if math.IsInf(v, 0) || math.Abs(v) > math.MaxFloat32 {
					return float32(math.MaxFloat32) // saturates instead
				}
				return float32(v)
			}
			return o
		},
		check: func(p probe.Profile) bool { return !p.InfNaNProduced },
		says:  "a machine saturating on overflow rather than producing an infinity",
	}, {
		name: "no NaN from zero over zero",
		ops: func(o probe.Ops) probe.Ops {
			o.Div = func(a, b float32) float32 {
				if b == 0 {
					return 0 // undefined, and this target chose zero
				}
				return a / b
			}
			return o
		},
		check: func(p probe.Profile) bool { return !p.InfNaNProduced },
		says:  "a machine where zero over zero is not a NaN",
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := probe.Measure("test", "synthetic", c.ops(probe.GoOps()))
			if !c.check(p) {
				t.Fatalf("the probe did not notice %s: %v", c.says, p)
			}
			// And such a machine cannot offer exact comparison, which is the
			// consequence that matters downstream.
			if p.ExactAvailable() && (!p.RoundToNearestEven || !p.ContractionOff) {
				t.Error("exact comparison should not be available")
			}
		})
	}
}

// roundHalfAway and roundHalfToward round an exact double to f32 with a
// tie-breaking rule other than nearest-even.
//
// Written from the definition rather than by nudging the value, so the tie is
// resolved deliberately rather than as a side effect of a fudge factor.
func roundHalfAway(v float64) float32 {
	lo, hi, tie := neighbours(v)
	if !tie {
		return float32(v)
	}
	if v < 0 {
		return lo
	}
	return hi
}

func roundHalfToward(v float64) float32 {
	lo, hi, tie := neighbours(v)
	if !tie {
		return float32(v)
	}
	if v < 0 {
		return hi
	}
	return lo
}

// neighbours reports the f32 values bracketing v, and whether v is exactly
// between them.
func neighbours(v float64) (lo, hi float32, tie bool) {
	near := float32(v)
	if float64(near) == v {
		return near, near, false
	}
	if float64(near) > v {
		hi = near
		lo = math.Float32frombits(math.Float32bits(near) - 1)
	} else {
		lo = near
		hi = math.Float32frombits(math.Float32bits(near) + 1)
	}
	return lo, hi, v-float64(lo) == float64(hi)-v
}
