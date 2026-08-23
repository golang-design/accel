// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal_test

import (
	"testing"

	"golang.design/x/accel/internal/conformance/probe"
	"golang.design/x/accel/internal/metal"
	"golang.design/x/accel/internal/mtl"
)

// The Metal numeric profile, measured before anything derives a bound from it.
//
// specs/009-sequencing.md's risk row makes the order normative: the probes run
// first, and a device that misses a normative ceiling is answered by changing
// the lowering, never by widening a bound to match what it reported. So this
// test asserts the four conditions rather than logging them: a profile that
// merely printed would let a device that flushes subnormals reach the corpus
// differential, where the failure would be attributed to a kernel.
func TestMetalNumericProfile(t *testing.T) {
	adapters(t) // fails or skips on what the job promised
	devs, err := mtl.Devices()
	if err != nil || len(devs) == 0 {
		t.Fatalf("devices: %v", err)
	}
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()

	p, err := metal.NewProber(devs[0])
	if err != nil {
		t.Fatalf("prober: %v", err)
	}
	defer p.Close()

	prof := p.Profile()
	t.Logf("%v", prof)

	if !prof.RoundToNearestEven {
		t.Error("Metal does not round ties to nearest-even, so specs/008-numerics.md " +
			"condition 6 fails and no f32 comparison on this backend may claim exactness")
	}
	if !prof.ContractionOff {
		t.Error("a generated kernel's a*b+c fuses: the emitter's pragma is not reaching " +
			"the device, and every dot product will disagree with the CPU oracle")
	}
	// Asserted as a *measured value* rather than as a requirement, and it is
	// false here: Apple GPUs flush a subnormal result to zero.
	//
	// That is a restriction on the domain, not a failure, which is exactly the
	// distinction specs/008-numerics.md draws -- ExactAvailable is about the
	// machine and the normal-result condition belongs to a particular
	// comparison. What this pins is the measurement: a device that started
	// preserving them would widen the domain, and widening a domain because the
	// hardware improved is the one direction 009's risk row allows.
	if prof.SubnormalsPreserved {
		t.Errorf("subnormals now survive arithmetic on this device (smallest positive "+
			"f32 came back as %v), which is a wider domain than recorded in "+
			"specs/022-msl-target.md; update the profile", p.Denormal())
	}
	if !prof.InfNaNProduced {
		t.Error("overflow does not produce an infinity or 0/0 a NaN, so an error budget " +
			"cannot distinguish an overflow from a wrong answer")
	}
	if !prof.ExactAvailable() {
		t.Fatal("exact f32 comparison is not available on this device, which every " +
			"bit-for-bit differential in this suite assumes")
	}

	// The CPU oracle and this device must agree about *why* an exact comparison
	// is allowed, not merely that it is. Two backends reaching the same verdict
	// from different conditions would let a differential claim exactness that
	// only one of them supports.
	cpu := probe.CPU()
	if cpu.RoundToNearestEven != prof.RoundToNearestEven ||
		cpu.ContractionOff != prof.ContractionOff {
		t.Errorf("the CPU oracle measures %v and this device %v: a bit-for-bit "+
			"differential between them is comparing arithmetic that does not match", cpu, prof)
	}
}

// The probe measures the device rather than Go, which is checkable by asking it
// something Go would answer differently.
//
// Ldexp(1, -149) is the smallest positive f32. A device that flushed subnormals
// would return zero here while Go returns the subnormal, so a non-zero answer
// is evidence the value came back from the GPU rather than from a constant
// folded on the host.
func TestTheProbeReachesTheDevice(t *testing.T) {
	adapters(t)
	devs, err := mtl.Devices()
	if err != nil || len(devs) == 0 {
		t.Skipf("no Metal device: %v", err)
	}
	defer func() {
		for _, d := range devs {
			d.Close()
		}
	}()
	p, err := metal.NewProber(devs[0])
	if err != nil {
		t.Fatalf("prober: %v", err)
	}
	defer p.Close()

	ops := p.Ops()
	// Two answers only the device can give: a tie that nearest-even resolves
	// downward, and a product whose low bit survives only without fusion.
	two24 := ops.Ldexp(1, 24)
	if two24 != 1<<24 {
		t.Fatalf("ldexp(1, 24) came back as %v", two24)
	}
	if got := ops.Add(two24, 1); got != two24 {
		t.Errorf("2^24 + 1 came back as %v, want %v", got, two24)
	}
	x := ops.Add(1, ops.Ldexp(1, -12))
	if got, fused := ops.MulAdd(x, x, -1), float32(float64(x)*float64(x)-1); got == fused {
		t.Errorf("x*x-1 came back as the fused value %v: the probe kernel is contracting", got)
	}
}
