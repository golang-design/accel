// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package accel_test

import (
	"testing"

	"golang.design/x/accel"
)

// The run half of specs/062-backend-parity.md: every registered case, on both
// backends, compared under its own ceiling.
//
// parity_matrix_test.go holds the cases and the completeness gate and has no
// build tag, so what is missing from this matrix is knowable on a machine with
// no GPU. This file is the part that needs one.
func TestTheParityMatrixAgreesOnCPUAndMetal(t *testing.T) {
	metal := openMetal(t)
	cpu := parityOracle(t, metal)

	for _, c := range parityCases() {
		t.Run(c.name, func(t *testing.T) {
			onCPU := c.run(t, cpu)
			assertNotDegenerate(t, "the CPU backend", onCPU)
			onMetal := c.run(t, metal)
			assertNotDegenerate(t, "Metal", onMetal)
			compareParity(t, c, onCPU, onMetal)
		})
	}
}

// parityOracle opens the CPU backend configured to the device it is checking.
//
// The subgroup width is the configuration that matters and it is not optional.
// The CPU backend emulates subgroups at a width the caller chooses and defaults
// to four; this device executes some other width, and a reduction over more
// lanes than the emulated width sums a different number of partial results. The
// comparison would then be between two different computations, and it would
// fail for a reason that has nothing to do with either backend being wrong.
//
// A device whose width is a range fails rather than skips: the oracle emulates
// one width, so a range needs a different arrangement rather than a quieter
// test. specs/062-backend-parity.md section 4.1.
//
// Permissive rather than strict, and that is a decision rather than an
// inheritance: strict mode forces capability absence to model the portable
// intersection of a stated target set, and a case comparing against Metal wants
// the CPU's natural behaviour on the capabilities Metal actually reports.
// Strict mode's own agreement obligation is specs/006-backends.md's.
func parityOracle(t *testing.T, metal *accel.Device) *accel.Device {
	t.Helper()
	lim := metal.Limits()
	if lim.MinSubgroupSize != lim.MaxSubgroupSize {
		t.Fatalf("this device reports a subgroup width range of [%d, %d]; the oracle "+
			"emulates one width, so a varying one needs a different arrangement",
			lim.MinSubgroupSize, lim.MaxSubgroupSize)
	}
	cpu, err := accel.OpenCPU(accel.CPUOptions{SubgroupSize: lim.MinSubgroupSize})
	if err != nil {
		t.Fatalf("open the CPU oracle at subgroup width %d: %v", lim.MinSubgroupSize, err)
	}
	t.Cleanup(func() { _ = cpu.Close() })
	return cpu
}
