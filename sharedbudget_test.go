// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// A kernel asking for more shared memory than the device reports is refused,
// naming both numbers.
//
// specs/016-graph-execution.md's validation rule V11. It was stated and could
// not fire: `requirementsOf` never set `Requirements.SharedBytes`, and
// `Kernel.SharedSizes` held element counts with no element size, so the number
// the rule compares against the budget did not exist. A kernel over budget
// reached the device and failed there, in that backend's words — or on the CPU
// backend, not at all.
func TestAKernelOverTheSharedBudgetIsRefused(t *testing.T) {
	// Mimic a device with a small shared-memory budget, which is what that mode
	// is for: the CPU backend's own profiles report the portable floor of 16
	// KiB or more, and the corpus asks for a kilobyte, so neither would fire
	// the rule.
	base := openDevice(t)
	profile := accel.DeviceProfile{Info: base.Info()}
	profile.Info.Limits.MaxSharedMemoryBytes = 128
	d, err := accel.OpenCPU(accel.CPUOptions{Mode: accel.CPUMimic, Mimic: &profile})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	defer d.Close()

	k := &testkernels.ExchangeKernel
	if k.SharedBytes == 0 {
		t.Fatal("the kernel reports no shared bytes, so the rule has nothing to compare")
	}
	if k.SharedBytes <= 128 {
		t.Fatalf("the kernel asks for %d bytes, which fits the %d-byte budget under test",
			k.SharedBytes, 128)
	}

	_, err = d.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: k, Label: "over"})
	if err == nil {
		t.Fatal("a kernel asking for more shared memory than the device reports was accepted")
	}
	for _, want := range []string{"MaxSharedMemoryBytes", "128"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal should name %q so a caller can act on it, got %v",
				want, err)
		}
	}
}

// A kernel within the budget is accepted, so the rule is a ceiling rather than
// a refusal of every cooperative kernel.
func TestAKernelWithinTheSharedBudgetIsAccepted(t *testing.T) {
	d := openDevice(t)
	k := &testkernels.ExchangeKernel
	if got := d.Limits().MaxSharedMemoryBytes; got <= k.SharedBytes {
		t.Skipf("the device reports %d shared bytes and the kernel needs %d", got, k.SharedBytes)
	}
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: k, Label: "fits"})
	if err != nil {
		t.Fatalf("a kernel within the budget was refused: %v", err)
	}
	defer p.Close()
}
