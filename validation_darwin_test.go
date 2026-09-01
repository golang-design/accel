// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package accel_test

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

// The Metal render path passes Metal's own API validation.
//
// The validation layer is what catches a pipeline that does not match its
// pass: a stencil draw ran into a Depth32FloatStencil8 texture through a
// pipeline declaring Depth32Float and no stencil format, and without
// validation the picture came out right anyway. With it, the draw is an
// assertion that aborts the process.
//
// # Why a child process
//
// METAL_DEVICE_WRAPPER_TYPE is read when Metal.framework loads, which in this
// binary is the first test that opens a device -- so setting it from inside a
// test is too late whenever another test ran first, and would be too late
// silently. The test binary re-runs itself with the variable set, on the
// parity cases that draw with stencil, and requires the child to say that
// validation was enabled: a child that ran without it would pass for the
// wrong reason.
func TestStencilDrawsPassMetalAPIValidation(t *testing.T) {
	openMetal(t) // skip or fail as the job promised, before spending a process on it
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("locating the test binary: %v", err)
	}
	cmd := exec.Command(exe,
		"-test.run", "^TestTheParityMatrixAgreesOnCPUAndMetal$/stencil",
		"-test.count=1", "-test.v")
	cmd.Env = append(os.Environ(), "METAL_DEVICE_WRAPPER_TYPE=1")
	out, err := cmd.CombinedOutput()
	text := string(out)
	if !strings.Contains(text, "Metal API Validation Enabled") {
		t.Fatalf("the child did not report validation enabled, so it proves nothing:\n%s", text)
	}
	if err != nil {
		t.Fatalf("the stencil draws failed under Metal API validation: %v\n%s", err, text)
	}
	if !strings.Contains(text, "--- PASS: TestTheParityMatrixAgreesOnCPUAndMetal/the_stencil_operation") {
		t.Fatalf("no stencil case ran in the child:\n%s", text)
	}
}
