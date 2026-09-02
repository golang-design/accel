// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/kernelabi"
)

// The CPU backend's cooperative diagnostics are on in every mode and off only
// by request.
//
// specs/006-backends.md section 5 defines a CPU mode as a capability and
// limits profile; instrumentation is not part of it. Strict used to switch the
// checks off as a side effect, and the public options had no way to ask for
// them back, so a caller verifying portability under Strict lost the oracle.
// NoDiagnostics is the request, and it is the only thing that turns them off.
func TestCPUDiagnosticsAreOnInEveryModeAndOffByRequest(t *testing.T) {
	// Lane 1 returns before the barrier every other lane reaches, which is
	// the arrival mismatch the instrumentation exists to report.
	early := &kernelabi.Kernel{
		Name: "EarlyReturn", WorkgroupSize: accel.ID3{X: 4, Y: 1, Z: 1},
		Digest: "test:EarlyReturn", Generator: kernelabi.Version, Suspensions: 1,
		Cooperative: func(th accel.Thread, _ kernelabi.Args, f *kernelabi.Frame) bool {
			if th.LocalID().X == 1 {
				return false
			}
			if f.Pass == 0 {
				f.Pass = 1
				f.Barrier = kernelabi.BarrierID{Index: 0, Pos: "early.go:3:2"}
				return true
			}
			return false
		},
	}
	run := func(t *testing.T, opts accel.CPUOptions) error {
		t.Helper()
		d, err := accel.OpenCPU(opts)
		if err != nil {
			t.Fatalf("open: %v", err)
		}
		defer d.Close()
		p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: early, Label: "early"})
		if err != nil {
			t.Fatalf("pipeline: %v", err)
		}
		defer p.Close()
		r := d.NewRecorder()
		r.Dispatch(p, nil, nil, accel.WorkgroupCount{X: 1})
		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		return d.Queue().Submit(g).Wait()
	}

	for _, c := range []struct {
		name string
		opts accel.CPUOptions
	}{
		{"developer", accel.CPUOptions{}},
		{"strict", accel.CPUOptions{Mode: accel.CPUStrict, StrictTargets: []accel.Backend{accel.BackendVulkan}}},
	} {
		t.Run(c.name, func(t *testing.T) {
			err := run(t, c.opts)
			if err == nil || !strings.Contains(err.Error(), "barrier arrival mismatch") {
				t.Fatalf("an early return before a barrier should be reported under %s, got %v",
					c.name, err)
			}
			off := c.opts
			off.NoDiagnostics = true
			if err := run(t, off); err != nil {
				t.Fatalf("with NoDiagnostics the same kernel should run unchecked, got %v", err)
			}
		})
	}
}
