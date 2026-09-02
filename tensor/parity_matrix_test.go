// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/parity"
	"golang.design/x/accel/tensor"
)

// The tensor operator set's CPU/Metal parity matrix, per
// specs/062-backend-parity.md section 6.6.
//
// The gate is here rather than in the darwin file for the reason section 2
// gives: it compares the exported operator set, read out of the source, against
// what the cases say they cover, and that comparison needs no device. An
// operator added without a case is a red build on Linux.
//
// Why an enumeration was needed here in particular: before this, five tests
// compared the two backends over composites -- an MLP, a feed-forward block, a
// decode step, a sampled token, and one graph of "the remaining operators" --
// and between them they reached about eight operators by name. The rest were
// incidental, so nobody could say which of the forty had ever been compared.

// tensorParityCase is one plan, run on both backends.
type tensorParityCase struct {
	name    string
	covers  parity.Covers
	ceiling parity.Ceiling
	run     func(t *testing.T, d *accel.Device) []float32
}

func tensorParityCases() []tensorParityCase {
	return []tensorParityCase{
		elementwiseParityCase(),
		viewParityCase(),
		matmulParityCase(),
		int4ParityCase(),
		quantParityCase(),
		groupedParityCase(),
		attentionParityCase(),
		linearAttentionParityCase(),
		samplingParityCase(),
		rowSamplingParityCase(),
	}
}

// Every exported operator has a parity case.
//
// The universe is every exported function whose first parameter is a *Builder,
// which is what an operator is in this package and what nothing else is. The
// port declarations -- Input, Weight, Output, Scalar, NewState -- are in it and
// are covered rather than excluded: a declaration that reached the wrong buffer
// changes the compared result, so a case using one does compare it.
func TestEveryTensorOperatorHasAParityCase(t *testing.T) {
	pkg, err := parity.Open(".")
	if err != nil {
		t.Fatalf("open the package: %v", err)
	}
	ops, err := parity.Funcs(pkg, "*Builder")
	if err != nil {
		t.Fatalf("enumerate the operators: %v", err)
	}

	var covered parity.Covers
	for _, c := range tensorParityCases() {
		if len(c.covers) == 0 {
			t.Errorf("parity case %q covers no operator", c.name)
		}
		if err := c.ceiling.Validate(); err != nil {
			t.Errorf("parity case %q: %v", c.name, err)
		}
		covered = append(covered, c.covers...)
	}
	for _, err := range parity.Check(ops, parity.Set{
		Surface: "the tensor operator set", Covered: covered,
	}) {
		t.Error(err)
	}
}

// Every case runs on the CPU backend and produces something worth comparing.
func TestEveryTensorParityCaseProducesAResultOnTheCPU(t *testing.T) {
	d := openCPU(t)
	for _, c := range tensorParityCases() {
		t.Run(c.name, func(t *testing.T) {
			assertNotAllZero(t, "the CPU backend", c.run(t, d))
		})
	}
}

func openCPU(t *testing.T) *accel.Device {
	t.Helper()
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("OpenCPU: %v", err)
	}
	t.Cleanup(func() { _ = d.Close() })
	return d
}

// assertNotAllZero refuses a result two backends could agree on for the wrong
// reason. Two buffers of zeros are equal, and a plan that never ran produces
// one.
func assertNotAllZero(t *testing.T, who string, got []float32) {
	t.Helper()
	if len(got) == 0 {
		t.Fatalf("%s produced nothing, so comparing it proves nothing", who)
	}
	for _, v := range got {
		if v != 0 {
			return
		}
	}
	t.Fatalf("%s produced %d values and every one is zero, so an agreement here "+
		"would be two empty results agreeing", who, len(got))
}

// parityRuntime builds a runtime on a caller's device.
//
// Separate from newRuntime, which opens a device of its own: a parity case is
// handed the device to run on, because running the two backends is the point.
func parityRuntime(t *testing.T, d *accel.Device) *tensor.Runtime {
	t.Helper()
	rt, err := tensor.NewRuntime(d)
	if err != nil {
		t.Fatalf("runtime on %v: %v", d.Info().Backend, err)
	}
	t.Cleanup(func() { _ = rt.Close() })
	return rt
}

// runPlan compiles, submits and reads one output back.
func runPlan(t *testing.T, d *accel.Device, rt *tensor.Runtime, b *tensor.Builder,
	label, out string, n int, bindings tensor.Bindings) []float32 {
	t.Helper()
	plan, err := b.Compile(rt, tensor.CompileOptions{Label: label})
	if err != nil {
		t.Fatalf("compile %s on %v: %v", label, d.Info().Backend, err)
	}
	defer func() {
		if err := plan.Close(); err != nil {
			t.Errorf("close %s: %v", label, err)
		}
	}()

	dst := f32Buffer(t, d, out, make([]float32, n))
	if bindings.Buffers == nil {
		bindings.Buffers = map[string]accel.BufferView{}
	}
	bindings.Buffers[out] = dst
	if err := plan.Submit(d.Queue(), bindings).Wait(); err != nil {
		t.Fatalf("submit %s on %v: %v", label, d.Info().Backend, err)
	}
	got := make([]float32, n)
	if err := d.Queue().ReadBuffer(dst.Buffer, 0, got); err != nil {
		t.Fatalf("read %s on %v: %v", label, d.Info().Backend, err)
	}
	return got
}

// ramp is a deterministic input that has no symmetry to hide a transposition.
func ramp(n int, phase float64) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(math.Sin(float64(i)*0.29 + phase))
	}
	return out
}
