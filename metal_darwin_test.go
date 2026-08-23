// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package accel_test

import (
	"math"
	"os"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// openMetal opens the enumerated Metal adapter by id.
//
// It fails rather than skips, per specs/006-backends.md section 7: a job that
// promises a backend and finds no device is a failure, not a skip. A skip here
// would let the whole backend rot green on a machine that stopped reporting a
// GPU.
func openMetal(t *testing.T) *accel.Device {
	t.Helper()
	e := accel.Enumerate()
	for _, info := range e.Devices {
		if info.Backend != accel.BackendMetal {
			continue
		}
		d, err := accel.OpenDevice(info.ID)
		if err != nil {
			t.Fatalf("OpenDevice(%s): %v", info.Name, err)
		}
		t.Cleanup(func() { _ = d.Close() })
		return d
	}
	// A failure or a skip depending on what the job promised: see
	// specs/006-backends.md section 7 and the header of
	// .github/workflows/ci.yml. Tier 1 runs on three platforms and promises
	// only the CPU backend, so a hosted macOS runner with no usable GPU must
	// not turn it red; Tier 2 promises Metal and sets ACCEL_REQUIRE_METAL.
	if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
		t.Fatalf("this job promises Metal and enumerated no adapter; diagnostics: %v",
			e.Diagnostics)
	}
	t.Skipf("no Metal adapter on this machine; diagnostics: %v", e.Diagnostics)
	return nil
}

// The M6 end-to-end scenario, and the differential that makes the device an
// oracle: the same recorded graph runs on the CPU backend and on Metal, from
// the same generated kernel record, and the two agree bit for bit.
//
// Bit for bit is the right bar and a weaker one would hide the failure this
// test exists to catch. Addition of f32 is exact in the IEEE sense on both
// backends by specs/008-numerics.md, so any difference is a lowering bug --
// and the most likely such bug, a contracted multiply-add, moves results by
// about one part in 2^24. A tolerance of 1e-6 would pass straight over it.
//
// The graph is recorded once per device rather than shared, because a Graph
// belongs to the device that built it. What is shared is the kernel record,
// which is the thing under test: the CPU backend runs its Flat lowering and
// Metal compiles its MSL, both generated from one IR.
func TestTheSameGraphAgreesOnCPUAndMetal(t *testing.T) {
	const n = 4096

	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		// Values that are not round, so that a lowering which dropped a term or
		// swapped two bindings produces a visibly different number rather than
		// a plausible one.
		a[i] = float32(math.Sin(float64(i)) * 3.5)
		b[i] = float32(math.Cos(float64(i)*0.25) * 1.25)
	}

	run := func(t *testing.T, d *accel.Device) []float32 {
		t.Helper()
		storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst

		p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
			Kernel: &testkernels.AddKernel, Label: "add",
		})
		if err != nil {
			t.Fatalf("pipeline: %v", err)
		}
		defer p.Close()

		ba := newBuffer(t, d, "a", n, storage)
		bb := newBuffer(t, d, "b", n, storage)
		out := newBuffer(t, d, "out", n, storage)

		r := d.NewRecorder()
		// Upload through the graph rather than through the queue, so the
		// recorded plan carries the host writes and the scenario is the one
		// specs/009-sequencing.md names: upload, dispatch, readback.
		r.CopyToBuffer(whole(t, ba), a)
		r.CopyToBuffer(whole(t, bb), b)
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: whole(t, ba)},
			{Index: 1, Buffer: whole(t, bb)},
			{Index: 2, Buffer: whole(t, out)},
		}, accel.WorkgroupCount{X: (n + 63) / 64})

		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		if err := d.Queue().Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		return readback(t, d, out)
	}

	cpu := run(t, openDevice(t))
	gpu := run(t, openMetal(t))

	for i := range cpu {
		if cpu[i] != gpu[i] {
			t.Fatalf("element %d: the CPU backend produced %v (%#08x) and Metal %v (%#08x); "+
				"both lowerings come from one IR, so a disagreement is the transform's",
				i, cpu[i], math.Float32bits(cpu[i]), gpu[i], math.Float32bits(gpu[i]))
		}
	}
	// A test comparing two backends passes trivially if both produced nothing,
	// and a graph whose dispatch silently did not run would do exactly that.
	nonzero := 0
	for _, v := range gpu {
		if v != 0 {
			nonzero++
		}
	}
	if nonzero < n/2 {
		t.Fatalf("only %d of %d outputs are non-zero, so the dispatch did not run", nonzero, n)
	}
}

// A kernel outside the MSL subset is refused by name, and never falls back to
// the Go lowering.
//
// The fallback is the failure worth guarding. Running the CPU lowering on a
// device the caller selected specifically would be correct, fast enough not to
// notice, and would mean the GPU was never exercised -- so it would pass every
// test that compares results, which is every test that would otherwise catch it.
func TestMetalRefusesAKernelItCannotLower(t *testing.T) {
	if testkernels.ReduceSumKernel.MSL != "" {
		t.Skip("ReduceSum is now inside the MSL subset; this test needs one outside it")
	}
	d := openMetal(t)
	const n = 256
	storage := accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst

	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{
		Kernel: &testkernels.ReduceSumKernel, Label: "reduce",
	})
	if err != nil {
		if !strings.Contains(err.Error(), "ReduceSum") {
			t.Errorf("the refusal should name the kernel: %v", err)
		}
		return
	}
	defer p.Close()

	in := newBuffer(t, d, "in", n, storage)
	out := newBuffer(t, d, "out", 1, storage)
	r := d.NewRecorder()
	r.Dispatch(p, []accel.Binding{
		{Index: 0, Buffer: whole(t, in)},
		{Index: 1, Buffer: whole(t, out)},
	}, accel.WorkgroupCount{X: 1})

	g, buildErr := r.Build()
	if buildErr == nil {
		defer g.Close()
		buildErr = d.Queue().Submit(g).Wait()
	}
	if buildErr == nil {
		t.Fatal("a kernel with no MSL artifact ran on Metal, which means something fell " +
			"back to the Go lowering: the GPU was never exercised")
	}
	if !strings.Contains(buildErr.Error(), "ReduceSum") {
		t.Errorf("the refusal should name the kernel that cannot be lowered: %v", buildErr)
	}
	if !strings.Contains(buildErr.Error(), "MSL") {
		t.Errorf("the refusal should name the target, per specs/004-kernel-authoring.md: %v", buildErr)
	}
	t.Logf("refused: %v", buildErr)
}
