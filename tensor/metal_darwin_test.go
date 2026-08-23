// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package tensor_test

import (
	"math"
	"os"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/numeq"
	"golang.design/x/accel/tensor"
)

// The same tensor plan runs on the CPU backend and on Metal and agrees.
//
// This is specs/024-tensor-bringup.md's fourth testing level, and it is the one
// that makes the tensor layer's claim of backend independence checkable rather
// than architectural. Nothing above the device layer knows which backend it is
// on: the same builder code compiles against either runtime.
//
// The ceiling follows specs/022-msl-target.md's rule rather than inventing one.
// Add and Mul are exact on both backends, so they must agree bit for bit; SiLU
// reaches exp, which specs/008-numerics.md section 6 bounds at 4 ULP, doubled
// for two implementations and rounded up for the division that follows.
func TestATensorPlanAgreesOnCPUAndMetal(t *testing.T) {
	const n = 512

	xs := make([]float32, n)
	ws := make([]float32, n)
	gs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(math.Sin(float64(i))) * 3
		ws[i] = float32(math.Cos(float64(i)*0.5)) * 2
		gs[i] = float32(i%23) - 11
	}

	run := func(t *testing.T, d *accel.Device) []float32 {
		t.Helper()
		rt, err := tensor.NewRuntime(d)
		if err != nil {
			t.Fatalf("runtime: %v", err)
		}
		b := rt.NewBuilder("mlp")
		x := tensor.Input(b, value("x", n))
		w := tensor.Weight(b, value("w", n))
		g := tensor.Weight(b, value("gate", n))
		tensor.Output(b, "y", tensor.SwiGLU(b, g, tensor.Add(b, x, w)))

		plan, err := b.Compile(rt, tensor.CompileOptions{Label: "mlp"})
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		out := f32Buffer(t, d, "y", make([]float32, n))
		f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
			"x": f32Buffer(t, d, "x", xs), "w": f32Buffer(t, d, "w", ws),
			"gate": f32Buffer(t, d, "gate", gs), "y": out,
		}})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		got := make([]float32, n)
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}
		if err := plan.Close(); err != nil {
			t.Fatalf("plan close: %v", err)
		}
		if err := rt.Close(); err != nil {
			t.Fatalf("runtime close: %v", err)
		}
		return got
	}

	cpuDev, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open CPU: %v", err)
	}
	defer cpuDev.Close()
	cpu := run(t, cpuDev)
	gpu := run(t, openMetalRuntimeDevice(t))

	// SwiGLU reaches exp through SiLU: 4 ULP each way from correctly rounded,
	// plus the division, rounded up to 16 the way specs/022-msl-target.md
	// derives its ceilings.
	if r := numeq.WithinULP(gpu, cpu, 16); !r.Equal {
		t.Fatalf("the same plan disagrees between backends: %v\n  the ceiling is SiLU's "+
			"exp and division; Add and Mul are exact and must not need one", r)
	}

	nonzero := 0
	for _, v := range gpu {
		if v != 0 {
			nonzero++
		}
	}
	if nonzero < n/2 {
		t.Fatalf("only %d of %d outputs are non-zero, so the plan did not run", nonzero, n)
	}
}

func openMetalRuntimeDevice(t *testing.T) *accel.Device {
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
	// specs/006-backends.md section 7 and .github/workflows/ci-metal.yml.
	if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
		t.Fatalf("this job promises Metal and enumerated no adapter; diagnostics: %v",
			e.Diagnostics)
	}
	t.Skipf("no Metal adapter on this machine; diagnostics: %v", e.Diagnostics)
	return nil
}
