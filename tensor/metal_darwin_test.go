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

// A feed-forward block agrees between the backends, which is the operator set
// checked rather than one operator.
//
// The ceiling is 022's rule applied: RMSNorm reaches rsqrt and SiLU reaches exp,
// both bounded by specs/008-numerics.md section 6, doubled for two
// implementations. The GEMM reaches neither and would have to agree exactly on
// its own -- which the corpus differential already asserts, so what this adds is
// that composing them does not lose it.
func TestAFeedForwardBlockAgreesOnCPUAndMetal(t *testing.T) {
	const rows, width, hidden = 4, 128, 64

	xs := make([]float32, rows*width)
	gs := make([]float32, width)
	ws := make([]float32, width*hidden)
	for i := range xs {
		xs[i] = float32(math.Sin(float64(i))) * 2
	}
	for i := range gs {
		gs[i] = 1 + float32(i%5)/8
	}
	for i := range ws {
		ws[i] = float32(i%7)/4 - 0.75
	}

	run := func(t *testing.T, d *accel.Device) []float32 {
		t.Helper()
		rt, err := tensor.NewRuntime(d)
		if err != nil {
			t.Fatalf("runtime: %v", err)
		}
		b := rt.NewBuilder("ffn")
		x := tensor.Input(b, tensor.ValueDesc{
			Name: "x", DType: accel.F32, Shape: tensor.Shape{rows, width},
		})
		gain := tensor.Weight(b, tensor.ValueDesc{
			Name: "gain", DType: accel.F32, Shape: tensor.Shape{width},
		})
		xf16 := tensor.Input(b, tensor.ValueDesc{
			Name: "xf16", DType: accel.F16, Shape: tensor.Shape{rows, width},
		})
		w1 := tensor.Weight(b, tensor.ValueDesc{
			Name: "w1", DType: accel.F16, Shape: tensor.Shape{width, hidden},
		})
		// Normalize, project, activate: three kernels of three different
		// shapes -- a row reduction, a tiled GEMM, and an elementwise pass.
		tensor.Output(b, "normed", tensor.RMSNorm(b, x, gain, 1e-5))
		tensor.Output(b, "h", tensor.SiLU(b, tensor.MatMul(b, xf16, w1)))

		plan, err := b.Compile(rt, tensor.CompileOptions{Label: "ffn"})
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		normed := f32Buffer(t, d, "normed", make([]float32, rows*width))
		h := f32Buffer(t, d, "h", make([]float32, rows*hidden))
		f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
			"x": f32Buffer(t, d, "x", xs), "xf16": f16Buffer(t, d, "xf16", xs),
			"gain": f32Buffer(t, d, "gain", gs), "w1": f16Buffer(t, d, "w1", ws),
			"normed": normed, "h": h,
		}})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		out := make([]float32, rows*width+rows*hidden)
		if err := d.Queue().ReadBuffer(normed.Buffer, 0, out[:rows*width]); err != nil {
			t.Fatalf("readback: %v", err)
		}
		if err := d.Queue().ReadBuffer(h.Buffer, 0, out[rows*width:]); err != nil {
			t.Fatalf("readback: %v", err)
		}
		if err := plan.Close(); err != nil {
			t.Fatalf("plan close: %v", err)
		}
		if err := rt.Close(); err != nil {
			t.Fatalf("runtime close: %v", err)
		}
		return out
	}

	cpuDev, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open CPU: %v", err)
	}
	defer cpuDev.Close()
	cpu := run(t, cpuDev)
	gpu := run(t, openMetalRuntimeDevice(t))

	if r := numeq.WithinULP(gpu, cpu, 16); !r.Equal {
		t.Fatalf("a feed-forward block disagrees between backends: %v\n  the ceiling is "+
			"RMSNorm's rsqrt and SiLU's exp, doubled for two implementations", r)
	}
}

// A decode step agrees between the backends.
//
// The last thing M7 needs on both: a state mutation, the barrier the graph
// inferred between the write and the read, and the fused attention kernel --
// none of which the earlier differentials touch. The ceiling is the softmax
// inside attention, which reaches exp.
func TestADecodeStepAgreesOnCPUAndMetal(t *testing.T) {
	const qHeads, kvHeads, headDim, capacity = 4, 2, 8, 8

	qs := make([]float32, qHeads*headDim)
	ks := make([]float32, kvHeads*headDim)
	vs := make([]float32, kvHeads*headDim)
	for i := range qs {
		qs[i] = float32(math.Sin(float64(i))) * 1.5
	}
	for i := range ks {
		ks[i] = float32(math.Cos(float64(i))) * 1.5
		vs[i] = float32(i)/4 - 1
	}

	run := func(t *testing.T, d *accel.Device) []float32 {
		t.Helper()
		rt, err := tensor.NewRuntime(d)
		if err != nil {
			t.Fatalf("runtime: %v", err)
		}
		b := rt.NewBuilder("decode")
		tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
		q := tensor.Input(b, tensor.ValueDesc{
			Name: "q", DType: accel.F32, Shape: tensor.Shape{qHeads, headDim},
		})
		nk := tensor.Input(b, tensor.ValueDesc{
			Name: "nk", DType: accel.F32, Shape: tensor.Shape{1, kvHeads * headDim},
		})
		nv := tensor.Input(b, tensor.ValueDesc{
			Name: "nv", DType: accel.F32, Shape: tensor.Shape{1, kvHeads * headDim},
		})
		slot := tensor.Input(b, tensor.ValueDesc{
			Name: "slot", DType: accel.U32, Shape: tensor.Shape{1},
		})
		kc := tensor.NewState(b, tensor.StateDesc{
			Name: "kc", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
		})
		vc := tensor.NewState(b, tensor.StateDesc{
			Name: "vc", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
		})
		lengths := tensor.Input(b, tensor.ValueDesc{
			Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
		})
		tensor.Output(b, "out", tensor.Attention(b, q,
			tensor.ScatterRows(b, kc, nk, slot),
			tensor.ScatterRows(b, vc, nv, slot),
			tensor.AttentionOptions{Lengths: lengths, ScaleName: "scale"}))

		plan, err := b.Compile(rt, tensor.CompileOptions{Label: "decode"})
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		out := f32Buffer(t, d, "out", make([]float32, qHeads*headDim))
		f := plan.Submit(d.Queue(), tensor.Bindings{
			Buffers: map[string]accel.BufferView{
				"q": f32Buffer(t, d, "q", qs), "nk": f32Buffer(t, d, "nk", ks),
				"nv":   f32Buffer(t, d, "nv", vs),
				"slot": u32Buffer(t, d, "slot", []uint32{0}),
				"kc":   f32Buffer(t, d, "kc", make([]float32, capacity*kvHeads*headDim)),
				"vc":   f32Buffer(t, d, "vc", make([]float32, capacity*kvHeads*headDim)),
				"out":  out,
				"len":  u32Buffer(t, d, "len", []uint32{1}),
			},
			Scalars: map[string]tensor.ScalarValue{
				"scale": tensor.F32(float32(1 / math.Sqrt(headDim))),
			},
		})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		got := make([]float32, qHeads*headDim)
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

	if r := numeq.WithinULP(gpu, cpu, 16); !r.Equal {
		t.Fatalf("a decode step disagrees between backends: %v\n  the ceiling is the "+
			"softmax inside attention, which reaches exp", r)
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
