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

// Contiguous runs on Metal, which it could not until the emitter learned to
// spell a std140 array member.
//
// accel issue 19. Pack was the only kernel in the corpus with no MSL artifact,
// because its uniform block holds two eight-element arrays and std140 gives an
// array member a sixteen-byte stride whatever its element type. Nothing above
// the emitter knew, so tensor.Contiguous was documented without qualification,
// compiled on a Metal device, and failed at plan compile -- after the weights
// were on the card.
//
// The shape here is the consumer's: slice the last row of a hidden state and
// pack it, which is the LM-head case and the one with no alternative. Running
// the head over every position instead costs T times the vocabulary, which for
// a 4B model at two thousand positions is more memory than the model.
func TestContiguousRunsOnMetal(t *testing.T) {
	const rows, cols = 4, 8
	d := openMetalRuntimeDevice(t)
	rt, err := tensor.NewRuntime(d)
	if err != nil {
		t.Fatalf("runtime: %v", err)
	}
	b := rt.NewBuilder("contiguous")
	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F32, Shape: tensor.Shape{rows, cols},
	})
	last := tensor.Slice(b, x, 0, rows-1, rows)
	tensor.Output(b, "y", tensor.Contiguous(b, last))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "contiguous"})
	if err != nil {
		t.Fatalf("compile on Metal: %v", err)
	}
	defer plan.Close()

	src := make([]float32, rows*cols)
	for i := range src {
		src[i] = float32(math.Sin(float64(i) * 0.5))
	}
	out := f32Buffer(t, d, "y", make([]float32, cols))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f32Buffer(t, d, "x", src), "y": out,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, cols)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}
	// Exactly: a pack moves values and computes nothing.
	for i := range got {
		if want := src[(rows-1)*cols+i]; got[i] != want {
			t.Fatalf("element %d is %v, want %v: a pack copies, so this is equality",
				i, got[i], want)
		}
	}
}

// The operators 025 lists that no other cross-backend test reaches: the views,
// Softmax, RoPE and GatherRows.
//
// specs/025-tensor-operators.md §6's first Done bullet says "every operator
// above builds, infers, lowers, and runs on both backends", and the audit on
// 2026-08-27 found that half of it rested on the corpus differential — which
// compares *kernels*, not the operators that select and drive them. The
// difference is real: an operator that computed the wrong grid, materialized the
// wrong view, or bound its operands in the wrong order would pass every kernel
// differential and fail here.
//
// Composed into one graph on purpose. Each operator alone is a shape the corpus
// already covers; what this adds is that a view feeding a kernel, and a kernel
// feeding a view, survive the trip on both backends.
func TestTheRemainingOperatorsAgreeOnCPUAndMetal(t *testing.T) {
	const rows, width = 4, 32
	const tableRows = 6

	table := make([]float32, tableRows*width)
	for i := range table {
		table[i] = float32(math.Sin(float64(i) * 0.13))
	}
	ids := []uint32{4, 0, 5, 1}
	pos := []uint32{0, 1, 2, 3}
	bias := make([]float32, rows*width)
	for i := range bias {
		bias[i] = float32(i%7) * 0.25
	}

	run := func(t *testing.T, d *accel.Device) []float32 {
		t.Helper()
		rt, err := tensor.NewRuntime(d)
		if err != nil {
			t.Fatalf("runtime: %v", err)
		}
		b := rt.NewBuilder("ops")
		tensor.Scalar(b, tensor.ScalarDesc{Name: "theta", Kind: tensor.ScalarF32})

		tbl := tensor.Weight(b, tensor.ValueDesc{
			Name: "table", DType: accel.F32, Shape: tensor.Shape{tableRows, width}})
		idv := tensor.Input(b, tensor.ValueDesc{
			Name: "ids", DType: accel.U32, Shape: tensor.Shape{rows}})
		posv := tensor.Input(b, tensor.ValueDesc{
			Name: "pos", DType: accel.U32, Shape: tensor.Shape{rows}})
		bv := tensor.Input(b, tensor.ValueDesc{
			Name: "bias", DType: accel.F32, Shape: tensor.Shape{rows, width}})

		// GatherRows feeding an elementwise add: a kernel's result used as an
		// operand. Broadcast is deliberately not here -- this version
		// materializes only a contiguous run repeated whole, so a broadcast of
		// [width] to [rows, width] is refused, and that refusal is 025's to
		// cover rather than something to work around in a backend agreement test.
		g := tensor.Add(b, tensor.GatherRows(b, tbl, idv), bv)

		// RoPE over the gathered rows, reshaped to the [tokens, heads, dim] the
		// operator takes and back again — a kernel feeding a view.
		r := tensor.Reshape(b, g, tensor.Shape{rows, 1, width})
		r = tensor.RoPE(b, r, width/2, "theta", posv)
		r = tensor.Reshape(b, r, tensor.Shape{rows, width})

		// A transposed operand materialized explicitly, which is the boundary
		// 025 draws: a copy of a matrix is asked for, never implied.
		tr := tensor.Contiguous(b, tensor.Transpose(b, r, 0, 1))

		// Softmax over the last axis of the transposed thing, then a slice.
		// Axis 1 is the last of the transposed [width, rows], and it is stated
		// rather than defaulted: Softmax refuses any other, so a default that
		// happened to be right would be an accident this test relied on.
		sm := tensor.Softmax(b, tr, tensor.SoftmaxOptions{Axis: 1})
		tensor.Output(b, "y", tensor.Contiguous(b, tensor.Slice(b, sm, 0, 1, width)))

		plan, err := b.Compile(rt, tensor.CompileOptions{Label: "ops"})
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		n := (width - 1) * rows
		out := f32Buffer(t, d, "y", make([]float32, n))
		f := plan.Submit(d.Queue(), tensor.Bindings{
			Buffers: map[string]accel.BufferView{
				"table": f32Buffer(t, d, "table", table),
				"ids":   u32Buffer(t, d, "ids", ids),
				"pos":   u32Buffer(t, d, "pos", pos),
				"bias":  f32Buffer(t, d, "bias", bias),
				"y":     out,
			},
			Scalars: map[string]tensor.ScalarValue{"theta": tensor.F32(10000)},
		})
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
		return got
	}

	cpuDev, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open CPU: %v", err)
	}
	defer cpuDev.Close()
	cpu := run(t, cpuDev)
	gpu := run(t, openMetalRuntimeDevice(t))

	// RoPE reaches sin and cos and Softmax reaches exp and a division, both
	// bounded by specs/008-numerics.md §6 and doubled for two implementations,
	// rounded the way specs/022-msl-target.md derives its ceilings. The views
	// and the gather are exact and contribute nothing to this.
	if r := numeq.WithinULP(gpu, cpu, 16); !r.Equal {
		t.Fatalf("the operator set disagrees between backends: %v\n  the ceiling is "+
			"RoPE's sin/cos and Softmax's exp; the views and the gather are exact", r)
	}

	nonzero := 0
	for _, v := range gpu {
		if v != 0 {
			nonzero++
		}
	}
	if nonzero < len(gpu)/2 {
		t.Fatalf("only %d of %d outputs are non-zero, so the plan did not run",
			nonzero, len(gpu))
	}
}
