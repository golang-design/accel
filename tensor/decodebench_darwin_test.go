// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package tensor_test

import (
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"
	"golang.design/x/accel/tensor"
)

// The two GEMM shapes a transformer spends its time in, on Metal: a decode
// step's [1, K] x [K, N] over f32 activations and f16 weights, and a prefill
// block's [M, K] x [K, N] over int8 weights. Each submits one plan per
// iteration and waits, so the figure is a whole step's latency including the
// host's part, which is what a caller pacing tokens sees.
func BenchmarkDecodeMatMulF32F16OnMetal(b *testing.B) {
	const k, n = 2048, 2048
	d := openMetalBenchDevice(b)
	rt, err := tensor.NewRuntime(d)
	if err != nil {
		b.Fatal(err)
	}
	defer rt.Close()
	bd := rt.NewBuilder("decode")
	x := tensor.Input(bd, tensor.ValueDesc{Name: "x", DType: accel.F32, Shape: tensor.Shape{1, k}})
	w := tensor.Weight(bd, tensor.ValueDesc{Name: "w", DType: accel.F16, Shape: tensor.Shape{k, n}})
	tensor.Output(bd, "y", tensor.MatMul(bd, x, w))
	plan, err := bd.Compile(rt, tensor.CompileOptions{})
	if err != nil {
		b.Fatal(err)
	}
	defer plan.Close()
	b.Logf("kernel: %s", plan.Selections()[0].Kernel)

	xs := make([]float32, k)
	ws := make([]float32, k*n)
	for i := range xs {
		xs[i] = float32(i%7) * 0.25
	}
	for i := range ws {
		ws[i] = float32(i%13) * 0.125
	}
	tb := &testing.T{}
	bind := tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f32Buffer(tb, d, "x", xs), "w": f16Buffer(tb, d, "w", ws),
		"y": f32Buffer(tb, d, "y", make([]float32, n)),
	}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := plan.Submit(d.Queue(), bind).Wait(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkPrefillQuantMatMulOnMetal(b *testing.B) {
	const m, k, n = 64, 1024, 1024
	d := openMetalBenchDevice(b)
	rt, err := tensor.NewRuntime(d)
	if err != nil {
		b.Fatal(err)
	}
	defer rt.Close()
	bd := rt.NewBuilder("prefill")
	x := tensor.Input(bd, tensor.ValueDesc{Name: "x", DType: accel.F32, Shape: tensor.Shape{m, k}})
	qw := tensor.Weight(bd, tensor.ValueDesc{Name: "wq", DType: accel.I8, Shape: tensor.Shape{k, n}})
	sw := tensor.Weight(bd, tensor.ValueDesc{Name: "ws", DType: accel.F16, Shape: tensor.Shape{k * n / 32}})
	tensor.Output(bd, "y", tensor.QuantMatMul(bd, x, tensor.Quantized{Quants: qw, Scales: sw}))
	plan, err := bd.Compile(rt, tensor.CompileOptions{})
	if err != nil {
		b.Fatal(err)
	}
	defer plan.Close()
	b.Logf("kernel: %s", plan.Selections()[0].Kernel)

	xs := make([]float32, m*k)
	ws := make([]float32, k*n)
	for i := range xs {
		xs[i] = float32(i%7) * 0.25
	}
	for i := range ws {
		ws[i] = float32(i%13)*0.125 - 0.75
	}
	wq, wsc := quant.Int8Quantize(ws)
	tb := &testing.T{}
	bind := tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f32Buffer(tb, d, "x", xs), "wq": i8Buffer(tb, d, "wq", wq),
		"ws": f16BitsBuffer(tb, d, "ws", wsc),
		"y":  f32Buffer(tb, d, "y", make([]float32, m*n)),
	}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := plan.Submit(d.Queue(), bind).Wait(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeQuantMatMulOnMetal(b *testing.B) {
	const k, n = 2048, 2048
	d := openMetalBenchDevice(b)
	rt, err := tensor.NewRuntime(d)
	if err != nil {
		b.Fatal(err)
	}
	defer rt.Close()
	bd := rt.NewBuilder("decode-int8")
	x := tensor.Input(bd, tensor.ValueDesc{Name: "x", DType: accel.F32, Shape: tensor.Shape{1, k}})
	qw := tensor.Weight(bd, tensor.ValueDesc{Name: "wq", DType: accel.I8, Shape: tensor.Shape{k, n}})
	sw := tensor.Weight(bd, tensor.ValueDesc{Name: "ws", DType: accel.F16, Shape: tensor.Shape{k * n / 32}})
	tensor.Output(bd, "y", tensor.QuantMatMul(bd, x, tensor.Quantized{Quants: qw, Scales: sw}))
	plan, err := bd.Compile(rt, tensor.CompileOptions{})
	if err != nil {
		b.Fatal(err)
	}
	defer plan.Close()
	b.Logf("kernel: %s", plan.Selections()[0].Kernel)

	xs := make([]float32, k)
	ws := make([]float32, k*n)
	for i := range xs {
		xs[i] = float32(i%7) * 0.25
	}
	for i := range ws {
		ws[i] = float32(i%13)*0.125 - 0.75
	}
	wq, wsc := quant.Int8Quantize(ws)
	tb := &testing.T{}
	bind := tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f32Buffer(tb, d, "x", xs), "wq": i8Buffer(tb, d, "wq", wq),
		"ws": f16BitsBuffer(tb, d, "ws", wsc),
		"y":  f32Buffer(tb, d, "y", make([]float32, n)),
	}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := plan.Submit(d.Queue(), bind).Wait(); err != nil {
			b.Fatal(err)
		}
	}
}
