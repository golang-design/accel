// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal

import (
	"fmt"

	"golang.design/x/accel/internal/conformance/probe"
	"golang.design/x/accel/internal/kernelc/emit"
	"golang.design/x/accel/internal/mtl"
)

// # Measuring Metal's arithmetic
//
// specs/009-sequencing.md's risk table makes this first: "M6 probes before
// other Metal numeric tests", answered by changing the lowering and never by
// widening a bound from what the device happened to report. So this exists
// before any test derives a tolerance from it.
//
// Every probe kernel carries [emit.MSLContractOff], because the question is not
// what Metal does in the abstract -- it is what the code this project generates
// does. A probe compiled with different options from the emitter would measure
// a compiler nobody ships.
//
// One dispatch per operation is slow and it is the only honest arrangement.
// Batching would put several operations in one kernel and let the compiler
// rearrange across them, which is the thing being measured.

// probeSource is one kernel per operation, selected by entry point.
//
// The operations take their inputs from a buffer rather than from literals for
// the reason specs/008-numerics.md gives about Go constants: a compiler that
// can see both operands folds the expression at build time in exact arithmetic,
// and the probe then measures the compiler instead of the machine.
const probeSource = `#include <metal_stdlib>
using namespace metal;
` + emit.MSLContractOff + `

kernel void p_add(const device float *in [[buffer(0)]], device float *out [[buffer(1)]]) {
  out[0] = in[0] + in[1];
}
kernel void p_sub(const device float *in [[buffer(0)]], device float *out [[buffer(1)]]) {
  out[0] = in[0] - in[1];
}
kernel void p_mul(const device float *in [[buffer(0)]], device float *out [[buffer(1)]]) {
  out[0] = in[0] * in[1];
}
kernel void p_div(const device float *in [[buffer(0)]], device float *out [[buffer(1)]]) {
  out[0] = in[0] / in[1];
}
kernel void p_muladd(const device float *in [[buffer(0)]], device float *out [[buffer(1)]]) {
  out[0] = in[0] * in[1] + in[2];
}
kernel void p_ldexp(const device float *in [[buffer(0)]], device float *out [[buffer(1)]]) {
  out[0] = ldexp(in[0], int(in[1]));
}
`

// Prober runs one arithmetic operation at a time on a device.
type Prober struct {
	dev   *mtl.Device
	queue *mtl.Queue
	in    *mtl.Buffer
	out   *mtl.Buffer
	pipes map[string]*mtl.Pipeline
}

// NewProber compiles the probe kernels.
func NewProber(d *mtl.Device) (*Prober, error) {
	p := &Prober{dev: d, pipes: map[string]*mtl.Pipeline{}}
	var err error
	if p.in, err = d.NewBuffer(16, mtl.StorageShared); err != nil {
		return nil, err
	}
	if p.out, err = d.NewBuffer(4, mtl.StorageShared); err != nil {
		p.in.Close()
		return nil, err
	}
	p.queue = d.NewQueue()
	for _, name := range []string{"p_add", "p_sub", "p_mul", "p_div", "p_muladd", "p_ldexp"} {
		pipe, err := d.Compile(probeSource, name)
		if err != nil {
			p.Close()
			return nil, fmt.Errorf("accel: compiling probe %s: %w", name, err)
		}
		p.pipes[name] = pipe
	}
	return p, nil
}

// Close releases everything the prober holds.
func (p *Prober) Close() {
	for _, pipe := range p.pipes {
		pipe.Close()
	}
	p.pipes = nil
	if p.queue != nil {
		p.queue.Close()
	}
	if p.in != nil {
		p.in.Close()
	}
	if p.out != nil {
		p.out.Close()
	}
}

// run executes one probe kernel over the given inputs.
//
// The inputs are written and the output read without any Go arithmetic in
// between: the kernel is the only thing that computes, which is the whole point
// of measuring rather than assuming.
func (p *Prober) run(name string, args ...float32) float32 {
	in := unsafeFloats(p.in.Bytes())
	for i := range in {
		in[i] = 0
	}
	copy(in, args)
	unsafeFloats(p.out.Bytes())[0] = 0

	cb := p.queue.Begin()
	defer cb.Close()
	e := cb.Compute()
	e.SetPipeline(p.pipes[name])
	e.SetBuffer(p.in, 0, 0)
	e.SetBuffer(p.out, 0, 1)
	one := mtl.Size{Width: 1, Height: 1, Depth: 1}
	e.Dispatch(one, one)
	e.End()
	cb.Commit()
	cb.Wait()
	return unsafeFloats(p.out.Bytes())[0]
}

// Ops presents this device's arithmetic to the probe harness.
func (p *Prober) Ops() probe.Ops {
	return probe.Ops{
		Add: func(a, b float32) float32 { return p.run("p_add", a, b) },
		Sub: func(a, b float32) float32 { return p.run("p_sub", a, b) },
		Mul: func(a, b float32) float32 { return p.run("p_mul", a, b) },
		Div: func(a, b float32) float32 { return p.run("p_div", a, b) },
		// Ldexp's exponent arrives as a float because the probe buffer is one
		// dtype. Every exponent this harness uses is small and exactly
		// representable, so the conversion is lossless; a larger one would not
		// be, which is why the assertion is here rather than assumed.
		Ldexp: func(m float32, e int) float32 {
			if float64(float32(e)) != float64(e) {
				panic(fmt.Sprintf("accel: probe exponent %d is not exact as f32", e))
			}
			return p.run("p_ldexp", m, float32(e))
		},
		MulAdd: func(a, b, c float32) float32 { return p.run("p_muladd", a, b, c) },
	}
}

// Profile measures this device.
func (p *Prober) Profile() probe.Profile {
	return probe.Measure("arm64", "metal", p.Ops())
}

// Denormal reports whether the device produced the value math expects, which
// [probe.Profile] folds into SubnormalsPreserved. Kept as its own helper
// because a caller debugging a flush-to-zero device wants the number.
func (p *Prober) Denormal() float32 { return p.run("p_ldexp", 1, -149) }
