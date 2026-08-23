// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal_test

import (
	"fmt"
	"math"
	"testing"

	"golang.design/x/accel/internal/conformance/numeq"
	"golang.design/x/accel/internal/kernelc/emit"
	"golang.design/x/accel/internal/mtl"
	"golang.design/x/accel/kmath"
)

// # Metal against specs/008-numerics.md section 6
//
// The ceilings in that table are a *library contract*, and measurement only
// proves an implementation meets it. Nothing else in this suite proves it for
// Metal: the corpus differential compares two lowerings against each other, so
// if this device's exp sat forty ULP from correctly rounded and kmath.Exp
// drifted the same way, that comparison would pass while Metal violated its
// normative bound. specs/009-sequencing.md's risk row asks for these probes
// before other Metal numeric tests for exactly that reason.
//
// The reference is float64 rounded once to f32. That is not correctly rounded
// by construction -- a double rounding can add half an ULP where the f64 result
// lands on an f32 tie -- and it is well inside every ceiling here, none of
// which is tighter than one ULP.
//
// **Both backends are measured against it**, not against each other. A ceiling
// is about the distance from the truth, and two implementations can be equally
// and identically wrong.

// primitive is one operation's Metal spelling, Go spelling, ceiling and domain.
type primitive struct {
	name string

	// msl is the expression, with x the input.
	msl string

	// ref is the higher-precision reference: f64 throughout, rounded once.
	ref func(float64) float64

	// cpu is the lowering the CPU backend actually runs, which is what the
	// ceiling has to hold for -- not math.Exp, but kmath.Exp.
	cpu func(float32) float32

	// ulp and abs are the ceiling, one or the other. Section 6 bounds sin and
	// cos absolutely because argument reduction dominates and a ULP count near
	// a zero crossing means nothing.
	ulp uint64
	abs float64

	// inputs are the domain the ceiling is claimed over.
	inputs []float32
}

func spread(lo, hi float64, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(lo + (hi-lo)*float64(i)/float64(n-1))
	}
	return out
}

func primitives() []primitive {
	// Positive normals across many binades, which is where a lowering that used
	// a cheap approximation diverges: a probe over [1, 2] measures almost
	// nothing.
	positive := []float32{
		1e-30, 1e-20, 1e-10, 1e-5, 0.001, 0.1, 0.5, 0.75, 1, 1.5, 2, 3, 7.25,
		100, 1e4, 1e10, 1e20, 1e30,
	}
	return []primitive{{
		name: "div", msl: "x / 3.0f",
		ref: func(x float64) float64 { return x / 3 },
		cpu: func(x float32) float32 { return x / 3 },
		// Section 6 allows 2.5 ULP. Two is what the operator delivers and ULP
		// counts are integral, so the assertion is stricter than the contract
		// -- which is the right direction: a regression to 3 would fail here
		// while still meeting the spec, and that is worth being told about.
		ulp: 2, inputs: append(spread(-1e6, 1e6, 64), positive...),
	}, {
		name: "sqrt", msl: "precise::sqrt(x)",
		ref: math.Sqrt, cpu: kmath.Sqrt,
		// One ULP, the tightest in the table, and deliberately so: allowing two
		// would hide an accidental x*rsqrt(x) lowering.
		ulp: 1, inputs: positive,
	}, {
		name: "rsqrt", msl: "rsqrt(x)",
		ref: func(x float64) float64 { return 1 / math.Sqrt(x) },
		cpu: kmath.RSqrt,
		ulp: 4, inputs: positive,
	}, {
		name: "exp", msl: "precise::exp(x)",
		ref: math.Exp, cpu: kmath.Exp,
		// Bounded so the result stays finite and normal, which is the domain
		// section 6 states.
		ulp: 4, inputs: spread(-80, 80, 128),
	}, {
		name: "log", msl: "precise::log(x)",
		ref: math.Log, cpu: kmath.Log,
		ulp: 4, inputs: positive,
	}, {
		name: "tanh", msl: "precise::tanh(x)",
		ref: math.Tanh, cpu: kmath.Tanh,
		ulp: 4, inputs: spread(-20, 20, 128),
	}, {
		name: "sin", msl: "precise::sin(x)",
		ref: math.Sin, cpu: kmath.Sin,
		// Absolute, and out to 2^16, which is the domain section 6 admits and
		// the range v0 RoPE positions reach. Large arguments are the point:
		// argument reduction is where a shader library gives up.
		abs: math.Ldexp(1, -20), inputs: spread(-65536, 65536, 256),
	}, {
		name: "cos", msl: "precise::cos(x)",
		ref: math.Cos, cpu: kmath.Cos,
		abs: math.Ldexp(1, -20), inputs: spread(-65536, 65536, 256),
	}}
}

// Every v0 primitive meets its normative ceiling on this device, and on the CPU
// oracle.
//
// A miss is answered by changing the lowering or narrowing the supported
// domain, never by widening the ceiling -- specs/008-numerics.md says so at the
// top of section 6, and this test is written so that widening it is a visible
// edit to a number the spec also states.
func TestPrimitiveCeilings(t *testing.T) {
	devs := requireDevices(t)
	d := devs[0]
	q := d.NewQueue()
	defer q.Close()

	for _, p := range primitives() {
		t.Run(p.name, func(t *testing.T) {
			want := make([]float32, len(p.inputs))
			for i, x := range p.inputs {
				want[i] = float32(p.ref(float64(x)))
			}

			// The CPU lowering first. If the oracle misses its own ceiling then
			// every comparison against it downstream is unfounded, and a Metal
			// failure reported alongside would be the less important half.
			host := make([]float32, len(p.inputs))
			for i, x := range p.inputs {
				host[i] = p.cpu(x)
			}
			check(t, "the CPU lowering", p, host, want)

			check(t, "Metal", p, runPrimitive(t, d, q, p), want)
		})
	}
}

func check(t *testing.T, who string, p primitive, got, want []float32) {
	t.Helper()
	var r numeq.Report
	if p.abs > 0 {
		r = withinAbsolute(got, want, p.abs)
	} else {
		r = numeq.WithinULP(got, want, p.ulp)
	}
	if r.Equal {
		return
	}
	t.Errorf("%s misses the ceiling specs/008-numerics.md section 6 sets for %s: %v\n"+
		"  a miss changes the lowering or the supported domain, never the ceiling",
		who, p.name, r)
}

func withinAbsolute(got, want []float32, ceiling float64) numeq.Report {
	r := numeq.Report{Equal: true, FirstDiff: -1, Len: len(got), WantLen: len(want)}
	for i := range got {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		if !(d <= ceiling) { // written so a NaN fails rather than passing
			r.Diffs++
			if r.FirstDiff < 0 {
				r.FirstDiff, r.Equal = i, false
				r.Got = fmt.Sprintf("%v", got[i])
				r.Want = fmt.Sprintf("%v, off by %g against a ceiling of %g",
					want[i], d, ceiling)
			}
		}
	}
	return r
}

// runPrimitive evaluates one operation over its inputs on the device.
func runPrimitive(t *testing.T, d *mtl.Device, q *mtl.Queue, p primitive) []float32 {
	t.Helper()
	src := fmt.Sprintf(`#include <metal_stdlib>
using namespace metal;
%s

kernel void probe(const device float *in [[buffer(0)]],
                  device float *out [[buffer(1)]],
                  uint gid [[thread_position_in_grid]]) {
  float x = in[gid];
  out[gid] = %s;
}`, emit.MSLContractOff, p.msl)

	pipe, err := d.Compile(src, "probe")
	if err != nil {
		t.Fatalf("compile %s: %v", p.name, err)
	}
	defer pipe.Close()

	n := len(p.inputs)
	in, err := d.NewBuffer(n*4, mtl.StorageShared)
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer in.Close()
	out, err := d.NewBuffer(n*4, mtl.StorageShared)
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer out.Close()
	copy(unsafeFloats(in.Bytes()), p.inputs)

	cb := q.Begin()
	defer cb.Close()
	e := cb.Compute()
	e.SetPipeline(pipe)
	e.SetBuffer(in, 0, 0)
	e.SetBuffer(out, 0, 1)
	e.Dispatch(mtl.Size{Width: uint64(n), Height: 1, Depth: 1},
		mtl.Size{Width: 1, Height: 1, Depth: 1})
	e.End()
	cb.Commit()
	cb.Wait()
	if err := cb.Err(); err != nil {
		t.Fatalf("submission: %v", err)
	}
	return append([]float32(nil), unsafeFloats(out.Bytes())[:n]...)
}
