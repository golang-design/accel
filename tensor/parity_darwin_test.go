// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package tensor_test

import (
	"math"
	"strconv"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/numeq"
)

// Every tensor operator, on both backends, per specs/062-backend-parity.md
// section 6.6.
//
// The gate that says which operators are missing is in parity_matrix_test.go
// and has no build tag. This is the half that needs a GPU.
func TestTheTensorOperatorsAgreeOnCPUAndMetal(t *testing.T) {
	gpu := openMetalRuntimeDevice(t)
	cpu := tensorParityOracle(t, gpu)

	for _, c := range tensorParityCases() {
		t.Run(c.name, func(t *testing.T) {
			onCPU := c.run(t, cpu)
			assertNotAllZero(t, "the CPU backend", onCPU)
			onMetal := c.run(t, gpu)
			assertNotAllZero(t, "Metal", onMetal)

			if len(onCPU) != len(onMetal) {
				t.Fatalf("%d values on the CPU backend and %d on Metal",
					len(onCPU), len(onMetal))
			}
			var r numeq.Report
			switch {
			case c.ceiling.Exact():
				r = numeq.Exact(onMetal, onCPU)
			case c.ceiling.Abs > 0:
				r = withinAbsolute(onMetal, onCPU, c.ceiling.Abs)
			default:
				r = numeq.WithinULP(onMetal, onCPU, c.ceiling.ULP)
			}
			if !r.Equal {
				t.Fatalf("%v\n  the ceiling is %v\n  both backends compile the same "+
					"plan from one builder, so a disagreement beyond it is the "+
					"operator's rather than the caller's", r, c.ceiling)
			}
		})
	}
}

// tensorParityOracle opens the CPU backend at the device's subgroup width.
//
// specs/062-backend-parity.md section 4.1: the CPU backend emulates subgroups
// at a width the caller chooses and defaults to four, and a reduction over more
// lanes than the emulated width sums a different number of partial results. The
// attention and the products here all reduce, so an oracle at the wrong width
// would be comparing two different computations.
func tensorParityOracle(t *testing.T, gpu *accel.Device) *accel.Device {
	t.Helper()
	lim := gpu.Limits()
	if lim.MinSubgroupSize != lim.MaxSubgroupSize {
		t.Fatalf("this device reports a subgroup width range of [%d, %d]; the oracle "+
			"emulates one width", lim.MinSubgroupSize, lim.MaxSubgroupSize)
	}
	cpu, err := accel.OpenCPU(accel.CPUOptions{SubgroupSize: lim.MinSubgroupSize})
	if err != nil {
		t.Fatalf("open the CPU oracle at subgroup width %d: %v", lim.MinSubgroupSize, err)
	}
	t.Cleanup(func() { _ = cpu.Close() })
	return cpu
}

// withinAbsolute compares against an absolute ceiling, which is what a value
// crossing zero needs: a ULP distance across a zero crossing is enormous and
// says nothing about whether the two agree.
func withinAbsolute(got, want []float32, ceiling float64) numeq.Report {
	r := numeq.Report{Equal: true, FirstDiff: -1, Len: len(got), WantLen: len(want)}
	if len(got) != len(want) {
		r.Equal = false
		return r
	}
	for i := range got {
		if d := math.Abs(float64(got[i]) - float64(want[i])); d > ceiling {
			r.Diffs++
			if r.Equal {
				r.Equal, r.FirstDiff = false, i
				r.Got, r.Want = fmtF32(got[i]), fmtF32(want[i])
			}
		}
	}
	return r
}

func fmtF32(v float32) string { return strconv.FormatFloat(float64(v), 'g', -1, 32) }
