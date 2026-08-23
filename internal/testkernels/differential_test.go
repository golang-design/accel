// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"golang.design/x/accel/kernelabi"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/direct"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
)

// The flat-versus-cooperative differential, which is spec 018's definition of
// done.
//
// Every kernel eligible for both lowerings runs both ways over the same inputs
// and the results are compared bit for bit. A disagreement is a bug in the
// resumable transform, localized to the transform, because both lowerings are
// generated from one IR and ran over the same data.
//
// Bit for bit rather than within a tolerance: one IR means one set of rounding
// points, so any difference at all is the transform's. A tolerance here would
// hide exactly what this exists to find.
//
// This is the same shape as spec 017's whole-plan oracle, which found three
// real bugs within minutes of first running. That is why it is the criterion
// rather than a note beside one.
func TestFlatAndCooperativeLoweringsAgree(t *testing.T) {
	const n = 130 // not a multiple of the workgroup, so the tail is exercised

	cases := []struct {
		name   string
		kernel *accel.Kernel
		args   func() (kernelabi.Args, []float32)
	}{{
		name:   "Scale",
		kernel: &testkernels.ScaleKernel,
		args: func() (kernelabi.Args, []float32) {
			in, out := ramp(n), make([]float32, n)
			return kernelabi.Args{Slices: []any{in, out}}, out
		},
	}, {
		name:   "Add",
		kernel: &testkernels.AddKernel,
		args: func() (kernelabi.Args, []float32) {
			a, b, out := ramp(n), ramp(n), make([]float32, n)
			for i := range b {
				b[i] = -b[i] * 0.5
			}
			return kernelabi.Args{Slices: []any{a, b, out}}, out
		},
	}, {
		name:   "SegmentSum",
		kernel: &testkernels.SegmentSumKernel,
		args: func() (kernelabi.Args, []float32) {
			in, out := ramp(n), make([]float32, n)
			return kernelabi.Args{Slices: []any{in, out}}, out
		},
	}, {
		name:   "Normalize",
		kernel: &testkernels.NormalizeKernel,
		args: func() (kernelabi.Args, []float32) {
			in := make([]accel.Float16, n)
			for i := range in {
				in[i] = accel.ToFloat16(float32(i)*0.25 - 3)
			}
			out, scratch := make([]float32, n), make([]float32, n)
			return kernelabi.Args{Slices: []any{in, out, scratch}}, out
		},
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if c.kernel.Flat == nil {
				t.Fatalf("%s has no flat lowering", c.name)
			}
			groups := accel.ID3{X: uint32((n + 63) / 64)}

			flatArgs, flatOut := c.args()
			if err := direct.Run(c.kernel, groups, flatArgs); err != nil {
				t.Fatalf("flat: %v", err)
			}

			// The same kernel, driven through the cooperative scheduler. A flat
			// kernel has no suspension points, so it finishes in one epoch --
			// which is the property that makes it eligible for both.
			coopArgs, coopOut := c.args()
			if err := runAsCooperative(c.kernel, groups, coopArgs); err != nil {
				t.Fatalf("cooperative: %v", err)
			}

			for i := range flatOut {
				if math.Float32bits(flatOut[i]) != math.Float32bits(coopOut[i]) {
					t.Fatalf("element %d: flat %v, cooperative %v", i, flatOut[i], coopOut[i])
				}
			}
		})
	}
}

// runAsCooperative drives a flat kernel through the cooperative scheduler by
// wrapping its entry point in one that never suspends.
//
// The wrapper is the whole trick, and it is why every flat kernel is eligible:
// a lowering with no barrier runs to completion on its first call, so "advance
// every invocation to its next suspension point" and "run every invocation" are
// the same thing. What this exercises is the scheduler's own machinery -- the
// per-invocation frames, the epoch loop, the id computation -- against the path
// that does not use any of it.
func runAsCooperative(k *accel.Kernel, count accel.ID3, args kernelabi.Args) error {
	wrapped := *k
	wrapped.Flat = nil
	wrapped.Suspensions = 0
	wrapped.Cooperative = func(t accel.Thread, a kernelabi.Args, f *kernelabi.Frame) bool {
		k.Flat(t, a)
		return false
	}
	return kernel.DispatchCooperative(&wrapped, count, args)
}

func ramp(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(i)*0.25 - 3
	}
	return out
}

// The scheduler computes the same ids the flat path does, all of them.
//
// All of them deliberately: an earlier version recorded only the global id, and
// a scheduler that swapped the local id's x and y axes passed it — because none
// of the differential's kernels reads the local id. A kernel that addresses a
// shared tile does, and getting that wrong is exactly the bug that would then
// reach M5's GEMM.
func TestTheSchedulerComputesTheSameIDs(t *testing.T) {
	size := accel.ID3{X: 8, Y: 4, Z: 2} // three non-trivial axes
	count := accel.ID3{X: 3, Y: 2, Z: 1}

	type ids struct{ global, local, group accel.ID3 }
	var flat, coop []ids
	record := func(dst *[]ids) *accel.Kernel {
		return &accel.Kernel{
			Name: "Ids", WorkgroupSize: size, Generator: kernelabi.Version,
			Bindings: []kernelabi.Binding{
				{Name: "out", DType: kernelabi.U32, Access: kernelabi.Write},
			},
			Flat: func(th accel.Thread, a kernelabi.Args) {
				*dst = append(*dst, ids{th.GlobalID(), th.LocalID(), th.GroupID()})
			},
		}
	}
	out := make([]uint32, 1)
	args := kernelabi.Args{Slices: []any{out}}

	if err := direct.Run(record(&flat), count, args); err != nil {
		t.Fatalf("flat: %v", err)
	}
	if err := runAsCooperative(record(&coop), count, args); err != nil {
		t.Fatalf("cooperative: %v", err)
	}

	if len(flat) != len(coop) {
		t.Fatalf("flat ran %d invocations and cooperative %d", len(flat), len(coop))
	}
	if want := 8 * 4 * 2 * 3 * 2; len(flat) != want {
		t.Fatalf("ran %d invocations, want %d", len(flat), want)
	}
	for i := range flat {
		if flat[i] != coop[i] {
			t.Fatalf("invocation %d: flat %+v, cooperative %+v", i, flat[i], coop[i])
		}
	}
}

// And the order is the same, which is what x-fastest linearization means. A
// kernel indexing shared memory by LocalIndex depends on which invocation gets
// which slot, so spec 002 section 1.4 guarantees the order rather than leaving
// it to the backend.
func TestTheSchedulerVisitsInvocationsXFastest(t *testing.T) {
	size := accel.ID3{X: 4, Y: 2, Z: 1}
	var seen []accel.ID3
	k := &accel.Kernel{
		Name: "Order", WorkgroupSize: size, Generator: kernelabi.Version,
		Bindings: []kernelabi.Binding{
			{Name: "out", DType: kernelabi.U32, Access: kernelabi.Write},
		},
		Flat: func(th accel.Thread, a kernelabi.Args) { seen = append(seen, th.LocalID()) },
	}
	if err := runAsCooperative(k, accel.ID3{X: 1}, kernelabi.Args{Slices: []any{make([]uint32, 1)}}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	i := 0
	for z := range size.Z {
		for y := range size.Y {
			for x := range size.X {
				want := accel.ID3{X: x, Y: y, Z: z}
				if seen[i] != want {
					t.Fatalf("invocation %d has local id %+v, want %+v: the order is "+
						"x-fastest and a shared tile's indexing depends on it", i, seen[i], want)
				}
				i++
			}
		}
	}
}
