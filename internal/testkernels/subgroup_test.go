// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"fmt"
	"golang.design/x/accel/kernelabi"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
)

// The subgroup sweep: the subgroup path and its fallback agree at sizes 1, 4,
// 32 and 64.
//
// Each size is in the list for a reason spec 020 states. At 1 every operation
// degenerates to the identity, so a kernel with an assumption about having
// neighbours breaks. At 4 a 64-wide workgroup spans sixteen subgroups, so the
// boundary is crossed repeatedly. 32 is NVIDIA and Apple, 64 is AMD and the
// case where one subgroup spans the whole workgroup.
//
// A reduction that agrees at 32 and disagrees at 4 has a boundary bug, and a
// boundary bug at v0 becomes a wrong answer on hardware nobody in this project
// owns.
func TestSubgroupSweepAgreesWithTheFallback(t *testing.T) {
	const group = 64
	for _, width := range []uint32{1, 4, 32, 64} {
		t.Run(fmt.Sprintf("size=%d", width), func(t *testing.T) {
			for _, n := range []int{group, group * 3, group*2 + 17} {
				t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
					in := make([]float32, n)
					for i := range in {
						in[i] = float32(i)*0.5 - 3
					}
					subgroups := (n + int(width) - 1) / int(width)

					viaSubgroup := make([]float32, subgroups)
					err := kernel.DispatchCooperativeWith(&testkernels.SubgroupReduceKernel,
						accel.ID3{X: uint32((n + group - 1) / group)},
						kernelabi.Args{Slices: []any{in, viaSubgroup}},
						kernel.Options{SubgroupSize: width, Diagnostics: true})
					if err != nil {
						t.Fatalf("subgroup path: %v", err)
					}

					// The fallback is flat, because it uses no subgroup
					// operation — which is the point. A fallback sharing any of
					// the subgroup path's machinery could not show the two
					// agree.
					viaFallback := make([]float32, subgroups)
					err = kernel.Dispatch(&testkernels.SubgroupReduceFallbackKernel,
						accel.ID3{X: uint32((n + group - 1) / group)},
						kernelabi.Args{Slices: []any{in, viaFallback, []uint32{width}}})
					if err != nil {
						t.Fatalf("fallback: %v", err)
					}

					for i := range viaSubgroup {
						if math.Float32bits(viaSubgroup[i]) != math.Float32bits(viaFallback[i]) {
							t.Fatalf("subgroup %d: subgroup path %v, fallback %v",
								i, viaSubgroup[i], viaFallback[i])
						}
					}
				})
			}
		})
	}
}

// At size 1 every operation is the identity, which is the case a kernel
// assuming it has neighbours gets wrong.
func TestASubgroupOfOneIsTheIdentity(t *testing.T) {
	const group, n = 64, 64
	in := make([]float32, n)
	for i := range in {
		in[i] = float32(i) + 1
	}
	out := make([]float32, n)

	err := kernel.DispatchCooperativeWith(&testkernels.SubgroupReduceKernel,
		accel.ID3{X: 1}, kernelabi.Args{Slices: []any{in, out}},
		kernel.Options{SubgroupSize: 1, Diagnostics: true})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	for i := range out {
		if out[i] != in[i] {
			t.Fatalf("lane %d received %v from a subgroup of one, want its own %v: a "+
				"reduction over one active lane is that lane's value, not v + 0",
				i, out[i], in[i])
		}
	}
}

// Elect is true for exactly one lane per subgroup, and accel pins which.
// Hardware guarantees only "exactly one", so an unpinned choice would make a
// correct kernel's output depend on the device.
func TestElectPicksTheLowestLane(t *testing.T) {
	const group = 64
	for _, width := range []uint32{1, 4, 32, 64} {
		t.Run(fmt.Sprintf("size=%d", width), func(t *testing.T) {
			in := make([]float32, group)
			for i := range in {
				in[i] = 1
			}
			subgroups := group / int(width)
			out := make([]float32, subgroups)
			// Poison, so a subgroup whose elected lane never wrote is visible.
			for i := range out {
				out[i] = -1
			}

			err := kernel.DispatchCooperativeWith(&testkernels.SubgroupReduceKernel,
				accel.ID3{X: 1}, kernelabi.Args{Slices: []any{in, out}},
				kernel.Options{SubgroupSize: width, Diagnostics: true})
			if err != nil {
				t.Fatalf("dispatch: %v", err)
			}
			for i, got := range out {
				if want := float32(width); got != want {
					t.Fatalf("subgroup %d holds %v, want %v: exactly one lane should have "+
						"written the subgroup's total", i, got, want)
				}
			}
		})
	}
}

// The id accessors follow the mapping spec 002 section 5.1 says the oracle
// uses, which is what a kernel building its own lane-to-data mapping needs to
// know.
func TestSubgroupIDsFollowTheOracleMapping(t *testing.T) {
	const group = 64
	for _, width := range []uint32{1, 4, 32, 64} {
		size := kernel.ID3{X: group, Y: 1, Z: 1}
		for l := range uint32(group) {
			th := kernel.NewThreadWithSubgroup(kernel.ID3{X: l}, kernel.ID3{X: l},
				kernel.ID3{}, size, kernel.ID3{X: 1}, width)
			if got, want := th.SubgroupSize(), width; got != want {
				t.Fatalf("SubgroupSize is %d, want %d", got, want)
			}
			if got, want := th.SubgroupIndex(), l/width; got != want {
				t.Errorf("width %d lane %d: SubgroupIndex is %d, want %d", width, l, got, want)
			}
			if got, want := th.SubgroupLane(), l%width; got != want {
				t.Errorf("width %d lane %d: SubgroupLane is %d, want %d",
					width, l, got, want)
			}
		}
	}
}

// The authored halves of the subgroup kernels, which is spec 004's fifth
// testing level.
//
// The rendezvous is emulated the way the scheduler implements it: every lane
// contributes, the combination happens, and every lane reads back. The authored
// method bodies return their own argument, so the emulation has to do the
// combining — which is exactly the point of running them, since it shows the
// authored form means what the lowering computes rather than being a stub
// nobody exercises.
func TestAuthoredSubgroupKernels(t *testing.T) {
	const group, width = 64, 16
	in := make([]float32, group)
	for i := range in {
		in[i] = float32(i) * 0.25
	}

	// The authored fallback runs directly: it is flat.
	authored := make([]float32, group/width)
	size := kernel.ID3{X: group, Y: 1, Z: 1}
	for l := range uint32(group) {
		th := kernel.NewThreadWithSubgroup(kernel.ID3{X: l}, kernel.ID3{X: l},
			kernel.ID3{}, size, kernel.ID3{X: 1}, width)
		testkernels.SubgroupReduceFallback(th, in, authored, []uint32{width})
	}

	// And the generated subgroup path agrees with it.
	generated := make([]float32, group/width)
	err := kernel.DispatchCooperativeWith(&testkernels.SubgroupReduceKernel,
		accel.ID3{X: 1}, kernelabi.Args{Slices: []any{in, generated}},
		kernel.Options{SubgroupSize: width, Diagnostics: true})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	for i := range authored {
		if math.Float32bits(authored[i]) != math.Float32bits(generated[i]) {
			t.Fatalf("subgroup %d: authored fallback %v, generated subgroup path %v",
				i, authored[i], generated[i])
		}
	}

	// The authored subgroup kernel itself, under an emulated rendezvous. Its
	// SubgroupAddF32 returns the lane's own value, so what it computes is the
	// per-lane identity — which is what a subgroup of one gives, and is
	// therefore the right expectation rather than a stub's artefact.
	perLane := make([]float32, group)
	for l := range uint32(group) {
		th := kernel.NewThreadWithSubgroup(kernel.ID3{X: l}, kernel.ID3{X: l},
			kernel.ID3{}, size, kernel.ID3{X: 1}, 1)
		testkernels.SubgroupReduce(th, in, perLane)
	}
	for i := range perLane {
		if perLane[i] != in[i] {
			t.Fatalf("lane %d wrote %v, want its own %v", i, perLane[i], in[i])
		}
	}
}

// The subgroup kernel's record carries the capabilities its body implies,
// inferred rather than declared.
func TestTheSubgroupKernelDeclaresItsCapabilities(t *testing.T) {
	caps := accel.Capability(testkernels.SubgroupReduceKernel.Caps)
	for _, want := range []struct {
		cap  accel.Capability
		name string
		why  string
	}{
		{accel.CapSubgroupArithmetic, "CapSubgroupArithmetic", "it reduces across lanes"},
		{accel.CapSubgroupBasic, "CapSubgroupBasic", "it elects a lane and reads SubgroupIndex"},
	} {
		if caps&want.cap == 0 {
			t.Errorf("the record does not require %s, and %s", want.name, want.why)
		}
	}
	// And the fallback requires none, which is what makes it a fallback.
	if got := testkernels.SubgroupReduceFallbackKernel.Caps; got != 0 {
		t.Errorf("the fallback requires capabilities %d and should require none", got)
	}
}
