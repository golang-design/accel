// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"golang.design/x/accel/kernelabi"

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

// The shuffles agree with a fallback that uses none of them, at every width in
// spec 020 section 4's sweep.
//
// Exactly, not within a tolerance: every read moves a value between lanes
// without arithmetic, and the two paths sum the same four terms in the same
// order, so a difference of one bit is a difference in what was read.
//
// The width matters more here than for the reduction. A reversal is
// `w-1-lane` and a butterfly is `lane^1`, so at width 1 both degenerate, at
// width 4 the subgroup boundary is crossed sixteen times across the workgroup,
// and at 64 the subgroup is the workgroup. A kernel that read the wrong
// neighbour at a boundary passes at 64 and fails at 4.
func TestTheShuffleSweepAgreesWithTheFallback(t *testing.T) {
	const group = 64
	in := make([]float32, group)
	for i := range in {
		// Dyadic and small, so every sum in either path is exact and the
		// comparison is about which lane was read.
		in[i] = float32(i)*0.25 - 8
	}

	for _, width := range []uint32{1, 4, 32, 64} {
		t.Run(fmt.Sprintf("size=%d", width), func(t *testing.T) {
			viaShuffle := make([]float32, group)
			err := kernel.DispatchCooperativeWith(&testkernels.SubgroupShuffleMixKernel,
				accel.ID3{X: 1}, kernelabi.Args{Slices: []any{in, viaShuffle}},
				kernel.Options{SubgroupSize: width, Diagnostics: true})
			if err != nil {
				t.Fatalf("shuffle path: %v", err)
			}

			viaFallback := make([]float32, group)
			err = kernel.Dispatch(&testkernels.SubgroupShuffleMixFallbackKernel,
				accel.ID3{X: 1},
				kernelabi.Args{Slices: []any{in, viaFallback, []uint32{width}}})
			if err != nil {
				t.Fatalf("fallback: %v", err)
			}

			for i := range viaShuffle {
				if math.Float32bits(viaShuffle[i]) != math.Float32bits(viaFallback[i]) {
					t.Fatalf("element %d: shuffle path %v, fallback %v (lane %d of subgroup %d)",
						i, viaShuffle[i], viaFallback[i], uint32(i)%width, uint32(i)/width)
				}
			}
		})
	}
}

// A shuffle whose partner lane is not there is undefined, and the oracle says
// so when the lane exists and is simply not taking part.
//
// The witness is a subgroup width that does not divide the workgroup, which is
// spec 002 section 5.1's partly filled last subgroup: at width 24 a 64-wide
// workgroup has two full subgroups and a tail of 16, so the tail's reversal
// reads lanes 16 to 23, which no invocation occupies. That is the case rule 3
// is written about, and it is reachable from the generated lowering rather than
// only from a hand-driven scheduler.
func TestTheTailSubgroupReportsItsMissingLanes(t *testing.T) {
	const group = 64
	in := make([]float32, group)
	out := make([]float32, group)
	err := kernel.DispatchCooperativeWith(&testkernels.SubgroupShuffleMixKernel,
		accel.ID3{X: 1}, kernelabi.Args{Slices: []any{in, out}},
		kernel.Options{SubgroupSize: 24, Diagnostics: true})
	if err == nil {
		t.Fatal("the last subgroup is 16 lanes wide out of 24, so its reversal reads eight " +
			"lanes that hold no invocation, and that is an undefined read spec 002 " +
			"section 5.2 rule 3 requires to be reported")
	}
	if !strings.Contains(err.Error(), "not active at that operation") {
		t.Errorf("the report should name the inactive lane, and says:\n%v", err)
	}
}

// The authored kernel and its generated lowering agree.
//
// At width 1, which is the width where the authored bodies are *correct*
// rather than a stub: a subgroup of one has every lane read itself, which is
// what `return v` computes. Every other read the kernel makes is out of range
// at that width and its result is discarded, so the two paths have to produce
// the same number for a reason rather than by luck.
func TestAuthoredShuffleKernel(t *testing.T) {
	const group = 64
	in := make([]float32, group)
	for i := range in {
		in[i] = float32(i)*0.5 - 3
	}

	authored := make([]float32, group)
	size := kernel.ID3{X: group, Y: 1, Z: 1}
	for l := range uint32(group) {
		th := kernel.NewThreadWithSubgroup(kernel.ID3{X: l}, kernel.ID3{X: l},
			kernel.ID3{}, size, kernel.ID3{X: 1}, 1)
		testkernels.SubgroupShuffleMix(th, in, authored)
	}

	generated := make([]float32, group)
	err := kernel.DispatchCooperativeWith(&testkernels.SubgroupShuffleMixKernel,
		accel.ID3{X: 1}, kernelabi.Args{Slices: []any{in, generated}},
		kernel.Options{SubgroupSize: 1, Diagnostics: true})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	for i := range authored {
		if math.Float32bits(authored[i]) != math.Float32bits(generated[i]) {
			t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
		}
	}
	// And it is not the trivially-zero function, or the comparison above would
	// hold for two kernels that both did nothing.
	if authored[7] != 5*in[7] {
		t.Fatalf("at a subgroup of one every read is the lane's own, so element 7 should "+
			"be five times its input: got %v, want %v", authored[7], 5*in[7])
	}
}

// The shuffle kernel's record requires the shuffle capability, inferred from
// its body.
//
// accel groups the lane-addressed reads under one capability, including the
// broadcast from a chosen lane that Vulkan files under its ballot feature. The
// deviation is recorded in spec 002 section 5.2, and this is where it is
// checked: a broadcast that inferred the ballot capability would make this
// kernel unavailable on a device that has every operation it uses.
func TestTheShuffleKernelDeclaresItsCapabilities(t *testing.T) {
	caps := accel.Capability(testkernels.SubgroupShuffleMixKernel.Caps)
	for _, want := range []struct {
		cap  accel.Capability
		name string
		why  string
	}{
		{accel.CapSubgroupShuffle, "CapSubgroupShuffle", "it reads four lanes by index"},
		{accel.CapSubgroupBasic, "CapSubgroupBasic", "it reads the lane index and the width"},
	} {
		if caps&want.cap == 0 {
			t.Errorf("the record does not require %s, and %s", want.name, want.why)
		}
	}
	if caps&accel.CapSubgroupBallot != 0 {
		t.Error("the record requires CapSubgroupBallot, which no operation in the kernel " +
			"produces: it would make the kernel unavailable on a device that has " +
			"everything it uses")
	}
	if got := testkernels.SubgroupShuffleMixFallbackKernel.Caps; got != 0 {
		t.Errorf("the fallback requires capabilities %d and should require none", got)
	}
}

// The scans agree with a fallback that sums the buffer, at every width in the
// sweep.
//
// Exactly, and the exactness is earned rather than assumed: spec 002 section
// 5.2 fixes the order at ascending active lane and the fallback sums in that
// order, so the two compute the same expression term by term. A fallback that
// summed downwards would differ in the last bit on inputs that round, and the
// difference would be the test's rather than the kernel's.
func TestTheScanSweepAgreesWithTheFallback(t *testing.T) {
	const group = 64
	in := make([]float32, group)
	for i := range in {
		// Values that round: a scan over these is not exact, which is the point
		// — two implementations that agree here agree on the order as well as
		// on the terms.
		in[i] = float32(i)*0.1 - 3.3
	}

	for _, width := range []uint32{1, 4, 32, 64} {
		t.Run(fmt.Sprintf("size=%d", width), func(t *testing.T) {
			incl := make([]float32, group)
			excl := make([]float32, group)
			err := kernel.DispatchCooperativeWith(&testkernels.SubgroupScanKernel,
				accel.ID3{X: 1}, kernelabi.Args{Slices: []any{in, incl, excl}},
				kernel.Options{SubgroupSize: width, Diagnostics: true})
			if err != nil {
				t.Fatalf("scan path: %v", err)
			}

			wantIncl := make([]float32, group)
			wantExcl := make([]float32, group)
			err = kernel.Dispatch(&testkernels.SubgroupScanFallbackKernel, accel.ID3{X: 1},
				kernelabi.Args{Slices: []any{in, wantIncl, wantExcl, []uint32{width}}})
			if err != nil {
				t.Fatalf("fallback: %v", err)
			}

			for i := range incl {
				if math.Float32bits(incl[i]) != math.Float32bits(wantIncl[i]) {
					t.Fatalf("inclusive element %d: scan %v, fallback %v",
						i, incl[i], wantIncl[i])
				}
				if math.Float32bits(excl[i]) != math.Float32bits(wantExcl[i]) {
					t.Fatalf("exclusive element %d: scan %v, fallback %v",
						i, excl[i], wantExcl[i])
				}
			}
			// And the two scans are not the same function, or the comparison
			// above would hold for a lowering that emitted one for both.
			if width > 1 && incl[1] == excl[1] {
				t.Fatalf("the inclusive and exclusive scans agree at lane 1 (%v), and they "+
					"differ by that lane's own value", incl[1])
			}
		})
	}
}

// The authored scan kernel agrees with its generated lowering.
//
// At width 1, where the authored bodies are correct rather than stubs: a
// subgroup of one has an inclusive scan of the lane's own value and an
// exclusive scan of nothing, which is what `return v` and `return 0` compute.
// The two rows of spec 002 section 5.2 disagree there, so a stub that returned
// the argument for both would fail this.
func TestAuthoredScanKernel(t *testing.T) {
	const group = 64
	in := make([]float32, group)
	for i := range in {
		in[i] = float32(i)*0.5 - 3
	}

	authoredIncl := make([]float32, group)
	authoredExcl := make([]float32, group)
	size := kernel.ID3{X: group, Y: 1, Z: 1}
	for l := range uint32(group) {
		th := kernel.NewThreadWithSubgroup(kernel.ID3{X: l}, kernel.ID3{X: l},
			kernel.ID3{}, size, kernel.ID3{X: 1}, 1)
		testkernels.SubgroupScan(th, in, authoredIncl, authoredExcl)
	}

	incl := make([]float32, group)
	excl := make([]float32, group)
	err := kernel.DispatchCooperativeWith(&testkernels.SubgroupScanKernel,
		accel.ID3{X: 1}, kernelabi.Args{Slices: []any{in, incl, excl}},
		kernel.Options{SubgroupSize: 1, Diagnostics: true})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	for i := range incl {
		if math.Float32bits(authoredIncl[i]) != math.Float32bits(incl[i]) {
			t.Fatalf("inclusive element %d: authored %v, generated %v",
				i, authoredIncl[i], incl[i])
		}
		if math.Float32bits(authoredExcl[i]) != math.Float32bits(excl[i]) {
			t.Fatalf("exclusive element %d: authored %v, generated %v",
				i, authoredExcl[i], excl[i])
		}
	}
	if incl[7] != in[7] || excl[7] != 0 {
		t.Fatalf("at a subgroup of one, element 7's inclusive scan is its own value and its "+
			"exclusive scan is the identity: got %v and %v, want %v and 0",
			incl[7], excl[7], in[7])
	}
}

// The scan kernel requires the arithmetic capability and nothing else.
//
// Nothing else matters as much as the requirement itself: a kernel that
// declared more than it uses is unavailable on a device that could run it, and
// the symptom is a device being skipped, which nobody notices.
func TestTheScanKernelDeclaresItsCapabilities(t *testing.T) {
	caps := accel.Capability(testkernels.SubgroupScanKernel.Caps)
	if caps&accel.CapSubgroupArithmetic == 0 {
		t.Error("the record does not require CapSubgroupArithmetic, and the kernel sums " +
			"across lanes")
	}
	if extra := caps &^ accel.CapSubgroupArithmetic; extra != 0 {
		t.Errorf("the record also requires %v, which the body does not use", extra)
	}
	if got := testkernels.SubgroupScanFallbackKernel.Caps; got != 0 {
		t.Errorf("the fallback requires capabilities %d and should require none", got)
	}
}
