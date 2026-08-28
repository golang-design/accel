// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"fmt"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
	"golang.design/x/accel/kernelabi"
)

// A lane publishes through a storage buffer and every lane reads what it wrote.
//
// specs/050-barrier-scopes.md §3's first assertion and the accepting half of
// the whole spec: this is the handoff 002 §2.5 says BarrierStorage orders. The
// payload crosses a []uint32 rather than a shared array, because a shared array
// is the class the cheaper barrier already orders and a kernel publishing
// through one would pass at every scope.
func TestAStorageBarrierPublishesToTheWorkgroup(t *testing.T) {
	const groups = 3
	width := int(testkernels.PublishStorageKernel.WorkgroupSize.X)

	scratch := make([]uint32, groups)
	out := make([]uint32, groups*width)
	if err := kernel.DispatchCooperative(&testkernels.PublishStorageKernel,
		accel.ID3{X: groups},
		kernelabi.Args{Slices: []any{scratch, out}}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	for g := range groups {
		want := uint32(g)*1000 + 7
		for lane := range width {
			if got := out[g*width+lane]; got != want {
				t.Fatalf("workgroup %d lane %d read %d, want %d: the value lane 0 wrote "+
					"to the storage buffer was not visible across the barrier",
					g, lane, got, want)
			}
		}
	}
}

// A shared barrier orders the shared array, which is what it is for.
//
// BarrierShared's own accepting half. Without it, asserting only that the
// emitted MSL carries a narrower mem_flags mask would leave a lowering that
// emitted the right text and rendezvoused nobody indistinguishable from a
// correct one.
func TestASharedBarrierOrdersTheSharedArray(t *testing.T) {
	const groups = 2
	width := int(testkernels.PublishSharedKernel.WorkgroupSize.X)

	out := make([]uint32, groups*width)
	if err := kernel.DispatchCooperative(&testkernels.PublishSharedKernel,
		accel.ID3{X: groups},
		kernelabi.Args{Slices: []any{out}}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	for g := range groups {
		for lane := range width {
			// Each lane reads the lane above it, wrapping.
			want := uint32(g)*1000 + uint32((lane+1)%width)
			if got := out[g*width+lane]; got != want {
				t.Fatalf("workgroup %d lane %d read %d, want %d: its neighbour's write "+
					"was not visible across the shared barrier", g, lane, got, want)
			}
		}
	}
}

// Each barrier lowers to the mem_flags mask 002 §2.5's table gives.
//
// specs/050-barrier-scopes.md §3's second and third assertions, asserted on the
// **emitted text** rather than on a result. That is forced rather than chosen:
// a kernel whose data fits in one threadgroup gets the right answer at any of
// the three scopes, and Apple hardware may make a storage write visible across
// a threadgroup barrier regardless -- "may" being what undefined behaviour
// looks like from the inside. §2.5's table gives the target text per backend,
// which is what makes the claim checkable at all.
func TestEachBarrierScopeLowersToItsOwnMask(t *testing.T) {
	for _, c := range []struct {
		kernel *accel.Kernel
		want   string
		absent string
	}{
		// Shared and storage, so BarrierStorage's own mask must not be the
		// whole of it.
		{
			&testkernels.RMSNormKernel,
			"threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)",
			"",
		},
		// Shared only. The absent string is what says it is not the default
		// barrier's text with a prefix: mem_device must not appear at all.
		{
			&testkernels.PublishSharedKernel,
			"threadgroup_barrier(mem_flags::mem_threadgroup)",
			"mem_device",
		},
		// Storage only.
		{
			&testkernels.PublishStorageKernel,
			"threadgroup_barrier(mem_flags::mem_device)",
			"mem_threadgroup",
		},
	} {
		t.Run(c.kernel.Name, func(t *testing.T) {
			if c.kernel.MSL == "" {
				t.Fatalf("%s carries no MSL, so this asserts nothing", c.kernel.Name)
			}
			if !strings.Contains(c.kernel.MSL, c.want) {
				t.Errorf("%s does not emit %q", c.kernel.Name, c.want)
			}
			if c.absent != "" && strings.Contains(c.kernel.MSL, c.absent) {
				t.Errorf("%s emits %q, and this barrier's scope does not include it",
					c.kernel.Name, c.absent)
			}
		})
	}
}

// A masked barrier is still a suspension point.
//
// The execution half is identical across the three scopes -- they differ only
// in what they make visible -- so a lowering that treated a masked one as an
// ordinary statement would produce a kernel with fewer states, running past a
// rendezvous its peers are waiting at. The count is in the source rather than
// per execution, which is why it can be pinned.
func TestAMaskedBarrierIsASuspensionPoint(t *testing.T) {
	for _, c := range []struct {
		kernel *accel.Kernel
		want   int
	}{
		{&testkernels.PublishStorageKernel, 1},
		{&testkernels.PublishSharedKernel, 1},
	} {
		if got := c.kernel.Suspensions; got != c.want {
			t.Errorf("%s has %d suspension points, want %d: a masked barrier "+
				"rendezvouses exactly as Barrier does", c.kernel.Name, got, c.want)
		}
	}
}

// A subgroup barrier rendezvouses one subgroup, at every emulated size.
//
// specs/050-barrier-scopes.md §3's fourth assertion and 002 §5.3. The sweep is
// what makes it meaningful: at size 1 every lane is its own subgroup and reads
// what it wrote, at 64 there is a single subgroup spanning the workgroup, and
// the answer is the same only if the indexing is right. 002 §5.4 gives the
// sweep's sizes and the reason -- an emulated size is a test instrument rather
// than a model of hardware.
func TestASubgroupBarrierPublishesWithinItsSubgroup(t *testing.T) {
	width := int(testkernels.SubgroupPublishKernel.WorkgroupSize.X)

	for _, size := range []uint32{1, 4, 32, 64} {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			out := make([]float32, width)
			err := kernel.DispatchCooperativeWith(&testkernels.SubgroupPublishKernel,
				accel.ID3{X: 1}, kernelabi.Args{Slices: []any{out}},
				kernel.Options{Diagnostics: true, SubgroupSize: size})
			if err != nil {
				t.Fatalf("dispatch at subgroup size %d: %v", size, err)
			}
			for lane := range width {
				// Lane l is in subgroup l/size, and that subgroup's lowest lane
				// wrote its index plus one.
				want := float32(uint32(lane)/size) + 1
				if got := out[lane]; got != want {
					t.Fatalf("at subgroup size %d, lane %d read %v, want %v: it did not "+
						"see its own subgroup's write", size, lane, got, want)
				}
			}
		})
	}
}

// A subgroup barrier lowers to simdgroup_barrier, not to a workgroup one.
//
// The scope is the set of *lanes*, and the memory mask stays wide: 002 §2.5's
// table gives this call shared and storage visibility, so what narrows is who
// rendezvouses. A lowering that emitted threadgroup_barrier would be correct on
// every result a test can produce -- a wider rendezvous includes the narrower
// one -- and wrong about what the caller asked for, which is why this is
// asserted on the text.
func TestASubgroupBarrierLowersToSubgroupScope(t *testing.T) {
	src := testkernels.SubgroupPublishKernel.MSL
	if src == "" {
		t.Fatal("SubgroupPublish carries no MSL, so this asserts nothing")
	}
	const want = "simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)"
	if !strings.Contains(src, want) {
		t.Errorf("SubgroupPublish does not emit %q", want)
	}
	if strings.Contains(src, "threadgroup_barrier(") {
		t.Error("SubgroupPublish emits a threadgroup_barrier, which rendezvouses the " +
			"whole workgroup rather than one subgroup")
	}
}

// Subgroups may sit at different subgroup barriers, and the dispatch completes.
//
// This is what the per-subgroup arrival check is *for*, and the case
// SubgroupPublish cannot reach: its barrier is at the top level, so every lane
// is at it and a workgroup-wide check passes anyway. Here the loop's trip count
// is SubgroupIndex, so at any epoch the subgroups are at different states --
// legal by 002 §5.3, and reported as a non-uniform arrival by a check that
// compares every invocation against one expected barrier.
//
// Diagnostics are on, which is what makes the assertion an assertion: the check
// being exercised lives in the CPU backend's developer mode
// (specs/006-backends.md §5), so running without it would pass whatever the
// scheduler did.
func TestSubgroupsMaySitAtDifferentSubgroupBarriers(t *testing.T) {
	width := int(testkernels.SubgroupStaggerKernel.WorkgroupSize.X)

	// Size 4 gives 16 subgroups over the workgroup, so the trip counts range
	// from 0 to 15 and most epochs have subgroups genuinely out of step. Size
	// 64 is the degenerate single subgroup, kept so the sweep says the shape
	// works when there is nothing to stagger.
	for _, size := range []uint32{4, 16, 64} {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			out := make([]float32, width)
			err := kernel.DispatchCooperativeWith(&testkernels.SubgroupStaggerKernel,
				accel.ID3{X: 1}, kernelabi.Args{Slices: []any{out}},
				kernel.Options{Diagnostics: true, SubgroupSize: size})
			if err != nil {
				t.Fatalf("at subgroup size %d the dispatch failed, and every lane of each "+
					"subgroup reaches its own barriers: %v", size, err)
			}
			for lane := range width {
				want := float32(uint32(lane)/size) + 1
				if got := out[lane]; got != want {
					t.Fatalf("at subgroup size %d, lane %d is %v, want %v", size, lane, got, want)
				}
			}
		})
	}
}

// The authored barrier-scope kernels and their generated lowerings agree.
//
// specs/010-kernel-corpus.md §6's obligation, and the one whose absence CI
// catches as a coverage number rather than as a name: every other test here
// runs the generated form, so nothing calls the authored function, and on Linux
// the Metal differential is not there to cover the lowering either.
//
// RunAuthored gives each invocation a goroutine behind a cyclic barrier, which
// is what a workgroup is. That rendezvous is workgroup-wide, so the two
// subgroup kernels are run one subgroup at a time -- a subgroup barrier
// rendezvouses its own lanes and this is how the authored form expresses that.
func TestTheAuthoredBarrierScopeKernelsMatchTheirLowerings(t *testing.T) {
	t.Run("PublishStorage", func(t *testing.T) {
		const groups = 3
		width := int(testkernels.PublishStorageKernel.WorkgroupSize.X)

		authored := make([]uint32, groups*width)
		scratch := make([]uint32, groups)
		for g := range uint32(groups) {
			kernel.RunAuthored(&testkernels.PublishStorageKernel, kernel.ID3{X: g},
				kernel.ID3{X: groups}, 128, func(th kernel.Thread) {
					testkernels.PublishStorage(th, scratch, authored)
				})
		}

		generated := make([]uint32, groups*width)
		if err := kernel.DispatchCooperative(&testkernels.PublishStorageKernel,
			accel.ID3{X: groups},
			kernelabi.Args{Slices: []any{make([]uint32, groups), generated}}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range authored {
			if authored[i] != generated[i] {
				t.Fatalf("element %d: authored %d, generated %d", i, authored[i], generated[i])
			}
		}
	})

	t.Run("PublishShared", func(t *testing.T) {
		const groups = 2
		width := int(testkernels.PublishSharedKernel.WorkgroupSize.X)

		authored := make([]uint32, groups*width)
		for g := range uint32(groups) {
			var sh [32]uint32
			kernelabi.Poison(sh[:])
			kernel.RunAuthored(&testkernels.PublishSharedKernel, kernel.ID3{X: g},
				kernel.ID3{X: groups}, 128, func(th kernel.Thread) {
					testkernels.PublishShared(th, authored, &sh)
				})
		}

		generated := make([]uint32, groups*width)
		if err := kernel.DispatchCooperative(&testkernels.PublishSharedKernel,
			accel.ID3{X: groups},
			kernelabi.Args{Slices: []any{generated}}); err != nil {
			t.Fatalf("dispatch: %v", err)
		}
		for i := range authored {
			if authored[i] != generated[i] {
				t.Fatalf("element %d: authored %d, generated %d", i, authored[i], generated[i])
			}
		}
	})

	// The two subgroup kernels, at an emulated size that gives more than one
	// subgroup per workgroup. At the degenerate single subgroup a subgroup
	// rendezvous and a workgroup one release the same lanes, so the comparison
	// would say nothing about the scope at all.
	for _, c := range []struct {
		name string
		k    *accel.Kernel
		run  func(th kernel.Thread, out []float32, sh *[64]float32)
	}{
		{"SubgroupPublish", &testkernels.SubgroupPublishKernel,
			func(th kernel.Thread, out []float32, sh *[64]float32) {
				testkernels.SubgroupPublish(th, out, sh)
			}},
		{"SubgroupStagger", &testkernels.SubgroupStaggerKernel,
			func(th kernel.Thread, out []float32, sh *[64]float32) {
				testkernels.SubgroupStagger(th, out, sh)
			}},
	} {
		t.Run(c.name, func(t *testing.T) {
			const size = 16
			width := int(c.k.WorkgroupSize.X)

			// One subgroup at a time. RunAuthored's rendezvous spans everything
			// it launches, so launching the workgroup would make a subgroup
			// barrier wait for lanes of other subgroups -- which is what
			// SubgroupStagger deliberately does not do, and would deadlock.
			authored := make([]float32, width)
			var sh [64]float32
			kernelabi.Poison(sh[:])
			kernel.RunAuthored(c.k, kernel.ID3{}, kernel.ID3{X: 1}, size,
				func(th kernel.Thread) { c.run(th, authored, &sh) })

			generated := make([]float32, width)
			err := kernel.DispatchCooperativeWith(c.k, accel.ID3{X: 1},
				kernelabi.Args{Slices: []any{generated}},
				kernel.Options{Diagnostics: true, SubgroupSize: size})
			if err != nil {
				t.Fatalf("dispatch: %v", err)
			}
			for i := range authored {
				if authored[i] != generated[i] {
					t.Fatalf("element %d: authored %v, generated %v",
						i, authored[i], generated[i])
				}
			}
		})
	}
}
