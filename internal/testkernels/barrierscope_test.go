// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
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
