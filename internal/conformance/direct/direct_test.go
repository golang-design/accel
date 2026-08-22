// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package direct_test

import (
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/direct"
)

// recorder is a kernel that writes down which invocations ran.
func recorder(size accel.ID3) (*accel.Kernel, *[]accel.ID3, *[]accel.ID3) {
	var globals, locals []accel.ID3
	k := &accel.Kernel{
		Name: "Record", WorkgroupSize: size, Generator: accel.KernelABIVersion,
		Bindings: []accel.KernelBinding{{Name: "out", DType: accel.KernelU32, Access: accel.KernelWrite}},
		Flat: func(t accel.Thread, a accel.KernelArgs) {
			globals = append(globals, t.GlobalID())
			locals = append(locals, t.LocalID())
			out := accel.KernelSlice[uint32](a, 0)
			if i := t.GlobalIndex(); int(i) < len(out) {
				out[i] = i + 1
			}
		},
	}
	return k, &globals, &locals
}

// TestRunCoversEveryInvocation checks that a dispatch runs the whole grid, in
// workgroups, with the ids a backend would supply.
func TestRunCoversEveryInvocation(t *testing.T) {
	size := accel.ID3{X: 2, Y: 2, Z: 1}
	k, globals, locals := recorder(size)
	count := accel.ID3{X: 3, Y: 1, Z: 1}

	out := make([]uint32, 12)
	if err := direct.Run(k, count, accel.KernelArgs{Slices: []any{out}}); err != nil {
		t.Fatalf("Run: %v", err)
	}

	if want := 12; len(*globals) != want {
		t.Fatalf("%d invocations, want %d", len(*globals), want)
	}
	for i, v := range out {
		if v != uint32(i)+1 {
			t.Errorf("element %d is %d, want %d: an invocation did not run", i, v, i+1)
		}
	}

	// Local ids stay inside the workgroup extent, which is what makes shared
	// memory addressable by them at M4.
	for i, l := range *locals {
		if l.X >= size.X || l.Y >= size.Y || l.Z >= size.Z {
			t.Fatalf("invocation %d has local id %+v, outside the %+v workgroup", i, l, size)
		}
	}

	// Global ids are the group's origin plus the local id, per spec 002.
	for i, g := range *globals {
		l := (*locals)[i]
		if (g.X-l.X)%size.X != 0 || (g.Y-l.Y)%size.Y != 0 {
			t.Errorf("invocation %d has global %+v and local %+v, which are not a workgroup apart", i, g, l)
		}
	}
}

// TestRunRefusesACooperativeKernel is the restriction that is not a
// simplification: a cooperative kernel's invocations rendezvous, so running
// them in sequence is a different program rather than a slower one.
func TestRunRefusesACooperativeKernel(t *testing.T) {
	k := &accel.Kernel{Name: "Reduce", WorkgroupSize: accel.ID3{X: 1, Y: 1, Z: 1}}
	err := direct.Run(k, accel.ID3{X: 1}, accel.KernelArgs{})
	if err == nil {
		t.Fatal("a kernel with no flat entry point ran")
	}
	for _, want := range []string{"Reduce", "cooperative", "rendezvous"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not carry %q", err, want)
		}
	}
}

// TestRunRejections covers what must fail before anything executes.
func TestRunRejections(t *testing.T) {
	ok := func() *accel.Kernel {
		k, _, _ := recorder(accel.ID3{X: 4, Y: 1, Z: 1})
		return k
	}

	if err := direct.Run(nil, accel.ID3{X: 1}, accel.KernelArgs{}); err == nil {
		t.Error("a nil kernel ran")
	}

	zero := ok()
	zero.WorkgroupSize = accel.ID3{X: 4, Y: 0, Z: 1}
	if err := direct.Run(zero, accel.ID3{X: 1}, accel.KernelArgs{Slices: []any{[]uint32{}}}); err == nil {
		t.Error("a workgroup with a zero axis ran")
	} else if !strings.Contains(err.Error(), "dispatches nothing") {
		t.Errorf("error %q does not say why", err)
	}

	mismatched := ok()
	if err := direct.Run(mismatched, accel.ID3{X: 1}, accel.KernelArgs{Slices: []any{[]float32{}}}); err == nil {
		t.Error("a mismatched argument set ran")
	}

	stale := ok()
	stale.Generator = accel.KernelABIVersion - 1
	if err := direct.Run(stale, accel.ID3{X: 1}, accel.KernelArgs{Slices: []any{[]uint32{}}}); err == nil {
		t.Error("a kernel generated against an older ABI ran")
	}
}

// TestRunTreatsAZeroGridAsOne checks the axis default, so a one-dimensional
// dispatch does not have to spell out two ones.
func TestRunTreatsAZeroGridAsOne(t *testing.T) {
	k, globals, _ := recorder(accel.ID3{X: 2, Y: 1, Z: 1})
	out := make([]uint32, 2)
	if err := direct.Run(k, accel.ID3{X: 1}, accel.KernelArgs{Slices: []any{out}}); err != nil {
		t.Fatal(err)
	}
	if len(*globals) != 2 {
		t.Errorf("%d invocations, want 2: a zero grid axis means one workgroup", len(*globals))
	}
}

// TestGroupsRoundsUp is why a kernel carries a bounds check: the last workgroup
// runs invocations past the end of the data, and nothing but the kernel's own
// guard stops them writing there.
func TestGroupsRoundsUp(t *testing.T) {
	for _, tc := range []struct{ n, extent, want uint32 }{
		{0, 64, 0},
		{1, 64, 1},
		{64, 64, 1},
		{65, 64, 2},
		{128, 64, 2},
		{129, 64, 3},
		{10, 1, 10},
		{10, 0, 0},
	} {
		if got := direct.Groups(tc.n, tc.extent); got != tc.want {
			t.Errorf("Groups(%d, %d) = %d, want %d", tc.n, tc.extent, got, tc.want)
		}
	}
}

// TestCoverIsTheOneDimensionalCase covers the helper a test reaches for.
func TestCoverIsTheOneDimensionalCase(t *testing.T) {
	k, _, _ := recorder(accel.ID3{X: 64, Y: 1, Z: 1})
	if got := direct.Cover(k, 130); got != (accel.ID3{X: 3, Y: 1, Z: 1}) {
		t.Errorf("Cover(130) = %+v, want 3,1,1", got)
	}
	if got := direct.Cover(nil, 4); got != (accel.ID3{}) {
		t.Errorf("Cover(nil) = %+v, want the zero grid", got)
	}
	if got := direct.Cover(k, -1); got != (accel.ID3{}) {
		t.Errorf("Cover(-1) = %+v, want the zero grid", got)
	}
}

// TestRunIsDeterministic checks that two runs of a flat kernel produce the same
// invocation order. A flat kernel has no ordering guarantee on a real device,
// and this executor is an oracle: an oracle that varied its own order would
// make a failure impossible to reproduce.
func TestRunIsDeterministic(t *testing.T) {
	first, second := make([]uint32, 16), make([]uint32, 16)
	for _, out := range [][]uint32{first, second} {
		k, _, _ := recorder(accel.ID3{X: 4, Y: 1, Z: 1})
		if err := direct.Run(k, direct.Cover(k, len(out)), accel.KernelArgs{Slices: []any{out}}); err != nil {
			t.Fatal(err)
		}
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("element %d differs between runs: %d and %d", i, first[i], second[i])
		}
	}
}
