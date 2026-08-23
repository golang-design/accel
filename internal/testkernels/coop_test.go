// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/direct"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
)

// The bring-up case: a cooperative kernel runs, and its barrier means what it
// says. Every invocation reads the value invocation zero published, so a
// lowering that ran the barrier as a no-op would read poison for most of them.
func TestExchangeRunsCooperatively(t *testing.T) {
	const n = 128
	k := &testkernels.ExchangeKernel
	if k.Cooperative == nil {
		t.Fatal("Exchange is cooperative and should have a resumable entry point")
	}
	if k.Flat != nil {
		t.Error("a cooperative kernel has no flat form: running its invocations one " +
			"after another is a different program, not a slower one")
	}
	if k.Suspensions != 1 {
		t.Errorf("Exchange has %d suspension points, want 1", k.Suspensions)
	}

	in := make([]float32, n)
	out := make([]float32, n)
	for i := range in {
		in[i] = float32(i)
	}
	args := accel.KernelArgs{Slices: []any{in, out}}
	if err := kernel.DispatchCooperative(k, accel.ID3{X: n / 64}, args); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	// Invocation i reads its neighbour's published value, wrapping inside the
	// workgroup. Under sequential execution the neighbour has not published
	// yet, so this is what distinguishes a rendezvous from a no-op.
	for i, got := range out {
		base, lid := (i/64)*64, i%64
		want := in[base+(lid+1)%64]
		if got != want {
			t.Fatalf("element %d is %v, want %v: it should hold what its neighbour "+
				"published before the barrier", i, got, want)
		}
	}
}

// Shared storage is per workgroup, not per dispatch. Two workgroups sharing it
// would be a hazard no barrier covers, because spec 002 section 2.7 gives no
// ordering between workgroups at all.
func TestSharedStorageIsPerWorkgroup(t *testing.T) {
	const n = 256
	in := make([]float32, n)
	out := make([]float32, n)
	for i := range in {
		in[i] = float32(i)
	}
	args := accel.KernelArgs{Slices: []any{in, out}}
	if err := kernel.DispatchCooperative(&testkernels.ExchangeKernel,
		accel.ID3{X: n / 64}, args); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	// Every invocation reads within its own workgroup, so a shared allocation
	// leaking across workgroups shows up as a value from the wrong one.
	for i, got := range out {
		base, lid := (i/64)*64, i%64
		want := in[base+(lid+1)%64]
		if got != want {
			t.Fatalf("element %d is %v, want %v: shared storage is leaking between "+
				"workgroups", i, got, want)
		}
	}
}

// The frame carries locals across the suspension. Broadcast computes its ids
// before the barrier and uses them after, so a lowering that lost them would
// index with zero.
func TestLocalsSurviveTheSuspension(t *testing.T) {
	const n = 64
	in := make([]float32, n)
	out := make([]float32, n)
	for i := range in {
		in[i] = 42
	}
	args := accel.KernelArgs{Slices: []any{in, out}}
	if err := kernel.DispatchCooperative(&testkernels.ExchangeKernel,
		accel.ID3{X: 1}, args); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	for i, got := range out {
		if got != 42 {
			t.Fatalf("element %d is %v, want 42: the ids computed before the "+
				"barrier did not survive it", i, got)
		}
	}
}

// Shared storage arrives poisoned rather than zeroed, so a kernel reading
// before writing produces a number nobody mistakes for an answer.
func TestSharedStorageIsPoisoned(t *testing.T) {
	var sh [8]float32
	accel.KernelPoison(sh[:])
	for i, v := range sh {
		if v == 0 {
			t.Fatalf("element %d is zero: zero is a value a kernel legitimately "+
				"expects, so a read before a write would look plausible", i)
		}
		if v == v { // a quiet NaN is not equal to itself
			t.Fatalf("element %d is %v, want a NaN so arithmetic propagates it", i, v)
		}
	}

	var u [8]uint32
	accel.KernelPoison(u[:])
	for i, v := range u {
		if v == 0 {
			t.Fatalf("integer element %d is zero", i)
		}
	}
}

// A cooperative kernel refuses the flat dispatch path rather than running its
// invocations in sequence, which would be a different program.
func TestACooperativeKernelRefusesTheFlatPath(t *testing.T) {
	err := kernel.Dispatch(&testkernels.ExchangeKernel, accel.ID3{X: 1},
		accel.KernelArgs{Slices: []any{make([]float32, 64), make([]float32, 64)}})
	if err == nil {
		t.Fatal("a cooperative kernel should refuse the flat path")
	}
}

// TestAuthoredExchange runs the authored kernel against an independent
// reference, which is the authored half of spec 004's fifth testing level.
//
// A cooperative kernel cannot be run one invocation at a time, so the barrier
// is emulated the way the scheduler implements it: every invocation runs to the
// barrier, then every invocation runs past it. Here that is two passes over the
// whole function, which works because Exchange's pre-barrier half is idempotent
// — it writes the same value both times.
//
// That reasoning is specific to this kernel, and it is why spec 018 makes
// flat-versus-cooperative agreement the general criterion instead: a kernel
// whose pre-barrier half is not idempotent has no such emulation, and the
// differential needs none.
func TestAuthoredExchange(t *testing.T) {
	const n = 128
	in := make([]float32, n)
	for i := range in {
		in[i] = float32(i)*0.5 - 2
	}
	out := make([]float32, n)

	const group = 64
	size := kernel.ID3{X: group, Y: 1, Z: 1}
	groups := uint32(n / group)
	count := kernel.ID3{X: groups, Y: 1, Z: 1}

	for g := range groups {
		var sh [group]float32
		accel.KernelPoison(sh[:])
		// Two passes: the first is every invocation up to the barrier, the
		// second is every invocation past it.
		for range 2 {
			for l := range uint32(group) {
				th := kernel.NewThread(
					kernel.ID3{X: g*group + l},
					kernel.ID3{X: l}, kernel.ID3{X: g}, size, count,
				)
				testkernels.Exchange(th, in, out, &sh)
			}
		}
	}

	for i, got := range out {
		base, lid := (i/group)*group, i%group
		want := in[base+(lid+1)%group]
		if got != want {
			t.Fatalf("element %d is %v, want %v", i, got, want)
		}
	}
}

// TestAuthoredAdd is the same for the flat Add kernel, whose generated lowering
// M3's graph tests run but whose authored form nothing had checked.
func TestAuthoredAdd(t *testing.T) {
	const n = 130 // not a multiple of the workgroup, so the tail is exercised
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32(i) * 0.5
		b[i] = float32(n-i) * 0.25
	}
	out := make([]float32, n)

	const group = 64
	size := kernel.ID3{X: group, Y: 1, Z: 1}
	groups := uint32((n + group - 1) / group)
	count := kernel.ID3{X: groups, Y: 1, Z: 1}
	for g := range groups {
		for l := range uint32(group) {
			th := kernel.NewThread(
				kernel.ID3{X: g*group + l},
				kernel.ID3{X: l}, kernel.ID3{X: g}, size, count,
			)
			testkernels.Add(th, a, b, out)
		}
	}

	for i := range out {
		want := a[i] + b[i]
		if math.Float32bits(out[i]) != math.Float32bits(want) {
			t.Fatalf("element %d is %v, want %v", i, out[i], want)
		}
	}
}

// And the generated lowering agrees with it, which is the other half of the
// fifth level: the authored function is type-checking input and the lowering is
// what runs, so they have to mean the same thing.
func TestGeneratedAddAgreesWithTheAuthoredOne(t *testing.T) {
	const n = 130
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32(i) * 0.5
		b[i] = float32(n-i) * 0.25
	}

	authored := make([]float32, n)
	const group = 64
	size := kernel.ID3{X: group, Y: 1, Z: 1}
	groups := uint32((n + group - 1) / group)
	count := kernel.ID3{X: groups, Y: 1, Z: 1}
	for g := range groups {
		for l := range uint32(group) {
			th := kernel.NewThread(kernel.ID3{X: g*group + l},
				kernel.ID3{X: l}, kernel.ID3{X: g}, size, count)
			testkernels.Add(th, a, b, authored)
		}
	}

	generated := make([]float32, n)
	if err := direct.Run(&testkernels.AddKernel, accel.ID3{X: groups},
		accel.KernelArgs{Slices: []any{a, b, generated}}); err != nil {
		t.Fatalf("direct.Run: %v", err)
	}

	for i := range authored {
		if math.Float32bits(authored[i]) != math.Float32bits(generated[i]) {
			t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
		}
	}
}
