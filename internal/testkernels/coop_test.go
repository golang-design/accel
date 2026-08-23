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

// The mid-loop split, checked against the split that already worked.
//
// ReduceLoop puts its barrier inside a halving-stride loop, which needs the
// state machine to resume mid-loop and carry the induction variable across the
// epoch. ReduceUnrolled computes the same thing with its barriers at the top
// level, using only the transform this child had already checked. Compared bit
// for bit, so a disagreement is the new state numbering's and nothing else's.
//
// This is the same shape as spec 017's whole-plan oracle and 018's
// flat-versus-cooperative differential, and it is preferred to a golden of the
// generated code for the same reason: a golden says the shape changed, not
// whether it is right.
func TestTheLoopedAndUnrolledReductionsAgree(t *testing.T) {
	const groups, group = 4, 64
	in := make([]float32, groups*group)
	for i := range in {
		in[i] = float32(i)*0.5 - 3
	}

	run := func(k *accel.Kernel) []float32 {
		t.Helper()
		out := make([]float32, groups)
		err := kernel.DispatchCooperative(k, accel.ID3{X: groups},
			accel.KernelArgs{Slices: []any{in, out}})
		if err != nil {
			t.Fatalf("%s: %v", k.Name, err)
		}
		return out
	}

	looped := run(&testkernels.ReduceLoopKernel)
	unrolled := run(&testkernels.ReduceUnrolledKernel)
	for i := range looped {
		if math.Float32bits(looped[i]) != math.Float32bits(unrolled[i]) {
			t.Fatalf("workgroup %d: looped %v, unrolled %v", i, looped[i], unrolled[i])
		}
	}

	// And both are the right answer, not merely the same wrong one. The
	// reference sums in the same tree order, so the comparison is exact.
	for g := range groups {
		acc := make([]float32, group)
		copy(acc, in[g*group:(g+1)*group])
		for stride := group / 2; stride > 0; stride /= 2 {
			for i := range stride {
				acc[i] += acc[i+stride]
			}
		}
		if math.Float32bits(looped[g]) != math.Float32bits(acc[0]) {
			t.Fatalf("workgroup %d is %v, want %v", g, looped[g], acc[0])
		}
	}
}

// The looped reduction suspends once per round plus the initial one, which is
// what bounds the scheduler's epoch loop. A count taken from the state total
// would be wrong, because a loop's check and post states never suspend.
func TestTheLoopedReductionCountsItsSuspensions(t *testing.T) {
	if got := testkernels.ReduceLoopKernel.Suspensions; got != 2 {
		t.Errorf("ReduceLoop has %d suspension points, want 2: one before the loop "+
			"and one inside it, however many times the loop runs", got)
	}
	if got := testkernels.ReduceUnrolledKernel.Suspensions; got != 7 {
		t.Errorf("ReduceUnrolled has %d suspension points, want 7", got)
	}
}

// The authored halves of the two reductions, which is spec 004's fifth testing
// level: the generated lowering is what runs, so nothing else would ever call
// these functions, and a kernel nobody executes means whatever the IR made of
// it.
//
// The rendezvous is emulated the way the scheduler implements it — every
// invocation to the barrier, then every invocation past it — which for these
// kernels means running the whole function once per round, since each round's
// work is guarded by the stride and the earlier rounds are idempotent once
// their inputs have settled.
func TestAuthoredReductionsMatchTheirLowerings(t *testing.T) {
	const groups, group = 2, 64
	in := make([]float32, groups*group)
	for i := range in {
		in[i] = float32(i)*0.25 - 1
	}

	// The reference: the same tree order, so the float sum is exact against it.
	want := make([]float32, groups)
	for g := range groups {
		acc := make([]float32, group)
		copy(acc, in[g*group:(g+1)*group])
		for stride := group / 2; stride > 0; stride /= 2 {
			for i := range stride {
				acc[i] += acc[i+stride]
			}
		}
		want[g] = acc[0]
	}

	// The authored ReduceUnrolled, driven with an explicit rendezvous: seven
	// passes, one per barrier, each running every invocation.
	authored := make([]float32, groups)
	size := kernel.ID3{X: group, Y: 1, Z: 1}
	count := kernel.ID3{X: groups, Y: 1, Z: 1}
	for g := range uint32(groups) {
		var sh [group]float32
		accel.KernelPoison(sh[:])
		for range 7 {
			for l := range uint32(group) {
				th := kernel.NewThread(kernel.ID3{X: g*group + l},
					kernel.ID3{X: l}, kernel.ID3{X: g}, size, count)
				testkernels.ReduceUnrolled(th, in, authored, &sh)
			}
		}
	}
	for g := range groups {
		if math.Float32bits(authored[g]) != math.Float32bits(want[g]) {
			t.Fatalf("authored ReduceUnrolled workgroup %d is %v, want %v",
				g, authored[g], want[g])
		}
	}

	// And the generated lowering agrees with it.
	generated := make([]float32, groups)
	if err := kernel.DispatchCooperative(&testkernels.ReduceUnrolledKernel,
		accel.ID3{X: groups}, accel.KernelArgs{Slices: []any{in, generated}}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	for g := range groups {
		if math.Float32bits(authored[g]) != math.Float32bits(generated[g]) {
			t.Fatalf("workgroup %d: authored %v, generated %v", g, authored[g], generated[g])
		}
	}
}

// The authored Exchange and the reductions all declare shared memory, so the
// record carries its extent: a backend needs the size at pipeline creation,
// since it appears in the GLSL layout qualifier and in Metal's threadgroup
// attribute.
func TestSharedExtentsReachTheRecord(t *testing.T) {
	for _, c := range []struct {
		k    *accel.Kernel
		want []int
	}{
		{&testkernels.ExchangeKernel, []int{64}},
		{&testkernels.ReduceLoopKernel, []int{64}},
		{&testkernels.ReduceUnrolledKernel, []int{64}},
	} {
		if len(c.k.SharedSizes) != len(c.want) {
			t.Errorf("%s declares %v shared arrays, want %v", c.k.Name, c.k.SharedSizes, c.want)
			continue
		}
		for i := range c.want {
			if c.k.SharedSizes[i] != c.want[i] {
				t.Errorf("%s shared array %d has extent %d, want %d",
					c.k.Name, i, c.k.SharedSizes[i], c.want[i])
			}
		}
		if c.k.NewShared == nil {
			t.Errorf("%s declares shared memory and has no allocator", c.k.Name)
		}
	}
}
