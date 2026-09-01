// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernel"
)

// readerKernel is a cooperative kernel that reads shared element `at` without
// anything having written it, after storing `stored` there directly.
//
// Directly, bypassing the tracker, which is how the bit-pattern sweep works:
// the memory holds a chosen value and nothing told the tracker it was written,
// which is exactly what a read of another workgroup's leftovers looks like.
func readerKernel(stored float32, at int) *kernel.Kernel {
	return &kernel.Kernel{
		Name: "Reader", WorkgroupSize: kernel.ID3{X: 4, Y: 1, Z: 1},
		Generator: kernel.ABIVersion, SharedSizes: []int{8},
		NewShared: func() []any {
			var sh [8]float32
			for i := range sh {
				sh[i] = stored
			}
			return []any{&sh}
		},
		Cooperative: func(t kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			sh := kernel.SharedSlice[[8]float32](a, 0)
			_ = sh[f.Shared.ReadAt(0, at)]
			return false
		},
	}
}

// A read of shared memory nothing wrote is reported for **every** stored bit
// pattern, which is spec 019's strong form.
//
// A sentinel-based implementation fails this by construction: a sentinel is a
// value the kernel could compute, so a check comparing against one misses the
// read for whichever pattern it chose as the sentinel. A shadow bit does not
// look at the value at all.
func TestAnUndefinedReadIsReportedForEveryStoredPattern(t *testing.T) {
	patterns := []uint32{
		0x00000000, // zero, the value a naive check treats as undefined
		0x7FC0DEAD, // this backend's own poison, which a check might special-case
		0x7FC00000, // a quiet NaN
		0x7F800000, // +Inf
		0xFF800000, // -Inf
		0x3F800000, // 1.0
		0x80000000, // -0.0
		0xFFFFFFFF, // all bits set
		0x00000001, // the smallest subnormal
		0xDEADBEEF,
	}
	// Plus a sweep of the exponent field, since a check keyed on NaN-ness would
	// pass the list above and fail here.
	for e := range uint32(256) {
		patterns = append(patterns, e<<23|0x0055AA)
	}

	for _, bits := range patterns {
		stored := math.Float32frombits(bits)
		err := kernel.DispatchCooperative(readerKernel(stored, 3), kernel.ID3{X: 1},
			kernel.Args{})
		if err == nil {
			t.Fatalf("a read of undefined shared memory holding 0x%08X was not reported: "+
				"the check is looking at the value, so a kernel that computed that value "+
				"would be reported too", bits)
		}
		if !strings.Contains(err.Error(), "undefined shared memory") {
			t.Fatalf("0x%08X: got %v", bits, err)
		}
	}
}

// A read of something this workgroup did write is not reported, or the check
// above would be passing because it reports everything.
func TestADefinedReadIsNotReported(t *testing.T) {
	k := &kernel.Kernel{
		Name: "Writer", WorkgroupSize: kernel.ID3{X: 4, Y: 1, Z: 1},
		Generator: kernel.ABIVersion, SharedSizes: []int{8},
		NewShared: func() []any {
			var sh [8]float32
			return []any{&sh}
		},
		Cooperative: func(t kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			sh := kernel.SharedSlice[[8]float32](a, 0)
			i := int(t.LocalID().X)
			f.Shared.Write(0, i)
			sh[i] = 1
			_ = sh[f.Shared.ReadAt(0, i)]
			return false
		},
	}
	if err := kernel.DispatchCooperative(k, kernel.ID3{X: 1}, kernel.Args{}); err != nil {
		t.Fatalf("a kernel reading what it wrote should not be reported: %v", err)
	}
}

// Definedness is per workgroup: what one workgroup wrote does not define the
// next one's storage, because the storage is fresh.
func TestDefinednessDoesNotCarryBetweenWorkgroups(t *testing.T) {
	var pass int
	k := &kernel.Kernel{
		Name: "Leak", WorkgroupSize: kernel.ID3{X: 1, Y: 1, Z: 1},
		Generator: kernel.ABIVersion, SharedSizes: []int{4},
		NewShared: func() []any {
			var sh [4]float32
			return []any{&sh}
		},
		Cooperative: func(t kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			sh := kernel.SharedSlice[[4]float32](a, 0)
			if t.GroupID().X == 0 {
				f.Shared.Write(0, 0)
				sh[0] = 1
			} else {
				// The second workgroup reads without writing. If definedness
				// carried, this would be silently accepted.
				_ = sh[f.Shared.ReadAt(0, 0)]
			}
			pass++
			return false
		},
	}
	err := kernel.DispatchCooperative(k, kernel.ID3{X: 2}, kernel.Args{})
	if err == nil {
		t.Fatal("the second workgroup read storage nothing wrote and was not reported")
	}
}

// Two invocations touching one element between the same pair of barriers, with
// at least one writing, is a race and is reported with both invocations.
func TestAConflictingAccessIsReported(t *testing.T) {
	k := &kernel.Kernel{
		Name: "Racer", WorkgroupSize: kernel.ID3{X: 4, Y: 1, Z: 1},
		Generator: kernel.ABIVersion, SharedSizes: []int{4},
		NewShared: func() []any {
			var sh [4]float32
			return []any{&sh}
		},
		Cooperative: func(t kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			sh := kernel.SharedSlice[[4]float32](a, 0)
			// Every invocation writes element 0, with no barrier between them.
			f.Shared.Write(0, 0)
			sh[0] = float32(t.LocalID().X)
			return false
		},
	}
	err := kernel.DispatchCooperative(k, kernel.ID3{X: 1}, kernel.Args{})
	if err == nil {
		t.Fatal("four invocations writing one element with nothing ordering them is a race")
	}
	if !strings.Contains(err.Error(), "conflicting access") {
		t.Fatalf("got %v", err)
	}
	if !strings.Contains(err.Error(), "against invocation") {
		t.Error("a conflict report names both invocations, or a reader cannot find the pair")
	}
}

// Two invocations reading one element is not a conflict. Without this the
// check above would be reporting every shared access.
func TestConcurrentReadsAreNotAConflict(t *testing.T) {
	k := &kernel.Kernel{
		Name: "Readers", WorkgroupSize: kernel.ID3{X: 4, Y: 1, Z: 1},
		Generator: kernel.ABIVersion, SharedSizes: []int{4},
		NewShared: func() []any {
			var sh [4]float32
			return []any{&sh}
		},
		Cooperative: func(t kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			sh := kernel.SharedSlice[[4]float32](a, 0)
			if f.Pass == 0 {
				// Invocation zero publishes; everyone reads after the barrier.
				if t.LocalID().X == 0 {
					f.Shared.Write(0, 0)
					sh[0] = 1
				}
				f.Pass = 1
				f.Barrier = kernel.BarrierID{Index: 0, Pos: "k.go:1:1"}
				return true
			}
			_ = sh[f.Shared.ReadAt(0, 0)]
			return false
		},
		Suspensions: 1,
	}
	if err := kernel.DispatchCooperative(k, kernel.ID3{X: 1}, kernel.Args{}); err != nil {
		t.Fatalf("four invocations reading one element is not a race: %v", err)
	}
}

// A barrier ends the window conflicts are compared in, which is what a barrier
// means. Without it every publish-then-read kernel would report a conflict.
func TestABarrierSeparatesConflictingAccesses(t *testing.T) {
	k := &kernel.Kernel{
		Name: "Ordered", WorkgroupSize: kernel.ID3{X: 2, Y: 1, Z: 1},
		Generator: kernel.ABIVersion, SharedSizes: []int{4}, Suspensions: 1,
		NewShared: func() []any {
			var sh [4]float32
			return []any{&sh}
		},
		Cooperative: func(t kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			sh := kernel.SharedSlice[[4]float32](a, 0)
			if f.Pass == 0 {
				if t.LocalID().X == 0 {
					f.Shared.Write(0, 0)
					sh[0] = 1
				}
				f.Pass = 1
				f.Barrier = kernel.BarrierID{Index: 0, Pos: "k.go:1:1"}
				return true
			}
			// A different invocation reads it, after the barrier.
			if t.LocalID().X == 1 {
				_ = sh[f.Shared.ReadAt(0, 0)]
			}
			return false
		},
	}
	if err := kernel.DispatchCooperative(k, kernel.ID3{X: 1}, kernel.Args{}); err != nil {
		t.Fatalf("a write and a read separated by a barrier are ordered: %v", err)
	}
}

// A nil tracker is every call a no-op, which is how a dispatch with diagnostics
// off pays nothing for a check the default wants.
func TestANilTrackerIsInert(t *testing.T) {
	var tr *kernel.SharedTracker
	tr.Begin(kernel.ID3{})
	tr.Write(0, 0)
	tr.Read(0, 0)
	tr.Epoch()
	tr.Reset(kernel.ID3{})
	if got := tr.ReadAt(0, 7); got != 7 {
		t.Errorf("ReadAt on a nil tracker returned %d, want the index unchanged", got)
	}
	if ds := tr.Diagnostics(); ds != nil {
		t.Errorf("a nil tracker reports %v", ds)
	}
}

// Diagnostics carry what a reader needs to act, and report the same order every
// run: a report whose order changes is one a developer learns to re-run past.
func TestDiagnosticsAreOrderedAndComplete(t *testing.T) {
	err := kernel.DispatchCooperative(readerKernel(0, 2), kernel.ID3{X: 1}, kernel.Args{})
	if err == nil {
		t.Fatal("expected a diagnostic")
	}
	var ds kernel.Diagnostics
	if !errors.As(err, &ds) {
		t.Fatalf("got %T, want kernel.Diagnostics", err)
	}
	first := ds.Error()
	for range 20 {
		again := kernel.DispatchCooperative(readerKernel(0, 2), kernel.ID3{X: 1}, kernel.Args{})
		if again.Error() != first {
			t.Fatalf("the report differs between runs:\n%s\nwant\n%s", again, first)
		}
	}
	for _, want := range []string{"Reader", "workgroup", "invocation", "element 2"} {
		if !strings.Contains(first, want) {
			t.Errorf("the report should say %q, got:\n%s", want, first)
		}
	}
}

func TestDiagKindsName(t *testing.T) {
	for _, c := range []struct {
		k    kernel.DiagKind
		want string
	}{
		{kernel.DiagUndefinedRead, "read of undefined shared memory"},
		{kernel.DiagArrival, "barrier arrival mismatch"},
		{kernel.DiagConflict, "conflicting access"},
		{kernel.DiagKind(9), "unknown"},
	} {
		if got := c.k.String(); got != c.want {
			t.Errorf("got %q, want %q", got, c.want)
		}
	}
}

// Invocations that do not all reach the same barrier are reported, with both
// positions, rather than hanging.
//
// Three cases, because they are three different mistakes with the same symptom
// on a GPU: an invocation returning early, two reaching different barriers, and
// a non-uniform arrival where only some invocations suspend.
func TestBarrierArrivalMismatchesAreReported(t *testing.T) {
	cases := []struct {
		name string
		says string
		coop func(t kernel.Thread, a kernel.Args, f *kernel.Frame) bool
	}{{
		name: "an invocation returns while its peers wait",
		says: "it returned while its peers wait",
		coop: func(th kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			if th.LocalID().X == 2 {
				return false // finishes without ever reaching the barrier
			}
			if f.Pass == 0 {
				f.Pass = 1
				f.Barrier = kernel.BarrierID{Index: 0, Pos: "k.go:10:2"}
				return true
			}
			return false
		},
	}, {
		name: "two invocations reach different barriers",
		says: "while its peer waits at",
		coop: func(th kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			if f.Pass > 0 {
				return false
			}
			f.Pass = 1
			// Half suspend at one barrier, half at another. On a GPU this is a
			// hang or a silent pairing; here it is a mismatch.
			if th.LocalID().X < 2 {
				f.Barrier = kernel.BarrierID{Index: 0, Pos: "k.go:10:2"}
			} else {
				f.Barrier = kernel.BarrierID{Index: 1, Pos: "k.go:14:2"}
			}
			return true
		},
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			k := &kernel.Kernel{
				Name: "Arrive", WorkgroupSize: kernel.ID3{X: 4, Y: 1, Z: 1},
				Generator: kernel.ABIVersion, SharedSizes: []int{1}, Suspensions: 2,
				NewShared: func() []any {
					var sh [1]float32
					return []any{&sh}
				},
				Cooperative: c.coop,
			}
			err := kernel.DispatchCooperative(k, kernel.ID3{X: 1}, kernel.Args{})
			if err == nil {
				t.Fatal("expected an arrival diagnostic")
			}
			if !strings.Contains(err.Error(), "barrier arrival mismatch") {
				t.Fatalf("got %v", err)
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Errorf("the message should say %q, got:\n%v", c.says, err)
			}
			// Both positions, or a reader cannot see which two lines disagreed.
			if !strings.Contains(err.Error(), "k.go:10:2") {
				t.Errorf("the report should name the barrier's position, got:\n%v", err)
			}
		})
	}
}

// Arrival is checked for a kernel with barriers and no shared arrays.
//
// The tracker used to exist only when a kernel declared shared memory, and the
// arrival check was skipped without one, so a kernel synchronising storage
// through a barrier -- PublishStorage, Ballot, every kernel whose barrier
// orders global memory -- had a returning invocation go unreported in
// developer mode. Every other arrival test declares SharedSizes, which is
// why none of them caught it.
func TestArrivalIsCheckedWithoutSharedArrays(t *testing.T) {
	k := &kernel.Kernel{
		Name: "NoShared", WorkgroupSize: kernel.ID3{X: 4, Y: 1, Z: 1},
		Generator: kernel.ABIVersion, Suspensions: 1,
		Cooperative: func(th kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			if th.LocalID().X == 2 {
				return false // finishes without ever reaching the barrier
			}
			if f.Pass == 0 {
				f.Pass = 1
				f.Barrier = kernel.BarrierID{Index: 0, Pos: "k.go:10:2"}
				return true
			}
			return false
		},
	}
	err := kernel.DispatchCooperative(k, kernel.ID3{X: 1}, kernel.Args{})
	if err == nil {
		t.Fatal("an invocation returned while its peers wait at a barrier and nothing " +
			"reported it: the arrival check is gated on shared arrays rather than on " +
			"diagnostics")
	}
	var ds kernel.Diagnostics
	if !errors.As(err, &ds) || ds[0].Kind != kernel.DiagArrival {
		t.Fatalf("got %v, want a barrier arrival mismatch", err)
	}
	if !strings.Contains(err.Error(), "it returned while its peers wait") {
		t.Errorf("the message should name the returning invocation, got:\n%v", err)
	}
}

// A workgroup where every invocation reaches the same barrier reports nothing,
// or the checks above would be firing on every correct kernel.
func TestUniformArrivalIsNotReported(t *testing.T) {
	k := &kernel.Kernel{
		Name: "Fine", WorkgroupSize: kernel.ID3{X: 4, Y: 1, Z: 1},
		Generator: kernel.ABIVersion, SharedSizes: []int{1}, Suspensions: 1,
		NewShared: func() []any {
			var sh [1]float32
			return []any{&sh}
		},
		Cooperative: func(th kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			if f.Pass == 0 {
				f.Pass = 1
				f.Barrier = kernel.BarrierID{Index: 0, Pos: "k.go:10:2"}
				return true
			}
			return false
		},
	}
	if err := kernel.DispatchCooperative(k, kernel.ID3{X: 1}, kernel.Args{}); err != nil {
		t.Fatalf("every invocation reached the same barrier: %v", err)
	}
}

// The report is the same whatever order the invocations happen to be advanced
// in, which is what "on the first offending run rather than on an unlucky
// interleaving" means: it is a detection from recorded state, not a timeout.
func TestArrivalReportsAreDeterministic(t *testing.T) {
	build := func() *kernel.Kernel {
		return &kernel.Kernel{
			Name: "Arrive", WorkgroupSize: kernel.ID3{X: 8, Y: 1, Z: 1},
			Generator: kernel.ABIVersion, SharedSizes: []int{1}, Suspensions: 1,
			NewShared: func() []any {
				var sh [1]float32
				return []any{&sh}
			},
			Cooperative: func(th kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
				if f.Pass > 0 {
					return false
				}
				f.Pass = 1
				if th.LocalID().X%3 == 0 {
					return false
				}
				f.Barrier = kernel.BarrierID{Index: 0, Pos: "k.go:10:2"}
				return true
			},
		}
	}
	first := kernel.DispatchCooperative(build(), kernel.ID3{X: 1}, kernel.Args{})
	if first == nil {
		t.Fatal("expected a diagnostic")
	}
	for range 25 {
		again := kernel.DispatchCooperative(build(), kernel.ID3{X: 1}, kernel.Args{})
		if again == nil || again.Error() != first.Error() {
			t.Fatalf("the report differs between runs:\n%v\nwant\n%v", again, first)
		}
	}
	// Every offending invocation is named, not just the first: a report that
	// stopped at one would send a reader round the loop once per mistake.
	if n := strings.Count(first.Error(), "barrier arrival mismatch"); n != 3 {
		t.Errorf("got %d reports for 3 offending invocations of 8", n)
	}
}

// The check does not infer from a count of who is blocked. That count falling
// short is one way an arrival becomes impossible and not the only one, so a
// kernel where every invocation suspends at the same barrier is fine however
// few they are.
func TestArrivalIsNotInferredFromACount(t *testing.T) {
	k := &kernel.Kernel{
		Name: "One", WorkgroupSize: kernel.ID3{X: 1, Y: 1, Z: 1},
		Generator: kernel.ABIVersion, SharedSizes: []int{1}, Suspensions: 1,
		NewShared: func() []any {
			var sh [1]float32
			return []any{&sh}
		},
		Cooperative: func(th kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			if f.Pass == 0 {
				f.Pass = 1
				f.Barrier = kernel.BarrierID{Index: 0}
				return true
			}
			return false
		},
	}
	if err := kernel.DispatchCooperative(k, kernel.ID3{X: 1}, kernel.Args{}); err != nil {
		t.Fatalf("a single-invocation workgroup reaching its barrier is fine: %v", err)
	}
}

func TestABarrierIDDescribesItself(t *testing.T) {
	withPos := kernel.BarrierID{Index: 2, Pos: "k.go:9:3"}
	err := kernel.Diagnostic{
		Kind: kernel.DiagArrival, Kernel: "K", Element: -1,
		Detail: "it suspended at " + withPos.Describe(),
	}
	if !strings.Contains(err.Error(), "barrier 2 (k.go:9:3)") {
		t.Errorf("got %v", err.Error())
	}
	if !strings.Contains(kernel.BarrierID{Index: 1}.Describe(), "barrier 1") {
		t.Error("a barrier with no position still names its index")
	}
	if strings.Contains(err.Error(), "element") {
		t.Error("an arrival mismatch has no element and should not claim one")
	}
}

// An invocation that suspends without saying which barrier it stopped at is a
// defect in the generated lowering, and is reported as one.
//
// It is worth its own case because the failure is silent otherwise: the check
// looks for a barrier id to compare against, and if no suspended invocation
// carries one there is nothing to compare, so the epoch would pass. That is
// absence of evidence read as evidence of absence, and the transform forgetting
// to emit the id is exactly the bug a second suspension site could introduce.
func TestASuspensionWithNoBarrierIDIsADefect(t *testing.T) {
	k := &kernel.Kernel{
		Name: "Silent", WorkgroupSize: kernel.ID3{X: 4, Y: 1, Z: 1},
		Generator: kernel.ABIVersion, SharedSizes: []int{1}, Suspensions: 1,
		NewShared: func() []any {
			var sh [1]float32
			return []any{&sh}
		},
		Cooperative: func(th kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			if f.Pass == 0 {
				f.Pass = 1
				// Suspends without setting f.Barrier.
				return true
			}
			return false
		},
	}
	err := kernel.DispatchCooperative(k, kernel.ID3{X: 1}, kernel.Args{})
	if err == nil {
		t.Fatal("an invocation suspended without identifying its barrier and nothing " +
			"was reported: the check found no id to compare against and read that as " +
			"nothing being wrong")
	}
	if !strings.Contains(err.Error(), "did not identify") {
		t.Fatalf("got %v", err)
	}
}
