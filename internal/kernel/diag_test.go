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

// A nil tracker is every call a no-op, which is how strict mode pays nothing
// for a check developer mode wants.
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
