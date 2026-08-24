// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"testing"

	"golang.design/x/accel"
)

// The workgroup count covers exactly the requested invocations and no fewer.
//
// The arithmetic is one line, and the reason it is a method is that the line
// needs the kernel's workgroup size — which is the kernel's, not the caller's.
// Written by the caller it lives wherever the size is not, and the two drift:
// too few workgroups leaves a tail untouched, which reads as a kernel bug at
// the boundary.
func TestWorkgroupsCoverTheRequestedInvocations(t *testing.T) {
	d := openDevice(t)
	p := addPipeline(t, d)
	size := int(p.Kernel().WorkgroupSize.X)
	if size <= 1 {
		t.Fatalf("the test kernel's workgroup is %d wide, so ceiling division is not "+
			"exercised", size)
	}

	for _, n := range []int{1, size - 1, size, size + 1, 2 * size, 3*size - 1, 1000} {
		got := p.Workgroups(n)
		if covered := got.X * size; covered < n {
			t.Errorf("%d invocations need %d workgroups of %d and got %d, covering only %d",
				n, (n+size-1)/size, size, got.X, covered)
		}
		// And no more than one workgroup of slack, which is what makes this
		// ceiling division rather than a round number a caller guessed.
		if covered := got.X * size; covered-n >= size {
			t.Errorf("%d invocations got %d workgroups of %d, covering %d — more than one "+
				"workgroup of waste", n, got.X, size, covered)
		}
		if got.Y != 1 || got.Z != 1 {
			t.Errorf("Workgroups(%d) gave Y=%d Z=%d, want one each", n, got.Y, got.Z)
		}
	}
}

// No work covers as no work.
//
// specs/003-command-graph.md makes a zero in any dimension a dispatch of
// nothing rather than an error, so covering an empty extent must produce zero —
// not one workgroup whose invocations all read past the end.
func TestCoveringNoWorkGivesNoWorkgroups(t *testing.T) {
	d := openDevice(t)
	p := addPipeline(t, d)

	for _, n := range []int{0, -1} {
		if got := p.Workgroups(n); got.X != 0 {
			t.Errorf("Workgroups(%d) gave X=%d, want 0: a zero extent is the specified "+
				"skip, and one workgroup would read past the end", n, got.X)
		}
	}
	if got := p.WorkgroupsFor(64, 0, 8); got.Y != 0 {
		t.Errorf("an empty Y axis gave %d workgroups, want 0", got.Y)
	}
}

// A covered dispatch computes every element, including the tail.
//
// The end-to-end form: a length that is not a multiple of the workgroup size is
// the case a hand-written count gets wrong, and the last elements are where it
// shows.
func TestACoveredDispatchReachesTheTail(t *testing.T) {
	const n = 1000 // deliberately not a multiple of 64
	d := openDevice(t)
	q := d.Queue()
	p := addPipeline(t, d)

	storage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	a := newBuffer(t, d, "a", n, storage)
	b := newBuffer(t, d, "b", n, storage)
	out := newBuffer(t, d, "out", n, storage)
	ones := make([]float32, n)
	for i := range ones {
		ones[i] = float32(i)
	}
	if err := q.WriteBuffer(a, 0, ones); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := q.WriteBuffer(b, 0, ones); err != nil {
		t.Fatalf("write: %v", err)
	}

	err := q.Run(func(r *accel.Recorder) {
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: whole(t, a)},
			{Index: 1, Buffer: whole(t, b)},
			{Index: 2, Buffer: whole(t, out)},
		}, nil, p.Workgroups(n))
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	got := readback(t, d, out)
	for i := range got {
		if want := float32(i) * 2; got[i] != want {
			t.Fatalf("element %d is %v, want %v; the count left a tail uncomputed",
				i, got[i], want)
		}
	}
}
