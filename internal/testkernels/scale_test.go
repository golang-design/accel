// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
)

// TestAuthoredScale runs the authored kernel against an independent reference.
//
// This is the authored half of spec 004's fifth testing level. The generated
// lowering is what the backend runs, so nothing else would ever call this
// function, and a kernel nobody executes is a kernel whose meaning is whatever
// the IR happened to make of it. Checking it against a reference written
// separately is what makes "the two agree" a statement about correctness rather
// than about a tautology.
//
// The comparison to the generated lowering arrives with the direct executor.
func TestAuthoredScale(t *testing.T) {
	const n = 130 // not a multiple of the 64 workgroup, so the tail is exercised

	in := make([]float32, n)
	for i := range in {
		in[i] = float32(i)*0.25 - 3
	}

	out := make([]float32, n)
	runAuthored(n, in, out)

	for i := range out {
		want := in[i] * 2
		if math.Float32bits(out[i]) != math.Float32bits(want) {
			t.Fatalf("element %d is %v, want %v", i, out[i], want)
		}
	}
}

// TestAuthoredScaleRespectsItsBound checks the guard the kernel writes by hand.
//
// A kernel is dispatched in whole workgroups, so the last one runs invocations
// past the end of the buffer, and the bounds check is the only thing standing
// between that and a write outside the binding. Testing it means dispatching a
// grid deliberately larger than the data.
func TestAuthoredScaleRespectsItsBound(t *testing.T) {
	const n = 10
	in := make([]float32, n)
	out := make([]float32, n)
	for i := range in {
		in[i] = 1
	}

	// One full workgroup of 64 over 10 elements: 54 invocations are out of range.
	runGrid(64, in, out, n)

	for i := range out {
		if out[i] != 2 {
			t.Fatalf("element %d is %v, want 2", i, out[i])
		}
	}
}

// TestAuthoredScaleOnAnEmptyBinding checks that a kernel over nothing writes
// nothing rather than faulting, which is what a zero-length dispatch means.
func TestAuthoredScaleOnAnEmptyBinding(t *testing.T) {
	runGrid(64, nil, nil, 0)
}

// runAuthored dispatches exactly enough invocations to cover the data.
func runAuthored(n int, in, out []float32) {
	runGrid(uint32(n), in, out, n)
}

// runGrid calls the authored kernel once per invocation, the way a backend
// would, so that the ids it reads are the ids a dispatch would give it.
func runGrid(invocations uint32, in, out []float32, _ int) {
	const group = 64
	size := kernel.ID3{X: group, Y: 1, Z: 1}
	groups := (invocations + group - 1) / group
	count := kernel.ID3{X: groups, Y: 1, Z: 1}

	for g := range groups {
		for l := range uint32(group) {
			t := kernel.NewThread(
				kernel.ID3{X: g*group + l},
				kernel.ID3{X: l},
				kernel.ID3{X: g},
				size, count,
			)
			testkernels.Scale(accel.Thread(t), in, out)
		}
	}
}
