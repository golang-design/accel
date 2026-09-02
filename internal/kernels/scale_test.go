// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels_test

import (
	"fmt"
	"golang.design/x/accel/kernelabi"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/direct"
	"golang.design/x/accel/internal/conformance/numeq"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/kernels"
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
			kernels.Scale(accel.Thread(t), in, out)
		}
	}
}

// TestGeneratedMatchesAuthored is spec 004's fifth testing level, and the only
// check that the generated lowering means what the authored source says.
//
// Since the authored function is no longer what a backend executes, a mistake
// in IR construction produces a CPU runner and every GPU artifact identically
// wrong, all derived from the same wrong IR, and differential execution across
// backends cannot see it: it is wrong the same way everywhere. Running the
// authored Go directly and comparing is what closes that hole.
//
// The comparison is under spec 008 rather than bit-for-bit, and the reason is
// the point of the exercise: the generated lowering emits an explicit float32
// at every rounding point and the authored function does not, so on a host with
// FMA the two may legitimately differ in the last bit. Here the expression is a
// single multiplication with no intermediate to widen, so the bound collapses to
// equality and the test asserts bits. A kernel with an accumulation would not.
func TestGeneratedMatchesAuthored(t *testing.T) {
	for _, n := range []int{0, 1, 63, 64, 65, 130, 1000} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			in := make([]float32, n)
			for i := range in {
				in[i] = float32(i)*0.375 - 7
			}

			authored := make([]float32, n)
			runGrid(uint32(n), in, authored, n)

			generated := make([]float32, n)
			if err := direct.Run(&kernels.ScaleKernel,
				direct.Cover(&kernels.ScaleKernel, n),
				kernelabi.Args{Slices: []any{in, generated}}); err != nil {
				t.Fatalf("direct.Run: %v", err)
			}

			if r := numeq.ExactBits(generated, authored, func(f float32) uint64 {
				return uint64(math.Float32bits(f))
			}); !r.Equal {
				t.Errorf("the generated lowering and the authored function disagree: %v", r)
			}
		})
	}
}

// TestGeneratedRecordDescribesTheSource checks that what the generator inferred
// is what the signature and body say, since every caller and every backend is
// told this rather than reading the source.
func TestGeneratedRecordDescribesTheSource(t *testing.T) {
	k := &kernels.ScaleKernel

	if k.Name != "Scale" {
		t.Errorf("Name = %q", k.Name)
	}
	if k.WorkgroupSize != (accel.ID3{X: 64, Y: 1, Z: 1}) {
		t.Errorf("WorkgroupSize = %+v, want 64,1,1", k.WorkgroupSize)
	}
	if k.Generator != kernelabi.Version {
		t.Errorf("Generator = %d, want %d", k.Generator, kernelabi.Version)
	}
	if k.Digest == "" {
		t.Error("the record carries no digest, so nothing can check it is fresh")
	}
	if len(k.Bindings) != 2 {
		t.Fatalf("%d bindings, want 2", len(k.Bindings))
	}
	// The accesses are inferred from the body. in is read, out is written, and
	// len(out) is not an element read.
	if got := k.Bindings[0]; got.Name != "in" || got.Access != kernelabi.Read {
		t.Errorf("binding 0 = %+v, want in/read", got)
	}
	if got := k.Bindings[1]; got.Name != "out" || got.Access != kernelabi.Write {
		t.Errorf("binding 1 = %+v, want out/write", got)
	}
}

// TestBindRejectsTheWrongBuffers checks that the argument set is validated
// against the record before anything runs.
func TestBindRejectsTheWrongBuffers(t *testing.T) {
	k := &kernels.ScaleKernel
	for _, tc := range []struct {
		name string
		args kernelabi.Args
	}{
		{"too few", kernelabi.Args{Slices: []any{make([]float32, 4)}}},
		{"wrong dtype", kernelabi.Args{Slices: []any{make([]float32, 4), make([]int32, 4)}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := direct.Run(k, direct.Cover(k, 4), tc.args); err == nil {
				t.Error("was accepted")
			}
		})
	}
}
