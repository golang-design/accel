// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"fmt"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/direct"
	"golang.design/x/accel/internal/conformance/numeq"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
)

// runAuthoredKernel calls an authored function once per invocation, the way a
// backend would, so the ids it reads are the ids a dispatch supplies.
func runAuthoredKernel(k *accel.Kernel, n int, call func(t accel.Thread)) {
	size := k.WorkgroupSize
	groups := direct.Groups(uint32(n), size.X)
	count := accel.ID3{X: groups, Y: 1, Z: 1}
	for g := range groups {
		for l := range size.X {
			call(kernel.NewThread(
				accel.ID3{X: g*size.X + l}, accel.ID3{X: l}, accel.ID3{X: g}, size, count))
		}
	}
}

// TestSegmentSumMatchesAuthored is spec 004's fifth level for a kernel whose
// body reaches two helpers and a loop.
//
// Scale's single multiplication has no intermediate a compiler could keep wider,
// so its comparison collapses to equality. This one accumulates, which is where
// the generated lowering's explicit rounding points would show up if the
// authored function were evaluated differently. They agree bit for bit here; on
// a host where they did not, that difference is what spec 008's budgets bound
// at M4, and asserting bits is what would surface it rather than hide it.
func TestSegmentSumMatchesAuthored(t *testing.T) {
	for _, n := range []int{0, 1, 7, 32, 33, 100} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			in := make([]float32, n*4)
			for i := range in {
				in[i] = float32(i)*0.1 - 5
			}

			authored := make([]float32, n)
			runAuthoredKernel(&testkernels.SegmentSumKernel, n, func(th accel.Thread) {
				testkernels.SegmentSum(th, in, authored)
			})

			generated := make([]float32, n)
			if err := direct.Run(&testkernels.SegmentSumKernel,
				direct.Cover(&testkernels.SegmentSumKernel, n),
				accel.KernelArgs{Slices: []any{in, generated}}); err != nil {
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

// TestSegmentSumAgainstAReference checks the kernel against an independently
// written sum, so "the two lowerings agree" is joined by "and they are right".
func TestSegmentSumAgainstAReference(t *testing.T) {
	const n = 16
	in := make([]float32, n*4)
	for i := range in {
		in[i] = float32(i)
	}
	out := make([]float32, n)
	if err := direct.Run(&testkernels.SegmentSumKernel,
		direct.Cover(&testkernels.SegmentSumKernel, n),
		accel.KernelArgs{Slices: []any{in, out}}); err != nil {
		t.Fatal(err)
	}

	for i := range out {
		// The kernel clamps both ends into the binding, so the reference does too.
		from := min(uint32(i)*4, uint32(len(in))-1)
		to := min(from+4, uint32(len(in))-1)
		want := float32(0)
		for j := from; j < to; j++ {
			want += in[j]
		}
		if out[i] != want {
			t.Errorf("element %d is %v, want %v", i, out[i], want)
		}
	}
}

// TestCountAboveMatchesAuthored covers the kernel that uses all three loop
// forms, so each has a lowering somebody runs rather than only one somebody
// emitted.
func TestCountAboveMatchesAuthored(t *testing.T) {
	for _, n := range []int{0, 1, 16, 17, 40} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			in := make([]float32, 64)
			for i := range in {
				in[i] = float32(i) - 32
			}

			authored := make([]int32, n)
			runAuthoredKernel(&testkernels.CountAboveKernel, n, func(th accel.Thread) {
				testkernels.CountAbove(th, in, authored)
			})

			generated := make([]int32, n)
			if err := direct.Run(&testkernels.CountAboveKernel,
				direct.Cover(&testkernels.CountAboveKernel, n),
				accel.KernelArgs{Slices: []any{in, generated}}); err != nil {
				t.Fatalf("direct.Run: %v", err)
			}

			// Integer results are bit-exact everywhere, and that is a hard
			// requirement rather than an observation: spec 006 puts integer
			// kernels in the exact class with no bound at all.
			if r := numeq.Exact(generated, authored); !r.Equal {
				t.Errorf("the generated lowering and the authored function disagree: %v", r)
			}

			// 31 of the 64 inputs are above zero.
			for i, got := range generated {
				if n > 0 && got != 31 {
					t.Fatalf("element %d counted %d, want 31", i, got)
				}
			}
		})
	}
}

// TestHelperAccessIsInferredThroughACall is the property a caller depends on:
// SegmentSum never indexes in directly, only through accumulate, and the record
// still says the binding is read.
func TestHelperAccessIsInferredThroughACall(t *testing.T) {
	k := &testkernels.SegmentSumKernel
	if len(k.Bindings) != 2 {
		t.Fatalf("%d bindings", len(k.Bindings))
	}
	if got := k.Bindings[0]; got.Name != "in" || got.Access != accel.KernelRead {
		t.Errorf("binding 0 is %+v; in is read only inside a helper, and the access still "+
			"has to reach the record", got)
	}
	if got := k.Bindings[1]; got.Name != "out" || got.Access != accel.KernelWrite {
		t.Errorf("binding 1 is %+v", got)
	}
}
