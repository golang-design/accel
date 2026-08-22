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

// TestNormalizeMatchesAuthored is spec 004's fifth level for a kernel that
// reaches narrow storage and two bounded intrinsics.
//
// This is the case where the generated lowering's explicit rounding points
// could genuinely diverge from the authored function, because the authored
// `kmath.Abs(x) + 1` has an intermediate a compiler may keep wider and the
// generated `float32(kmath.Abs(x) + float32(1))` does not. They agree here; the
// value of asserting bits is that a host where they did not would say so rather
// than pass with a tolerance nobody chose.
func TestNormalizeMatchesAuthored(t *testing.T) {
	for _, n := range []int{0, 1, 32, 33, 70} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			in := make([]accel.Float16, n)
			for i := range in {
				in[i] = accel.ToFloat16(float32(i)*0.25 - 8)
			}

			authored := make([]float32, n)
			authoredScratch := make([]float32, n)
			runAuthoredKernel(&testkernels.NormalizeKernel, n, func(th accel.Thread) {
				testkernels.Normalize(th, in, authored, authoredScratch)
			})

			generated := make([]float32, n)
			generatedScratch := make([]float32, n)
			if err := direct.Run(&testkernels.NormalizeKernel,
				direct.Cover(&testkernels.NormalizeKernel, n),
				accel.KernelArgs{Slices: []any{in, generated, generatedScratch}}); err != nil {
				t.Fatalf("direct.Run: %v", err)
			}

			bits := func(f float32) uint64 { return uint64(math.Float32bits(f)) }
			if r := numeq.ExactBits(generated, authored, bits); !r.Equal {
				t.Errorf("outputs disagree: %v", r)
			}
			if r := numeq.ExactBits(generatedScratch, authoredScratch, bits); !r.Equal {
				t.Errorf("scratch disagrees: %v", r)
			}
		})
	}
}

// TestNormalizeWidensItsInput checks the conversion a kernel has to write by
// hand, since accel.Float16 carries no arithmetic operators.
//
// The values are chosen to include one that f16 cannot represent exactly, so
// the test says what the kernel actually computes rather than what a reader
// might assume it does.
func TestNormalizeWidensItsInput(t *testing.T) {
	const n = 4
	in := []accel.Float16{
		accel.ToFloat16(0),
		accel.ToFloat16(1),
		accel.ToFloat16(-2.5),
		accel.ToFloat16(0.1), // not exactly representable in f16
	}
	out := make([]float32, n)
	scratch := make([]float32, n)
	if err := direct.Run(&testkernels.NormalizeKernel,
		direct.Cover(&testkernels.NormalizeKernel, n),
		accel.KernelArgs{Slices: []any{in, out, scratch}}); err != nil {
		t.Fatal(err)
	}

	for i := range in {
		// The reference widens the same stored value the kernel sees, not the
		// f32 it was converted from: 0.1 is not 0.1 once it is an f16, and a
		// reference that used the original would be testing the conversion twice.
		x := in[i].F32()
		wantMagnitude := float32(math.Sqrt(float64(float32(math.Abs(float64(x))) + 1)))
		if scratch[i] != wantMagnitude {
			t.Errorf("element %d magnitude is %v, want %v", i, scratch[i], wantMagnitude)
		}
	}
}

// TestNarrowBindingIsDeclaredAsSuch checks that the record names the storage
// format, since a caller allocates against that dtype and a mismatch would bind
// a buffer of the wrong element width.
func TestNarrowBindingIsDeclaredAsSuch(t *testing.T) {
	k := &testkernels.NormalizeKernel
	if got := k.Bindings[0]; got.DType != accel.KernelF16 {
		t.Errorf("binding 0 is %v, want f16", got.DType)
	}
	if err := k.Bind(accel.KernelArgs{Slices: []any{
		make([]uint16, 4), make([]float32, 4), make([]float32, 4),
	}}); err == nil {
		t.Error("a []uint16 was accepted for an f16 binding: a narrow value is a distinct type " +
			"precisely so its bit pattern cannot be mistaken for an integer")
	}
}
