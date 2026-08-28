// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"fmt"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
	"golang.design/x/accel/kernelabi"
)

// intReduceInputs is a shuffled set spanning both signs.
//
// Shuffled rather than monotone: with an increasing input the minimum is always
// lane 0's and the maximum always the last lane's, so a kernel returning a fixed
// lane's value instead of reducing would pass. Spanning both signs is what makes
// the signed and unsigned answers land on different lanes.
func intReduceInputs(n int) []int32 {
	in := make([]int32, n)
	for i := range in {
		// A multiplier coprime with the width permutes the lanes, and the
		// offset puts about a third of them below zero.
		in[i] = int32((i*37)%n) - int32(n/3)
	}
	return in
}

// Each integer reduction equals a scalar loop over the same lanes.
//
// specs/059-subgroup-reductions.md §7's first assertion. The reference uses no
// subgroup operation at all, so a disagreement is the subgroup path's rather
// than the kernel's -- the same shape [020](020-cooperative-atomics.md)'s
// fallback comparison uses.
//
// Swept across the emulated sizes 002 §5.4 gives, because a reduction that
// ignored subgroup boundaries agrees at the degenerate single subgroup and
// disagrees at every other width.
func TestTheIntegerReductionsMatchAScalarLoop(t *testing.T) {
	width := int(testkernels.IntReduceKernel.WorkgroupSize.X)
	in := intReduceInputs(width)

	for _, size := range []uint32{1, 4, 32, 64} {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			minI := make([]int32, width)
			maxI := make([]int32, width)
			minU := make([]uint32, width)
			maxU := make([]uint32, width)

			err := kernel.DispatchCooperativeWith(&testkernels.IntReduceKernel,
				accel.ID3{X: 1},
				kernelabi.Args{Slices: []any{in, minI, maxI, minU, maxU}},
				kernel.Options{Diagnostics: true, SubgroupSize: size})
			if err != nil {
				t.Fatalf("dispatch at subgroup size %d: %v", size, err)
			}

			for i := range width {
				// This lane's subgroup, computed here rather than read from
				// the kernel.
				base := i - i%int(size)
				wantMinI, wantMaxI := in[base], in[base]
				wantMinU, wantMaxU := uint32(in[base]), uint32(in[base])
				for l := base; l < base+int(size) && l < width; l++ {
					wantMinI = min(wantMinI, in[l])
					wantMaxI = max(wantMaxI, in[l])
					wantMinU = min(wantMinU, uint32(in[l]))
					wantMaxU = max(wantMaxU, uint32(in[l]))
				}

				if minI[i] != wantMinI {
					t.Fatalf("lane %d: MinI32 is %d, want %d", i, minI[i], wantMinI)
				}
				if maxI[i] != wantMaxI {
					t.Fatalf("lane %d: MaxI32 is %d, want %d", i, maxI[i], wantMaxI)
				}
				if minU[i] != wantMinU {
					t.Fatalf("lane %d: MinU32 is %d, want %d", i, minU[i], wantMinU)
				}
				if maxU[i] != wantMaxU {
					t.Fatalf("lane %d: MaxU32 is %d, want %d", i, maxU[i], wantMaxU)
				}
			}
		})
	}
}

// The four reductions give four different answers.
//
// §7's second assertion, and what a family of near-identical cases needs: a
// transposed comparison in one of them produces its neighbour's answer, which
// every "equals a scalar loop" check above would still catch -- but only
// because the loop is written per operation too. This asserts the four are
// *distinguishable on this input*, so the comparison above is not four
// agreements on one number.
func TestTheFourIntegerReductionsDisagree(t *testing.T) {
	width := int(testkernels.IntReduceKernel.WorkgroupSize.X)
	in := intReduceInputs(width)

	minI := make([]int32, width)
	maxI := make([]int32, width)
	minU := make([]uint32, width)
	maxU := make([]uint32, width)
	err := kernel.DispatchCooperativeWith(&testkernels.IntReduceKernel,
		accel.ID3{X: 1},
		kernelabi.Args{Slices: []any{in, minI, maxI, minU, maxU}},
		kernel.Options{Diagnostics: true, SubgroupSize: 16})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	for _, c := range []struct {
		name string
		a, b any
	}{
		{"MinI32 and MaxI32", minI[0], maxI[0]},
		{"MinU32 and MaxU32", minU[0], maxU[0]},
		// The pair that says the two carriers are separate: a negative i32 is
		// a large u32, so the signed minimum and the unsigned minimum are
		// different lanes' values.
		{"MinI32 and MinU32", uint32(minI[0]), minU[0]},
		{"MaxI32 and MaxU32", uint32(maxI[0]), maxU[0]},
	} {
		if c.a == c.b {
			t.Errorf("%s agree at %v on this input, so a test comparing either "+
				"against a reference cannot tell them apart", c.name, c.a)
		}
	}
}

// The authored integer reductions and the generated lowering agree, at one lane.
//
// specs/010-kernel-corpus.md §6, run the way every authored subgroup collective
// is: Thread.SubgroupMinI32 returns the lane's own value because combining is
// the scheduler's job, and at a subgroup of one that is the right answer rather
// than a stub's artefact.
func TestTheAuthoredIntegerReductionsMatchTheirLoweringAtOneLane(t *testing.T) {
	width := int(testkernels.IntReduceKernel.WorkgroupSize.X)
	in := intReduceInputs(width)

	aMinI := make([]int32, width)
	aMaxI := make([]int32, width)
	aMinU := make([]uint32, width)
	aMaxU := make([]uint32, width)
	kernel.RunAuthored(&testkernels.IntReduceKernel, kernel.ID3{}, kernel.ID3{X: 1}, 1,
		func(th kernel.Thread) {
			testkernels.IntReduce(th, in, aMinI, aMaxI, aMinU, aMaxU)
		})

	gMinI := make([]int32, width)
	gMaxI := make([]int32, width)
	gMinU := make([]uint32, width)
	gMaxU := make([]uint32, width)
	err := kernel.DispatchCooperativeWith(&testkernels.IntReduceKernel,
		accel.ID3{X: 1},
		kernelabi.Args{Slices: []any{in, gMinI, gMaxI, gMinU, gMaxU}},
		kernel.Options{Diagnostics: true, SubgroupSize: 1})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	for i := range width {
		if aMinI[i] != gMinI[i] || aMaxI[i] != gMaxI[i] ||
			aMinU[i] != gMinU[i] || aMaxU[i] != gMaxU[i] {
			t.Fatalf("lane %d: authored (%d %d %d %d), generated (%d %d %d %d)",
				i, aMinI[i], aMaxI[i], aMinU[i], aMaxU[i],
				gMinI[i], gMaxI[i], gMinU[i], gMaxU[i])
		}
	}
	// And it is the lane's own value at a subgroup of one, which is what says
	// the comparison ran rather than agreeing on zeros.
	if aMinI[0] != in[0] {
		t.Fatalf("MinI32 at a subgroup of one is %d and lane 0 holds %d", aMinI[0], in[0])
	}
}

// bitReduceInputs gives every lane a shared set of bits and a private one.
//
// # Why not a pseudorandom pattern
//
// The obvious input -- a hash per lane -- makes And zero and Or all-ones over
// any subgroup wider than a handful of lanes, because a random bit is clear in
// *some* lane and set in *some* lane. Both are then exactly what a kernel
// ignoring its input produces, so the comparison against a scalar loop is three
// agreements on two constants. TestTheThreeBitwiseReductionsDisagree caught
// that, on the first version of this function.
//
// So the pattern is constructed rather than scrambled: a fixed mask every lane
// carries, which survives And, plus one bit per lane from a small rotating set,
// which does not. And is then the shared mask, Or is the mask plus the union,
// and Xor is neither.
func bitReduceInputs(n int) []int32 {
	const shared = 0x00F0F00F
	in := make([]int32, n)
	for i := range in {
		// A private bit from a set of eight, so a subgroup of any width sees
		// several distinct values and none of them clears the shared mask.
		in[i] = int32(shared) | int32(1)<<(uint(i)%8+20)
	}
	return in
}

// Each bitwise reduction equals a scalar loop over the same lanes.
//
// specs/059-subgroup-reductions.md §7. The reference uses no subgroup operation,
// and the sweep is what says the kernel respects subgroup boundaries: an
// implementation reducing the whole workgroup agrees at size 64 and disagrees
// everywhere else.
func TestTheBitwiseReductionsMatchAScalarLoop(t *testing.T) {
	width := int(testkernels.BitReduceKernel.WorkgroupSize.X)
	in := bitReduceInputs(width)

	for _, size := range []uint32{1, 4, 32, 64} {
		t.Run(fmt.Sprintf("size=%d", size), func(t *testing.T) {
			andI := make([]int32, width)
			orI := make([]int32, width)
			xorI := make([]int32, width)
			andU := make([]uint32, width)
			orU := make([]uint32, width)
			xorU := make([]uint32, width)

			err := kernel.DispatchCooperativeWith(&testkernels.BitReduceKernel,
				accel.ID3{X: 1},
				kernelabi.Args{Slices: []any{in, andI, orI, xorI, andU, orU, xorU}},
				kernel.Options{Diagnostics: true, SubgroupSize: size})
			if err != nil {
				t.Fatalf("dispatch at subgroup size %d: %v", size, err)
			}

			for i := range width {
				base := i - i%int(size)
				wa, wo, wx := in[base], in[base], in[base]
				for l := base + 1; l < base+int(size) && l < width; l++ {
					wa &= in[l]
					wo |= in[l]
					wx ^= in[l]
				}

				if andI[i] != wa {
					t.Fatalf("lane %d: AndI32 is %#x, want %#x", i, andI[i], wa)
				}
				if orI[i] != wo {
					t.Fatalf("lane %d: OrI32 is %#x, want %#x", i, orI[i], wo)
				}
				if xorI[i] != wx {
					t.Fatalf("lane %d: XorI32 is %#x, want %#x", i, xorI[i], wx)
				}
				// The unsigned answers are the same bits, which is what says
				// the two carriers did not cross: a u32 reduction reading the
				// i32 carrier would still agree here, and the *compiler*
				// rejects that -- see 059 §2.
				if andU[i] != uint32(wa) {
					t.Fatalf("lane %d: AndU32 is %#x, want %#x", i, andU[i], uint32(wa))
				}
				if orU[i] != uint32(wo) {
					t.Fatalf("lane %d: OrU32 is %#x, want %#x", i, orU[i], uint32(wo))
				}
				if xorU[i] != uint32(wx) {
					t.Fatalf("lane %d: XorU32 is %#x, want %#x", i, xorU[i], uint32(wx))
				}
			}
		})
	}
}

// And, Or and Xor give three different answers on this input.
//
// Without it the comparison above could be three agreements on one number:
// And over consecutive integers is zero and Or is all-ones, and a kernel
// ignoring its input satisfies both.
func TestTheThreeBitwiseReductionsDisagree(t *testing.T) {
	width := int(testkernels.BitReduceKernel.WorkgroupSize.X)
	in := bitReduceInputs(width)

	andI := make([]int32, width)
	orI := make([]int32, width)
	xorI := make([]int32, width)
	andU := make([]uint32, width)
	orU := make([]uint32, width)
	xorU := make([]uint32, width)
	err := kernel.DispatchCooperativeWith(&testkernels.BitReduceKernel,
		accel.ID3{X: 1},
		kernelabi.Args{Slices: []any{in, andI, orI, xorI, andU, orU, xorU}},
		kernel.Options{Diagnostics: true, SubgroupSize: 16})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	for _, c := range []struct {
		name string
		a, b int32
	}{
		{"And and Or", andI[0], orI[0]},
		{"And and Xor", andI[0], xorI[0]},
		{"Or and Xor", orI[0], xorI[0]},
	} {
		if c.a == c.b {
			t.Errorf("%s agree at %#x on this input, so a test comparing either "+
				"against a reference cannot tell them apart", c.name, c.a)
		}
	}
	// And neither And nor Or is the degenerate answer a kernel ignoring its
	// input would give.
	if andI[0] == 0 {
		t.Error("And is zero, which a kernel ignoring its input also produces")
	}
	if orI[0] == -1 {
		t.Error("Or is all ones, which a kernel ignoring its input also produces")
	}
}

// The authored bitwise reductions and the generated lowering agree, at one lane.
//
// specs/010-kernel-corpus.md §6, at a subgroup of one for the reason every
// authored subgroup collective is run there: the method returns the lane's own
// value because combining is the scheduler's job, and over one lane that is the
// right answer rather than a stub's artefact.
func TestTheAuthoredBitwiseReductionsMatchTheirLoweringAtOneLane(t *testing.T) {
	width := int(testkernels.BitReduceKernel.WorkgroupSize.X)
	in := bitReduceInputs(width)

	mk := func() ([]int32, []int32, []int32, []uint32, []uint32, []uint32) {
		return make([]int32, width), make([]int32, width), make([]int32, width),
			make([]uint32, width), make([]uint32, width), make([]uint32, width)
	}
	aA, aO, aX, aAU, aOU, aXU := mk()
	kernel.RunAuthored(&testkernels.BitReduceKernel, kernel.ID3{}, kernel.ID3{X: 1}, 1,
		func(th kernel.Thread) {
			testkernels.BitReduce(th, in, aA, aO, aX, aAU, aOU, aXU)
		})

	gA, gO, gX, gAU, gOU, gXU := mk()
	err := kernel.DispatchCooperativeWith(&testkernels.BitReduceKernel,
		accel.ID3{X: 1},
		kernelabi.Args{Slices: []any{in, gA, gO, gX, gAU, gOU, gXU}},
		kernel.Options{Diagnostics: true, SubgroupSize: 1})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	for i := range width {
		if aA[i] != gA[i] || aO[i] != gO[i] || aX[i] != gX[i] ||
			aAU[i] != gAU[i] || aOU[i] != gOU[i] || aXU[i] != gXU[i] {
			t.Fatalf("lane %d: authored (%#x %#x %#x %#x %#x %#x), "+
				"generated (%#x %#x %#x %#x %#x %#x)", i,
				aA[i], aO[i], aX[i], aAU[i], aOU[i], aXU[i],
				gA[i], gO[i], gX[i], gAU[i], gOU[i], gXU[i])
		}
	}
	// Over one lane every operation is the identity on that lane's value, so
	// this is what says the comparison ran rather than agreeing on zeros.
	if aA[0] != in[0] {
		t.Fatalf("And at a subgroup of one is %#x and lane 0 holds %#x", aA[0], in[0])
	}
}
