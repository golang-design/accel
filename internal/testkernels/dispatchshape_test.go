// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/direct"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
	"golang.design/x/accel/kernelabi"
)

// shapeDispatch is the grid this runs on: three axes that differ, so a lowering
// that transposed one is visible rather than symmetric.
var shapeGroups = accel.ID3{X: 5, Y: 3, Z: 2}

// Each accessor reports what the dispatch was recorded with.
//
// specs/052-dispatch-shape.md §3's first assertion. The kernel writes what the
// thread says and the test compares it against what the host asked for, so a
// kernel reading its shape from the uniform would be agreeing with itself.
func TestTheDispatchShapeIsWhatTheDispatchWasRecordedWith(t *testing.T) {
	const stride = 3
	d := testkernels.ShapeDims{Stride: stride}
	out := make([]uint32, 3*stride)

	if err := direct.Run(&testkernels.DispatchShapeKernel, shapeGroups,
		kernelabi.Args{Uniforms: []any{d}, Slices: []any{out}}); err != nil {
		t.Fatalf("run: %v", err)
	}

	ws := testkernels.DispatchShapeKernel.WorkgroupSize
	for _, c := range []struct {
		name string
		at   int
		want accel.ID3
	}{
		// The declared extent, which the accel:kernel directive fixed at 4,2,1
		// and which the kernel record carries. Compared against the record
		// rather than against a literal here, so the two cannot drift.
		{"WorkgroupSize", 0, ws},
		{"NumGroups", 1, shapeGroups},
		{"GlobalSize", 2, accel.ID3{
			X: ws.X * shapeGroups.X, Y: ws.Y * shapeGroups.Y, Z: ws.Z * shapeGroups.Z,
		}},
	} {
		got := accel.ID3{
			X: out[c.at*stride+0], Y: out[c.at*stride+1], Z: out[c.at*stride+2],
		}
		if got != c.want {
			t.Errorf("%s is %+v, want %+v", c.name, got, c.want)
		}
	}
}

// GlobalSize is the product rather than a fourth thing bound separately.
//
// §2: two numbers that must agree eventually disagree. This is what says the
// derivation is the derivation, and it holds across grids rather than at the
// one shape the test above uses.
func TestTheGlobalSizeIsTheProductOfTheOtherTwo(t *testing.T) {
	const stride = 3
	for _, groups := range []accel.ID3{
		{X: 1, Y: 1, Z: 1},
		{X: 5, Y: 3, Z: 2},
		{X: 7, Y: 1, Z: 1},
	} {
		out := make([]uint32, 3*stride)
		if err := direct.Run(&testkernels.DispatchShapeKernel, groups,
			kernelabi.Args{
				Uniforms: []any{testkernels.ShapeDims{Stride: stride}},
				Slices:   []any{out},
			}); err != nil {
			t.Fatalf("run at %+v: %v", groups, err)
		}
		for axis := range 3 {
			ws, ng, gs := out[0*stride+axis], out[1*stride+axis], out[2*stride+axis]
			if ws*ng != gs {
				t.Errorf("at grid %+v axis %d: WorkgroupSize %d times NumGroups %d is "+
					"%d, and GlobalSize reports %d", groups, axis, ws, ng, ws*ng, gs)
			}
		}
	}
}

// A workgroup extent used as a loop bound still compiles with a barrier inside.
//
// §3's third assertion, and the property that makes the accessor worth having
// over the uniform field every corpus kernel carries. A barrier must be reached
// by every invocation, so a loop containing one needs a bound the *compiler*
// can see is the same for all of them. WorkgroupSize is a compile-time constant
// and a uniform field is not.
//
// That ShapeBoundedSum exists at all is most of the assertion: it is in the
// corpus, so `go generate` lowered it for both targets, and a bound the barrier
// analysis rejected would have failed generation rather than this test. What
// this adds is that the kernel computes the right thing, since a kernel that
// compiles and sums the wrong slice would satisfy the first half.
func TestAWorkgroupBoundedLoopWithABarrierRuns(t *testing.T) {
	const groups = 3
	width := int(testkernels.ShapeBoundedSumKernel.WorkgroupSize.X)

	in := make([]float32, groups*width)
	for i := range in {
		in[i] = float32(i) + 1
	}
	out := make([]float32, len(in))
	// The grid is a workgroup count, not an invocation count.
	if err := kernel.DispatchCooperative(&testkernels.ShapeBoundedSumKernel,
		accel.ID3{X: groups},
		kernelabi.Args{Slices: []any{in, out}}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	// Every lane of a workgroup sums that workgroup's whole slice, so the
	// answer is the same for all of them and differs between workgroups. A
	// kernel whose loop ran the wrong number of times gets a different total,
	// which no tolerance hides because the inputs are integers.
	for g := range groups {
		want := float32(0)
		for i := range width {
			want += in[g*width+i]
		}
		for lane := range width {
			if got := out[g*width+lane]; got != want {
				t.Fatalf("workgroup %d lane %d summed to %v, want %v: the loop bounded "+
					"by WorkgroupSize ran a different number of times", g, lane, got, want)
			}
		}
	}
}

// The authored form and the generated lowering agree on the shape.
//
// The two read it differently -- the authored function asks the Thread, the
// generated Go lowering asks the same Thread, and the MSL lowering has a
// literal baked in by the generator -- so this is what says the three spell one
// number. The Metal half is the corpus differential.
func TestTheAuthoredDispatchShapeMatchesItsLowering(t *testing.T) {
	const stride = 3
	d := testkernels.ShapeDims{Stride: stride}

	authored := make([]uint32, 3*stride)
	for gz := range shapeGroups.Z {
		for gy := range shapeGroups.Y {
			for gx := range shapeGroups.X {
				kernel.RunAuthored(&testkernels.DispatchShapeKernel,
					kernel.ID3{X: gx, Y: gy, Z: gz}, shapeGroups, 128,
					func(th kernel.Thread) {
						testkernels.DispatchShape(th, d, authored)
					})
			}
		}
	}

	generated := make([]uint32, 3*stride)
	if err := direct.Run(&testkernels.DispatchShapeKernel, shapeGroups,
		kernelabi.Args{Uniforms: []any{d}, Slices: []any{generated}}); err != nil {
		t.Fatalf("run: %v", err)
	}
	for i := range authored {
		if authored[i] != generated[i] {
			t.Fatalf("slot %d: authored %d, generated %d", i, authored[i], generated[i])
		}
	}
	// And it is not nine zeros agreeing, which every accessor returning nothing
	// would also produce.
	if authored[0] == 0 {
		t.Fatal("WorkgroupSize.X came back zero from both, so the comparison says nothing")
	}
}
