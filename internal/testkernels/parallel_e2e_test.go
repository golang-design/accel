// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"runtime"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// A dispatch answers the same bytes however many workers ran it.
//
// specs/006-backends.md §5 states the rule and internal/kernel checks the
// decision that carries it. This checks the whole path a caller uses instead:
// the generated kernel carries the order-independence the compiler inferred,
// the pipeline accepts it, and the backend reads it. A rule can be right in
// every unit and still reach nothing, and the flag is inferred rather than
// declared, so nothing in this file names it.
//
// The two runs are the same program on the same input, so any difference is
// the worker count. A million elements is enough to be over the pool
// threshold several hundred times over.
// It cannot be a parallel test: it moves GOMAXPROCS, which is process-global,
// so a sibling running at the same time would see the processor count this test
// pinned rather than the one it was started with.
func TestADispatchAnswersTheSameOnOneWorkerAndMany(t *testing.T) {
	if runtime.NumCPU() < 2 {
		t.Skip("one processor: there is no second worker count to disagree with")
	}
	dev, err := accel.OpenBest(accel.Policy{AllowCPU: true, Prefer: []accel.Backend{accel.BackendCPU}})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer dev.Close()

	pipe, err := dev.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: &testkernels.SiLUKernel})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer pipe.Close()

	const n = 1 << 20
	usage := accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst
	in, err := dev.NewBuffer(accel.BufferDescriptor{DType: accel.F32, Count: n, Usage: usage, Label: "in"})
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer in.Close()
	out, err := dev.NewBuffer(accel.BufferDescriptor{DType: accel.F32, Count: n, Usage: usage, Label: "out"})
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer out.Close()

	// Values that are neither uniform nor monotone, so a lane picking up the
	// wrong index is visible in the result rather than hidden by a smooth ramp.
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(i%13) - 6.5
	}
	if err := dev.Queue().WriteBuffer(in, 0, src); err != nil {
		t.Fatalf("write: %v", err)
	}
	inView, err := in.View(0, n)
	if err != nil {
		t.Fatalf("view: %v", err)
	}
	outView, err := out.View(0, n)
	if err != nil {
		t.Fatalf("view: %v", err)
	}

	wg := int(testkernels.SiLUKernel.WorkgroupSize.X)
	// Cleared before each run, because the output buffer survives between them:
	// a run that never wrote an element would otherwise be compared against the
	// previous run's correct value for it, and a workgroup that never executed
	// would read as agreement.
	zero := make([]float32, n)
	run := func(procs int) []float32 {
		defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(procs))
		if err := dev.Queue().WriteBuffer(out, 0, zero); err != nil {
			t.Fatalf("clear: %v", err)
		}
		if err := dev.Queue().Run(func(r *accel.Recorder) {
			r.Dispatch(pipe, []accel.Binding{
				{Index: 0, Buffer: inView},
				{Index: 1, Buffer: outView},
			}, nil, accel.WorkgroupCount{X: n / wg})
		}); err != nil {
			t.Fatalf("dispatch on %d processors: %v", procs, err)
		}
		got := make([]float32, n)
		if err := dev.Queue().ReadBuffer(out, 0, got); err != nil {
			t.Fatalf("read: %v", err)
		}
		return got
	}

	one, many := run(1), run(runtime.NumCPU())
	for i := range one {
		if one[i] != many[i] {
			t.Fatalf("element %d: one worker answered %v and %d workers answered %v, so this "+
				"backend's answer depends on how many goroutines ran it",
				i, one[i], runtime.NumCPU(), many[i])
		}
	}
}
