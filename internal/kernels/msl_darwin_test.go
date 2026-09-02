// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package kernels_test

import (
	"os"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernels"
	"golang.design/x/accel/internal/mtl"
)

// Every kernel the MSL emitter produced source for compiles on the device.
//
// This is the check that a text golden cannot make. The offline Metal toolchain
// is frequently not installed, and it is not needed: -newLibraryWithSource: is
// the Metal compiler, so the thing that accepts the emitted text here is the
// same thing that will accept it in production. specs/021-metal-bringup.md
// section 1 argues that this is stronger evidence than a parse, and it is the
// reason the emitter is allowed to be a text transform at all.
//
// The corpus is read by reflection over the generated file rather than from a
// list, because a list goes stale the moment the corpus grows and the failure
// is silence: a kernel added and never compiled here would look exactly like a
// kernel that passes.
func TestEveryEmittedKernelCompilesOnTheDevice(t *testing.T) {
	devs, err := mtl.Devices()
	if err != nil || len(devs) == 0 {
		// A failure or a skip depending on what the job promised: see
		// specs/006-backends.md section 7 and the header of
		// .github/workflows/ci.yml. Tier 2 sets ACCEL_REQUIRE_METAL.
		if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
			t.Fatalf("this job promises Metal and found no device (err=%v)", err)
		}
		t.Skipf("no Metal device on this machine (err=%v)", err)
	}
	d := devs[0]
	defer func() {
		for _, x := range devs {
			x.Close()
		}
	}()

	kernels := corpus()
	if len(kernels) == 0 {
		t.Fatal("no kernel in the corpus carries MSL, so this test proves nothing")
	}
	for _, k := range kernels {
		t.Run(k.Name, func(t *testing.T) {
			p, err := d.Compile(k.MSL, k.Name)
			if err != nil {
				t.Fatalf("the device compiler rejected the emitted MSL:\n%v\n\n%s", err, k.MSL)
			}
			defer p.Close()
			if p.ThreadExecutionWidth == 0 {
				t.Error("a pipeline reporting a zero execution width is not a pipeline")
			}
			// The pipeline's own ceiling can be below the device's when a kernel
			// uses many registers, and it is the number a dispatch must respect.
			// A kernel whose declared workgroup does not fit is one this backend
			// must refuse rather than clamp, so the fact is checked here where
			// the number first exists.
			want := int(k.WorkgroupSize.X * k.WorkgroupSize.Y * k.WorkgroupSize.Z)
			if want > p.MaxTotalThreadsPerThreadgroup {
				t.Errorf("declared workgroup %v is %d invocations, above this pipeline's "+
					"ceiling of %d", k.WorkgroupSize, want, p.MaxTotalThreadsPerThreadgroup)
			}
		})
	}
}

// corpus returns every generated kernel carrying MSL.
func corpus() []*accel.Kernel {
	var out []*accel.Kernel
	for _, k := range kernels.Kernels {
		if k.MSL != "" {
			out = append(out, k)
		}
	}
	return out
}

// Every stage the MSL emitter produced source for compiles on the device.
//
// The same argument as the kernel case and the same compiler:
// -newLibraryWithSource: is what accepts the text in production. What differs is
// that a stage cannot go through Compile -- a vertex or fragment function is not
// a compute kernel and -newComputePipelineStateWithFunction: rejects one -- so
// this resolves the function instead, which also catches source that parses but
// declares the entry point under another name.
//
// A render pipeline state would prove more and needs more: both stages together
// plus attachment formats. That belongs with the Metal render path; this is the
// claim the emitter alone can make.
func TestEveryEmittedStageCompilesOnTheDevice(t *testing.T) {
	devs, err := mtl.Devices()
	if err != nil || len(devs) == 0 {
		if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
			t.Fatalf("this job promises Metal and found no device (err=%v)", err)
		}
		t.Skipf("no Metal device on this machine (err=%v)", err)
	}
	d := devs[0]
	defer func() {
		for _, x := range devs {
			x.Close()
		}
	}()

	var withMSL int
	for _, s := range kernels.Stages {
		if s.MSL == "" {
			t.Errorf("stage %s carries no MSL; every stage in the corpus is inside the "+
				"subset, so an empty one is a lowering that silently stopped working",
				s.Name)
			continue
		}
		withMSL++
		t.Run(s.Name, func(t *testing.T) {
			f, err := d.CompileFunction(s.MSL, s.Name)
			if err != nil {
				t.Fatalf("the device compiler rejected the emitted MSL:\n%v\n\n%s",
					err, s.MSL)
			}
			f.Close()
		})
	}
	if withMSL == 0 {
		t.Fatal("no stage in the corpus carries MSL, so this test proves nothing")
	}
}
