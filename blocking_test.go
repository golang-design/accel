// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/kernelabi"
)

// blocker is a kernel that does not return until released.
//
// It is what makes "while a submission is running" a state a test can hold
// the graph in, rather than a window it hopes to hit: started receives once
// per run when the kernel is inside its body, and release lets it leave. A
// test that hammered a queue from eight goroutines and counted refusals could
// only log when the machine serialized them all; this one knows.
type blocker struct {
	kernel  *kernelabi.Kernel
	started chan struct{}
	release chan struct{}
}

func newBlocker() *blocker {
	b := &blocker{started: make(chan struct{}, 64), release: make(chan struct{})}
	b.kernel = &kernelabi.Kernel{
		Name: "Block", Generator: kernelabi.Version,
		WorkgroupSize: kernelabi.ID3{X: 1, Y: 1, Z: 1},
		Flat: func(accel.Thread, kernelabi.Args) {
			b.started <- struct{}{}
			<-b.release
		},
	}
	return b
}

// pipeline compiles the blocker for d.
func (b *blocker) pipeline(t *testing.T, d *accel.Device) *accel.ComputePipeline {
	t.Helper()
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: b.kernel, Label: "block"})
	if err != nil {
		t.Fatalf("blocking pipeline: %v", err)
	}
	t.Cleanup(func() { _ = p.Close() })
	return p
}
