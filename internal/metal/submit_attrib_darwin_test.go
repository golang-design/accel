// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal_test

import (
	"fmt"
	"testing"
	"time"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/metal"
	"golang.design/x/accel/internal/testkernels"
)

// BenchmarkSubmitAttribution splits a submission into the host half and the
// device half, and measures each against the node count.
//
// A consumer reported the whole interval as 15.6% of a decode step on a graph
// of about 790 nodes -- the largest cost in the step that is not a kernel --
// and could not see which part it was. specs/006-backends.md section 4.3
// predicted the shape and said re-encoding "stops being fine somewhere in the
// thousands of nodes", so the question is not the total but whether it is paid
// per node or per submission. Those have different fixes: a per-node cost is
// what an indirect command buffer removes, and a per-submission one is not.
//
// Submit is the host half: a fresh command buffer, the encode, and the commit.
// Wait is the device half plus the fence round trip. Reported separately
// because only the first is accel's to spend.
//
// Run with -bench SubmitAttribution -benchtime 100x.
func BenchmarkSubmitAttribution(b *testing.B) {
	for _, nodes := range []int{1, 64, 256, 790} {
		b.Run(fmt.Sprintf("nodes=%d", nodes), func(b *testing.B) {
			e := benchExecutable(b, nodes)

			var submit, wait time.Duration
			b.ResetTimer()
			for range b.N {
				t0 := time.Now()
				f, err := e.Submit()
				if err != nil {
					b.Fatalf("submit: %v", err)
				}
				t1 := time.Now()
				if err := f.Wait(); err != nil {
					b.Fatalf("wait: %v", err)
				}
				t2 := time.Now()
				submit += t1.Sub(t0)
				wait += t2.Sub(t1)
			}
			b.StopTimer()

			n := time.Duration(b.N)
			b.ReportMetric(float64(submit/n)/1e3, "us/submit")
			b.ReportMetric(float64(wait/n)/1e3, "us/wait")
			// The figure that decides the fix. A cost that is flat in the node
			// count is paid once a submission and an indirect command buffer
			// would not touch it.
			b.ReportMetric(float64((submit/n).Nanoseconds())/float64(nodes), "ns/node")
		})
	}
}

// benchExecutable compiles a plan of n independent dispatches over one buffer.
//
// No barrier between them, so the plan encodes into one encoder rather than n.
// That is deliberate: it isolates the per-dispatch encoding cost, which is what
// scales with a model's node count, from the per-encoder cost a barrier adds.
// A real decode graph has both, and separating them is the point of the
// measurement.
func benchExecutable(b *testing.B, n int) driver.Executable {
	b.Helper()
	ads, err := metal.Adapters()
	if err != nil || len(ads) == 0 {
		b.Skipf("no Metal adapter on this machine (err=%v)", err)
	}
	d, err := ads[0].Open(nil)
	if err != nil {
		b.Fatalf("open: %v", err)
	}
	b.Cleanup(func() { _ = d.Close() })

	const count = 256
	const bytes = count * 4
	mk := func(label string) driver.Block {
		blk, err := d.Alloc(driver.MemoryShared, bytes, label)
		if err != nil {
			b.Fatalf("alloc %s: %v", label, err)
		}
		b.Cleanup(blk.Free)
		return blk
	}
	in1, in2, out := mk("in1"), mk("in2"), mk("out")
	whole := func(blk driver.Block) driver.Operand {
		o, err := driver.BlockOperand(blk, 0, bytes)
		if err != nil {
			b.Fatalf("operand: %v", err)
		}
		return o
	}
	plan := &driver.Plan{Label: "attrib"}
	for i := range n {
		plan.Nodes = append(plan.Nodes, driver.PlanNode{
			Op: driver.OpDispatch, ID: i,
			Dispatch: &driver.Dispatch{
				Kernel:   &testkernels.AddKernel,
				Count:    kernel.ID3{X: count / 64, Y: 1, Z: 1},
				Bindings: []driver.Operand{whole(in1), whole(in2), whole(out)},
			},
		})
	}
	exe, err := d.(driver.GraphCompiler).Compile(plan)
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	b.Cleanup(func() { _ = exe.Close() })
	return exe
}
