// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal_test

import (
	"testing"
	"time"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/mtl"
)

// A submission releases the command buffer of the one before it.
//
// Queue.Begin retains the buffer and nothing released it until the executable
// closed, so an executable submitted a thousand times held a thousand command
// buffers. Metal does not report that and Go's heap does not show it, because
// the objects live on the Objective-C side; what shows it is counting what was
// retained against what was released.
func TestASubmissionReleasesThePreviousCommandBuffer(t *testing.T) {
	e := benchExecutable(t, 4, false)
	before := mtl.LiveCommandBuffers()

	const runs = 50
	for range runs {
		f, err := e.Submit()
		if err != nil {
			t.Fatalf("submit: %v", err)
		}
		if err := f.Wait(); err != nil {
			t.Fatalf("wait: %v", err)
		}
	}
	// One is allowed: the most recent submission's, which the executable keeps
	// so Elapsed and IndirectStats can answer for it.
	if held := mtl.LiveCommandBuffers() - before; held > 1 {
		t.Fatalf("%d submissions left %d command buffers retained; the executable "+
			"should hold at most the last one", runs, held)
	}
	if err := e.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if held := mtl.LiveCommandBuffers() - before; held != 0 {
		t.Fatalf("a closed executable still holds %d command buffers", held)
	}
}

// A fence the caller still holds after its command buffer was released keeps
// answering: complete, with the error and the device time the submission had.
func TestAReplacedFenceStillAnswers(t *testing.T) {
	e := benchExecutable(t, 4, false)
	first, err := e.Submit()
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	if err := first.Wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
	second, err := e.Submit()
	if err != nil {
		t.Fatalf("second submit: %v", err)
	}
	if err := second.Wait(); err != nil {
		t.Fatalf("second wait: %v", err)
	}
	if !first.Done() {
		t.Error("the first fence reports not done after its successor replaced it")
	}
	if err := first.Wait(); err != nil {
		t.Errorf("the first fence reports %v after its successor replaced it", err)
	}
}

// spinSource is a kernel that runs for as long as its argument says.
//
// The iteration count comes from a buffer rather than a literal so that the
// compiler cannot reason about the loop, and the chain is data-dependent so it
// cannot be vectorised away. What it computes is irrelevant; that it takes a
// measurable time is the whole point.
const spinSource = `#include <metal_stdlib>
using namespace metal;
kernel void spin(device float *out [[buffer(0)]],
                 const device uint *n [[buffer(1)]],
                 constant uint *_lens [[buffer(2)]],
                 uint3 _gid [[thread_position_in_grid]]) {
  float x = float(_gid.x);
  uint count = n[0];
  for (uint k = 0; k < count; k++) { x = x * 1.0000001f + 0.5f; }
  out[_gid.x] = x;
}`

// spinExecutable compiles a one-node plan running spinSource for the given
// iteration count.
func spinExecutable(t *testing.T, d driver.Device, iterations uint32) driver.Executable {
	t.Helper()
	k := synthetic("spin", spinSource, kernel.ID3{X: 64, Y: 1, Z: 1})
	k.Bindings = []kernel.Binding{
		{Name: "out", DType: kernel.F32, Access: kernel.Write},
		{Name: "n", DType: kernel.U32, Access: kernel.Read},
	}
	out := mustAlloc(t, d, 64*4)
	n := mustAlloc(t, d, 4)
	copy(n.Bytes(), []byte{byte(iterations), byte(iterations >> 8),
		byte(iterations >> 16), byte(iterations >> 24)})
	whole := func(b driver.Block) driver.Operand {
		o, err := driver.BlockOperand(b, 0, b.Size())
		if err != nil {
			t.Fatalf("operand: %v", err)
		}
		return o
	}
	e, err := d.(driver.GraphCompiler).Compile(&driver.Plan{Nodes: []driver.PlanNode{{
		Op: driver.OpDispatch, Dispatch: &driver.Dispatch{
			Kernel: k, Count: kernel.ID3{X: 1, Y: 1, Z: 1},
			Bindings: []driver.Operand{whole(out), whole(n)},
		},
	}}})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	t.Cleanup(func() { _ = e.Close() })
	return e
}

// Done answers at once while another goroutine is inside Wait.
//
// Wait held the fence's mutex across the GPU wait, so Done -- documented
// non-blocking, and what Submit, Rebind and Close consult through busy() --
// blocked until the kernel finished. A caller polling a fence from one
// goroutine while another waited on it was a caller who could not poll.
func TestDoneDoesNotBlockBehindAWaiter(t *testing.T) {
	d := open(t)
	e := spinExecutable(t, d, spinIterations(t, d))

	f, err := e.Submit()
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	waited := make(chan time.Duration, 1)
	go func() {
		t0 := time.Now()
		_ = f.Wait()
		waited <- time.Since(t0)
	}()
	// Enough for the goroutine to reach the wait. A Done that returned before
	// the waiter got there would prove nothing.
	time.Sleep(20 * time.Millisecond)

	t0 := time.Now()
	done := f.Done()
	took := time.Since(t0)
	waitTook := <-waited

	t.Logf("Done took %v and reported %v; Wait took %v", took, done, waitTook)
	if waitTook < 100*time.Millisecond {
		t.Fatalf("the kernel finished in %v, which is too fast for Done to have had "+
			"anything to block behind", waitTook)
	}
	if took > waitTook/2 {
		t.Fatalf("Done took %v while a Wait of %v was in progress, so it blocked "+
			"behind the waiter", took, waitTook)
	}
}

// spinIterations picks a count that keeps the device busy for a fraction of a
// second on this machine: long enough that a blocked Done is visible, short
// enough to stay well clear of the GPU watchdog.
func spinIterations(t *testing.T, d driver.Device) uint32 {
	t.Helper()
	const probe = 1 << 22
	e := spinExecutable(t, d, probe)
	f, err := e.Submit()
	if err != nil {
		t.Fatalf("submit: %v", err)
	}
	t0 := time.Now()
	if err := f.Wait(); err != nil {
		t.Fatalf("wait: %v", err)
	}
	per := time.Since(t0) / probe
	if per == 0 {
		t.Fatalf("%d iterations took no measurable time", probe)
	}
	want := uint32(300 * time.Millisecond / per)
	t.Logf("%v per iteration; using %d iterations", per, want)
	return want
}
