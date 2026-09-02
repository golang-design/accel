// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel_test

import (
	"runtime"
	"sync/atomic"
	"testing"

	"golang.design/x/accel/internal/kernel"
)

// rendezvous waits until every one of want workgroups has arrived.
//
// It yields rather than sleeps, so the wait is a number of scheduler turns
// rather than a clock: on one processor Gosched hands the processor to the
// next runnable goroutine, so every workgroup arrives within a bounded number
// of turns whatever the machine. The bound is what turns a pool that did not
// run the workgroups together into a failure rather than a hang.
func rendezvous(arrived *atomic.Int64, want int64) bool {
	arrived.Add(1)
	for turn := 0; turn < 1<<20; turn++ {
		if arrived.Load() >= want {
			return true
		}
		runtime.Gosched()
	}
	return false
}

// The pool runs an order-independent kernel's workgroups together, and under
// that contention a load-add-store on a shared location loses updates.
//
// This is the test that gives the order-independence gate its teeth on this
// backend, and it is why the gate cannot be loosened for kernels that use
// atomics. The CPU backend's atomics are plain reads and writes (atomic.go),
// safe only because a kernel that reaches one is inferred order-dependent and
// runs on one worker. A kernel that used `counts[b]++` where it should have
// used AddU32 is inferred order-*independent*, because the compiler infers the
// property as the absence of any atomic (specs/006-backends.md section 5, rule
// 1), and so it is exactly the kernel the pool runs at once -- and this is
// what happens to it: every workgroup reads the counter, waits until all the
// others have read it too, and stores its increment. Eight workgroups, one
// survivor, every time.
//
// The update is written with an atomic load and an atomic store rather than a
// bare read and write so that the race detector sees a synchronized program:
// what is under test is the lost update, which a load-then-store has whether
// or not its two halves are individually atomic, and a bare race would end
// the run under -race before the count could be read.
func TestThePoolContendsAndALoadAddStoreLosesUpdates(t *testing.T) {
	const groups = 8
	var arrived, late atomic.Int64
	counter := []uint32{0}
	k := &kernel.Kernel{
		Name: "Contend", Generator: kernel.ABIVersion,
		WorkgroupSize: kernel.ID3{X: 1, Y: 1, Z: 1},
		// What the compiler infers for a body with no atomic in it.
		OrderIndependent: true,
		Bindings: []kernel.Binding{
			{Name: "counter", DType: kernel.U32, Access: kernel.Read | kernel.Write},
		},
		Flat: func(_ kernel.Thread, a kernel.Args) {
			c := kernel.Slice[uint32](a, 0)
			v := atomic.LoadUint32(&c[0])
			if !rendezvous(&arrived, groups) {
				late.Add(1)
				return
			}
			atomic.StoreUint32(&c[0], v+1)
		},
	}
	if err := kernel.DispatchWith(k, kernel.ID3{X: groups},
		kernel.Args{Slices: []any{counter}}, kernel.Options{Workers: groups}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if n := late.Load(); n != 0 {
		t.Fatalf("%d of %d workgroups never met the others: the pool did not run them "+
			"together, so nothing here was under contention", n, groups)
	}
	if got := counter[0]; got != 1 {
		t.Fatalf("a load-add-store from %d workgroups that had all read the counter "+
			"left it at %d; every one of them read 0 and stored 1", groups, got)
	}
}
