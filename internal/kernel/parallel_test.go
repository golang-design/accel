// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel_test

import (
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"golang.design/x/accel/internal/kernel"
)

// gauge records how many workgroups were inside a kernel body at once.
//
// It is how "this dispatch ran in parallel" is asserted by observation rather
// than by inference from a clock: a timing comparison would be a machine test,
// and a rendezvous that only completes when two workgroups meet would hang
// forever the day the pool is switched off. So each workgroup announces itself,
// waits a bounded time for company, and leaves. The wait is what makes an
// overlap observable; the bound is what keeps a serial run a fast test rather
// than a hang.
type gauge struct {
	live atomic.Int64
	most atomic.Int64
}

func (g *gauge) visit(want int64, wait time.Duration) {
	n := g.live.Add(1)
	for {
		most := g.most.Load()
		if n <= most || g.most.CompareAndSwap(most, n) {
			break
		}
	}
	deadline := time.Now().Add(wait)
	for g.most.Load() < want && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	g.live.Add(-1)
}

// gaugeKernel announces every workgroup on entry, once per workgroup.
func gaugeKernel(g *gauge, orderIndependent bool, size uint32) *kernel.Kernel {
	return &kernel.Kernel{
		Name: "Gauge", Generator: kernel.ABIVersion,
		WorkgroupSize:    kernel.ID3{X: size, Y: 1, Z: 1},
		OrderIndependent: orderIndependent,
		Flat: func(t kernel.Thread, a kernel.Args) {
			if t.LocalID().X == 0 {
				g.visit(2, 25*time.Millisecond)
			}
		},
	}
}

// A kernel the compiler proved order-independent runs its workgroups at once.
//
// Observed rather than timed: the gauge reports the largest number of
// workgroups that were inside the body together, and one is what the old
// single-goroutine loop reported.
func TestAnOrderIndependentDispatchRunsWorkgroupsAtOnce(t *testing.T) {
	var g gauge
	k := gaugeKernel(&g, true, 1)
	if err := kernel.DispatchWith(k, kernel.ID3{X: 8}, kernel.Args{},
		kernel.Options{Workers: 8}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if got := g.most.Load(); got < 2 {
		t.Fatalf("at most %d workgroups were running together: the pool is not running "+
			"anything at once, so the dispatch is still the one-goroutine loop", got)
	}
}

// A kernel the compiler did not prove order-independent runs on one worker,
// whatever the caller asked for.
//
// This is the oracle rule in one test. Every atomic accel offers returns the
// value the location held before it, so an atomic increment is a ticket
// dispenser: the counter's total does not depend on the order the workgroups
// ran in and the tickets do. Running the grid in order is what makes the ticket
// each workgroup drew reproducible, and reproducible is the whole promise of
// this backend.
func TestAnOrderDependentDispatchRunsInGridOrder(t *testing.T) {
	// Large enough that one worker draining the queue before its peers wake is
	// unlikely rather than usual: at 64 workgroups of one multiply, that
	// schedule is common enough that the rule can be removed and this test
	// still passes.
	const groups = 4096
	k := &kernel.Kernel{
		Name: "Ticket", Generator: kernel.ABIVersion,
		WorkgroupSize:    kernel.ID3{X: 1, Y: 1, Z: 1},
		OrderIndependent: false,
		Bindings: []kernel.Binding{
			{Name: "counter", DType: kernel.U32, Access: kernel.Read | kernel.Write},
			{Name: "tickets", DType: kernel.U32, Access: kernel.Write},
		},
		Flat: func(t kernel.Thread, a kernel.Args) {
			counter := kernel.Slice[uint32](a, 0)
			tickets := kernel.Slice[uint32](a, 1)
			tickets[t.GroupID().X] = kernel.AddU32(counter, 0, 1)
		},
	}

	// Workers is asked for explicitly, so a pass here is the order-dependence
	// gate holding rather than the size threshold hiding the question.
	//
	// What a pass is not is proof: this watches the rule from the outside, and
	// whether two workers overlap at all is the scheduler's to decide. See
	// TestOrderDependenceOverrulesAnExplicitWorkerCount for the guard that has
	// no race in it.
	counter, tickets := make([]uint32, 1), make([]uint32, groups)
	if err := kernel.DispatchWith(k, kernel.ID3{X: groups},
		kernel.Args{Slices: []any{counter, tickets}}, kernel.Options{Workers: 8}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	for i, v := range tickets {
		if v != uint32(i) {
			t.Fatalf("workgroup %d drew ticket %d: the grid did not run in order, so a "+
				"kernel whose result depends on that order got a different answer than "+
				"the serial oracle gives", i, v)
		}
	}
	if counter[0] != groups {
		t.Fatalf("the counter reached %d over %d workgroups: an update was lost",
			counter[0], groups)
	}
}

// A dispatch too small to pay for a pool runs on one worker.
//
// The threshold is measured on the cheapest kernel there is, so this is the
// promise that adding the pool cannot make an existing small dispatch slower.
func TestASmallDispatchRunsOnOneWorker(t *testing.T) {
	var g gauge
	// Eight workgroups of one invocation, which is three orders of magnitude
	// below the threshold and two workgroups above the one-workgroup case, so
	// it is the size rule being tested and not the count rule.
	k := gaugeKernel(&g, true, 1)
	if err := kernel.Dispatch(k, kernel.ID3{X: 8}, kernel.Args{}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if got := g.most.Load(); got != 1 {
		t.Fatalf("%d workgroups ran together in a dispatch of 8 invocations: a pool over "+
			"a dispatch this small costs more than the loop it replaces", got)
	}
}

// A dispatch of one workgroup runs on one worker, because a pool over one
// workgroup is a goroutine and a join around the loop that was going to run.
func TestOneWorkgroupRunsOnOneWorker(t *testing.T) {
	var g gauge
	k := gaugeKernel(&g, true, 1)
	if err := kernel.DispatchWith(k, kernel.ID3{X: 1}, kernel.Args{},
		kernel.Options{Workers: 8}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if got := g.most.Load(); got != 1 {
		t.Fatalf("%d workgroups ran together in a one-workgroup dispatch", got)
	}
}

// indexKernel writes a value derived only from the invocation's own id, which
// is what a workgroup-independent kernel looks like.
func indexKernel() *kernel.Kernel {
	return &kernel.Kernel{
		Name: "Index", Generator: kernel.ABIVersion,
		WorkgroupSize:    kernel.ID3{X: 4, Y: 2, Z: 2},
		OrderIndependent: true,
		Bindings:         []kernel.Binding{{Name: "out", DType: kernel.F32, Access: kernel.Write}},
		Flat: func(t kernel.Thread, a kernel.Args) {
			out := kernel.Slice[float32](a, 0)
			i := t.GlobalIndex()
			if int(i) < len(out) {
				g := t.GroupID()
				out[i] = float32(g.X)*0.5 - float32(g.Y)*0.25 + float32(g.Z)*0.125 +
					float32(t.LocalIndex())
			}
		},
	}
}

// A parallel dispatch produces the bytes a serial one produces, over a
// three-dimensional grid and every worker count between them.
//
// Three-dimensional because that is where a linearization can be wrong without
// being obviously wrong: a pool numbers its work and a nested loop does not, so
// the two agree only if the numbering walks x fastest the way the loop did.
func TestAParallelDispatchAgreesWithASerialOne(t *testing.T) {
	k := indexKernel()
	count := kernel.ID3{X: 5, Y: 3, Z: 2}
	n := int(count.X*count.Y*count.Z) * 16

	serial := make([]float32, n)
	if err := kernel.DispatchWith(k, count, kernel.Args{Slices: []any{serial}},
		kernel.Options{Workers: 1}); err != nil {
		t.Fatalf("serial dispatch: %v", err)
	}

	for _, workers := range []int{2, 3, 7, 8, 64} {
		got := make([]float32, n)
		if err := kernel.DispatchWith(k, count, kernel.Args{Slices: []any{got}},
			kernel.Options{Workers: workers}); err != nil {
			t.Fatalf("%d workers: %v", workers, err)
		}
		for i := range got {
			if got[i] != serial[i] {
				t.Fatalf("%d workers: element %d is %v and the serial run gives %v: the "+
					"grid walk disagrees with the loop it replaced",
					workers, i, got[i], serial[i])
			}
		}
	}
}

// The same dispatch run many times gives byte-identical results.
//
// The important test of the pair. Agreement with a serial run once could be one
// lucky interleaving; this is the property a caller relies on, and it is the
// one a pool over shared state would break intermittently rather than always.
func TestAParallelDispatchIsRepeatable(t *testing.T) {
	k := indexKernel()
	count := kernel.ID3{X: 9, Y: 4, Z: 3}
	n := int(count.X*count.Y*count.Z) * 16

	var first []float32
	for run := range 50 {
		got := make([]float32, n)
		if err := kernel.Dispatch(k, count, kernel.Args{Slices: []any{got}}); err != nil {
			t.Fatalf("run %d: %v", run, err)
		}
		if first == nil {
			first = got
			continue
		}
		for i := range got {
			if got[i] != first[i] {
				t.Fatalf("run %d differs from run 0 at element %d: %v against %v",
					run, i, got[i], first[i])
			}
		}
	}
}

// A panicking kernel panics on the goroutine that called the dispatch.
//
// The CPU backend recovers there and reports through the fence, which is how an
// out-of-bounds kernel becomes an error rather than a dead process. A panic
// left inside a worker would abort from a goroutine nobody can recover in,
// which is the thing that recover was put there to prevent.
func TestAPanicInAWorkerReachesTheCaller(t *testing.T) {
	k := &kernel.Kernel{
		Name: "Overrun", Generator: kernel.ABIVersion,
		WorkgroupSize:    kernel.ID3{X: 1, Y: 1, Z: 1},
		OrderIndependent: true,
		Flat: func(t kernel.Thread, a kernel.Args) {
			panic(fmt.Sprintf("workgroup %d", t.GroupID().X))
		},
	}

	var got any
	func() {
		defer func() { got = recover() }()
		_ = kernel.DispatchWith(k, kernel.ID3{X: 32}, kernel.Args{},
			kernel.Options{Workers: 8})
	}()
	if got == nil {
		t.Fatal("a panicking kernel did not panic on the calling goroutine: the CPU " +
			"backend recovers there, so the process would have died instead")
	}
	// Workgroup zero is claimed before any worker can have failed, so it is the
	// lowest-numbered failure every time. That is what the serial loop would
	// have reported.
	if s, _ := got.(string); s != "workgroup 0" {
		t.Fatalf("the panic reported is %v, and the serial loop would have stopped at "+
			"workgroup 0", got)
	}
}

// tileKernel sums each workgroup's invocations through shared memory, which
// makes it cooperative: a barrier separates the publish from the read.
func tileKernel() *kernel.Kernel {
	return &kernel.Kernel{
		Name: "Tile", Generator: kernel.ABIVersion,
		WorkgroupSize:    kernel.ID3{X: 8, Y: 1, Z: 1},
		OrderIndependent: true,
		SharedSizes:      []int{8}, Suspensions: 1,
		Bindings: []kernel.Binding{{Name: "out", DType: kernel.F32, Access: kernel.Write}},
		NewShared: func() []any {
			var sh [8]float32
			kernel.Poison(sh[:])
			return []any{&sh}
		},
		Cooperative: func(t kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			sh := kernel.SharedSlice[[8]float32](a, 0)
			out := kernel.Slice[float32](a, 0)
			l := t.LocalIndex()
			if f.Pass == 0 {
				f.Shared.Write(0, int(l))
				sh[l] = float32(t.GlobalIndex()) * 0.25
				f.Pass = 1
				f.Barrier = kernel.BarrierID{Index: 0}
				return true
			}
			if l == 0 {
				var sum float32
				for i := range sh {
					sum += sh[f.Shared.ReadAt(0, i)]
				}
				out[t.GroupIndex()] = sum
			}
			return false
		},
	}
}

// A parallel cooperative dispatch produces the bytes a serial one produces.
//
// The scheduler carries state a workgroup owns -- frames, ids, shared storage,
// the access tracker -- so this is where a pool over one copy of it would show
// up: two workgroups advancing through one set of frames is two programs
// writing one scratchpad, and the answer would move between runs.
func TestAParallelCooperativeDispatchAgreesWithASerialOne(t *testing.T) {
	k := tileKernel()
	count := kernel.ID3{X: 7, Y: 3, Z: 2}
	n := int(count.X * count.Y * count.Z)

	serial := make([]float32, n)
	if err := kernel.DispatchCooperativeWith(k, count,
		kernel.Args{Slices: []any{serial}},
		kernel.Options{Diagnostics: true, Workers: 1}); err != nil {
		t.Fatalf("serial dispatch: %v", err)
	}

	for _, workers := range []int{2, 5, 8, 64} {
		for run := range 5 {
			got := make([]float32, n)
			if err := kernel.DispatchCooperativeWith(k, count,
				kernel.Args{Slices: []any{got}},
				kernel.Options{Diagnostics: true, Workers: workers}); err != nil {
				t.Fatalf("%d workers run %d: %v", workers, run, err)
			}
			for i := range got {
				if got[i] != serial[i] {
					t.Fatalf("%d workers run %d: workgroup %d summed to %v and the serial "+
						"run gives %v", workers, run, i, got[i], serial[i])
				}
			}
		}
	}
}

// The shuffle mode stays reproducible when the grid runs on a pool.
//
// A seed exists so a failing run re-runs from the seed alone. The permutation
// is per epoch inside one workgroup, so it has to be unaffected by which worker
// picked that workgroup up, or the seed would name a schedule nobody could
// reproduce.
func TestAShuffledParallelDispatchIsRepeatable(t *testing.T) {
	k := tileKernel()
	count := kernel.ID3{X: 11, Y: 2, Z: 1}
	n := int(count.X * count.Y * count.Z)

	var first []float32
	for run := range 20 {
		got := make([]float32, n)
		if err := kernel.DispatchCooperativeWith(k, count,
			kernel.Args{Slices: []any{got}},
			kernel.Options{Diagnostics: true, ShuffleSeed: 0x5EED, Workers: 8}); err != nil {
			t.Fatalf("run %d: %v", run, err)
		}
		if first == nil {
			first = got
			continue
		}
		for i := range got {
			if got[i] != first[i] {
				t.Fatalf("run %d differs from run 0 at workgroup %d: a shuffled dispatch "+
					"has to reproduce from its seed", run, i)
			}
		}
	}
}

// A diagnostic found on a pool names the workgroup the serial loop would have
// stopped at.
//
// The tracker is per worker, so several workgroups can report at once. Which
// one a reader is shown cannot depend on which worker got there first: a report
// that named a different workgroup each run is a report a developer learns to
// re-run past.
func TestTheLowestNumberedWorkgroupIsReported(t *testing.T) {
	// Every workgroup reads shared memory nothing wrote, so every one of them
	// has something to report and the choice between them is the whole test.
	k := &kernel.Kernel{
		Name: "Reader", Generator: kernel.ABIVersion,
		WorkgroupSize:    kernel.ID3{X: 4, Y: 1, Z: 1},
		OrderIndependent: true,
		SharedSizes:      []int{8},
		NewShared: func() []any {
			var sh [8]float32
			return []any{&sh}
		},
		Cooperative: func(t kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			sh := kernel.SharedSlice[[8]float32](a, 0)
			_ = sh[f.Shared.ReadAt(0, 3)]
			return false
		},
	}

	for run := range 20 {
		err := kernel.DispatchCooperativeWith(k, kernel.ID3{X: 32}, kernel.Args{},
			kernel.Options{Diagnostics: true, Workers: 8})
		if err == nil {
			t.Fatal("a read of undefined shared memory was not reported")
		}
		if !strings.Contains(err.Error(), "workgroup 0,0,0") {
			t.Fatalf("run %d reports %v, and the serial loop would have stopped at "+
				"workgroup 0,0,0", run, err)
		}
	}
}

// The stuck-scheduler backstop still reports when the grid runs on a pool.
//
// It is the one failure that is not a diagnostic, so it takes the other path
// out of a worker, and a pool that dropped it would turn a report into a hang.
func TestAStuckKernelIsReportedFromAWorker(t *testing.T) {
	k := &kernel.Kernel{
		Name: "Stuck", Generator: kernel.ABIVersion,
		WorkgroupSize:    kernel.ID3{X: 1, Y: 1, Z: 1},
		OrderIndependent: true,
		Cooperative: func(t kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			// Never advances, which is what a generated program counter that
			// does not move looks like. It still names its barrier, as a
			// generated lowering does: every invocation is at the same one
			// every epoch, so the arrival check has nothing to report and the
			// epoch bound is what catches it.
			f.Barrier = kernel.BarrierID{Index: 0}
			return true
		},
	}
	err := kernel.DispatchCooperativeWith(k, kernel.ID3{X: 16}, kernel.Args{},
		kernel.Options{Diagnostics: true, Workers: 8})
	if err == nil || !strings.Contains(err.Error(), "rendezvous epochs") {
		t.Fatalf("a kernel that never finishes should be reported, got %v", err)
	}
}

// A cooperative kernel that is not order-independent runs on one worker, for
// the reason the flat one does.
func TestAnOrderDependentCooperativeDispatchRunsOnOneWorker(t *testing.T) {
	var g gauge
	k := &kernel.Kernel{
		Name: "GaugeCoop", Generator: kernel.ABIVersion,
		WorkgroupSize:    kernel.ID3{X: 1, Y: 1, Z: 1},
		OrderIndependent: false,
		Cooperative: func(t kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			g.visit(2, 25*time.Millisecond)
			return false
		},
	}
	if err := kernel.DispatchCooperativeWith(k, kernel.ID3{X: 8}, kernel.Args{},
		kernel.Options{Diagnostics: true, Workers: 8}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if got := g.most.Load(); got != 1 {
		t.Fatalf("%d workgroups of an order-dependent kernel ran together", got)
	}
}
