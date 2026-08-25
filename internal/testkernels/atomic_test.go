// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"golang.design/x/accel/kernelabi"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/direct"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
)

// Every atomic returns the previous value, which is what every target's
// instruction returns.
//
// Asserted operation by operation rather than through a total, because a sum
// hides a wrong answer: an add and an exchange that both left the right final
// value would pass a reduction test while returning different things.
func TestAtomicsReturnThePreviousValue(t *testing.T) {
	state := []uint32{10, 10, 10, 10, 0xFF, 0x0F, 0xAA, 1, 1, 2}
	before := append([]uint32(nil), state...)
	prev := make([]uint32, 10)

	if err := direct.Run(&testkernels.AtomicOpsKernel, accel.ID3{X: 1},
		kernelabi.Args{Slices: []any{state, prev}}); err != nil {
		t.Fatalf("run: %v", err)
	}

	for i, want := range before {
		if prev[i] != want {
			t.Errorf("operation %d returned %d, want the previous value %d", i, prev[i], want)
		}
	}

	// And each left the value its own definition says.
	for _, c := range []struct {
		at   int
		want uint32
		what string
	}{
		{0, 17, "add"},
		{1, 7, "sub"},
		{2, 5, "min, which stores because 5 < 10"},
		{3, 10, "max, which does not store because 5 < 10"},
		{4, 0x0F, "and"},
		{5, 0xFF, "or"},
		{6, 0x55, "xor"},
		{7, 42, "exchange, which stores unconditionally"},
		{8, 99, "compare-exchange whose comparison matched"},
		{9, 2, "compare-exchange whose comparison did not match"},
	} {
		if state[c.at] != c.want {
			t.Errorf("%s left %d, want %d", c.what, state[c.at], c.want)
		}
	}
}

// Compare-exchange is strong: success is exactly `returned == cmp`, and it
// never fails spuriously. Promising weak would invent a hazard for callers to
// loop around, since every target's is strong.
func TestCompareExchangeIsStrong(t *testing.T) {
	b := []uint32{5}
	for range 1000 {
		if got := kernel.CompareExchangeU32(b, 0, 5, 5); got != 5 {
			t.Fatalf("a matching compare-exchange returned %d and should never fail "+
				"spuriously", got)
		}
	}
	if got := kernel.CompareExchangeU32(b, 0, 4, 9); got != 5 {
		t.Errorf("a non-matching compare-exchange returned %d, want the observed 5", got)
	}
	if b[0] != 5 {
		t.Errorf("a non-matching compare-exchange stored %d and should store nothing", b[0])
	}
}

// The histogram is right, which it is not without atomics: every invocation
// updates a bucket some other invocation may be updating.
func TestHistogramCountsEveryElement(t *testing.T) {
	const n = 500
	in := make([]float32, n)
	for i := range in {
		in[i] = float32(i%100) / 100
	}
	counts := make([]uint32, 4)

	if err := direct.Run(&testkernels.HistogramKernel, direct.Cover(&testkernels.HistogramKernel, n),
		kernelabi.Args{Slices: []any{in, counts}}); err != nil {
		t.Fatalf("run: %v", err)
	}

	total := uint32(0)
	for _, c := range counts {
		total += c
	}
	if total != n {
		t.Errorf("the buckets total %d, want %d: an update was lost", total, n)
	}
	// Each quarter of the range gets a quarter of the inputs.
	for i, got := range counts {
		if want := uint32(n / 4); got != want {
			t.Errorf("bucket %d has %d, want %d", i, got, want)
		}
	}
}

// An atomic reads and writes the binding it names, which is what the graph
// builder infers dependency edges from. A binding that looked untouched would
// be a missing barrier, and therefore a race.
func TestAnAtomicMarksItsBindingReadWrite(t *testing.T) {
	var counts *kernelabi.Binding
	for i := range testkernels.HistogramKernel.Bindings {
		if b := &testkernels.HistogramKernel.Bindings[i]; b.Name == "counts" {
			counts = b
		}
	}
	if counts == nil {
		t.Fatal("Histogram has no counts binding")
	}
	if counts.Access != kernelabi.Read|kernelabi.Write {
		t.Errorf("the atomic's binding is %v, want read-write: an atomic is a "+
			"read-modify-write and the graph builder infers barriers from that",
			counts.Access)
	}
}

// The float atomic is a capability, and the record says so. A kernel using it
// on a device that lacks it must be refused rather than lowered to something
// else.
func TestTheFloatAtomicIsACapability(t *testing.T) {
	if testkernels.HistogramKernel.Caps != 0 {
		t.Errorf("Histogram uses only integer atomics and requires %d", testkernels.HistogramKernel.Caps)
	}
}

// The authored halves of the atomic kernels, which is spec 004's fifth testing
// level: the generated lowering is what runs, so nothing else calls these, and
// a kernel nobody executes means whatever the IR made of it.
func TestAuthoredAtomicKernels(t *testing.T) {
	const group = 64
	size := kernel.ID3{X: group, Y: 1, Z: 1}

	t.Run("Histogram", func(t *testing.T) {
		const n = 200
		in := make([]float32, n)
		for i := range in {
			in[i] = float32(i%100) / 100
		}
		counts := make([]uint32, 4)
		groups := uint32((n + group - 1) / group)
		for g := range groups {
			for l := range uint32(group) {
				th := kernel.NewThread(kernel.ID3{X: g*group + l},
					kernel.ID3{X: l}, kernel.ID3{X: g}, size, kernel.ID3{X: groups})
				testkernels.Histogram(th, in, counts)
			}
		}
		total := uint32(0)
		for _, c := range counts {
			total += c
		}
		if total != n {
			t.Errorf("the buckets total %d, want %d", total, n)
		}

		// And the generated lowering agrees.
		generated := make([]uint32, 4)
		if err := direct.Run(&testkernels.HistogramKernel, accel.ID3{X: groups},
			kernelabi.Args{Slices: []any{in, generated}}); err != nil {
			t.Fatalf("run: %v", err)
		}
		for i := range counts {
			if counts[i] != generated[i] {
				t.Errorf("bucket %d: authored %d, generated %d", i, counts[i], generated[i])
			}
		}
	})

	t.Run("AtomicOps", func(t *testing.T) {
		start := []uint32{10, 10, 10, 10, 0xFF, 0x0F, 0xAA, 1, 1, 2}

		authoredState := append([]uint32(nil), start...)
		authoredPrev := make([]uint32, 10)
		th := kernel.NewThread(kernel.ID3{}, kernel.ID3{}, kernel.ID3{},
			kernel.ID3{X: 1, Y: 1, Z: 1}, kernel.ID3{X: 1, Y: 1, Z: 1})
		testkernels.AtomicOps(th, authoredState, authoredPrev)

		genState := append([]uint32(nil), start...)
		genPrev := make([]uint32, 10)
		if err := direct.Run(&testkernels.AtomicOpsKernel, accel.ID3{X: 1},
			kernelabi.Args{Slices: []any{genState, genPrev}}); err != nil {
			t.Fatalf("run: %v", err)
		}

		for i := range authoredPrev {
			if authoredPrev[i] != genPrev[i] {
				t.Errorf("operation %d returned %d authored and %d generated",
					i, authoredPrev[i], genPrev[i])
			}
			if authoredState[i] != genState[i] {
				t.Errorf("operation %d left %d authored and %d generated",
					i, authoredState[i], genState[i])
			}
		}
	})
}

// The signed atomics arrive signed, and min and max are what say so.
//
// Every unsigned atomic was reached by a corpus kernel and no signed one was,
// so the emitter's lowering and the MSL spelling for all six were declared and
// never executed. This is the accepting half.
//
// Four of the six cannot distinguish a signed lowering from an unsigned one:
// add, sub, exchange and compare-exchange are the same machine operation on the
// same bits either way. They are here for reachability. **Min and max are the
// assertion**, because a comparison is where two's complement stops hiding the
// difference: state[2] holds -5 and the operand is 3, so a signed minimum keeps
// -5 and an unsigned one keeps 3.
func TestTheSignedAtomicsArriveSigned(t *testing.T) {
	state := []int32{10, 10, -5, -5, 10, -1, 7}
	prev := make([]int32, len(state))

	// One workgroup of one invocation, as [AtomicOps]'s test does: these
	// operations are a fixed sequence rather than a grid, and running the
	// sequence sixty-four times over would be testing the dispatch.
	if err := direct.Run(&testkernels.AtomicOpsI32Kernel, accel.ID3{X: 1},
		kernelabi.Args{Slices: []any{state, prev}}); err != nil {
		t.Fatalf("run: %v", err)
	}

	// Every atomic returns the value the location held before it.
	wantPrev := []int32{10, 10, -5, -5, 10, -1, 7}
	for i, w := range wantPrev {
		if prev[i] != w {
			t.Errorf("operation %d returned %d, want the previous %d", i, prev[i], w)
		}
	}

	wantState := []int32{
		3,   // 10 + (-7)
		13,  // 10 - (-3)
		-5,  // min(-5, 3) signed. Unsigned this is 3, which is the whole point
		3,   // max(-5, 3) signed. Unsigned this is -5, likewise
		-42, // exchanged
		-99, // compare-exchange matched -1 and stored
		7,   // compare-exchange did not match, so it left 7
	}
	for i, w := range wantState {
		if state[i] != w {
			t.Errorf("state %d is %d, want %d", i, state[i], w)
		}
	}

	// Stated separately so a failure here reads as what it is rather than as
	// one of seven equal assertions.
	if state[2] != -5 || state[3] != 3 {
		t.Fatalf("min gave %d and max gave %d over (-5, 3); signed they are -5 and 3, "+
			"and unsigned they are 3 and -5. This is the pair that tells a signed "+
			"lowering from an unsigned one, because add and exchange cannot",
			state[2], state[3])
	}
}

// The float atomic adds, and one invocation of it is deterministic.
//
// accel.AddF32 was the last operation in the atomic surface no kernel reached.
// It was left out of the signed-atomic sweep on the grounds that its result is
// numerics class E, and that conflated two things: **class E is about
// contention, not about the operation.** What specs/008-numerics.md classifies
// as non-deterministic is a reduction -- many invocations adding into one
// location in an order nobody fixes. One invocation doing one read-modify-write
// is an ordinary float addition and is exactly reproducible, which is why this
// asserts exact values rather than a bound.
func TestTheFloatAtomicReachesAKernel(t *testing.T) {
	state := []float32{1.5, 1.5}
	prev := make([]float32, 2)

	if err := direct.Run(&testkernels.AtomicAddF32Kernel, accel.ID3{X: 1},
		kernelabi.Args{Slices: []any{state, prev}}); err != nil {
		t.Fatalf("run: %v", err)
	}

	// Every atomic returns the value the location held before it, which is the
	// property the whole family shares and the one a ticket dispenser needs.
	for i, w := range []float32{1.5, 1.5} {
		if prev[i] != w {
			t.Errorf("operation %d returned %v, want the previous %v", i, prev[i], w)
		}
	}
	// Exact, not within a bound: the addends are small integers over a power of
	// two, so a difference here is the atomic and not the arithmetic.
	for i, w := range []float32{2.0, 1.25} {
		if state[i] != w {
			t.Errorf("state %d is %v, want %v", i, state[i], w)
		}
	}
}
