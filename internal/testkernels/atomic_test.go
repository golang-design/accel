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
