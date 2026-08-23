// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel_test

import (
	"testing"

	"golang.design/x/accel/internal/kernel"
)

// orderKernel records the order its invocations are advanced in.
//
// Each invocation appends its own local x on its first pass and finishes on the
// second, so one epoch produces one permutation of 0..n-1.
func orderKernel(n int, seen *[]uint32) *kernel.Kernel {
	return &kernel.Kernel{
		Name: "Order", WorkgroupSize: kernel.ID3{X: uint32(n), Y: 1, Z: 1},
		Generator: kernel.ABIVersion, Suspensions: 1,
		Cooperative: func(th kernel.Thread, a kernel.Args, f *kernel.Frame) bool {
			if f.Pass == 0 {
				*seen = append(*seen, th.LocalID().X)
				f.Pass = 1
				f.Barrier = kernel.BarrierID{Index: 0}
				return true
			}
			return false
		},
	}
}

func advanceOrder(t *testing.T, n int, seed uint64) []uint32 {
	t.Helper()
	var seen []uint32
	k := orderKernel(n, &seen)
	err := kernel.DispatchCooperativeWith(k, kernel.ID3{X: 1}, kernel.Args{},
		kernel.Options{ShuffleSeed: seed})
	if err != nil {
		t.Fatalf("dispatch (seed %d): %v", seed, err)
	}
	return seen[:n] // the first epoch
}

// Without a seed the advance order is signature order, which is what every
// reproducible run depends on.
func TestAdvanceOrderIsSignatureOrderByDefault(t *testing.T) {
	const n = 8
	got := advanceOrder(t, n, 0)
	for i, v := range got {
		if v != uint32(i) {
			t.Fatalf("invocation %d advanced at position %d; the default order is the "+
				"signature one", v, i)
		}
	}
}

// A seed permutes the order within an epoch, and does it deterministically.
//
// The option existed on CPUOptions, was stored on the device, and was readable
// back — and never reached the scheduler, so setting it changed nothing. A
// caller sweeping seeds to shake out an order-dependent kernel got the same
// order every time and concluded the kernel was fine.
//
// specs/018-cooperative-lowering.md §2 and specs/019-cooperative-diagnostics.md
// §5 both claimed this worked. The spec audit found the knob was not wired.
func TestShuffleSeedPermutesTheAdvanceOrder(t *testing.T) {
	const n = 16

	shuffled := advanceOrder(t, n, 0x5EED)
	if len(shuffled) != n {
		t.Fatalf("saw %d advances in the first epoch, want %d", len(shuffled), n)
	}

	// It is a permutation: every invocation runs exactly once per epoch.
	seen := make([]bool, n)
	for _, v := range shuffled {
		if int(v) >= n || seen[v] {
			t.Fatalf("invocation %d ran twice or out of range; the shuffle must permute, "+
				"not resample", v)
		}
		seen[v] = true
	}

	// It differs from signature order, or the option does nothing.
	same := true
	for i, v := range shuffled {
		if v != uint32(i) {
			same = false
			break
		}
	}
	if same {
		t.Error("a seeded run advanced in signature order, so the seed changed nothing")
	}

	// And it is deterministic in the seed: a failing run has to reproduce from
	// the seed alone, or it reports a bug nobody can re-run.
	again := advanceOrder(t, n, 0x5EED)
	for i := range shuffled {
		if shuffled[i] != again[i] {
			t.Fatalf("two runs with the same seed advanced differently at position %d", i)
		}
	}

	// A different seed gives a different order, or the seed is being ignored in
	// favour of one fixed permutation.
	other := advanceOrder(t, n, 0xC0FFEE)
	differs := false
	for i := range shuffled {
		if shuffled[i] != other[i] {
			differs = true
			break
		}
	}
	if !differs {
		t.Error("two different seeds produced the same order")
	}
}
