// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel

import (
	"runtime"
	"testing"
)

// The determinism rule is a property of a decision, so it is tested on the
// decision.
//
// TestAnOrderDependentDispatchRunsInGridOrder watches the same rule from the
// outside, by reading the tickets a grid of workgroups drew. That test is worth
// having and it is not a gate: with the rule removed it reports the violation
// about nineteen runs in twenty, because whether two workers overlap at all is
// up to the scheduler, and one worker draining a queue of cheap workgroups
// before its peers wake is a legal schedule that happens to produce grid order.
// A guard that holds only most of the time is not holding the line; it is
// waiting to be the flake somebody deletes.
//
// [workerCount] is where the rule is actually written down, and asking it
// directly has no race in it at all.
func TestOrderDependenceOverrulesAnExplicitWorkerCount(t *testing.T) {
	for _, c := range []struct {
		name             string
		orderIndependent bool
		invocations      int
		want             int
		workers          int
	}{
		// The gate is checked before anything else, so neither a caller naming
		// a pool size nor a dispatch large enough to want one can get past it.
		{"asked for a pool", false, parallelThreshold * 8, 8, 1},
		{"large enough for one", false, parallelThreshold * 8, 0, 1},
		{"asked for one worker", false, parallelThreshold * 8, 1, 1},

		// An order-independent kernel is the only one the size question is
		// asked about.
		{"order independent, asked for a pool", true, 1, 8, 8},
		{"order independent, below the threshold", true, parallelThreshold - 1, 0, 1},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := workerCount(c.orderIndependent, c.invocations, c.want)
			if got != c.workers {
				why := "an order-dependent kernel on more than one worker can report a " +
					"different answer than the serial oracle gives"
				if c.orderIndependent {
					why = "the pool this dispatch was sized for is not the one it got"
				}
				t.Fatalf("workerCount(orderIndependent=%v, invocations=%d, want=%d) = %d, want %d: %s",
					c.orderIndependent, c.invocations, c.want, got, c.workers, why)
			}
		})
	}
}

// A dispatch at or above the threshold asks for the machine it is on.
func TestALargeOrderIndependentDispatchAsksForEveryProcessor(t *testing.T) {
	if got, want := workerCount(true, parallelThreshold, 0), runtime.GOMAXPROCS(0); got != want {
		t.Fatalf("workerCount at the threshold = %d, want %d", got, want)
	}
}
