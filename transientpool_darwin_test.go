// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package accel_test

import "testing"

// A shared pool grows on Metal without invalidating the graphs already built
// into it.
//
// This is the one scenario the handle indirection exists for, and the CPU
// backend cannot prove it: it resolves a block in one accessor, where Metal
// reaches its own block type in three separate places. Dropping driver.Unwrap
// from those three fails this test by name -- "names a *accel.poolBlock, which
// is not Metal memory" -- which is what makes them load-bearing rather than
// defensive.
//
// The order matters: the small graph is built first, the large one second so
// the growth happens under it, and only then is the small one submitted. What
// this does *not* prove is that Metal reports a freed block, because it does
// not: handing the graph the pre-growth allocation directly still produced the
// right answer here. The CPU backend is where that failure is visible, and it
// is where the bug was found.
func TestSharedTransientPoolGrowsOnMetal(t *testing.T) {
	d := openMetal(t)

	pool, err := d.NewTransientPool("metal")
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	defer pool.Close()

	small, smallOut := buildInto(t, d, pool, 64)
	before := pool.Bytes()

	// Four times the elements, so the pool has to reallocate under `small`.
	buildInto(t, d, pool, 256)
	if pool.Bytes() <= before {
		t.Fatalf("the pool holds %d bytes after a larger graph and %d before, so it never "+
			"grew and this test proves nothing", pool.Bytes(), before)
	}

	if err := d.Queue().Submit(small).Wait(); err != nil {
		t.Fatalf("the graph built before the pool grew: %v", err)
	}
	for i, v := range readback(t, d, smallOut) {
		if v != 4 {
			t.Fatalf("element %d is %v, want 4, after the pool grew underneath the graph",
				i, v)
		}
	}
}
