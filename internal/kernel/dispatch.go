// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel

import "fmt"

// Dispatch runs a flat kernel over a grid of workgroups.
//
// It lives here rather than in either of its two callers because both of them
// need exactly this loop: the CPU backend executing a graph node, and the
// bring-up path that runs a generated kernel without a device. Two copies would
// be two definitions of what a workgroup id is, and the second one to be
// updated is the one that quietly stops agreeing with spec 002.
//
// count is in workgroups, not invocations, for the same reason a dispatch is:
// the workgroup is the unit the kernel's own extent divides the grid into, and
// a thread count would make the caller do a division the kernel already knows
// the answer to.
func Dispatch(k *Kernel, count ID3, args Args) error {
	return DispatchWith(k, count, args, Options{})
}

// DispatchWith runs a flat kernel with explicit options.
//
// It exists for the reason [DispatchCooperativeWith] does, and it is the seam a
// determinism test needs: [Options.Workers] pins the pool size, so "parallel
// agrees with serial" is an assertion about the two strategies rather than
// about the GOMAXPROCS of whatever machine ran the test.
//
// Arguments are checked once, before the first invocation. The signature is the
// binding layout, so a mismatch is something generation already proved, and
// checking it inside the loop would report it once per invocation instead of
// once with the binding's name.
func DispatchWith(k *Kernel, count ID3, args Args, opts Options) error {
	if k == nil {
		return fmt.Errorf("accel: dispatch: no kernel")
	}
	if k.Flat == nil {
		return fmt.Errorf("accel: dispatch: kernel %q has no flat entry point, so it is "+
			"cooperative: its invocations rendezvous, and running them in sequence would be a "+
			"different program rather than a slower one", k.Name)
	}
	size := k.WorkgroupSize
	if size.X == 0 || size.Y == 0 || size.Z == 0 {
		return fmt.Errorf("accel: dispatch: kernel %q has workgroup extent %d,%d,%d, and an "+
			"axis of zero dispatches nothing", k.Name, size.X, size.Y, size.Z)
	}
	if err := k.Bind(args); err != nil {
		return err
	}

	grid := normalizeCount(count)
	groups := int(grid.X) * int(grid.Y) * int(grid.Z)
	workers := workerCount(k.OrderIndependent, groups*int(linear(size)), opts.Workers)
	// Every worker shares one argument set, and that is what a binding is: one
	// buffer the whole dispatch writes. A flat kernel carries no per-workgroup
	// state at all -- no frame, no shared storage, no scheduler -- so a worker
	// here is a loop and nothing else.
	return runGrid(grid, workers, func(_ int, group ID3) error {
		runFlat(k, args, group, size, grid)
		return nil
	})
}

// runFlat runs one workgroup's invocations, x fastest.
func runFlat(k *Kernel, args Args, group, size, count ID3) {
	for lz := range size.Z {
		for ly := range size.Y {
			for lx := range size.X {
				t := NewThread(
					ID3{
						X: group.X*size.X + lx,
						Y: group.Y*size.Y + ly,
						Z: group.Z*size.Z + lz,
					},
					ID3{X: lx, Y: ly, Z: lz}, group, size, count,
				)
				k.Flat(t, args)
			}
		}
	}
}
