// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package direct runs a generated flat kernel without a device.
//
// It is compiler bring-up and test infrastructure, not a submission API. There
// is no [accel.Graph] until M3, and until there is, this is the only way to
// execute what the generator produced and find out whether it means what the
// authored source said.
//
// It disappears behind the common harness once graphs exist. Whether it should
// survive as harness-internal is spec 012's open question, and the argument for
// keeping it is that a planning bug and a lowering bug look alike from a graph:
// a path that does not depend on planning being correct is worth something when
// the two have to be told apart.
//
// # Flat only, and why that is a restriction rather than a simplification
//
// A cooperative kernel has no direct-call form by construction. Its invocations
// rendezvous at barriers, so running them one after another does not produce a
// slower version of the right answer; it produces a different program. The
// generated record carries no flat entry point for one, and this refuses rather
// than inventing an order.
package direct

import (
	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
)

// Run executes a kernel over a grid of workgroups.
//
// count is in workgroups, not invocations, for the same reason a dispatch is:
// the workgroup is the unit the kernel's own extent divides the grid into, and
// a thread count would make the caller do a division the kernel already knows
// the answer to.
//
// The arguments are checked once, before the first invocation. The signature is
// the binding layout, so a mismatch is something generation already proved, and
// checking it inside the loop would report it once per invocation instead of
// once with the binding's name.
func Run(k *accel.Kernel, count accel.ID3, args accel.KernelArgs) error {
	return kernel.Dispatch(k, count, args)
}

// Groups is how many workgroups cover n invocations along one axis.
//
// It rounds up, which is what a dispatch does, and it is the reason a kernel
// carries a bounds check: the last workgroup runs invocations past the end of
// the data, and nothing but the kernel's own guard stops them writing there.
func Groups(n, extent uint32) uint32 {
	if extent == 0 {
		return 0
	}
	return (n + extent - 1) / extent
}

// Cover is [Groups] for the common one-dimensional case.
func Cover(k *accel.Kernel, n int) accel.ID3 {
	if k == nil || n < 0 {
		return accel.ID3{}
	}
	return accel.ID3{X: Groups(uint32(n), k.WorkgroupSize.X), Y: 1, Z: 1}
}
