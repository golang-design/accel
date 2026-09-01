// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package intrin_test

import (
	"testing"

	"golang.design/x/accel/internal/kernelc/intrin"
	"golang.design/x/accel/internal/kernelc/ir"
)

// The table is the only statement of each intrinsic's uniformity, and the
// analysis reads it through ByOp. An entry that states none would be read as
// the zero value, so the zero value states nothing and this test refuses it.
func TestEveryIntrinsicStatesItsUniformity(t *testing.T) {
	all := intrin.All()
	if len(all) == 0 {
		t.Fatal("the table is empty, so this checked nothing")
	}
	for _, in := range all {
		if in.Uniformity.String() == "unstated" {
			t.Errorf("%s states no uniformity: the analysis would read it as nothing", in.Authored)
		}
		byOp, ok := intrin.ByOp(in.Op)
		if !ok || byOp.Op != in.Op {
			t.Errorf("%s is not reachable through ByOp(%v)", in.Authored, in.Op)
		}
	}
	if _, ok := intrin.ByOp(ir.OpInvalid); ok {
		t.Error("ByOp resolved the invalid opcode")
	}
}

// Spec 002 section 3.3's seed table, as the intrinsic table states it. These
// are the rows the analysis used to keep a second copy of, and SubgroupIndex
// is the one the two copies disagreed on.
func TestSeedRowsMatchSpec002(t *testing.T) {
	for _, c := range []struct {
		op   ir.Opcode
		want intrin.Uniformity
	}{
		{ir.OpGroupID, intrin.PerWorkgroup},
		{ir.OpGroupIndex, intrin.PerWorkgroup},
		{ir.OpWorkgroupSize, intrin.PerWorkgroup},
		{ir.OpNumGroups, intrin.PerWorkgroup},
		{ir.OpGlobalSize, intrin.PerWorkgroup},
		{ir.OpSubgroupSize, intrin.PerWorkgroup},
		{ir.OpSubgroupID, intrin.PerSubgroup},
		{ir.OpLocalID, intrin.PerInvocation},
		{ir.OpLocalIndex, intrin.PerInvocation},
		{ir.OpGlobalID, intrin.PerInvocation},
		{ir.OpGlobalIndex, intrin.PerInvocation},
		{ir.OpSubgroupInvocationID, intrin.PerInvocation},
		{ir.OpAtomicAddU32, intrin.PerInvocation},
		{ir.OpAtomicAddF32, intrin.PerInvocation},
		{ir.OpSubgroupAddF32, intrin.PerInvocation},
		{ir.OpBroadcastFirstF32, intrin.PerInvocation},
		{ir.OpBallot, intrin.PerInvocation},
		{ir.OpSqrt, intrin.OfOperands},
		{ir.OpMaskCount, intrin.OfOperands},
		{ir.OpF16ToF32, intrin.OfOperands},
	} {
		in, ok := intrin.ByOp(c.op)
		if !ok {
			t.Errorf("no intrinsic lowers to %v", c.op)
			continue
		}
		if in.Uniformity != c.want {
			t.Errorf("%s is %v, want %v", in.Authored, in.Uniformity, c.want)
		}
	}
}

func TestUniformityStringsAreDistinct(t *testing.T) {
	seen := map[string]bool{}
	for _, u := range []intrin.Uniformity{
		intrin.PerInvocation, intrin.PerSubgroup, intrin.PerWorkgroup, intrin.OfOperands,
	} {
		s := u.String()
		if s == "unstated" || seen[s] {
			t.Errorf("Uniformity(%d).String() = %q", int(u), s)
		}
		seen[s] = true
	}
	if intrin.Uniformity(0).String() != "unstated" {
		t.Error("the zero value should read as unstated")
	}
}
