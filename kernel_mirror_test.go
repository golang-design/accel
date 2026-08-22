// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"testing"

	"golang.design/x/accel/internal/kernel"
)

// TestDTypeMirrorsTheKernelPackage guards a duplication the compiler cannot.
//
// Limits and Capabilities cross the backend seam as structs, so the whole-struct
// conversion between them stops compiling the moment either side changes. DType
// crosses as an enum, and a defined type over int converts to another whatever
// the constants mean, so nothing fails to build if the two orders drift. A
// drifted order does not error either: it silently binds an f32 buffer to a
// kernel that reads i32, which is the wrong-answer class this project keeps
// out of the design everywhere else.
//
// So the parity is asserted here, by name and by width, over every value.
func TestDTypeMirrorsTheKernelPackage(t *testing.T) {
	pairs := []struct {
		public DType
		kern   kernel.DType
	}{
		{F32, kernel.F32},
		{F16, kernel.F16},
		{BF16, kernel.BF16},
		{I32, kernel.I32},
		{U32, kernel.U32},
		{I8, kernel.I8},
		{U8, kernel.U8},
	}

	if len(pairs) != len(dtypeInfo) {
		t.Fatalf("this test covers %d dtypes and accel declares %d: a dtype was added "+
			"without extending the mirror", len(pairs), len(dtypeInfo))
	}

	for _, p := range pairs {
		if int(p.public) != int(p.kern) {
			t.Errorf("accel.%v is %d and kernel.%v is %d: the enum orders have drifted, "+
				"so a converted dtype names a different type on each side",
				p.public, int(p.public), p.kern, int(p.kern))
		}
		if p.public.String() != p.kern.String() {
			t.Errorf("accel.DType(%d) is %q and kernel.DType(%d) is %q",
				int(p.public), p.public, int(p.kern), p.kern)
		}
	}

	// The conversion the generator relies on, exercised rather than assumed.
	for _, p := range pairs {
		if got := kernel.DType(p.public); got != p.kern {
			t.Errorf("kernel.DType(accel.%v) = %v, want %v", p.public, got, p.kern)
		}
	}
}

// TestThreadAliasIsTheSameType checks that the alias is an alias. If it ever
// became a defined type, generated code saying accel.Thread and a Kernel entry
// point saying kernel.Thread would stop being assignable, and the two halves of
// the pipeline would need a conversion that has nothing to convert.
func TestThreadAliasIsTheSameType(t *testing.T) {
	var a Thread = kernel.NewThread(ID3{X: 1}, ID3{}, ID3{}, ID3{X: 1, Y: 1, Z: 1}, ID3{X: 1, Y: 1, Z: 1})
	var k kernel.Thread = a
	if k.GlobalID().X != 1 {
		t.Errorf("GlobalID = %+v", k.GlobalID())
	}

	var i ID3 = kernel.ID3{X: 2}
	var j kernel.ID3 = i
	if j.X != 2 {
		t.Errorf("ID3 = %+v", j)
	}
}
