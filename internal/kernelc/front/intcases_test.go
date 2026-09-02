// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package front_test

import (
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/front"
)

// The excluded integer cases are build errors when the excluding operand is a
// constant.
//
// specs/008-numerics.md section 3: a shift count outside [0, 31] is an error,
// not a value, and Go admits it. With a constant the compiler can say so at
// the line; with a variable the CPU and Metal agree on Go's result. A constant
// zero divisor never reaches the front end: Go's type checker refuses it.
func TestConstantExcludedIntegerCasesAreBuildErrors(t *testing.T) {
	cases := []struct {
		name, body, says string
	}{
		{"shift by 32", "out[i] = in[i] << 32", "a shift by 32 is outside [0, 31]"},
		{"shift by 40", "out[i] = in[i] >> 40", "a shift by 40 is outside [0, 31]"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			pkg := checkSource(t, "package k\n\nimport \"golang.design/x/accel\"\n\n"+
				"//accel:kernel workgroup=64\n"+
				"func K(t accel.Thread, in []uint32, out []uint32) {\n\ti := t.GlobalID().X\n\t"+c.body+"\n}\n")
			if pkg == nil {
				t.Fatal("the source did not type-check")
			}
			_, diags := front.Check(pkg)
			if len(diags) == 0 {
				t.Fatalf("accepted; want %q", c.says)
			}
			if !strings.Contains(diags.Error(), c.says) {
				t.Fatalf("the diagnostic does not say %q:\n%v", c.says, diags)
			}
		})
	}
	// And a shift by 31 or a division by two is ordinary.
	pkg := checkSource(t, "package k\n\nimport \"golang.design/x/accel\"\n\n"+
		"//accel:kernel workgroup=64\n"+
		"func K(t accel.Thread, in []uint32, out []uint32) {\n\ti := t.GlobalID().X\n\tout[i] = (in[i] << 31) / 2\n}\n")
	if pkg == nil {
		t.Fatal("the source did not type-check")
	}
	if _, diags := front.Check(pkg); len(diags) > 0 {
		t.Fatalf("a shift by 31 and a division by two were refused: %v", diags)
	}
}
