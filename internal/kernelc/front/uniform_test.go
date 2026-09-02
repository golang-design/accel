// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package front_test

import (
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/front"
)

// A uniform member with more than four rows is refused at the parameter,
// naming the field. It was laid out as a matrix with the column count read as
// the row count too: `A [3][8]float32` got a 416-byte block, a `float A[3][4]`
// in MSL and an encoder writing three of its eight rows, while the body
// indexed A[2][7].
func TestAUniformMatrixWithMoreThanFourRowsIsRefused(t *testing.T) {
	const body = `type P struct {
	A [3][8]float32
}

//accel:kernel workgroup=64
func K(t accel.Thread, p P, out []float32) {
	out[t.GlobalID().X] = p.A[2][7]
}`
	pkgs := loadOverlay(t, map[string]string{"widematrix": header("widematrix") + body + "\n"},
		[]string{"./internal/kernelc/front/widematrix"})
	_, diags := front.Check(pkgs["./internal/kernelc/front/widematrix"])
	if len(diags) == 0 {
		t.Fatal("a [3][8]float32 uniform member was accepted, and no std140 matrix has eight rows")
	}
	d := diags[0]
	if d.Pos.Line != headerLines+6 {
		t.Errorf("reported at line %d, want the parameter's line %d: %v", d.Pos.Line, headerLines+6, d)
	}
	for _, want := range []string{"field A", "at most four rows", "storage buffer"} {
		if !strings.Contains(d.Msg, want) {
			t.Errorf("the message should carry %q: %s", want, d.Msg)
		}
	}
}
