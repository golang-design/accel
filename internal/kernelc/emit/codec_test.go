// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package emit_test

import (
	"strings"
	"testing"
)

// The generated encoder writes every element of a non-square matrix at its
// column's sixteen-byte stride, and every element of an array at its own.
//
// The encoder looped the column count for the rows as well, so a 2x4 wrote
// M[c][0] and M[c][1] only, and a 4x2 wrote M[c][2] and M[c][3] from past the
// end of the Go array. The corpus's MatrixShapes is the fixture, and its
// differential test is where the block declaration and the codec meet.
func TestCodecWritesEveryMatrixElementAtItsStride(t *testing.T) {
	out := string(generateCorpus(t))
	for _, want := range []string{
		// Wide [2][4]: two columns of four, sixteen bytes apart.
		"w.F32(0, value.Wide[0][0])", "w.F32(12, value.Wide[0][3])",
		"w.F32(16, value.Wide[1][0])", "w.F32(28, value.Wide[1][3])",
		// Tall [4][2] at 32: four columns of two, sixteen bytes apart.
		"w.F32(32, value.Tall[0][0])", "w.F32(36, value.Tall[0][1])",
		"w.F32(80, value.Tall[3][0])", "w.F32(84, value.Tall[3][1])",
		// Column [6] at 96: one element per sixteen-byte slot.
		"w.F32(96, value.Column[0])", "w.F32(176, value.Column[5])",
		"const MatrixParamsBlockSize = 192",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the generated codec does not carry %q", want)
		}
	}
	for _, reject := range []string{
		"value.Wide[2]", "value.Wide[0][4]", "value.Tall[0][2]", "value.Column[6]",
	} {
		if strings.Contains(out, reject) {
			t.Errorf("the generated codec indexes past the Go array: %q", reject)
		}
	}
}
