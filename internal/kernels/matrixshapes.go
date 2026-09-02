// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import "golang.design/x/accel"

// MatrixParams holds the uniform member shapes whose two extents differ.
//
// A square matrix cannot tell a codec that reads its column count as its row
// count from one that does not, and the corpus had only a 4x4. These are the
// shapes that can: a 2x4 encoded as a 2x2 loses six of its eight values, a 4x2
// encoded as a 4x4 writes eight values past the struct, and an array of six
// has a sixteen-byte stride a vector does not.
type MatrixParams struct {
	Wide   [2][4]float32
	Tall   [4][2]float32
	Column [6]float32
}

// MatrixShapes writes every member of its uniform to a distinct element of
// out, which is the device check specs/014-kernel-uniforms.md section 4 asks
// for: the CPU reads the Go struct directly and Metal reads the std140 bytes
// the generated codec wrote, so the two agree only if the codec, the block
// declaration and the layout all agree.
//
// Element i of out is, in order, Wide[c][r] for c then r, Tall[c][r] for c
// then r, and Column[i].
//
//accel:kernel workgroup=32
func MatrixShapes(t accel.Thread, p MatrixParams, out []float32) {
	i := t.GlobalID().X
	if i >= uint32(len(out)) {
		return
	}
	if i < 8 {
		out[i] = p.Wide[i/4][i%4]
		return
	}
	if i < 16 {
		out[i] = p.Tall[(i-8)/2][(i-8)%2]
		return
	}
	if i < 22 {
		out[i] = p.Column[i-16]
		return
	}
	out[i] = 0
}
