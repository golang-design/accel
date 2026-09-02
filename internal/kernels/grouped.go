// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import "golang.design/x/accel"

// GroupedDims is a grouped product's shape.
type GroupedDims struct {
	// Experts is how many weight matrices there are, so the offsets are
	// Experts+1 long. An expert with no tokens is one of them.
	Experts uint32

	// K is the contraction width and N the output width, shared by every
	// expert: they are the same layer, differing only in weights.
	K uint32
	N uint32
}

// GroupedMatVec multiplies each token by the weight matrix its segment names.
//
//	out[t][n] = Σₖ x[t][k] · w[e(t)][k][n]
//
// specs/049-grouped-gemm.md. A mixture-of-experts layer is E independent
// products whose row counts are runtime values, which is
// specs/046-segmented-extents.md's extent with the row being an expert rather
// than a sequence. Tokens arrive already ordered by expert, so each expert's
// rows are contiguous and the offsets say where.
//
// # Why the lookup is a count
//
// The same reason [AttentionRagged]'s is. Testing whether a token falls in a
// half-open range is correct, and is *also* correct with the wrong comparison
// on its upper bound so long as the loop keeps the last match -- so a reader who
// added a break would turn a typo into a token multiplied by the wrong expert's
// matrix. Counting the segments that end at or before the token has no such
// reading, and an expert with no tokens is counted for every token at or after
// it, which is what makes an empty segment cost nothing rather than a case.
//
// # A matvec per token, not a tiled product per expert
//
// Decode is where a mixture of experts earns its keep: a step routes one token
// to k experts and reads those k matrices rather than all E. The tiled form is
// what a prefill wants, over this same extent, and is not this kernel.
//
//accel:uniform offsets
//accel:kernel workgroup=128
func GroupedMatVec(t accel.Thread, d GroupedDims, x []float32, w []float32,
	offsets []uint32, out []float32, sh *[128]float32) {

	group := t.GroupID().X
	lid := t.LocalID().X

	tok := group / d.N
	col := group % d.N

	// Which expert this token routed to.
	e := uint32(0)
	for r := uint32(0); r < d.Experts; r++ {
		if offsets[r+1] <= tok {
			e = e + 1
		}
	}
	// A token past the last expert's segment routed nowhere, and is padding.
	//
	// [AttentionRagged]'s guard, for its reason, and one index short of it: the
	// count loop stops at Experts so it reads no offset out of range, but `e`
	// reaching Experts makes `wBase` address a matrix past the weight tensor --
	// on a GPU, whatever allocation follows it, read as weights.
	// specs/049-grouped-gemm.md §1.
	if e >= d.Experts {
		if lid == 0 {
			out[tok*d.N+col] = 0
		}
		return
	}
	wBase := e * d.K * d.N

	acc := float32(0)
	for k := lid; k < d.K; k += RowWidth {
		acc = acc + x[tok*d.K+k]*w[wBase+k*d.N+col]
	}
	sh[lid] = acc
	t.Barrier()

	for stride := uint32(RowWidth / 2); stride > 0; stride /= 2 {
		if lid < stride {
			sh[lid] = sh[lid] + sh[lid+stride]
		}
		t.Barrier()
	}

	if lid == 0 {
		out[tok*d.N+col] = sh[0]
	}
}
