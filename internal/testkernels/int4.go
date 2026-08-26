// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import "golang.design/x/accel"

// Int4Group is how many weights share one scale and one zero point.
//
// A hundred and twenty-eight, matching quant.Int4Group. Stated here rather than
// imported because a kernel's constants are what the generator reads, and
// specs/048-int4.md §2 has the argument: at four bits the metadata's share of
// the payload doubles, so the group grows to keep it at the 6.2% an fp16 scale
// per 32 costs an 8-bit format.
const Int4Group = 128

// QuantMatVecInt4 is a matrix-vector product over asymmetric 4-bit weights.
//
//	out[n] = Σₖ a[k] · ((nibble(bq, k·N+n) − z[g]) · s[g]),  g = (k·N+n)/Int4Group
//
// # Why the zero point, and why it is not free
//
// specs/048-int4.md §1. Sixteen codes spent symmetrically about zero are mostly
// wasted on a group of weights clustered away from it, so a group carries a
// scale *and* a zero point and the reconstruction is (q − z)·s rather than q·s.
// That is one extra subtract per weight and one extra f16 load per group, and
// it is what makes four bits usable at all.
//
// # The nibble order is a contract, not a detail
//
// Weight i lives in word i/8 at nibble i%8, low first. quant.Int4Quantize
// writes it that way and quant.Int4Nibble reads it that way, and this kernel is
// the third place the order is spelled. Two of the three agreeing is not enough
// -- a packer and an unpacker that agree with each other about the wrong order
// round-trip perfectly, and only a kernel reading real weights notices.
//
// Words rather than bytes because a kernel cannot do arithmetic on a u8:
// specs/002-compute-model.md makes narrow dtypes storage, converted to f32 on
// load, so a shift and a mask on one is outside the subset. The generator
// refuses it by name, which is how this representation was corrected before it
// shipped rather than after.
//
// # The products widen before accumulating
//
// [QuantMatVec]'s reason: the scale varies per group, so an integer accumulator
// would sum codes that mean different things. specs/048-int4.md §3's bound is
// stated against this evaluation.
//
//accel:kernel workgroup=128
func QuantMatVecInt4(t accel.Thread, d GEMMDims, a []float32, bq []uint32,
	bs []accel.Float16, bz []accel.Float16, out []float32, sh *[128]float32) {

	col := t.GroupID().X
	lid := t.LocalID().X

	acc := float32(0)
	if col < d.N {
		for k := lid; k < d.K; k += RowWidth {
			w := k*d.N + col

			// Weight w is in word w/8 at nibble w%8, low first. The same
			// spelling quant.Int4Nibble uses, and the third place this order is
			// written down.
			code := bq[w/8] >> (4 * (w % 8)) & 0xF

			g := w / Int4Group
			s := bs[g].F32()
			z := bz[g].F32()
			acc = acc + a[k]*((float32(code)-z)*s)
		}
	}
	sh[lid] = acc
	t.Barrier()

	for stride := uint32(RowWidth / 2); stride > 0; stride /= 2 {
		if lid < stride {
			sh[lid] = sh[lid] + sh[lid+stride]
		}
		t.Barrier()
	}

	if lid == 0 && col < d.N {
		out[col] = sh[0]
	}
}
