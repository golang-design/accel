// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"fmt"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
	"golang.design/x/accel/quant"
)

// Int4 is a weight matrix stored as packed 4-bit codes with a scale and a zero
// point per group.
//
// Three planes rather than two, and the third is what makes it four bits.
// [Quantized]'s int8 is symmetric, so a scale is enough; at four bits the codes
// have to be spent where the weights actually are, which needs an offset as
// well. specs/048-int4.md §1.
//
// Bundled for [Quantized]'s reason: binding one matrix's codes against another
// matrix's scales compiles, runs, and produces a matrix of noise.
type Int4 struct {
	// Codes is u32, eight weights per element, low nibble first. See
	// [quant.Int4Quantize] — a caller does not pack this by hand.
	Codes *Tensor

	// Scales and Zeros are f16, one each per [quant.Int4Group] weights.
	Scales *Tensor
	Zeros  *Tensor

	// Weights is how many weights Codes holds, which the element count cannot
	// give: eight per word means a matrix of 8n and one of 8n-3 pack into the
	// same number of words. specs/048-int4.md §4.
	Weights int
}

// checkInt4 reports why a packed matrix is not usable, or "".
func checkInt4(q Int4, what string) string {
	switch {
	case q.Codes == nil || q.Scales == nil || q.Zeros == nil:
		return what + " needs a code plane, a scale plane and a zero plane; the zero " +
			"point is what makes four bits usable and is not optional " +
			"(specs/048-int4.md section 1)"
	case q.Codes.dtype != accel.U32:
		return fmt.Sprintf("%s codes are %v and must be u32: eight weights per word, "+
			"and a kernel cannot do arithmetic on a narrower type", what, q.Codes.dtype)
	case q.Scales.dtype != accel.F16:
		return fmt.Sprintf("%s scales are %v and must be f16", what, q.Scales.dtype)
	case q.Zeros.dtype != accel.F16:
		return fmt.Sprintf("%s zeros are %v and must be f16", what, q.Zeros.dtype)
	case q.Weights <= 0:
		return fmt.Sprintf("%s declares %d weights; the count cannot be derived from "+
			"the word count, because eight weights per word makes several matrices "+
			"pack into the same words", what, q.Weights)
	}

	if want := (q.Weights + 7) / 8; q.Codes.shape.Elements() != want {
		return fmt.Sprintf("%s declares %d weights, which is %d words, and the code "+
			"plane holds %d", what, q.Weights, want, q.Codes.shape.Elements())
	}
	// The group counts follow from the weight count, so a mismatch means the
	// planes describe different matrices -- what bundling them prevents, caught
	// here for a caller who built the triple by hand.
	groups := (q.Weights + quant.Int4Group - 1) / quant.Int4Group
	for _, p := range []struct {
		name string
		t    *Tensor
	}{{"scales", q.Scales}, {"zeros", q.Zeros}} {
		if got := p.t.shape.Elements(); got != groups {
			return fmt.Sprintf("%s has %d weights, which is %d groups of %d, and %d %s",
				what, q.Weights, groups, quant.Int4Group, got, p.name)
		}
	}
	return ""
}

// Int4MatVec multiplies f32 activations by a packed 4-bit weight matrix.
//
//	out[n] = Σₖ a[k] · ((code(k·N+n) − z[g]) · s[g])
//
// # Why a separate operator
//
// [QuantMatMul]'s reason, one width down: the cost is different and a caller
// should see which one they wrote. What differs here is also the *accuracy*,
// and not in one direction — specs/048-int4.md §3 states the bound as a group's
// range over 30 where int8's is a peak over 254, so a matrix whose weights
// cluster away from zero is represented better by these four bits than by eight
// symmetric ones, and one centred on zero is about seventeen times worse.
//
// # A vector, not a matrix
//
// This is the shape a decode step has, which is where the memory pressure
// four-bit weights exist to relieve is felt: a decode reads the whole model to
// produce one token. A prefill wants a tiled form over the same representation
// and specs/048-int4.md §5 records it as not built.
func Int4MatVec(b *Builder, a *Tensor, w Int4) *Tensor {
	// The triple is checked before the poison test, and the order matters. A
	// nil plane is poison as far as poisoned() is concerned, so testing that
	// first would return a silent poison where a caller who left out the zero
	// plane deserves to be told which one is missing -- and the graph would
	// then fail with "declares no output", naming nothing. Found by writing the
	// refusal test, which is what a refusal test is for.
	if why := checkInt4(w, "the weight matrix"); why != "" {
		return b.fail(1, "Int4MatVec", "%s", why)
	}
	if poisoned(a) || poisoned(w.Codes, w.Scales, w.Zeros) {
		return b.poison()
	}
	if a.dtype != accel.F32 {
		return b.fail(1, "Int4MatVec", "activations are %v and this kernel reads f32",
			a.dtype)
	}
	if len(a.shape) != 1 {
		return b.fail(1, "Int4MatVec", "activations are %v; a matvec takes a vector, "+
			"and a batch of them is the tiled form specs/048-int4.md section 5 records "+
			"as not built", a.shape)
	}
	k := a.shape[0]
	if k == 0 || w.Weights%k != 0 {
		return b.fail(1, "Int4MatVec", "activations are %d long and the matrix holds "+
			"%d weights; the matrix is [K, N] with K the activation width, so the "+
			"second is a multiple of the first", k, w.Weights)
	}
	n := w.Weights / k

	return b.record(node{
		op: "Int4MatVec", inputs: []*Tensor{a, w.Codes, w.Scales, w.Zeros},
		kernel: &testkernels.QuantMatVecInt4Kernel,
		uniform: func(map[string]ScalarValue) any {
			return testkernels.GEMMDims{K: uint32(k), N: uint32(n)}
		},
		grid: func(*Tensor) accel.WorkgroupCount {
			// One workgroup per output column, each reducing over K. The row
			// kernels' shape, and the reason Int4Group is 128: a group is
			// exactly one workgroup's width.
			return accel.WorkgroupCount{X: n}
		},
		reason: fmt.Sprintf("the 4-bit row kernel: %d columns over %d weights, "+
			"unpacked eight per word with a scale and a zero per %d",
			n, k, quant.Int4Group),
		rejected: []string{"the int8 row kernel: it reads one scale per block and no " +
			"zero point, so it cannot express where a 4-bit group's codes sit"},
	}, accel.F32, Shape{n})
}

// Int4MatMul multiplies a batch of f32 activations by a packed 4-bit matrix.
//
//	out[m][n] = Σₖ a[m][k] · ((code(k·N+n) − z[g]) · s[g])
//
// # Why a separate operator from [Int4MatVec]
//
// The shapes a decode step and a prefill have, and they want different kernels
// for the reason specs/048-int4.md §5 gives. A decode reads the whole model to
// produce one token, so its matvec is bound by how fast the weights arrive. A
// prefill has many tokens against the same matrix, so the matrix is read once
// per tile of tokens and the unpacking is amortised over the tile rather than
// repeated per token.
//
// Taking a vector here would work and would be slower than [Int4MatVec], and
// taking a matrix there would not compile. A caller should see which one they
// wrote, which is [QuantMatMul]'s argument one width down.
//
// The accuracy is [Int4MatVec]'s: the same representation, the same
// reconstruction, and specs/048-int4.md §3's bound covers both.
func Int4MatMul(b *Builder, a *Tensor, w Int4) *Tensor {
	// Checked before the poison test, for the reason Int4MatVec records.
	if why := checkInt4(w, "the weight matrix"); why != "" {
		return b.fail(1, "Int4MatMul", "%s", why)
	}
	if poisoned(a) || poisoned(w.Codes, w.Scales, w.Zeros) {
		return b.poison()
	}
	if a.dtype != accel.F32 {
		return b.fail(1, "Int4MatMul", "activations are %v and this kernel reads f32",
			a.dtype)
	}
	if len(a.shape) != 2 {
		return b.fail(1, "Int4MatMul", "activations are %v; this is the batched form "+
			"and takes [rows, K]. One row is Int4MatVec, which is the decode shape "+
			"and reads the matrix once per token", a.shape)
	}
	m, k := a.shape[0], a.shape[1]
	if k == 0 || w.Weights%k != 0 {
		return b.fail(1, "Int4MatMul", "activations are %d wide and the matrix holds "+
			"%d weights; the matrix is [K, N] with K the activation width, so the "+
			"second is a multiple of the first", k, w.Weights)
	}
	n := w.Weights / k

	return b.record(node{
		op: "Int4MatMul", inputs: []*Tensor{a, w.Codes, w.Scales, w.Zeros},
		kernel: &testkernels.QuantMatMulInt4Kernel,
		uniform: func(map[string]ScalarValue) any {
			return testkernels.GEMMDims{M: uint32(m), K: uint32(k), N: uint32(n)}
		},
		grid: func(*Tensor) accel.WorkgroupCount {
			// One workgroup per output tile. The tile is the reason this
			// operator exists: every weight it unpacks is read TileM times.
			return accel.WorkgroupCount{
				X: (n + testkernels.TileN - 1) / testkernels.TileN,
				Y: (m + testkernels.TileM - 1) / testkernels.TileM,
			}
		},
		reason: fmt.Sprintf("the tiled 4-bit kernel: %d rows over %d columns, each "+
			"weight unpacked once per tile and read %d times", m, n, testkernels.TileM),
		rejected: []string{"the 4-bit row kernel: it reduces one row per workgroup, so " +
			"a batch would re-read and re-unpack the whole matrix per token"},
	}, accel.F32, Shape{m, n})
}
