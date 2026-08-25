// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels

import (
	"golang.design/x/accel"
	"golang.design/x/accel/kmath"
)

// The flat family of spec 010: elementwise arithmetic, activations, gather and
// scatter, and rotary position embedding.
//
// Flat because none of them needs its invocations to cooperate — each output
// element depends on its own inputs and nothing else — which is what makes the
// flat lowering the right one and why spec 018's selection rule picks it.

// ElemAdd sums two tensors elementwise.
//
//accel:kernel workgroup=64
func ElemAdd(t accel.Thread, a []float32, b []float32, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		out[i] = a[i] + b[i]
	}
}

// ElemMul multiplies two tensors elementwise.
//
//accel:kernel workgroup=64
func ElemMul(t accel.Thread, a []float32, b []float32, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		out[i] = a[i] * b[i]
	}
}

// ScaleParams carries a runtime scalar.
//
// A uniform rather than a one-element binding, because it is one value for the
// whole dispatch and a binding would make the barrier analysis treat it as
// memory some invocation might write. Spec 010 says the scalar is decoded from
// uniform data for that reason.
type ScaleParams struct{ Factor float32 }

// ElemScale multiplies by a runtime scalar.
//
//accel:kernel workgroup=64
func ElemScale(t accel.Thread, p ScaleParams, in []float32, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		out[i] = in[i] * p.Factor
	}
}

// SiLU is x·σ(x), the sigmoid linear unit.
//
//	silu(x) = x / (1 + exp(-x))
//
// # Why the negated exponent
//
// Writing it as x·exp(x)/(1+exp(x)) is algebraically identical and overflows for
// x above about 88, where exp(x) is +Inf and the quotient is Inf/Inf. With the
// exponent negated, a large positive x gives exp(-x) → 0 and the result → x,
// which is what the function does; a large negative x gives a large exp(-x) and
// the result → 0, which is also right. Neither end overflows.
//
//accel:kernel workgroup=64
func SiLU(t accel.Thread, in []float32, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		x := in[i]
		out[i] = x / (float32(1) + kmath.Exp(-x))
	}
}

// SwiGLU is silu(a)·b, fused.
//
// Fused rather than two kernels because the intermediate is the same size as
// the output and writing it costs a full round trip through memory for one
// multiply. Spec 010 lists it as an authored fused kernel for that reason.
//
//accel:kernel workgroup=64
func SwiGLU(t accel.Thread, a []float32, b []float32, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		x := a[i]
		out[i] = x / (float32(1) + kmath.Exp(-x)) * b[i]
	}
}

// RowParams is a gather or scatter's shape.
type RowParams struct {
	// Rows is how many rows are moved and Width how wide each is.
	Rows  uint32
	Width uint32

	// Capacity is how many rows the table or state holds, which is what an id
	// is checked against.
	Capacity uint32
}

// GatherRows copies table[ids[r]] into out[r].
//
// # Why the id is range-checked
//
// An id past the table reads whatever follows it in memory, which for a token
// embedding is another token's vector: a plausible number, in the right shape,
// that silently changes what the model saw. Spec 010 makes the strict path
// check the range, and this checks in every mode — a bounds check on a gather
// costs one comparison against a load that already went to memory.
//
// An out-of-range id writes zeros rather than skipping the row, so the output
// is defined whatever the input was.
//
//accel:kernel workgroup=64
func GatherRows(t accel.Thread, p RowParams, table []float32, ids []uint32,
	out []float32) {

	i := t.GlobalID().X
	if i < p.Rows*p.Width {
		r := i / p.Width
		c := i % p.Width
		id := ids[r]
		if id < p.Capacity {
			out[i] = table[id*p.Width+c]
		} else {
			out[i] = float32(0)
		}
	}
}

// GatherRowsF16 is [GatherRows] over an f16 table, widening on load.
//
// # Why the table may be narrow where an accumulator may not
//
// specs/002-compute-model.md's rule is that a narrow type is storage which
// converts on load, and this is that case exactly: a gather performs no
// arithmetic at all. There is nothing to accumulate and therefore nothing to
// lose -- the value read is the value written, one conversion wider.
//
// It exists because the embedding table is the largest single tensor in a small
// model and had no width between f32 and int8. A consumer costed it at 1.56 GB
// against 778 MB for a 151936 x 2560 vocabulary, inside an 8 GB budget, and the
// choice was to hold the one tensor most sensitive to quantization at full
// width or to quantize it (accel issue 11).
//
// The output is f32 because what follows an embedding lookup is a normalize,
// and specs/010-kernel-corpus.md registers that at f32. Narrowing here would
// make the caller widen again at the next operator.
//
//accel:kernel workgroup=64
func GatherRowsF16(t accel.Thread, p RowParams, table []accel.Float16, ids []uint32,
	out []float32) {

	i := t.GlobalID().X
	if i < p.Rows*p.Width {
		r := i / p.Width
		c := i % p.Width
		id := ids[r]
		if id < p.Capacity {
			out[i] = table[id*p.Width+c].F32()
		} else {
			out[i] = float32(0)
		}
	}
}

// ScatterRows writes rows[r] into state[ids[r]].
//
// The inverse of [GatherRows] and the one that mutates persistent state, which
// is why spec 010 says it first executes through a public graph: a kernel that
// writes state the caller keeps between dispatches is exactly what the graph's
// hazard tracking exists for.
//
// An out-of-range id writes nothing, rather than clamping to the last row.
// Clamping would corrupt a real row with another's contents, which is worse
// than dropping the write and much harder to notice.
//
//accel:kernel workgroup=64
func ScatterRows(t accel.Thread, p RowParams, rows []float32, ids []uint32,
	state []float32) {

	i := t.GlobalID().X
	if i < p.Rows*p.Width {
		r := i / p.Width
		c := i % p.Width
		id := ids[r]
		if id < p.Capacity {
			state[id*p.Width+c] = rows[i]
		}
	}
}

// ScatterRowsF16 is [ScatterRows] over an f16 state, from f16 rows.
//
// # Why the state may be narrow
//
// The same argument as [GatherRowsF16] and for the same reason: a scatter
// performs no arithmetic, so specs/002-compute-model.md's storage rule applies
// with nothing to lose. The value read is the value written, at the same width.
//
// It exists because an f16 KV cache was readable and not writable. A decode
// step's sequence is prefill, scatter one row per step, then attention -- and
// [AttentionDecodeF16] read a narrow cache that no kernel could populate from
// inside the graph, so the saving it argues for was not reachable by a model
// (accel issue 13).
//
// # Why the rows are f16 rather than f32
//
// The state's width is the cache's, and a row is one position of it. Narrowing
// here rather than in the caller would make this kernel a cast and a scatter,
// and specs/010-kernel-corpus.md registers the cast separately: a caller whose
// rows are f32 runs [CastF32ToF16] first, which is one operator rather than a
// second scatter variant per input width.
//
// An out-of-range id writes nothing, for [ScatterRows]'s reason.
//
//accel:kernel workgroup=64
func ScatterRowsF16(t accel.Thread, p RowParams, rows []accel.Float16, ids []uint32,
	state []accel.Float16) {

	i := t.GlobalID().X
	if i < p.Rows*p.Width {
		r := i / p.Width
		c := i % p.Width
		id := ids[r]
		if id < p.Capacity {
			state[id*p.Width+c] = rows[i]
		}
	}
}

// RoPEParams carries the rotation's runtime inputs.
type RoPEParams struct {
	// Rows is how many positions, Width the model dimension.
	Rows  uint32
	Width uint32

	// RotaryDim is how much of each row rotates; the tail passes through. It is
	// declared rather than assumed equal to Width because models differ, and a
	// rotation applied to the wrong span is a plausible tensor.
	RotaryDim uint32

	// Base is the frequency base, conventionally 10000.
	//
	// It stays a scalar where the position does not, and the line between them
	// is specs/043-per-row-values.md's: a value every row of a dispatch shares
	// is a uniform, and a value that differs per row is device data. The base
	// is a property of the model; the position is a property of the sequence.
	Base float32
}

// RoPE rotates each row's leading pairs by an angle that depends on the
// position and the pair index.
//
//	θ = (pos + offset) · base^(-2i/rotaryDim)
//	(x₂ᵢ, x₂ᵢ₊₁) ← (x₂ᵢcos θ − x₂ᵢ₊₁sin θ, x₂ᵢsin θ + x₂ᵢ₊₁cos θ)
//
// One invocation per *pair* rather than per element, because a rotation reads
// both halves and writes both: splitting it across two invocations would make
// each read a value the other is writing, which is a race a barrier cannot fix
// since they may be in different workgroups.
//
//accel:kernel workgroup=64
func RoPE(t accel.Thread, p RoPEParams, positions []uint32, inout []float32) {
	i := t.GlobalID().X
	pairs := p.RotaryDim / 2
	if i < p.Rows*pairs {
		r := i / pairs
		k := i % pairs
		// The row's own position, read from device data rather than derived
		// from the row index plus a shared offset. specs/043-per-row-values.md:
		// in a batched decode the row index is the *slot*, so a shared offset
		// rotates exactly one member of the batch at its own cache length, and
		// the output stays finite and fluent while long-range coherence rots.
		pos := float32(positions[r])

		// base^(-2k/rotaryDim), written as an exponential of a logarithm
		// because the subset has no power operator and the two are the same
		// function.
		exponent := float32(-2) * float32(k) / float32(p.RotaryDim)
		freq := kmath.Exp(exponent * kmath.Log(p.Base))
		theta := pos * freq
		c := kmath.Cos(theta)
		s := kmath.Sin(theta)

		lo := r*p.Width + 2*k
		hi := lo + 1
		x := inout[lo]
		y := inout[hi]
		inout[lo] = x*c - y*s
		inout[hi] = x*s + y*c
	}
}

// BiasParams carries a signed offset.
//
// It exists because `int32` is a legal uniform field type and nothing proved
// it. specs/014-kernel-uniforms.md admits three scalars and the emitter maps
// each to an [accel.UniformWriter] method, but every uniform in this corpus was
// float32 or uint32 — so `UniformWriter.I32`, the emitter's `int32` case, and
// the MSL spelling of a signed uniform were all declared and never executed. A
// kernel author writing this struct today would be the first caller of that
// path, which is not where a caller should find out.
//
// Signed rather than a second unsigned field for the same reason the value is
// negative in the tests: `uint32(-1)` and `int32(-1)` are the same four bytes,
// so a codec that wrote the wrong method would still round-trip. What
// distinguishes them is arithmetic on the far side.
type BiasParams struct{ Offset int32 }

// ElemBias applies a signed offset, and branches on its sign.
//
// # Why it compares rather than only adds
//
// The first version of this added the offset and nothing else, on the reasoning
// that "adding a negative offset to a positive input" would expose a uniform
// read as unsigned. That reasoning is wrong, and reinstating the bug is what
// said so: two's-complement addition is sign-agnostic. int32(-3) and
// uint32(4294967293) are the same four bytes and adding either produces the
// same thirty-two bits, so a Metal side declaring the field `uint` passed the
// differential unchanged.
//
// Signedness is only observable where the operation itself differs: comparison,
// division, modulo, and the right shift. This compares against zero, which is
// the cheapest of them and the one a reader recognises as being about sign.
//
//accel:kernel workgroup=64
func ElemBias(t accel.Thread, p BiasParams, in []int32, out []int32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		// Read as unsigned, a negative offset is a very large positive number,
		// so this branch is never taken and every element takes the other one.
		if p.Offset < 0 {
			out[i] = in[i] - p.Offset
		} else {
			out[i] = in[i] + p.Offset
		}
	}
}
