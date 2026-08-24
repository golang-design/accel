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
