// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"fmt"

	"golang.design/x/accel"
)

// Materializing a view into packed storage.
//
// The corpus kernels index their operands contiguously, so a view whose
// elements are not adjacent cannot be read as it stands. Two ways out and this
// package takes both, in different places:
//
//   - An elementwise operator materializes a broadcast operand, because
//     specs/007-tensor-layer.md gives Add and Mul NumPy broadcasting and a
//     caller who wrote it expects it to work.
//   - MatMul does not, because that spec explicitly requires unit stride on the
//     contracted axes "without silently materializing either one": a copy of a
//     weight matrix is a cost large enough that a caller must ask for it.
//
// Either way the copy is **reported** in [Plan.Selections]. A materialization
// nobody can see is a performance cliff nobody can explain, which is the same
// argument that makes kernel choice a report rather than an implementation
// detail.

// repeats describes a broadcast this package can materialize with copies alone.
//
// The shape it admits is the one that matters: a contiguous block repeated a
// whole number of times. A gain vector broadcast across rows, a bias across a
// batch. Anything else -- an interior axis expanding, a strided operand -- would
// need a gather kernel with the strides in a uniform block, which the corpus
// does not have and which specs/025-tensor-operators.md would have to add.
type repeats struct {
	block int // elements in one contiguous run
	times int // how many times it repeats
}

// broadcastRepeats reports how to build want from t with copies, if it can be.
func broadcastRepeats(t *Tensor, want Shape) (repeats, bool) {
	if !t.contiguousLayout() {
		return repeats{}, false
	}
	block := t.shape.Elements()
	total := want.Elements()
	if block == 0 || total%block != 0 {
		return repeats{}, false
	}
	// The operand must be a *suffix* of the result: its axes align to the right
	// and each is either equal or absent. An operand with a size-one axis in the
	// middle would repeat with a stride and is not a run.
	off := len(want) - len(t.shape)
	if off < 0 {
		return repeats{}, false
	}
	for i := range t.shape {
		if t.shape[i] != want[off+i] {
			return repeats{}, false
		}
	}
	return repeats{block: block, times: total / block}, true
}

// materialize records the copies that pack an operand into a transient.
func (p *Plan) materialize(r *accel.Recorder, op string, t *Tensor, want Shape,
	src accel.Binding, label string) (accel.Binding, error) {

	rep, ok := broadcastRepeats(t, want)
	if !ok {
		return accel.Binding{}, fmt.Errorf("an operand of shape %v against a result of %v "+
			"is not a contiguous run repeated a whole number of times, which is the only "+
			"broadcast this version materializes; make it contiguous in the shape you want",
			t.shape, want)
	}

	dst := r.Transient(accel.BufferDescriptor{
		DType: t.dtype, Count: want.Elements(),
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
		Label: label,
	})
	// A BufferView is a range rather than a resource, so each repeat is the
	// same transient at a different offset. Written out rather than expressed
	// as one strided copy, because a strided copy is a texture operation here
	// and this is a buffer.
	for i := range rep.times {
		slice := accel.BufferView{
			Buffer: dst.Buffer, DType: dst.DType,
			Offset: dst.Offset + i*rep.block, Count: rep.block,
		}
		if src.Buffer.Buffer != nil {
			r.CopyBuffer(slice, src.Buffer)
			continue
		}
		r.CopyFromSlot(slice, src.Slot, 0, rep.block)
	}
	p.selections = append(p.selections, KernelSelection{
		Op:     op,
		Kernel: "copy",
		Reason: fmt.Sprintf("materializing a %v operand into %v: %d copies of a %d-element "+
			"run, because the elementwise kernels index their operands together",
			t.shape, want, rep.times, rep.block),
	})
	return accel.Binding{Buffer: dst}, nil
}
