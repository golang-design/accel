// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"errors"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/kernelabi"
)

// uniformLoadKernel reads a routing table declared uniform and writes out.
//
// Hand-built rather than generated, because the property under test is the
// graph's, not the compiler's: the record carries kernelabi.UniformLoad on
// the table, which is what //accel:uniform generates.
func uniformLoadKernel() *kernelabi.Kernel {
	return &kernelabi.Kernel{
		Name: "Routed", WorkgroupSize: accel.ID3{X: 8, Y: 1, Z: 1},
		Digest: "test:Routed", Generator: kernelabi.Version, OrderIndependent: true,
		Bindings: []kernelabi.Binding{
			{Name: "table", DType: kernelabi.U32, Access: kernelabi.Read | kernelabi.UniformLoad},
			{Name: "out", DType: kernelabi.U32, Access: kernelabi.Write},
		},
		Flat: func(t accel.Thread, args kernelabi.Args) {
			table := kernelabi.Slice[uint32](args, 0)
			out := kernelabi.Slice[uint32](args, 1)
			i := t.GlobalID().X
			if i < uint32(len(out)) {
				out[i] = table[0]
			}
		},
	}
}

// A dispatch that writes bytes one of its bindings declared uniform is refused.
//
// specs/063-uniform-loads.md. The compiler accepted the kernel's barriers on
// the author's promise that no invocation of the dispatch writes the table; a
// write binding of the same dispatch over the same bytes breaks it, and the
// graph is the only thing that can see both sides. Concrete resources are
// checked when the dispatch is recorded, and a slot when it is bound, because
// that is when each is known (there by V24, which refuses any slot over any
// writer). A non-overlapping pair passes, so the check is about the bytes and
// not the buffer.
func TestADispatchCannotWriteWhatItDeclaredUniform(t *testing.T) {
	d := openDevice(t)
	p, err := d.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: uniformLoadKernel(), Label: "routed"})
	if err != nil {
		t.Fatalf("pipeline: %v", err)
	}
	defer p.Close()
	buf := newBuffer(t, d, "shared", 64*4, accel.BufferStorage)
	view := func(off, count int) accel.BufferView {
		v, err := buf.ViewAs(accel.U32, off, count)
		if err != nil {
			t.Fatalf("view: %v", err)
		}
		return v
	}

	t.Run("concrete, overlapping", func(t *testing.T) {
		r := d.NewRecorder()
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: view(0, 32)},
			{Index: 1, Buffer: view(16, 32)},
		}, nil, accel.WorkgroupCount{X: 4})
		_, err := r.Build()
		if !errors.Is(err, accel.ErrUniformLoadAliased) {
			t.Fatalf("a write over the declared-uniform bytes built: %v", err)
		}
	})
	t.Run("concrete, disjoint", func(t *testing.T) {
		r := d.NewRecorder()
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Buffer: view(0, 32)},
			{Index: 1, Buffer: view(32, 32)},
		}, nil, accel.WorkgroupCount{X: 4})
		g, err := r.Build()
		if err != nil {
			t.Fatalf("disjoint ranges of one buffer were refused: %v", err)
		}
		defer g.Close()
		if err := d.Queue().Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
	})
	t.Run("through a slot", func(t *testing.T) {
		r := d.NewRecorder()
		table := r.Slot(accel.SlotDescriptor{Name: "table", DType: accel.U32, MinCount: 32, Access: accel.AccessRead})
		r.Dispatch(p, []accel.Binding{
			{Index: 0, Slot: table},
			{Index: 1, Buffer: view(0, 16)},
		}, nil, accel.WorkgroupCount{X: 2})
		g, err := r.Build()
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		defer g.Close()
		// A slot overlapping any writer is V24's refusal (ErrRebindOverlap),
		// which is stricter than the uniform rule and therefore covers it:
		// the graph-wide check cannot tell a uniform reader from any other.
		err = g.Bind(accel.SlotBinding{Slot: table, Buffer: view(0, 32)})
		if !errors.Is(err, accel.ErrRebindOverlap) {
			t.Fatalf("binding the table over the dispatch's own write bound: %v", err)
		}
		if err := g.Bind(accel.SlotBinding{Slot: table, Buffer: view(32, 32)}); err != nil {
			t.Fatalf("a disjoint table was refused: %v", err)
		}
		if err := d.Queue().Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
	})
}
