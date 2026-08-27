// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

func TestBlockPool(t *testing.T) {
	p, err := tensor.NewBlockPool(4, 8) // 4 blocks of 8 positions
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	if p.BlockSize() != 8 || p.Available() != 4 {
		t.Fatalf("a fresh pool reports block %d and %d available", p.BlockSize(), p.Available())
	}

	// Growing takes only what it needs, and rounds up to whole blocks.
	a, err := p.Grow(nil, 9) // two blocks
	if err != nil {
		t.Fatalf("grow: %v", err)
	}
	if len(a) != 2 || p.Available() != 2 {
		t.Fatalf("9 positions took %d blocks and left %d", len(a), p.Available())
	}
	if p.Positions(a) != 16 {
		t.Errorf("two blocks of eight address %d positions", p.Positions(a))
	}

	// Growing again keeps the earlier pages where they are: a decode step's
	// cache must not move under it.
	before := append([]uint32(nil), a...)
	a, err = p.Grow(a, 17) // a third block
	if err != nil {
		t.Fatalf("grow: %v", err)
	}
	if len(a) != 3 {
		t.Fatalf("17 positions took %d blocks", len(a))
	}
	for i := range before {
		if a[i] != before[i] {
			t.Fatalf("page %d moved from %d to %d when the sequence grew",
				i, before[i], a[i])
		}
	}

	// Growing to something the existing pages already hold takes nothing.
	same, err := p.Grow(a, 20)
	if err != nil {
		t.Fatalf("grow: %v", err)
	}
	if len(same) != 3 || p.Available() != 1 {
		t.Fatalf("a grow inside the existing pages took a block")
	}

	// And the pool refuses rather than evicting.
	if _, err := p.Grow(a, 100); err == nil {
		t.Fatal("the pool handed out blocks it does not have")
	} else if !strings.Contains(err.Error(), "does not evict") {
		t.Errorf("the refusal should say why: %v", err)
	}

	if err := p.Free(a); err != nil {
		t.Fatalf("free: %v", err)
	}
	if p.Available() != 4 {
		t.Errorf("after freeing three blocks the pool has %d of 4", p.Available())
	}

	// Freeing twice is refused, because it would hand one block to two
	// sequences -- and the symptom is one conversation reading another's.
	if err := p.Free(a); err == nil {
		t.Error("a block was freed twice")
	}

	if _, err := tensor.NewBlockPool(0, 8); err == nil {
		t.Error("a pool of no blocks was accepted")
	}
	if _, err := tensor.NewBlockPool(4, 0); err == nil {
		t.Error("a block of no positions was accepted")
	}
	if _, err := p.Grow(nil, -1); err == nil {
		t.Error("a negative length was accepted")
	}
}

// A pool-allocated page table drives a paged decode, and two sequences from one
// pool never see each other's tokens.
//
// specs/030-paged-kv.md §4.1 says the pool was built as `tensor.BlockPool`, and
// until 2026-08-27 it was not: it lived in tensor/internal/pagetable, whose doc
// said it stayed internal "until an operator accepts a page table". Attention
// takes one, so the condition was met and unnoticed.
//
// This is the test that makes the export mean something. Exporting a type only
// proves a caller can name it; what a caller needs is that its output binds, so
// the pool's pages go straight into AttentionOptions.Pages here rather than
// being asserted as a slice.
//
// The disjointness half is the property the pool exists for. Two sequences
// holding one block is one sequence reading another's tokens, which reads as a
// model answering from the wrong conversation rather than as a fault.
func TestAPoolAllocatedPageTableDrivesAPagedDecode(t *testing.T) {
	const qHeads, kvHeads, headDim = 2, 1, 8
	const block, blocks = 2, 6
	const width = kvHeads * headDim

	pool, err := tensor.NewBlockPool(blocks, block)
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	if got := pool.BlockSize(); got != block {
		t.Fatalf("BlockSize = %d, want %d", got, block)
	}

	// Two sequences, grown from the same pool.
	a, err := pool.Grow(nil, 4)
	if err != nil {
		t.Fatalf("grow a: %v", err)
	}
	b, err := pool.Grow(nil, 4)
	if err != nil {
		t.Fatalf("grow b: %v", err)
	}

	// Disjoint, which is what stops one sequence reading the other's tokens.
	seen := map[uint32]bool{}
	for _, p := range a {
		seen[p] = true
	}
	for _, p := range b {
		if seen[p] {
			t.Fatalf("block %d was handed to both sequences; page tables %v and %v", p, a, b)
		}
	}
	if pool.Positions(a) < 4 {
		t.Fatalf("Positions(%v) = %d, want at least the 4 asked for", a, pool.Positions(a))
	}
	if want := blocks - len(a) - len(b); pool.Available() != want {
		t.Fatalf("Available = %d, want %d", pool.Available(), want)
	}

	// The pages bind, and the answer is the one the same positions give when
	// the page table is written by hand. That is the whole claim: a pool's
	// output is a page table Attention accepts.
	rt := newRuntime(t)
	d := rt.Device()
	scale := float32(1 / math.Sqrt(headDim))

	qs := make([]float32, qHeads*headDim)
	for i := range qs {
		qs[i] = float32(math.Sin(float64(i) * 0.41))
	}
	phys := make([]float32, blocks*block*width)
	for i := range phys {
		phys[i] = float32(math.Cos(float64(i) * 0.23))
	}

	run := func(label string, pages []uint32) []float32 {
		t.Helper()
		bl := rt.NewBuilder(label)
		tensor.Scalar(bl, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
		q := tensor.Input(bl, tensor.ValueDesc{
			Name: "q", DType: accel.F32, Shape: tensor.Shape{qHeads, headDim}})
		pg := tensor.Input(bl, tensor.ValueDesc{
			Name: "pages", DType: accel.U32, Shape: tensor.Shape{len(pages)}})
		ln := tensor.Input(bl, tensor.ValueDesc{
			Name: "len", DType: accel.U32, Shape: tensor.Shape{1}})
		kc := tensor.NewState(bl, tensor.StateDesc{
			Name: "kc", DType: accel.F32,
			Shape: tensor.Shape{blocks * block, kvHeads, headDim}})
		vc := tensor.NewState(bl, tensor.StateDesc{
			Name: "vc", DType: accel.F32,
			Shape: tensor.Shape{blocks * block, kvHeads, headDim}})
		tensor.Output(bl, "out", tensor.Attention(bl, q, kc, vc, tensor.AttentionOptions{
			Lengths: ln, Pages: pg, Block: block, ScaleName: "scale",
		}))

		plan, err := bl.Compile(rt, tensor.CompileOptions{Label: label})
		if err != nil {
			t.Fatalf("compile %s: %v", label, err)
		}
		defer plan.Close()

		out := f32Buffer(t, d, "out", make([]float32, qHeads*headDim))
		f := plan.Submit(d.Queue(), tensor.Bindings{
			Buffers: map[string]accel.BufferView{
				"q":     f32Buffer(t, d, "q", qs),
				"pages": u32Buffer(t, d, "pages", pages),
				"len":   u32Buffer(t, d, "len", []uint32{4}),
				"kc":    f32Buffer(t, d, "kc", phys),
				"vc":    f32Buffer(t, d, "vc", phys),
				"out":   out,
			},
			Scalars: map[string]tensor.ScalarValue{"scale": tensor.F32(scale)},
		})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit %s: %v", label, err)
		}
		got := make([]float32, qHeads*headDim)
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("readback %s: %v", label, err)
		}
		return got
	}

	fromPool := run("pool", a)
	byHand := run("hand", append([]uint32(nil), a...))
	for i := range fromPool {
		if fromPool[i] != byHand[i] {
			t.Fatalf("element %d: pool pages %v, hand-written %v", i, fromPool[i], byHand[i])
		}
	}
	nonzero := false
	for _, v := range fromPool {
		if v != 0 {
			nonzero = true
		}
	}
	if !nonzero {
		t.Fatal("every output is zero, so the comparison says nothing")
	}

	// Freeing returns them, and freeing twice is refused rather than handing one
	// block to two sequences later.
	if err := pool.Free(a); err != nil {
		t.Fatalf("free: %v", err)
	}
	if err := pool.Free(a); err == nil {
		t.Fatal("freeing the same pages twice was accepted")
	}
}
