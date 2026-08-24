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

// A paged cache produces what a contiguous one does over the same logical
// positions, including when the blocks are out of order.
//
// # Why this matters more than it looks
//
// A consumer reported that the block pool was unreachable: `pagetable` is
// internal *because* no exported operator accepted a page table, so the only
// cache anyone could build was one contiguous State per session, sized for the
// longest sequence the server would ever accept. On a 36-layer model that is
// 9.66 GB per sequence at 32k context — reserved whether or not anything is
// that long, and larger than the int8 weights of the model being served.
//
// The kernels for this existed and had since specs/030-paged-kv.md. Nothing in
// this package referenced them.
//
// # Why paging is not a second kind of cache
//
// A State addressed through a page table is the same State; what differs is
// the binding. Introducing a PagedState beside State would make a caller ask
// which to build, and the answer would depend on a scheduler they have not
// written — which is the non-orthogonal growth specs/043-per-row-values.md
// exists to avoid.
func TestAPagedCacheMatchesAContiguousOne(t *testing.T) {
	const qHeads, kvHeads, headDim = 2, 1, 8
	const block, blocks = 2, 3
	const kvLen = block * blocks

	rt := newRuntime(t)
	d := rt.Device()
	scale := float32(1 / math.Sqrt(headDim))

	qs := make([]float32, qHeads*headDim)
	for i := range qs {
		qs[i] = float32(math.Sin(float64(i) * 0.37))
	}
	// The logical cache: kvLen positions in order.
	logical := make([]float32, kvLen*kvHeads*headDim)
	for i := range logical {
		logical[i] = float32(math.Cos(float64(i) * 0.19))
	}

	// The physical cache holds the same blocks somewhere else, and the page
	// table says where. Deliberately out of order and not starting at zero,
	// because an implementation that ignored the table would still produce a
	// plausible tensor from a contiguous read.
	pages := []uint32{4, 1, 3}
	physical := make([]float32, 8*block*kvHeads*headDim)
	per := block * kvHeads * headDim
	for b := range blocks {
		copy(physical[int(pages[b])*per:], logical[b*per:(b+1)*per])
	}

	run := func(t *testing.T, paged bool) []float32 {
		t.Helper()
		bl := rt.NewBuilder("paged")
		tensor.Scalar(bl, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
		q := tensor.Input(bl, tensor.ValueDesc{
			Name: "q", DType: accel.F32, Shape: tensor.Shape{qHeads, headDim},
		})
		lengths := tensor.Input(bl, tensor.ValueDesc{
			Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
		})
		capacity := kvLen
		opts := tensor.AttentionOptions{Lengths: lengths, ScaleName: "scale"}
		if paged {
			capacity = 8 * block
			opts.Pages = tensor.Input(bl, tensor.ValueDesc{
				Name: "pages", DType: accel.U32, Shape: tensor.Shape{blocks},
			})
			opts.Block = block
		}
		kc := tensor.NewState(bl, tensor.StateDesc{
			Name: "k", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
		})
		vc := tensor.NewState(bl, tensor.StateDesc{
			Name: "v", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
		})
		tensor.Output(bl, "out", tensor.Attention(bl, q, kc, vc, opts))

		plan, err := bl.Compile(rt, tensor.CompileOptions{Label: "paged"})
		if err != nil {
			t.Fatalf("compile (paged=%v): %v", paged, err)
		}
		defer plan.Close()

		cache := logical
		if paged {
			cache = physical
		}
		out := f32Buffer(t, d, "out", make([]float32, qHeads*headDim))
		bufs := map[string]accel.BufferView{
			"q": f32Buffer(t, d, "q", qs),
			"k": f32Buffer(t, d, "k", cache), "v": f32Buffer(t, d, "v", cache),
			"out": out,
			"len": u32Buffer(t, d, "len", []uint32{kvLen}),
		}
		if paged {
			bufs["pages"] = u32Buffer(t, d, "pages", pages)
		}
		f := plan.Submit(d.Queue(), tensor.Bindings{
			Buffers: bufs,
			Scalars: map[string]tensor.ScalarValue{"scale": tensor.F32(scale)},
		})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit (paged=%v): %v", paged, err)
		}
		got := make([]float32, qHeads*headDim)
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}
		return got
	}

	contiguous, viaPages := run(t, false), run(t, true)
	for i := range contiguous {
		if math.Abs(float64(contiguous[i]-viaPages[i])) > 1e-6 {
			t.Fatalf("element %d is %v contiguous and %v through the page table; the two "+
				"address the same logical positions", i, contiguous[i], viaPages[i])
		}
	}
	// And it drew on the whole cache, so the two agreeing is not two zeros.
	var live bool
	for _, v := range contiguous {
		if v != 0 {
			live = true
		}
	}
	if !live {
		t.Fatal("the contiguous run produced zeros, so agreement proves nothing")
	}
}

// The refusals a page table owns.
func TestPagedAttentionRefusals(t *testing.T) {
	const qHeads, kvHeads, headDim, capacity = 2, 1, 8, 8
	rt := newRuntime(t)

	build := func(mut func(*tensor.Builder, *tensor.AttentionOptions)) error {
		b := rt.NewBuilder("refusal")
		tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
		q := tensor.Input(b, tensor.ValueDesc{
			Name: "q", DType: accel.F32, Shape: tensor.Shape{qHeads, headDim},
		})
		opts := tensor.AttentionOptions{
			Lengths: tensor.Input(b, tensor.ValueDesc{
				Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
			}),
			ScaleName: "scale",
		}
		mut(b, &opts)
		kc := tensor.NewState(b, tensor.StateDesc{
			Name: "k", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
		})
		vc := tensor.NewState(b, tensor.StateDesc{
			Name: "v", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
		})
		tensor.Attention(b, q, kc, vc, opts)
		return b.Err()
	}

	t.Run("a page table with no block size", func(t *testing.T) {
		err := build(func(b *tensor.Builder, o *tensor.AttentionOptions) {
			o.Pages = tensor.Input(b, tensor.ValueDesc{
				Name: "p", DType: accel.U32, Shape: tensor.Shape{4},
			})
		})
		if err == nil || !strings.Contains(err.Error(), "how many positions one holds") {
			t.Errorf("a page table without Block gave %v", err)
		}
	})

	t.Run("a page table of the wrong dtype", func(t *testing.T) {
		err := build(func(b *tensor.Builder, o *tensor.AttentionOptions) {
			o.Pages = tensor.Input(b, tensor.ValueDesc{
				Name: "p", DType: accel.F32, Shape: tensor.Shape{4},
			})
			o.Block = 2
		})
		if err == nil || !strings.Contains(err.Error(), "a page table is u32") {
			t.Errorf("an f32 page table gave %v", err)
		}
	})
}
