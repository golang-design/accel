// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// An attribute an operator records must change the graph's identity.
//
// # The class this exists for, and it is not the same as reaches_test's
//
// [tensor.Builder.Identity] covers every operator's operands, kernel, shapes,
// ports and scalars. What it could not see was a value an operator *recorded*
// and passed to its kernel through the uniform closure: an eps, a rotary width,
// a top-k, a page-block size. Those reach the kernel and change what it
// computes, and two graphs differing only in one of them hashed alike.
//
// That is not cosmetic, because [tensor.PlanCache.Compile] records the graph to
// learn its identity and then, on a hit, **discards it and returns the cached
// plan**. So a serving process that compiled a top-k of 40 for one request and
// asked for a top-k of 5 for the next got the first plan back and a token drawn
// from forty candidates. Nothing fails, nothing logs, and the token is
// plausible.
//
// reaches_test asks whether a field reaches a *kernel*. This asks whether it
// reaches the *key*, which is a different question with the same shape: a field
// can do the first and not the second, and every row below did exactly that
// before `node.attrs` existed.
func TestARecordedAttributeReachesTheIdentity(t *testing.T) {
	rt := newRuntime(t)

	// Each row builds the same graph twice, changing one recorded attribute and
	// nothing else. Ports and scalars are declared identically in both, so a
	// digest that moves moves because the *operator* carried the value.
	rows := map[string]func(b *tensor.Builder, alt bool){
		"RMSNorm.eps": func(b *tensor.Builder, alt bool) {
			x := tensor.Input(b, tensor.ValueDesc{
				Name: "x", DType: accel.F32, Shape: tensor.Shape{2, 8},
			})
			g := tensor.Weight(b, tensor.ValueDesc{
				Name: "g", DType: accel.F32, Shape: tensor.Shape{8},
			})
			eps := float32(1e-5)
			if alt {
				eps = 1e-2
			}
			tensor.Output(b, "o", tensor.RMSNorm(b, x, g, eps))
		},
		"RoPE.rotaryDim": func(b *tensor.Builder, alt bool) {
			tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarF32})
			x := tensor.Input(b, tensor.ValueDesc{
				Name: "x", DType: accel.F32, Shape: tensor.Shape{2, 8},
			})
			pos := tensor.Input(b, tensor.ValueDesc{
				Name: "pos", DType: accel.U32, Shape: tensor.Shape{2},
			})
			dim := 4
			if alt {
				dim = 8
			}
			tensor.Output(b, "o", tensor.RoPE(b, x, dim, "base", pos))
		},
		// The page table is present in *both* halves, which is what makes this
		// row real. reaches_test's Block row compares a graph with no page
		// table against one with a table and a block size, so it moved on the
		// table's presence whatever Block did.
		"AttentionOptions.Block": func(b *tensor.Builder, alt bool) {
			tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
			q := tensor.Input(b, tensor.ValueDesc{
				Name: "q", DType: accel.F32, Shape: tensor.Shape{4, 8},
			})
			k := tensor.NewState(b, tensor.StateDesc{
				Name: "k", DType: accel.F32, Shape: tensor.Shape{4, 2, 8},
			})
			v := tensor.NewState(b, tensor.StateDesc{
				Name: "v", DType: accel.F32, Shape: tensor.Shape{4, 2, 8},
			})
			lens := tensor.Input(b, tensor.ValueDesc{
				Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
			})
			pages := tensor.Input(b, tensor.ValueDesc{
				Name: "pages", DType: accel.U32, Shape: tensor.Shape{2},
			})
			block := 2
			if alt {
				block = 4
			}
			tensor.Output(b, "o", tensor.Attention(b, q, k, v, tensor.AttentionOptions{
				Lengths: lens, ScaleName: "scale", Pages: pages, Block: block,
			}))
		},
	}

	for name, build := range rows {
		t.Run(name, func(t *testing.T) {
			digest := func(alt bool) tensor.Identity {
				b := rt.NewBuilder("attrs")
				build(b, alt)
				if err := b.Err(); err != nil {
					t.Fatalf("the graph does not build: %v", err)
				}
				return b.Identity()
			}
			if digest(false) == digest(true) {
				t.Errorf("changing %s left the identity unchanged, so a PlanCache keyed "+
					"on it hands one configuration's plan to the other and the result "+
					"is a plausible wrong answer", name)
			}
		})
	}
}

// A plan cache asked for two rotary widths returns two plans.
//
// The end-to-end statement of what the digest is for, and the one that fails on
// *values* rather than on a hash. The two graphs differ only in how much of each
// row rotates, so a cache that could not tell them apart hands the first plan to
// the second caller and the rotation silently covers the wrong span -- which
// specs/043-per-row-values.md describes for the position: output that stays
// finite and fluent while long-range coherence degrades.
func TestAPlanCacheTellsTwoRotaryWidthsApart(t *testing.T) {
	rt := newRuntime(t)
	cache := tensor.NewPlanCache(rt)
	defer cache.Close()

	const width = 8
	xs := make([]float32, width)
	for i := range xs {
		xs[i] = 1
	}

	rotate := func(rotaryDim int) []float32 {
		plan, err := cache.Compile(func(b *tensor.Builder) {
			tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarF32})
			x := tensor.Input(b, tensor.ValueDesc{
				Name: "x", DType: accel.F32, Shape: tensor.Shape{1, width},
			})
			pos := tensor.Input(b, tensor.ValueDesc{
				Name: "pos", DType: accel.U32, Shape: tensor.Shape{1},
			})
			tensor.Output(b, "o", tensor.RoPE(b, x, rotaryDim, "base", pos))
		}, tensor.CompileOptions{Label: "rope"})
		if err != nil {
			t.Fatalf("compile rotaryDim=%d: %v", rotaryDim, err)
		}
		d := rt.Device()
		out := f32Buffer(t, d, "o", make([]float32, width))
		f := plan.Submit(d.Queue(), tensor.Bindings{
			Buffers: map[string]accel.BufferView{
				"x": f32Buffer(t, d, "x", xs), "pos": u32Buffer(t, d, "pos", []uint32{1}),
				"o": out,
			},
			Scalars: map[string]tensor.ScalarValue{"base": tensor.F32(10000)},
		})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit rotaryDim=%d: %v", rotaryDim, err)
		}
		got := make([]float32, width)
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}
		return got
	}

	// The wider one first, so a cache that cannot tell them apart answers the
	// second call with a plan that rotates more of the row than was asked for.
	wide := rotate(width)
	narrow := rotate(2)

	// Everything past the narrow span must be untouched, and the wide run must
	// have touched it. Two assertions rather than one, because "the outputs
	// differ" would also pass if both plans were wrong in different ways.
	for i := 2; i < width; i++ {
		if narrow[i] != xs[i] {
			t.Fatalf("a rotaryDim of 2 changed element %d: %v", i, narrow[i])
		}
		if wide[i] == xs[i] {
			t.Fatalf("a rotaryDim of %d left element %d alone: %v", width, i, wide[i])
		}
	}
	if cache.Len() != 2 {
		t.Errorf("two rotary widths produced %d cached plans; the cache cannot tell "+
			"them apart, so the second caller ran the first caller's rotation",
			cache.Len())
	}
}
