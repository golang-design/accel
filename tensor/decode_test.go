// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// A decode step: append this token's key and value to the cache, then attend
// over everything the cache holds.
//
// This is what specs/026-tensor-decode.md exists for, and the properties worth
// asserting are the ones a single submission cannot show. The plan is submitted
// once per token with the same buffers, a different current length, and nothing
// rebuilt -- which is the claim that a decode plan is "reusable for every token
// up to KV capacity".
//
// The write and the read of the cache are ordered by the graph rather than by
// this test: they declare overlapping byte ranges, so the planner infers a
// read-after-write edge and emits a barrier. That was checked before M7 began,
// and this is the first thing that depends on it.
func TestADecodeStepAppendsAndAttends(t *testing.T) {
	const (
		qHeads   = 4
		kvHeads  = 2
		headDim  = 8
		capacity = 8
	)
	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("decode")

	tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})

	q := tensor.Input(b, tensor.ValueDesc{
		Name: "q", DType: accel.F32, Shape: tensor.Shape{qHeads, headDim},
	})
	newK := tensor.Input(b, tensor.ValueDesc{
		Name: "newk", DType: accel.F32, Shape: tensor.Shape{1, kvHeads * headDim},
	})
	newV := tensor.Input(b, tensor.ValueDesc{
		Name: "newv", DType: accel.F32, Shape: tensor.Shape{1, kvHeads * headDim},
	})
	slot := tensor.Input(b, tensor.ValueDesc{
		Name: "slot", DType: accel.U32, Shape: tensor.Shape{1},
	})
	// One sequence, so one length. specs/043-per-row-values.md: the same path a
	// batch takes, with one row.
	lengths := tensor.Input(b, tensor.ValueDesc{
		Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
	})

	kc := tensor.NewState(b, tensor.StateDesc{
		Name: "kcache", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
	})
	vc := tensor.NewState(b, tensor.StateDesc{
		Name: "vcache", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
	})

	// The next version of each cache, which is what Attention reads. Reading
	// the version *before* the scatter would be a different node and would miss
	// this token, which is what the versions are for.
	kc2 := tensor.ScatterRows(b, kc, newK, slot)
	vc2 := tensor.ScatterRows(b, vc, newV, slot)
	tensor.Output(b, "out", tensor.Attention(b, q, kc2, vc2, tensor.AttentionOptions{
		Lengths: lengths, ScaleName: "scale",
	}))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "decode"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	// The caches are caller-owned and survive between submissions, which is the
	// point: they are bound once and never re-uploaded.
	kBuf := f32Buffer(t, d, "kcache", make([]float32, capacity*kvHeads*headDim))
	vBuf := f32Buffer(t, d, "vcache", make([]float32, capacity*kvHeads*headDim))
	out := f32Buffer(t, d, "out", make([]float32, qHeads*headDim))

	// A host-side model of the same computation, stepped alongside.
	var keys, vals [][]float32
	scale := float32(1 / math.Sqrt(headDim))

	for step := range 4 {
		qs := make([]float32, qHeads*headDim)
		ks := make([]float32, kvHeads*headDim)
		vs := make([]float32, kvHeads*headDim)
		for i := range qs {
			qs[i] = float32(math.Sin(float64(step*13 + i)))
		}
		for i := range ks {
			ks[i] = float32(math.Cos(float64(step*7 + i)))
			vs[i] = float32(step) + float32(i)/8
		}
		keys = append(keys, ks)
		vals = append(vals, vs)

		f := plan.Submit(d.Queue(), tensor.Bindings{
			Buffers: map[string]accel.BufferView{
				"q":      f32Buffer(t, d, "q", qs),
				"newk":   f32Buffer(t, d, "newk", ks),
				"newv":   f32Buffer(t, d, "newv", vs),
				"slot":   u32Buffer(t, d, "slot", []uint32{uint32(step)}),
				"kcache": kBuf, "vcache": vBuf, "out": out,
				"len": u32Buffer(t, d, "len", []uint32{uint32(step + 1)}),
			},
			Scalars: map[string]tensor.ScalarValue{
				"scale": tensor.F32(scale),
			},
		})
		if err := f.Wait(); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}

		got := make([]float32, qHeads*headDim)
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}

		// The reference, in f64: score every cached position, softmax, combine.
		group := qHeads / kvHeads
		for h := range qHeads {
			kvHead := h / group
			scores := make([]float64, len(keys))
			maxScore := math.Inf(-1)
			for pos := range keys {
				var acc float64
				for i := range headDim {
					acc += float64(qs[h*headDim+i]) *
						float64(keys[pos][kvHead*headDim+i])
				}
				scores[pos] = acc * float64(scale)
				maxScore = math.Max(maxScore, scores[pos])
			}
			var sum float64
			for i := range scores {
				scores[i] = math.Exp(scores[i] - maxScore)
				sum += scores[i]
			}
			for i := range headDim {
				var acc float64
				for pos := range keys {
					acc += scores[pos] / sum * float64(vals[pos][kvHead*headDim+i])
				}
				if g := float64(got[h*headDim+i]); math.Abs(g-acc) > 1e-4*(1+math.Abs(acc)) {
					t.Fatalf("step %d head %d element %d is %v, want about %v",
						step, h, i, g, acc)
				}
			}
		}
	}

	// The cache holds every token, which is what makes the next step's
	// attention right and is the thing a per-step check cannot see.
	gotK := make([]float32, capacity*kvHeads*headDim)
	if err := d.Queue().ReadBuffer(kBuf.Buffer, 0, gotK); err != nil {
		t.Fatalf("readback: %v", err)
	}
	for pos := range keys {
		for i := range kvHeads * headDim {
			if want := keys[pos][i]; gotK[pos*kvHeads*headDim+i] != want {
				t.Fatalf("cache position %d element %d is %v, want %v",
					pos, i, gotK[pos*kvHeads*headDim+i], want)
			}
		}
	}
	// And nothing past the written positions was touched.
	for pos := len(keys); pos < capacity; pos++ {
		for i := range kvHeads * headDim {
			if v := gotK[pos*kvHeads*headDim+i]; v != 0 {
				t.Fatalf("cache position %d was written and never should have been: %v", pos, v)
			}
		}
	}
}

// The state and attention refusals.
func TestStateAndAttentionRefusals(t *testing.T) {
	rt := newRuntime(t)

	f32 := func(b *tensor.Builder, n string, dims ...int) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: n, DType: accel.F32, Shape: dims})
	}
	u32 := func(b *tensor.Builder, n string, dims ...int) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: n, DType: accel.U32, Shape: dims})
	}
	cache := func(b *tensor.Builder, n string, dims ...int) *tensor.State {
		return tensor.NewState(b, tensor.StateDesc{
			Name: n, DType: accel.F32, Shape: dims,
		})
	}
	scalars := func(b *tensor.Builder) {
		tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
	}
	opts := func(b *tensor.Builder) tensor.AttentionOptions {
		return tensor.AttentionOptions{Lengths: u32(b, "len", 1), ScaleName: "scale"}
	}

	for _, tc := range []struct {
		name  string
		build func(b *tensor.Builder)
		want  string
	}{{
		name: "rows that are not the cache's width",
		build: func(b *tensor.Builder) {
			tensor.ScatterRows(b, cache(b, "c", 8, 4), f32(b, "r", 1, 3), u32(b, "i", 1))
		},
		want: "wide",
	}, {
		name: "ids that do not match the rows",
		build: func(b *tensor.Builder) {
			tensor.ScatterRows(b, cache(b, "c", 8, 4), f32(b, "r", 2, 4), u32(b, "i", 1))
		},
		want: "2 rows and 1 ids",
	}, {
		name: "ids that are not u32",
		build: func(b *tensor.Builder) {
			tensor.ScatterRows(b, cache(b, "c", 8, 4), f32(b, "r", 1, 4), f32(b, "i", 1))
		},
		want: "ids are f32",
	}, {
		name: "query heads that do not group",
		build: func(b *tensor.Builder) {
			scalars(b)
			tensor.Attention(b, f32(b, "q", 3, 8), cache(b, "k", 4, 2, 8),
				cache(b, "v", 4, 2, 8), opts(b))
		},
		want: "share one cache entry",
	}, {
		name: "caches that differ",
		build: func(b *tensor.Builder) {
			scalars(b)
			tensor.Attention(b, f32(b, "q", 4, 8), cache(b, "k", 4, 2, 8),
				cache(b, "v", 4, 2, 4), opts(b))
		},
		want: "both",
	}, {
		name: "a prefill with no base",
		build: func(b *tensor.Builder) {
			scalars(b)
			tensor.Attention(b, f32(b, "q", 2, 4, 8), cache(b, "k", 4, 2, 8),
				cache(b, "v", 4, 2, 8), opts(b))
		},
		want: "a prefill needs BaseName",
	}, {
		name: "a query of the wrong rank",
		build: func(b *tensor.Builder) {
			scalars(b)
			tensor.Attention(b, f32(b, "q", 8), cache(b, "k", 4, 2, 8),
				cache(b, "v", 4, 2, 8), opts(b))
		},
		want: "for a prefill",
	}, {
		name: "no lengths tensor",
		build: func(b *tensor.Builder) {
			tensor.Attention(b, f32(b, "q", 4, 8), cache(b, "k", 4, 2, 8),
				cache(b, "v", 4, 2, 8),
				tensor.AttentionOptions{ScaleName: "scale"})
		},
		want: "Lengths is required",
	}, {
		name: "lengths of the wrong dtype",
		build: func(b *tensor.Builder) {
			tensor.Attention(b, f32(b, "q", 4, 8), cache(b, "k", 4, 2, 8),
				cache(b, "v", 4, 2, 8),
				tensor.AttentionOptions{
					Lengths: f32(b, "len", 1), ScaleName: "scale",
				})
		},
		want: "Lengths is f32",
	}, {
		// accel issue 10: a prefill compiled with Pages, ignored it, and
		// returned a plausible wrong answer. The Block guard lived inside the
		// decode switch, so a prefill reached neither it nor the page table;
		// both checks are above the shape split now, and this is the one that
		// is still a refusal after the paged prefill kernel landed.
		name: "a page table with no block size, on a prefill",
		build: func(b *tensor.Builder) {
			scalars(b)
			tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarU32})
			o := opts(b)
			o.Pages = u32(b, "pages", 2)
			o.BaseName = "base"
			tensor.Attention(b, f32(b, "q", 2, 4, 8), cache(b, "k", 4, 2, 8),
				cache(b, "v", 4, 2, 8), o)
		},
		want: "how many positions one holds is required",
	}, {
		// A prefill read row zero and ignored the rest: a value the caller
		// supplied that reached nothing, found while adding the batch axis
		// (accel issue 12).
		name: "more lengths than sequences",
		build: func(b *tensor.Builder) {
			scalars(b)
			tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarU32})
			o := tensor.AttentionOptions{
				Lengths: u32(b, "len", 4), ScaleName: "scale", BaseName: "base",
			}
			tensor.Attention(b, f32(b, "q", 2, 4, 8), cache(b, "k", 4, 2, 8),
				cache(b, "v", 4, 2, 8), o)
		},
		want: "one length per sequence",
	}, {
		// The three below are the cases that were accepted and computed a
		// plausible wrong answer. A rank-3 or rank-4 q is flattened to rank 2
		// before lowering, and the flattening built a fresh tensor with
		// contiguous strides and offset zero -- so a permuted or sliced q
		// passed the lowering's strided-view check and the kernel read the
		// buffer as recorded, unpermuted and unsliced.
		name: "a permuted prefill q",
		build: func(b *tensor.Builder) {
			scalars(b)
			tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarU32})
			o := opts(b)
			o.BaseName = "base"
			// [qHeads, qSeq, headDim] permuted into the [qSeq, qHeads, headDim]
			// a prefill takes: the shape is right and the bytes are not.
			q := tensor.Permute(b, f32(b, "q", 2, 4, 8), 1, 0, 2)
			tensor.Attention(b, q, cache(b, "k", 8, 2, 8), cache(b, "v", 8, 2, 8), o)
		},
		want: "q is a strided view",
	}, {
		name: "a sliced prefill q",
		build: func(b *tensor.Builder) {
			scalars(b)
			tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarU32})
			o := opts(b)
			o.BaseName = "base"
			// The second and third tokens of three: contiguous strides at a
			// nonzero offset, which the flattening dropped.
			q := tensor.Slice(b, f32(b, "q", 3, 2, 8), 0, 1, 3)
			tensor.Attention(b, q, cache(b, "k", 8, 2, 8), cache(b, "v", 8, 2, 8), o)
		},
		want: "q is a strided view",
	}, {
		name: "a transposed decode q",
		build: func(b *tensor.Builder) {
			scalars(b)
			q := tensor.Transpose(b, f32(b, "q", 8, 2), 0, 1)
			tensor.Attention(b, q, cache(b, "k", 8, 2, 8), cache(b, "v", 8, 2, 8), opts(b))
		},
		want: "q is a strided view",
	}, {
		name: "a batch with no page table",
		build: func(b *tensor.Builder) {
			scalars(b)
			o := tensor.AttentionOptions{Lengths: u32(b, "len", 3), ScaleName: "scale"}
			tensor.Attention(b, f32(b, "q", 3, 1, 4, 8), cache(b, "k", 16, 2, 8),
				cache(b, "v", 16, 2, 8), o)
		},
		want: "a contiguous cache would pad every sequence to the longest",
	}, {
		name: "a batched prefill",
		build: func(b *tensor.Builder) {
			scalars(b)
			o := tensor.AttentionOptions{
				Lengths: u32(b, "len", 3), ScaleName: "scale",
				Pages: u32(b, "pages", 3, 2), Block: 4,
			}
			tensor.Attention(b, f32(b, "q", 3, 5, 4, 8), cache(b, "k", 16, 2, 8),
				cache(b, "v", 16, 2, 8), o)
		},
		want: "a batched *prefill* is specs/040-batch-scheduler.md's",
	}, {
		name: "a page table whose rows are not the batch",
		build: func(b *tensor.Builder) {
			scalars(b)
			o := tensor.AttentionOptions{
				Lengths: u32(b, "len", 3), ScaleName: "scale",
				Pages: u32(b, "pages", 2, 2), Block: 4,
			}
			tensor.Attention(b, f32(b, "q", 3, 1, 4, 8), cache(b, "k", 16, 2, 8),
				cache(b, "v", 16, 2, 8), o)
		},
		want: "one row of block ids per sequence",
	}, {
		name: "a layer that does not exist",
		build: func(b *tensor.Builder) {
			tensor.LayerState(b, cache(b, "c", 2, 8, 4), 5)
		},
		want: "layer 5",
	}, {
		// specs/046 and 010: the query is f32 in every registered kernel, so a
		// narrowed query is refused rather than silently widened.
		name: "a query that is not f32",
		build: func(b *tensor.Builder) {
			scalars(b)
			q := tensor.Input(b, tensor.ValueDesc{
				Name: "q", DType: accel.F16, Shape: tensor.Shape{4, 8}})
			tensor.Attention(b, q, cache(b, "k", 4, 2, 8), cache(b, "v", 4, 2, 8), opts(b))
		},
		want: "registered kernels read an f32",
	}, {
		name: "a cache that is neither f32 nor f16",
		build: func(b *tensor.Builder) {
			scalars(b)
			kc := tensor.NewState(b, tensor.StateDesc{
				Name: "k", DType: accel.U32, Shape: tensor.Shape{4, 2, 8}})
			vc := tensor.NewState(b, tensor.StateDesc{
				Name: "v", DType: accel.U32, Shape: tensor.Shape{4, 2, 8}})
			tensor.Attention(b, f32(b, "q", 4, 8), kc, vc, opts(b))
		},
		want: "registered kernels read f32 or",
	}, {
		name: "extents that are not u32",
		build: func(b *tensor.Builder) {
			scalars(b)
			o := opts(b)
			o.QueryExtents = f32(b, "ext", 2)
			o.Pages = u32(b, "pages", 2, 3)
			o.Block = 4
			tensor.Attention(b, f32(b, "q", 3, 4, 8), cache(b, "k", 16, 2, 8),
				cache(b, "v", 16, 2, 8), o)
		},
		want: "it is a count per sequence",
	}, {
		name: "a query head narrower than the cache's",
		build: func(b *tensor.Builder) {
			scalars(b)
			tensor.Attention(b, f32(b, "q", 4, 8), cache(b, "k", 4, 2, 16),
				cache(b, "v", 4, 2, 16), opts(b))
		},
		want: "q's head is 8 wide and the cache's is 16",
	}, {
		name: "a scale name that names no scalar",
		build: func(b *tensor.Builder) {
			scalars(b)
			o := opts(b)
			o.ScaleName = "nosuch"
			tensor.Attention(b, f32(b, "q", 4, 8), cache(b, "k", 4, 2, 8),
				cache(b, "v", 4, 2, 8), o)
		},
		want: `"nosuch" is not a declared f32 scalar`,
	}, {
		// The key cache's staleness is covered elsewhere; the value cache has
		// its own branch and its own message, and a copy-paste that checked k
		// twice would pass every test that only writes to k.
		name: "a stale value cache",
		build: func(b *tensor.Builder) {
			scalars(b)
			kc, vc := cache(b, "k", 8, 2, 8), cache(b, "v", 8, 2, 8)
			stale := vc
			vc = tensor.ScatterRows(b, vc, f32(b, "r", 1, 16), u32(b, "i", 1))
			_ = vc
			tensor.Attention(b, f32(b, "q", 2, 8), kc, stale, opts(b))
		},
		want: "the value cache is",
	}, {
		name: "a ragged page table whose rows are not the sequences",
		build: func(b *tensor.Builder) {
			scalars(b)
			o := opts(b)
			o.Lengths = u32(b, "len2", 2)
			o.QueryExtents = u32(b, "ext", 2)
			o.Pages = u32(b, "pages", 5, 3)
			o.Block = 4
			tensor.Attention(b, f32(b, "q", 3, 4, 8), cache(b, "k", 16, 2, 8),
				cache(b, "v", 16, 2, 8), o)
		},
		want: "one row of block ids per sequence",
	}, {
		name: "a batched decode over an f16 cache",
		build: func(b *tensor.Builder) {
			scalars(b)
			kc := tensor.NewState(b, tensor.StateDesc{
				Name: "k", DType: accel.F16, Shape: tensor.Shape{16, 2, 8}})
			vc := tensor.NewState(b, tensor.StateDesc{
				Name: "v", DType: accel.F16, Shape: tensor.Shape{16, 2, 8}})
			o := opts(b)
			o.Lengths = u32(b, "len2", 2)
			o.Pages = u32(b, "pages", 2, 3)
			o.Block = 4
			tensor.Attention(b, f32(b, "q", 2, 1, 4, 8), kc, vc, o)
		},
		want: "batched decode kernel",
	}, {
		name: "a base name that names no u32 scalar",
		build: func(b *tensor.Builder) {
			scalars(b)
			tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarF32})
			o := opts(b)
			o.BaseName = "base"
			tensor.Attention(b, f32(b, "q", 2, 4, 8), cache(b, "k", 8, 2, 8),
				cache(b, "v", 8, 2, 8), o)
		},
		want: `"base" is not a declared u32 scalar`,
	}, {
		// Reachable only through a view. Input refuses a dimension of zero, so
		// the empty tensor this needs cannot be declared -- but Slice permits
		// an empty half-open range, and the result is a legal zero-element
		// operand. Checking reachability against the constructor alone says
		// this branch is dead, and it is not.
		name: "extents sliced empty",
		build: func(b *tensor.Builder) {
			scalars(b)
			o := opts(b)
			o.Lengths = u32(b, "len2", 2)
			o.QueryExtents = tensor.Slice(b, u32(b, "ext", 2), 0, 0, 0)
			tensor.Attention(b, f32(b, "q", 3, 4, 8), cache(b, "k", 16, 2, 8),
				cache(b, "v", 16, 2, 8), o)
		},
		want: "QueryExtents is empty",
	}} {
		t.Run(tc.name, func(t *testing.T) {
			b := rt.NewBuilder(tc.name)
			tc.build(b)
			err := b.Err()
			if err == nil {
				t.Fatalf("%s was accepted", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("the refusal should say %q, got %v", tc.want, err)
			}
		})
	}

	// A poisoned state flows through without a second diagnostic.
	b := rt.NewBuilder("poison")
	scalars(b)
	bad := tensor.ScatterRows(b, cache(b, "c", 8, 4), f32(b, "r", 1, 3), u32(b, "i", 1))
	before := b.Err().Error()
	tensor.ScatterRows(b, bad, f32(b, "r2", 1, 4), u32(b, "i2", 1))
	tensor.LayerState(b, bad, 0)
	tensor.ReadState(b, bad)
	tensor.Attention(b, f32(b, "q", 4, 8), bad, bad, opts(b))
	if b.Err().Error() != before {
		t.Errorf("an operator on a poisoned state recorded a diagnostic:\n%v", b.Err())
	}
}

// A state version that has been superseded is refused, at the caller's line.
//
// This is the test that made the version chain mean something. The first
// implementation compiled a graph where Attention read the version *before* the
// scatter, ordered nothing differently, and produced the right answer anyway --
// because both versions are one caller-owned buffer and the read happened after
// the write regardless. A distinction that cannot be violated is not a
// distinction, so an older version is now an error rather than a silent
// synonym for the newer one.
func TestAStaleStateVersionIsRefused(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("stale")
	tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})

	c := tensor.NewState(b, tensor.StateDesc{
		Name: "cache", DType: accel.F32, Shape: tensor.Shape{4, 2, 8},
	})
	rows := tensor.Input(b, tensor.ValueDesc{
		Name: "rows", DType: accel.F32, Shape: tensor.Shape{1, 16},
	})
	ids := tensor.Input(b, tensor.ValueDesc{
		Name: "ids", DType: accel.U32, Shape: tensor.Shape{1},
	})
	q := tensor.Input(b, tensor.ValueDesc{
		Name: "q", DType: accel.F32, Shape: tensor.Shape{4, 8},
	})

	tensor.ScatterRows(b, c, rows, ids) // c is now version 0 of 1

	// Reading the old version, three ways.
	for _, tc := range []struct {
		name string
		call func()
	}{
		{"through ReadState", func() { tensor.ReadState(b, c) }},
		{"through Attention", func() {
			tensor.Attention(b, q, c, c, tensor.AttentionOptions{
				Lengths: tensor.Input(b, tensor.ValueDesc{
					Name: "attnlen", DType: accel.U32, Shape: tensor.Shape{1},
				}), ScaleName: "scale",
			})
		}},
		{"through a second write", func() { tensor.ScatterRows(b, c, rows, ids) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := ""
			if err := b.Err(); err != nil {
				before = err.Error()
			}
			tc.call()
			err := b.Err()
			if err == nil || err.Error() == before {
				t.Fatalf("a stale version was accepted %s", tc.name)
			}
			if !strings.Contains(err.Error(), "written 1 time(s) since") {
				t.Errorf("the refusal should say how far behind: %v", err)
			}
			if !strings.Contains(err.Error(), "decode_test.go:") {
				t.Errorf("the refusal should name the caller's line rather than a line "+
					"inside the package: %v", err)
			}
		})
	}
}

// A decode step over a cache far longer than a workgroup, end to end through
// the public operator.
//
// This is the case accel issue 8 reported: Attention refused any cache past 128
// positions, so no model was servable and no test against real weights was
// writable. 4096 is specs/044-unbounded-context.md section 7's figure, and it
// is thirty-two blocks rather than one, with the length landing mid-block so
// the tail is masked rather than aligned.
func TestADecodeStepOverALongCache(t *testing.T) {
	const (
		qHeads   = 4
		kvHeads  = 2
		headDim  = 8
		capacity = 4096
		kvLen    = 2001
	)
	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("longdecode")

	tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
	q := tensor.Input(b, tensor.ValueDesc{
		Name: "q", DType: accel.F32, Shape: tensor.Shape{qHeads, headDim},
	})
	lengths := tensor.Input(b, tensor.ValueDesc{
		Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
	})
	kc := tensor.NewState(b, tensor.StateDesc{
		Name: "kcache", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
	})
	vc := tensor.NewState(b, tensor.StateDesc{
		Name: "vcache", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
	})
	tensor.Output(b, "out", tensor.Attention(b, q, kc, vc, tensor.AttentionOptions{
		Lengths: lengths, ScaleName: "scale",
	}))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "longdecode"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	// The cache is filled past kvLen, so a kernel that ignored the length would
	// read values that are present rather than zeros and get a different answer
	// instead of the same one.
	ks := make([]float32, capacity*kvHeads*headDim)
	vs := make([]float32, capacity*kvHeads*headDim)
	for i := range ks {
		ks[i] = float32(math.Cos(float64(i) * 0.017))
		vs[i] = float32(math.Sin(float64(i) * 0.013))
	}
	qs := make([]float32, qHeads*headDim)
	for i := range qs {
		qs[i] = float32(math.Sin(float64(i) * 0.29))
	}
	scale := float32(1 / math.Sqrt(headDim))

	out := f32Buffer(t, d, "out", make([]float32, qHeads*headDim))
	f := plan.Submit(d.Queue(), tensor.Bindings{
		Buffers: map[string]accel.BufferView{
			"q":      f32Buffer(t, d, "q", qs),
			"kcache": f32Buffer(t, d, "kcache", ks),
			"vcache": f32Buffer(t, d, "vcache", vs),
			"len":    u32Buffer(t, d, "len", []uint32{kvLen}),
			"out":    out,
		},
		Scalars: map[string]tensor.ScalarValue{"scale": tensor.F32(scale)},
	})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, qHeads*headDim)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}

	// The reference, in f64, over exactly the positions the length names.
	group := qHeads / kvHeads
	for h := range qHeads {
		kvHead := h / group
		scores := make([]float64, kvLen)
		best := math.Inf(-1)
		for pos := range kvLen {
			var acc float64
			for i := range headDim {
				acc += float64(qs[h*headDim+i]) *
					float64(ks[(pos*kvHeads+kvHead)*headDim+i])
			}
			scores[pos] = acc * float64(scale)
			best = math.Max(best, scores[pos])
		}
		var sum float64
		for i := range scores {
			scores[i] = math.Exp(scores[i] - best)
			sum += scores[i]
		}
		for i := range headDim {
			var acc float64
			for pos := range kvLen {
				acc += scores[pos] / sum * float64(vs[(pos*kvHeads+kvHead)*headDim+i])
			}
			if g := float64(got[h*headDim+i]); math.Abs(g-acc) > 1e-4*(1+math.Abs(acc)) {
				t.Fatalf("head %d element %d is %v, want about %v", h, i, g, acc)
			}
		}
	}
}

// A cache holding every layer, bound once.
//
// This is accel issue 9: Attention refused a LayerState view, so a 36-layer
// model needed 72 states and 72 bindings for what is one allocation. The view
// arithmetic was built and tested the whole time -- what was missing is that
// LayerState computed an offset and the compiler never read it, so the refusal
// was honest.
//
// A window of a port is now its own graph slot, derived from the one view the
// caller binds. Each layer's kernel indexes its own layer from zero and has no
// idea which layer it is, which is why the offset has to be the binding's and
// cannot be the kernel's.
func TestALayeredCacheBindsOnce(t *testing.T) {
	const (
		layers   = 6
		qHeads   = 4
		kvHeads  = 2
		headDim  = 8
		capacity = 16
		perLayer = capacity * kvHeads * headDim
	)
	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("layered")

	tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
	lengths := tensor.Input(b, tensor.ValueDesc{
		Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
	})
	slot := tensor.Input(b, tensor.ValueDesc{
		Name: "slot", DType: accel.U32, Shape: tensor.Shape{1},
	})
	// One state for every layer's keys and one for every layer's values, which
	// is two buffers rather than 2*layers.
	kc := tensor.NewState(b, tensor.StateDesc{
		Name: "kcache", DType: accel.F32,
		Shape: tensor.Shape{layers, capacity, kvHeads, headDim},
	})
	vc := tensor.NewState(b, tensor.StateDesc{
		Name: "vcache", DType: accel.F32,
		Shape: tensor.Shape{layers, capacity, kvHeads, headDim},
	})

	for l := range layers {
		q := tensor.Input(b, tensor.ValueDesc{
			Name: fmt.Sprintf("q%d", l), DType: accel.F32,
			Shape: tensor.Shape{qHeads, headDim},
		})
		newK := tensor.Input(b, tensor.ValueDesc{
			Name: fmt.Sprintf("k%d", l), DType: accel.F32,
			Shape: tensor.Shape{1, kvHeads * headDim},
		})
		newV := tensor.Input(b, tensor.ValueDesc{
			Name: fmt.Sprintf("v%d", l), DType: accel.F32,
			Shape: tensor.Shape{1, kvHeads * headDim},
		})
		lk := tensor.ScatterRows(b, tensor.LayerState(b, kc, l), newK, slot)
		lv := tensor.ScatterRows(b, tensor.LayerState(b, vc, l), newV, slot)
		tensor.Output(b, fmt.Sprintf("out%d", l),
			tensor.Attention(b, q, lk, lv, tensor.AttentionOptions{
				Lengths: lengths, ScaleName: "scale",
			}))
	}

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "layered"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	kBuf := f32Buffer(t, d, "kcache", make([]float32, layers*perLayer))
	vBuf := f32Buffer(t, d, "vcache", make([]float32, layers*perLayer))
	scale := float32(1 / math.Sqrt(headDim))

	// A host model of every layer's cache, stepped alongside.
	keys := make([][][]float32, layers)
	vals := make([][][]float32, layers)

	for step := range 3 {
		bufs := map[string]accel.BufferView{
			"kcache": kBuf, "vcache": vBuf,
			"len":  u32Buffer(t, d, "len", []uint32{uint32(step + 1)}),
			"slot": u32Buffer(t, d, "slot", []uint32{uint32(step)}),
		}
		outs := map[string]accel.BufferView{}
		qs := make([][]float32, layers)
		for l := range layers {
			q := make([]float32, qHeads*headDim)
			ks := make([]float32, kvHeads*headDim)
			vs := make([]float32, kvHeads*headDim)
			for i := range q {
				q[i] = float32(math.Sin(float64(step*31 + l*7 + i)))
			}
			for i := range ks {
				// Distinct per layer, so a layer reading another layer's slice
				// is a wrong answer rather than a coincidence.
				ks[i] = float32(math.Cos(float64(l*100 + step*7 + i)))
				vs[i] = float32(l*10) + float32(step) + float32(i)/8
			}
			qs[l] = q
			keys[l] = append(keys[l], ks)
			vals[l] = append(vals[l], vs)
			bufs[fmt.Sprintf("q%d", l)] = f32Buffer(t, d, "q", q)
			bufs[fmt.Sprintf("k%d", l)] = f32Buffer(t, d, "k", ks)
			bufs[fmt.Sprintf("v%d", l)] = f32Buffer(t, d, "v", vs)
			o := f32Buffer(t, d, "out", make([]float32, qHeads*headDim))
			outs[fmt.Sprintf("out%d", l)] = o
			bufs[fmt.Sprintf("out%d", l)] = o
		}

		f := plan.Submit(d.Queue(), tensor.Bindings{
			Buffers: bufs,
			Scalars: map[string]tensor.ScalarValue{"scale": tensor.F32(scale)},
		})
		if err := f.Wait(); err != nil {
			t.Fatalf("step %d: %v", step, err)
		}

		for l := range layers {
			got := make([]float32, qHeads*headDim)
			if err := d.Queue().ReadBuffer(outs[fmt.Sprintf("out%d", l)].Buffer, 0, got); err != nil {
				t.Fatalf("readback: %v", err)
			}
			want := attentionReference(qs[l], keys[l], vals[l],
				qHeads, kvHeads, headDim, float64(scale))
			for i := range got {
				if diff := math.Abs(float64(got[i]) - want[i]); diff > 1e-4*(1+math.Abs(want[i])) {
					t.Fatalf("step %d layer %d element %d is %v, want about %v",
						step, l, i, got[i], want[i])
				}
			}
		}
	}

	// Every layer's slice holds its own keys and no other layer's, which a
	// per-step output check cannot see: a wrong offset that is consistent
	// between the scatter and the attention would agree with itself.
	gotK := make([]float32, layers*perLayer)
	if err := d.Queue().ReadBuffer(kBuf.Buffer, 0, gotK); err != nil {
		t.Fatalf("readback: %v", err)
	}
	for l := range layers {
		base := l * perLayer
		for pos := range keys[l] {
			for i := range kvHeads * headDim {
				want := keys[l][pos][i]
				if g := gotK[base+pos*kvHeads*headDim+i]; g != want {
					t.Fatalf("layer %d position %d element %d is %v, want %v",
						l, pos, i, g, want)
				}
			}
		}
	}
}

// attentionReference is one decode step in f64.
func attentionReference(q []float32, keys, vals [][]float32,
	qHeads, kvHeads, headDim int, scale float64) []float64 {

	out := make([]float64, qHeads*headDim)
	group := qHeads / kvHeads
	for h := range qHeads {
		kvHead := h / group
		scores := make([]float64, len(keys))
		best := math.Inf(-1)
		for pos := range keys {
			var acc float64
			for i := range headDim {
				acc += float64(q[h*headDim+i]) * float64(keys[pos][kvHead*headDim+i])
			}
			scores[pos] = acc * scale
			best = math.Max(best, scores[pos])
		}
		var sum float64
		for i := range scores {
			scores[i] = math.Exp(scores[i] - best)
			sum += scores[i]
		}
		for i := range headDim {
			var acc float64
			for pos := range keys {
				acc += scores[pos] / sum * float64(vals[pos][kvHead*headDim+i])
			}
			out[h*headDim+i] = acc
		}
	}
	return out
}

// Attention over a layer this plan did not write.
//
// The companion to TestALayeredCacheBindsOnce, and not a duplicate of it: there
// every layer was scattered first, so Attention read the State that ScatterRows
// produced and the binding came from the slot that node wrote. The read window
// was never consulted. This reads a cache filled by an earlier submission,
// which is what a decode plan does on every step after the first, and it is the
// only path where a layer view's offset reaches the binding through the *read*.
func TestAttentionReadsALayerItDidNotWrite(t *testing.T) {
	const (
		layers   = 4
		qHeads   = 2
		kvHeads  = 1
		headDim  = 8
		capacity = 4
		perLayer = capacity * kvHeads * headDim
	)
	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("readlayer")

	tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
	lengths := tensor.Input(b, tensor.ValueDesc{
		Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
	})
	kc := tensor.NewState(b, tensor.StateDesc{
		Name: "kcache", DType: accel.F32,
		Shape: tensor.Shape{layers, capacity, kvHeads, headDim},
	})
	vc := tensor.NewState(b, tensor.StateDesc{
		Name: "vcache", DType: accel.F32,
		Shape: tensor.Shape{layers, capacity, kvHeads, headDim},
	})
	for l := range layers {
		q := tensor.Input(b, tensor.ValueDesc{
			Name: fmt.Sprintf("q%d", l), DType: accel.F32,
			Shape: tensor.Shape{qHeads, headDim},
		})
		tensor.Output(b, fmt.Sprintf("out%d", l),
			tensor.Attention(b, q, tensor.LayerState(b, kc, l), tensor.LayerState(b, vc, l),
				tensor.AttentionOptions{Lengths: lengths, ScaleName: "scale"}))
	}
	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "readlayer"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	// Each layer's slice holds values only that layer should see.
	ks := make([]float32, layers*perLayer)
	vs := make([]float32, layers*perLayer)
	for l := range layers {
		for i := range perLayer {
			ks[l*perLayer+i] = float32(math.Cos(float64(l*50 + i)))
			vs[l*perLayer+i] = float32(l*10) + float32(i)/4
		}
	}
	scale := float32(1 / math.Sqrt(headDim))

	bufs := map[string]accel.BufferView{
		"kcache": f32Buffer(t, d, "kcache", ks),
		"vcache": f32Buffer(t, d, "vcache", vs),
		"len":    u32Buffer(t, d, "len", []uint32{capacity}),
	}
	outs := map[string]accel.BufferView{}
	qs := make([][]float32, layers)
	for l := range layers {
		q := make([]float32, qHeads*headDim)
		for i := range q {
			q[i] = float32(math.Sin(float64(l*3 + i)))
		}
		qs[l] = q
		bufs[fmt.Sprintf("q%d", l)] = f32Buffer(t, d, "q", q)
		o := f32Buffer(t, d, "out", make([]float32, qHeads*headDim))
		outs[fmt.Sprintf("out%d", l)] = o
		bufs[fmt.Sprintf("out%d", l)] = o
	}
	f := plan.Submit(d.Queue(), tensor.Bindings{
		Buffers: bufs,
		Scalars: map[string]tensor.ScalarValue{"scale": tensor.F32(scale)},
	})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	for l := range layers {
		got := make([]float32, qHeads*headDim)
		if err := d.Queue().ReadBuffer(outs[fmt.Sprintf("out%d", l)].Buffer, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}
		keys := make([][]float32, capacity)
		vals := make([][]float32, capacity)
		for pos := range capacity {
			base := l*perLayer + pos*kvHeads*headDim
			keys[pos] = ks[base : base+kvHeads*headDim]
			vals[pos] = vs[base : base+kvHeads*headDim]
		}
		want := attentionReference(qs[l], keys, vals, qHeads, kvHeads, headDim, float64(scale))
		for i := range got {
			if diff := math.Abs(float64(got[i]) - want[i]); diff > 1e-4*(1+math.Abs(want[i])) {
				t.Fatalf("layer %d element %d is %v, want about %v: a layer read must "+
					"bind that layer's slice", l, i, got[i], want[i])
			}
		}
	}
}

// A paged prefill writes blocks and a paged decode reads them, in one plan.
//
// This is the workflow accel issue 10 says was inexpressible: a paged decode is
// only useful over blocks a paged prefill wrote, and until the paged prefill
// kernel existed, `Attention` accepted a page table on a prefill and ignored
// it. The reporter's point was that this is not a corner but the first
// operation of every request in a paged design.
//
// The pool is larger than the sequence and the table is scattered and out of
// order, so a prefill that walked the pool in order reads the wrong blocks --
// which is what the bug did.
func TestAPagedPrefillFeedsAPagedDecode(t *testing.T) {
	const (
		qHeads, kvHeads, headDim = 4, 2, 16
		block                    = 4
		qSeq                     = 9
		pageCount                = 4 // 16 positions, enough for the prompt and a step
		poolBlocks               = 12
		width                    = kvHeads * headDim
	)
	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("pagedprefill")

	tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
	tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarU32})
	pages := tensor.Input(b, tensor.ValueDesc{
		Name: "pages", DType: accel.U32, Shape: tensor.Shape{pageCount},
	})
	lengths := tensor.Input(b, tensor.ValueDesc{
		Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
	})
	q := tensor.Input(b, tensor.ValueDesc{
		Name: "q", DType: accel.F32, Shape: tensor.Shape{qSeq, qHeads, headDim},
	})
	kc := tensor.NewState(b, tensor.StateDesc{
		Name: "kpool", DType: accel.F32,
		Shape: tensor.Shape{poolBlocks * block, kvHeads, headDim},
	})
	vc := tensor.NewState(b, tensor.StateDesc{
		Name: "vpool", DType: accel.F32,
		Shape: tensor.Shape{poolBlocks * block, kvHeads, headDim},
	})
	tensor.Output(b, "out", tensor.Attention(b, q, kc, vc, tensor.AttentionOptions{
		Lengths: lengths, Pages: pages, Block: block,
		ScaleName: "scale", BaseName: "base",
	}))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "pagedprefill"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	// The selection is reported, which is what makes the choice reviewable --
	// the bug's worst property was that Selections named the contiguous kernel
	// while a page table was bound.
	sel := plan.Selections()
	if len(sel) != 1 || sel[0].Kernel != "AttentionPrefillPaged" {
		t.Fatalf("selections are %+v, want the paged prefill kernel", sel)
	}

	rng := rand.New(rand.NewPCG(41, 3))
	pool := func() []float32 {
		s := make([]float32, poolBlocks*block*width)
		for i := range s {
			s[i] = float32(rng.NormFloat64())
		}
		return s
	}
	pk, pv := pool(), pool()
	qs := make([]float32, qSeq*qHeads*headDim)
	for i := range qs {
		qs[i] = float32(rng.NormFloat64())
	}
	// Scattered and out of order.
	table := []uint32{7, 2, 10, 5}
	scale := float32(1 / math.Sqrt(headDim))

	out := f32Buffer(t, d, "out", make([]float32, qSeq*qHeads*headDim))
	f := plan.Submit(d.Queue(), tensor.Bindings{
		Buffers: map[string]accel.BufferView{
			"q": f32Buffer(t, d, "q", qs), "kpool": f32Buffer(t, d, "kpool", pk),
			"vpool": f32Buffer(t, d, "vpool", pv),
			"pages": u32Buffer(t, d, "pages", table),
			"len":   u32Buffer(t, d, "len", []uint32{qSeq}),
			"out":   out,
		},
		Scalars: map[string]tensor.ScalarValue{
			"scale": tensor.F32(scale), "base": tensor.U32(0),
		},
	})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, qSeq*qHeads*headDim)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}

	// The reference gathers the same positions through the same table and
	// attends causally in f64.
	gk := make([][]float32, qSeq)
	gv := make([][]float32, qSeq)
	for j := range qSeq {
		phys := int(table[j/block])*block + j%block
		gk[j] = pk[phys*width : (phys+1)*width]
		gv[j] = pv[phys*width : (phys+1)*width]
	}
	for s := range qSeq {
		want := attentionReference(qs[s*qHeads*headDim:(s+1)*qHeads*headDim],
			gk[:s+1], gv[:s+1], qHeads, kvHeads, headDim, float64(scale))
		for i := range qHeads * headDim {
			g := float64(got[s*qHeads*headDim+i])
			if math.Abs(g-want[i]) > 1e-4*(1+math.Abs(want[i])) {
				t.Fatalf("query %d element %d is %v, want about %v: a paged prefill "+
					"attends over the blocks its table names", s, i, g, want[i])
			}
		}
	}
}

// A batch of sequences steps together, each over its own pages and its own
// length.
//
// accel issue 12: everything around attention was already batched -- Lengths is
// per sequence, Pages is [s][i], RoPE takes per-row positions, the samplers
// draw per row -- and q's shape was the one thing that was not, so a batched
// decode was read as a prefill and refused. The kernel had been in the corpus
// and tested since 030 with no operator reaching it.
//
// The lengths differ on purpose. That is what continuous batching *is*, and a
// batch padded to its longest sequence would be the allocation paging exists to
// avoid.
func TestABatchOfSequencesStepsTogether(t *testing.T) {
	const (
		batch, qHeads, kvHeads, headDim = 3, 4, 2, 8
		block, maxPages                 = 4, 3
		poolBlocks                      = 16
		width                           = kvHeads * headDim
	)
	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("batched")

	tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
	q := tensor.Input(b, tensor.ValueDesc{
		Name: "q", DType: accel.F32, Shape: tensor.Shape{batch, 1, qHeads, headDim},
	})
	pages := tensor.Input(b, tensor.ValueDesc{
		Name: "pages", DType: accel.U32, Shape: tensor.Shape{batch, maxPages},
	})
	lengths := tensor.Input(b, tensor.ValueDesc{
		Name: "len", DType: accel.U32, Shape: tensor.Shape{batch},
	})
	kc := tensor.NewState(b, tensor.StateDesc{
		Name: "kpool", DType: accel.F32,
		Shape: tensor.Shape{poolBlocks * block, kvHeads, headDim},
	})
	vc := tensor.NewState(b, tensor.StateDesc{
		Name: "vpool", DType: accel.F32,
		Shape: tensor.Shape{poolBlocks * block, kvHeads, headDim},
	})
	tensor.Output(b, "out", tensor.Attention(b, q, kc, vc, tensor.AttentionOptions{
		Lengths: lengths, Pages: pages, Block: block, ScaleName: "scale",
	}))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "batched"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	sel := plan.Selections()
	if len(sel) != 1 || sel[0].Kernel != "AttentionDecodeBatched" {
		t.Fatalf("selections are %+v, want the batched kernel", sel)
	}

	rng := rand.New(rand.NewPCG(13, 29))
	pk := make([]float32, poolBlocks*block*width)
	pv := make([]float32, len(pk))
	for i := range pk {
		pk[i] = float32(rng.NormFloat64())
		pv[i] = float32(rng.NormFloat64())
	}
	qs := make([]float32, batch*qHeads*headDim)
	for i := range qs {
		qs[i] = float32(rng.NormFloat64())
	}
	// Disjoint, scattered page rows, so a sequence reading another's blocks is
	// a wrong answer rather than a coincidence.
	table := []uint32{
		9, 2, 14,
		0, 11, 5,
		7, 3, 12,
	}
	lens := []uint32{10, 3, 7} // deliberately unequal, and none a block multiple
	scale := float32(1 / math.Sqrt(headDim))

	out := f32Buffer(t, d, "out", make([]float32, batch*qHeads*headDim))
	f := plan.Submit(d.Queue(), tensor.Bindings{
		Buffers: map[string]accel.BufferView{
			"q": f32Buffer(t, d, "q", qs), "kpool": f32Buffer(t, d, "kpool", pk),
			"vpool": f32Buffer(t, d, "vpool", pv),
			"pages": u32Buffer(t, d, "pages", table),
			"len":   u32Buffer(t, d, "len", lens),
			"out":   out,
		},
		Scalars: map[string]tensor.ScalarValue{"scale": tensor.F32(scale)},
	})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, batch*qHeads*headDim)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}

	for s := range batch {
		n := int(lens[s])
		keys := make([][]float32, n)
		vals := make([][]float32, n)
		for j := range n {
			phys := int(table[s*maxPages+j/block])*block + j%block
			keys[j] = pk[phys*width : (phys+1)*width]
			vals[j] = pv[phys*width : (phys+1)*width]
		}
		want := attentionReference(qs[s*qHeads*headDim:(s+1)*qHeads*headDim],
			keys, vals, qHeads, kvHeads, headDim, float64(scale))
		for i := range qHeads * headDim {
			g := float64(got[s*qHeads*headDim+i])
			if math.Abs(g-want[i]) > 1e-4*(1+math.Abs(want[i])) {
				t.Fatalf("sequence %d (length %d) element %d is %v, want about %v",
					s, n, i, g, want[i])
			}
		}
	}
}

// An output naming a state is refused rather than silently producing zeros.
//
// The lowering matches an output to the node that produces it by tensor
// identity, and ReadState builds a fresh tensor rather than returning the
// writing node's result. So this bound a port nothing ever wrote: the caller
// read zeros while the state's own buffer held the right answer the whole time.
//
// Found by asking whether a repetition penalty composes from what exists --
// gather the seen tokens' logits, scale them, scatter them back -- which is a
// shape nothing in the suite had written, because every state test reads the
// state's bound buffer directly rather than naming it as an output.
func TestAnOutputNamingAStateIsRefused(t *testing.T) {
	const vocab = 16
	rt := newRuntime(t)
	b := rt.NewBuilder("stateout")

	st := tensor.NewState(b, tensor.StateDesc{
		Name: "s", DType: accel.F32, Shape: tensor.Shape{vocab, 1},
	})
	rows := tensor.Input(b, tensor.ValueDesc{
		Name: "rows", DType: accel.F32, Shape: tensor.Shape{3, 1},
	})
	ids := tensor.Input(b, tensor.ValueDesc{
		Name: "ids", DType: accel.U32, Shape: tensor.Shape{3},
	})
	next := tensor.ScatterRows(b, st, rows, ids)
	tensor.Output(b, "out", tensor.ReadState(b, next))

	_, err := b.Compile(rt, tensor.CompileOptions{Label: "stateout"})
	if err == nil {
		t.Fatal("an output naming a state was accepted, and it produces zeros")
	}
	for _, want := range []string{`names state "s"`, "bind it and read it"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal should say %q, got %v", want, err)
		}
	}
}

// A paged prefill over an f16 cache selects the narrow paged kernel and agrees
// with the f32 one.
//
// [#25](https://github.com/golang-design/accel/issues/25). The narrow cache
// existed for a contiguous prefill and for a paged decode, and not for the
// combination -- which is the one a consumer actually reaches: a shared block
// pool is addressed through a page table by construction, and every
// conversation begins with a prefill. So the f16 cache was available to a
// single-sequence contiguous caller and to nobody who pages, which is the
// opposite of who wants it.
//
// The selection is asserted rather than only the compile, because the bug's
// shape was a *silent* one. The narrow kernel was chosen and then overwritten
// by the paged branch, so what a caller saw was a binding-width complaint about
// a plan they had assembled correctly.
func TestAPagedPrefillOverAnF16CacheSelectsTheNarrowKernel(t *testing.T) {
	const (
		qHeads, kvHeads, headDim = 4, 2, 16
		block                    = 4
		qSeq                     = 9
		pageCount                = 4
		poolBlocks               = 12
		width                    = kvHeads * headDim
	)
	rt := newRuntime(t)
	d := rt.Device()

	rng := rand.New(rand.NewPCG(17, 29))
	pool := make([]float32, poolBlocks*block*width)
	for i := range pool {
		pool[i] = float32(rng.NormFloat64())
	}
	qs := make([]float32, qSeq*qHeads*headDim)
	for i := range qs {
		qs[i] = float32(rng.NormFloat64())
	}
	pageTable := []uint32{3, 1, 5, 2}

	// Both caches hold the same values, the narrow one rounded. Comparing the
	// two answers is then a statement about the kernel rather than about the
	// data, since the f32 run reads exactly what the f16 run rounds.
	rounded := make([]float32, len(pool))
	narrow := make([]accel.Float16, len(pool))
	for i, v := range pool {
		narrow[i] = accel.ToFloat16(v)
		rounded[i] = narrow[i].F32()
	}

	run := func(label string, dt accel.DType) ([]float32, string) {
		t.Helper()
		b := rt.NewBuilder(label)
		tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
		tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarU32})
		pages := tensor.Input(b, tensor.ValueDesc{
			Name: "pages", DType: accel.U32, Shape: tensor.Shape{pageCount}})
		lengths := tensor.Input(b, tensor.ValueDesc{
			Name: "len", DType: accel.U32, Shape: tensor.Shape{1}})
		q := tensor.Input(b, tensor.ValueDesc{
			Name: "q", DType: accel.F32, Shape: tensor.Shape{qSeq, qHeads, headDim}})
		kc := tensor.NewState(b, tensor.StateDesc{
			Name: "kpool", DType: dt,
			Shape: tensor.Shape{poolBlocks * block, kvHeads, headDim}})
		vc := tensor.NewState(b, tensor.StateDesc{
			Name: "vpool", DType: dt,
			Shape: tensor.Shape{poolBlocks * block, kvHeads, headDim}})
		tensor.Output(b, "out", tensor.Attention(b, q, kc, vc, tensor.AttentionOptions{
			Lengths: lengths, Pages: pages, Block: block,
			ScaleName: "scale", BaseName: "base",
		}))

		plan, err := b.Compile(rt, tensor.CompileOptions{Label: label})
		if err != nil {
			t.Fatalf("compile %s: %v", label, err)
		}
		defer plan.Close()

		sel := plan.Selections()
		if len(sel) != 1 {
			t.Fatalf("%s: selections are %+v, want one", label, sel)
		}

		bufs := map[string]accel.BufferView{
			"pages": u32Buffer(t, d, "pages", pageTable),
			"len":   u32Buffer(t, d, "len", []uint32{qSeq}),
			"q":     f32Buffer(t, d, "q", qs),
		}
		out := f32Buffer(t, d, "out", make([]float32, qSeq*qHeads*headDim))
		bufs["out"] = out
		if dt == accel.F16 {
			bufs["kpool"] = f16Buffer(t, d, "kpool", pool)
			bufs["vpool"] = f16Buffer(t, d, "vpool", pool)
		} else {
			bufs["kpool"] = f32Buffer(t, d, "kpool", rounded)
			bufs["vpool"] = f32Buffer(t, d, "vpool", rounded)
		}
		f := plan.Submit(d.Queue(), tensor.Bindings{
			Buffers: bufs,
			Scalars: map[string]tensor.ScalarValue{
				"scale": tensor.F32(float32(1 / math.Sqrt(headDim))),
				"base":  tensor.U32(0),
			},
		})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit %s: %v", label, err)
		}
		got := make([]float32, qSeq*qHeads*headDim)
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("readback %s: %v", label, err)
		}
		return got, sel[0].Kernel
	}

	wide, wideKernel := run("pagedprefill32", accel.F32)
	got, gotKernel := run("pagedprefill16", accel.F16)

	if wideKernel != "AttentionPrefillPaged" {
		t.Fatalf("the f32 cache selected %q", wideKernel)
	}
	if gotKernel != "AttentionPrefillPagedF16" {
		t.Fatalf("a paged prefill over an f16 cache selected %q; the f32 kernel would "+
			"read the halves as full floats and answer plausibly and wrongly", gotKernel)
	}

	// The f16 cache holds exactly what the f32 one holds, so the two differ
	// only by the order the widened values are summed in --
	// specs/008-numerics.md §7's reduction bound, not a quantization one.
	for i := range wide {
		if e := math.Abs(float64(got[i] - wide[i])); e > 1e-4 {
			t.Fatalf("element %d: f16 cache %v, f32 cache %v (off %v)",
				i, got[i], wide[i], e)
		}
	}
	nonzero := false
	for _, v := range wide {
		if v != 0 {
			nonzero = true
		}
	}
	if !nonzero {
		t.Fatal("every output is zero, so the comparison says nothing")
	}
}

// Every attention shape, over both cache widths, either selects a kernel or is
// refused by name.
//
// [#25](https://github.com/golang-design/accel/issues/25)'s real lesson, and
// the reporter's framing rather than mine: #13 closed on "ScatterRows, prefill
// and paged decode all take f16", and each of those three was true. The
// *combination* was not. A gap in a matrix is invisible to any test that checks
// the axes one at a time, which is what every f16 test here had been doing.
//
// The failure this forbids is the silent one. A missing combination that
// refuses by name costs a caller one line of output; one that quietly selects
// the wrong width costs them an afternoon inside a binding-size complaint about
// a plan they assembled correctly. So an unbuilt cell is allowed and an
// unnamed one is not.
func TestEveryAttentionShapeOverBothCacheWidths(t *testing.T) {
	const (
		qHeads, kvHeads, headDim = 2, 1, 8
		block                    = 4
		capacity                 = 32
	)
	rt := newRuntime(t)

	// Each shape as the caller writes it: the query rank and the options that
	// select a path. The cache width is the other axis.
	shapes := []struct {
		name string
		q    tensor.Shape
		opts func(b *tensor.Builder) tensor.AttentionOptions
	}{{
		name: "decode",
		q:    tensor.Shape{qHeads, headDim},
		opts: func(b *tensor.Builder) tensor.AttentionOptions {
			return tensor.AttentionOptions{
				Lengths: tensor.Input(b, tensor.ValueDesc{
					Name: "len", DType: accel.U32, Shape: tensor.Shape{1}}),
				ScaleName: "scale",
			}
		},
	}, {
		name: "paged decode",
		q:    tensor.Shape{qHeads, headDim},
		opts: func(b *tensor.Builder) tensor.AttentionOptions {
			return tensor.AttentionOptions{
				Lengths: tensor.Input(b, tensor.ValueDesc{
					Name: "len", DType: accel.U32, Shape: tensor.Shape{1}}),
				Pages: tensor.Input(b, tensor.ValueDesc{
					Name: "pages", DType: accel.U32, Shape: tensor.Shape{4}}),
				Block: block, ScaleName: "scale",
			}
		},
	}, {
		name: "prefill",
		q:    tensor.Shape{3, qHeads, headDim},
		opts: func(b *tensor.Builder) tensor.AttentionOptions {
			return tensor.AttentionOptions{
				Lengths: tensor.Input(b, tensor.ValueDesc{
					Name: "len", DType: accel.U32, Shape: tensor.Shape{1}}),
				ScaleName: "scale", BaseName: "base",
			}
		},
	}, {
		name: "paged prefill",
		q:    tensor.Shape{3, qHeads, headDim},
		opts: func(b *tensor.Builder) tensor.AttentionOptions {
			return tensor.AttentionOptions{
				Lengths: tensor.Input(b, tensor.ValueDesc{
					Name: "len", DType: accel.U32, Shape: tensor.Shape{1}}),
				Pages: tensor.Input(b, tensor.ValueDesc{
					Name: "pages", DType: accel.U32, Shape: tensor.Shape{4}}),
				Block: block, ScaleName: "scale", BaseName: "base",
			}
		},
	}, {
		name: "batched decode",
		q:    tensor.Shape{2, 1, qHeads, headDim},
		opts: func(b *tensor.Builder) tensor.AttentionOptions {
			return tensor.AttentionOptions{
				Lengths: tensor.Input(b, tensor.ValueDesc{
					Name: "len", DType: accel.U32, Shape: tensor.Shape{2}}),
				Pages: tensor.Input(b, tensor.ValueDesc{
					Name: "pages", DType: accel.U32, Shape: tensor.Shape{2, 4}}),
				Block: block, ScaleName: "scale",
			}
		},
	}, {
		name: "ragged",
		q:    tensor.Shape{3, qHeads, headDim},
		opts: func(b *tensor.Builder) tensor.AttentionOptions {
			return tensor.AttentionOptions{
				Lengths: tensor.Input(b, tensor.ValueDesc{
					Name: "len", DType: accel.U32, Shape: tensor.Shape{2}}),
				Pages: tensor.Input(b, tensor.ValueDesc{
					Name: "pages", DType: accel.U32, Shape: tensor.Shape{2, 4}}),
				QueryExtents: tensor.Input(b, tensor.ValueDesc{
					Name: "ext", DType: accel.U32, Shape: tensor.Shape{2}}),
				Block: block, ScaleName: "scale",
			}
		},
	}}

	for _, sh := range shapes {
		for _, dt := range []accel.DType{accel.F32, accel.F16} {
			t.Run(sh.name+"/"+dt.String(), func(t *testing.T) {
				b := rt.NewBuilder("matrix")
				tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
				tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarU32})
				q := tensor.Input(b, tensor.ValueDesc{
					Name: "q", DType: accel.F32, Shape: sh.q})
				kc := tensor.NewState(b, tensor.StateDesc{
					Name: "kc", DType: dt,
					Shape: tensor.Shape{capacity, kvHeads, headDim}})
				vc := tensor.NewState(b, tensor.StateDesc{
					Name: "vc", DType: dt,
					Shape: tensor.Shape{capacity, kvHeads, headDim}})
				tensor.Output(b, "out", tensor.Attention(b, q, kc, vc, sh.opts(b)))

				plan, err := b.Compile(rt, tensor.CompileOptions{Label: "matrix"})
				if err != nil {
					// An unbuilt cell is allowed, and must say so in words a
					// caller can act on: which width, and that a kernel is what
					// is missing. A binding-size complaint is the failure this
					// test exists to forbid.
					msg := err.Error()
					if strings.Contains(msg, "binding") && strings.Contains(msg, "slot") {
						t.Fatalf("%s over an %v cache failed as a binding mismatch "+
							"rather than a named refusal, which is #25's shape:\n%v",
							sh.name, dt, err)
					}
					if !strings.Contains(msg, dt.String()) {
						t.Errorf("the refusal should name the cache width %v, got %v",
							dt, err)
					}
					t.Logf("not built, refused by name: %v", err)
					return
				}
				defer plan.Close()

				// The attention node, not the only node: a ragged step also
				// records the prefix sum its extents need.
				picked := ""
				for _, s := range plan.Selections() {
					if s.Op == "Attention" {
						picked = s.Kernel
					}
				}
				if picked == "" {
					t.Fatalf("no attention node in %+v", plan.Selections())
				}
				// A narrow cache must reach a kernel that reads a narrow cache.
				// This is the assertion #25 lacked: the plan compiled for f32
				// and would have compiled for f16 too if the widths had not
				// disagreed at bind time.
				narrow := strings.HasSuffix(picked, "F16")
				if want := dt == accel.F16; narrow != want {
					t.Fatalf("%s over an %v cache selected %q", sh.name, dt, picked)
				}
			})
		}
	}
}

// Selections reports how many tiles the block loop runs, not only the kernel.
//
// specs/044-unbounded-context.md §7 asks for "the kernel and the tile count",
// and §9 recorded it as not done: the reason string carried the cached-position
// count, which is the number a caller already had. The tile count is the one
// that grows with context, and it is what says whether a step is one pass or
// forty.
//
// Asserted at two lengths so the number is read rather than pattern-matched: a
// reason that hard-coded "1 tile" would pass a single-length test.
func TestSelectionsReportsTheTileCount(t *testing.T) {
	rt := newRuntime(t)

	// AttnBlock is the loop's width; these straddle it deliberately.
	for _, c := range []struct {
		cached int
		tiles  string
	}{
		{64, "1 tile(s)"},
		{300, "3 tile(s)"},
	} {
		b := rt.NewBuilder("tiles")
		tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
		q := tensor.Input(b, tensor.ValueDesc{
			Name: "q", DType: accel.F32, Shape: tensor.Shape{2, 8}})
		ln := tensor.Input(b, tensor.ValueDesc{
			Name: "len", DType: accel.U32, Shape: tensor.Shape{1}})
		kc := tensor.NewState(b, tensor.StateDesc{
			Name: "kc", DType: accel.F32, Shape: tensor.Shape{c.cached, 1, 8}})
		vc := tensor.NewState(b, tensor.StateDesc{
			Name: "vc", DType: accel.F32, Shape: tensor.Shape{c.cached, 1, 8}})
		tensor.Output(b, "out", tensor.Attention(b, q, kc, vc, tensor.AttentionOptions{
			Lengths: ln, ScaleName: "scale",
		}))

		plan, err := b.Compile(rt, tensor.CompileOptions{Label: "tiles"})
		if err != nil {
			t.Fatalf("compile at %d: %v", c.cached, err)
		}
		reason := ""
		for _, s := range plan.Selections() {
			if s.Op == "Attention" {
				reason = s.Reason
			}
		}
		plan.Close()

		if !strings.Contains(reason, c.tiles) {
			t.Errorf("a %d-position cache should report %q, got:\n%s",
				c.cached, c.tiles, reason)
		}
	}
}
