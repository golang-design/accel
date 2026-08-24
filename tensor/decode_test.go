// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"fmt"
	"math"
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
		// accel issue 10: this compiled, ignored Pages, and returned a
		// plausible wrong answer -- the first accepted-and-silently-wrong case
		// reported here rather than a refusal.
		name: "a page table on a prefill",
		build: func(b *tensor.Builder) {
			scalars(b)
			tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarU32})
			o := opts(b)
			o.Pages = u32(b, "pages", 2)
			o.Block = 2
			o.BaseName = "base"
			tensor.Attention(b, f32(b, "q", 2, 4, 8), cache(b, "k", 4, 2, 8),
				cache(b, "v", 4, 2, 8), o)
		},
		want: "only the decode kernels read a page table",
	}, {
		// The Block guard lived inside the decode switch, so a prefill reached
		// neither it nor the page table.
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
		name: "a layer that does not exist",
		build: func(b *tensor.Builder) {
			tensor.LayerState(b, cache(b, "c", 2, 8, 4), 5)
		},
		want: "layer 5",
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
