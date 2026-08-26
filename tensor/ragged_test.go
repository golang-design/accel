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

// raggedCase is one step's shape and contents, shared by the tests below.
type raggedCase struct {
	batch, qHeads, kvHeads, headDim int
	block, maxPages                 int
	counts                          []uint32
	lengths                         []uint32

	// pageBase shifts this case's page table so a single-sequence run can
	// address the same physical blocks a member of a larger batch used. Without
	// it a member run reads block 0 where the mixed run read block r*maxPages,
	// which are different caches and would make the comparison meaningless.
	pageBase int

	// capacityOf overrides the cache allocation, so a member run allocates the
	// cache the mixed run did and its shifted page ids stay in range.
	capacityOf int

	// qBase shifts where this case's queries are seeded from, so a member run
	// asks the same questions the mixed run asked for that sequence. Without it
	// a member seeds from flat index zero and the two runs would be compared on
	// different inputs -- which passes for sequence zero and says nothing about
	// the rest.
	qBase int
}

// capacity is the cache allocation, which a member run inherits from the batch
// it was a member of.
func (c raggedCase) capacity() int {
	if c.capacityOf != 0 {
		return c.capacityOf
	}
	return c.batch * c.maxPages * c.block
}

func (c raggedCase) tokens() int {
	n := 0
	for _, v := range c.counts {
		n += int(v)
	}
	return n
}

// runRaggedStep compiles and runs one ragged attention step.
//
// The caches are filled per sequence through its own page table, so a sequence
// reading another's blocks is visible as a wrong answer rather than as a crash.
func runRaggedStep(t *testing.T, c raggedCase) []float32 {
	t.Helper()
	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("ragged")

	tokens := c.tokens()
	tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})

	q := tensor.Input(b, tensor.ValueDesc{
		Name: "q", DType: accel.F32, Shape: tensor.Shape{tokens, c.qHeads, c.headDim},
	})
	extents := tensor.Input(b, tensor.ValueDesc{
		Name: "extents", DType: accel.U32, Shape: tensor.Shape{c.batch},
	})
	lengths := tensor.Input(b, tensor.ValueDesc{
		Name: "len", DType: accel.U32, Shape: tensor.Shape{c.batch},
	})
	pages := tensor.Input(b, tensor.ValueDesc{
		Name: "pages", DType: accel.U32, Shape: tensor.Shape{c.batch, c.maxPages},
	})

	capacity := c.capacity()
	kc := tensor.NewState(b, tensor.StateDesc{
		Name: "kcache", DType: accel.F32,
		Shape: tensor.Shape{capacity, c.kvHeads, c.headDim},
	})
	vc := tensor.NewState(b, tensor.StateDesc{
		Name: "vcache", DType: accel.F32,
		Shape: tensor.Shape{capacity, c.kvHeads, c.headDim},
	})

	tensor.Output(b, "out", tensor.Attention(b, q, kc, vc, tensor.AttentionOptions{
		Lengths: lengths, Pages: pages, Block: c.block,
		QueryExtents: extents, ScaleName: "scale",
	}))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "ragged"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	qs, ks, vs, pageTable := raggedInputs(c)
	out := f32Buffer(t, d, "out", make([]float32, tokens*c.qHeads*c.headDim))
	f := plan.Submit(d.Queue(), tensor.Bindings{
		Buffers: map[string]accel.BufferView{
			"q":       f32Buffer(t, d, "q", qs),
			"extents": u32Buffer(t, d, "extents", c.counts),
			"len":     u32Buffer(t, d, "len", c.lengths),
			"pages":   u32Buffer(t, d, "pages", pageTable),
			"kcache":  f32Buffer(t, d, "kcache", ks),
			"vcache":  f32Buffer(t, d, "vcache", vs),
			"out":     out,
		},
		Scalars: map[string]tensor.ScalarValue{
			"scale": tensor.F32(float32(1 / math.Sqrt(float64(c.headDim)))),
		},
	})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, tokens*c.qHeads*c.headDim)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}
	return got
}

// raggedInputs builds queries, caches and a page table for a case.
func raggedInputs(c raggedCase) (q, k, v []float32, pages []uint32) {
	tokens := c.tokens()
	q = make([]float32, tokens*c.qHeads*c.headDim)
	for i := range q {
		q[i] = float32(math.Sin(float64(c.qBase+i) * 0.41))
	}
	pages = make([]uint32, c.batch*c.maxPages)
	for i := range pages {
		pages[i] = uint32(c.pageBase + i)
	}
	capacity := c.capacity()
	k = make([]float32, capacity*c.kvHeads*c.headDim)
	v = make([]float32, len(k))
	for i := range k {
		k[i] = float32(math.Cos(float64(i) * 0.29))
		// Positive, so a softmax-weighted sum cannot cancel and a comparison
		// measures the step rather than the conditioning.
		v[i] = float32(i%11+1) / 8
	}
	return q, k, v, pages
}

// A ragged batch of one sequence equals the prefill it generalises.
//
// specs/046-segmented-extents.md §5's first assertion, and the accepting half
// of the whole spec: if the ragged path disagrees with the path it generalises,
// nothing else about it matters.
//
// Compared against a single-sequence paged prefill over the same cache, built
// through the ordinary operator rather than against a second reference — the
// question is whether the two operators agree, not whether either matches an
// oracle, which the corpus differential already answers.
func TestARaggedBatchOfOneEqualsAPrefill(t *testing.T) {
	c := raggedCase{
		batch: 1, qHeads: 2, kvHeads: 1, headDim: 8,
		block: 4, maxPages: 3,
		counts: []uint32{3}, lengths: []uint32{5},
	}
	ragged := runRaggedStep(t, c)

	// The same step as a prefill: BaseName is the position of its first token,
	// which is L-n = 2.
	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("prefill")
	tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
	tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarU32})

	tokens := c.tokens()
	q := tensor.Input(b, tensor.ValueDesc{
		Name: "q", DType: accel.F32, Shape: tensor.Shape{tokens, c.qHeads, c.headDim},
	})
	lengths := tensor.Input(b, tensor.ValueDesc{
		Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
	})
	pages := tensor.Input(b, tensor.ValueDesc{
		Name: "pages", DType: accel.U32, Shape: tensor.Shape{c.maxPages},
	})
	capacity := c.batch * c.maxPages * c.block
	kc := tensor.NewState(b, tensor.StateDesc{
		Name: "kcache", DType: accel.F32,
		Shape: tensor.Shape{capacity, c.kvHeads, c.headDim},
	})
	vc := tensor.NewState(b, tensor.StateDesc{
		Name: "vcache", DType: accel.F32,
		Shape: tensor.Shape{capacity, c.kvHeads, c.headDim},
	})
	tensor.Output(b, "out", tensor.Attention(b, q, kc, vc, tensor.AttentionOptions{
		Lengths: lengths, Pages: pages, Block: c.block,
		ScaleName: "scale", BaseName: "base",
	}))
	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "prefill"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	qs, ks, vs, pageTable := raggedInputs(c)
	out := f32Buffer(t, d, "out", make([]float32, tokens*c.qHeads*c.headDim))
	f := plan.Submit(d.Queue(), tensor.Bindings{
		Buffers: map[string]accel.BufferView{
			"q":      f32Buffer(t, d, "q", qs),
			"len":    u32Buffer(t, d, "len", c.lengths),
			"pages":  u32Buffer(t, d, "pages", pageTable),
			"kcache": f32Buffer(t, d, "kcache", ks),
			"vcache": f32Buffer(t, d, "vcache", vs),
			"out":    out,
		},
		Scalars: map[string]tensor.ScalarValue{
			"scale": tensor.F32(float32(1 / math.Sqrt(float64(c.headDim)))),
			"base":  tensor.U32(uint32(int(c.lengths[0]) - tokens)),
		},
	})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	want := make([]float32, tokens*c.qHeads*c.headDim)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, want); err != nil {
		t.Fatalf("readback: %v", err)
	}

	for i := range want {
		if math.Abs(float64(ragged[i]-want[i])) > 1e-5 {
			t.Fatalf("element %d: the ragged path gave %v and the prefill it generalises "+
				"gave %v", i, ragged[i], want[i])
		}
	}
}

// A ragged step's tokens match the same tokens run as separate steps.
//
// §5's second assertion. Running the members separately is the oracle, which is
// the shape 043 §8 used for the batch axis: a mixed step is right when each
// sequence's rows are what that sequence alone would have produced.
func TestAMixedStepMatchesItsMembersRunSeparately(t *testing.T) {
	mixed := raggedCase{
		batch: 3, qHeads: 2, kvHeads: 1, headDim: 8,
		block: 4, maxPages: 3,
		counts:  []uint32{3, 1, 2},
		lengths: []uint32{5, 2, 6},
	}
	got := runRaggedStep(t, mixed)
	width := mixed.qHeads * mixed.headDim

	base := 0
	for seq := range mixed.counts {
		n := int(mixed.counts[seq])
		// The member run addresses the *same physical blocks* the mixed run
		// gave this sequence, over a cache allocated to the same size and
		// filled the same way. Without that it would read block 0 where the
		// mixed run read block seq*maxPages, and the comparison would be
		// between two different caches.
		alone := raggedCase{
			batch: 1, qHeads: mixed.qHeads, kvHeads: mixed.kvHeads,
			headDim: mixed.headDim, block: mixed.block, maxPages: mixed.maxPages,
			counts: []uint32{mixed.counts[seq]}, lengths: []uint32{mixed.lengths[seq]},
			pageBase: seq * mixed.maxPages, capacityOf: mixed.capacity(),
			qBase: base,
		}
		want := runRaggedStep(t, alone)

		for i := range n * width {
			if math.Abs(float64(got[base+i]-want[i])) > 1e-5 {
				t.Fatalf("sequence %d token %d element %d: the mixed step gave %v and "+
					"the same sequence run alone gave %v", seq, i/width, i%width,
					got[base+i], want[i])
			}
		}
		base += n * width
	}

	if len(got) != mixed.tokens()*width {
		t.Fatalf("the flat output is %d elements for %d tokens", len(got), mixed.tokens())
	}
}

// A sequence contributing nothing occupies no rows and shifts none.
func TestARaggedStepAcceptsAZeroCount(t *testing.T) {
	c := raggedCase{
		batch: 3, qHeads: 2, kvHeads: 1, headDim: 8,
		block: 4, maxPages: 3,
		counts: []uint32{2, 0, 2}, lengths: []uint32{3, 4, 3},
	}
	got := runRaggedStep(t, c)
	if want := c.tokens() * c.qHeads * c.headDim; len(got) != want {
		t.Fatalf("the output is %d elements and four tokens is %d", len(got), want)
	}

	// The same two contributing sequences without the empty one between them
	// must produce the same rows: an off-by-one in the segment lookup shows up
	// here as a shifted batch rather than as a fault.
	without := raggedCase{
		batch: 2, qHeads: 2, kvHeads: 1, headDim: 8,
		block: 4, maxPages: 3,
		counts: []uint32{2, 2}, lengths: []uint32{3, 3},
	}
	// Only the first sequence's rows can be compared: the second sequence's
	// page table moves when the empty row is removed, so its cache is not the
	// same cache. The first row is where the lookup's boundary sits.
	ref := runRaggedStep(t, without)
	width := c.qHeads * c.headDim
	for i := range 2 * width {
		if math.Abs(float64(got[i]-ref[i])) > 1e-5 {
			t.Fatalf("token %d element %d is %v with an empty row after it and %v "+
				"without; an empty row shifted the rows before it",
				i/width, i%width, got[i], ref[i])
		}
	}
}

// The refusals 046 §4 states, each naming what it refused.
func TestARaggedStepRefusesWhatItCannotCheckOnTheDevice(t *testing.T) {
	base := raggedCase{
		batch: 2, qHeads: 2, kvHeads: 1, headDim: 8,
		block: 4, maxPages: 3,
		counts: []uint32{2, 1}, lengths: []uint32{3, 2},
	}
	for _, c := range []struct {
		name string
		mut  func(*tensor.AttentionOptions, *tensor.Tensor) *tensor.Tensor
		want string
	}{
		{"no pages", func(o *tensor.AttentionOptions, q *tensor.Tensor) *tensor.Tensor {
			o.Pages = nil
			return q
		}, "needs Pages"},
		{"a base as well", func(o *tensor.AttentionOptions, q *tensor.Tensor) *tensor.Tensor {
			o.BaseName = "base"
			return q
		}, "nothing reads"},
	} {
		t.Run(c.name, func(t *testing.T) {
			rt := newRuntime(t)
			b := rt.NewBuilder("refusal")
			tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
			tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarU32})
			q := tensor.Input(b, tensor.ValueDesc{
				Name: "q", DType: accel.F32,
				Shape: tensor.Shape{base.tokens(), base.qHeads, base.headDim},
			})
			opts := tensor.AttentionOptions{
				Lengths: tensor.Input(b, tensor.ValueDesc{
					Name: "len", DType: accel.U32, Shape: tensor.Shape{base.batch},
				}),
				Pages: tensor.Input(b, tensor.ValueDesc{
					Name: "pages", DType: accel.U32,
					Shape: tensor.Shape{base.batch, base.maxPages},
				}),
				Block: base.block,
				QueryExtents: tensor.Input(b, tensor.ValueDesc{
					Name: "extents", DType: accel.U32, Shape: tensor.Shape{base.batch},
				}),
				ScaleName: "scale",
			}
			q = c.mut(&opts, q)
			capacity := base.batch * base.maxPages * base.block
			kc := tensor.NewState(b, tensor.StateDesc{
				Name: "kcache", DType: accel.F32,
				Shape: tensor.Shape{capacity, base.kvHeads, base.headDim},
			})
			vc := tensor.NewState(b, tensor.StateDesc{
				Name: "vcache", DType: accel.F32,
				Shape: tensor.Shape{capacity, base.kvHeads, base.headDim},
			})
			tensor.Output(b, "out", tensor.Attention(b, q, kc, vc, opts))
			_, err := b.Compile(rt, tensor.CompileOptions{Label: "refusal"})
			if err == nil {
				t.Fatal("accepted")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Fatalf("refused with %q, which does not mention %q", err, c.want)
			}
		})
	}
}

// QueryExtents on a rank-2 or rank-4 q is refused.
//
// The flat form or nothing: rank 2 is one token and rank 4 is the rectangular
// batch whose sequences all contribute the same count.
func TestQueryExtentsNeedsTheFlatShape(t *testing.T) {
	for _, shape := range []tensor.Shape{{2, 8}, {2, 3, 2, 8}} {
		rt := newRuntime(t)
		b := rt.NewBuilder("shape")
		tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
		q := tensor.Input(b, tensor.ValueDesc{Name: "q", DType: accel.F32, Shape: shape})
		kc := tensor.NewState(b, tensor.StateDesc{
			Name: "kcache", DType: accel.F32, Shape: tensor.Shape{16, 1, 8},
		})
		vc := tensor.NewState(b, tensor.StateDesc{
			Name: "vcache", DType: accel.F32, Shape: tensor.Shape{16, 1, 8},
		})
		tensor.Output(b, "out", tensor.Attention(b, q, kc, vc, tensor.AttentionOptions{
			Lengths: tensor.Input(b, tensor.ValueDesc{
				Name: "len", DType: accel.U32, Shape: tensor.Shape{2},
			}),
			QueryExtents: tensor.Input(b, tensor.ValueDesc{
				Name: "extents", DType: accel.U32, Shape: tensor.Shape{2},
			}),
			ScaleName: "scale",
		}))
		_, err := b.Compile(rt, tensor.CompileOptions{Label: "shape"})
		if err == nil {
			t.Fatalf("q of %v with QueryExtents was accepted", shape)
		}
		if !strings.Contains(err.Error(), "flat") {
			t.Fatalf("q of %v refused with %q, which does not say the flat form is "+
				"what a ragged step takes", shape, err)
		}
	}
}

// A ragged plan and the prefill plan of the same rank are different plans.
//
// specs/046-segmented-extents.md §5, and the assertion §3's design rests on. A
// flat ragged q is rank 3 and so is a single-sequence prefill: the two are
// separated only by whether QueryExtents was supplied, which is the
// presence-of-a-field shape this project refused twice this milestone.
//
// §3 claims it is safe here because the two readings select different kernels,
// so 029's digest tells the plans apart structurally. That is a claim about the
// digest, so it is asserted through Identity rather than reasoned from what the
// hash covers. If it failed, a plan cache would serve a ragged plan for a
// prefill — and the tokens it returned would be plausible.
func TestARaggedPlanIsNotAPrefillPlan(t *testing.T) {
	const tokens, qHeads, kvHeads, headDim = 3, 2, 1, 8
	const block, maxPages, batch = 4, 3, 1

	build := func(ragged bool) tensor.Identity {
		rt := newRuntime(t)
		b := rt.NewBuilder("identity")
		tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
		tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarU32})
		q := tensor.Input(b, tensor.ValueDesc{
			Name: "q", DType: accel.F32, Shape: tensor.Shape{tokens, qHeads, headDim},
		})
		lengths := tensor.Input(b, tensor.ValueDesc{
			Name: "len", DType: accel.U32, Shape: tensor.Shape{batch},
		})
		capacity := batch * maxPages * block
		kc := tensor.NewState(b, tensor.StateDesc{
			Name: "kcache", DType: accel.F32,
			Shape: tensor.Shape{capacity, kvHeads, headDim},
		})
		vc := tensor.NewState(b, tensor.StateDesc{
			Name: "vcache", DType: accel.F32,
			Shape: tensor.Shape{capacity, kvHeads, headDim},
		})
		opts := tensor.AttentionOptions{
			Lengths: lengths, Block: block, ScaleName: "scale",
		}
		if ragged {
			// The page table a ragged step takes: one row per sequence.
			opts.Pages = tensor.Input(b, tensor.ValueDesc{
				Name: "pages", DType: accel.U32, Shape: tensor.Shape{batch, maxPages},
			})
			opts.QueryExtents = tensor.Input(b, tensor.ValueDesc{
				Name: "extents", DType: accel.U32, Shape: tensor.Shape{batch},
			})
		} else {
			opts.Pages = tensor.Input(b, tensor.ValueDesc{
				Name: "pages", DType: accel.U32, Shape: tensor.Shape{maxPages},
			})
			opts.BaseName = "base"
		}
		tensor.Output(b, "out", tensor.Attention(b, q, kc, vc, opts))
		return b.Identity()
	}

	if a, c := build(true), build(false); a == c {
		t.Fatalf("a ragged step and a prefill over the same rank-3 q have the same plan "+
			"identity %q; a plan cache would answer one with the other, and the tokens "+
			"it returned would be plausible", a)
	}

	// And the same reading twice is the same plan, or the assertion above holds
	// for any two graphs and says nothing about QueryExtents.
	if a, c := build(true), build(true); a != c {
		t.Fatalf("two identical ragged graphs have different identities %q and %q", a, c)
	}
}

// A ragged step runs over an f16 cache, and picks the kernel that reads one.
//
// accel issue 23: an f16 cache halves the largest allocation a serving process
// has after the weights, and a ragged step is the only way to express a batched
// prefill — so refusing the pair made a server give up one to have the other.
//
// Asserted through Selections rather than only by the graph compiling, because
// a graph that silently used the f32 kernel over a narrow cache would read the
// halves as full floats and produce a plausible wrong picture.
func TestARaggedStepRunsOverAnF16Cache(t *testing.T) {
	c := raggedCase{
		batch: 2, qHeads: 2, kvHeads: 1, headDim: 8,
		block: 4, maxPages: 3,
		counts: []uint32{2, 1}, lengths: []uint32{3, 2},
	}
	rt := newRuntime(t)
	b := rt.NewBuilder("f16ragged")

	tokens := c.tokens()
	tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
	q := tensor.Input(b, tensor.ValueDesc{
		Name: "q", DType: accel.F32, Shape: tensor.Shape{tokens, c.qHeads, c.headDim},
	})
	extents := tensor.Input(b, tensor.ValueDesc{
		Name: "extents", DType: accel.U32, Shape: tensor.Shape{c.batch},
	})
	lengths := tensor.Input(b, tensor.ValueDesc{
		Name: "len", DType: accel.U32, Shape: tensor.Shape{c.batch},
	})
	pages := tensor.Input(b, tensor.ValueDesc{
		Name: "pages", DType: accel.U32, Shape: tensor.Shape{c.batch, c.maxPages},
	})
	capacity := c.capacity()
	kc := tensor.NewState(b, tensor.StateDesc{
		Name: "kcache", DType: accel.F16,
		Shape: tensor.Shape{capacity, c.kvHeads, c.headDim},
	})
	vc := tensor.NewState(b, tensor.StateDesc{
		Name: "vcache", DType: accel.F16,
		Shape: tensor.Shape{capacity, c.kvHeads, c.headDim},
	})
	tensor.Output(b, "out", tensor.Attention(b, q, kc, vc, tensor.AttentionOptions{
		Lengths: lengths, Pages: pages, Block: c.block,
		QueryExtents: extents, ScaleName: "scale",
	}))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "f16ragged"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	var picked string
	for _, s := range plan.Selections() {
		if s.Op == "Attention" {
			picked = s.Kernel
		}
	}
	if picked != "AttentionRaggedF16" {
		t.Fatalf("a ragged step over an f16 cache selected %q; the f32 kernel would "+
			"read the halves as full floats and draw a plausible wrong picture", picked)
	}
}
