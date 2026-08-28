// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/direct"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
	"golang.design/x/accel/kernelabi"
)

// The offsets are the exclusive prefix sum, and a zero count is legal.
//
// specs/046-segmented-extents.md §1. The last entry is the total, which is what
// lets a caller be refused for a flat buffer whose length disagrees with the
// counts — so it is asserted rather than left as an artefact of the loop.
func TestSegmentOffsetsArePrefixSums(t *testing.T) {
	for _, c := range []struct {
		name   string
		counts []uint32
		want   []uint32
	}{
		{"a mixed step", []uint32{512, 1, 1, 1}, []uint32{0, 512, 513, 514, 515}},
		{"one row", []uint32{7}, []uint32{0, 7}},
		// A row contributing nothing is an ordinary member: its offsets repeat,
		// so it contains no token and nothing below needs a case for it.
		{"a zero between two rows", []uint32{3, 0, 4}, []uint32{0, 3, 3, 7}},
		{"every row zero", []uint32{0, 0}, []uint32{0, 0, 0}},
	} {
		t.Run(c.name, func(t *testing.T) {
			rows := len(c.counts)
			got := make([]uint32, rows+1)
			if err := direct.Run(&testkernels.SegmentOffsetsKernel, accel.ID3{X: 1},
				kernelabi.Args{
					Uniforms: []any{testkernels.SegmentDims{Rows: uint32(rows)}},
					Slices:   []any{c.counts, got},
				}); err != nil {
				t.Fatalf("run: %v", err)
			}
			for i := range c.want {
				if got[i] != c.want[i] {
					t.Fatalf("offsets are %v, want %v", got, c.want)
				}
			}
		})
	}
}

// raggedRef is an independent reference for one ragged step.
//
// Written from specs/046-segmented-extents.md §2.2 rather than from the kernel:
// it computes each token's position as L-n+i and attends over 0..position, in
// f64, with no blocking. A reference that mirrored the kernel's block loop
// would agree with a kernel whose blocking was wrong.
func raggedRef(t *testing.T, d testkernels.RaggedDims, q, k, v []float32,
	pages, lengths, offsets []uint32) []float32 {

	t.Helper()
	total := int(offsets[d.Batch])
	out := make([]float32, total*int(d.QHeads)*int(d.HeadDim))
	group := d.QHeads / d.KVHeads

	for seq := uint32(0); seq < d.Batch; seq++ {
		n := offsets[seq+1] - offsets[seq]
		for i := uint32(0); i < n; i++ {
			tok := offsets[seq] + i
			pos := lengths[seq] - n + i
			for h := uint32(0); h < d.QHeads; h++ {
				kvHead := h / group
				scores := make([]float64, pos+1)
				maxScore := math.Inf(-1)
				for p := uint32(0); p <= pos; p++ {
					phys := pages[seq*d.MaxPages+p/d.Block]*d.Block + p%d.Block
					dot := 0.0
					for e := uint32(0); e < d.HeadDim; e++ {
						qi := float64(q[(tok*d.QHeads+h)*d.HeadDim+e])
						ki := float64(k[phys*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+e])
						dot += qi * ki
					}
					scores[p] = dot * float64(d.Scale)
					maxScore = math.Max(maxScore, scores[p])
				}
				sum := 0.0
				for p := range scores {
					scores[p] = math.Exp(scores[p] - maxScore)
					sum += scores[p]
				}
				for e := uint32(0); e < d.HeadDim; e++ {
					acc := 0.0
					for p := uint32(0); p <= pos; p++ {
						phys := pages[seq*d.MaxPages+p/d.Block]*d.Block + p%d.Block
						acc += scores[p] * float64(v[phys*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+e])
					}
					out[(tok*d.QHeads+h)*d.HeadDim+e] = float32(acc / sum)
				}
			}
		}
	}
	return out
}

// raggedFixture builds a cache and queries for one ragged step.
func raggedFixture(d testkernels.RaggedDims, counts []uint32) (q, k, v []float32,
	pages, lengths, offsets []uint32) {

	offsets = make([]uint32, d.Batch+1)
	sum := uint32(0)
	for r, c := range counts {
		offsets[r] = sum
		sum += c
	}
	offsets[d.Batch] = sum

	// Each sequence's cache holds its own tokens, the last n of which are this
	// step's. A page table per sequence, blocks handed out in order.
	lengths = make([]uint32, d.Batch)
	for r := range counts {
		lengths[r] = counts[r] + uint32(r) + 1 // some prior occupancy, varying
	}
	pages = make([]uint32, d.Batch*d.MaxPages)
	next := uint32(0)
	for r := uint32(0); r < d.Batch; r++ {
		for p := uint32(0); p < d.MaxPages; p++ {
			pages[r*d.MaxPages+p] = next
			next++
		}
	}
	blocks := next
	k = make([]float32, blocks*d.Block*d.KVHeads*d.HeadDim)
	v = make([]float32, len(k))
	for i := range k {
		k[i] = float32(math.Cos(float64(i) * 0.37))
		v[i] = float32(math.Sin(float64(i)*0.19)) * 2
	}
	q = make([]float32, sum*d.QHeads*d.HeadDim)
	for i := range q {
		q[i] = float32(math.Sin(float64(i) * 0.53))
	}
	return q, k, v, pages, lengths, offsets
}

func runRagged(t *testing.T, d testkernels.RaggedDims, q, k, v []float32,
	pages, lengths, offsets []uint32) []float32 {

	t.Helper()
	total := offsets[d.Batch]
	out := make([]float32, total*d.QHeads*d.HeadDim)
	// Cooperative: the kernel holds barriers across its block loop, so its
	// invocations rendezvous and running them in sequence would be a different
	// program. Same entry point every attention kernel here uses.
	if err := kernel.DispatchCooperative(&testkernels.AttentionRaggedKernel,
		accel.ID3{X: total * d.QHeads},
		kernelabi.Args{
			Uniforms: []any{d},
			Slices:   []any{q, k, v, pages, lengths, offsets, out},
		}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	return out
}

// A ragged step matches a reference written from the spec's arithmetic.
//
// The mixed shape is the point: one sequence contributing several tokens beside
// sequences contributing one, which is what a prefill chunk sharing a dispatch
// with decodes looks like.
func TestARaggedStepMatchesItsReference(t *testing.T) {
	d := testkernels.RaggedDims{
		Batch: 3, QHeads: 2, KVHeads: 1, HeadDim: 8,
		Block: 4, MaxPages: 3, Scale: float32(1 / math.Sqrt(8)),
	}
	counts := []uint32{3, 1, 2}
	q, k, v, pages, lengths, offsets := raggedFixture(d, counts)

	got := runRagged(t, d, q, k, v, pages, lengths, offsets)
	want := raggedRef(t, d, q, k, v, pages, lengths, offsets)

	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			tok := i / int(d.QHeads*d.HeadDim)
			t.Fatalf("element %d (token %d) is %v, want %v", i, tok, got[i], want[i])
		}
	}
}

// A sequence contributing no tokens disturbs nothing around it.
//
// specs/046-segmented-extents.md §1 makes a zero count legal rather than an
// error, and a row of zero between two non-empty ones is where an off-by-one in
// the segment lookup shows up as a shifted batch.
func TestARowContributingNothingIsSkipped(t *testing.T) {
	d := testkernels.RaggedDims{
		Batch: 4, QHeads: 2, KVHeads: 1, HeadDim: 8,
		Block: 4, MaxPages: 3, Scale: float32(1 / math.Sqrt(8)),
	}
	counts := []uint32{2, 0, 3, 0}
	q, k, v, pages, lengths, offsets := raggedFixture(d, counts)

	got := runRagged(t, d, q, k, v, pages, lengths, offsets)
	want := raggedRef(t, d, q, k, v, pages, lengths, offsets)
	if len(got) != int(offsets[d.Batch])*int(d.QHeads)*int(d.HeadDim) {
		t.Fatalf("the flat output is %d elements for %d tokens", len(got), offsets[d.Batch])
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-5 {
			t.Fatalf("element %d is %v, want %v: a row contributing nothing shifted "+
				"the rows after it", i, got[i], want[i])
		}
	}
}

// A token attends its own position and not the one after it.
//
// The leak causal masking exists to prevent, and it is invisible in a smooth
// distribution: the cache position immediately after a token holds a value that
// changes the output if it is read, so the reference and the kernel disagree
// only if the bound is off by one.
func TestARaggedTokenDoesNotSeeThePositionAfterIt(t *testing.T) {
	d := testkernels.RaggedDims{
		Batch: 1, QHeads: 1, KVHeads: 1, HeadDim: 8,
		Block: 4, MaxPages: 3, Scale: float32(1 / math.Sqrt(8)),
	}
	counts := []uint32{2}
	q, k, v, pages, lengths, offsets := raggedFixture(d, counts)

	// The baseline, before anything is made loud.
	quiet := runRagged(t, d, q, k, v, pages, lengths, offsets)

	// Now make the *last* cached position loud. It is token 1's own position
	// and the position immediately after token 0's, so a correct bound lets
	// token 1 read it and stops token 0.
	last := lengths[0] - 1
	phys := pages[last/d.Block]*d.Block + last%d.Block
	for e := uint32(0); e < d.HeadDim; e++ {
		v[phys*d.KVHeads*d.HeadDim+e] = 1000
	}
	loud := runRagged(t, d, q, k, v, pages, lengths, offsets)

	// Comparing two runs rather than testing a magnitude: how much the loud
	// value moves an output depends on its softmax weight, which is a property
	// of the fixture rather than of the bound. Whether it moves at all is not.
	w := int(d.QHeads * d.HeadDim)
	for e := range w {
		if loud[e] != quiet[e] {
			t.Fatalf("token 0's output element %d moved from %v to %v when the position "+
				"after it was changed: it attends past its own position, which is the "+
				"causal leak", e, quiet[e], loud[e])
		}
	}
	moved := false
	for e := range w {
		if loud[w+e] != quiet[w+e] {
			moved = true
		}
	}
	if !moved {
		t.Fatal("token 1's output did not move when its own last cached position was " +
			"changed, so the fixture does not reach the bound it claims to test and " +
			"the assertion above is about nothing")
	}

	// And the kernel still matches the reference with the loud value in place,
	// which is what says both tokens took the right bound rather than both
	// taking a wrong one.
	want := raggedRef(t, d, q, k, v, pages, lengths, offsets)
	for i := range want {
		if math.Abs(float64(loud[i]-want[i])) > 1e-3 {
			t.Fatalf("element %d is %v, want %v", i, loud[i], want[i])
		}
	}
}

// The authored ragged kernel and its generated lowering agree.
//
// specs/010-kernel-corpus.md §6's per-kernel obligation. Every other test here
// runs the generated form, which is what a device executes, so nothing else
// calls the authored function -- and a function nothing calls is a function
// that can drift from the lowering it is the source of.
//
// It also matters where the differential does not run. On darwin the CPU/Metal
// comparison exercises this kernel; on Linux nothing does, which is how the
// package's coverage fell below its gate on one platform and not the other.
func TestTheAuthoredRaggedKernelMatchesItsLowering(t *testing.T) {
	d := testkernels.RaggedDims{
		Batch: 3, QHeads: 2, KVHeads: 1, HeadDim: 8,
		Block: 4, MaxPages: 3, Scale: float32(1 / math.Sqrt(8)),
	}
	counts := []uint32{3, 1, 2}
	q, k, v, pages, lengths, offsets := raggedFixture(d, counts)

	total := offsets[d.Batch]
	groups := kernel.ID3{X: total * d.QHeads, Y: 1, Z: 1}

	authored := make([]float32, total*d.QHeads*d.HeadDim)
	for g := range groups.X {
		var scores, red [128]float32
		kernel.RunAuthored(&testkernels.AttentionRaggedKernel, kernel.ID3{X: g},
			groups, 128, func(th kernel.Thread) {
				testkernels.AttentionRagged(th, d, q, k, v, pages, lengths,
					offsets, authored, &scores, &red)
			})
	}

	generated := runRagged(t, d, q, k, v, pages, lengths, offsets)

	// Within a bound rather than exact, and the reason is specs/006-backends.md
	// §5's rounding obligation rather than laxity. The generated form spells the
	// dot product as float32(dot + float32(a*b)) -- an explicit rounding of the
	// product, which is what stops a backend fusing the multiply and the add.
	// The authored form is ordinary Go, where the compiler may fuse on a target
	// that has FMA. So the two differ by an ulp on arm64 and agree exactly on a
	// target without it, and an exact comparison here would pass on one machine
	// and fail on the other. Every authored-versus-generated comparison over
	// f32 in this package carries the same bound for the same reason.
	for i := range authored {
		if math.Abs(float64(authored[i]-generated[i])) > 1e-5 {
			t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
		}
	}
}

// The authored prefix sum and its generated lowering agree, for the same reason.
func TestTheAuthoredSegmentOffsetsMatchesItsLowering(t *testing.T) {
	counts := []uint32{3, 0, 4, 1}
	dims := testkernels.SegmentDims{Rows: uint32(len(counts))}

	authored := make([]uint32, len(counts)+1)
	kernel.RunAuthored(&testkernels.SegmentOffsetsKernel, kernel.ID3{X: 0},
		kernel.ID3{X: 1, Y: 1, Z: 1}, 1, func(th kernel.Thread) {
			testkernels.SegmentOffsets(th, dims, counts, authored)
		})

	generated := make([]uint32, len(counts)+1)
	if err := direct.Run(&testkernels.SegmentOffsetsKernel, accel.ID3{X: 1},
		kernelabi.Args{Uniforms: []any{dims}, Slices: []any{counts, generated}}); err != nil {
		t.Fatalf("run: %v", err)
	}
	for i := range authored {
		if authored[i] != generated[i] {
			t.Fatalf("offset %d: authored %d, generated %d", i, authored[i], generated[i])
		}
	}
}

// The f16 ragged kernel equals the f32 one on values f16 holds exactly.
//
// accel issue 23. The two kernels differ in three lines -- the cache bindings
// and the two loads that widen -- so what is worth asserting is that they agree
// where they should, and the way to make that meaningful is to feed both a
// cache f16 represents without rounding. Then a difference is the *lowering*
// rather than the storage, which is the only thing this variant can get wrong.
func TestTheF16RaggedStepEqualsTheF32One(t *testing.T) {
	d := testkernels.RaggedDims{
		Batch: 3, QHeads: 2, KVHeads: 1, HeadDim: 8,
		Block: 4, MaxPages: 3, Scale: float32(1 / math.Sqrt(8)),
	}
	counts := []uint32{3, 1, 2}
	q, k, v, pages, lengths, offsets := raggedFixture(d, counts)

	// Small integers over a power of two: exact in f16, so the halves carry
	// what was intended rather than what rounding produced.
	for i := range k {
		k[i] = float32(i%13-6) / 4
		v[i] = float32(i%11-5) / 2
	}

	wide := runRagged(t, d, q, k, v, pages, lengths, offsets)

	narrowK := make([]accel.Float16, len(k))
	narrowV := make([]accel.Float16, len(v))
	for i := range k {
		narrowK[i] = accel.ToFloat16(k[i])
		narrowV[i] = accel.ToFloat16(v[i])
		if narrowK[i].F32() != k[i] || narrowV[i].F32() != v[i] {
			t.Fatalf("element %d does not survive f16, so this test would be about "+
				"rounding rather than about the lowering", i)
		}
	}

	total := offsets[d.Batch]
	narrow := make([]float32, total*d.QHeads*d.HeadDim)
	if err := kernel.DispatchCooperative(&testkernels.AttentionRaggedF16Kernel,
		accel.ID3{X: total * d.QHeads},
		kernelabi.Args{
			Uniforms: []any{d},
			Slices:   []any{q, narrowK, narrowV, pages, lengths, offsets, narrow},
		}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	for i := range wide {
		if narrow[i] != wide[i] {
			t.Fatalf("element %d: the f16 cache gave %v and the f32 cache %v over a "+
				"cache f16 holds exactly, so the two lowerings disagree",
				i, narrow[i], wide[i])
		}
	}
}

// The authored f16 ragged kernel and its generated lowering agree.
//
// specs/010-kernel-corpus.md §6's obligation, and the reason it is not implied
// by the test above: that one runs the *generated* form on both sides of the
// f16/f32 comparison, so the authored function it was derived from is called by
// nothing. On darwin the Metal differential would reach it; on Linux nothing
// does, which is how this package's coverage gate failed twice this milestone.
func TestTheAuthoredF16RaggedKernelMatchesItsLowering(t *testing.T) {
	d := testkernels.RaggedDims{
		Batch: 2, QHeads: 2, KVHeads: 1, HeadDim: 8,
		Block: 4, MaxPages: 3, Scale: float32(1 / math.Sqrt(8)),
	}
	counts := []uint32{2, 1}
	q, k, v, pages, lengths, offsets := raggedFixture(d, counts)

	narrowK := make([]accel.Float16, len(k))
	narrowV := make([]accel.Float16, len(v))
	for i := range k {
		narrowK[i] = accel.ToFloat16(k[i])
		narrowV[i] = accel.ToFloat16(v[i])
	}

	total := offsets[d.Batch]
	groups := kernel.ID3{X: total * d.QHeads, Y: 1, Z: 1}
	authored := make([]float32, total*d.QHeads*d.HeadDim)
	for g := range groups.X {
		var scores, red [128]float32
		kernel.RunAuthored(&testkernels.AttentionRaggedF16Kernel, kernel.ID3{X: g},
			groups, 128, func(th kernel.Thread) {
				testkernels.AttentionRaggedF16(th, d, q, narrowK, narrowV, pages,
					lengths, offsets, authored, &scores, &red)
			})
	}

	generated := make([]float32, total*d.QHeads*d.HeadDim)
	if err := kernel.DispatchCooperative(&testkernels.AttentionRaggedF16Kernel,
		accel.ID3{X: total * d.QHeads},
		kernelabi.Args{
			Uniforms: []any{d},
			Slices:   []any{q, narrowK, narrowV, pages, lengths, offsets, generated},
		}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	for i := range authored {
		if math.Abs(float64(authored[i]-generated[i])) > 1e-5 {
			t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
		}
	}
}

// shortenOffsets rebuilds an offsets array from counts that sum to less than
// the query buffer holds, which is the shape a padded batch has.
func shortenOffsets(batch uint32, counts []uint32) []uint32 {
	off := make([]uint32, batch+1)
	sum := uint32(0)
	for r, c := range counts {
		off[r] = sum
		sum += c
	}
	off[batch] = sum
	return off
}

// A token past the last segment is padding: it scores nothing, writes zero, and
// reads nothing out of bounds.
//
// The bug this pins ([#24](https://github.com/golang-design/accel/issues/24)):
// the segment lookup counts the rows that end at or before a token, so a token
// past every row counted every row and produced seq == Batch. That is one past
// the end of offsets, of lengths, and of the page table's rows -- a panic on the
// CPU backend and, on a GPU, another sequence's cache read as this one's.
//
// specs/046-segmented-extents.md §1 property 3 is why the check is here and not
// in tensor.Attention: the sum of a device count buffer is not a value the host
// can see at record time, so the kernel is the only place that can enforce it.
func TestARaggedTokenPastTheLastSegmentIsPadding(t *testing.T) {
	d := testkernels.RaggedDims{
		Batch: 3, QHeads: 2, KVHeads: 1, HeadDim: 8,
		Block: 4, MaxPages: 3, Scale: float32(1 / math.Sqrt(8)),
	}
	// q holds four rows; the extents claim three. The fourth is padding.
	q, k, v, pages, lengths, _ := raggedFixture(d, []uint32{3, 1, 0})
	short := shortenOffsets(d.Batch, []uint32{2, 1, 0})

	const rows = 4
	if got := int(short[d.Batch]); got != rows-1 {
		t.Fatalf("the fixture is not padded: extents sum to %d and q has %d rows",
			got, rows)
	}

	out := make([]float32, rows*d.QHeads*d.HeadDim)
	if err := kernel.DispatchCooperative(&testkernels.AttentionRaggedKernel,
		accel.ID3{X: rows * d.QHeads},
		kernelabi.Args{
			Uniforms: []any{d},
			Slices:   []any{q, k, v, pages, lengths, short, out},
		}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	// The real rows are unaffected: padding must not perturb the tokens that do
	// belong to a sequence. The reference runs over the same short extents.
	want := raggedRef(t, d, q, k, v, pages, lengths, short)
	for i := range want {
		if math.Abs(float64(out[i]-want[i])) > 1e-5 {
			t.Fatalf("padding changed a real token: element %d is %v, want %v",
				i, out[i], want[i])
		}
	}

	// The pad row is zero. Asserted rather than left untouched, because a test
	// that only checks the dispatch returned would pass against a buffer that
	// happened to hold anything.
	for i := len(want); i < len(out); i++ {
		if out[i] != 0 {
			t.Fatalf("the pad row is not zero: element %d is %v", i, out[i])
		}
	}

	// The authored form takes the same branch, which is what the Linux gate
	// sees: the differential against the lowering only runs where Metal does.
	authored := make([]float32, len(out))
	groups := kernel.ID3{X: rows * d.QHeads, Y: 1, Z: 1}
	for g := range groups.X {
		var scores, red [128]float32
		kernel.RunAuthored(&testkernels.AttentionRaggedKernel, kernel.ID3{X: g},
			groups, 128, func(th kernel.Thread) {
				testkernels.AttentionRagged(th, d, q, k, v, pages, lengths,
					short, authored, &scores, &red)
			})
	}
	for i := range authored {
		if math.Abs(float64(authored[i]-out[i])) > 1e-5 {
			t.Fatalf("element %d: authored %v, generated %v", i, authored[i], out[i])
		}
	}
}

// The f16 cache variant guards the same way, and is checked because it is a
// separate kernel rather than a flag: a fix applied to one and not the other is
// exactly the divergence a mechanically derived variant invites.
func TestAnF16RaggedTokenPastTheLastSegmentIsPadding(t *testing.T) {
	d := testkernels.RaggedDims{
		Batch: 2, QHeads: 2, KVHeads: 1, HeadDim: 8,
		Block: 4, MaxPages: 3, Scale: float32(1 / math.Sqrt(8)),
	}
	q, k32, v32, pages, lengths, _ := raggedFixture(d, []uint32{2, 1})
	short := shortenOffsets(d.Batch, []uint32{1, 1})

	k16 := make([]accel.Float16, len(k32))
	v16 := make([]accel.Float16, len(v32))
	for i := range k32 {
		k16[i] = accel.ToFloat16(k32[i])
		v16[i] = accel.ToFloat16(v32[i])
	}

	const rows = 3
	out := make([]float32, rows*d.QHeads*d.HeadDim)
	if err := kernel.DispatchCooperative(&testkernels.AttentionRaggedF16Kernel,
		accel.ID3{X: rows * d.QHeads},
		kernelabi.Args{
			Uniforms: []any{d},
			Slices:   []any{q, k16, v16, pages, lengths, short, out},
		}); err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	real := int(short[d.Batch]) * int(d.QHeads) * int(d.HeadDim)
	for i := real; i < len(out); i++ {
		if out[i] != 0 {
			t.Fatalf("the pad row is not zero: element %d is %v", i, out[i])
		}
	}
	// A real row still attended: a guard that returned for every token would
	// pass the zero check above and nothing else.
	nonzero := false
	for i := range real {
		if out[i] != 0 {
			nonzero = true
		}
	}
	if !nonzero {
		t.Fatal("every real row is zero, so the guard is firing for tokens that " +
			"belong to a sequence")
	}

	authored := make([]float32, len(out))
	groups := kernel.ID3{X: rows * d.QHeads, Y: 1, Z: 1}
	for g := range groups.X {
		var scores, red [128]float32
		kernel.RunAuthored(&testkernels.AttentionRaggedF16Kernel, kernel.ID3{X: g},
			groups, 128, func(th kernel.Thread) {
				testkernels.AttentionRaggedF16(th, d, q, k16, v16, pages, lengths,
					short, authored, &scores, &red)
			})
	}
	for i := range authored {
		if math.Abs(float64(authored[i]-out[i])) > 1e-5 {
			t.Fatalf("element %d: authored %v, generated %v", i, authored[i], out[i])
		}
	}
}
