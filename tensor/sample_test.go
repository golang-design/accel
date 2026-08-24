// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"math"
	"sort"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// The host oracles below are written from the definitions in
// specs/028-sampling.md, not from the kernels. That is the point of them: a
// reimplementation of the kernel would agree with the kernel about a shared
// mistake, and the mistakes worth catching here -- a tie going the other way, a
// walk stopping one entry early, a mask keeping the largest set below p rather
// than the smallest set reaching it -- are exactly the ones both would make.

// hostArgmax is "the largest value, and the lowest index among equals".
func hostArgmax(row []float32) uint32 {
	best, at := row[0], 0
	for i, v := range row {
		if v > best {
			best, at = v, i
		}
	}
	return uint32(at)
}

// hostCategorical walks the row in index order and takes the first index whose
// running sum exceeds the draw scaled by the row's own total, or the last index
// if none does.
func hostCategorical(row []float32, draw float32) uint32 {
	if draw < 0 {
		draw = 0
	}
	if draw > 0.99999994 {
		draw = 0.99999994
	}
	var total float32
	for _, v := range row {
		total += v
	}
	target := draw * total
	var acc float32
	for i, v := range row {
		acc += v
		if acc > target {
			return uint32(i)
		}
	}
	return uint32(len(row) - 1)
}

// descending orders indices by (value descending, index ascending), which is
// the lexicographic comparison specs/028-sampling.md fixes so that an entry
// ties only with itself.
func descending(row []float32) []int {
	idx := make([]int, len(row))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		if row[idx[a]] != row[idx[b]] {
			return row[idx[a]] > row[idx[b]]
		}
		return idx[a] < idx[b]
	})
	return idx
}

// hostTopK keeps the k largest and zeroes the rest.
func hostTopK(row []float32, k int) []float32 {
	out := make([]float32, len(row))
	for _, i := range descending(row)[:k] {
		out[i] = row[i]
	}
	return out
}

// hostTopP keeps the smallest set of largest entries whose mass reaches p of
// the row's own total -- including the entry that crosses, which is what makes
// the set the smallest one *reaching* p rather than the largest one below it.
func hostTopP(row []float32, p float32) []float32 {
	var total float32
	for _, v := range row {
		total += v
	}
	target := total * p
	out := make([]float32, len(row))
	var acc float32
	for _, i := range descending(row) {
		if acc >= target {
			break
		}
		out[i] = row[i]
		acc += row[i]
	}
	return out
}

// A batch of logits is one buffer of rows*vocab, which is what every operator
// here reads.
func flatten(rows [][]float32) []float32 {
	out := make([]float32, 0, len(rows)*len(rows[0]))
	for _, r := range rows {
		out = append(out, r...)
	}
	return out
}

// Argmax names the largest logit in every row, and the lowest of tied maxima.
//
// The tie rule is the substance and the batching is the shape. A row of distinct
// values would pass for an implementation with no tie rule at all, and a
// single-row case would pass for one whose rows all read row zero -- so every
// row here has a plateau, and the rows disagree about where it is.
func TestArgmaxNamesTheLowestTiedMaximumInEachRow(t *testing.T) {
	const vocab = 300
	rows := [][]float32{make([]float32, vocab), make([]float32, vocab), make([]float32, vocab)}
	for r, row := range rows {
		for i := range row {
			row[i] = float32(math.Sin(float64(i)*0.21 + float64(r)))
		}
	}
	// Equal maxima spread so the pairs that meet them form at different depths
	// of the reduction tree: an implementation whose tie rule only held between
	// adjacent lanes passes a plateau and fails this.
	rows[0][19], rows[0][240] = 5, 5
	rows[1][7], rows[1][8], rows[1][201] = 4, 4, 4
	rows[2][299], rows[2][0] = 9, 9

	rt := newRuntime(t)
	b := rt.NewBuilder("argmax")
	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F32, Shape: tensor.Shape{len(rows), vocab},
	})
	tensor.Output(b, "token", tensor.Argmax(b, x))
	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "argmax"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	// The selection is asserted, not just the value: specs/007-tensor-layer.md
	// makes which kernel ran something a caller can read, and a graph that
	// compiled says nothing about which one it picked.
	sel := plan.Selections()
	if len(sel) != 1 || sel[0].Op != "Argmax" || sel[0].Kernel != "SampleArgmax" {
		t.Fatalf("selections are %+v, want one Argmax on SampleArgmax", sel)
	}
	if !strings.Contains(sel[0].Reason, "lowest index") {
		t.Errorf("the reason does not state the tie rule, which is the one thing a "+
			"caller cannot infer from the result: %q", sel[0].Reason)
	}

	d := rt.Device()
	out := u32Buffer(t, d, "token", make([]uint32, len(rows)))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f32Buffer(t, d, "x", flatten(rows)), "token": out,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]uint32, len(rows))
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}

	// Against the oracle *and* against a number written here, so a row reading
	// another row's data cannot agree with an oracle that read the same wrong
	// row.
	want := []uint32{19, 7, 0}
	for r := range rows {
		if o := hostArgmax(rows[r]); o != want[r] {
			t.Fatalf("the oracle says row %d is %d and this test says %d", r, o, want[r])
		}
		if got[r] != want[r] {
			t.Errorf("row %d sampled %d, want %d", r, got[r], want[r])
		}
	}
}

// The categorical walk takes each row against that row's own draw.
//
// Two claims in one graph. The first is exactness: with a known draw the chosen
// index is determined, so this asserts the token rather than a distribution, and
// it asserts it at the cumulative boundaries where an off-by-one lives -- a walk
// comparing with >= instead of > shifts every answer by one for exactly the
// draws that land on a partial sum.
//
// The second is independence, which is specs/043-per-row-values.md section 8's
// assertion: rows 4 and 5 hold the *same* distribution and different draws, and
// must emit different tokens. A sampler taking one draw per dispatch stays
// perfectly reproducible and fails this, which is why reproducibility tests
// alone cannot see it.
func TestCategoricalDrawsEachRowAgainstItsOwnUniform(t *testing.T) {
	// Cumulative: 0.1, 0.3, 0.6, 1.0.
	dist := []float32{0.1, 0.2, 0.3, 0.4}
	draws := []float32{
		0,     // the very bottom
		0.1,   // exactly the first boundary: > means it moves on
		0.6,   // the third boundary
		0.999, // the top
		0.05,  // the same distribution as the next row...
		0.95,  // ...and a different draw
	}
	want := []uint32{0, 1, 3, 3, 0, 3}

	rows := make([][]float32, len(draws))
	for i := range rows {
		rows[i] = dist
	}

	rt := newRuntime(t)
	b := rt.NewBuilder("categorical")
	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F32, Shape: tensor.Shape{len(rows), len(dist)},
	})
	u := tensor.Input(b, tensor.ValueDesc{
		Name: "u", DType: accel.F32, Shape: tensor.Shape{len(draws)},
	})
	tensor.Output(b, "token", tensor.SampleCategorical(b, x, u))
	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "categorical"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	sel := plan.Selections()
	if len(sel) != 1 || sel[0].Kernel != "SampleCategorical" {
		t.Fatalf("selections are %+v, want one SampleCategorical", sel)
	}

	d := rt.Device()
	out := u32Buffer(t, d, "token", make([]uint32, len(rows)))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f32Buffer(t, d, "x", flatten(rows)),
		"u": f32Buffer(t, d, "u", draws), "token": out,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]uint32, len(rows))
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}

	for r := range rows {
		if o := hostCategorical(rows[r], draws[r]); o != want[r] {
			t.Fatalf("the oracle says a draw of %v takes %d and this test says %d",
				draws[r], o, want[r])
		}
		if got[r] != want[r] {
			t.Errorf("row %d drew %v and sampled %d, want %d", r, draws[r], got[r], want[r])
		}
	}
	if got[4] == got[5] {
		t.Errorf("rows 4 and 5 hold the same distribution and different draws and both "+
			"sampled %d; a batch sharing one draw stays reproducible and makes two "+
			"sequences converge (specs/043-per-row-values.md)", got[4])
	}
}

// A zero-weight entry can never be drawn, whatever the draw is.
//
// That is what makes a mask a mask, and it is the property the whole composition
// rests on: [tensor.TopKMask] and [tensor.TopPMask] express "dropped" as a
// weight of zero, so a walk that could still land on one would make truncation
// decorative. The kernel's partial sum does not increase on a zero and the
// comparison is strict, so this holds for every draw rather than for most.
func TestAZeroWeightIsNeverDrawn(t *testing.T) {
	// Only indices 1 and 4 survive, and the kept mass is 0.3 rather than 1 --
	// so this also pins that no renormalizing pass is needed or performed.
	dist := []float32{0, 0.1, 0, 0, 0.2, 0, 0}
	const rows = 32
	draws := make([]float32, rows)
	for i := range draws {
		draws[i] = float32(i) / rows
	}
	all := make([][]float32, rows)
	for i := range all {
		all[i] = dist
	}

	rt := newRuntime(t)
	b := rt.NewBuilder("masked")
	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F32, Shape: tensor.Shape{rows, len(dist)},
	})
	u := tensor.Input(b, tensor.ValueDesc{
		Name: "u", DType: accel.F32, Shape: tensor.Shape{rows},
	})
	tensor.Output(b, "token", tensor.SampleCategorical(b, x, u))
	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "masked"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	d := rt.Device()
	out := u32Buffer(t, d, "token", make([]uint32, rows))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f32Buffer(t, d, "x", flatten(all)),
		"u": f32Buffer(t, d, "u", draws), "token": out,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]uint32, rows)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}

	seen := map[uint32]int{}
	for r, tok := range got {
		if tok != 1 && tok != 4 {
			t.Fatalf("a draw of %v took index %d, whose weight is zero", draws[r], tok)
		}
		if want := hostCategorical(dist, draws[r]); tok != want {
			t.Errorf("a draw of %v took %d, and the oracle says %d", draws[r], tok, want)
		}
		seen[tok]++
	}
	// Both survivors must actually be reachable, or "never draws a zero" would
	// be satisfied by a sampler that always returned the same kept index.
	if len(seen) != 2 {
		t.Errorf("a sweep of %d draws over two kept entries reached %d of them", rows, len(seen))
	}
}

// Top-k keeps exactly k, and top-p keeps the set that reaches p, per row.
//
// Both are checked against oracles written from the definitions and both rows
// carry a plateau at the boundary, which is where the two designs
// specs/028-sampling.md rejected differ from the one it took: a threshold search
// keeps however many entries happen to sit above wherever it stopped, which is k
// only when nothing ties.
func TestTruncationsKeepWhatTheirDefinitionsSay(t *testing.T) {
	const vocab = 48
	// Non-negative, because a nucleus over signed weights is not a nucleus.
	// Sixteen entries share the boundary value, so a k of 12 has to choose
	// twelve of them by index.
	rowA := make([]float32, vocab)
	for i := range rowA {
		rowA[i] = float32(i%7) + 1
		if i%16 == 3 {
			rowA[i] = 9
		}
	}
	// A second row at a different scale, so a mask reading row zero's frontier
	// or row zero's total is visible.
	rowB := make([]float32, vocab)
	for i := range rowB {
		rowB[i] = 40 * (float32((i*5)%11) + 1)
	}
	rows := [][]float32{rowA, rowB}

	for _, tc := range []struct {
		name   string
		build  func(b *tensor.Builder, x *tensor.Tensor) *tensor.Tensor
		kernel string
		oracle func(row []float32) []float32
	}{
		{
			name:   "TopKMask",
			build:  func(b *tensor.Builder, x *tensor.Tensor) *tensor.Tensor { return tensor.TopKMask(b, x, 12) },
			kernel: "TopKMask",
			oracle: func(row []float32) []float32 { return hostTopK(row, 12) },
		},
		{
			name:   "TopPMask",
			build:  func(b *tensor.Builder, x *tensor.Tensor) *tensor.Tensor { return tensor.TopPMask(b, x, 0.5) },
			kernel: "TopPMask",
			oracle: func(row []float32) []float32 { return hostTopP(row, 0.5) },
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rt := newRuntime(t)
			b := rt.NewBuilder(tc.name)
			x := tensor.Input(b, tensor.ValueDesc{
				Name: "x", DType: accel.F32, Shape: tensor.Shape{len(rows), vocab},
			})
			tensor.Output(b, "mask", tc.build(b, x))
			plan, err := b.Compile(rt, tensor.CompileOptions{Label: tc.name})
			if err != nil {
				t.Fatalf("compile: %v", err)
			}
			defer plan.Close()
			if sel := plan.Selections(); len(sel) != 1 || sel[0].Kernel != tc.kernel {
				t.Fatalf("selections are %+v, want one %s", sel, tc.kernel)
			}

			d := rt.Device()
			out := f32Buffer(t, d, "mask", make([]float32, len(rows)*vocab))
			f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
				"x": f32Buffer(t, d, "x", flatten(rows)), "mask": out,
			}})
			if err := f.Wait(); err != nil {
				t.Fatalf("submit: %v", err)
			}
			got := make([]float32, len(rows)*vocab)
			if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
				t.Fatalf("readback: %v", err)
			}

			for r, row := range rows {
				want := tc.oracle(row)
				kept := 0
				for i := range row {
					if got[r*vocab+i] != want[i] {
						t.Fatalf("row %d element %d is %v, and the definition says %v",
							r, i, got[r*vocab+i], want[i])
					}
					if want[i] != 0 {
						kept++
					}
				}
				if tc.name == "TopKMask" && kept != 12 {
					t.Errorf("row %d kept %d entries for a k of 12; a plateau spans the "+
						"boundary, so anything but exactly 12 means the boundary was "+
						"chosen by value rather than by (value, index)", r, kept)
				}
			}
		})
	}
}

// The composed pipeline is the order specs/039-sampling-policy.md section 5
// fixes, and the token it produces is one of the entries the masks kept.
//
// Softmax, then top-k, then top-p, then the draw. Nothing renormalizes in
// between, so the weights the walk sees sum to well under one -- and the walk
// scales its draw by that total rather than by one, which is what makes the
// masks composable at all. Inserting a second Softmax after a mask would make
// every dropped entry weigh exp(0) = 1 against the kept ones' exp(p) and this
// fails immediately, which is the mutation section 5 names.
func TestTheComposedPipelineSamplesOnlyFromTheNucleus(t *testing.T) {
	const vocab, rows = 64, 16
	logits := make([]float32, vocab)
	for i := range logits {
		logits[i] = float32(math.Sin(float64(i)*0.7)) * 3
	}
	all := make([][]float32, rows)
	draws := make([]float32, rows)
	for i := range all {
		all[i] = logits
		draws[i] = float32(i) / rows
	}

	rt := newRuntime(t)
	b := rt.NewBuilder("pipeline")
	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F32, Shape: tensor.Shape{rows, vocab},
	})
	u := tensor.Input(b, tensor.ValueDesc{
		Name: "u", DType: accel.F32, Shape: tensor.Shape{rows},
	})
	probs := tensor.Softmax(b, x, tensor.SoftmaxOptions{Axis: 1})
	masked := tensor.TopPMask(b, tensor.TopKMask(b, probs, 8), 0.6)
	tensor.Output(b, "mask", masked)
	tensor.Output(b, "token", tensor.SampleCategorical(b, masked, u))
	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "pipeline"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	// The order is asserted on the plan rather than inferred from the result:
	// a pipeline that ran top-p first would still produce a plausible token.
	var ops []string
	for _, s := range plan.Selections() {
		ops = append(ops, s.Kernel)
	}
	want := []string{"Softmax", "TopKMask", "TopPMask", "SampleCategorical"}
	if strings.Join(ops, ",") != strings.Join(want, ",") {
		t.Fatalf("the plan runs %v, and specs/039-sampling-policy.md section 5 fixes %v",
			ops, want)
	}

	d := rt.Device()
	mask := f32Buffer(t, d, "mask", make([]float32, rows*vocab))
	out := u32Buffer(t, d, "token", make([]uint32, rows))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f32Buffer(t, d, "x", flatten(all)),
		"u": f32Buffer(t, d, "u", draws), "mask": mask, "token": out,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	gotMask := make([]float32, rows*vocab)
	if err := d.Queue().ReadBuffer(mask.Buffer, 0, gotMask); err != nil {
		t.Fatalf("readback mask: %v", err)
	}
	got := make([]uint32, rows)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback token: %v", err)
	}

	// The kept set, and the mass it carries. Under one by a margin, which is
	// what a renormalizing pass or a second softmax would destroy.
	var mass float32
	kept := map[uint32]bool{}
	for i := range vocab {
		if v := gotMask[i]; v != 0 {
			kept[uint32(i)] = true
			mass += v
		}
	}
	if len(kept) == 0 || len(kept) > 8 {
		t.Fatalf("a top-8 followed by a top-p kept %d entries", len(kept))
	}
	if mass >= 0.95 {
		t.Errorf("the kept mass is %v; a mask that left it at one is not a mask, and a "+
			"second Softmax after it would put every dropped entry at exp(0) = 1", mass)
	}

	seen := map[uint32]bool{}
	for r, tok := range got {
		if !kept[tok] {
			t.Errorf("a draw of %v sampled %d, which the masks dropped", draws[r], tok)
		}
		seen[tok] = true
	}
	if len(seen) < 2 {
		t.Errorf("a sweep of %d draws over %d kept entries reached %d of them",
			rows, len(kept), len(seen))
	}
}

// A batch of one is the same path, not a special case.
//
// specs/043-per-row-values.md section 3 makes that the orthogonality test: after
// this there is no question of the form "which of the two do I use?". A rank-1
// logits tensor is one sequence and gets a one-element token buffer, and running
// three sequences together gives what running them one at a time gives.
func TestABatchOfOneIsTheSamePath(t *testing.T) {
	const vocab = 40
	rows := [][]float32{make([]float32, vocab), make([]float32, vocab), make([]float32, vocab)}
	for r, row := range rows {
		for i := range row {
			row[i] = float32((i*(r+2))%13) + 1
		}
	}
	draws := []float32{0.05, 0.5, 0.9}

	rt := newRuntime(t)
	d := rt.Device()

	// The batched run, over a [3, vocab] tensor.
	run := func(rows [][]float32, draws []float32) []uint32 {
		b := rt.NewBuilder("batch")
		shape := tensor.Shape{len(rows), vocab}
		if len(rows) == 1 {
			// A single sequence, written the way a caller with one sequence
			// writes it: no batch axis at all.
			shape = tensor.Shape{vocab}
		}
		x := tensor.Input(b, tensor.ValueDesc{Name: "x", DType: accel.F32, Shape: shape})
		u := tensor.Input(b, tensor.ValueDesc{
			Name: "u", DType: accel.F32, Shape: tensor.Shape{len(draws)},
		})
		masked := tensor.TopKMask(b, x, 6)
		tensor.Output(b, "token", tensor.SampleCategorical(b, masked, u))
		plan, err := b.Compile(rt, tensor.CompileOptions{Label: "batch"})
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		defer plan.Close()

		// A rank-1 input names one token, not none: the leading axes of the
		// logits are the result, and there are none.
		for _, p := range plan.Ports() {
			if p.Name == "token" && !p.Shape.Equal(tensor.Shape{len(rows)}) {
				t.Fatalf("a %v logits tensor declares a token port of %v, want %v",
					shape, p.Shape, tensor.Shape{len(rows)})
			}
		}

		out := u32Buffer(t, d, "token", make([]uint32, len(rows)))
		f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
			"x": f32Buffer(t, d, "x", flatten(rows)),
			"u": f32Buffer(t, d, "u", draws), "token": out,
		}})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		got := make([]uint32, len(rows))
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}
		return got
	}

	together := run(rows, draws)
	for r := range rows {
		alone := run(rows[r:r+1], draws[r:r+1])
		if alone[0] != together[r] {
			t.Errorf("sequence %d sampled %d in a batch of three and %d on its own",
				r, together[r], alone[0])
		}
		// And against the definition, so three wrong-but-equal answers do not
		// pass. The mask is applied first, exactly as the graph does it.
		if want := hostCategorical(hostTopK(rows[r], 6), draws[r]); together[r] != want {
			t.Errorf("sequence %d sampled %d and the definition says %d",
				r, together[r], want)
		}
	}
}

// A configuration a kernel could not report is refused before it is compiled.
//
// This layer is the first one that can report an error -- a kernel cannot, which
// is why specs/028-sampling.md has the draw clamp and the round count truncate
// down there. Every row below is a value that reaches the kernel and produces a
// plausible token: a k above the bound silently becomes top-128, a p of zero
// erases the row and the walk returns the last index, and a draws tensor of the
// wrong length leaves rows reading whatever follows it.
func TestSamplingRefusesWhatAKernelCannotReport(t *testing.T) {
	rt := newRuntime(t)
	for _, tc := range []struct {
		name  string
		build func(b *tensor.Builder)
		want  string
	}{
		{
			name: "f16 logits",
			build: func(b *tensor.Builder) {
				x := tensor.Input(b, tensor.ValueDesc{
					Name: "x", DType: accel.F16, Shape: tensor.Shape{8},
				})
				tensor.Output(b, "t", tensor.Argmax(b, x))
			},
			want: "Cast",
		},
		{
			name: "one draw for three rows",
			build: func(b *tensor.Builder) {
				x := tensor.Input(b, tensor.ValueDesc{
					Name: "x", DType: accel.F32, Shape: tensor.Shape{3, 8},
				})
				u := tensor.Input(b, tensor.ValueDesc{
					Name: "u", DType: accel.F32, Shape: tensor.Shape{1},
				})
				tensor.Output(b, "t", tensor.SampleCategorical(b, x, u))
			},
			want: "one per row",
		},
		{
			name: "u32 draws",
			build: func(b *tensor.Builder) {
				x := tensor.Input(b, tensor.ValueDesc{
					Name: "x", DType: accel.F32, Shape: tensor.Shape{8},
				})
				u := tensor.Input(b, tensor.ValueDesc{
					Name: "u", DType: accel.U32, Shape: tensor.Shape{1},
				})
				tensor.Output(b, "t", tensor.SampleCategorical(b, x, u))
			},
			want: "draws are u32",
		},
		{
			name: "k above the bound",
			build: func(b *tensor.Builder) {
				x := tensor.Input(b, tensor.ValueDesc{
					Name: "x", DType: accel.F32, Shape: tensor.Shape{4096},
				})
				tensor.Output(b, "m", tensor.TopKMask(b, x, 4096))
			},
			want: "the bound is 128",
		},
		{
			name: "k of zero",
			build: func(b *tensor.Builder) {
				x := tensor.Input(b, tensor.ValueDesc{
					Name: "x", DType: accel.F32, Shape: tensor.Shape{8},
				})
				tensor.Output(b, "m", tensor.TopKMask(b, x, 0))
			},
			want: "leave this operator out",
		},
		{
			name: "k wider than the vocabulary",
			build: func(b *tensor.Builder) {
				x := tensor.Input(b, tensor.ValueDesc{
					Name: "x", DType: accel.F32, Shape: tensor.Shape{8},
				})
				tensor.Output(b, "m", tensor.TopKMask(b, x, 9))
			},
			want: "wider than",
		},
		{
			name: "p of zero",
			build: func(b *tensor.Builder) {
				x := tensor.Input(b, tensor.ValueDesc{
					Name: "x", DType: accel.F32, Shape: tensor.Shape{8},
				})
				tensor.Output(b, "m", tensor.TopPMask(b, x, 0))
			},
			want: "mass in (0, 1]",
		},
		{
			name: "p above one",
			build: func(b *tensor.Builder) {
				x := tensor.Input(b, tensor.ValueDesc{
					Name: "x", DType: accel.F32, Shape: tensor.Shape{8},
				})
				tensor.Output(b, "m", tensor.TopPMask(b, x, 1.5))
			},
			want: "mass in (0, 1]",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := rt.NewBuilder("refuse")
			tc.build(b)
			err := b.Err()
			if err == nil {
				t.Fatalf("the graph built; the value reaches the kernel and the token it " +
					"produces is plausible")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("the diagnostic does not say %q: %v", tc.want, err)
			}
			// One mistake, one diagnostic, and it names the call site.
			if n := strings.Count(err.Error(), "accel/tensor:"); n != 1 {
				t.Errorf("one mistake produced %d diagnostics: %v", n, err)
			}
			if !strings.Contains(err.Error(), "sample_test.go:") {
				t.Errorf("the diagnostic does not name the call site: %v", err)
			}
		})
	}
}
