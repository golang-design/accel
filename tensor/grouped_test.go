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

// A grouped product through the public surface matches each expert alone.
//
// specs/049-grouped-gemm.md §4. The whole claim, end to end: each segment is
// the ungrouped product of its own rows against its own weights.
func TestAGroupedMatVecMatchesEachExpertAlone(t *testing.T) {
	const experts, K, N = 3, 64, 4
	counts := []uint32{2, 0, 3}
	tokens := 5

	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("grouped")

	xv := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F32, Shape: tensor.Shape{tokens, K},
	})
	wv := tensor.Weight(b, tensor.ValueDesc{
		Name: "w", DType: accel.F32, Shape: tensor.Shape{experts, K, N},
	})
	cv := tensor.Input(b, tensor.ValueDesc{
		Name: "counts", DType: accel.U32, Shape: tensor.Shape{experts},
	})
	tensor.Output(b, "out", tensor.GroupedMatVec(b, xv, wv, cv))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "grouped"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	x := make([]float32, tokens*K)
	for i := range x {
		x[i] = float32(math.Sin(float64(i) * 0.19))
	}
	// Each expert's matrix scaled by its index, so the wrong one is wrong by a
	// factor rather than by a rounding.
	w := make([]float32, experts*K*N)
	for e := range experts {
		for i := range K * N {
			w[e*K*N+i] = float32(math.Cos(float64(i)*0.13)) * float32(e+1)
		}
	}

	out := f32Buffer(t, d, "out", make([]float32, tokens*N))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f32Buffer(t, d, "x", x), "w": f32Buffer(t, d, "w", w),
		"counts": u32Buffer(t, d, "counts", counts), "out": out,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, tokens*N)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}

	off := 0
	for e := range experts {
		for i := range int(counts[e]) {
			tok := off + i
			for n := range N {
				want := 0.0
				for k := range K {
					want += float64(x[tok*K+k]) * float64(w[e*K*N+k*N+n])
				}
				if diff := math.Abs(float64(got[tok*N+n]) - want); diff > 1e-4 {
					t.Fatalf("expert %d token %d column %d is %v, want %v",
						e, tok, n, got[tok*N+n], want)
				}
			}
		}
		off += int(counts[e])
	}
}

// The shapes a grouped product needs are refused when they disagree.
func TestAGroupedMatVecRefusesMismatchedShapes(t *testing.T) {
	for _, c := range []struct {
		name       string
		xs, ws, cs tensor.Shape
		want       string
	}{
		{"a contraction width that differs", tensor.Shape{4, 64},
			tensor.Shape{2, 32, 4}, tensor.Shape{2}, "same width"},
		{"one count per expert", tensor.Shape{4, 64},
			tensor.Shape{2, 64, 4}, tensor.Shape{3}, "one token count per expert"},
		{"x that is not a matrix", tensor.Shape{4, 64, 2},
			tensor.Shape{2, 64, 4}, tensor.Shape{2}, "end to end"},
	} {
		t.Run(c.name, func(t *testing.T) {
			rt := newRuntime(t)
			b := rt.NewBuilder("refusal")
			x := tensor.Input(b, tensor.ValueDesc{Name: "x", DType: accel.F32, Shape: c.xs})
			w := tensor.Weight(b, tensor.ValueDesc{Name: "w", DType: accel.F32, Shape: c.ws})
			cnt := tensor.Input(b, tensor.ValueDesc{Name: "c", DType: accel.U32, Shape: c.cs})
			tensor.Output(b, "out", tensor.GroupedMatVec(b, x, w, cnt))
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

// The dtypes and the degenerate shapes a grouped product refuses.
func TestAGroupedMatVecRefusesWrongTypes(t *testing.T) {
	for _, c := range []struct {
		name       string
		xd, wd, cd tensor.DType
		xs, ws, cs tensor.Shape
		nilCounts  bool
		want       string
	}{
		{name: "no counts", xd: accel.F32, wd: accel.F32, cd: accel.U32,
			xs: tensor.Shape{4, 64}, ws: tensor.Shape{2, 64, 4}, cs: tensor.Shape{2},
			nilCounts: true, want: "one token count per expert"},
		{name: "f16 activations", xd: accel.F16, wd: accel.F32, cd: accel.U32,
			xs: tensor.Shape{4, 64}, ws: tensor.Shape{2, 64, 4}, cs: tensor.Shape{2},
			want: "x is"},
		{name: "counts that are not u32", xd: accel.F32, wd: accel.F32, cd: accel.F32,
			xs: tensor.Shape{4, 64}, ws: tensor.Shape{2, 64, 4}, cs: tensor.Shape{2},
			want: "counts is"},
		{name: "w that is not a stack of matrices", xd: accel.F32, wd: accel.F32, cd: accel.U32,
			xs: tensor.Shape{4, 64}, ws: tensor.Shape{64, 4}, cs: tensor.Shape{2},
			want: "one matrix per expert"},
	} {
		t.Run(c.name, func(t *testing.T) {
			rt := newRuntime(t)
			b := rt.NewBuilder("refusal")
			x := tensor.Input(b, tensor.ValueDesc{Name: "x", DType: c.xd, Shape: c.xs})
			w := tensor.Weight(b, tensor.ValueDesc{Name: "w", DType: c.wd, Shape: c.ws})
			var cnt *tensor.Tensor
			if !c.nilCounts {
				cnt = tensor.Input(b, tensor.ValueDesc{Name: "c", DType: c.cd, Shape: c.cs})
			}
			tensor.Output(b, "out", tensor.GroupedMatVec(b, x, w, cnt))
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

// A weight tensor with no experts is refused.
//
// Reachable only through a view: Input refuses a dimension of zero, so the
// empty operand this needs cannot be declared, but Slice permits an empty
// half-open range and yields a legal zero-element tensor. The counts are
// sliced to match, so the earlier count-versus-expert check passes and this
// branch is the one that answers.
//
// Recorded because the first reachability check got it wrong: testing against
// the constructor alone reported this refusal as dead code.
func TestAGroupedProductRefusesAWeightWithNoExperts(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("grouped")

	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F32, Shape: tensor.Shape{2, 4}})
	w := tensor.Input(b, tensor.ValueDesc{
		Name: "w", DType: accel.F32, Shape: tensor.Shape{3, 4, 3}})
	counts := tensor.Input(b, tensor.ValueDesc{
		Name: "counts", DType: accel.U32, Shape: tensor.Shape{3}})

	tensor.GroupedMatVec(b, x, tensor.Slice(b, w, 0, 0, 0),
		tensor.Slice(b, counts, 0, 0, 0))

	err := b.Err()
	if err == nil {
		t.Fatal("a weight tensor with no experts was accepted")
	}
	if !strings.Contains(err.Error(), "declares no experts") {
		t.Errorf("the refusal should name the missing experts, got: %v", err)
	}
}

// The batched grouped operator equals the row operator through the public
// surface, over a batch no expert divides evenly.
//
// specs/049-grouped-gemm.md §5. Same inputs, same answer, different kernel:
// what a caller gets by switching is speed, not arithmetic.
func TestAGroupedMatMulEqualsTheMatVecThroughThePlan(t *testing.T) {
	rt := newRuntime(t)
	d := rt.Device()

	const K, N = 40, 20
	counts := []uint32{5, 0, 9}
	experts := len(counts)
	tokens := 0
	for _, c := range counts {
		tokens += int(c)
	}

	x := make([]float32, tokens*K)
	for i := range x {
		x[i] = float32(math.Sin(float64(i) * 0.19))
	}
	w := make([]float32, experts*K*N)
	for e := range experts {
		for i := range K * N {
			w[e*K*N+i] = float32(math.Cos(float64(i)*0.11)) * float32(e+1)
		}
	}

	run := func(label string, tiled bool) []float32 {
		t.Helper()
		b := rt.NewBuilder(label)
		xv := tensor.Input(b, tensor.ValueDesc{
			Name: "x", DType: accel.F32, Shape: tensor.Shape{tokens, K}})
		wv := tensor.Weight(b, tensor.ValueDesc{
			Name: "w", DType: accel.F32, Shape: tensor.Shape{experts, K, N}})
		cv := tensor.Input(b, tensor.ValueDesc{
			Name: "counts", DType: accel.U32, Shape: tensor.Shape{experts}})
		if tiled {
			tensor.Output(b, "out", tensor.GroupedMatMul(b, xv, wv, cv))
		} else {
			tensor.Output(b, "out", tensor.GroupedMatVec(b, xv, wv, cv))
		}

		plan, err := b.Compile(rt, tensor.CompileOptions{Label: label})
		if err != nil {
			t.Fatalf("compile %s: %v", label, err)
		}
		defer plan.Close()

		out := f32Buffer(t, d, "out", make([]float32, tokens*N))
		f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
			"x":      f32Buffer(t, d, "x", x),
			"w":      f32Buffer(t, d, "w", w),
			"counts": u32Buffer(t, d, "counts", counts),
			"out":    out,
		}})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit %s: %v", label, err)
		}
		got := make([]float32, tokens*N)
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("readback %s: %v", label, err)
		}
		return got
	}

	tiled, row := run("grouped-tiled", true), run("grouped-row", false)
	for i := range row {
		if e := math.Abs(float64(tiled[i] - row[i])); e > 1e-4 {
			t.Fatalf("element %d (token %d, column %d): tiled %v, row %v",
				i, i/N, i%N, tiled[i], row[i])
		}
	}
	// A real answer, not two agreeing zeros.
	nonzero := false
	for _, v := range row {
		if v != 0 {
			nonzero = true
		}
	}
	if !nonzero {
		t.Fatal("every element is zero, so the comparison says nothing")
	}
}

// Every refusal both grouped operators make, each naming what it refused.
//
// Written with the operators rather than after them: the audit in
// specs/009-sequencing.md found 39 refusals in this package that no test had
// ever made fire, and GroupedMatMul shipped seven more of them. A refusal
// nothing exercises may have the wrong condition or name the wrong value, and
// the graph still compiles either way.
//
// Both operators are driven through one table because they take the same
// arguments and make the same checks. A refusal added to one and not the other
// shows up here as a case that passes for the wrong operator.
func TestTheGroupedOperatorsRefusals(t *testing.T) {
	rt := newRuntime(t)

	f32 := func(b *tensor.Builder, n string, dims ...int) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: n, DType: accel.F32, Shape: dims})
	}
	u32 := func(b *tensor.Builder, n string, dims ...int) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: n, DType: accel.U32, Shape: dims})
	}

	for _, tc := range []struct {
		name  string
		build func(b *tensor.Builder) (x, w, counts *tensor.Tensor)
		want  string
	}{{
		name: "no counts at all",
		build: func(b *tensor.Builder) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
			return f32(b, "x", 2, 4), f32(b, "w", 2, 4, 3), nil
		},
		want: "counts is nil",
	}, {
		name: "activations that are not f32",
		build: func(b *tensor.Builder) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
			x := tensor.Input(b, tensor.ValueDesc{
				Name: "x", DType: accel.F16, Shape: tensor.Shape{2, 4}})
			return x, f32(b, "w", 2, 4, 3), u32(b, "c", 2)
		},
		want: "this kernel reads",
	}, {
		name: "counts that are not u32",
		build: func(b *tensor.Builder) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
			return f32(b, "x", 2, 4), f32(b, "w", 2, 4, 3), f32(b, "c", 2)
		},
		want: "this kernel reads",
	}, {
		name: "activations of the wrong rank",
		build: func(b *tensor.Builder) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
			return f32(b, "x", 8), f32(b, "w", 2, 4, 3), u32(b, "c", 2)
		},
		want: "it is [tokens, K]",
	}, {
		name: "weights of the wrong rank",
		build: func(b *tensor.Builder) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
			return f32(b, "x", 2, 4), f32(b, "w", 4, 3), u32(b, "c", 2)
		},
		want: "it is [experts, K, N]",
	}, {
		name: "a contraction width the weights do not share",
		build: func(b *tensor.Builder) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
			return f32(b, "x", 2, 4), f32(b, "w", 2, 5, 3), u32(b, "c", 2)
		},
		want: "every expert contracts",
	}, {
		name: "one count per expert, and not that many",
		build: func(b *tensor.Builder) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
			return f32(b, "x", 2, 4), f32(b, "w", 2, 4, 3), u32(b, "c", 3)
		},
		want: "one token count per expert",
	}, {
		// Reachable only through a view: Input refuses a dimension of zero, and
		// Slice permits an empty half-open range. The counts are sliced to match
		// so the count-versus-expert check passes and this one answers.
		name: "weights with no experts",
		build: func(b *tensor.Builder) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
			return f32(b, "x", 2, 4),
				tensor.Slice(b, f32(b, "w", 2, 4, 3), 0, 0, 0),
				tensor.Slice(b, u32(b, "c", 2), 0, 0, 0)
		},
		want: "declares no experts",
	}} {
		for _, op := range []struct {
			name string
			call func(b *tensor.Builder, x, w, c *tensor.Tensor) *tensor.Tensor
		}{
			{"MatVec", tensor.GroupedMatVec},
			{"MatMul", tensor.GroupedMatMul},
		} {
			t.Run(tc.name+"/"+op.name, func(t *testing.T) {
				b := rt.NewBuilder("grouped-refusal")
				x, w, c := tc.build(b)
				op.call(b, x, w, c)
				err := b.Err()
				if err == nil {
					t.Fatalf("Grouped%s accepted %s", op.name, tc.name)
				}
				if !strings.Contains(err.Error(), tc.want) {
					t.Errorf("the refusal should say %q, got %v", tc.want, err)
				}
				if !strings.Contains(err.Error(), "Grouped"+op.name) {
					t.Errorf("the refusal should name Grouped%s, got %v", op.name, err)
				}
			})
		}
	}
}
