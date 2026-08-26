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
