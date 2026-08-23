// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package testkernels_test

import (
	"golang.design/x/accel/kernelabi"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/direct"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/testkernels"
)

// The elementwise family against independent references, at a length that is
// not a multiple of the workgroup so the guarded tail is exercised.
func TestTheElementwiseFamily(t *testing.T) {
	const n = 130

	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32(math.Sin(float64(i)) * 3)
		b[i] = float32(math.Cos(float64(i)) * 2)
	}

	cases := []struct {
		name string
		run  func(out []float32)
		want func(i int) float64
	}{{
		name: "add",
		run: func(out []float32) {
			runFlat(t, &testkernels.ElemAddKernel, n, kernelabi.Args{
				Slices: []any{a, b, out}})
		},
		want: func(i int) float64 { return float64(a[i]) + float64(b[i]) },
	}, {
		name: "mul",
		run: func(out []float32) {
			runFlat(t, &testkernels.ElemMulKernel, n, kernelabi.Args{
				Slices: []any{a, b, out}})
		},
		want: func(i int) float64 { return float64(a[i]) * float64(b[i]) },
	}, {
		name: "scale",
		run: func(out []float32) {
			runFlat(t, &testkernels.ElemScaleKernel, n, kernelabi.Args{
				Slices:   []any{a, out},
				Uniforms: []any{testkernels.ScaleParams{Factor: 0.25}},
			})
		},
		want: func(i int) float64 { return float64(a[i]) * 0.25 },
	}, {
		name: "silu",
		run: func(out []float32) {
			runFlat(t, &testkernels.SiLUKernel, n, kernelabi.Args{Slices: []any{a, out}})
		},
		want: func(i int) float64 {
			x := float64(a[i])
			return x / (1 + math.Exp(-x))
		},
	}, {
		name: "swiglu",
		run: func(out []float32) {
			runFlat(t, &testkernels.SwiGLUKernel, n, kernelabi.Args{
				Slices: []any{a, b, out}})
		},
		want: func(i int) float64 {
			x := float64(a[i])
			return x / (1 + math.Exp(-x)) * float64(b[i])
		},
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out := make([]float32, n)
			c.run(out)
			for i := range out {
				want := c.want(i)
				if diff := math.Abs(float64(out[i]) - want); diff > math.Abs(want)*1e-5+1e-6 {
					t.Fatalf("element %d is %v, want about %v", i, out[i], want)
				}
			}
		})
	}
}

func runFlat(t *testing.T, k *accel.Kernel, n int, args kernelabi.Args) {
	t.Helper()
	if err := direct.Run(k, direct.Cover(k, n), args); err != nil {
		t.Fatalf("run %s: %v", k.Name, err)
	}
}

// SiLU is finite at both ends, which is what negating the exponent buys.
//
// Written as x·exp(x)/(1+exp(x)) it is algebraically identical and overflows
// above about 88, where exp(x) is +Inf and the quotient is Inf/Inf. So this is
// not a rearrangement for tidiness: it is the difference between an activation
// and a NaN on inputs a real model produces.
func TestSiLUIsFiniteAtBothExtremes(t *testing.T) {
	in := []float32{-1e30, -200, -100, -88, 0, 88, 100, 200, 1e30}
	out := make([]float32, len(in))
	runFlat(t, &testkernels.SiLUKernel, len(in), kernelabi.Args{Slices: []any{in, out}})

	for i, x := range in {
		got := float64(out[i])
		if math.IsNaN(got) {
			t.Fatalf("silu(%v) is NaN: writing the exponent positive overflows above "+
				"about 88 and the quotient becomes Inf over Inf", x)
		}
		// Large positive: silu(x) → x. Large negative: silu(x) → 0.
		if x > 100 && math.Abs(got-float64(x)) > math.Abs(float64(x))*1e-5 {
			t.Errorf("silu(%v) is %v, want about %v", x, got, x)
		}
		if x < -100 && math.Abs(got) > 1e-6 {
			t.Errorf("silu(%v) is %v, want about 0", x, got)
		}
	}
}

// A gather checks its ids, because an id past the table reads another row: a
// plausible vector, in the right shape, that silently changes what the model
// saw.
func TestGatherRowsChecksItsIDs(t *testing.T) {
	const rows, width, capacity = 4, 3, 5
	p := testkernels.RowParams{Rows: rows, Width: width, Capacity: capacity}

	table := make([]float32, capacity*width)
	for i := range table {
		table[i] = float32(i) + 1
	}
	ids := []uint32{2, 0, capacity, 4} // the third is out of range
	out := make([]float32, rows*width)

	runFlat(t, &testkernels.GatherRowsKernel, rows*width, kernelabi.Args{
		Slices: []any{table, ids, out}, Uniforms: []any{p}})

	for r := range rows {
		for c := range width {
			got := out[r*width+c]
			if ids[r] >= capacity {
				if got != 0 {
					t.Errorf("row %d came back as %v for an out-of-range id, want zeros: "+
						"reading past the table returns another row's vector", r, got)
				}
				continue
			}
			if want := table[ids[r]*width+uint32(c)]; got != want {
				t.Errorf("row %d column %d is %v, want %v", r, c, got, want)
			}
		}
	}
}

// A scatter drops an out-of-range write rather than clamping.
//
// Clamping would corrupt a real row with another's contents, which is worse
// than dropping the write and much harder to notice: the model would keep
// running with one position of its cache silently wrong.
func TestScatterRowsDropsAnOutOfRangeWrite(t *testing.T) {
	const rows, width, capacity = 3, 2, 4
	p := testkernels.RowParams{Rows: rows, Width: width, Capacity: capacity}

	state := make([]float32, capacity*width)
	for i := range state {
		state[i] = -1 // a value no write produces
	}
	rowsIn := make([]float32, rows*width)
	for i := range rowsIn {
		rowsIn[i] = float32(i) + 1
	}
	ids := []uint32{1, capacity + 7, 3}

	runFlat(t, &testkernels.ScatterRowsKernel, rows*width, kernelabi.Args{
		Slices: []any{rowsIn, ids, state}, Uniforms: []any{p}})

	for r := range rows {
		if ids[r] >= capacity {
			continue
		}
		for c := range width {
			if got, want := state[ids[r]*width+uint32(c)], rowsIn[r*width+c]; got != want {
				t.Errorf("state row %d column %d is %v, want %v", ids[r], c, got, want)
			}
		}
	}
	// The rows nothing legally targeted are untouched, which is what says the
	// out-of-range write was dropped rather than clamped onto one of them.
	for _, untouched := range []uint32{0, 2} {
		for c := range width {
			if got := state[untouched*width+uint32(c)]; got != -1 {
				t.Errorf("state row %d column %d is %v and nothing targeted it: an "+
					"out-of-range write was clamped onto a real row", untouched, c, got)
			}
		}
	}
}

// RoPE rotates each pair by the angle its position and index imply, and leaves
// the tail past the rotary dimension alone.
func TestRoPERotatesPairsAndLeavesTheTail(t *testing.T) {
	const rows, width, rotary = 3, 8, 4
	p := testkernels.RoPEParams{
		Rows: rows, Width: width, RotaryDim: rotary, Base: 10000, Offset: 2,
	}

	inout := make([]float32, rows*width)
	for i := range inout {
		inout[i] = float32(i%7) - 3
	}
	before := append([]float32(nil), inout...)

	runFlat(t, &testkernels.RoPEKernel, rows*rotary/2, kernelabi.Args{
		Slices: []any{inout}, Uniforms: []any{p}})

	for r := range rows {
		pos := float64(r) + float64(p.Offset)
		for k := range rotary / 2 {
			freq := math.Pow(float64(p.Base), -2*float64(k)/float64(rotary))
			theta := pos * freq
			lo, hi := r*width+2*k, r*width+2*k+1
			x, y := float64(before[lo]), float64(before[hi])
			wantLo := x*math.Cos(theta) - y*math.Sin(theta)
			wantHi := x*math.Sin(theta) + y*math.Cos(theta)
			if math.Abs(float64(inout[lo])-wantLo) > 1e-4 {
				t.Fatalf("row %d pair %d low is %v, want %v", r, k, inout[lo], wantLo)
			}
			if math.Abs(float64(inout[hi])-wantHi) > 1e-4 {
				t.Fatalf("row %d pair %d high is %v, want %v", r, k, inout[hi], wantHi)
			}
		}
		// The tail past the rotary dimension is untouched, which models rely on
		// and which a rotation over the whole width would silently break.
		for c := rotary; c < width; c++ {
			if inout[r*width+c] != before[r*width+c] {
				t.Fatalf("row %d column %d moved and is past the rotary dimension", r, c)
			}
		}
	}
}

// The offset is what makes a decode step's rotation match the prefill's.
//
// A decode step's row is at position cacheLen, not zero, so an implementation
// ignoring the offset rotates every token as though it were the first — which
// produces a fluent model that has lost track of order.
func TestRoPEHonoursTheOffset(t *testing.T) {
	const width, rotary = 4, 4
	base := func(offset uint32) []float32 {
		p := testkernels.RoPEParams{
			Rows: 1, Width: width, RotaryDim: rotary, Base: 10000, Offset: offset,
		}
		inout := []float32{1, 0, 1, 0}
		runFlat(t, &testkernels.RoPEKernel, rotary/2, kernelabi.Args{
			Slices: []any{inout}, Uniforms: []any{p}})
		return inout
	}
	at0, at5 := base(0), base(5)
	same := true
	for i := range at0 {
		if math.Abs(float64(at0[i]-at5[i])) > 1e-5 {
			same = false
		}
	}
	if same {
		t.Fatal("the rotation is identical at offsets 0 and 5, so the offset is being " +
			"ignored: a decode step's row is at position cacheLen, not zero")
	}
}

// The authored halves, which for a flat kernel is a direct call: no rendezvous
// to emulate, because nothing cooperates.
func TestAuthoredElementwiseFamily(t *testing.T) {
	const n = 70
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32(i%11) - 5
		b[i] = float32(i%7) - 3
	}

	run := func(k *accel.Kernel, args kernelabi.Args, authored func(th kernel.Thread)) []float32 {
		t.Helper()
		const group = 64
		size := kernel.ID3{X: group, Y: 1, Z: 1}
		groups := uint32((n + group - 1) / group)
		for g := range groups {
			for l := range uint32(group) {
				authored(kernel.NewThread(kernel.ID3{X: g*group + l}, kernel.ID3{X: l},
					kernel.ID3{X: g}, size, kernel.ID3{X: groups}))
			}
		}
		gen := make([]float32, n)
		genArgs := args
		genArgs.Slices = append(append([]any(nil), args.Slices[:len(args.Slices)-1]...), gen)
		if err := direct.Run(k, accel.ID3{X: groups}, genArgs); err != nil {
			t.Fatalf("run %s: %v", k.Name, err)
		}
		return gen
	}

	t.Run("add", func(t *testing.T) {
		au := make([]float32, n)
		gen := run(&testkernels.ElemAddKernel, kernelabi.Args{Slices: []any{a, b, au}},
			func(th kernel.Thread) { testkernels.ElemAdd(th, a, b, au) })
		compareExact(t, au, gen)
	})
	t.Run("mul", func(t *testing.T) {
		au := make([]float32, n)
		gen := run(&testkernels.ElemMulKernel, kernelabi.Args{Slices: []any{a, b, au}},
			func(th kernel.Thread) { testkernels.ElemMul(th, a, b, au) })
		compareExact(t, au, gen)
	})
	t.Run("scale", func(t *testing.T) {
		p := testkernels.ScaleParams{Factor: 0.5}
		au := make([]float32, n)
		gen := run(&testkernels.ElemScaleKernel,
			kernelabi.Args{Slices: []any{a, au}, Uniforms: []any{p}},
			func(th kernel.Thread) { testkernels.ElemScale(th, p, a, au) })
		compareExact(t, au, gen)
	})
	t.Run("silu", func(t *testing.T) {
		au := make([]float32, n)
		gen := run(&testkernels.SiLUKernel, kernelabi.Args{Slices: []any{a, au}},
			func(th kernel.Thread) { testkernels.SiLU(th, a, au) })
		compareExact(t, au, gen)
	})
	t.Run("swiglu", func(t *testing.T) {
		au := make([]float32, n)
		gen := run(&testkernels.SwiGLUKernel, kernelabi.Args{Slices: []any{a, b, au}},
			func(th kernel.Thread) { testkernels.SwiGLU(th, a, b, au) })
		compareExact(t, au, gen)
	})
}

func compareExact(t *testing.T, authored, generated []float32) {
	t.Helper()
	for i := range authored {
		if math.Float32bits(authored[i]) != math.Float32bits(generated[i]) {
			t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
		}
	}
}

// The authored gather, scatter and rotation, which take uniforms and so are
// driven separately.
func TestAuthoredRowAndRotationKernels(t *testing.T) {
	const group = 64
	size := kernel.ID3{X: group, Y: 1, Z: 1}
	drive := func(n int, body func(th kernel.Thread)) {
		groups := uint32((n + group - 1) / group)
		for g := range groups {
			for l := range uint32(group) {
				body(kernel.NewThread(kernel.ID3{X: g*group + l}, kernel.ID3{X: l},
					kernel.ID3{X: g}, size, kernel.ID3{X: groups}))
			}
		}
	}

	t.Run("gather", func(t *testing.T) {
		p := testkernels.RowParams{Rows: 3, Width: 4, Capacity: 5}
		table := make([]float32, p.Capacity*p.Width)
		for i := range table {
			table[i] = float32(i)
		}
		ids := []uint32{1, 9, 4}
		au := make([]float32, p.Rows*p.Width)
		drive(int(p.Rows*p.Width), func(th kernel.Thread) {
			testkernels.GatherRows(th, p, table, ids, au)
		})
		gen := make([]float32, p.Rows*p.Width)
		runFlat(t, &testkernels.GatherRowsKernel, int(p.Rows*p.Width),
			kernelabi.Args{Slices: []any{table, ids, gen}, Uniforms: []any{p}})
		compareExact(t, au, gen)
	})

	t.Run("scatter", func(t *testing.T) {
		p := testkernels.RowParams{Rows: 2, Width: 3, Capacity: 4}
		rows := []float32{1, 2, 3, 4, 5, 6}
		ids := []uint32{0, 9}
		au := make([]float32, p.Capacity*p.Width)
		drive(int(p.Rows*p.Width), func(th kernel.Thread) {
			testkernels.ScatterRows(th, p, rows, ids, au)
		})
		gen := make([]float32, p.Capacity*p.Width)
		runFlat(t, &testkernels.ScatterRowsKernel, int(p.Rows*p.Width),
			kernelabi.Args{Slices: []any{rows, ids, gen}, Uniforms: []any{p}})
		compareExact(t, au, gen)
	})

	t.Run("rope", func(t *testing.T) {
		p := testkernels.RoPEParams{Rows: 2, Width: 6, RotaryDim: 4, Base: 10000, Offset: 1}
		start := make([]float32, p.Rows*p.Width)
		for i := range start {
			start[i] = float32(i%5) - 2
		}
		au := append([]float32(nil), start...)
		drive(int(p.Rows*p.RotaryDim/2), func(th kernel.Thread) {
			testkernels.RoPE(th, p, au)
		})
		gen := append([]float32(nil), start...)
		runFlat(t, &testkernels.RoPEKernel, int(p.Rows*p.RotaryDim/2),
			kernelabi.Args{Slices: []any{gen}, Uniforms: []any{p}})
		compareExact(t, au, gen)
	})
}
