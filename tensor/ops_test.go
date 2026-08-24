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

func f16Buffer(t *testing.T, d *accel.Device, label string, vals []float32) accel.BufferView {
	t.Helper()
	bits := make([]uint16, len(vals))
	for i, v := range vals {
		bits[i] = accel.ToFloat16(v).Bits()
	}
	b, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.F16, Count: len(vals), Label: label,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
	})
	if err != nil {
		t.Fatalf("buffer %s: %v", label, err)
	}
	t.Cleanup(func() { _ = b.Close() })
	if err := d.Queue().WriteBuffer(b, 0, bits); err != nil {
		t.Fatalf("write %s: %v", label, err)
	}
	v, err := b.View(0, len(vals))
	if err != nil {
		t.Fatalf("view %s: %v", label, err)
	}
	return v
}

func u32Buffer(t *testing.T, d *accel.Device, label string, vals []uint32) accel.BufferView {
	t.Helper()
	b, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.U32, Count: len(vals), Label: label,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
	})
	if err != nil {
		t.Fatalf("buffer %s: %v", label, err)
	}
	t.Cleanup(func() { _ = b.Close() })
	if err := d.Queue().WriteBuffer(b, 0, vals); err != nil {
		t.Fatalf("write %s: %v", label, err)
	}
	v, err := b.View(0, len(vals))
	if err != nil {
		t.Fatalf("view %s: %v", label, err)
	}
	return v
}

// A transformer feed-forward block compiles and runs.
//
// specs/025-tensor-operators.md's third done criterion, and the smallest thing
// that shows the operator set *composes*: a normalization feeding a projection
// feeding an activation feeding a projection, with the intermediates aliased by
// the graph planner and every kernel choice reported.
//
// It is checked against a reference computed here in f64 rather than against a
// stored answer, so a change in any kernel shows up as a numeric failure rather
// than as a golden nobody can evaluate.
func TestAFeedForwardBlockRuns(t *testing.T) {
	const rows, width, hidden = 4, 128, 64

	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("ffn")

	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F32, Shape: tensor.Shape{rows, width},
	})
	gain := tensor.Weight(b, tensor.ValueDesc{
		Name: "gain", DType: accel.F32, Shape: tensor.Shape{width},
	})
	// The projection is f16, which is what a transformer's weights are and what
	// the registered GEMM reads.
	w1 := tensor.Weight(b, tensor.ValueDesc{
		Name: "w1", DType: accel.F16, Shape: tensor.Shape{width, hidden},
	})
	xf16 := tensor.Input(b, tensor.ValueDesc{
		Name: "xf16", DType: accel.F16, Shape: tensor.Shape{rows, width},
	})

	normed := tensor.RMSNorm(b, x, gain, 1e-5)
	tensor.Output(b, "normed", normed)
	proj := tensor.MatMul(b, xf16, w1)
	tensor.Output(b, "h", tensor.SiLU(b, proj))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "ffn"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	xs := make([]float32, rows*width)
	gs := make([]float32, width)
	ws := make([]float32, width*hidden)
	for i := range xs {
		xs[i] = float32(math.Sin(float64(i))) * 2
	}
	for i := range gs {
		gs[i] = 1 + float32(i%5)/8
	}
	for i := range ws {
		ws[i] = float32(i%7)/4 - 0.75
	}

	normedOut := f32Buffer(t, d, "normed", make([]float32, rows*width))
	hOut := f32Buffer(t, d, "h", make([]float32, rows*hidden))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x":      f32Buffer(t, d, "x", xs),
		"xf16":   f16Buffer(t, d, "xf16", xs),
		"gain":   f32Buffer(t, d, "gain", gs),
		"w1":     f16Buffer(t, d, "w1", ws),
		"normed": normedOut, "h": hOut,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	gotNorm := make([]float32, rows*width)
	if err := d.Queue().ReadBuffer(normedOut.Buffer, 0, gotNorm); err != nil {
		t.Fatalf("readback: %v", err)
	}
	for r := range rows {
		var sum float64
		for c := range width {
			v := float64(xs[r*width+c])
			sum += v * v
		}
		scale := 1 / math.Sqrt(sum/float64(width)+1e-5)
		for c := range width {
			want := float32(float64(xs[r*width+c]) * scale * float64(gs[c]))
			if got := gotNorm[r*width+c]; math.Abs(float64(got-want)) > 1e-4 {
				t.Fatalf("normed (%d,%d) is %v, want about %v", r, c, got, want)
			}
		}
	}

	gotH := make([]float32, rows*hidden)
	if err := d.Queue().ReadBuffer(hOut.Buffer, 0, gotH); err != nil {
		t.Fatalf("readback: %v", err)
	}
	for r := range rows {
		for c := range hidden {
			var acc float64
			for k := range width {
				acc += float64(accel.ToFloat16(xs[r*width+k]).F32()) *
					float64(accel.ToFloat16(ws[k*hidden+c]).F32())
			}
			want := acc / (1 + math.Exp(-acc))
			if got := float64(gotH[r*hidden+c]); math.Abs(got-want) > 1e-2*(1+math.Abs(want)) {
				t.Fatalf("h (%d,%d) is %v, want about %v", r, c, got, want)
			}
		}
	}

	// Every operator reported what it became, and the GEMM reported which of
	// the two it chose.
	var gemm string
	for _, s := range plan.Selections() {
		if s.Op == "MatMul" {
			gemm = s.Kernel
		}
	}
	if gemm != "MatMulTiled" {
		t.Errorf("a %d-row MatMul selected %q, want the tiled GEMM", rows, gemm)
	}
}

// A matrix-vector product selects the M=1 kernel and says so.
//
// The selection is the point: specs/007-tensor-layer.md makes MatVec "the
// selected M=1 implementation, not a distinct public semantic operation", so a
// caller writes MatMul and is told which one they got. This is the shape every
// later selection takes.
func TestMatVecIsASelection(t *testing.T) {
	const k, n = 40, 12
	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("mv")

	x := tensor.Input(b, tensor.ValueDesc{
		Name: "x", DType: accel.F16, Shape: tensor.Shape{1, k},
	})
	w := tensor.Weight(b, tensor.ValueDesc{
		Name: "w", DType: accel.F16, Shape: tensor.Shape{k, n},
	})
	tensor.Output(b, "y", tensor.MatMul(b, x, w))

	plan, err := b.Compile(rt, tensor.CompileOptions{})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	sel := plan.Selections()[0]
	if sel.Kernel != "MatVec" {
		t.Errorf("an M=1 MatMul selected %q", sel.Kernel)
	}
	if !strings.Contains(sel.Reason, "M is 1") {
		t.Errorf("the selection should say why: %q", sel.Reason)
	}
	if len(sel.Rejected) == 0 {
		t.Error("the selection should say what it rejected, which is how a caller learns " +
			"the other kernel exists")
	}

	xs := make([]float32, k)
	ws := make([]float32, k*n)
	for i := range xs {
		xs[i] = float32(i%5) - 2
	}
	for i := range ws {
		ws[i] = float32(i%3) - 1
	}
	out := f32Buffer(t, d, "y", make([]float32, n))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f16Buffer(t, d, "x", xs), "w": f16Buffer(t, d, "w", ws), "y": out,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, n)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}
	for c := range n {
		var want float64
		for i := range k {
			want += float64(xs[i]) * float64(ws[i*n+c])
		}
		if math.Abs(float64(got[c])-want) > 1e-3 {
			t.Fatalf("column %d is %v, want %v", c, got[c], want)
		}
	}
}

// Rows, Softmax and RoPE run, and each reports what it became.
func TestIndexingNormalizationAndRotation(t *testing.T) {
	rt := newRuntime(t)
	d := rt.Device()

	t.Run("Rows", func(t *testing.T) {
		const vocab, width, n = 8, 4, 3
		b := rt.NewBuilder("rows")
		table := tensor.Weight(b, tensor.ValueDesc{
			Name: "table", DType: accel.F32, Shape: tensor.Shape{vocab, width},
		})
		ids := tensor.Input(b, tensor.ValueDesc{
			Name: "ids", DType: accel.U32, Shape: tensor.Shape{n},
		})
		out := tensor.GatherRows(b, table, ids)
		if !out.Shape().Equal(tensor.Shape{n, width}) {
			t.Fatalf("Rows inferred %v, want [3 4]", out.Shape())
		}
		tensor.Output(b, "y", out)
		plan, err := b.Compile(rt, tensor.CompileOptions{})
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		defer plan.Close()

		tab := make([]float32, vocab*width)
		for i := range tab {
			tab[i] = float32(i)
		}
		// The last id is out of range, which the kernel answers with zeros:
		// a GPU cannot report one, and specs/007-tensor-layer.md makes it a
		// caller error rather than a device error.
		idv := []uint32{5, 0, vocab + 1}
		res := f32Buffer(t, d, "y", make([]float32, n*width))
		f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
			"table": f32Buffer(t, d, "table", tab),
			"ids":   u32Buffer(t, d, "ids", idv), "y": res,
		}})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		got := make([]float32, n*width)
		if err := d.Queue().ReadBuffer(res.Buffer, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}
		for c := range width {
			if want := tab[5*width+c]; got[c] != want {
				t.Errorf("row 0 column %d is %v, want %v", c, got[c], want)
			}
			if got[2*width+c] != 0 {
				t.Errorf("an out-of-range id produced %v rather than zero", got[2*width+c])
			}
		}
	})

	t.Run("Softmax", func(t *testing.T) {
		const rows, width = 2, 128
		b := rt.NewBuilder("sm")
		x := tensor.Input(b, tensor.ValueDesc{
			Name: "x", DType: accel.F32, Shape: tensor.Shape{rows, width},
		})
		tensor.Output(b, "y", tensor.Softmax(b, x, tensor.SoftmaxOptions{Axis: -1}))
		plan, err := b.Compile(rt, tensor.CompileOptions{})
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		defer plan.Close()

		xs := make([]float32, rows*width)
		for i := range xs {
			xs[i] = float32(i%13) - 6
		}
		out := f32Buffer(t, d, "y", make([]float32, rows*width))
		f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
			"x": f32Buffer(t, d, "x", xs), "y": out,
		}})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		got := make([]float32, rows*width)
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}
		for r := range rows {
			var sum float64
			for c := range width {
				v := got[r*width+c]
				if v < 0 || v > 1 {
					t.Fatalf("row %d column %d is %v, which is not a probability", r, c, v)
				}
				sum += float64(v)
			}
			if math.Abs(sum-1) > 1e-4 {
				t.Errorf("row %d sums to %v, want 1", r, sum)
			}
		}
	})

	t.Run("RoPE", func(t *testing.T) {
		const rows, width, rotary = 4, 16, 8
		b := rt.NewBuilder("rope")
		tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarF32})
		tensor.Scalar(b, tensor.ScalarDesc{Name: "pos", Kind: tensor.ScalarU32})
		x := tensor.Input(b, tensor.ValueDesc{
			Name: "x", DType: accel.F32, Shape: tensor.Shape{rows, width},
		})
		tensor.Output(b, "y", tensor.RoPE(b, x, rotary, "base", "pos"))
		plan, err := b.Compile(rt, tensor.CompileOptions{})
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		defer plan.Close()

		xs := make([]float32, rows*width)
		for i := range xs {
			xs[i] = float32(i%9) - 4
		}
		out := f32Buffer(t, d, "y", make([]float32, rows*width))
		f := plan.Submit(d.Queue(), tensor.Bindings{
			Buffers: map[string]accel.BufferView{
				"x": f32Buffer(t, d, "x", xs), "y": out,
			},
			Scalars: map[string]tensor.ScalarValue{
				"base": tensor.F32(10000), "pos": tensor.U32(3),
			},
		})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		got := make([]float32, rows*width)
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}
		for r := range rows {
			for k := range rotary / 2 {
				pos := float64(r + 3)
				freq := math.Exp(-2 * float64(k) / rotary * math.Log(10000))
				th := pos * freq
				lo, hi := r*width+2*k, r*width+2*k+1
				wantLo := float64(xs[lo])*math.Cos(th) - float64(xs[hi])*math.Sin(th)
				if math.Abs(float64(got[lo])-wantLo) > 1e-3 {
					t.Fatalf("row %d pair %d low is %v, want %v", r, k, got[lo], wantLo)
				}
			}
			// The tail past rotaryDim passes through untouched, which is what
			// the copy before the in-place dispatch is for.
			for c := rotary; c < width; c++ {
				if got[r*width+c] != xs[r*width+c] {
					t.Fatalf("row %d column %d was rotated and is past rotaryDim", r, c)
				}
			}
		}
	})
}

// The operator refusals, which are where v0 is narrower than
// specs/007-tensor-layer.md and where a caller meets that narrowness.
//
// Each says which spec owns the gap rather than "unsupported", because a caller
// who hits one needs to know whether to wait, work around it, or write the
// kernel.
func TestOperatorRefusals(t *testing.T) {
	rt := newRuntime(t)

	f32 := func(b *tensor.Builder, name string, dims ...int) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: name, DType: accel.F32, Shape: dims})
	}
	f16 := func(b *tensor.Builder, name string, dims ...int) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: name, DType: accel.F16, Shape: dims})
	}
	u32 := func(b *tensor.Builder, name string, dims ...int) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: name, DType: accel.U32, Shape: dims})
	}

	for _, tc := range []struct {
		name  string
		build func(b *tensor.Builder)
		want  string
	}{{
		name:  "a table that is not a matrix",
		build: func(b *tensor.Builder) { tensor.GatherRows(b, f32(b, "t", 8), u32(b, "i", 2)) },
		want:  "must be [vocab, width]",
	}, {
		name:  "ids that are not u32",
		build: func(b *tensor.Builder) { tensor.GatherRows(b, f32(b, "t", 8, 4), f32(b, "i", 2)) },
		want:  "ids are f32",
	}, {
		name:  "an f16 table",
		build: func(b *tensor.Builder) { tensor.GatherRows(b, f16(b, "t", 8, 4), u32(b, "i", 2)) },
		want:  "010-kernel-corpus.md owns the f16 variant",
	}, {
		name:  "a gain that is not one per feature",
		build: func(b *tensor.Builder) { tensor.RMSNorm(b, f32(b, "x", 4, 8), f32(b, "g", 4), 1e-5) },
		want:  "one value per feature",
	}, {
		name:  "an eps that is not positive",
		build: func(b *tensor.Builder) { tensor.RMSNorm(b, f32(b, "x", 4, 8), f32(b, "g", 8), 0) },
		want:  "positive and finite",
	}, {
		name:  "an f16 RMSNorm",
		build: func(b *tensor.Builder) { tensor.RMSNorm(b, f16(b, "x", 4, 8), f16(b, "g", 8), 1e-5) },
		want:  "the registered kernel reads",
	}, {
		name: "a softmax over an axis that is not the last",
		build: func(b *tensor.Builder) {
			tensor.Softmax(b, f32(b, "x", 4, 8), tensor.SoftmaxOptions{Axis: 0})
		},
		want: "needs a transpose",
	}, {
		name: "a softmax axis outside the rank",
		build: func(b *tensor.Builder) {
			tensor.Softmax(b, f32(b, "x", 4, 8), tensor.SoftmaxOptions{Axis: 7})
		},
		want: "outside a rank-2 tensor",
	}, {
		name:  "an f16 softmax",
		build: func(b *tensor.Builder) { tensor.Softmax(b, f16(b, "x", 4, 8), tensor.SoftmaxOptions{}) },
		want:  "reads f32",
	}, {
		name: "a rotary dimension that is odd",
		build: func(b *tensor.Builder) {
			tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarF32})
			tensor.Scalar(b, tensor.ScalarDesc{Name: "pos", Kind: tensor.ScalarU32})
			tensor.RoPE(b, f32(b, "x", 4, 8), 3, "base", "pos")
		},
		want: "positive, even, and no wider",
	}, {
		name: "a rotary base that is not declared",
		build: func(b *tensor.Builder) {
			tensor.RoPE(b, f32(b, "x", 4, 8), 4, "nope", "alsonope")
		},
		want: "not a declared f32 scalar",
	}, {
		name: "a rotary offset of the wrong kind",
		build: func(b *tensor.Builder) {
			tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarF32})
			tensor.Scalar(b, tensor.ScalarDesc{Name: "pos", Kind: tensor.ScalarF32})
			tensor.RoPE(b, f32(b, "x", 4, 8), 4, "base", "pos")
		},
		want: "not a declared u32 scalar",
	}, {
		name:  "a MatMul of f32 operands",
		build: func(b *tensor.Builder) { tensor.MatMul(b, f32(b, "x", 4, 8), f32(b, "w", 8, 4)) },
		want:  "010-kernel-corpus.md owns an f32 variant",
	}, {
		name:  "contracted axes that disagree",
		build: func(b *tensor.Builder) { tensor.MatMul(b, f16(b, "x", 4, 8), f16(b, "w", 5, 4)) },
		want:  "must agree",
	}, {
		name:  "a batched MatMul",
		build: func(b *tensor.Builder) { tensor.MatMul(b, f16(b, "x", 2, 4, 8), f16(b, "w", 8, 4)) },
		want:  "v0 multiplies two matrices",
	}, {
		name: "a bias that is not one per column",
		build: func(b *tensor.Builder) {
			tensor.Linear(b, f16(b, "x", 4, 8), f16(b, "w", 8, 6), f32(b, "c", 4))
		},
		want: "one value per output column",
	}, {
		name: "an f16 bias",
		build: func(b *tensor.Builder) {
			tensor.Linear(b, f16(b, "x", 4, 8), f16(b, "w", 8, 6), f16(b, "c", 6))
		},
		want: "the epilogue adds in f32",
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

	// And a poisoned operand flows through every one of them silently.
	b := rt.NewBuilder("poison")
	bad := tensor.MatMul(b, f16(b, "x", 4, 8), f16(b, "w", 5, 4))
	before := b.Err().Error()
	tensor.GatherRows(b, bad, u32(b, "i", 2))
	tensor.RMSNorm(b, bad, bad, 1e-5)
	tensor.Softmax(b, bad, tensor.SoftmaxOptions{})
	tensor.RoPE(b, bad, 2, "base", "pos")
	tensor.MatMul(b, bad, bad)
	tensor.Linear(b, bad, bad, bad)
	tensor.Scale(b, bad, "f")
	tensor.SwiGLU(b, bad, bad)
	if b.Err().Error() != before {
		t.Errorf("an operator on a poisoned tensor recorded a diagnostic:\n%v", b.Err())
	}
}

// Linear adds its bias in the same kernel, and says what it rejected.
func TestLinearFusesItsBias(t *testing.T) {
	const m, k, n = 9, 23, 19
	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("lin")

	x := tensor.Input(b, tensor.ValueDesc{Name: "x", DType: accel.F16, Shape: tensor.Shape{m, k}})
	w := tensor.Weight(b, tensor.ValueDesc{Name: "w", DType: accel.F16, Shape: tensor.Shape{k, n}})
	c := tensor.Weight(b, tensor.ValueDesc{Name: "c", DType: accel.F32, Shape: tensor.Shape{n}})
	tensor.Output(b, "y", tensor.Linear(b, x, w, c))

	plan, err := b.Compile(rt, tensor.CompileOptions{})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()
	if sel := plan.Selections()[0]; len(sel.Rejected) == 0 {
		t.Error("Linear should report rejecting the composed MatMul-then-Add form")
	}

	xs := make([]float32, m*k)
	ws := make([]float32, k*n)
	cs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(i%5) - 2
	}
	for i := range ws {
		ws[i] = float32(i%3) - 1
	}
	for i := range cs {
		cs[i] = float32(i%4) - 1.5
	}
	out := f32Buffer(t, d, "y", make([]float32, m*n))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f16Buffer(t, d, "x", xs), "w": f16Buffer(t, d, "w", ws),
		"c": f32Buffer(t, d, "c", cs), "y": out,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, m*n)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}
	for r := range m {
		for col := range n {
			want := float64(cs[col])
			for i := range k {
				want += float64(xs[r*k+i]) * float64(ws[i*n+col])
			}
			if math.Abs(float64(got[r*n+col])-want) > 1e-3 {
				t.Fatalf("(%d,%d) is %v, want %v", r, col, got[r*n+col], want)
			}
		}
	}
}
