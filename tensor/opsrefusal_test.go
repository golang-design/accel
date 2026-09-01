// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/quant"
	"golang.design/x/accel/tensor"
)

// The dtype and shape refusals the operator layer makes, each naming what it
// refused.
//
// specs/009-sequencing.md's refusal audit: these exist in code and no test had
// made any of them fire, so nothing said whether the condition was right or the
// message named the right value. They are gathered in one table because they
// are one kind of check -- an operator declining a dtype or a shape it has no
// kernel for -- and reading them together is how a missing one is noticed.
func TestTheOperatorDtypeAndShapeRefusals(t *testing.T) {
	rt := newRuntime(t)

	f32 := func(b *tensor.Builder, n string, dims ...int) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: n, DType: accel.F32, Shape: dims})
	}
	typed := func(b *tensor.Builder, n string, dt accel.DType, dims ...int) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: n, DType: dt, Shape: dims})
	}

	for _, tc := range []struct {
		name  string
		build func(b *tensor.Builder)
		want  string
	}{{
		name:  "a port with no name",
		build: func(b *tensor.Builder) { f32(b, "", 4) },
		want:  "a port needs a name",
	}, {
		name: "a port with no shape",
		build: func(b *tensor.Builder) {
			tensor.Input(b, tensor.ValueDesc{Name: "x", DType: accel.F32})
		},
		want: "a scalar is a shape of [1]",
	}, {
		// The diagnostic names the operator a caller wrote. It said
		// "Persistent", an operator that no longer exists, so a reader
		// searching for it found nothing.
		name: "a state with no name",
		build: func(b *tensor.Builder) {
			tensor.NewState(b, tensor.StateDesc{DType: accel.F32, Shape: tensor.Shape{4}})
		},
		want: "NewState at",
	}, {
		name:  "an output with no name",
		build: func(b *tensor.Builder) { tensor.Output(b, "", f32(b, "x", 4)) },
		want:  "an output needs a name",
	}, {
		name: "an elementwise op over an integer dtype",
		build: func(b *tensor.Builder) {
			tensor.Add(b, typed(b, "x", accel.U32, 4), typed(b, "y", accel.U32, 4))
		},
		want: "is not an elementwise dtype",
	}, {
		name: "Scale over an integer dtype",
		build: func(b *tensor.Builder) {
			tensor.Scalar(b, tensor.ScalarDesc{Name: "s", Kind: tensor.ScalarF32})
			tensor.Scale(b, typed(b, "x", accel.U32, 4), "s")
		},
		want: "is not an elementwise dtype",
	}, {
		name: "SwiGLU over two different dtypes",
		build: func(b *tensor.Builder) {
			tensor.SwiGLU(b, f32(b, "g", 4), typed(b, "v", accel.F16, 4))
		},
		want: "operands are",
	}, {
		name: "SwiGLU over an integer dtype",
		build: func(b *tensor.Builder) {
			tensor.SwiGLU(b, typed(b, "g", accel.U32, 4), typed(b, "v", accel.U32, 4))
		},
		want: "is not an elementwise dtype",
	}, {
		name: "RoPE over a narrow dtype",
		build: func(b *tensor.Builder) {
			tensor.Scalar(b, tensor.ScalarDesc{Name: "theta", Kind: tensor.ScalarF32})
			pos := typed(b, "pos", accel.U32, 1)
			tensor.RoPE(b, typed(b, "x", accel.F16, 1, 2, 4), 4, "theta", pos)
		},
		want: "the registered kernel reads f32",
	}, {
		name: "a Cast the corpus does not register",
		build: func(b *tensor.Builder) {
			tensor.Cast(b, typed(b, "x", accel.F16, 4), accel.BF16)
		},
		want: "registers f32 to f16",
	}, {
		// Both axes, because the two checks are separate lines carrying the
		// same sentence. One case reaches whichever is second and leaves the
		// first untested, and the message cannot tell them apart -- so the
		// axis number in each expectation is what says which one answered.
		name: "a Transpose first axis outside the tensor",
		build: func(b *tensor.Builder) {
			tensor.Transpose(b, f32(b, "x", 2, 3), 7, 1)
		},
		want: "axis 7 is outside a rank-2 tensor",
	}, {
		name: "a Transpose second axis outside the tensor",
		build: func(b *tensor.Builder) {
			tensor.Transpose(b, f32(b, "x", 2, 3), 0, 5)
		},
		want: "axis 5 is outside a rank-2 tensor",
	}, {
		// Contiguous returns its operand untouched when the layout is already
		// packed, so every refusal below it needs a *strided* operand. A
		// transpose is what makes one.
		name: "packing a dtype the kernel does not move",
		build: func(b *tensor.Builder) {
			tensor.Contiguous(b, tensor.Transpose(b, typed(b, "x", accel.U32, 2, 3), 0, 1))
		},
		want: "not a dtype the packing kernel moves",
	}, {
		name: "packing a strided view with an empty axis",
		build: func(b *tensor.Builder) {
			x := tensor.Transpose(b, f32(b, "x", 2, 3), 0, 1)
			tensor.Contiguous(b, tensor.Slice(b, x, 0, 1, 1))
		},
		want: "needs every extent to compute an index",
	}} {
		t.Run(tc.name, func(t *testing.T) {
			b := rt.NewBuilder("refusal")
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
}

// The two refusals Compile makes that a caller can actually reach.
//
// specs/009-sequencing.md's refusal audit counted ten error sites in
// compile.go, and the count was too broad. Eight are internal invariants of the
// lowering -- a pipeline that will not build, a slot that is missing, a value
// written nowhere -- reached only if the builder handed the compiler something
// it does not construct. They return a plain error rather than calling b.fail,
// and that difference is the line: b.fail records a refusal against the
// caller's own line, so it is something a caller can provoke and should be
// told about. These two are that kind.
func TestCompileRefusals(t *testing.T) {
	rt := newRuntime(t)

	t.Run("no runtime", func(t *testing.T) {
		b := rt.NewBuilder("nort")
		tensor.Output(b, "o", tensor.Input(b, tensor.ValueDesc{
			Name: "x", DType: accel.F32, Shape: tensor.Shape{4}}))

		_, err := b.Compile(nil, tensor.CompileOptions{Label: "nort"})
		if err == nil {
			t.Fatal("compiling against no runtime was accepted")
		}
		if !strings.Contains(err.Error(), "Compile needs a runtime") {
			t.Errorf("the refusal should name the missing runtime, got %v", err)
		}
	})

	// A strided operand reaching a kernel that indexes contiguously. This is
	// the refusal that keeps specs/007-tensor-layer.md's rule true: a copy of a
	// matrix is something a caller asks for, not something that happens -- so
	// the compiler declines rather than inserting a pack nobody wrote.
	t.Run("a strided view into a contiguous kernel", func(t *testing.T) {
		b := rt.NewBuilder("strided")
		g := tensor.Input(b, tensor.ValueDesc{
			Name: "g", DType: accel.F32, Shape: tensor.Shape{2}})
		x := tensor.Input(b, tensor.ValueDesc{
			Name: "x", DType: accel.F32, Shape: tensor.Shape{2, 3}})
		// Transposed, so the last axis is 2 and the layout is not packed.
		tr := tensor.Transpose(b, x, 0, 1)
		tensor.Output(b, "o", tensor.RMSNorm(b, tr, g, 1e-5))

		_, err := b.Compile(rt, tensor.CompileOptions{Label: "strided"})
		if err == nil {
			t.Fatal("a strided operand was accepted by a contiguous kernel")
		}
		for _, want := range []string{"operand 0 is a strided view", "Insert Contiguous"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("the refusal should say %q, got %v", want, err)
			}
		}

		// And the fix the message names actually works, which is what makes it
		// advice rather than an apology.
		b2 := rt.NewBuilder("packed")
		g2 := tensor.Input(b2, tensor.ValueDesc{
			Name: "g", DType: accel.F32, Shape: tensor.Shape{2}})
		x2 := tensor.Input(b2, tensor.ValueDesc{
			Name: "x", DType: accel.F32, Shape: tensor.Shape{2, 3}})
		packed := tensor.Contiguous(b2, tensor.Transpose(b2, x2, 0, 1))
		tensor.Output(b2, "o", tensor.RMSNorm(b2, packed, g2, 1e-5))
		plan, err := b2.Compile(rt, tensor.CompileOptions{Label: "packed"})
		if err != nil {
			t.Fatalf("the fix the refusal names does not compile: %v", err)
		}
		plan.Close()
	})
}

// The quantized and sampling operators' remaining refusals.
//
// The audit's caller-facing half: these call b.fail, so a caller can provoke
// them and is told at their own line. Every one had shipped without a test.
func TestTheQuantizedAndSamplingRefusals(t *testing.T) {
	rt := newRuntime(t)

	f32 := func(b *tensor.Builder, n string, dims ...int) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: n, DType: accel.F32, Shape: dims})
	}
	typed := func(b *tensor.Builder, n string, dt accel.DType, dims ...int) *tensor.Tensor {
		return tensor.Input(b, tensor.ValueDesc{Name: n, DType: dt, Shape: dims})
	}
	// A well-formed int8 pair, so a case changes one thing at a time.
	pair := func(b *tensor.Builder, rows, cols int) tensor.Quantized {
		return tensor.Quantized{
			Quants: tensor.Weight(b, tensor.ValueDesc{
				Name: "q", DType: accel.I8, Shape: tensor.Shape{rows, cols}}),
			Scales: tensor.Weight(b, tensor.ValueDesc{
				Name: "s", DType: accel.F16,
				Shape: tensor.Shape{rows * cols / quant.Int8Block}}),
		}
	}

	for _, tc := range []struct {
		name  string
		build func(b *tensor.Builder)
		want  string
	}{{
		name: "a quantized product over something that is not two matrices",
		build: func(b *tensor.Builder) {
			tensor.QuantMatMul(b, f32(b, "x", 64), pair(b, 64, 4))
		},
		want: "v0 multiplies two matrices",
	}, {
		name: "a gather from a table missing its scale plane",
		build: func(b *tensor.Builder) {
			tbl := pair(b, 64, 4)
			tbl.Scales = nil
			tensor.QuantGatherRows(b, tbl, typed(b, "ids", accel.U32, 2))
		},
		want: "needs both a quant plane and a scale plane",
	}, {
		name: "a gather from a table whose planes disagree",
		build: func(b *tensor.Builder) {
			tbl := pair(b, 64, 4)
			// A scale plane for a different matrix than the codes describe.
			tbl.Scales = tensor.Weight(b, tensor.ValueDesc{
				Name: "s2", DType: accel.F16, Shape: tensor.Shape{1}})
			tensor.QuantGatherRows(b, tbl, typed(b, "ids", accel.U32, 2))
		},
		want: "scale",
	}, {
		name: "sampling from logits that are not f32",
		build: func(b *tensor.Builder) {
			tensor.SampleCategorical(b, typed(b, "w", accel.F16, 8), f32(b, "d", 1))
		},
		want: "a narrow head needs an explicit Cast",
	}, {
		name: "a top-k mask over logits that are not f32",
		build: func(b *tensor.Builder) {
			tensor.TopKMask(b, typed(b, "w", accel.F16, 8), 2)
		},
		want: "a narrow head needs an explicit Cast",
	}, {
		name: "a top-p mask over logits that are not f32",
		build: func(b *tensor.Builder) {
			tensor.TopPMask(b, typed(b, "w", accel.F16, 8), 0.9)
		},
		want: "a narrow head needs an explicit Cast",
	}, {
		name: "scattering into a state the kernels cannot write",
		build: func(b *tensor.Builder) {
			s := tensor.NewState(b, tensor.StateDesc{
				Name: "c", DType: accel.U32, Shape: tensor.Shape{8, 4}})
			tensor.ScatterRows(b, s, f32(b, "r", 1, 4), typed(b, "i", accel.U32, 1))
		},
		want: "the registered kernels write f32 or f16",
	}, {
		name: "scattering rows whose dtype is not the state's",
		build: func(b *tensor.Builder) {
			s := tensor.NewState(b, tensor.StateDesc{
				Name: "c", DType: accel.F16, Shape: tensor.Shape{8, 4}})
			tensor.ScatterRows(b, s, f32(b, "r", 1, 4), typed(b, "i", accel.U32, 1))
		},
		want: "so they share a dtype -- Cast the rows",
	}} {
		t.Run(tc.name, func(t *testing.T) {
			b := rt.NewBuilder("qs-refusal")
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
}

// A rank-zero state, which only a repeated LayerState produces.
//
// The last two caller-facing refusals from the audit, and the reason they are
// worth a note. NewState refuses a shape with no axes, so reasoning from the
// constructor says both are dead code -- which is exactly the mistake
// specs/009-sequencing.md's correction records, made once and nearly made
// again here.
//
// LayerState is the operator that reduces rank: it indexes a layer off the
// front. Applied to a rank-1 state it yields a rank-zero one, and applied twice
// it reaches the guard. So the state's rank is not bounded below by its
// declaration, and the ways it arrives have to be enumerated rather than
// assumed from how it was built.
func TestARankZeroStateIsRefused(t *testing.T) {
	rt := newRuntime(t)

	// LayerState off a rank-1 state is legal and leaves nothing to index.
	t.Run("LayerState on what has no axes left", func(t *testing.T) {
		b := rt.NewBuilder("layer0")
		s := tensor.NewState(b, tensor.StateDesc{
			Name: "c", DType: accel.F32, Shape: tensor.Shape{4}})
		flat := tensor.LayerState(b, s, 0)
		if err := b.Err(); err != nil {
			t.Fatalf("one LayerState off a rank-1 state should be legal: %v", err)
		}
		tensor.LayerState(b, flat, 0)

		err := b.Err()
		if err == nil {
			t.Fatal("a second LayerState was accepted")
		}
		if !strings.Contains(err.Error(), "has no shape") {
			t.Errorf("the refusal should name the missing shape, got %v", err)
		}
	})

	t.Run("ScatterRows into what has no axes left", func(t *testing.T) {
		b := rt.NewBuilder("scatter0")
		s := tensor.NewState(b, tensor.StateDesc{
			Name: "c", DType: accel.F32, Shape: tensor.Shape{4}})
		flat := tensor.LayerState(b, s, 0)
		if err := b.Err(); err != nil {
			t.Fatalf("one LayerState off a rank-1 state should be legal: %v", err)
		}
		rows := tensor.Input(b, tensor.ValueDesc{
			Name: "r", DType: accel.F32, Shape: tensor.Shape{1, 4}})
		ids := tensor.Input(b, tensor.ValueDesc{
			Name: "i", DType: accel.U32, Shape: tensor.Shape{1}})
		tensor.ScatterRows(b, flat, rows, ids)

		err := b.Err()
		if err == nil {
			t.Fatal("scattering into a rank-zero state was accepted")
		}
		if !strings.Contains(err.Error(), "ScatterRows") {
			t.Errorf("the refusal should name ScatterRows, got %v", err)
		}
	})
}
