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

func newRuntime(t *testing.T) *tensor.Runtime {
	t.Helper()
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	rt, err := tensor.NewRuntime(d)
	if err != nil {
		t.Fatalf("runtime: %v", err)
	}
	t.Cleanup(func() {
		if err := rt.Close(); err != nil {
			t.Errorf("runtime close: %v", err)
		}
		_ = d.Close()
	})
	return rt
}

func f32Buffer(t *testing.T, d *accel.Device, label string, vals []float32) accel.BufferView {
	t.Helper()
	b, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: len(vals), Label: label,
		Usage: accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst,
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

func value(name string, n int) tensor.ValueDesc {
	return tensor.ValueDesc{Name: name, DType: accel.F32, Shape: tensor.Shape{n}}
}

// The end-to-end shape of the API: declare, build, compile, bind, submit, read.
//
// specs/024-tensor-bringup.md's fifth testing level, and the only one that
// checks the parts fit together. Everything below tests a piece; this checks
// that a caller who knows nothing about the pieces can use it.
func TestAPlanComputesWhatItsGraphSays(t *testing.T) {
	const n = 256
	rt := newRuntime(t)
	b := rt.NewBuilder("mlp")

	x := tensor.Input(b, value("x", n))
	g := tensor.Weight(b, value("gate", n))
	w := tensor.Weight(b, value("w", n))
	// y = (x + w) * SiLU(gate), which is three operators, two of them consuming
	// an intermediate. A single operator would not show that intermediates
	// become transients.
	sum := tensor.Add(b, x, w)
	act := tensor.SiLU(b, g)
	tensor.Output(b, "y", tensor.Mul(b, sum, act))

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "mlp"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	xs := make([]float32, n)
	ws := make([]float32, n)
	gs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(i%7) - 3
		ws[i] = float32(i%5) - 2
		gs[i] = float32(i%11) - 5
	}
	d := rt.Device()
	out := f32Buffer(t, d, "y", make([]float32, n))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f32Buffer(t, d, "x", xs), "w": f32Buffer(t, d, "w", ws),
		"gate": f32Buffer(t, d, "gate", gs), "y": out,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}

	got := make([]float32, n)
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}
	for i := range got {
		silu := gs[i] / (1 + exp(-gs[i]))
		if want := (xs[i] + ws[i]) * silu; !close32(got[i], want) {
			t.Fatalf("element %d is %v, want %v", i, got[i], want)
		}
	}

	// The intermediates went through the recorder, which is what makes the
	// graph's aliasing apply to them. A tensor layer that allocated its own
	// would report nothing here.
	if plan.Memory().TransientBytes == 0 {
		t.Error("the plan reports no transient memory, so its two intermediates did not " +
			"go through the graph planner")
	}
	if len(plan.Selections()) != 3 {
		t.Errorf("three operators produced %d selections", len(plan.Selections()))
	}
	for _, s := range plan.Selections() {
		if s.Kernel == "" || s.Reason == "" {
			t.Errorf("selection for %s reports no kernel or no reason: %+v", s.Op, s)
		}
	}
}

// One mistake produces one diagnostic, and it names the operator and the line.
//
// The rule is what makes a poisoned tensor safe rather than merely convenient:
// without it, a wrong shape near the top of a model produces a page of errors
// that all describe the same thing in different words, and the real one is
// indistinguishable from its echoes.
func TestOneMistakeIsOneDiagnostic(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("bad")

	x := tensor.Input(b, value("x", 8))
	y := tensor.Input(b, tensor.ValueDesc{
		Name: "y", DType: accel.F32, Shape: tensor.Shape{3, 5},
	})
	bad := tensor.Add(b, x, y) // the only mistake
	// Ten consumers of the poisoned value, none of which should say anything.
	for range 10 {
		bad = tensor.SiLU(b, bad)
	}
	tensor.Output(b, "out", bad)

	_, err := b.Compile(rt, tensor.CompileOptions{})
	if err == nil {
		t.Fatal("a graph with mismatched shapes compiled")
	}
	msg := err.Error()
	if n := strings.Count(msg, "accel/tensor:"); n != 1 {
		t.Errorf("one mistake produced %d diagnostics:\n%s", n, msg)
	}
	if !strings.Contains(msg, "Add") {
		t.Errorf("the diagnostic does not name the operator: %s", msg)
	}
	if !strings.Contains(msg, "tensor_test.go:") {
		t.Errorf("the diagnostic does not name the call site, which is the only place a "+
			"tensor graph has one: %s", msg)
	}
	if !strings.Contains(msg, "broadcast") {
		t.Errorf("the diagnostic does not say what was wrong: %s", msg)
	}
}

func exp(x float32) float32 { return float32(math.Exp(float64(x))) }

func close32(a, b float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	m := b
	if m < 0 {
		m = -m
	}
	return d <= 1e-5*(1+m)
}

// A submission validates its whole binding set before binding any of it.
//
// All at once, because a caller cannot see which half of a partial bind landed
// and could not recover if they could. And every failure arrives through the
// fence rather than as a second return value, which is what makes Submit one
// thing to check -- specs/007-tensor-layer.md asks for that and it is why
// accel.FailedFence exists.
func TestBindingValidation(t *testing.T) {
	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("v")
	x := tensor.Input(b, value("x", 8))
	tensor.Output(b, "y", tensor.SiLU(b, x))
	plan, err := b.Compile(rt, tensor.CompileOptions{})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	good := func() map[string]accel.BufferView {
		return map[string]accel.BufferView{
			"x": f32Buffer(t, d, "x", make([]float32, 8)),
			"y": f32Buffer(t, d, "y", make([]float32, 8)),
		}
	}

	for _, tc := range []struct {
		name string
		mut  func(map[string]accel.BufferView)
		want string
	}{{
		name: "a missing port",
		mut:  func(m map[string]accel.BufferView) { delete(m, "x") },
		want: `input "x" is not bound`,
	}, {
		name: "a port nobody declared",
		mut: func(m map[string]accel.BufferView) {
			m["z"] = f32Buffer(t, d, "z", make([]float32, 8))
		},
		want: `"z" is bound and this plan declares no such port`,
	}, {
		name: "a view that is too small",
		mut: func(m map[string]accel.BufferView) {
			m["x"] = f32Buffer(t, d, "small", make([]float32, 4))
		},
		want: "the bound view has 4",
	}, {
		name: "a view of the wrong dtype",
		mut: func(m map[string]accel.BufferView) {
			buf, err := d.NewBuffer(accel.BufferDescriptor{
				DType: accel.U32, Count: 8, Usage: accel.UsageStorage, Label: "u32",
			})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = buf.Close() })
			v, err := buf.View(0, 8)
			if err != nil {
				t.Fatal(err)
			}
			m["x"] = v
		},
		want: "the bound view is u32",
	}} {
		t.Run(tc.name, func(t *testing.T) {
			m := good()
			tc.mut(m)
			err := plan.Submit(d.Queue(), tensor.Bindings{Buffers: m}).Wait()
			if err == nil {
				t.Fatalf("%s was accepted", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("the error should contain %q, got %v", tc.want, err)
			}
		})
	}

	// And the whole set, correct, still works afterwards: a rejected binding
	// must leave the plan exactly as it was.
	if err := plan.Submit(d.Queue(), tensor.Bindings{Buffers: good()}).Wait(); err != nil {
		t.Fatalf("a valid submission after four rejected ones: %v", err)
	}
}

// Inference accepts what specs/007-tensor-layer.md says it should and refuses
// the rest by name.
//
// A table because the interesting cases are the refusals, and a refusal that
// says the wrong thing is worse than one that says nothing: it sends a reader
// to the wrong line.
func TestInference(t *testing.T) {
	rt := newRuntime(t)

	for _, tc := range []struct {
		name  string
		build func(b *tensor.Builder) *tensor.Tensor
		shape tensor.Shape
		want  string
	}{{
		name: "equal shapes",
		build: func(b *tensor.Builder) *tensor.Tensor {
			return tensor.Add(b, tensor.Input(b, value("a", 8)), tensor.Input(b, value("c", 8)))
		},
		shape: tensor.Shape{8},
	}, {
		name: "a size-one axis expands",
		build: func(b *tensor.Builder) *tensor.Tensor {
			x := tensor.Input(b, tensor.ValueDesc{Name: "a", DType: accel.F32, Shape: tensor.Shape{4, 8}})
			y := tensor.Input(b, tensor.ValueDesc{Name: "c", DType: accel.F32, Shape: tensor.Shape{1, 8}})
			return tensor.Mul(b, x, y)
		},
		shape: tensor.Shape{4, 8},
	}, {
		name: "shorter shapes align from the right",
		build: func(b *tensor.Builder) *tensor.Tensor {
			x := tensor.Input(b, tensor.ValueDesc{Name: "a", DType: accel.F32, Shape: tensor.Shape{4, 8}})
			y := tensor.Input(b, value("c", 8))
			return tensor.Add(b, x, y)
		},
		shape: tensor.Shape{4, 8},
	}, {
		name: "mismatched dtypes",
		build: func(b *tensor.Builder) *tensor.Tensor {
			x := tensor.Input(b, value("a", 8))
			y := tensor.Input(b, tensor.ValueDesc{Name: "c", DType: accel.F16, Shape: tensor.Shape{8}})
			return tensor.Add(b, x, y)
		},
		want: "elementwise operators need one dtype",
	}, {
		name: "a dtype v0 does not compute over",
		build: func(b *tensor.Builder) *tensor.Tensor {
			x := tensor.Input(b, tensor.ValueDesc{Name: "a", DType: accel.U32, Shape: tensor.Shape{8}})
			return tensor.SiLU(b, x)
		},
		want: "not an elementwise dtype",
	}, {
		name: "shapes that do not broadcast",
		build: func(b *tensor.Builder) *tensor.Tensor {
			x := tensor.Input(b, value("a", 8))
			y := tensor.Input(b, value("c", 3))
			return tensor.Add(b, x, y)
		},
		want: "do not broadcast",
	}, {
		name: "SwiGLU with unequal shapes",
		build: func(b *tensor.Builder) *tensor.Tensor {
			x := tensor.Input(b, tensor.ValueDesc{Name: "a", DType: accel.F32, Shape: tensor.Shape{4, 8}})
			y := tensor.Input(b, value("c", 8))
			return tensor.SwiGLU(b, x, y)
		},
		want: "the fused kernel indexes both operands together",
	}, {
		name: "a duplicate port name",
		build: func(b *tensor.Builder) *tensor.Tensor {
			tensor.Input(b, value("a", 8))
			return tensor.SiLU(b, tensor.Input(b, value("a", 8)))
		},
		want: "declared twice",
	}, {
		name: "a dimension that is not positive",
		build: func(b *tensor.Builder) *tensor.Tensor {
			return tensor.SiLU(b, tensor.Input(b, value("a", 0)))
		},
		want: "positive concrete integer",
	}} {
		t.Run(tc.name, func(t *testing.T) {
			b := rt.NewBuilder(tc.name)
			out := tc.build(b)
			err := b.Err()
			if tc.want == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if !out.Shape().Equal(tc.shape) {
					t.Errorf("inferred %v, want %v", out.Shape(), tc.shape)
				}
				return
			}
			if err == nil {
				t.Fatalf("%s was accepted, inferring %v", tc.name, out.Shape())
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("the refusal should say %q, got %v", tc.want, err)
			}
		})
	}
}

// A runtime refuses to close while a plan built from it is open.
//
// Checked rather than assumed, because a plan outliving its runtime holds a
// pipeline nobody owns, and the failure would be a use of a closed device
// somewhere unrelated.
func TestRuntimeOutlivesItsPlans(t *testing.T) {
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer d.Close()
	rt, err := tensor.NewRuntime(d)
	if err != nil {
		t.Fatalf("runtime: %v", err)
	}

	b := rt.NewBuilder("p")
	tensor.Output(b, "y", tensor.SiLU(b, tensor.Input(b, value("x", 8))))
	plan, err := b.Compile(rt, tensor.CompileOptions{})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	if err := rt.Close(); err == nil {
		t.Error("a runtime closed with a plan still open")
	}
	if err := plan.Close(); err != nil {
		t.Fatalf("plan close: %v", err)
	}
	if err := plan.Close(); err != nil {
		t.Errorf("closing a plan twice must be harmless: %v", err)
	}
	if err := rt.Close(); err != nil {
		t.Errorf("runtime close after its plan: %v", err)
	}
}

// A plan reports the ports a caller has to bind, so they need not have kept the
// builder.
func TestPortsAreDiscoverable(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("p")
	x := tensor.Input(b, value("x", 8))
	w := tensor.Weight(b, value("w", 8))
	tensor.Output(b, "y", tensor.Add(b, x, w))
	plan, err := b.Compile(rt, tensor.CompileOptions{})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	want := []tensor.PortDesc{
		{Name: "x", DType: accel.F32, Shape: tensor.Shape{8}, Kind: tensor.PortInput},
		{Name: "w", DType: accel.F32, Shape: tensor.Shape{8}, Kind: tensor.PortWeight},
		{Name: "y", DType: accel.F32, Shape: tensor.Shape{8}, Kind: tensor.PortOutput},
	}
	got := plan.Ports()
	if len(got) != len(want) {
		t.Fatalf("reported %d ports, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].Name != want[i].Name || got[i].Kind != want[i].Kind ||
			got[i].DType != want[i].DType || !got[i].Shape.Equal(want[i].Shape) {
			t.Errorf("port %d is %+v, want %+v", i, got[i], want[i])
		}
	}
	// Declaration order, not name order: it is the order a reader of the model
	// met them, and sorting would put the output in the middle.
	if got[0].Name != "x" || got[2].Name != "y" {
		t.Error("ports are not in declaration order")
	}
	// Returned by value, so a caller cannot reach into the plan.
	got[0].Name = "mutated"
	if plan.Ports()[0].Name != "x" {
		t.Error("Ports hands out the plan's own slice")
	}
}

// The refusals that belong to lowering rather than to inference.
//
// Broadcasting is inferred and not yet materialized, so a shape that infers
// correctly can still fail to lower. Refusing by name is what keeps that
// honest: the alternative is a kernel indexing operands of different extents,
// which reads the wrong elements rather than repeating them.
func TestLoweringRefusals(t *testing.T) {
	rt := newRuntime(t)

	t.Run("an operand that needs broadcasting", func(t *testing.T) {
		b := rt.NewBuilder("bc")
		x := tensor.Input(b, tensor.ValueDesc{
			Name: "x", DType: accel.F32, Shape: tensor.Shape{4, 8},
		})
		y := tensor.Input(b, tensor.ValueDesc{
			Name: "y", DType: accel.F32, Shape: tensor.Shape{1, 8},
		})
		tensor.Output(b, "z", tensor.Add(b, x, y))
		if _, err := b.Compile(rt, tensor.CompileOptions{}); err == nil {
			t.Fatal("a broadcast operand lowered, and the kernel indexes operands together")
		} else if !strings.Contains(err.Error(), "broadcast copy") {
			t.Errorf("the refusal should name what is missing: %v", err)
		}
	})

	t.Run("a graph with no output", func(t *testing.T) {
		b := rt.NewBuilder("no-out")
		tensor.SiLU(b, tensor.Input(b, value("x", 8)))
		if _, err := b.Compile(rt, tensor.CompileOptions{}); err == nil {
			t.Fatal("a graph computing nothing readable compiled")
		} else if !strings.Contains(err.Error(), "declares no output") {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("an output that is an input", func(t *testing.T) {
		b := rt.NewBuilder("passthrough")
		tensor.Output(b, "y", tensor.Input(b, value("x", 8)))
		if _, err := b.Compile(rt, tensor.CompileOptions{}); err == nil {
			t.Fatal("a plan that only copies compiled")
		} else if !strings.Contains(err.Error(), "a copy, not a plan") {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("a value that is both an output and an operand", func(t *testing.T) {
		b := rt.NewBuilder("both")
		x := tensor.Input(b, value("x", 8))
		mid := tensor.SiLU(b, x)
		tensor.Output(b, "mid", mid)
		tensor.Output(b, "y", tensor.Add(b, mid, x))
		if _, err := b.Compile(rt, tensor.CompileOptions{}); err == nil {
			t.Fatal("a value read after being written into an output lowered")
		} else if !strings.Contains(err.Error(), "not lowered yet") {
			t.Errorf("the refusal should say it is unbuilt rather than illegal: %v", err)
		}
	})

	t.Run("a duplicate output name", func(t *testing.T) {
		b := rt.NewBuilder("dup")
		x := tensor.Input(b, value("x", 8))
		tensor.Output(b, "y", tensor.SiLU(b, x))
		tensor.Output(b, "y", tensor.Add(b, x, x))
		if err := b.Err(); err == nil {
			t.Fatal("two outputs share a name")
		} else if !strings.Contains(err.Error(), "declared twice") {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("a builder from another runtime", func(t *testing.T) {
		other := newRuntime(t)
		b := rt.NewBuilder("x")
		tensor.Output(b, "y", tensor.SiLU(b, tensor.Input(b, value("x", 8))))
		if _, err := b.Compile(other, tensor.CompileOptions{}); err == nil {
			t.Fatal("a builder compiled against a runtime that is not its own")
		}
	})

	if _, err := tensor.NewRuntime(nil); err == nil {
		t.Error("NewRuntime accepted no device")
	}
}

// The small formatting surfaces, which appear in every diagnostic above.
func TestFormatting(t *testing.T) {
	if got := (tensor.Shape{2, 3}).String(); got != "[2 3]" {
		t.Errorf("Shape.String = %q", got)
	}
	if got := (tensor.Shape{}).String(); got != "[]" {
		t.Errorf("empty Shape.String = %q", got)
	}
	if n := (tensor.Shape{2, 3, 4}).Elements(); n != 24 {
		t.Errorf("Elements = %d, want 24", n)
	}
	if (tensor.Shape{2}).Equal(tensor.Shape{2, 1}) {
		t.Error("shapes of different rank are not equal")
	}
	for _, k := range []tensor.PortKind{
		tensor.PortInput, tensor.PortWeight, tensor.PortState, tensor.PortOutput,
	} {
		if k.String() == "" || strings.Contains(k.String(), "PortKind(") {
			t.Errorf("PortKind(%d) has no name", uint8(k))
		}
	}
	if got := tensor.PortKind(9).String(); !strings.Contains(got, "PortKind(9)") {
		t.Errorf("an unknown PortKind should say so, got %q", got)
	}

	rt := newRuntime(t)
	b := rt.NewBuilder("f")
	x := tensor.Input(b, value("x", 8))
	if got := x.String(); got != "f32[8]" {
		t.Errorf("Tensor.String = %q, want %q", got, "f32[8]")
	}
	if x.DType() != accel.F32 {
		t.Error("Tensor.DType")
	}
	bad := tensor.Add(b, x, tensor.Input(b, value("y", 3)))
	if got := bad.String(); !strings.Contains(got, "invalid") {
		t.Errorf("a poisoned tensor should say so, got %q", got)
	}
	var nilT *tensor.Tensor
	if got := nilT.String(); !strings.Contains(got, "nil") {
		t.Errorf("a nil tensor should format rather than panic, got %q", got)
	}
}

// A named scalar changes between submissions of one plan, which is the whole
// reason it is named rather than a Go value baked in at build.
//
// Two submissions with different factors, and the second result checked: a
// mechanism that wrote the value somewhere the plan does not read would pass
// any test that submitted once.
func TestAScalarVariesBetweenSubmissions(t *testing.T) {
	const n = 32
	rt := newRuntime(t)
	d := rt.Device()
	b := rt.NewBuilder("scale")
	tensor.Scalar(b, tensor.ScalarDesc{Name: "factor", Kind: tensor.ScalarF32})
	x := tensor.Input(b, value("x", n))
	tensor.Output(b, "y", tensor.Scale(b, x, "factor"))

	plan, err := b.Compile(rt, tensor.CompileOptions{})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()
	if len(plan.Scalars()) != 1 || plan.Scalars()[0].Name != "factor" {
		t.Fatalf("the plan reports scalars %+v", plan.Scalars())
	}

	xs := make([]float32, n)
	for i := range xs {
		xs[i] = float32(i) + 1
	}
	xv := f32Buffer(t, d, "x", xs)
	out := f32Buffer(t, d, "y", make([]float32, n))

	for _, factor := range []float32{2, -0.5, 100} {
		f := plan.Submit(d.Queue(), tensor.Bindings{
			Buffers: map[string]accel.BufferView{"x": xv, "y": out},
			Scalars: map[string]tensor.ScalarValue{"factor": tensor.F32(factor)},
		})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit at %v: %v", factor, err)
		}
		got := make([]float32, n)
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}
		for i := range got {
			if want := xs[i] * factor; got[i] != want {
				t.Fatalf("at factor %v element %d is %v, want %v", factor, i, got[i], want)
			}
		}
	}
}

// The scalar refusals, which are the half that keeps a misspelling from
// becoming a value nobody binds.
func TestScalarValidation(t *testing.T) {
	rt := newRuntime(t)
	d := rt.Device()

	t.Run("an undeclared name", func(t *testing.T) {
		b := rt.NewBuilder("u")
		tensor.Output(b, "y", tensor.Scale(b, tensor.Input(b, value("x", 8)), "nope"))
		if err := b.Err(); err == nil {
			t.Fatal("an undeclared scalar was accepted")
		} else if !strings.Contains(err.Error(), "not a declared scalar") {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("the wrong kind at declaration", func(t *testing.T) {
		b := rt.NewBuilder("k")
		tensor.Scalar(b, tensor.ScalarDesc{Name: "n", Kind: tensor.ScalarU32})
		tensor.Output(b, "y", tensor.Scale(b, tensor.Input(b, value("x", 8)), "n"))
		if err := b.Err(); err == nil {
			t.Fatal("a u32 scalar was accepted where f32 is needed")
		} else if !strings.Contains(err.Error(), "needs f32") {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("a duplicate declaration", func(t *testing.T) {
		b := rt.NewBuilder("d")
		tensor.Scalar(b, tensor.ScalarDesc{Name: "f", Kind: tensor.ScalarF32})
		tensor.Scalar(b, tensor.ScalarDesc{Name: "f", Kind: tensor.ScalarF32})
		if err := b.Err(); err == nil || !strings.Contains(err.Error(), "declared twice") {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("no name", func(t *testing.T) {
		b := rt.NewBuilder("n")
		tensor.Scalar(b, tensor.ScalarDesc{Kind: tensor.ScalarF32})
		if err := b.Err(); err == nil || !strings.Contains(err.Error(), "needs a name") {
			t.Errorf("unexpected error: %v", err)
		}
	})

	t.Run("an unknown kind", func(t *testing.T) {
		b := rt.NewBuilder("bk")
		tensor.Scalar(b, tensor.ScalarDesc{Name: "f", Kind: tensor.ScalarKind(9)})
		if err := b.Err(); err == nil || !strings.Contains(err.Error(), "f32 or u32") {
			t.Errorf("unexpected error: %v", err)
		}
	})

	// And at submission: the wrong kind is the case worth catching, because the
	// bytes pack either way and the kernel reads a float as an integer.
	b := rt.NewBuilder("s")
	tensor.Scalar(b, tensor.ScalarDesc{Name: "factor", Kind: tensor.ScalarF32})
	tensor.Output(b, "y", tensor.Scale(b, tensor.Input(b, value("x", 8)), "factor"))
	plan, err := b.Compile(rt, tensor.CompileOptions{})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	bufs := map[string]accel.BufferView{
		"x": f32Buffer(t, d, "x", make([]float32, 8)),
		"y": f32Buffer(t, d, "y", make([]float32, 8)),
	}
	for _, tc := range []struct {
		name    string
		scalars map[string]tensor.ScalarValue
		want    string
	}{
		{"unbound", nil, `the scalar "factor" is not bound`},
		{"the wrong kind", map[string]tensor.ScalarValue{"factor": tensor.U32(3)},
			"the bound value is u32"},
		{"a scalar nobody declared", map[string]tensor.ScalarValue{
			"factor": tensor.F32(1), "extra": tensor.F32(1),
		}, `the scalar "extra" is bound`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := plan.Submit(d.Queue(), tensor.Bindings{Buffers: bufs, Scalars: tc.scalars}).Wait()
			if err == nil {
				t.Fatalf("%s was accepted", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("the error should say %q, got %v", tc.want, err)
			}
		})
	}

	if got := tensor.F32(1.5).String(); got != "1.5" {
		t.Errorf("ScalarValue.String = %q", got)
	}
	if got := tensor.U32(7).String(); got != "7" {
		t.Errorf("ScalarValue.String = %q", got)
	}
	if got := tensor.ScalarKind(9).String(); !strings.Contains(got, "ScalarKind(9)") {
		t.Errorf("an unknown kind should say so, got %q", got)
	}
}
