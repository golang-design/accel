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

// bf16Buffer uploads BFloat16 values as the bit patterns a bf16 buffer holds.
func bf16Buffer(t *testing.T, d *accel.Device, label string, vals []accel.BFloat16) accel.BufferView {
	t.Helper()
	bits := make([]uint16, len(vals))
	for i, v := range vals {
		bits[i] = v.Bits()
	}
	b, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.BF16, Count: len(vals), Label: label,
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

// Cast widens a bf16 weight to f32, exactly, and it then multiplies.
//
// # Why this is worth an operator
//
// A checkpoint ships bf16 -- Qwen3 does -- and nothing in the corpus read it,
// so a loader had to convert on the host: a readback, a loop, and an upload
// inside what should be one graph, or else a conversion done before the weights
// ever reach the device. bf16 to f32 is exact, because bf16 is f32's top half,
// so the conversion adds nothing to the model's error and the comparison here
// is equality rather than a tolerance.
//
// The cast feeding a MatMul is the shape a loader actually has, and it also
// checks the thing a widening kernel alone would not: that the widened value is
// f32 as far as the rest of the graph is concerned.
func TestCastWidensBF16Exactly(t *testing.T) {
	const k, n = 8, 4
	rt := newRuntime(t)
	d := rt.Device()

	// Magnitudes spanning f32's exponent, including one f16 cannot hold: that
	// range is bf16's reason to exist and the reason this widening goes to f32
	// rather than through f16.
	raw := make([]accel.BFloat16, k*n)
	want := make([]float32, len(raw))
	for i := range raw {
		v := float32(math.Ldexp(1+float64(i%8)/8, (i%9)*5-20))
		if i%2 == 1 {
			v = -v
		}
		raw[i] = accel.ToBFloat16(v)
		want[i] = raw[i].F32()
	}

	b := rt.NewBuilder("bf16cast")
	w := tensor.Weight(b, tensor.ValueDesc{
		Name: "w", DType: accel.BF16, Shape: tensor.Shape{k, n},
	})
	wide := tensor.Cast(b, w, accel.F32)
	tensor.Output(b, "wide", wide)

	plan, err := b.Compile(rt, tensor.CompileOptions{Label: "bf16cast"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer plan.Close()

	sel := plan.Selections()
	if len(sel) != 1 || sel[0].Kernel != "CastBF16ToF32" {
		t.Fatalf("selections are %+v, want the bf16 widening", sel)
	}
	if !strings.Contains(sel[0].Reason, "16-bit shift") {
		t.Errorf("the selection does not say why the widening is exact: %q", sel[0].Reason)
	}

	out := f32Buffer(t, d, "wide", make([]float32, len(raw)))
	f := plan.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"w": bf16Buffer(t, d, "w", raw), "wide": out,
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]float32, len(raw))
	if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("element %d widened to %v, want exactly %v; bf16 is f32's top "+
				"half, so a widening that is not equality is a shift done wrong",
				i, got[i], want[i])
		}
	}
	// And at least one value really is outside f16's range, or the spread above
	// says nothing about why the target is f32.
	var beyondF16 bool
	for _, v := range want {
		if math.IsInf(float64(accel.ToFloat16(v).F32()), 0) {
			beyondF16 = true
		}
	}
	if !beyondF16 {
		t.Error("no value in this test is outside f16's range, so it does not exercise " +
			"bf16's eight-bit exponent -- which is the reason the widening targets f32")
	}
}

// Narrowing bf16 is refused, and the refusal names the route.
//
// bf16 to f16 is the one lossy step this pipeline would have, since bf16 has
// f32's eight-bit exponent and f16 has five. Registering it would fold that
// loss into a conversion that looks like the others; refusing it makes a caller
// write the two casts and see where the error enters.
func TestNarrowingBF16IsRefusedWithTheRoute(t *testing.T) {
	rt := newRuntime(t)
	b := rt.NewBuilder("bf16narrow")
	w := tensor.Weight(b, tensor.ValueDesc{
		Name: "w", DType: accel.BF16, Shape: tensor.Shape{4, 4},
	})
	tensor.Cast(b, w, accel.F16)
	if err := b.Err(); err == nil {
		t.Fatal("bf16 to f16 was accepted")
	} else {
		msg := err.Error()
		if !strings.Contains(msg, "Cast to f32 first") {
			t.Errorf("the refusal should name the route, got %v", err)
		}
		if !strings.Contains(msg, "eight-bit exponent") {
			t.Errorf("the refusal should say what is lost, got %v", err)
		}
	}
}
