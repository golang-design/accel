// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// Prefilling N tokens equals decoding them one at a time.
//
// specs/009-sequencing.md's M7 parity criterion, at the plan level rather than
// the kernel's. It is the criterion that matters most in a model runtime: if
// the two paths disagreed, the same prompt would produce different text
// depending on whether it had been prompted or generated, and the difference
// would appear only in the output of a model nobody could bisect.
//
// Two plans over one cache, which is the arrangement a real runtime has: a
// prefill plan for the prompt and a decode plan reused per token. They share
// the caller's buffers and nothing else -- the plans are compiled separately,
// select different kernels, and are told so by Selections.
//
// Equal within the softmax's reduction budget rather than bit for bit: the two
// kernels reduce over different numbers of lanes, so the additions happen in a
// different order, which specs/008-numerics.md section 7 bounds rather than
// forbids.
func TestPrefillAndDecodeAgree(t *testing.T) {
	const (
		qHeads   = 4
		kvHeads  = 2
		headDim  = 8
		width    = qHeads * headDim
		capacity = 8
		n        = 5 // tokens to prefill, then to decode one at a time
	)
	scale := float32(1 / math.Sqrt(headDim))

	// The same inputs for both paths.
	qs := make([][]float32, n)
	ks := make([][]float32, n)
	vs := make([][]float32, n)
	for s := range n {
		qs[s] = make([]float32, width)
		ks[s] = make([]float32, kvHeads*headDim)
		vs[s] = make([]float32, kvHeads*headDim)
		for i := range qs[s] {
			qs[s][i] = float32(math.Sin(float64(s*17+i) * 0.29))
		}
		for i := range ks[s] {
			ks[s][i] = float32(math.Cos(float64(s*11+i) * 0.19))
			vs[s][i] = float32((s*3+i)%7) - 3
		}
	}

	rt := newRuntime(t)
	d := rt.Device()

	kBuf := f32Buffer(t, d, "k", make([]float32, capacity*kvHeads*headDim))
	vBuf := f32Buffer(t, d, "v", make([]float32, capacity*kvHeads*headDim))

	// The prefill plan: n query tokens at once, over a cache holding n.
	prefill := func() []float32 {
		b := rt.NewBuilder("prefill")
		tensor.Scalar(b, tensor.ScalarDesc{Name: "len", Kind: tensor.ScalarU32})
		tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarU32})
		tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
		q := tensor.Input(b, tensor.ValueDesc{
			Name: "q", DType: accel.F32, Shape: tensor.Shape{n, qHeads, headDim},
		})
		kc := tensor.Persistent(b, tensor.StateDesc{
			Name: "k", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
		})
		vc := tensor.Persistent(b, tensor.StateDesc{
			Name: "v", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
		})
		tensor.Output(b, "out", tensor.Attention(b, q, kc, vc, tensor.AttentionOptions{
			CurrentLengthName: "len", ScaleName: "scale", BaseName: "base",
		}))
		plan, err := b.Compile(rt, tensor.CompileOptions{Label: "prefill"})
		if err != nil {
			t.Fatalf("compile prefill: %v", err)
		}
		defer plan.Close()
		if sel := plan.Selections()[0]; sel.Kernel != "AttentionPrefill" {
			t.Fatalf("the prefill plan selected %q", sel.Kernel)
		}

		// The cache is filled by the caller here rather than by a scatter,
		// because the prompt's keys and values are known before the plan runs
		// and this test is about attention rather than about the cache.
		flatK := make([]float32, capacity*kvHeads*headDim)
		flatV := make([]float32, capacity*kvHeads*headDim)
		flatQ := make([]float32, n*width)
		for s := range n {
			copy(flatK[s*kvHeads*headDim:], ks[s])
			copy(flatV[s*kvHeads*headDim:], vs[s])
			copy(flatQ[s*width:], qs[s])
		}
		if err := d.Queue().WriteBuffer(kBuf.Buffer, 0, flatK); err != nil {
			t.Fatalf("write k: %v", err)
		}
		if err := d.Queue().WriteBuffer(vBuf.Buffer, 0, flatV); err != nil {
			t.Fatalf("write v: %v", err)
		}

		out := f32Buffer(t, d, "out", make([]float32, n*width))
		f := plan.Submit(d.Queue(), tensor.Bindings{
			Buffers: map[string]accel.BufferView{
				"q": f32Buffer(t, d, "q", flatQ), "k": kBuf, "v": vBuf, "out": out,
			},
			Scalars: map[string]tensor.ScalarValue{
				"len": tensor.U32(n), "base": tensor.U32(0), "scale": tensor.F32(scale),
			},
		})
		if err := f.Wait(); err != nil {
			t.Fatalf("prefill: %v", err)
		}
		got := make([]float32, n*width)
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}
		return got
	}

	// The decode plan: one token, over a cache that grows by one each step.
	decode := func() []float32 {
		b := rt.NewBuilder("decode")
		tensor.Scalar(b, tensor.ScalarDesc{Name: "len", Kind: tensor.ScalarU32})
		tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
		q := tensor.Input(b, tensor.ValueDesc{
			Name: "q", DType: accel.F32, Shape: tensor.Shape{qHeads, headDim},
		})
		kc := tensor.Persistent(b, tensor.StateDesc{
			Name: "k", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
		})
		vc := tensor.Persistent(b, tensor.StateDesc{
			Name: "v", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
		})
		tensor.Output(b, "out", tensor.Attention(b, q, kc, vc, tensor.AttentionOptions{
			CurrentLengthName: "len", ScaleName: "scale",
		}))
		plan, err := b.Compile(rt, tensor.CompileOptions{Label: "decode"})
		if err != nil {
			t.Fatalf("compile decode: %v", err)
		}
		defer plan.Close()
		if sel := plan.Selections()[0]; sel.Kernel != "AttentionDecode" {
			t.Fatalf("the decode plan selected %q", sel.Kernel)
		}

		out := f32Buffer(t, d, "dout", make([]float32, width))
		all := make([]float32, n*width)
		for s := range n {
			f := plan.Submit(d.Queue(), tensor.Bindings{
				Buffers: map[string]accel.BufferView{
					"q": f32Buffer(t, d, "dq", qs[s]), "k": kBuf, "v": vBuf, "out": out,
				},
				Scalars: map[string]tensor.ScalarValue{
					// The cache holds every token, and the length is what makes
					// step s see only the first s+1 -- which is the same
					// masking the prefill does with its base.
					"len": tensor.U32(uint32(s + 1)), "scale": tensor.F32(scale),
				},
			})
			if err := f.Wait(); err != nil {
				t.Fatalf("decode step %d: %v", s, err)
			}
			step := make([]float32, width)
			if err := d.Queue().ReadBuffer(out.Buffer, 0, step); err != nil {
				t.Fatalf("readback: %v", err)
			}
			copy(all[s*width:], step)
		}
		return all
	}

	// The prefill first, because it fills the cache the decode reads.
	p := prefill()
	dec := decode()

	for i := range p {
		if diff := math.Abs(float64(p[i] - dec[i])); diff > 1e-5*(1+math.Abs(float64(dec[i]))) {
			t.Fatalf("token %d element %d: prefill gives %v and decode gives %v; the two "+
				"paths must compute the same function, or a model produces different text "+
				"depending on whether it was prompted or generated",
				i/width, i%width, p[i], dec[i])
		}
	}

	// And that they are not both zero, which would satisfy the comparison and
	// mean neither ran.
	nonzero := 0
	for _, v := range p {
		if v != 0 {
			nonzero++
		}
	}
	if nonzero < len(p)/2 {
		t.Fatalf("only %d of %d prefill outputs are non-zero", nonzero, len(p))
	}
}
