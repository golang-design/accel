// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package tensor_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// The composed sampling pipeline emits the same token on both backends.
//
// specs/039-sampling-policy.md section 9's last assertion, and the one the
// corpus differential cannot make. That differential compares each kernel's two
// lowerings, and every one of them passes here; what it cannot see is the
// composition, where a difference in *where* a boundary falls decides a
// different token from arithmetic that agreed everywhere.
//
// The comparison is on the token, not on the distribution, and that is the
// point. Two backends whose probabilities differ by an ulp emit the same token
// almost always -- so a distribution asserted within a ULP ceiling would pass
// while the thing a caller reads disagreed. The distribution below is built to
// make the boundaries decidable: a plateau at the top-k edge and a nucleus
// boundary a differing tie rule would move.
func TestASampledTokenAgreesOnCPUAndMetal(t *testing.T) {
	const vocab = 256
	const historyCap = 32

	// A distribution with a deliberate plateau where top-k cuts, so the mask
	// keeps a set only a shared tie rule makes well defined, and a long tail
	// that top-p has to stop inside.
	logits := make([]float32, vocab)
	for i := range logits {
		switch {
		case i%16 == 3:
			// Sixteen entries share the boundary value.
			logits[i] = 4.0
		default:
			logits[i] = float32(math.Sin(float64(i)*0.37)) * 2
		}
	}
	history := make([]uint32, historyCap)
	for i := range history {
		history[i] = uint32(i*7) % vocab
	}

	o := tensor.SamplingOptions{
		Temperature: 0.9, TopK: 12, TopP: 0.85,
		Repetition: 1.2, Presence: 0.15, Frequency: 0.05,
	}

	run := func(t *testing.T, d *accel.Device, draw float32) uint32 {
		t.Helper()
		rt, err := tensor.NewRuntime(d)
		if err != nil {
			t.Fatalf("runtime: %v", err)
		}
		defer rt.Close()
		b := rt.NewBuilder("policy")
		tensor.DeclareSamplingScalars(b, o, "s")
		in := tensor.Input(b, tensor.ValueDesc{
			Name: "logits", DType: accel.F32, Shape: tensor.Shape{vocab},
		})
		draws := tensor.Input(b, tensor.ValueDesc{
			Name: "draws", DType: accel.F32, Shape: tensor.Shape{1},
		})
		hist := tensor.NewState(b, tensor.StateDesc{
			Name: "history", DType: accel.U32, Shape: tensor.Shape{historyCap},
		})
		counts := tensor.NewState(b, tensor.StateDesc{
			Name: "counts", DType: accel.U32, Shape: tensor.Shape{vocab},
		})
		tensor.Output(b, "token", tensor.Sample(b, in, draws, hist, counts, o, "s"))

		plan, err := b.Compile(rt, tensor.CompileOptions{Label: "policy"})
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		defer plan.Close()
		sc, err := o.Scalars("s", historyCap, historyCap)
		if err != nil {
			t.Fatalf("scalars: %v", err)
		}
		out := u32Buffer(t, d, "token", make([]uint32, 1))
		f := plan.Submit(d.Queue(), tensor.Bindings{
			Buffers: map[string]accel.BufferView{
				"logits":  f32Buffer(t, d, "logits", logits),
				"draws":   f32Buffer(t, d, "draws", []float32{draw}),
				"history": u32Buffer(t, d, "history", history),
				"counts":  u32Buffer(t, d, "counts", make([]uint32, vocab)),
				"token":   out,
			},
			Scalars: sc,
		})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		got := make([]uint32, 1)
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}
		return got[0]
	}

	cpuDev, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open CPU: %v", err)
	}
	defer cpuDev.Close()
	gpuDev := openMetalRuntimeDevice(t)

	// Several draws, because one draw reads one point of the cumulative walk
	// and the boundaries are where the backends could differ. A draw close to 1
	// lands at the far end of the kept set, which is where top-p stopped.
	stream := tensor.Stream{Seed: 0x5DEECE66D}
	seen := map[uint32]bool{}
	for step := range uint64(24) {
		draw := stream.Draw(step)
		cpu := run(t, cpuDev, draw)
		gpu := run(t, gpuDev, draw)
		if cpu != gpu {
			t.Fatalf("draw %v (step %d) sampled token %d on the CPU backend and %d on "+
				"Metal; the kernels agree bit for bit, so this is the composition -- a "+
				"boundary that falls one entry differently decides a different token "+
				"from arithmetic that matched", draw, step, cpu, gpu)
		}
		if cpu >= vocab {
			t.Fatalf("step %d sampled token %d, which is outside a vocabulary of %d",
				step, cpu, vocab)
		}
		seen[cpu] = true
	}

	// Without this the test passes for a sampler that returns one token
	// whatever the draw -- which is what a mask keeping nothing, or a walk
	// whose comparison is always false, both produce. Agreement between two
	// backends that are equally wrong is not evidence.
	if len(seen) < 2 {
		t.Fatalf("24 draws sampled %d distinct token(s): the draw is not reaching the "+
			"walk, so the two backends agree about a constant", len(seen))
	}
}
