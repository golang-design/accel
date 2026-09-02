// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels_test

import (
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/kernels"
	"golang.design/x/accel/kernelabi"
)

// The per-row kernels' authored forms agree with their lowerings.
//
// specs/064-per-row-sampling.md. Two rows with different parameters, so a
// form reading one row's parameter for both would show, over inputs with few
// significant bits, so a fused multiply-add in the authored form rounds where
// the lowering does (see the quantized matvec case in prefill_test.go).
func TestRowSamplingAuthoredFormsAgreeWithTheirLowerings(t *testing.T) {
	const rows, vocab = 2, 40
	d := kernels.RowSampleDims{Vocab: vocab, Rows: rows, History: 6, Slots: 2}
	weights := make([]float32, rows*vocab)
	for i := range weights {
		weights[i] = float32((i*37)%vocab) / 64
	}

	sameF32 := func(t *testing.T, authored, generated []float32) {
		t.Helper()
		for i := range authored {
			if authored[i] != generated[i] {
				t.Fatalf("element %d: authored %v, generated %v", i, authored[i], generated[i])
			}
		}
	}
	sameU32 := func(t *testing.T, authored, generated []uint32) {
		t.Helper()
		for i := range authored {
			if authored[i] != generated[i] {
				t.Fatalf("element %d: authored %d, generated %d", i, authored[i], generated[i])
			}
		}
	}
	perRow := func(t *testing.T, k *accel.Kernel, body func(th kernel.Thread)) {
		t.Helper()
		for r := uint32(0); r < rows; r++ {
			kernel.RunAuthored(k, kernel.ID3{X: r}, kernel.ID3{X: rows, Y: 1, Z: 1}, 128, body)
		}
	}

	t.Run("TopKMaskRows", func(t *testing.T) {
		ks := []uint32{3, 0}
		authored := make([]float32, rows*vocab)
		var best [128]float32
		var at [128]uint32
		perRow(t, &kernels.TopKMaskRowsKernel, func(th kernel.Thread) {
			kernels.TopKMaskRows(th, d, weights, ks, authored, &best, &at)
		})
		generated := make([]float32, rows*vocab)
		if err := kernel.DispatchCooperative(&kernels.TopKMaskRowsKernel, accel.ID3{X: rows},
			kernelabi.Args{Slices: []any{weights, ks, generated}, Uniforms: []any{d}}); err != nil {
			t.Fatal(err)
		}
		sameF32(t, authored, generated)
		if authored[vocab] == 0 && authored[vocab+1] == 0 {
			t.Fatal("row 1 with k = 0 should keep every entry")
		}
	})

	t.Run("TopPMaskRows", func(t *testing.T) {
		ps := []float32{0.5, 0}
		authored := make([]float32, rows*vocab)
		var best [128]float32
		var at [128]uint32
		perRow(t, &kernels.TopPMaskRowsKernel, func(th kernel.Thread) {
			kernels.TopPMaskRows(th, d, weights, ps, authored, &best, &at)
		})
		generated := make([]float32, rows*vocab)
		if err := kernel.DispatchCooperative(&kernels.TopPMaskRowsKernel, accel.ID3{X: rows},
			kernelabi.Args{Slices: []any{weights, ps, generated}, Uniforms: []any{d}}); err != nil {
			t.Fatal(err)
		}
		sameF32(t, authored, generated)
	})

	flat := func(k *accel.Kernel, n int, args kernelabi.Args) error {
		wg := int(k.WorkgroupSize.X)
		return kernel.Dispatch(k, accel.ID3{X: uint32((n + wg - 1) / wg)}, args)
	}

	t.Run("ScaleRows", func(t *testing.T) {
		factors := []float32{0, 1.5}
		authored := make([]float32, rows*vocab)
		for i := 0; i < rows*vocab; i++ {
			kernels.ScaleRows(flatThread(i, rows*vocab), d, weights, factors, authored)
		}
		generated := make([]float32, rows*vocab)
		if err := flat(&kernels.ScaleRowsKernel, rows*vocab, kernelabi.Args{
			Slices: []any{weights, factors, generated}, Uniforms: []any{d}}); err != nil {
			t.Fatal(err)
		}
		sameF32(t, authored, generated)
	})

	t.Run("SampleRows", func(t *testing.T) {
		draws := []float32{0.3, 0.7}
		factors := []float32{0, 1}
		authored := make([]uint32, rows)
		for i := 0; i < rows; i++ {
			kernels.SampleRows(flatThread(i, rows), d, weights, draws, factors, authored)
		}
		generated := make([]uint32, rows)
		if err := flat(&kernels.SampleRowsKernel, rows, kernelabi.Args{
			Slices: []any{weights, draws, factors, generated}, Uniforms: []any{d}}); err != nil {
			t.Fatal(err)
		}
		sameU32(t, authored, generated)
	})

	t.Run("LogitBias", func(t *testing.T) {
		ids := []uint32{5, vocab, 9, 9}
		vals := []float32{1, 0, 0.5, 0.25}
		authored := make([]float32, rows*vocab)
		for i := 0; i < rows*vocab; i++ {
			kernels.LogitBias(flatThread(i, rows*vocab), d, weights, ids, vals, authored)
		}
		generated := make([]float32, rows*vocab)
		if err := flat(&kernels.LogitBiasKernel, rows*vocab, kernelabi.Args{
			Slices: []any{weights, ids, vals, generated}, Uniforms: []any{d}}); err != nil {
			t.Fatal(err)
		}
		sameF32(t, authored, generated)
		if authored[vocab+9] != weights[vocab+9]+0.75 {
			t.Fatalf("row 1 id 9 with two slots is %v, want %v", authored[vocab+9], weights[vocab+9]+0.75)
		}
	})

	t.Run("the penalties", func(t *testing.T) {
		history := []uint32{1, 2, 2, 0, 0, 0, 7, 7, 7, 7, 3, 99}
		filled := []uint32{3, 6}
		rep := []float32{1.5, 1}
		pres := []float32{0, 0.5}
		freq := []float32{0.25, 0}

		counts := make([]uint32, rows*vocab)
		for i := range counts {
			counts[i] = 9
		}
		for i := 0; i < rows*vocab; i++ {
			kernels.PenaltyClearRows(flatThread(i, rows*vocab), d, counts)
		}
		for i := 0; i < rows*int(d.History); i++ {
			kernels.PenaltyCountRows(flatThread(i, rows*int(d.History)), d, history, filled, counts)
		}
		authored := make([]float32, rows*vocab)
		for i := 0; i < rows*vocab; i++ {
			kernels.PenaltyApplyRows(flatThread(i, rows*vocab), d, weights, counts, rep, pres, freq, authored)
		}

		gcounts := make([]uint32, rows*vocab)
		for i := range gcounts {
			gcounts[i] = 9
		}
		if err := flat(&kernels.PenaltyClearRowsKernel, rows*vocab, kernelabi.Args{
			Slices: []any{gcounts}, Uniforms: []any{d}}); err != nil {
			t.Fatal(err)
		}
		if err := flat(&kernels.PenaltyCountRowsKernel, rows*int(d.History), kernelabi.Args{
			Slices: []any{history, filled, gcounts}, Uniforms: []any{d}}); err != nil {
			t.Fatal(err)
		}
		sameU32(t, counts, gcounts)
		if counts[2] != 2 || counts[vocab+7] != 4 || counts[vocab+3] != 1 || counts[vocab+99%vocab] != 0 {
			t.Fatalf("the counts are not the history's: %v", counts)
		}
		generated := make([]float32, rows*vocab)
		if err := flat(&kernels.PenaltyApplyRowsKernel, rows*vocab, kernelabi.Args{
			Slices: []any{weights, gcounts, rep, pres, freq, generated}, Uniforms: []any{d}}); err != nil {
			t.Fatal(err)
		}
		sameF32(t, authored, generated)
	})
}
