// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels_test

import (
	"encoding/binary"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/direct"
	"golang.design/x/accel/internal/kernels"
	"golang.design/x/accel/kernelabi"
)

func sampleMatrixParams() kernels.MatrixParams {
	var p kernels.MatrixParams
	for c := range p.Wide {
		for r := range p.Wide[c] {
			p.Wide[c][r] = float32(10*c+r) + 0.5
		}
	}
	for c := range p.Tall {
		for r := range p.Tall[c] {
			p.Tall[c][r] = -float32(10*c+r) - 0.25
		}
	}
	for i := range p.Column {
		p.Column[i] = float32(100 + i)
	}
	return p
}

// TestMatrixCodecWritesEveryElementAtItsColumnStride is the host half of the
// device check: every element of a non-square matrix lands at column times
// sixteen plus row times four, and an array element at its own sixteen-byte
// slot.
//
// Before the layout carried rows and columns separately the encoder looped
// the column count both ways: Wide[c][2] and Wide[c][3] were never written,
// and Tall[c][2] and Tall[c][3] were written from beyond the Go array.
func TestMatrixCodecWritesEveryElementAtItsColumnStride(t *testing.T) {
	if got := kernels.MatrixParamsBlockSize; got != 32+64+96 {
		t.Fatalf("MatrixParamsBlockSize = %d, want %d", got, 32+64+96)
	}
	p := sampleMatrixParams()
	dst := make([]byte, kernels.MatrixParamsBlockSize)
	if err := (kernels.MatrixParamsCodec{}).Encode(dst, p); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	at := func(off int) float32 {
		return math.Float32frombits(binary.LittleEndian.Uint32(dst[off:]))
	}
	for c := range 2 {
		for r := range 4 {
			if got, want := at(c*16+r*4), p.Wide[c][r]; got != want {
				t.Errorf("Wide[%d][%d] at %d is %v, want %v", c, r, c*16+r*4, got, want)
			}
		}
	}
	for c := range 4 {
		for r := range 2 {
			if got, want := at(32+c*16+r*4), p.Tall[c][r]; got != want {
				t.Errorf("Tall[%d][%d] at %d is %v, want %v", c, r, 32+c*16+r*4, got, want)
			}
		}
		// The two rows a square reading would have written are padding and
		// stay zero: they are past the end of the Go array.
		for r := 2; r < 4; r++ {
			if got := at(32 + c*16 + r*4); got != 0 {
				t.Errorf("padding at %d is %v, want 0", 32+c*16+r*4, got)
			}
		}
	}
	for i := range 6 {
		if got, want := at(96+i*16), p.Column[i]; got != want {
			t.Errorf("Column[%d] at %d is %v, want %v", i, 96+i*16, got, want)
		}
	}
}

// TestMatrixShapesMatchesAuthored runs the generated lowering against the
// authored kernel on the CPU. The Metal half is the differential corpus, which
// is where a codec disagreeing with the block declaration shows.
func TestMatrixShapesMatchesAuthored(t *testing.T) {
	const n = 24
	p := sampleMatrixParams()

	authored := make([]float32, n)
	runAuthoredKernel(&kernels.MatrixShapesKernel, n, func(th accel.Thread) {
		kernels.MatrixShapes(th, p, authored)
	})

	generated := make([]float32, n)
	if err := direct.Run(&kernels.MatrixShapesKernel,
		direct.Cover(&kernels.MatrixShapesKernel, n),
		kernelabi.Args{Slices: []any{generated}, Uniforms: []any{p}}); err != nil {
		t.Fatalf("direct.Run: %v", err)
	}
	for i := range n {
		if generated[i] != authored[i] {
			t.Errorf("out[%d] = %v generated, %v authored", i, generated[i], authored[i])
		}
	}
	want := []float32{0.5, 1.5, 2.5, 3.5, 10.5, 11.5, 12.5, 13.5,
		-0.25, -1.25, -10.25, -11.25, -20.25, -21.25, -30.25, -31.25,
		100, 101, 102, 103, 104, 105, 0, 0}
	for i := range want {
		if generated[i] != want[i] {
			t.Errorf("out[%d] = %v, want %v", i, generated[i], want[i])
		}
	}
}
