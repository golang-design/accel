// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels_test

import (
	"encoding/binary"
	"fmt"
	"golang.design/x/accel/kernelabi"
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/direct"
	"golang.design/x/accel/internal/conformance/numeq"
	"golang.design/x/accel/internal/kernels"
)

func sampleParams() kernels.Params {
	p := kernels.Params{
		Scale:  2.5,
		Origin: [3]float32{1, -2, 0.5},
		Steps:  24,
	}
	for c := range p.Inverse {
		for r := range p.Inverse[c] {
			p.Inverse[c][r] = float32(c*4+r) * 0.125
		}
	}
	return p
}

// TestCodecMatchesTheSpecExample is spec 001 section 3.3's worked example, as
// bytes rather than as a table.
//
// The offsets are the ones a reader would not guess, and they are the reason a
// codec is generated rather than a struct being cast: Steps occupies the tail of
// the sixteen-byte slot Origin's three components only half fill, so an unsafe
// cast puts it at 28 in std140's world and at 16 in Go's.
func TestCodecMatchesTheSpecExample(t *testing.T) {
	if got := kernels.ParamsBlockSize; got != 96 {
		t.Fatalf("ParamsBlockSize = %d, want 96", got)
	}

	p := sampleParams()
	dst := make([]byte, kernels.ParamsBlockSize)
	if err := (kernels.ParamsCodec{}).Encode(dst, p); err != nil {
		t.Fatalf("Encode: %v", err)
	}

	f32At := func(off int) float32 {
		return math.Float32frombits(binary.LittleEndian.Uint32(dst[off:]))
	}
	u32At := func(off int) uint32 { return binary.LittleEndian.Uint32(dst[off:]) }

	if got := f32At(0); got != p.Scale {
		t.Errorf("Scale at 0 is %v, want %v", got, p.Scale)
	}
	for i, want := range p.Origin {
		if got := f32At(16 + i*4); got != want {
			t.Errorf("Origin[%d] at %d is %v, want %v", i, 16+i*4, got, want)
		}
	}
	// The one that matters: a scalar after a three-vector shares its slot.
	if got := u32At(28); got != p.Steps {
		t.Errorf("Steps at 28 is %d, want %d: a scalar after a three-component vector occupies "+
			"the tail of the same sixteen-byte slot", got, p.Steps)
	}
	for c := range p.Inverse {
		for r := range p.Inverse[c] {
			off := 32 + c*16 + r*4
			if got := f32At(off); got != p.Inverse[c][r] {
				t.Errorf("Inverse[%d][%d] at %d is %v, want %v", c, r, off, got, p.Inverse[c][r])
			}
		}
	}
}

// TestCodecRefusesAShortBuffer checks the guard, since a caller who sized a
// buffer from anything but EncodedSize would otherwise write past it or write
// half a block.
func TestCodecRefusesAShortBuffer(t *testing.T) {
	err := (kernels.ParamsCodec{}).Encode(make([]byte, 32), sampleParams())
	if err == nil {
		t.Fatal("a short destination was accepted")
	}
}

// TestTransformMatchesAuthored is spec 004's fifth level for a kernel taking a
// uniform, whose loop bound comes from the uniform rather than from len.
func TestTransformMatchesAuthored(t *testing.T) {
	for _, n := range []int{0, 1, 24, 32, 40} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			p := sampleParams()
			in := make([]float32, n)
			for i := range in {
				in[i] = float32(i)*0.5 - 3
			}

			authored := make([]float32, n)
			runAuthoredKernel(&kernels.TransformKernel, n, func(th accel.Thread) {
				kernels.Transform(th, p, in, authored)
			})

			generated := make([]float32, n)
			if err := direct.Run(&kernels.TransformKernel,
				direct.Cover(&kernels.TransformKernel, n),
				kernelabi.Args{
					Slices:   []any{in, generated},
					Uniforms: []any{p},
				}); err != nil {
				t.Fatalf("direct.Run: %v", err)
			}

			if r := numeq.ExactBits(generated, authored, func(f float32) uint64 {
				return uint64(math.Float32bits(f))
			}); !r.Equal {
				t.Errorf("the generated lowering and the authored function disagree: %v", r)
			}
		})
	}
}

// TestUniformBoundIsRespected checks that the loop bound really comes from the
// uniform, which is the reason spec 001 calls this path required.
func TestUniformBoundIsRespected(t *testing.T) {
	const n = 40
	p := sampleParams()
	p.Steps = 10

	in := make([]float32, n)
	out := make([]float32, n)
	for i := range out {
		out[i] = -999
	}
	if err := direct.Run(&kernels.TransformKernel,
		direct.Cover(&kernels.TransformKernel, n),
		kernelabi.Args{Slices: []any{in, out}, Uniforms: []any{p}}); err != nil {
		t.Fatal(err)
	}

	for i := range out {
		if i < int(p.Steps) {
			if out[i] == -999 {
				t.Errorf("element %d was not written, and Steps is %d", i, p.Steps)
			}
			continue
		}
		if out[i] != -999 {
			t.Errorf("element %d was written past the uniform's Steps of %d", i, p.Steps)
		}
	}
}

// TestKernelDeclaresItsUniform checks that the record says what a caller has to
// supply, since a caller reads the record rather than the source.
func TestKernelDeclaresItsUniform(t *testing.T) {
	k := &kernels.TransformKernel
	if len(k.Uniforms) != 1 {
		t.Fatalf("%d uniforms, want 1", len(k.Uniforms))
	}
	u := k.Uniforms[0]
	if u.Name != "p" || u.Type != "Params" || u.Size != kernels.ParamsBlockSize {
		t.Errorf("uniform is %+v, want p of Params at %d bytes", u, kernels.ParamsBlockSize)
	}

	// A missing uniform is refused before anything runs.
	if err := k.Bind(kernelabi.Args{Slices: []any{[]float32{}, []float32{}}}); err == nil {
		t.Error("an argument set with no uniform was accepted")
	}
}

// TestUniformBufferRoundTrip covers the caller-facing path: allocate sized by
// the codec, write a value through a queue, read the block back.
func TestUniformBufferRoundTrip(t *testing.T) {
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()

	ub, err := accel.NewUniformBuffer[kernels.Params](d, kernels.ParamsCodec{})
	if err != nil {
		t.Fatalf("NewUniformBuffer: %v", err)
	}
	defer ub.Close()

	p := sampleParams()
	if err := ub.Write(d.Queue(), p); err != nil {
		t.Fatalf("Write: %v", err)
	}

	got := make([]byte, kernels.ParamsBlockSize)
	if err := d.Queue().ReadBuffer(ub.Buffer(), 0, got); err != nil {
		t.Fatalf("ReadBuffer: %v", err)
	}

	want := make([]byte, kernels.ParamsBlockSize)
	if err := (kernels.ParamsCodec{}).Encode(want, p); err != nil {
		t.Fatal(err)
	}
	if r := numeq.Exact(got, want); !r.Equal {
		t.Errorf("the block read back differs from the encoding: %v", r)
	}

	// The buffer is declared as bytes, which is one of exactly two places a
	// dtype means bytes rather than elements.
	if b := ub.Buffer(); b.DType() != accel.U8 || b.Count() != kernels.ParamsBlockSize {
		t.Errorf("the uniform buffer is %v of %d, want u8 of %d",
			b.DType(), b.Count(), kernels.ParamsBlockSize)
	}

	if _, err := ub.View(); err != nil {
		t.Errorf("View: %v", err)
	}
}

// TestUniformBufferRejectsAnOversizedBlock checks the device limit, using a
// mimicked profile so the path does not wait for hardware with a small one.
func TestUniformBufferRejectsAnOversizedBlock(t *testing.T) {
	small := accel.DeviceProfile{Info: accel.DeviceInfo{
		Name: "small-uniform",
		Limits: accel.Limits{
			MaxUniformBlockBytes: 64, MaxPools: 8, MaxPoolBytes: 1 << 20,
			MaxBufferBytes: 1 << 20, MinStorageBufferOffsetAlignment: 256,
			MinUniformBufferOffsetAlignment: 256, MinBufferCopyOffsetAlignment: 16,
		},
	}}
	d, err := accel.OpenCPU(accel.CPUOptions{Mode: accel.CPUMimic, Mimic: &small})
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()

	_, err = accel.NewUniformBuffer[kernels.Params](d, kernels.ParamsCodec{})
	if err == nil {
		t.Fatal("a 96-byte block was accepted on a device limited to 64")
	}
	if got := err.Error(); !contains(got, "MaxUniformBlockBytes") || !contains(got, "std140 pads") {
		t.Errorf("the error does not say what to do about it: %v", err)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
