// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels_test

import (
	"encoding/binary"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/kernels"
	"golang.design/x/accel/kernelabi"
)

// PackParams encodes to the std140 layout the emitted MSL declares.
//
// # Why this is worth a test of its own
//
// The block carries two eight-element arrays, and std140 gives an array member
// a sixteen-byte stride *whatever its element type* — so eight uint32 occupy
// 128 bytes, not 32. A reader guessing the layout from the Go struct is wrong
// by a factor of four, and an unsafe cast would put every extent and stride
// somewhere the shader does not look.
//
// Nothing else checks it on a machine without Metal: the codec exists so the
// Metal backend can fill a uniform block, so on every other platform it is
// generated code that runs nowhere. That is exactly the code a golden cannot
// vouch for — it compiles either way.
func TestPackParamsEncodesToStd140(t *testing.T) {
	// One uint32 per sixteen-byte slot for each array, plus the three scalars
	// at the front. The scalars pack tightly; the arrays do not.
	const scalars = 3 * 4
	const arrays = 2 * kernels.PackRank * 16
	if got, want := kernels.PackParamsBlockSize, scalars+4+arrays; got != want {
		t.Fatalf("PackParamsBlockSize is %d, want %d: three uint32 padded to a "+
			"sixteen-byte boundary, then two arrays at a sixteen-byte stride each",
			got, want)
	}

	p := kernels.PackParams{Rank: 3, Count: 24, Offset: 5}
	for i := range kernels.PackRank {
		p.Extent[i] = uint32(100 + i)
		p.Stride[i] = uint32(200 + i)
	}

	dst := make([]byte, kernels.PackParamsBlockSize)
	if err := (kernels.PackParamsCodec{}).Encode(dst, p); err != nil {
		t.Fatalf("Encode: %v", err)
	}
	u32 := func(off int) uint32 { return binary.LittleEndian.Uint32(dst[off:]) }

	for _, c := range []struct {
		name string
		off  int
		want uint32
	}{
		{"Rank", 0, 3}, {"Count", 4, 24}, {"Offset", 8, 5},
	} {
		if got := u32(c.off); got != c.want {
			t.Errorf("%s is %d at offset %d, want %d", c.name, got, c.off, c.want)
		}
	}

	// The arrays, each element in its own sixteen-byte slot. This is the part
	// a reader would get wrong, so it is asserted element by element rather
	// than as a length.
	base := scalars + 4
	for i := range kernels.PackRank {
		if got := u32(base + i*16); got != uint32(100+i) {
			t.Errorf("Extent[%d] is %d at offset %d, want %d; a std140 array member has "+
				"a sixteen-byte stride whatever its element type",
				i, got, base+i*16, 100+i)
		}
	}
	base += kernels.PackRank * 16
	for i := range kernels.PackRank {
		if got := u32(base + i*16); got != uint32(200+i) {
			t.Errorf("Stride[%d] is %d at offset %d, want %d", i, got, base+i*16, 200+i)
		}
	}

	// A destination that is too small is refused rather than truncating: a
	// short write would leave the tail of the block holding whatever was there,
	// and the shader would read it as strides.
	if err := (kernels.PackParamsCodec{}).Encode(dst[:8], p); err == nil {
		t.Error("Encode wrote into a buffer smaller than the block")
	}
	if got := (kernels.PackParamsCodec{}).EncodedSize(); got != kernels.PackParamsBlockSize {
		t.Errorf("EncodedSize is %d and the block is %d", got, kernels.PackParamsBlockSize)
	}
}

// The pack kernel gathers a strided view, checked at the corpus level rather
// than only through the tensor layer.
//
// Every axis has a distinct extent and a stride that is not the contiguous one,
// so a kernel that ignored the strides, or that decomposed the linear index
// from the wrong end, produces a differently-ordered tensor of the right size —
// which a length check would accept.
func TestPackGathersAStridedView(t *testing.T) {
	// A 2x3x4 view over a 2x3x4 source read in reverse axis order, which is
	// what a double transpose produces.
	const d0, d1, d2 = 2, 3, 4
	src := make([]float32, d0*d1*d2)
	for i := range src {
		src[i] = float32(i)
	}

	p := kernels.PackParams{Rank: 3, Count: d0 * d1 * d2}
	// Destination axis order (d2, d1, d0) over a source laid out (d0, d1, d2).
	p.Extent[0], p.Extent[1], p.Extent[2] = d2, d1, d0
	p.Stride[0], p.Stride[1], p.Stride[2] = 1, d2, d1*d2

	dst := make([]float32, len(src))
	err := kernel.Dispatch(&kernels.PackKernel, accel.ID3{X: uint32(len(src))},
		kernelabi.Args{Slices: []any{src, dst}, Uniforms: []any{p}})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}

	// The transpose computed here: destination (a,b,c) is source (c,b,a).
	for a := range d2 {
		for b := range d1 {
			for c := range d0 {
				got := dst[(a*d1+b)*d0+c]
				want := src[(c*d1+b)*d2+a]
				if got != want {
					t.Fatalf("destination (%d,%d,%d) is %v, want %v; the gather read the "+
						"wrong element rather than the wrong count", a, b, c, got, want)
				}
			}
		}
	}
}

// The offset is applied, which a transpose alone does not exercise.
func TestPackAppliesTheOffset(t *testing.T) {
	src := make([]float32, 20)
	for i := range src {
		src[i] = float32(i)
	}
	p := kernels.PackParams{Rank: 1, Count: 4, Offset: 7}
	p.Extent[0], p.Stride[0] = 4, 2

	dst := make([]float32, 4)
	err := kernel.Dispatch(&kernels.PackKernel, accel.ID3{X: 4},
		kernelabi.Args{Slices: []any{src, dst}, Uniforms: []any{p}})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	for i := range dst {
		if want := float32(7 + i*2); dst[i] != want {
			t.Errorf("element %d is %v, want %v; the offset or the stride was dropped",
				i, dst[i], want)
		}
	}
}
