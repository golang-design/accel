// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

// Format is a texture's pixel format.
//
// Formats are separate from [DType] even where they name the same width, because
// a texture format carries sampling and colour-space meaning a buffer dtype does
// not. Bytes per pixel always comes from the format: assuming a value is a real
// bug, and one that shows up as an out-of-range panic during readback rather than
// a wrong image. See docs/conventions.md.
type Format int

const (
	RGBA8Unorm Format = iota
	RGBA8UnormSRGB
	BGRA8Unorm
	R16Float
	RG16Float
	RGBA16Float
	R32Float
	RG32Float
	RGBA32Float

	// Depth formats carry backend constraints colour formats do not, including a
	// macOS requirement that they be device-private. The implementation enforces
	// that rather than the caller discovering it.
	Depth32Float
	Depth24PlusStencil8
)

// BytesPerPixel returns the format's size. Always ask rather than assume.
func (f Format) BytesPerPixel() int { panic(ErrNotImplemented) }

// IsDepth reports whether the format is a depth or depth-stencil format.
func (f Format) IsDepth() bool { panic(ErrNotImplemented) }

// String returns the format's name.
func (f Format) String() string { panic(ErrNotImplemented) }

// TextureUsage declares how a texture will be used.
type TextureUsage uint32

const (
	TextureSampled TextureUsage = 1 << iota
	TextureStorage
	TextureRenderTarget
	TextureCopySrc
	TextureCopyDst
)

// Extent is a texture's size in pixels.
type Extent struct{ Width, Height, Depth int }

// TextureDescriptor describes a texture to create.
type TextureDescriptor struct {
	Format Format
	Size   Extent
	Usage  TextureUsage

	// MipLevels of 0 means one level.
	MipLevels int

	// ArrayLayers of 0 means one layer.
	ArrayLayers int

	Label string
}

// Texture is an image in device memory.
type Texture struct{ _ noCopy }

// Format reports the texture's format.
func (t *Texture) Format() Format { panic(ErrNotImplemented) }

// Size reports the texture's extent.
func (t *Texture) Size() Extent { panic(ErrNotImplemented) }

// Read copies the texture back to the host, blocking until it is ready.
//
// Rows arrive in caller order on every backend. Both GL and Metal read render
// targets back bottom-origin natively, and Metal does so despite its top-left
// texture origin, so the backend flips rather than the caller. See
// docs/conventions.md.
func (t *Texture) Read(into []byte) error { panic(ErrNotImplemented) }

// Close releases the texture.
func (t *Texture) Close() error { panic(ErrNotImplemented) }

// FilterMode is how a sampler interpolates between texels.
type FilterMode int

const (
	FilterNearest FilterMode = iota
	FilterLinear
)

// AddressMode is how a sampler handles coordinates outside [0, 1].
type AddressMode int

const (
	AddressClampToEdge AddressMode = iota
	AddressRepeat
	AddressMirrorRepeat
)

// SamplerDescriptor describes a sampler to create.
type SamplerDescriptor struct {
	Min, Mag, Mip FilterMode
	AddressU      AddressMode
	AddressV      AddressMode
	AddressW      AddressMode
	Label         string
}

// Sampler describes how a texture is read in a kernel.
type Sampler struct{ _ noCopy }

// Close releases the sampler.
func (s *Sampler) Close() error { panic(ErrNotImplemented) }
