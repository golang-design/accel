// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"fmt"
	"strings"

	"golang.design/x/accel/internal/alloc"
)

// Format is a texture's pixel format.
//
// Formats are separate from [DType] even where they name the same width, because
// a texture format carries sampling and colour-space meaning a buffer dtype does
// not. Bytes per pixel always comes from the format: assuming a value is a real
// bug, and one that shows up as an out-of-range panic during readback rather than
// a wrong image. See docs/conventions.md.
type Format int

const (
	// FormatInvalid is not a creatable format. It is the zero-value sentinel used
	// by optional format constraints such as a graph slot.
	FormatInvalid Format = iota
	RGBA8Unorm
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

// TextureUsage declares how a texture will be used.
type TextureUsage uint32

const (
	TextureSampled TextureUsage = 1 << iota
	TextureStorage
	TextureRenderTarget
	TextureCopySrc
	TextureCopyDst
)

var textureUsageNames = []string{"TextureSampled", "TextureStorage",
	"TextureRenderTarget", "TextureCopySrc", "TextureCopyDst"}

// String names the usages set, so a diagnostic reads as what a caller wrote
// rather than as a number they have to decode.
func (u TextureUsage) String() string {
	if u == 0 {
		return "no usage"
	}
	var parts []string
	for i, name := range textureUsageNames {
		if u&(1<<uint(i)) != 0 {
			parts = append(parts, name)
		}
	}
	if rest := u &^ (1<<uint(len(textureUsageNames)) - 1); rest != 0 {
		parts = append(parts, fmt.Sprintf("TextureUsage(%d)", uint32(rest)))
	}
	return strings.Join(parts, "|")
}

// Extent is a texture's size in pixels.
type Extent struct{ Width, Height, Depth int }

// TextureDescriptor describes a texture to create.
type TextureDescriptor struct {
	Format Format
	Size   Extent
	Usage  TextureUsage

	// MipLevels of 0 means one level. v0 rejects values greater than one until the
	// public API can name subresources.
	MipLevels int

	// ArrayLayers of 0 means one layer. v0 rejects values greater than one until
	// the public API can name subresources.
	ArrayLayers int

	Label string
}

// Texture is an image in device memory.
type Texture struct {
	_ noCopy

	pool  *Pool
	desc  TextureDescriptor
	alloc *alloc.Allocation
	bytes int
	state resourceState
}

// Format reports the texture's format.
func (t *Texture) Format() Format { return t.desc.Format }

// Size reports the texture's extent.
func (t *Texture) Size() Extent { return t.desc.Size }

// Bytes reports the texture's device footprint, which counts the padding a
// backend's row alignment adds and is therefore at least width*height*bpp.
func (t *Texture) Bytes() int { return t.bytes }

// Close releases the texture.
func (t *Texture) Close() error {
	if !t.state.beginClose() {
		return nil
	}
	if t.state.release() {
		t.free()
	}
	return nil
}

func (t *Texture) free() {
	p := t.pool
	p.mu.Lock()
	// A linear pool frees by Reset, so an individual free is a no-op against
	// the memory. The accounting still retires, because the handle is gone
	// either way -- the same rule a buffer follows.
	if p.desc.Policy == PoolGeneral {
		_ = p.alloc.Free(t.alloc)
	}
	for i, o := range p.liveTextures {
		if o == t {
			p.liveTextures = append(p.liveTextures[:i], p.liveTextures[i+1:]...)
			break
		}
	}
	p.mu.Unlock()
}

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
