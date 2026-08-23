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

	// Kind is the memory the implicit pool behind [Device.NewTexture] is taken
	// from. The zero value is [MemoryDevice], which is the right default and the
	// wrong one for exactly one thing: [Queue.ReadTexture] needs mappable
	// memory, so a texture you intend to read back on the host must ask for
	// [MemoryReadback] here.
	//
	// The field exists because without it the convenience constructor produced
	// a texture the only readback method could never read, and nothing said so
	// until the call failed. It is ignored by [Pool.AllocTexture], where the
	// pool already fixed the answer.
	Kind MemoryKind

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

	// ownsPool marks a texture from [Device.NewTexture], whose pool exists only
	// for it and is closed with it.
	ownsPool bool
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
	if t.ownsPool {
		return t.pool.Close()
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

// The sampler family is withdrawn from the public API until there is a render
// pass to sample in.
//
// It was exported with no producer, no consumer, and a Close that panicked, so
// a caller who found it could construct a descriptor, never obtain a *sampler,
// and crash if they somehow did. Naming a type in a public API is a promise to
// keep it; this one promised something specs/005-graphics.md has not designed
// the shape of yet.
//
// specs/033-render-api.md and specs/032-stage-abi.md decide what it becomes.
// 032 admits an integer texel fetch and continues to refuse filtered sampling,
// on the evidence specs/004-kernel-authoring.md records, so the eventual public
// shape may not be a sampler at all.
type filterMode int

const (
	filterNearest filterMode = iota
	filterLinear
)

type addressMode int

const (
	addressClampToEdge addressMode = iota
	addressRepeat
	addressMirrorRepeat
)

type samplerDescriptor struct {
	Min, Mag, Mip filterMode
	AddressU      addressMode
	AddressV      addressMode
	AddressW      addressMode
	Label         string
}

// sampler describes how a texture is read in a kernel.
type sampler struct{ _ noCopy }
