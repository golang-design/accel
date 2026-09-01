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
	// The constants below are the formats' own names, unprefixed, because that
	// is how they read at the point of use: `Format: accel.RGBA8Unorm`. The
	// zero value is the one exception, and carries the type name because "no
	// format" has no name of its own.
	//
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

	// Depth32FloatStencil8 is the one depth format with a stencil aspect whose
	// layout is not device-defined: 32-bit float depth, then one byte of
	// stencil, then three reserved bytes that are written zero. It exists
	// because specs/033-render-api.md section 2.1's stencil state needs a
	// format that can carry it, and Depth24PlusStencil8 cannot -- that one is
	// refused precisely because "24 plus" has two defensible encodings, and
	// this is the answer to what the refusal left missing.
	Depth32FloatStencil8
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

	// MipLevels of 0 means one level.
	//
	// Each level halves both axes and never goes below one, so a 16x8 texture
	// has five: 16x8, 8x4, 4x2, 2x1, 1x1. More than the extent has is refused
	// naming the number it does have, because the off-by-one at the end of a
	// chain is where a mip count is usually wrong.
	//
	// [Texture.Subresource] states where each level's bytes are, and a
	// [TextureView] naming one is what a pass renders into, a stage fetches
	// from, and the graph hazards against. What still addresses the base level
	// only is the host copy -- [Queue.ReadTexture] and the recorded
	// texture-buffer copies take a texture rather than a view, so there is no
	// subresource for a caller to name.
	MipLevels int

	// ArrayLayers of 0 means one layer. Layers are consecutive within a level,
	// and more than the device reports is refused naming both numbers.
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
//
// Closing while something using this texture is still outstanding is reported
// rather than crashing, by the rule [Buffer.Close] follows: the texture stays
// valid until every hold on it is gone, its memory comes back then, and the
// caller learns that their teardown ordering was wrong. Returning nil here
// was the version that forgot the texture and left the pool counting a child
// nobody could reach.
func (t *Texture) Close() error {
	if !t.state.beginClose() {
		return nil
	}
	if t.state.release() {
		return t.free()
	}
	return &LifetimeError{
		Op:       "Close",
		Resource: t.desc.Label,
		Reason:   reasonPending,
		InFlight: t.state.holds(),
	}
}

// release drops one hold, freeing the texture if it was the last.
func (t *Texture) release() {
	if t.state.release() {
		_ = t.free()
	}
}

// free returns the texture's memory to its pool, and closes the pool when the
// texture owns it. It runs exactly once, when the last hold goes away, which
// is either inside Close or inside the path that completes the work still
// holding it.
func (t *Texture) free() error {
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
	if t.ownsPool {
		return p.Close()
	}
	return nil
}
