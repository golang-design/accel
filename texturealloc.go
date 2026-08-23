// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import "fmt"

// allocTexture suballocates a texture from a texture pool.
//
// # Why textures get their own pools
//
// Texture placement alignment is far coarser than any buffer alignment — 64 KiB
// on D3D12 against 256 bytes for a bound storage range — so a pool holding both
// would either pad every buffer to the texture alignment or track two alignment
// classes in one allocator. specs/001-device-resources.md section 4.4 makes a
// pool one or the other, and [Pool.Alloc] refuses a texture pool for the same
// reason this refuses a buffer one.
func (p *Pool) allocTexture(desc TextureDescriptor) (*Texture, error) {
	if err := p.state.checkOpen("AllocTexture"); err != nil {
		return nil, err
	}
	if !p.desc.Textures {
		return nil, fmt.Errorf("%w: AllocTexture %q: pool %q holds buffers. Texture "+
			"placement alignment is far coarser than any buffer alignment, so a pool is "+
			"one or the other and never both (spec 001 section 4.4)",
			ErrUsage, desc.Label, p.desc.Label)
	}
	if err := validateTexture(p.dev, desc); err != nil {
		return nil, err
	}

	size := textureBytes(p.dev, desc)
	align := p.dev.info.Limits.MinTexturePlacementAlignment

	p.mu.Lock()
	a, err := p.alloc.Alloc(size, align)
	if err != nil {
		s := p.alloc.Stats()
		p.mu.Unlock()
		return nil, &AllocError{
			Label: desc.Label, Pool: p.desc.Label, Kind: p.desc.Kind,
			Requested: size, Alignment: align,
			Free: s.Free, LargestFree: s.LargestFree, PoolSize: s.Size,
		}
	}
	t := &Texture{pool: p, desc: desc, alloc: a, bytes: size}
	t.state.init(desc.Label)
	p.liveTextures = append(p.liveTextures, t)
	p.mu.Unlock()
	return t, nil
}

// validateTexture is every rule a descriptor has to satisfy.
func validateTexture(d *Device, desc TextureDescriptor) error {
	if !desc.Format.valid() {
		return fmt.Errorf("%w: texture %q: %v is not a creatable format",
			ErrFormat, desc.Label, desc.Format)
	}
	if desc.Size.Width <= 0 || desc.Size.Height <= 0 {
		return fmt.Errorf("accel: texture %q: extent %dx%d has a non-positive axis",
			desc.Label, desc.Size.Width, desc.Size.Height)
	}
	if d := desc.Size.Depth; d > 1 {
		return fmt.Errorf("accel: texture %q: depth %d is a 3D texture, and every v0 "+
			"operation addresses a single layer (spec 001 section 4)", desc.Label, d)
	}
	// Zero normalizes to one; anything larger is rejected until the public API
	// can name a subresource. A texture whose mips could be created but not
	// bound, copied, or transitioned is worse than one that cannot be created.
	if desc.MipLevels > 1 {
		return fmt.Errorf("accel: texture %q: %d mip levels, and v0 addresses the base "+
			"level only: mip chains are deferred until the API can name a subresource "+
			"in creation, binding, copying, hazards, and layout (spec 001 section 4)",
			desc.Label, desc.MipLevels)
	}
	if desc.ArrayLayers > 1 {
		return fmt.Errorf("accel: texture %q: %d array layers, and v0 addresses a single "+
			"layer for the same reason mip chains are deferred (spec 001 section 4)",
			desc.Label, desc.ArrayLayers)
	}
	if desc.Usage == 0 {
		return fmt.Errorf("%w: texture %q declares no usage, so nothing can be done "+
			"with it", ErrUsage, desc.Label)
	}

	info := d.FormatInfo(desc.Format)
	for _, c := range []struct {
		usage TextureUsage
		have  bool
		what  string
	}{
		{TextureRenderTarget, info.Renderable, "used as a render target"},
		{TextureSampled, info.Sampleable, "sampled"},
		{TextureStorage, info.StorageRead || info.StorageWrite, "used as a storage image"},
	} {
		if desc.Usage&c.usage != 0 && !c.have {
			return fmt.Errorf("%w: texture %q: %v cannot be %s on %q",
				ErrFormat, desc.Label, desc.Format, c.what, d.info.Name)
		}
	}

	lim := d.info.Limits
	if w, h := desc.Size.Width, desc.Size.Height; w > lim.MaxTextureExtent2D ||
		h > lim.MaxTextureExtent2D {
		return fmt.Errorf("accel: texture %q: %dx%d exceeds %q's %d limit",
			desc.Label, w, h, d.info.Name, lim.MaxTextureExtent2D)
	}
	return nil
}

// textureBytes is a texture's device footprint.
//
// The *aligned* row pitch, not the tight one: this is what the device stores,
// and the tight packing spec 001 section 4.2 guarantees is a property of the
// API boundary rather than of the allocation. Sizing from the tight pitch would
// under-allocate on any backend that pads.
func textureBytes(d *Device, desc TextureDescriptor) int {
	pitch := d.AlignedRowPitch(desc.Format, desc.Size.Width)
	if pitch == 0 {
		// A device-defined layout, as for Depth24PlusStencil8. Four bytes per
		// pixel is the largest any backend uses for it, so this over-allocates
		// rather than under-allocating, and over-allocating is the direction
		// that is merely wasteful.
		pitch = alignUp(desc.Size.Width*4, d.info.Limits.MinBufferCopyRowPitchAlignment)
	}
	return pitch * desc.Size.Height
}

// readTexture returns the base level as tightly packed rows.
//
// # The guarantee, and who pays for it
//
// Row r begins at r*width*bpp with no padding, so a caller sizes a readback as
// width*height*bpp and is always right. The device may store rows padded to its
// own alignment; where it does, the repack happens here, in an intermediate the
// caller never sees.
//
// The cost is one extra full-size copy of the image, paid by the caller whose
// row length is not already aligned and by nobody else — and it is counted in
// [QueueStats] rather than returned, because a cost that is proportional to the
// image and otherwise silent is the wrong kind of silent on a performance path.
func (q *Queue) readTexture(src *Texture, into []byte) error {
	if src == nil {
		return fmt.Errorf("accel: ReadTexture: no texture")
	}
	if err := src.state.checkOpen("ReadTexture"); err != nil {
		return err
	}
	if src.pool.dev != q.dev {
		return fmt.Errorf("accel: ReadTexture %q: the texture belongs to a different device",
			src.desc.Label)
	}
	if !q.dev.FormatInfo(src.desc.Format).HostCopyable {
		return fmt.Errorf("%w: ReadTexture %q: %v is device-private, which several "+
			"backends require of depth formats and this one enforces so the rule is not "+
			"discovered in production (spec 001 section 4)",
			ErrFormat, src.desc.Label, src.desc.Format)
	}
	if src.desc.Usage&TextureCopySrc == 0 {
		return fmt.Errorf("%w: ReadTexture %q: it needs %v and was created with %v",
			ErrUsage, src.desc.Label, TextureCopySrc, src.desc.Usage)
	}

	tight := tightRowPitch(src.desc.Format, src.desc.Size.Width)
	want := tight * src.desc.Size.Height
	if len(into) != want {
		return fmt.Errorf("accel: ReadTexture %q: the destination is %d bytes and a "+
			"tightly packed %dx%d %v is %d",
			src.desc.Label, len(into), src.desc.Size.Width, src.desc.Size.Height,
			src.desc.Format, want)
	}

	// Everything queued ahead of this must land first, or a readback would
	// report the image as it was before the writes a caller just made.
	if err := q.Flush().Wait(); err != nil {
		return err
	}

	mem := src.pool.block.Bytes()
	if mem == nil {
		return fmt.Errorf("accel: ReadTexture %q: its pool is device-local and cannot be "+
			"mapped", src.desc.Label)
	}
	pitch := q.dev.AlignedRowPitch(src.desc.Format, src.desc.Size.Width)
	base := src.alloc.Offset

	q.mu.Lock()
	q.stats.ImmediateReads++
	if pitch != tight {
		q.stats.Repacks++
	}
	q.mu.Unlock()

	// Row by row, because the device's rows are padded and the caller's are
	// not. When the two pitches agree this is one contiguous copy, which the
	// loop degenerates to.
	for r := range src.desc.Size.Height {
		copy(into[r*tight:(r+1)*tight], mem[base+r*pitch:base+r*pitch+tight])
	}
	return nil
}
