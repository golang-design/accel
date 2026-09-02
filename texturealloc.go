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
// pool one or the other, and [Pool.AllocBuffer] refuses a texture pool for the
// same reason this refuses a buffer one.
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
	// Zero normalizes to one. Mip chains and array layers are admitted since
	// 2026-08-30: [Texture.Subresource] states the layout, textureBytes sizes
	// every level, and a recorded access covers the *subresource's* range
	// rather than the whole allocation. What still addresses the base level
	// only is the host copy, and that is refused per view rather than per
	// texture -- see readTexture and Recorder.textureCopy.
	if levels := desc.MipLevels; levels > 1 {
		if w, h := desc.Size.Width, desc.Size.Height; levels > mipChainLength(w, h) {
			return fmt.Errorf("%w: texture %q: %d mip levels, and a %dx%d texture has %d "+
				"before an axis reaches one", ErrUsage, desc.Label, levels, w, h,
				mipChainLength(w, h))
		}
	}
	if lim := d.info.Limits.MaxTextureArrayLayers; lim > 0 && desc.ArrayLayers > lim {
		return fmt.Errorf("%w: texture %q: %d array layers and %q reports a limit of %d",
			ErrUsage, desc.Label, desc.ArrayLayers, d.info.Name, lim)
	}
	if desc.Usage == 0 {
		return fmt.Errorf("%w: texture %q declares no usage, so nothing can be done "+
			"with it", ErrUsage, desc.Label)
	}

	// A *FormatError, so a caller can branch on the class and read which
	// capability the format lacks; errors.Is on ErrFormat still holds through
	// its Unwrap.
	info := d.FormatInfo(desc.Format)
	for _, c := range []struct {
		usage TextureUsage
		have  bool
		want  string
	}{
		{TextureRenderTarget, info.Renderable, "renderable"},
		{TextureSampled, info.Sampleable, "sampleable"},
		{TextureStorage, info.StorageRead || info.StorageWrite, "usable as a storage image"},
	} {
		if desc.Usage&c.usage != 0 && !c.have {
			return &FormatError{
				Format: desc.Format, Want: c.want, Device: d.info.Name, Resource: desc.Label,
			}
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
	levels, layers := max(desc.MipLevels, 1), max(desc.ArrayLayers, 1)
	total := 0
	for m := range levels {
		w := mipExtent(desc.Size.Width, m)
		h := mipExtent(desc.Size.Height, m)
		// Both planes, for a planar format; the second term is zero otherwise.
		total += (levelPitch(d, desc.Format, w) +
			d.StencilPlanePitch(desc.Format, w)) * h * layers
	}
	return total
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
			"mapped. Allocate it from readback memory: set Kind to MemoryReadback in the "+
			"TextureDescriptor you pass to Device.NewTexture, or allocate it out of a pool "+
			"created with that Kind", src.desc.Label)
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

// newTexture allocates a texture from an implicit texture pool.
//
// The convenience form, for a caller who wants one image rather than an arena.
// It is a real pool underneath — a texture pool, since the placement alignment
// makes that a different kind of pool — and the device owns it, which is why a
// texture from here is a live child of the device exactly as one from an
// explicit pool is a live child of that pool.
func (d *Device) newTexture(desc TextureDescriptor) (*Texture, error) {
	if err := d.state.checkOpen("NewTexture"); err != nil {
		return nil, err
	}
	if lost := d.dev.Lost(); lost != nil {
		return nil, lost
	}
	if err := validateTexture(d, desc); err != nil {
		return nil, err
	}

	// One pool per texture rather than a shared arena. A shared one would need
	// the growth policy the implicit buffer pool has, and a texture's placement
	// alignment makes the wasted tail much larger; a caller allocating many
	// images wants an explicit pool and should be nudged toward one rather than
	// quietly given a worse arena.
	size := textureBytes(d, desc)
	kind := desc.Kind
	if kind == 0 {
		kind = MemoryDevice
	}
	p, err := d.NewPool(PoolDescriptor{
		Kind: kind, Bytes: size + d.info.Limits.MinTexturePlacementAlignment,
		Textures: true, Label: "implicit texture pool for " + desc.Label,
	})
	if err != nil {
		return nil, err
	}
	t, err := p.allocTexture(desc)
	if err != nil {
		_ = p.Close()
		return nil, err
	}
	t.ownsPool = true
	return t, nil
}

// copyTextureToBuffer and copyBufferToTexture record a texture-buffer copy.
//
// The buffer side is tightly packed and the texture side is padded to the
// device's row alignment, so the plan carries both pitches and the backend
// steps by each. Where they agree it degenerates to one contiguous copy.
//
// This is where the guarantee of specs/001-device-resources.md section 4.2 is
// paid for in a graph, rather than in an intermediate as the immediate path
// does: a recorded copy knows both pitches at build, so there is nothing to
// allocate and nothing to copy twice.
func (r *Recorder) textureCopy(op string, buf BufferView, tex *Texture, toBuffer bool) NodeID {
	if tex == nil {
		r.fail("%s: no texture", op)
		return r.node(textureNodeKind(toBuffer), op, nil, nil)
	}
	if err := tex.state.checkOpen(op); err != nil {
		r.state.errs = append(r.state.errs, err)
		return r.node(textureNodeKind(toBuffer), op, nil, nil)
	}
	if tex.pool.dev != r.state.dev {
		r.fail("%s %q: the texture belongs to a different device", op, tex.desc.Label)
		return r.node(textureNodeKind(toBuffer), op, nil, nil)
	}

	need := TextureCopyDst
	if toBuffer {
		need = TextureCopySrc
	}
	if tex.desc.Usage&need == 0 {
		r.fail("%s %q: it needs %v and was created with %v",
			op, tex.desc.Label, need, tex.desc.Usage)
		return r.node(textureNodeKind(toBuffer), op, nil, nil)
	}

	tight := tightRowPitch(tex.desc.Format, tex.desc.Size.Width)
	if tight == 0 {
		r.fail("%s %q: %v has a device-defined layout, so a host-side copy has no "+
			"pitch to pack to", op, tex.desc.Label, tex.desc.Format)
		return r.node(textureNodeKind(toBuffer), op, nil, nil)
	}
	want := tight * tex.desc.Size.Height
	if _, size := buf.byteRange(); size != want {
		r.fail("%s %q: the buffer side is %d bytes and a tightly packed %dx%d %v is %d",
			op, tex.desc.Label, size, tex.desc.Size.Width, tex.desc.Size.Height,
			tex.desc.Format, want)
		return r.node(textureNodeKind(toBuffer), op, nil, nil)
	}

	mode := AccessWrite
	if toBuffer {
		mode = AccessRead
	}
	bufAccess, ok := r.declare(op, buf, oppositeAccess(mode))
	if !ok {
		return r.node(textureNodeKind(toBuffer), op, nil, nil)
	}

	// A copy moves the **base level**, and its access says exactly that.
	//
	// It takes a texture rather than a view, so there is no subresource for a
	// caller to name; the size check above is against the base extent for the
	// same reason. Declaring the whole allocation instead would make a copy of
	// the base level hazard against a pass writing mip 3, which is a barrier
	// the plan does not need and a serialization a caller cannot explain.
	base := tex.Subresource(0, 0)
	texAccess := access{
		res: resourceRef{tex: tex}, off: base.Offset, size: base.Size, mode: mode,
	}

	kind := textureNodeKind(toBuffer)
	var accesses []access
	if toBuffer {
		accesses = []access{bufAccess, texAccess} // destination first
	} else {
		accesses = []access{texAccess, bufAccess}
	}
	id := r.node(kind, op, accesses, nil)
	n := &r.state.nodes[id]
	n.texture = tex
	return id
}

func textureNodeKind(toBuffer bool) NodeKind {
	if toBuffer {
		return NodeCopyTextureToBuffer
	}
	return NodeCopyBufferToTexture
}

// oppositeAccess is the buffer side's mode given the texture side's.
func oppositeAccess(tex Access) Access {
	if tex == AccessRead {
		return AccessWrite
	}
	return AccessRead
}

// mipChainLength is how many levels a texture of this extent has before both
// axes reach one.
//
// Stated as a rule rather than left to a caller's arithmetic, because the
// off-by-one at the end of a chain is where a mip count is usually wrong: a
// 1024x1024 texture has eleven levels, not ten, since level ten is the 1x1.
func mipChainLength(w, h int) int {
	n := 1
	for w > 1 || h > 1 {
		w, h = max(1, w/2), max(1, h/2)
		n++
	}
	return n
}
