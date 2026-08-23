// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"strings"
	"testing"

	"golang.design/x/accel"
)

// Spec 001 section 4.1's format table, row by row.
//
// Bytes per pixel is asserted because assuming it is a real bug, and one that
// surfaces as an out-of-range panic during readback rather than as a wrong
// image — loud, but only on the machine whose format happened to differ.
func TestTheFormatTable(t *testing.T) {
	cases := []struct {
		f        accel.Format
		bpp      int
		channels int
		depth    bool
		srgb     bool
		storage  bool
	}{
		{accel.RGBA8Unorm, 4, 4, false, false, true},
		{accel.RGBA8UnormSRGB, 4, 4, false, true, false},
		{accel.BGRA8Unorm, 4, 4, false, false, false},
		{accel.R16Float, 2, 1, false, false, false},
		{accel.RG16Float, 4, 2, false, false, false},
		{accel.RGBA16Float, 8, 4, false, false, true},
		{accel.R32Float, 4, 1, false, false, true},
		{accel.RG32Float, 8, 2, false, false, false},
		{accel.RGBA32Float, 16, 4, false, false, true},
		{accel.Depth32Float, 4, 1, true, false, false},
		// Its layout is device-defined: "24 plus" means at least 24 bits of
		// depth, and a backend may store it as 32 with 8 unused or pack it with
		// the stencil. Reporting zero says so rather than guessing four.
		{accel.Depth24PlusStencil8, 0, 2, true, false, false},
	}

	d := openDevice(t)
	for _, c := range cases {
		t.Run(c.f.String(), func(t *testing.T) {
			if got := c.f.BytesPerPixel(); got != c.bpp {
				t.Errorf("BytesPerPixel is %d, want %d", got, c.bpp)
			}
			if got := c.f.IsDepth(); got != c.depth {
				t.Errorf("IsDepth is %v, want %v", got, c.depth)
			}
			info := d.FormatInfo(c.f)
			if info.Format != c.f {
				t.Errorf("FormatInfo reports format %v", info.Format)
			}
			if info.Channels != c.channels {
				t.Errorf("Channels is %d, want %d", info.Channels, c.channels)
			}
			if info.IsSRGB != c.srgb {
				t.Errorf("IsSRGB is %v, want %v", info.IsSRGB, c.srgb)
			}
			if got := info.StorageRead || info.StorageWrite; got != c.storage {
				t.Errorf("storage is %v, want %v", got, c.storage)
			}
			// A depth format is device-private on macOS, which several backends
			// require and this one enforces so the rule is not discovered in
			// production.
			if info.HostCopyable == c.depth {
				t.Errorf("HostCopyable is %v for a depth=%v format", info.HostCopyable, c.depth)
			}
		})
	}

	// An unsupported format reports a zero FormatInfo and no error: absence is
	// a capability answer, which is decision 6 applied to formats.
	if got := d.FormatInfo(accel.Format(999)); got != (accel.FormatInfo{}) {
		t.Errorf("an unknown format reports %+v, want the zero value", got)
	}
	if got := accel.FormatInvalid.String(); got != "FormatInvalid" {
		t.Errorf("FormatInvalid names itself %q", got)
	}
	if got := accel.Format(999).String(); !strings.Contains(got, "999") {
		t.Errorf("an unknown format names itself %q", got)
	}
}

// An sRGB format is never a storage image, on any backend.
//
// Not "unsupported": architecturally absent. Its transfer function is applied
// by fixed-function hardware that a storage write bypasses, so a storage image
// of one would write values the sampler then decodes a second time.
func TestSRGBIsNeverAStorageFormat(t *testing.T) {
	d := openDevice(t)
	info := d.FormatInfo(accel.RGBA8UnormSRGB)
	if info.StorageRead || info.StorageWrite {
		t.Error("an sRGB format reports storage, and its transfer function is applied " +
			"by hardware a storage write bypasses")
	}
	if !info.Renderable || !info.Sampleable {
		t.Error("an sRGB format is renderable and sampleable")
	}
}

// The row-pitch guarantee: at the API boundary, texture data is tightly packed.
//
//	tight  = w · bpp
//	device = ⌈w · bpp / A⌉ · A
//
// A caller sizes a readback as width*height*bpp and is always right, whatever
// the device stores.
func TestRowPitchIsTightAtTheBoundaryAndPaddedOnTheDevice(t *testing.T) {
	d := openDevice(t)
	align := d.Limits().MinBufferCopyRowPitchAlignment

	cases := []struct {
		width   int
		f       accel.Format
		repacks bool
	}{
		// The spec's own examples: 1024-wide RGBA8Unorm has a 4096-byte row,
		// already a multiple of 256, and repacks nowhere. 100-wide has 400,
		// which is not, and repacks.
		{1024, accel.RGBA8Unorm, false},
		{100, accel.RGBA8Unorm, true},
		{64, accel.RGBA8Unorm, false},
		{1, accel.RGBA8Unorm, true},
	}
	for _, c := range cases {
		tight := c.width * c.f.BytesPerPixel()
		got := d.AlignedRowPitch(c.f, c.width)
		want := ((tight + align - 1) / align) * align
		if got != want {
			t.Errorf("width %d: aligned pitch is %d, want %d", c.width, got, want)
		}
		if repacks := got != tight; repacks != c.repacks {
			t.Errorf("width %d: tight is %d and aligned is %d, so repacks=%v, want %v",
				c.width, tight, got, repacks, c.repacks)
		}
	}

	// A format with no defined layout has no pitch to report, rather than a
	// guessed one.
	if got := d.AlignedRowPitch(accel.Depth24PlusStencil8, 64); got != 0 {
		t.Errorf("a device-defined layout reports pitch %d, want 0", got)
	}
	if got := d.AlignedRowPitch(accel.RGBA8Unorm, 0); got != 0 {
		t.Errorf("a zero width reports pitch %d", got)
	}
}

// A texture is allocated from a texture pool, read back tightly packed, and the
// repack is counted rather than returned.
func TestATextureRoundTripsTightlyPacked(t *testing.T) {
	const w, h = 100, 7 // 400-byte rows, not a multiple of 256, so it repacks

	d := openDevice(t)
	p, err := d.NewPoolWith(accel.PoolDescriptor{
		Kind: accel.MemoryShared, Bytes: 1 << 20, Textures: true, Label: "images",
	})
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	defer p.Close()

	tex, err := p.AllocTexture(accel.TextureDescriptor{
		Format: accel.RGBA8Unorm, Size: accel.Extent{Width: w, Height: h},
		Usage: accel.TextureCopySrc | accel.TextureCopyDst, Label: "image",
	})
	if err != nil {
		t.Fatalf("texture: %v", err)
	}
	defer tex.Close()

	if got := tex.Format(); got != accel.RGBA8Unorm {
		t.Errorf("Format is %v", got)
	}
	if got := tex.Size(); got.Width != w || got.Height != h {
		t.Errorf("Size is %+v", got)
	}
	// The footprint counts the padding, so it is at least the tight size and
	// larger where the device pads.
	tight := w * h * accel.RGBA8Unorm.BytesPerPixel()
	if tex.Bytes() < tight {
		t.Errorf("the footprint is %d and a tight image is %d", tex.Bytes(), tight)
	}
	if !d.AlignedRowPitchRepacks(accel.RGBA8Unorm, w) {
		t.Skip("this device does not pad a 400-byte row")
	}

	before := d.Queue().Stats().Repacks
	out := make([]byte, tight)
	if err := d.Queue().ReadTexture(tex, out); err != nil {
		t.Fatalf("readback: %v", err)
	}
	if got := d.Queue().Stats().Repacks - before; got != 1 {
		t.Errorf("the repack count moved by %d, want 1: a cost proportional to the "+
			"image and otherwise silent is the wrong kind of silent", got)
	}

	// A destination of the wrong size is refused rather than reading past.
	if err := d.Queue().ReadTexture(tex, make([]byte, tight-1)); err == nil {
		t.Error("a destination smaller than the tight image should be refused")
	}
}

// Every rule a texture descriptor has to satisfy.
func TestTextureValidationRows(t *testing.T) {
	d := openDevice(t)
	pool := func(t *testing.T, textures bool) *accel.Pool {
		t.Helper()
		p, err := d.NewPoolWith(accel.PoolDescriptor{
			Kind: accel.MemoryShared, Bytes: 1 << 20, Textures: textures, Label: "p",
		})
		if err != nil {
			t.Fatalf("pool: %v", err)
		}
		t.Cleanup(func() { _ = p.Close() })
		return p
	}

	base := accel.TextureDescriptor{
		Format: accel.RGBA8Unorm, Size: accel.Extent{Width: 8, Height: 8},
		Usage: accel.TextureCopySrc, Label: "tex",
	}
	cases := []struct {
		name    string
		says    string
		buffers bool
		mutate  func(*accel.TextureDescriptor)
	}{
		{"a buffer pool", "holds buffers", true, nil},
		{"an invalid format", "not a creatable format", false,
			func(d *accel.TextureDescriptor) { d.Format = accel.FormatInvalid }},
		{"a zero extent", "non-positive axis", false,
			func(d *accel.TextureDescriptor) { d.Size.Width = 0 }},
		{"a 3D extent", "every v0 operation addresses a single layer", false,
			func(d *accel.TextureDescriptor) { d.Size.Depth = 4 }},
		{"mip levels", "mip chains are deferred", false,
			func(d *accel.TextureDescriptor) { d.MipLevels = 4 }},
		{"array layers", "array layers", false,
			func(d *accel.TextureDescriptor) { d.ArrayLayers = 6 }},
		{"no usage", "declares no usage", false,
			func(d *accel.TextureDescriptor) { d.Usage = 0 }},
		{"storage on an sRGB format", "cannot be used as a storage image", false,
			func(d *accel.TextureDescriptor) {
				d.Format = accel.RGBA8UnormSRGB
				d.Usage = accel.TextureStorage
			}},
		{"past the extent limit", "exceeds", false,
			func(d *accel.TextureDescriptor) { d.Size.Width = 1 << 20 }},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			desc := base
			if c.mutate != nil {
				c.mutate(&desc)
			}
			p := pool(t, !c.buffers)
			_, err := p.AllocTexture(desc)
			if err == nil {
				t.Fatal("expected a rejection")
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Errorf("the message should say %q, got:\n%v", c.says, err)
			}
		})
	}

	// And a descriptor satisfying every rule is accepted, or the rows above
	// would be passing against a pool that refuses everything.
	tex, err := pool(t, true).AllocTexture(base)
	if err != nil {
		t.Fatalf("a valid descriptor was refused: %v", err)
	}
	if err := tex.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := tex.Close(); err != nil {
		t.Errorf("Close should be idempotent: %v", err)
	}
}

// A depth texture is device-private, which several backends require and this
// one enforces so the rule is not discovered in production.
func TestADepthTextureIsNotHostReadable(t *testing.T) {
	d := openDevice(t)
	p, err := d.NewPoolWith(accel.PoolDescriptor{
		Kind: accel.MemoryShared, Bytes: 1 << 20, Textures: true, Label: "depth",
	})
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	defer p.Close()

	tex, err := p.AllocTexture(accel.TextureDescriptor{
		Format: accel.Depth32Float, Size: accel.Extent{Width: 8, Height: 8},
		Usage: accel.TextureRenderTarget | accel.TextureCopySrc, Label: "depth",
	})
	if err != nil {
		t.Fatalf("a depth texture should be creatable: %v", err)
	}
	defer tex.Close()

	err = d.Queue().ReadTexture(tex, make([]byte, 8*8*4))
	if err == nil {
		t.Fatal("reading a depth texture back to the host should be refused")
	}
	if !strings.Contains(err.Error(), "device-private") {
		t.Errorf("the message should say why, got:\n%v", err)
	}
}

// A pool with a live texture refuses to close, exactly as one with a live
// buffer does: closing children out from under a caller who still holds them
// turns a leak into a use-after-free.
func TestAPoolWillNotCloseUnderALiveTexture(t *testing.T) {
	d := openDevice(t)
	p, err := d.NewPoolWith(accel.PoolDescriptor{
		Kind: accel.MemoryShared, Bytes: 1 << 20, Textures: true, Label: "p",
	})
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	tex, err := p.AllocTexture(accel.TextureDescriptor{
		Format: accel.RGBA8Unorm, Size: accel.Extent{Width: 4, Height: 4},
		Usage: accel.TextureCopySrc, Label: "tex",
	})
	if err != nil {
		t.Fatalf("texture: %v", err)
	}
	if err := p.Close(); err == nil {
		t.Error("closing a pool with a live texture should fail")
	}
	if err := tex.Close(); err != nil {
		t.Fatalf("close texture: %v", err)
	}
	if err := p.Close(); err != nil {
		t.Errorf("close pool: %v", err)
	}
}

func TestTextureUsageNamesItself(t *testing.T) {
	got := (accel.TextureSampled | accel.TextureCopyDst).String()
	for _, want := range []string{"TextureSampled", "TextureCopyDst"} {
		if !strings.Contains(got, want) {
			t.Errorf("%q should contain %q", got, want)
		}
	}
	if accel.TextureUsage(0).String() != "no usage" {
		t.Errorf("the empty set is %q", accel.TextureUsage(0).String())
	}
}

// A texture round-trips through a graph: buffer in, texture, buffer out.
//
// The width is chosen so the device pads: a 100-pixel RGBA8Unorm row is 400
// bytes, not a multiple of 256, so the texture side steps by 512 while the
// buffer side steps by 400. A copy that used one pitch for both would shear the
// image — every row after the first offset by the difference — which is a
// plausible-looking picture rather than an error.
func TestATextureRoundTripsThroughAGraph(t *testing.T) {
	const w, h = 100, 5
	const bpp = 4

	d := openDevice(t)
	if !d.AlignedRowPitchRepacks(accel.RGBA8Unorm, w) {
		t.Skip("this device does not pad a 400-byte row, so the two pitches agree")
	}

	p, err := d.NewPoolWith(accel.PoolDescriptor{
		Kind: accel.MemoryShared, Bytes: 1 << 20, Textures: true, Label: "images",
	})
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	defer p.Close()

	tex, err := p.AllocTexture(accel.TextureDescriptor{
		Format: accel.RGBA8Unorm, Size: accel.Extent{Width: w, Height: h},
		Usage: accel.TextureCopySrc | accel.TextureCopyDst, Label: "image",
	})
	if err != nil {
		t.Fatalf("texture: %v", err)
	}
	defer tex.Close()

	tight := w * h * bpp
	src := newBytes(t, d, "src", tight)
	dst := newBytes(t, d, "dst", tight)

	// A pattern whose row is identifiable, so a sheared copy is visible rather
	// than merely different.
	pattern := make([]byte, tight)
	for r := range h {
		for i := range w * bpp {
			pattern[r*w*bpp+i] = byte(r*16 + i%16)
		}
	}
	if err := d.Queue().WriteBuffer(src, 0, pattern); err != nil {
		t.Fatalf("upload: %v", err)
	}

	r := d.NewRecorder()
	r.CopyBufferToTexture(tex, whole(t, src))
	r.CopyTextureToBuffer(whole(t, dst), tex)
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	defer g.Close()

	// The two nodes touch one texture, one writing, so they are ordered and a
	// barrier separates them. Without that the readback could precede the
	// upload.
	if g.Hazards() != 1 {
		t.Errorf("got %d hazards, want the read-after-write on the texture", g.Hazards())
	}
	if n := g.NodeStats(1); n.BarriersBefore != 1 {
		t.Errorf("the readback needs a barrier after the upload, got %d", n.BarriersBefore)
	}

	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("submit: %v", err)
	}
	got := make([]byte, tight)
	if err := d.Queue().ReadBuffer(dst, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}
	for i := range pattern {
		if got[i] != pattern[i] {
			row, col := i/(w*bpp), i%(w*bpp)
			t.Fatalf("row %d byte %d is %d, want %d: the two sides have different row "+
				"pitches and a copy using one for both shears the image",
				row, col, got[i], pattern[i])
		}
	}
}

func newBytes(t *testing.T, d *accel.Device, label string, n int) *accel.Buffer {
	t.Helper()
	b, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.U8, Count: n, Label: label,
		Usage: accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst,
	})
	if err != nil {
		t.Fatalf("buffer %q: %v", label, err)
	}
	t.Cleanup(func() { _ = b.Close() })
	return b
}

// The validation rows a texture copy adds.
func TestTextureCopyValidationRows(t *testing.T) {
	d := openDevice(t)
	p, err := d.NewPoolWith(accel.PoolDescriptor{
		Kind: accel.MemoryShared, Bytes: 1 << 20, Textures: true, Label: "images",
	})
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	defer p.Close()

	mk := func(t *testing.T, usage accel.TextureUsage, f accel.Format) *accel.Texture {
		t.Helper()
		tex, err := p.AllocTexture(accel.TextureDescriptor{
			Format: f, Size: accel.Extent{Width: 8, Height: 4},
			Usage: usage, Label: "tex",
		})
		if err != nil {
			t.Fatalf("texture: %v", err)
		}
		t.Cleanup(func() { _ = tex.Close() })
		return tex
	}

	cases := []struct {
		name string
		says string
		rec  func(t *testing.T, r *accel.Recorder)
	}{{
		name: "a texture without copy-source usage",
		says: "it needs TextureCopySrc",
		rec: func(t *testing.T, r *accel.Recorder) {
			tex := mk(t, accel.TextureCopyDst, accel.RGBA8Unorm)
			r.CopyTextureToBuffer(whole(t, newBytes(t, d, "b", 8*4*4)), tex)
		},
	}, {
		name: "a texture without copy-destination usage",
		says: "it needs TextureCopyDst",
		rec: func(t *testing.T, r *accel.Recorder) {
			tex := mk(t, accel.TextureCopySrc, accel.RGBA8Unorm)
			r.CopyBufferToTexture(tex, whole(t, newBytes(t, d, "b", 8*4*4)))
		},
	}, {
		name: "a buffer of the wrong size",
		says: "tightly packed 8x4",
		rec: func(t *testing.T, r *accel.Recorder) {
			tex := mk(t, accel.TextureCopySrc, accel.RGBA8Unorm)
			r.CopyTextureToBuffer(whole(t, newBytes(t, d, "b", 8*4*4-1)), tex)
		},
	}, {
		name: "a format with a device-defined layout",
		says: "device-defined layout",
		rec: func(t *testing.T, r *accel.Recorder) {
			tex := mk(t, accel.TextureCopySrc, accel.Depth24PlusStencil8)
			r.CopyTextureToBuffer(whole(t, newBytes(t, d, "b", 8*4*4)), tex)
		},
	}, {
		name: "no texture at all",
		says: "no texture",
		rec: func(t *testing.T, r *accel.Recorder) {
			r.CopyTextureToBuffer(whole(t, newBytes(t, d, "b", 16)), nil)
		},
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := d.NewRecorder()
			c.rec(t, r)
			_, err := r.Build()
			if err == nil {
				t.Fatal("expected a rejection")
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Errorf("the message should say %q, got:\n%v", c.says, err)
			}
		})
	}
}
