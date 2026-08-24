// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import "fmt"

// A texture view names one subresource, per specs/045-texture-attachments.md.

// TextureViewDesc selects a subresource of a texture, and optionally reads it
// as a different format.
type TextureViewDesc struct {
	// Mip is the mip level, zero being the base.
	Mip int

	// Layer is the array layer, zero being the first.
	Layer int

	// Format reinterprets the same bytes. Zero means the texture's own, which
	// is what almost every view wants; see [Texture.View] for what a non-zero
	// value may be.
	Format Format
}

// TextureView is one subresource of a texture, read as one format.
//
// # Why a view rather than fields on an attachment
//
// specs/033-render-api.md section 3.3 forbids an overlapping subresource being
// both an attachment and shader-visible, and permits disjoint ones -- a
// different mip or a different array layer is a different subresource. That
// comparison is sound only if both sides name a subresource the same way.
//
// Spelling the mip and the layer on the attachment, and again on a stage's
// binding, would make the rule compare two shapes. This project has withdrawn
// two validation rules already -- check V23, and 033 section 6's
// undeclared-slot rule -- and both were written against a shape nobody had
// tested a caller against. One type is how the third is avoided.
//
// It is a value rather than a handle for [BufferView]'s reason: a view owns
// nothing, so there is nothing to close and no lifetime to get wrong. The
// texture it names is the thing with a lifetime.
type TextureView struct {
	Texture *Texture
	Mip     int
	Layer   int

	// Format is always concrete here, resolved from the texture when the
	// descriptor left it zero. A view that carried the zero value would make
	// every consumer repeat the resolution, and one of them would forget.
	Format Format
}

// View names a subresource of t.
//
// # Format reinterpretation, and the one pair it exists for
//
// A non-zero Format reads the same bytes as another format, and is legal only
// within a compatible family: the same bytes per pixel, the same channel count,
// and a difference only in the numeric *encoding*. The format table has exactly
// one such pair today, RGBA8Unorm and RGBA8UnormSRGB, and that pair is why this
// exists:
//
//	write linear through one view, present through an sRGB view of the same
//	texture
//
// which is how every target expresses it -- Vulkan's image-view format, D3D12's
// view casting, Metal's newTextureViewWithPixelFormat:, WebGPU's viewFormats --
// and what makes sRGB a property of the *view* rather than of the texture.
// specs/035-cpu-rasterizer.md section 5 says sRGB converts on write and on
// read; this is what owns that conversion.
//
// Anything outside a family is refused by name, saying both formats and which
// clause failed, because "incompatible" alone sends a caller to guess.
func (t *Texture) View(d TextureViewDesc) (TextureView, error) {
	if t == nil {
		return TextureView{}, fmt.Errorf("accel: Texture.View on a nil texture")
	}
	if err := t.state.checkOpen("Texture.View"); err != nil {
		return TextureView{}, err
	}
	levels, layers := max(t.desc.MipLevels, 1), max(t.desc.ArrayLayers, 1)
	if d.Mip < 0 || d.Mip >= levels {
		return TextureView{}, fmt.Errorf("%w: mip %d of texture %q, which has %d level(s)",
			ErrUsage, d.Mip, t.desc.Label, levels)
	}
	if d.Layer < 0 || d.Layer >= layers {
		return TextureView{}, fmt.Errorf("%w: layer %d of texture %q, which has %d layer(s)",
			ErrUsage, d.Layer, t.desc.Label, layers)
	}

	format := d.Format
	if format == FormatInvalid {
		format = t.desc.Format
	} else if why := incompatible(t.pool.dev, t.desc.Format, format); why != "" {
		return TextureView{}, fmt.Errorf("%w: a view of texture %q reads %v as %v, and %s. "+
			"A view reinterprets bytes and does not convert them, so the two formats must "+
			"describe the same bytes (specs/045-texture-attachments.md section 2.1)",
			ErrUsage, t.desc.Label, t.desc.Format, format, why)
	}
	return TextureView{Texture: t, Mip: d.Mip, Layer: d.Layer, Format: format}, nil
}

// Whole is the view of a texture's base level and first layer, in its own
// format.
//
// It exists because that is what almost every attachment is, and a caller
// writing three zero fields to say "all of it" is a caller the API is making
// work for.
func (t *Texture) Whole() (TextureView, error) { return t.View(TextureViewDesc{}) }

// incompatible reports why two formats do not describe the same bytes, or "".
//
// The three clauses are separate so the refusal can say which failed. A caller
// who wrote the wrong channel count and a caller who wrote the wrong width have
// made different mistakes, and one message for both tells neither of them
// anything.
func incompatible(d *Device, have, want Format) string {
	a, b := d.FormatInfo(have), d.FormatInfo(want)
	switch {
	case a.BytesPerPixel != b.BytesPerPixel || a.BytesPerPixel == 0:
		// Zero means device-defined, as for a combined depth-stencil layout,
		// and two device-defined layouts are not known to agree.
		return fmt.Sprintf("they are %d and %d bytes per pixel", a.BytesPerPixel, b.BytesPerPixel)
	case a.Channels != b.Channels:
		return fmt.Sprintf("they have %d and %d channels", a.Channels, b.Channels)
	case a.IsDepth != b.IsDepth || a.IsStencil != b.IsStencil:
		return "one is a depth or stencil format and the other is not"
	case a.IsSRGB == b.IsSRGB:
		// Same width, same channels, same aspect and the same encoding: these
		// are either the same format or two spellings this table does not
		// distinguish, and reinterpreting between them buys nothing.
		if have == want {
			return ""
		}
		return "they differ in something other than the numeric encoding, which is the " +
			"only difference a view may reinterpret"
	}
	return ""
}
