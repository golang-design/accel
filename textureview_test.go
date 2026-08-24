// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"strings"
	"testing"

	"golang.design/x/accel"
)

func newTex(t *testing.T, d *accel.Device, f accel.Format, label string) *accel.Texture {
	t.Helper()
	tex, err := d.NewTexture(accel.TextureDescriptor{
		Format: f, Size: accel.Extent{Width: 8, Height: 4},
		Usage: accel.TextureCopySrc | accel.TextureCopyDst, Label: label,
	})
	if err != nil {
		t.Fatalf("texture %q: %v", label, err)
	}
	t.Cleanup(func() { _ = tex.Close() })
	return tex
}

// A view resolves the texture's own format, so no consumer repeats the
// resolution and none of them forgets.
func TestAViewResolvesTheTexturesFormat(t *testing.T) {
	d := openDevice(t)
	tex := newTex(t, d, accel.RGBA8Unorm, "base")

	v, err := tex.Whole()
	if err != nil {
		t.Fatalf("Whole: %v", err)
	}
	if v.Format != accel.RGBA8Unorm {
		t.Errorf("the view reads %v, want the texture's %v", v.Format, accel.RGBA8Unorm)
	}
	if v.Texture != tex || v.Mip != 0 || v.Layer != 0 {
		t.Errorf("Whole is %+v, want the base level of the first layer", v)
	}
}

// The pair the Format field exists for: linear and sRGB over the same bytes.
//
// specs/045-texture-attachments.md section 2.1 -- write linear through one
// view, present through an sRGB view of the same texture. The accepting half is
// asserted here because a rule that only ever refuses is one whose other half
// nobody has tested.
func TestAViewReinterpretsWithinAFamily(t *testing.T) {
	d := openDevice(t)
	tex := newTex(t, d, accel.RGBA8Unorm, "linear")

	v, err := tex.View(accel.TextureViewDesc{Format: accel.RGBA8UnormSRGB})
	if err != nil {
		t.Fatalf("an sRGB view of a linear texture of the same width and channel "+
			"count should be legal: %v", err)
	}
	if v.Format != accel.RGBA8UnormSRGB {
		t.Errorf("the view reads %v, want the reinterpreted %v", v.Format, accel.RGBA8UnormSRGB)
	}
	if v.Texture != tex {
		t.Error("the view names a different texture")
	}
	// And back the other way, since a view of an sRGB texture as linear is the
	// half a renderer uses to write.
	srgb := newTex(t, d, accel.RGBA8UnormSRGB, "srgb")
	if _, err := srgb.View(accel.TextureViewDesc{Format: accel.RGBA8Unorm}); err != nil {
		t.Errorf("a linear view of an sRGB texture should be legal: %v", err)
	}
}

// Every refusal names both formats and which clause failed, because
// "incompatible" alone sends a caller to guess.
func TestAViewRefusesAnIncompatibleFormat(t *testing.T) {
	d := openDevice(t)
	for _, c := range []struct {
		name   string
		of, as accel.Format
		says   string
	}{
		// Both are four bytes per pixel, so the width clause passes and the
		// channel clause is the one that catches it. Asserted because the
		// clauses are ordered and a refusal naming the wrong one sends a
		// caller to change the wrong thing.
		{"a different channel count", accel.RGBA8Unorm, accel.R32Float, "4 and 1 channels"},
		// A device-defined layout has no width to compare, so it can never be
		// reinterpreted into or out of.
		{"a device-defined layout", accel.RGBA8Unorm, accel.Depth24PlusStencil8, "bytes per pixel"},
		{"colour as depth", accel.RGBA8Unorm, accel.Depth32Float, ""},
		{"depth as colour", accel.Depth32Float, accel.RGBA8Unorm, ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			tex := newTex(t, d, c.of, "t")
			_, err := tex.View(accel.TextureViewDesc{Format: c.as})
			if err == nil {
				t.Fatalf("reading %v as %v was accepted", c.of, c.as)
			}
			// Both formats, always: a caller with several views needs to know
			// which one this is.
			for _, want := range []string{c.of.String(), c.as.String()} {
				if !strings.Contains(err.Error(), want) {
					t.Errorf("the refusal should name %q, got %v", want, err)
				}
			}
			if c.says != "" && !strings.Contains(err.Error(), c.says) {
				t.Errorf("the refusal should say which clause failed (%q), got %v", c.says, err)
			}
		})
	}
}

// A subresource outside the texture is refused, and the refusal says how many
// there are rather than only that the index is wrong.
func TestAViewRefusesASubresourceThatDoesNotExist(t *testing.T) {
	d := openDevice(t)
	tex := newTex(t, d, accel.RGBA8Unorm, "one")

	for _, c := range []struct {
		name string
		desc accel.TextureViewDesc
		says string
	}{
		{"a mip that does not exist", accel.TextureViewDesc{Mip: 1}, "which has 1 level(s)"},
		{"a negative mip", accel.TextureViewDesc{Mip: -1}, "level(s)"},
		{"a layer that does not exist", accel.TextureViewDesc{Layer: 1}, "which has 1 layer(s)"},
		{"a negative layer", accel.TextureViewDesc{Layer: -1}, "layer(s)"},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := tex.View(c.desc); err == nil {
				t.Fatalf("%+v was accepted", c.desc)
			} else if !strings.Contains(err.Error(), c.says) {
				t.Errorf("the refusal should say %q, got %v", c.says, err)
			}
		})
	}
}

// A view of a closed texture is refused rather than handing back a value that
// names freed memory.
func TestAViewOfAClosedTextureIsRefused(t *testing.T) {
	d := openDevice(t)
	tex, err := d.NewTexture(accel.TextureDescriptor{
		Format: accel.RGBA8Unorm, Size: accel.Extent{Width: 4, Height: 4},
		Usage: accel.TextureCopySrc | accel.TextureCopyDst, Label: "gone",
	})
	if err != nil {
		t.Fatalf("texture: %v", err)
	}
	if err := tex.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if _, err := tex.Whole(); err == nil {
		t.Fatal("a view of a closed texture was accepted")
	}
}
