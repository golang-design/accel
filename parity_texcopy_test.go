// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"fmt"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/parity"
)

// specs/062-backend-parity.md section 6.8: the format enumeration crossed with
// the copy entry points.
//
// # Why this is a second surface and not more claims on the first
//
// Section 6.2 already covers every colour format, through a render pass. That
// says nothing about `Recorder.CopyBufferToTexture` and `CopyTextureToBuffer`,
// which are a different path with a different failure: the render path compares
// the *encoding* of a texel, and the copy path compares the *addressing* of the
// rows around it. A backend can encode RGBA16Float correctly and still copy it
// at the wrong pitch.
//
// Adding "Format.RGBA16Float" to a copy case's covers would have changed no
// number the gate reports, because the render case already claims it -- so the
// gate could never have said "this format has never been copied". A separate
// surface over the same enumeration is what makes that sentence sayable.
const formatCopySurface = "FormatCopy"

// The two widths every case runs, and the reason there are two.
//
// specs/001-device-resources.md section 4.2 guarantees that at the accel API
// boundary texture data is tightly packed: row r begins at r*width*bpp with no
// padding, and accel pays for any repack the device's row-pitch alignment
// forces. copyAlignedW is a width whose rows are already aligned for the
// four-byte formats, and copyRepackW is one whose rows are not aligned for any
// format on either backend -- so the second width is the guarantee under load
// and the first is the path that does not exercise it.
//
// An odd height, so a vertical flip cannot coincide with the identity.
const (
	copyAlignedW = 64
	copyRepackW  = 13
	copyH        = 7
)

// formatCopyParityCases is one case per host-copyable format.
func formatCopyParityCases() []parityCase {
	copyable := []accel.Format{
		accel.RGBA8Unorm, accel.RGBA8UnormSRGB, accel.BGRA8Unorm,
		accel.R16Float, accel.RG16Float, accel.RGBA16Float,
		accel.R32Float, accel.RG32Float, accel.RGBA32Float,
		// The depth formats a copy reaches. FormatInfo reports them not
		// host-copyable, which is about Queue.ReadTexture mapping the memory --
		// a recorded copy is a different thing and both backends lower it.
		accel.Depth32Float, accel.Depth32FloatStencil8,
	}
	cases := make([]parityCase, 0, len(copyable))
	for _, f := range copyable {
		cases = append(cases, parityCase{
			name:   "a " + f.String() + " texture through a copy in each direction",
			covers: parity.Covers{formatCopySurface + "." + f.String()},
			run: func(t *testing.T, d *accel.Device) []byte {
				t.Helper()
				// Both widths in one result: the aligned one and the one that
				// forces a repack. Concatenated rather than split into two
				// cases because they are one guarantee, and a backend that
				// repacked wrongly would pass the aligned half.
				out := textureCopyRoundTrip(t, d, f, copyAlignedW, copyH)
				return append(out, textureCopyRoundTrip(t, d, f, copyRepackW, copyH)...)
			},
		})
	}
	return cases
}

// formatCopyParityExclusions are the two members no copy case can have.
func formatCopyParityExclusions() []parity.Excluded {
	return []parity.Excluded{
		{Name: formatCopySurface + ".FormatInvalid", Why: "the zero-value sentinel " +
			"for an optional format constraint, not a creatable format: there is no " +
			"texture to copy"},
		{Name: formatCopySurface + ".Depth24PlusStencil8", Why: "BytesPerPixel is 0 " +
			"because the layout is device-defined, so there is no tightly packed size " +
			"for a copy to name, and the CPU backend refuses the format outright " +
			"(internal/cpu/texel.go). A comparison needs an oracle"},
	}
}

// textureCopyRoundTrip uploads a row-identifiable pattern into a texture and
// copies it straight back out.
//
// It asserts caller order against the pattern before returning, on whichever
// device it ran on, and that assertion is the reason this is not left to the
// matrix's comparison alone. docs/conventions.md records that reading a render
// target back yields bottom-origin rows on GL and on Metal -- on Metal *despite*
// its top-left texture origin, so reasoning from the documented origin gives
// the wrong answer -- while accel guarantees caller order and the backend
// flips. Two backends that flipped identically would agree perfectly, and the
// pattern is what notices.
func textureCopyRoundTrip(t *testing.T, d *accel.Device, f accel.Format, w, h int) []byte {
	t.Helper()
	bpp := f.BytesPerPixel()
	if bpp == 0 {
		t.Fatalf("%v has a device-defined layout and no case should reach here", f)
	}
	stride := w * bpp

	// Every byte encodes the row it belongs to, so a misplaced byte names where
	// it came from rather than merely differing. The row is recoverable as
	// b/16, which holds while the height is at most 16.
	pattern := make([]byte, stride*h)
	for r := range h {
		for i := range stride {
			pattern[r*stride+i] = byte(r*16 + i%16)
		}
	}

	desc := accel.TextureDescriptor{
		Format: f, Size: accel.Extent{Width: w, Height: h},
		Usage: accel.TextureCopySrc | accel.TextureCopyDst,
		Label: fmt.Sprintf("%v %dx%d", f, w, h),
	}
	if f.IsDepth() {
		// A depth texture is device-private on macOS, and the CPU backend
		// enforces the same rule so it is not discovered in production.
		desc.Kind = accel.MemoryDevice
	}
	tex, err := d.NewTexture(desc)
	if err != nil {
		t.Fatalf("%v: %v texture %dx%d: %v", d.Info().Backend, f, w, h, err)
	}
	defer tex.Close()

	src := newBytes(t, d, "copy src", len(pattern))
	dst := newBytes(t, d, "copy dst", len(pattern))
	if err := d.Queue().WriteBuffer(src, 0, pattern); err != nil {
		t.Fatalf("%v: upload %v: %v", d.Info().Backend, f, err)
	}

	r := d.NewRecorder()
	r.CopyBufferToTexture(tex, whole(t, src))
	r.CopyTextureToBuffer(whole(t, dst), tex)
	g, err := r.Build()
	if err != nil {
		t.Fatalf("%v: build %v %dx%d: %v", d.Info().Backend, f, w, h, err)
	}
	defer g.Close()
	if err := d.Queue().Submit(g).Wait(); err != nil {
		t.Fatalf("%v: submit %v %dx%d: %v", d.Info().Backend, f, w, h, err)
	}

	got := make([]byte, len(pattern))
	if err := d.Queue().ReadBuffer(dst, 0, got); err != nil {
		t.Fatalf("%v: readback %v: %v", d.Info().Backend, f, err)
	}

	for i := range pattern {
		if got[i] == pattern[i] {
			continue
		}
		row, col := i/stride, i%stride
		t.Fatalf("%v: %v %dx%d row %d byte %d is %d, want %d; that byte belongs to "+
			"row %d, so this is %s\n  the row pitch is %d tightly packed and %d on "+
			"this device, which %s",
			d.Info().Backend, f, w, h, row, col, got[i], pattern[i], int(got[i])/16,
			flipOrShear(int(got[i])/16, row, h), stride, d.AlignedRowPitch(f, w),
			repackNote(d.AlignedRowPitchRepacks(f, w)))
	}
	return got
}

func repackNote(repacks bool) string {
	if repacks {
		return "means accel repacked this copy and owes the caller tight rows"
	}
	return "are the same, so no repack was involved"
}

// flipOrShear names which convention failure a misplaced row indicates, because
// "different" is the least useful thing a parity failure can say.
func flipOrShear(sawRow, wantRow, height int) string {
	if sawRow == height-1-wantRow {
		return "a vertical flip: the backend returned bottom-origin rows and accel " +
			"guarantees caller order"
	}
	return "neither a clean flip nor the identity, so it is a shear or a pitch " +
		"mismatch rather than an origin convention"
}
