// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"testing"

	"golang.design/x/accel"
)

// A texture round trip returns rows in caller order, on Metal as on the CPU.
//
// # Why this entry exists before the feature that needs it
//
// docs/conventions.md records a divergence and the lesson that goes with it:
// reading a render target back yields bottom-origin rows on GL and on Metal --
// on Metal *despite* its top-left texture origin, so reasoning from the
// documented origin gives the wrong answer -- while a storage buffer written by
// a compute kernel reads back linearly. The guarantee accel makes is that
// readback is in caller order and the backend flips.
//
// The lesson recorded beside it is that "a parity test proves only the path it
// exercises… a compute-path test passes while the texture path is mirrored".
// Every texture test in this package ran on the CPU device, so the texture path
// had no GPU comparison at all and the one convention that is known to diverge
// was covered by nothing.
//
// specs/042-surface-completion.md section 5.4 is why it is written now rather
// than later: the render path has exactly one resource kind today, because an
// attachment is a buffer. The moment attachments become textures accel acquires
// a second kind, and this is the bug that survives its own tests -- the pass
// that reads storage output never sees the flip.
//
// # It is skipped today, and that is the design
//
// Metal does not lower a texture copy yet -- it refuses by name, pointing at
// specs/021-metal-bringup.md -- so there is nothing to compare. The entry
// skips with that reason rather than being absent, so the comparison happens on
// the first day it can rather than being remembered afterwards. A convention
// bug is cheapest to catch in the commit that makes it reachable.
//
// # What it asserts
//
// Byte-for-byte equality against the CPU oracle over a pattern whose row index
// is recoverable from any byte of it, so a flipped or sheared result is
// identifiable rather than merely unequal. An odd height, so a flip cannot
// coincide with the identity.
func TestATextureRoundTripKeepsCallerOrderOnMetal(t *testing.T) {
	const w, h, bpp = 64, 7, 4

	// A row-identifiable pattern: every byte encodes its row, so a mirrored
	// readback names the row it came from.
	pattern := make([]byte, w*h*bpp)
	for r := range h {
		for i := range w * bpp {
			pattern[r*w*bpp+i] = byte(r*16 + i%16)
		}
	}

	// The skip that waited for Metal to lower a texture copy is gone: it does,
	// this entry compares, and the refusal it watched for exists nowhere in the
	// tree. A skip that can never fire is a comparison nobody notices is not
	// happening.
	roundTrip := func(t *testing.T, d *accel.Device) []byte {
		t.Helper()
		tex, err := d.NewTexture(accel.TextureDescriptor{
			Format: accel.RGBA8Unorm, Size: accel.Extent{Width: w, Height: h},
			Usage: accel.TextureCopySrc | accel.TextureCopyDst, Label: "image",
		})
		if err != nil {
			t.Fatalf("texture: %v", err)
		}
		defer tex.Close()

		src := newBytes(t, d, "src", len(pattern))
		dst := newBytes(t, d, "dst", len(pattern))
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
		if err := d.Queue().Submit(g).Wait(); err != nil {
			t.Fatalf("submit: %v", err)
		}
		got := make([]byte, len(pattern))
		if err := d.Queue().ReadBuffer(dst, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}
		return got
	}

	// The oracle half runs today, so this entry asserts something now rather
	// than only waiting: the CPU backend must already return caller order.
	cpu := roundTrip(t, openDevice(t))
	for i := range pattern {
		if cpu[i] != pattern[i] {
			row, col := i/(w*bpp), i%(w*bpp)
			t.Fatalf("the CPU oracle disagrees with the caller at row %d byte %d: %d, "+
				"want %d; that byte belongs to row %d, so this is %s",
				row, col, cpu[i], pattern[i], int(cpu[i])/16,
				flipOrShear(int(cpu[i])/16, row, h))
		}
	}

	gpu := roundTrip(t, openMetal(t))
	for i := range pattern {
		row, col := i/(w*bpp), i%(w*bpp)
		if gpu[i] != cpu[i] {
			// Name the row the byte actually came from, which is the
			// measurement docs/conventions.md says identifies a flip fastest.
			t.Fatalf("Metal row %d byte %d is %d and the CPU oracle has %d; that byte "+
				"belongs to row %d, so this is %s", row, col, gpu[i], cpu[i],
				int(gpu[i])/16, flipOrShear(int(gpu[i])/16, row, h))
		}
	}
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
