// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"fmt"

	"golang.design/x/accel/internal/driver"
)

// The format table of specs/001-device-resources.md section 4.1.
//
// # What is here and what is the device's
//
// This table holds what a format *is*: its width, its channel count, whether it
// carries a transfer function, whether it is a depth format. Those are
// properties of the encoding and do not vary by device.
//
// What a device can *do* with a format is not here. A format renderable on one
// backend may be sampleable only on another, which is why [Device.FormatInfo]
// is a method: 006's table marks several entries `cap`, meaning queryable and
// device-dependent, and one `?`, meaning unknown until measured. Baking any of
// those into a constant would be the confidently-wrong-number failure that spec
// warns about.
type formatEntry struct {
	name     string
	bpp      int
	channels int
	depth    bool
	stencil  bool
	srgb     bool

	// baseline is what every backend in 006's table marks `yes` for this
	// format. A device may report more; it may not report less.
	renderable   bool
	sampleable   bool
	filterable   bool
	storageRead  bool
	storageWrite bool
	blendable    bool
}

var formatTable = map[Format]formatEntry{
	RGBA8Unorm: {
		name: "RGBA8Unorm", bpp: 4, channels: 4,
		renderable: true, sampleable: true, filterable: true,
		storageRead: true, storageWrite: true, blendable: true,
	},
	RGBA8UnormSRGB: {
		name: "RGBA8UnormSRGB", bpp: 4, channels: 4, srgb: true,
		renderable: true, sampleable: true, filterable: true, blendable: true,
		// No storage, on any backend. The transfer function is applied by
		// fixed-function hardware that a storage write bypasses, so a storage
		// image of an sRGB format would write values the sampler then decodes a
		// second time. Architecturally absent rather than merely unsupported.
	},
	BGRA8Unorm: {
		name: "BGRA8Unorm", bpp: 4, channels: 4,
		renderable: true, sampleable: true, filterable: true, blendable: true,
	},
	R16Float: {
		name: "R16Float", bpp: 2, channels: 1,
		renderable: true, sampleable: true, filterable: true, blendable: true,
	},
	RG16Float: {
		name: "RG16Float", bpp: 4, channels: 2,
		renderable: true, sampleable: true, filterable: true, blendable: true,
	},
	RGBA16Float: {
		name: "RGBA16Float", bpp: 8, channels: 4,
		renderable: true, sampleable: true, filterable: true,
		storageRead: true, storageWrite: true, blendable: true,
	},
	R32Float: {
		name: "R32Float", bpp: 4, channels: 1,
		renderable: true, sampleable: true,
		// Filterable is deliberately false: the 32-bit float formats are
		// sampleable everywhere and linearly filterable only where the device
		// says so. Assuming otherwise is how a resolve ends up with
		// nearest-neighbour artefacts on one vendor and nobody else's.
		storageRead: true, storageWrite: true,
	},
	RG32Float: {
		name: "RG32Float", bpp: 8, channels: 2,
		renderable: true, sampleable: true,
	},
	RGBA32Float: {
		name: "RGBA32Float", bpp: 16, channels: 4,
		renderable: true, sampleable: true,
		storageRead: true, storageWrite: true,
	},
	Depth32Float: {
		name: "Depth32Float", bpp: 4, channels: 1, depth: true,
		renderable: true, sampleable: true,
	},
	Depth24PlusStencil8: {
		name: "Depth24PlusStencil8", channels: 2, depth: true, stencil: true,
		// bpp is deliberately zero: the layout is device-defined. "24 plus"
		// means at least 24 bits of depth, and a backend may store it as 32
		// with 8 unused or pack it with the stencil. A caller sizing a readback
		// from a guessed width would be wrong on some device, so the format
		// says it does not know rather than guessing four.
		renderable: true,
	},
}

// BytesPerPixel returns the format's size, or zero when the layout is
// device-defined.
//
// Always ask rather than assume. Assuming is a real bug and it surfaces as an
// out-of-range panic during readback rather than as a wrong image, which is at
// least loud — but only on the machine whose format happened to differ.
func (f Format) BytesPerPixel() int { return formatTable[f].bpp }

// IsDepth reports whether the format is a depth or depth-stencil format.
func (f Format) IsDepth() bool { return formatTable[f].depth }

// String returns the format's name.
func (f Format) String() string {
	if e, ok := formatTable[f]; ok {
		return e.name
	}
	if f == FormatInvalid {
		return "FormatInvalid"
	}
	return fmt.Sprintf("Format(%d)", int(f))
}

// valid reports whether a format is one a texture may be created with.
func (f Format) valid() bool {
	_, ok := formatTable[f]
	return ok
}

// plan is the format's spelling in a [driver.Plan].
//
// Written out rather than converted numerically, for the reason
// [driver.Format] gives: two enumerations that happen to agree today are one
// insertion away from shifting every value after it, and the result of a shift
// is a plausible image rather than an error.
//
// An unknown format maps to [driver.FormatInvalid] rather than being guessed at.
// A plan that carried it is refused by [driver.Plan.Validate], which is where a
// backend learns rather than where it decodes bytes by assumption.
func (f Format) plan() driver.Format {
	switch f {
	case RGBA8Unorm:
		return driver.RGBA8Unorm
	case RGBA8UnormSRGB:
		return driver.RGBA8UnormSRGB
	case BGRA8Unorm:
		return driver.BGRA8Unorm
	case R16Float:
		return driver.R16Float
	case RG16Float:
		return driver.RG16Float
	case RGBA16Float:
		return driver.RGBA16Float
	case R32Float:
		return driver.R32Float
	case RG32Float:
		return driver.RG32Float
	case RGBA32Float:
		return driver.RGBA32Float
	case Depth32Float:
		return driver.Depth32Float
	case Depth24PlusStencil8:
		return driver.Depth24PlusStencil8
	}
	return driver.FormatInvalid
}

// FormatInfo reports what this device can do with a format.
//
// An unsupported format reports a zero FormatInfo and no error: absence is a
// capability answer rather than a failure, which is decision 6 applied to
// formats. A caller checks the fields and picks a different format; it does not
// discover the answer at a draw call.
func (d *Device) FormatInfo(f Format) FormatInfo {
	e, ok := formatTable[f]
	if !ok {
		return FormatInfo{}
	}
	info := FormatInfo{
		Format:        f,
		BytesPerPixel: e.bpp,
		Channels:      e.channels,
		IsDepth:       e.depth,
		IsStencil:     e.stencil,
		IsSRGB:        e.srgb,
		Renderable:    e.renderable,
		Sampleable:    e.sampleable,
		Filterable:    e.filterable,
		StorageRead:   e.storageRead,
		StorageWrite:  e.storageWrite,
		Blendable:     e.blendable,
	}

	// A device may report more than the baseline and never less. The CPU
	// backend is the one place where "what the hardware can do" is a software
	// question, so it reports the baseline plus filtering on the float formats,
	// which it can do exactly.
	if d.info.Backend == BackendCPU && !e.depth {
		info.Filterable = e.sampleable
	}
	// Host copyability follows the memory kind rather than the format, and a
	// depth texture is device-private on macOS, so it is not host-copyable on
	// any backend that shares that constraint. The CPU backend enforces it
	// too: a rule no device enforces on a laptop is a rule discovered in
	// production.
	info.HostCopyable = !e.depth
	return info
}

// AlignedRowPitch returns the row pitch a texture-to-buffer copy of the given
// format and width will use on this device.
//
// # Why a caller does not need this
//
// specs/001-device-resources.md section 4.2 guarantees that at the accel API
// boundary texture data is tightly packed: row r begins at r*width*bpp with no
// padding. A caller sizes a readback as width*height*bpp and is always right,
// and accel pays for any repack.
//
// This exists so a caller sizing its *own* staging buffer can size it right the
// first time, and so a caller on a performance path can see the repack coming
// rather than measuring it.
func (d *Device) AlignedRowPitch(f Format, width int) int {
	bpp := f.BytesPerPixel()
	if bpp == 0 || width <= 0 {
		return 0
	}
	return alignUp(width*bpp, d.info.Limits.MinBufferCopyRowPitchAlignment)
}

// AlignedRowPitchRepacks reports whether a copy of this shape needs an
// intermediate padded buffer on this device.
//
// Exposed alongside [Device.AlignedRowPitch] for the same reason: a caller on a
// performance path can see the extra full-size copy coming rather than
// measuring it.
func (d *Device) AlignedRowPitchRepacks(f Format, width int) bool {
	return d.repacks(f, width)
}

// tightRowPitch is what the caller sees: width times bytes per pixel, with no
// padding between rows.
func tightRowPitch(f Format, width int) int { return width * f.BytesPerPixel() }

// repacks reports whether a copy of this shape needs an intermediate buffer.
//
//	tight  = w · bpp
//	device = ⌈w · bpp / A⌉ · A
//	repack ⟺ device ≠ tight
//
// A 1024-wide RGBA8Unorm target has a 4096-byte row, already a multiple of 256,
// and repacks nowhere. A 100-wide one has 400 bytes and repacks on any backend
// whose alignment is 256.
func (d *Device) repacks(f Format, width int) bool {
	return d.AlignedRowPitch(f, width) != tightRowPitch(f, width)
}
