// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

// Limits are the device's numeric bounds.
//
// They are separate from [Capabilities] on purpose. A capability is a boolean
// that gates a code path; a limit is an integer that appears in arithmetic, and
// the failure modes differ. A backend that forgets to report a capability leaves
// it false, and the affected path is simply not taken. A backend that forgets to
// report a limit leaves it zero, and zero is a divide-by-zero or an alignment of
// one, which is worse than useless. Keeping them in separate structs makes the
// missing-limit case obvious at the point a backend is written.
//
// See specs/001-device-resources.md.
type Limits struct {
	// MinStorageBufferOffsetAlignment and MinUniformBufferOffsetAlignment are the
	// alignments a bound buffer range must satisfy. They constrain suballocation
	// directly: a pool hands out offsets that satisfy the strictest alignment any
	// declared usage requires.
	MinStorageBufferOffsetAlignment int
	MinUniformBufferOffsetAlignment int

	// OptimalBufferCopyRowPitchAlignment is the row pitch a texture-to-buffer copy
	// wants. D3D12 requires 256 bytes and others differ; accel guarantees tightly
	// packed rows to the caller and pays for the repack itself, so this is
	// reported for callers sizing their own staging, not imposed on them. See
	// docs/conventions.md.
	OptimalBufferCopyRowPitchAlignment int
	OptimalBufferCopyOffsetAlignment   int

	MaxBufferBytes        int
	MaxStorageBufferRange int
	MaxUniformBufferRange int
	MaxTextureDimension2D int
	MaxTextureArrayLayers int

	// MaxBoundBuffers and MaxBoundTextures cap one pipeline's binding layout.
	MaxBoundBuffers  int
	MaxBoundTextures int
}

// Limits reports the device's numeric bounds.
func (d *Device) Limits() Limits { panic(ErrNotImplemented) }

// FormatInfo describes what a [Format] is and what a device can do with it.
//
// Capability is per device, not per format: a format that is renderable on one
// backend may be sampleable only on another. Asking the device is the only
// correct way to find out, which is why this is a method rather than a table
// constant.
type FormatInfo struct {
	Format        Format
	BytesPerPixel int
	Channels      int

	Renderable   bool
	Sampleable   bool
	Storage      bool
	Blendable    bool
	HostCopyable bool
}

// FormatInfo reports what this device can do with a format. An unsupported format
// reports a zero FormatInfo and no error: absence is a capability answer, not a
// failure.
func (d *Device) FormatInfo(f Format) FormatInfo { panic(ErrNotImplemented) }

// AlignedRowPitch returns the row pitch a texture-to-buffer copy of the given
// format and width will use on this device, in bytes.
//
// Callers do not need this for correctness. Readback through [Texture.Read]
// returns tightly packed rows in caller order regardless, and accel pays for the
// repack. It exists so a caller sizing its own staging buffer can size it right
// the first time.
func (d *Device) AlignedRowPitch(f Format, width int) int { panic(ErrNotImplemented) }
