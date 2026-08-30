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
	// declared usage requires. A multiple of 256 is always sufficient on every
	// backend; these report what this device actually needs, for callers who want
	// the waste back.
	MinStorageBufferOffsetAlignment int
	MinUniformBufferOffsetAlignment int

	// MinBufferCopyOffsetAlignment and MinBufferCopyRowPitchAlignment constrain
	// transfers rather than bindings, which is why a texture readback can cost a
	// repack. accel guarantees tightly packed rows to the caller and pays for the
	// repack itself, so these are reported for callers sizing their own staging,
	// not imposed on them. See docs/conventions.md.
	MinBufferCopyOffsetAlignment   int
	MinBufferCopyRowPitchAlignment int

	// MinTexturePlacementAlignment is the alignment a texture's backing memory must
	// start at inside a pool. It is far coarser than any buffer alignment on some
	// backends, which is why a pool is either a buffer pool or a texture pool and
	// never both.
	MinTexturePlacementAlignment int

	// MaxBufferBytes is the largest single buffer, MaxPoolBytes the largest single
	// device allocation, and MaxPools the driver's cap on live allocations.
	// MaxPools is the number that makes pooling mandatory rather than merely
	// efficient.
	MaxBufferBytes int
	MaxPoolBytes   int
	MaxPools       int

	MaxTextureExtent2D    int
	MaxTextureExtent3D    int
	MaxTextureArrayLayers int

	// MaxUniformBlockBytes is the largest std140 block a uniform binding may
	// carry. A kernel's uniform struct is encoded to a block whose size the
	// generator bakes into the pipeline, so without this there is no device number
	// to validate that size against, and a struct that is too large for the device
	// would be discovered at pipeline creation on somebody else's machine.
	MaxUniformBlockBytes int

	// Compute limits constrain generated kernel metadata and dispatches. Feature
	// availability remains in Capabilities.
	MaxWorkgroupSize             [3]int
	MaxWorkgroupInvocations      int
	MaxWorkgroupCount            [3]int
	MaxSharedMemoryBytes         int
	MaxStorageBufferBindingBytes int
	MaxBindingsPerKind           int

	// MaxColorAttachments is how many colour targets one render pass may have.
	// Zero means the backend does not report one, which at v0 means it has no
	// render path — a limit of zero and "no limit" are the same answer for a
	// backend that cannot draw.
	MaxColorAttachments int

	// MaxVertexBuffers is how many vertex buffers one draw may bind.
	//
	// Reported per device rather than taken from a constant. It used to be
	// mslabi.StageVertexBufferLimit, which is *Metal's* reservation -- the
	// index a stage's uniforms begin at -- so every device, including the CPU
	// oracle, was refused at a ceiling one backend's ABI happens to have.
	// specs/042-surface-completion.md section 5.3 names that as a
	// backend-specific type shaping the public API, which is
	// specs/000-decisions.md's layering rule 3.
	MaxVertexBuffers int

	// Devices without subgroups report 1/1 while Capabilities.Subgroups is false,
	// so every opened device still has positive numeric limits.
	MinSubgroupSize int
	MaxSubgroupSize int
}

// FormatInfo describes what a [Format] is and what a device can do with it.
//
// Capability is per device, not per format: a format that is renderable on one
// backend may be sampleable only on another. Asking the device is the only
// correct way to find out, which is why this is a method rather than a table
// constant.
type FormatInfo struct {
	Format        Format
	BytesPerPixel int // 0 when the layout is device-defined, as for Depth24PlusStencil8
	Channels      int

	IsDepth   bool
	IsStencil bool
	IsSRGB    bool

	Renderable bool
	Sampleable bool

	// Filterable is not implied by Sampleable. The 32-bit float formats are
	// sampleable everywhere and linearly filterable only where the device says so,
	// and assuming otherwise is how a resolve ends up with nearest-neighbour
	// artefacts on one vendor only.
	Filterable bool

	// StorageRead and StorageWrite are separate because an sRGB format is neither:
	// its transfer function is applied by fixed-function hardware that a storage
	// write bypasses.
	StorageRead  bool
	StorageWrite bool

	Blendable    bool
	HostCopyable bool
}

// CopyStats reports what a transfer does.
//
// It is a plan-time fact rather than a measurement: a backend knows its own pitch
// rules before anything executes, so a recorded copy carries this from
// [Graph.NodeStats] as soon as the graph is built. The immediate transfer path has
// no node, so its repacks are counted in [QueueStats] instead of returned per
// call, which keeps an observability concern out of two signatures every caller
// touches.
type CopyStats struct {
	Bytes    int
	Repacked bool // an intermediate padded-pitch buffer is used
	RowPitch int  // the pitch the backend uses on the device side
}
