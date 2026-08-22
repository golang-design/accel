// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"strconv"
	"strings"
	"sync"

	"golang.design/x/accel/internal/alloc"
	"golang.design/x/accel/internal/driver"
)

// MemoryKind is where a pool's memory lives and who can reach it. It is the
// property that actually decides performance, so it is chosen explicitly rather
// than inferred. See specs/001-device-resources.md.
type MemoryKind int

const (
	// MemoryDevice is fast for the GPU and not host-visible. Weights, activations,
	// render targets.
	MemoryDevice MemoryKind = iota

	// MemoryUpload is host-writable and GPU-readable. Staging.
	MemoryUpload

	// MemoryReadback is GPU-writable and host-readable. Results.
	MemoryReadback

	// MemoryShared is host-visible and device-local at once. This is a real
	// capability on unified-memory hardware, where it removes a copy entirely,
	// not an alias for something else. Reported per device rather than assumed
	// from the platform: check Capabilities.SharedMemoryKind.
	MemoryShared
)

// String returns the memory kind's name.
func (k MemoryKind) String() string {
	switch k {
	case MemoryDevice:
		return "Device"
	case MemoryUpload:
		return "Upload"
	case MemoryReadback:
		return "Readback"
	case MemoryShared:
		return "Shared"
	}
	return "MemoryKind(" + strconv.Itoa(int(k)) + ")"
}

// PoolPolicy selects how a pool carves itself up. See specs/001-device-resources.md.
type PoolPolicy int

const (
	// PoolGeneral is a general-purpose pool: arbitrary allocation and free order,
	// O(1) allocate and free through a two-level segregated fit allocator, and
	// bounded internal fragmentation. The default, and what a caller holding
	// weights and caches wants.
	PoolGeneral PoolPolicy = iota

	// PoolLinear allocates by bumping a cursor and frees only by resetting the
	// whole pool, so an individual Close is a no-op against the memory. This is
	// what a Graph's transient pool uses: the graph computed every offset at build,
	// so that pool needs no runtime allocator at all.
	PoolLinear
)

// PoolDescriptor describes a pool to create.
type PoolDescriptor struct {
	Kind   MemoryKind
	Bytes  int
	Policy PoolPolicy

	// Textures reserves the pool for textures. Texture placement alignment is far
	// coarser than buffer alignment on some backends and some forbid the mixture
	// outright, so it is a pool property rather than a per-allocation one. Mixing
	// them would apply a texture's granularity to a model's thousands of tensors,
	// which is not a tax but a fatal multiplier.
	Textures bool

	// Label appears in allocation errors and in backend debug tooling.
	Label string
}

// Pool is a device memory allocation that buffers are suballocated from.
//
// Pooling exists because one device allocation per buffer is fine for a renderer
// with a handful and wrong for a model with thousands: allocation is expensive,
// drivers cap how many you may hold, and per-resource allocation forecloses the
// transient aliasing a [Graph] does when it plans memory.
type Pool struct {
	_ noCopy

	dev   *Device
	desc  PoolDescriptor
	block driver.Block
	alloc alloc.Allocator
	state resourceState

	mu   sync.Mutex
	live []*Buffer
}

// AllocTexture suballocates a texture from a pool created with Textures set.
// Buffer pools reject it, and texture pools reject [Pool.Alloc].
func (p *Pool) AllocTexture(desc TextureDescriptor) (*Texture, error) {
	panic(ErrNotImplemented)
}

// PoolStats reports a pool's occupancy.
type PoolStats struct {
	Size int
	Used int // sum of allocation sizes, which includes alignment padding
	Free int // Size - Used

	// LargestFree is the biggest single allocation this pool can still serve. The
	// gap between Free and LargestFree is fragmentation, and it is what predicts
	// an allocation failure rather than reporting it afterwards. A pool never
	// compacts, because a device address is already baked into descriptor sets and
	// recorded commands, so fragmentation inside a pool is permanent for its life.
	LargestFree int

	// Allocations is the live count and Blocks the number of free blocks. Rising
	// Blocks against flat Allocations is fragmentation accumulating.
	Allocations int
	Blocks      int
}

// DType is the element type of a buffer.
//
// Arithmetic inside a kernel is f32 unless the kernel asks otherwise: narrow
// types are storage formats that convert on load and store. That default is a
// correctness choice, not a convenience, since accumulating a long dot product
// in f16 loses accuracy badly. See specs/002-compute-model.md.
type DType int

const (
	F32 DType = iota

	// F16 and BF16 storage are universal. Native arithmetic on them is separately
	// gated by CapF16Arithmetic and CapBF16Arithmetic. BF16 trades precision for
	// the range of f32 at the same width.
	F16
	BF16

	// I32 and U32 are universal. Atomics operate on these.
	I32
	U32

	// I8 and U8 are storage and conversion types, for quantized weights.
	I8
	U8
)

// dtypeInfo is the one place a dtype's width and name are written down. A
// storage buffer is a tightly packed array of one dtype, so this size is also
// its element stride: there is no padding anywhere, ever. See
// specs/001-device-resources.md section 3.2.
var dtypeInfo = [...]struct {
	name string
	size int
}{
	F32:  {"f32", 4},
	F16:  {"f16", 2},
	BF16: {"bf16", 2},
	I32:  {"i32", 4},
	U32:  {"u32", 4},
	I8:   {"i8", 1},
	U8:   {"u8", 1},
}

// Size returns the dtype's size in bytes.
func (d DType) Size() int {
	if d < 0 || int(d) >= len(dtypeInfo) {
		return 0
	}
	return dtypeInfo[d].size
}

// String returns the dtype's name.
func (d DType) String() string {
	if d < 0 || int(d) >= len(dtypeInfo) {
		return "DType(" + strconv.Itoa(int(d)) + ")"
	}
	return dtypeInfo[d].name
}

// BufferUsage declares how a buffer will be used.
//
// Usage is declared at creation because backends need it: it decides the
// underlying allocation flags. Using a buffer in a way it did not declare is a
// validation error when the graph is built, not undefined behaviour when it runs.
type BufferUsage uint32

const (
	UsageStorage BufferUsage = 1 << iota
	UsageUniform
	UsageIndex
	UsageVertex
	UsageIndirect
	UsageCopySrc
	UsageCopyDst
)

// usageNames is the declaration order of the usage bits, for String.
var usageNames = []string{"UsageStorage", "UsageUniform", "UsageIndex", "UsageVertex",
	"UsageIndirect", "UsageCopySrc", "UsageCopyDst"}

// String returns the usage set as a |-separated list, which is how validation
// errors name what a buffer declared against what it needs.
func (u BufferUsage) String() string {
	if u == 0 {
		return "no usage"
	}
	var parts []string
	for i, name := range usageNames {
		if u&(1<<uint(i)) != 0 {
			parts = append(parts, name)
		}
	}
	if rest := u &^ (1<<uint(len(usageNames)) - 1); rest != 0 {
		parts = append(parts, "BufferUsage("+strconv.FormatUint(uint64(rest), 10)+")")
	}
	return strings.Join(parts, "|")
}

// BufferDescriptor describes a buffer to create.
type BufferDescriptor struct {
	// DType and Count give the buffer's element type and length. Size in bytes is
	// DType.Size() * Count.
	DType DType
	Count int

	Usage BufferUsage

	// Label appears in validation errors and in backend debug tooling. Worth
	// setting: a build error naming "kv_cache_layer_7" beats one naming a pointer.
	Label string
}

// Buffer is a typed, sized range of device memory.
type Buffer struct {
	_ noCopy

	pool  *Pool
	desc  BufferDescriptor
	alloc *alloc.Allocation
	bytes int
	state resourceState
}

// BufferView is a sub-range of a [Buffer], possibly at a different dtype. It is a
// value: copying it is fine and does not copy any memory.
type BufferView struct {
	Buffer *Buffer
	DType  DType
	Offset int
	Count  int
}
