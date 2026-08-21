// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

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

// Pool is a device memory allocation that buffers are suballocated from.
//
// Pooling exists because one device allocation per buffer is fine for a renderer
// with a handful and wrong for a model with thousands: allocation is expensive,
// drivers cap how many you may hold, and per-resource allocation forecloses the
// transient aliasing a [Graph] does when it plans memory.
type Pool struct{ _ noCopy }

// Alloc suballocates a buffer from the pool.
func (p *Pool) Alloc(desc BufferDescriptor) (*Buffer, error) { panic(ErrNotImplemented) }

// Stats reports the pool's size, how much is in use, and how much is free.
func (p *Pool) Stats() PoolStats { panic(ErrNotImplemented) }

// Close releases the pool. Buffers suballocated from it must be closed first.
func (p *Pool) Close() error { panic(ErrNotImplemented) }

// PoolStats reports a pool's occupancy.
type PoolStats struct {
	Size int
	Used int
	Free int
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

	// F16 and BF16 are capabilities, not guarantees. BF16 trades precision for
	// the range of f32 at the same width, and backend support for it is thinner.
	F16
	BF16

	// I32 and U32 are universal. Atomics operate on these.
	I32
	U32

	// I8 and U8 are storage and conversion types, for quantized weights.
	I8
	U8
)

// Size returns the dtype's size in bytes.
func (d DType) Size() int { panic(ErrNotImplemented) }

// String returns the dtype's name.
func (d DType) String() string { panic(ErrNotImplemented) }

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
type Buffer struct{ _ noCopy }

// DType reports the buffer's element type.
func (b *Buffer) DType() DType { panic(ErrNotImplemented) }

// Count reports the buffer's element count.
func (b *Buffer) Count() int { panic(ErrNotImplemented) }

// View returns a sub-range of the buffer as a [BufferView].
//
// Views are what let a caller slice a KV cache or address one attention head
// without copying. A view does not own memory and must not outlive its buffer.
func (b *Buffer) View(offset, count int) (BufferView, error) { panic(ErrNotImplemented) }

// ViewAs is [Buffer.View] with a reinterpreted element type. It reports an error
// if the byte ranges do not divide evenly at the new dtype.
func (b *Buffer) ViewAs(d DType, offset, count int) (BufferView, error) {
	panic(ErrNotImplemented)
}

// Write copies host data into the buffer. It is immediate from the caller's point
// of view and asynchronous with respect to the device: it returns once the data
// has been staged, not once the device can see it.
//
// It takes part in no graph dependency tracking. For a transfer that must be
// ordered against other work, record it with [Recorder.CopyToBuffer] instead.
func (b *Buffer) Write(offset int, data any) error { panic(ErrNotImplemented) }

// Read copies buffer contents back to the host, blocking until the data is ready.
//
// Unlike [Buffer.Write] this is synchronous, because there is nothing useful to
// return early with. It therefore drains outstanding work touching this buffer,
// which makes it the wrong call in a hot loop.
func (b *Buffer) Read(offset int, into any) error { panic(ErrNotImplemented) }

// Close releases the buffer.
//
// Closing while a submission using this buffer is in flight is reported rather
// than crashing: the implementation keeps a submission's resources alive until
// its fence signals.
func (b *Buffer) Close() error { panic(ErrNotImplemented) }

// BufferView is a sub-range of a [Buffer], possibly at a different dtype. It is a
// value: copying it is fine and does not copy any memory.
type BufferView struct {
	Buffer *Buffer
	DType  DType
	Offset int
	Count  int
}
