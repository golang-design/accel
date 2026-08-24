// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"encoding/binary"
	"fmt"
	"math"
)

// UniformCodec is generated for one by-value kernel parameter type. It owns the
// std140 size, alignment, field offsets, and typed encoder.
//
// A caller never writes one and never spells an offset. The padding std140
// imposes is not Go's, and a caller who computed it by hand would be right for a
// struct of four floats and wrong for the first one containing a
// three-component vector. See specs/001-device-resources.md section 3.3.
type UniformCodec[T any] interface {
	// EncodedSize is the block's size in bytes, rounded up to sixteen. It is
	// what a uniform buffer is allocated at and what a pipeline's declared block
	// size is checked against.
	EncodedSize() int

	// Encode writes value into dst, which must be at least EncodedSize long.
	Encode(dst []byte, value T) error
}

// UniformWriter is the low-level half a generated codec is built from.
//
// It exists so a generated encoder is a list of field writes rather than a list
// of byte offsets: the offsets are computed once, by the layout, and the
// generated code names members. A caller who already manages a uniform arena
// may use it directly and still never spells padding.
type UniformWriter struct {
	dst []byte
	err error
}

// NewUniformWriter wraps a destination buffer.
func NewUniformWriter(dst []byte) *UniformWriter { return &UniformWriter{dst: dst} }

// Err reports the first failure, so a generated encoder can write every field
// and check once rather than checking after each.
func (w *UniformWriter) Err() error { return w.err }

func (w *UniformWriter) fail(format string, args ...any) {
	if w.err == nil {
		w.err = fmt.Errorf("accel: "+format, args...)
	}
}

// room reports whether n bytes fit at offset, recording the failure if not.
func (w *UniformWriter) room(offset, n int) bool {
	if w.err != nil {
		return false
	}
	if offset < 0 || offset+n > len(w.dst) {
		w.fail("uniform encode writes bytes [%d, %d) of a %d-byte block: the destination is "+
			"smaller than the codec's encoded size", offset, offset+n, len(w.dst))
		return false
	}
	return true
}

// F32 writes a 32-bit float at a std140 offset.
func (w *UniformWriter) F32(offset int, v float32) {
	if !w.room(offset, 4) {
		return
	}
	binary.LittleEndian.PutUint32(w.dst[offset:], math.Float32bits(v))
}

// I32 writes a signed 32-bit integer at a std140 offset.
func (w *UniformWriter) I32(offset int, v int32) {
	if !w.room(offset, 4) {
		return
	}
	binary.LittleEndian.PutUint32(w.dst[offset:], uint32(v))
}

// U32 writes an unsigned 32-bit integer at a std140 offset.
func (w *UniformWriter) U32(offset int, v uint32) {
	if !w.room(offset, 4) {
		return
	}
	binary.LittleEndian.PutUint32(w.dst[offset:], v)
}

// Little-endian, because accel requires the host and the device to share byte
// order and every supported platform is little-endian. A big-endian host is
// rejected at device open rather than byte-swapped here, per spec 001 section
// 3.5.

// UniformBuffer owns a generated-codec uniform allocation.
//
// It exists so that a value may change between submissions without changing
// graph structure: [UniformBuffer.Write] encodes through the queue, and the
// graph's binding does not move. A caller who wrote std140 bytes into an
// ordinary buffer would be doing the codec's job with the padding hidden.
type UniformBuffer[T any] struct {
	_ noCopy

	buf   *Buffer
	codec UniformCodec[T]
	scrat []byte
}

// NewUniformBuffer allocates storage sized and aligned by codec.
func NewUniformBuffer[T any](d *Device, codec UniformCodec[T]) (*UniformBuffer[T], error) {
	if d == nil || codec == nil {
		return nil, fmt.Errorf("accel: NewUniformBuffer needs a device and a codec")
	}
	size := codec.EncodedSize()
	if size <= 0 {
		return nil, fmt.Errorf("accel: NewUniformBuffer: the codec reports a size of %d", size)
	}
	// The block size is checked against the device rather than assumed. Without
	// this the failure lands on whichever machine happened to have the smaller
	// number, which is spec 001 section 3.3's reason for the limit existing.
	if limit := d.Limits().MaxUniformBlockBytes; size > limit {
		return nil, fmt.Errorf("%w: a uniform block of %s exceeds this device's "+
			"MaxUniformBlockBytes of %s. std140 pads: an array of scalars has a stride of "+
			"sixteen, so arrays belong in storage buffers (spec 001 section 3.3)",
			ErrUnsupported, humanBytes(size), humanBytes(limit))
	}

	// U8 with Count equal to the encoded size: a std140 block has no scalar
	// dtype, so on a uniform binding a dtype means bytes rather than elements.
	// That is one of exactly two exceptions in the design, and spec 001 section
	// 3.3 names both.
	buf, err := d.NewBuffer(BufferDescriptor{
		DType: U8, Count: size,
		Usage: BufferUniform | BufferCopyDst,
		Label: "uniform",
	})
	if err != nil {
		return nil, err
	}
	return &UniformBuffer[T]{buf: buf, codec: codec, scrat: make([]byte, size)}, nil
}

// Write encodes value into the buffer through a queue.
func (u *UniformBuffer[T]) Write(q *Queue, value T) error {
	if q == nil {
		return fmt.Errorf("accel: UniformBuffer.Write needs a queue: the encode goes through one, " +
			"so that it is ordered against that queue's other work")
	}
	if err := u.codec.Encode(u.scrat, value); err != nil {
		return err
	}
	return q.WriteBuffer(u.buf, 0, u.scrat)
}

// Buffer is the underlying allocation, for binding it into a graph.
func (u *UniformBuffer[T]) Buffer() *Buffer { return u.buf }

// View is the whole block as a binding range.
func (u *UniformBuffer[T]) View() (BufferView, error) {
	return u.buf.View(0, u.codec.EncodedSize())
}

// Close releases the allocation.
func (u *UniformBuffer[T]) Close() error { return u.buf.Close() }
