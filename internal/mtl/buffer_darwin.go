// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl

import (
	"unsafe"

	"github.com/ebitengine/purego/objc"
)

// StorageMode is the MTLResourceOptions storage-mode field, already shifted
// into place.
//
// The shift is why these are constants rather than an iota: MTLResourceOptions
// packs the storage mode at bit 4, and a value written unshifted is a legal
// options word meaning something else, so it would allocate rather than fail.
type StorageMode uint

const (
	StorageShared  StorageMode = 0 << 4
	StoragePrivate StorageMode = 2 << 4
)

// hostVisible reports whether a mode's memory may be mapped.
//
// The mode is the authority, and -contents is not. Metal documents -contents as
// nil for private storage; on Apple silicon it returns a valid pointer instead,
// because the memory genuinely is addressable and only the API contract says
// otherwise. Trusting the object over the mode therefore makes every buffer
// mappable on this machine and not on an Intel Mac, which converts a
// portability rule into a machine-specific one. See docs/conventions.md.
func (m StorageMode) hostVisible() bool { return m == StorageShared }

// Buffer is an MTLBuffer, retained.
type Buffer struct {
	id    objc.ID
	size  int
	bytes []byte // nil for private storage
}

var (
	selNewBufferWithLength = objc.RegisterName("newBufferWithLength:options:")
	selContents            = objc.RegisterName("contents")
)

// NewBuffer allocates device memory.
//
// The returned buffer is +1 from a new* selector and is not retained again, per
// the ownership rule in objc_darwin.go.
func (d *Device) NewBuffer(size int, mode StorageMode) (*Buffer, error) {
	if size <= 0 {
		return nil, errSize(size)
	}
	b := &Buffer{size: size}
	withPool(func() {
		b.id = d.id.Send(selNewBufferWithLength, uintptr(size), uintptr(mode))
		if b.id == 0 {
			return
		}
		if !mode.hostVisible() {
			return
		}
		if p := objc.Send[unsafe.Pointer](b.id, selContents); p != nil {
			b.bytes = unsafe.Slice((*byte)(p), size)
		}
	})
	if b.id == 0 {
		return nil, errAlloc(size)
	}
	return b, nil
}

// Bytes is the host mapping, or nil when this memory is not host-visible.
func (b *Buffer) Bytes() []byte { return b.bytes }

// Size is the allocation length in bytes.
func (b *Buffer) Size() int { return b.size }

// Close releases the buffer.
func (b *Buffer) Close() {
	release(b.id)
	b.id = 0
	b.bytes = nil
}
