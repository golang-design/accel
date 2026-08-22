// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import "fmt"

// DType reports the buffer's element type.
func (b *Buffer) DType() DType { return b.desc.DType }

// Count reports the buffer's element count.
func (b *Buffer) Count() int { return b.desc.Count }

// Usage reports what the buffer declared at creation.
func (b *Buffer) Usage() BufferUsage { return b.desc.Usage }

// Bytes reports the buffer's size in bytes, which is DType.Size() times Count.
// It is the caller's number and is never rounded; the pool's allocation size,
// which includes alignment padding, shows up in [PoolStats] instead.
func (b *Buffer) Bytes() int { return b.bytes }

// View returns a sub-range of the buffer as a [BufferView].
//
// Views are what let a caller slice a KV cache or address one attention head
// without copying. A view does not own memory and must not outlive its buffer.
//
// Offset and count are in elements of the buffer's dtype. Creating a view is not
// an operation the device sees and carries no alignment requirement of its own:
// the alignment a view owes depends on what it is *for*, so it is checked where
// the view reaches a binding or a copy rather than here. See
// specs/001-device-resources.md section 6.1.
func (b *Buffer) View(offset, count int) (BufferView, error) {
	return b.ViewAs(b.desc.DType, offset, count)
}

// ViewAs is [Buffer.View] with a reinterpreted element type. It reports an error
// if the byte ranges do not divide evenly at the new dtype.
//
// Offset and count are in elements of d, the *new* dtype, not of the buffer's.
// Anything else would make ViewAs require the caller to do the arithmetic ViewAs
// exists to do.
//
// It reinterprets and never converts. The bytes are unchanged, so a u32 view of
// an f32 buffer sees the IEEE 754 binary32 encoding of each value: that is what
// lets a kernel read a quantized plane as u8 and a scale plane as bit-packed
// u16, and what lets a debug path dump any buffer as u8. Reinterpreting across
// widths is exact byte-wise and is only meaningful because accel requires the
// host and device to share byte order, which every supported platform does.
func (b *Buffer) ViewAs(d DType, offset, count int) (BufferView, error) {
	if err := b.state.checkOpen("ViewAs"); err != nil {
		return BufferView{}, err
	}
	elem := d.Size()
	if elem == 0 {
		return BufferView{}, fmt.Errorf("accel: ViewAs on %q: %v is not a dtype", b.desc.Label, d)
	}
	if offset < 0 || count < 0 {
		return BufferView{}, fmt.Errorf("accel: ViewAs(%v) on %q: offset %d and count %d must not be negative",
			d, b.desc.Label, offset, count)
	}
	if (offset+count)*elem > b.bytes {
		return BufferView{}, fmt.Errorf("accel: ViewAs(%v) on %q (%v, %d elements): elements "+
			"[%d, %d) are bytes [%d, %d), past the buffer's %d",
			d, b.desc.Label, b.desc.DType, b.desc.Count,
			offset, offset+count, offset*elem, (offset+count)*elem, b.bytes)
	}
	// A view covering the whole buffer at a new dtype has to divide evenly. That
	// is the "sizes work out" case, and it is the only one: a partial view needs
	// nothing beyond its own range lying inside the buffer.
	if offset == 0 && count == b.bytes/elem && count*elem != b.bytes {
		return BufferView{}, fmt.Errorf("accel: ViewAs(%v) on %q (%v, %d elements): byte length %d "+
			"is not a multiple of %d", d, b.desc.Label, b.desc.DType, b.desc.Count, b.bytes, elem)
	}
	return BufferView{Buffer: b, DType: d, Offset: offset, Count: count}, nil
}

// Close releases the buffer.
//
// Closing while something using this buffer is still outstanding is reported
// rather than crashing: the implementation keeps a resource alive until every
// hold on it is gone, so the memory comes back later and the caller learns that
// their teardown ordering was wrong.
func (b *Buffer) Close() error {
	if !b.state.beginClose() {
		return nil
	}
	if b.state.release() {
		b.free()
		return nil
	}
	// Something still holds it. The memory is not returned, the buffer stays
	// valid for whatever holds it, and the caller is told rather than crashed.
	return &LifetimeError{
		Op:       "Close",
		Resource: b.desc.Label,
		Reason:   reasonPending,
		InFlight: b.state.holds(),
	}
}

// free returns the buffer's memory to its pool. It runs exactly once, when the
// last hold goes away, which is either inside Close or inside the path that
// completes the work still holding it.
func (b *Buffer) free() {
	p := b.pool
	p.mu.Lock()
	// A linear pool frees by Reset, so an individual free is a no-op against the
	// memory. The accounting still retires, because the handle is gone either way.
	if p.desc.Policy == PoolGeneral {
		_ = p.alloc.Free(b.alloc)
	}
	p.mu.Unlock()
	p.forget(b)
}

// release drops one hold, freeing the buffer if it was the last.
func (b *Buffer) release() {
	if b.state.release() {
		b.free()
	}
}

// check validates a view at the point of use.
//
// A BufferView is a plain value with an exported *Buffer field, so it can be
// copied, stored, and constructed by hand with nonsense in it. That makes "a
// view must not outlive its buffer" a rule needing a mechanism rather than a
// sentence: the view holds no reference, the Go pointer keeps the Buffer object
// alive so the closed flag is always readable, and every use re-validates the
// range against the live buffer. The worst outcome is a rejection. See
// specs/001-device-resources.md section 7.3.
func (v BufferView) check(op string) error {
	if v.Buffer == nil {
		return fmt.Errorf("accel: %s: the view names no buffer", op)
	}
	if err := v.Buffer.state.checkOpen(op); err != nil {
		return err
	}
	elem := v.DType.Size()
	if elem == 0 {
		return fmt.Errorf("accel: %s on %q: %v is not a dtype", op, v.Buffer.desc.Label, v.DType)
	}
	if v.Offset < 0 || v.Count < 0 || (v.Offset+v.Count)*elem > v.Buffer.bytes {
		return fmt.Errorf("accel: %s on %q: elements [%d, %d) of %v are bytes [%d, %d), "+
			"outside the buffer's %d", op, v.Buffer.desc.Label, v.Offset, v.Offset+v.Count,
			v.DType, v.Offset*elem, (v.Offset+v.Count)*elem, v.Buffer.bytes)
	}
	return nil
}

// byteRange reports the view's extent relative to the start of its buffer.
func (v BufferView) byteRange() (offset, length int) {
	elem := v.DType.Size()
	return v.Offset * elem, v.Count * elem
}
