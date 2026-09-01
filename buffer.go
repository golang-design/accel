// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"fmt"

	"golang.design/x/accel/internal/driver"
)

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

// Access hands the buffer's host mapping to fn, for the duration of the call.
//
// # What this is for
//
// A model loader reads a shard, converts it, and uploads it — and without this
// it holds all three at once: the shard, a fully converted host tensor, and the
// device allocation. On a multi-gigabyte checkpoint that middle term is the
// largest transient allocation in the process, and it exists only because the
// converted bytes have nowhere to go but a slice of the caller's own.
//
// With this the loader converts *into* the destination. The middle allocation
// does not shrink; it does not exist.
//
// On unified-memory hardware — which [Capabilities.SharedMemoryKind] reports —
// the mapping *is* device memory, so a [MemoryShared] pool makes the upload
// free rather than fast. On a discrete device the mapping is the staging
// buffer, so the copy that remains is the one the hardware requires.
//
// # Why a callback and not a returned slice
//
// A returned slice is a promise about a lifetime accel cannot see. The pool can
// be closed, the device can be lost, and a slice that outlived either would be
// a use-after-free whose symptom is a plausible tensor. Scoped to the call, the
// borrow is bounded by something the compiler and the reader can both see, and
// the buffer's lifetime stays accel's.
//
// The slice is exactly this buffer's bytes: writing past its end is not
// possible, and writing to it is visible to the device without a further call.
//
// # When it refuses
//
// Memory that is not host-visible has no mapping to hand out. That is a
// property of the pool's [MemoryKind], reported rather than discovered:
// [MemoryDevice] refuses here even on the CPU backend, where the memory
// physically could be mapped, because a rule only one backend enforces is a
// rule that fails in production.
//
// Do not retain the slice. Do not call this while a graph reading this buffer
// is in flight; the bytes are the device's during a submission.
func (b *Buffer) Access(fn func([]byte) error) error {
	if fn == nil {
		return fmt.Errorf("accel: Buffer.Access needs a function; the mapping is valid " +
			"only for the duration of the call, so there is nothing to return")
	}
	if err := b.checkUsable("Buffer.Access"); err != nil {
		return err
	}
	blk, base := blockFor(b)
	host := driver.Unwrap(blk).Bytes()
	if host == nil {
		return fmt.Errorf("%w: Buffer.Access on %q: its pool is %v, which has no host "+
			"mapping. Allocate from a %v pool to write into device memory directly, or "+
			"use Queue.WriteBuffer, which stages the copy",
			ErrUsage, b.desc.Label, b.pool.desc.Kind, MemoryShared)
	}
	if base < 0 || base+b.bytes > len(host) {
		return fmt.Errorf("accel: Buffer.Access on %q: its %d bytes at offset %d are "+
			"outside the pool's %d", b.desc.Label, b.bytes, base, len(host))
	}
	return fn(host[base : base+b.bytes])
}

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
	if outOfRange(offset, count, elem, b.bytes) {
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
	// A graph transient's memory belongs to the builder, not to a pool, so
	// there is nothing here to return and b.pool is nil. Refusing is the same
	// answer BufferView.check gives for every other way of touching one from
	// outside its graph; without this, free dereferences the nil pool and the
	// process dies, which is exactly what that method's doc promises cannot
	// happen. See specs/001-device-resources.md section 7.3.
	if b.transient != nil {
		return &LifetimeError{
			Op:       "Close",
			Resource: b.desc.Label,
			Reason:   reasonTransient,
		}
	}
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
	if err := v.Buffer.checkUsable(op); err != nil {
		return err
	}
	return v.checkRange(op)
}

// checkUsable is the guard every entry point that touches a buffer's memory
// from outside a graph starts with: there is a buffer, the handle is open, and
// the buffer is not a graph transient.
//
// It runs before anything reads a field, so that a nil *Buffer is a named
// error at the call rather than a fault inside the library: WriteBuffer and
// ReadBuffer used to read the label and dtype first.
//
// A transient has no pool. Its memory belongs to the graph that declared it,
// arrives at Build, and may be reused between nodes, so only that graph may
// touch it. Every path that reaches for pool, alloc or block has to check this
// first, and it is one function because each path that carried its own copy
// was one that could lack it: Buffer.Access did, and dereferenced the nil
// block before Build and the nil pool after. See
// specs/001-device-resources.md section 7.3.
func (b *Buffer) checkUsable(op string) error {
	if b == nil {
		return fmt.Errorf("accel: %s: no buffer", op)
	}
	if err := b.state.checkOpen(op); err != nil {
		return err
	}
	if b.transient != nil {
		return fmt.Errorf("accel: %s on %q: it is a graph transient, whose memory the builder "+
			"owns and may reuse between nodes, so only the graph that declared it may touch it",
			op, b.desc.Label)
	}
	return nil
}

// checkRange is the half of [BufferView.check] that measures.
//
// It is separate because the graph builder needs the range check and not the
// rest: a transient is exactly the case check rejects, and a graph is the one
// caller allowed to touch one.
func (v BufferView) checkRange(op string) error {
	elem := v.DType.Size()
	if elem == 0 {
		return fmt.Errorf("accel: %s on %q: %v is not a dtype", op, v.Buffer.desc.Label, v.DType)
	}
	if v.Offset < 0 || v.Count < 0 || outOfRange(v.Offset, v.Count, elem, v.Buffer.bytes) {
		return fmt.Errorf("accel: %s on %q: elements [%d, %d) of %v are bytes [%d, %d), "+
			"outside the buffer's %d", op, v.Buffer.desc.Label, v.Offset, v.Offset+v.Count,
			v.DType, v.Offset*elem, (v.Offset+v.Count)*elem, v.Buffer.bytes)
	}
	return nil
}

// outOfRange reports whether count elements of elem bytes starting at element
// offset run past a buffer of the given byte length.
//
// The comparison stays in elements until it is known to be safe. Multiplying
// first is the bug: a large offset times an element size wraps, lands back
// inside the buffer, and turns a rejection into a silent write at the wrong
// address. Spec 001 section 7.3 promises a hand-constructed view's worst
// outcome is a rejection, and a wrapped offset is not that.
//
// offset and count must already be non-negative.
func outOfRange(offset, count, elem, bytes int) bool {
	limit := bytes / elem
	return offset > limit || count > limit-offset
}

// byteRange reports the view's extent relative to the start of its buffer.
func (v BufferView) byteRange() (offset, length int) {
	elem := v.DType.Size()
	return v.Offset * elem, v.Count * elem
}
