// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"fmt"
	"sync"
	"unsafe"
)

// hostIsLittleEndian is checked once at startup.
//
// accel requires the host and the device to share byte order, and every
// supported platform is little-endian. That is what makes Queue.WriteBuffer a
// memory copy and never a byte swap, ViewAs a reinterpretation and never a
// conversion, and a kernel reading u32 see exactly the bytes the host wrote.
// See specs/001-device-resources.md section 3.5.
var hostIsLittleEndian = func() bool {
	x := uint16(1)
	return *(*byte)(unsafe.Pointer(&x)) == 1
}()

// pendingWrite is one staged transfer waiting for a flush.
//
// Only a write to memory the device does not map needs staging. A write to a
// host-visible kind lands in the mapping immediately, which is the whole point
// of MemoryShared on unified hardware: there is no staging block and no copy.
type pendingWrite struct {
	dst    *Buffer
	offset int // bytes from the start of the buffer
	data   []byte
}

// WriteBuffer copies data into queue-owned staging and appends the transfer to
// this queue's next submission prologue. It returns once data no longer aliases
// the caller's value, not when the device has consumed it.
//
// offset is in elements of the buffer's dtype and the write touches only the
// range it names. data must be a slice of the buffer's dtype: []float32 for f32,
// []uint16 for the 16-bit float storage types, []int32, []uint32, []int8, or
// []byte.
//
// The caller may reuse or modify their slice the moment this returns, which is
// what "asynchronous" has to mean for it to be safe. Every write issued before a
// flush is visible to that flush, so waiting on the returned fence proves the
// bytes landed.
func (q *Queue) WriteBuffer(dst *Buffer, offset int, data any) error {
	src, err := hostBytes("WriteBuffer", dst.desc.Label, dst.desc.DType, data)
	if err != nil {
		return err
	}
	byteOffset, err := q.checkRange("WriteBuffer", dst, offset, len(src))
	if err != nil {
		return err
	}

	if mapped := dst.mapping(); mapped != nil {
		// Host-visible memory takes the bytes now: one copy, straight from the
		// caller's slice into the mapping, and that is all. Staging first would
		// move the payload twice for memory that needs no staging, which is the
		// exact cost MemoryShared exists to remove. Nothing is counted as staged
		// because nothing was.
		//
		// This still satisfies the promise that the caller may reuse their slice
		// the moment this returns: the bytes are in the mapping, and nothing
		// holds a reference to the slice.
		//
		// The ordering obligation does not vanish with the copy. A shared buffer
		// written while the device reads it is a race with no copy in between to
		// hide it, and accel does not detect that.
		copy(mapped[byteOffset:], src)
		return nil
	}

	// Otherwise the bytes have to wait for a flush, so they are copied out of the
	// caller's slice now. Holding the slice until then would make what lands
	// depend on when the caller last touched it, which is the same reason a
	// recorded host write copies at record time.
	staged := make([]byte, len(src))
	copy(staged, src)

	q.mu.Lock()
	q.stats.BytesStaged += int64(len(staged))
	q.mu.Unlock()

	// Otherwise the destination is memory the device does not map, so the bytes
	// wait for a flush. The destination is retained until then: closing it now
	// retires the caller's handle and reports, and the queued write still lands.
	if !dst.state.retain() {
		return &LifetimeError{Op: "WriteBuffer", Resource: dst.desc.Label, Reason: reasonClosed}
	}
	q.mu.Lock()
	q.pending = append(q.pending, pendingWrite{dst: dst, offset: byteOffset, data: staged})
	q.mu.Unlock()
	return nil
}

// ReadBuffer flushes this queue's pending writes, waits for prior work on the
// queue, and copies a buffer range into host memory.
//
// It blocks, which is what makes it wrong in a hot loop and right for reading
// final results. offset is in elements of the buffer's dtype and into must be a
// slice of that dtype.
//
// A read orders only the queue it is called on. If another queue owns a pending
// write to the same resource, flush that queue first: an immediate read never
// silently searches or drains unrelated queues.
func (q *Queue) ReadBuffer(src *Buffer, offset int, into any) error {
	dst, err := hostBytes("ReadBuffer", src.desc.Label, src.desc.DType, into)
	if err != nil {
		return err
	}
	byteOffset, err := q.checkRange("ReadBuffer", src, offset, len(dst))
	if err != nil {
		return err
	}

	// A pending write the caller has not flushed would otherwise be invisible
	// here, so the read flushes first. Without this a write followed by a read
	// with no submission in between would return stale bytes.
	if err := q.Flush().Wait(); err != nil {
		return err
	}

	q.mu.Lock()
	q.stats.ImmediateReads++
	q.mu.Unlock()

	if mapped := src.mapping(); mapped != nil {
		copy(dst, mapped[byteOffset:])
		return nil
	}
	return src.pool.block.Read(src.alloc.Offset+byteOffset, dst)
}

// Flush submits this queue's pending immediate writes without a graph. When no
// writes are pending it returns an already-signalled fence.
//
// It exists for the caller who wants to flush without reading. Without it a
// caller who wrote and then expected the bytes to be there some other way would
// wait forever, because nothing else forces the batch out.
func (q *Queue) Flush() *Fence {
	q.mu.Lock()
	batch := q.pending
	q.pending = nil
	if len(batch) == 0 && settled(q.prev) {
		// Nothing to flush and nothing ahead of it, so the fence's promise --
		// everything enqueued on this queue up to here is complete -- already
		// holds. Going through the stream would only add a scheduling hop.
		q.mu.Unlock()
		f := newFence()
		f.signal()
		return f
	}
	if len(batch) > 0 {
		q.stats.Submissions++
	}
	q.mu.Unlock()

	return q.enqueue(func() error { return runWrites(batch) })
}

// runWrites performs one staged batch.
func runWrites(batch []pendingWrite) error {
	var first error
	for _, w := range batch {
		// The destination is retained, so it is still valid even if the caller
		// closed their handle in the meantime.
		if err := w.dst.pool.block.Write(w.dst.alloc.Offset+w.offset, w.data); err != nil && first == nil {
			first = err
		}
		w.dst.release()
	}
	return first
}

// settled reports whether a queue's previous unit of work has finished.
func settled(prev chan struct{}) bool {
	if prev == nil {
		return true
	}
	select {
	case <-prev:
		return true
	default:
		return false
	}
}

// enqueue puts one unit of work on this queue's serial stream and returns a
// fence for it, without blocking.
//
// # Why a queue is a stream and not a set of goroutines
//
// specs/003-command-graph.md requires that two submissions to one queue are
// fully ordered: the second begins no earlier than the first ends, and every
// write by the first is visible to the second with no caller action. That is a
// deliberate cost -- the alternative, starting submissions in order and leaving
// memory hazards to the caller, is what the underlying APIs provide and is
// faster -- and it is rejected because its failure mode is a race that appears
// under load on one backend.
//
// The mechanism is a chain rather than a worker goroutine: each unit captures
// the previous unit's completion channel under the queue lock, so the order
// units run in is exactly the order Submit and Flush were called in, and a
// caller that never waits still gets ordering.
func (q *Queue) enqueue(work func() error) *Fence {
	f := newFence()
	done := make(chan struct{})

	q.mu.Lock()
	prev := q.prev
	q.prev = done
	q.mu.Unlock()

	go func() {
		defer close(done)
		if prev != nil {
			<-prev
		}
		f.state.err = work()
		f.signal()
	}()
	return f
}

// Stats reports cumulative queue counters since device open. They are counters,
// not a profiler: nothing here is per node and nothing here costs a readback.
func (q *Queue) Stats() QueueStats {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.stats
}

// pendingCount reports how many writes are waiting for a flush, which
// Device.Close reports as live children rather than hiding.
func (q *Queue) pendingCount() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.pending)
}

// checkRange validates a transfer endpoint and returns its byte offset.
func (q *Queue) checkRange(op string, b *Buffer, offset, length int) (int, error) {
	if err := b.state.checkOpen(op); err != nil {
		return 0, err
	}
	if offset < 0 {
		return 0, fmt.Errorf("accel: %s %q: element offset %d is negative", op, b.desc.Label, offset)
	}
	elem := b.desc.DType.Size()
	// The offset is bounded in elements before it is scaled to bytes. Scaling an
	// unbounded one wraps, lands back inside the buffer, and writes at an address
	// nobody asked for.
	if outOfRange(offset, (length+elem-1)/elem, elem, b.bytes) {
		return 0, fmt.Errorf("accel: %s %q: elements [%d, %d) of %v are outside the buffer's "+
			"%d elements", op, b.desc.Label, offset, offset+(length+elem-1)/elem, b.desc.DType, b.desc.Count)
	}
	byteOffset := offset * elem
	if byteOffset+length > b.bytes {
		return 0, fmt.Errorf("accel: %s %q: elements [%d, %d) of %v are bytes [%d, %d), "+
			"outside the buffer's %d", op, b.desc.Label, offset,
			offset+length/elem, b.desc.DType, byteOffset, byteOffset+length, b.bytes)
	}
	return byteOffset, nil
}

// mapping returns the buffer's host-visible bytes, or nil when the device does
// not map that memory.
//
// The backend is the authority here, not this package. A Metal private buffer
// has no host pointer, and the CPU backend reports MemoryDevice unmappable even
// though the memory physically could be mapped, so that the one backend able to
// enforce the rule does. See specs/006-backends.md section 1.
func (b *Buffer) mapping() []byte {
	blk := b.pool.block.Bytes()
	if blk == nil {
		return nil
	}
	return blk[b.alloc.Offset : b.alloc.Offset+b.alloc.Size]
}

// hostBytes reinterprets a caller's typed slice as bytes, rejecting one whose
// element type does not match the buffer's dtype.
//
// The reinterpretation is exact rather than a conversion, which is only
// meaningful because accel fixes the byte order. A mismatched slice is rejected
// naming both types, because silently accepting one would write the right number
// of bytes with the wrong meaning.
// hostBytes reinterprets a caller's typed slice as the bytes a device holds.
//
// It takes a dtype and a label rather than a *Buffer because a view may be at a
// different dtype than the buffer it names, and a graph slot has a dtype and no
// buffer at all. Matching against the buffer's own dtype would reject a legal
// reinterpreting view and would have nothing to match a slot against.
func hostBytes(op, label string, dt DType, data any) ([]byte, error) {
	if !hostIsLittleEndian {
		return nil, fmt.Errorf("accel: %s: this architecture is big-endian, which accel does "+
			"not support: host and device must share byte order (spec 001 section 3.5)", op)
	}

	mismatch := func(got string) error {
		return fmt.Errorf("accel: %s %q: buffer is %v, so the host slice must be %s, not %s",
			op, label, dt, goSliceFor(dt), got)
	}

	switch v := data.(type) {
	case []byte:
		if dt != U8 {
			return nil, mismatch("[]byte")
		}
		return v, nil
	case []int8:
		if dt != I8 {
			return nil, mismatch("[]int8")
		}
		return asBytes(v), nil
	case []uint16:
		if dt != F16 && dt != BF16 {
			return nil, mismatch("[]uint16")
		}
		return asBytes(v), nil
	case []int32:
		if dt != I32 {
			return nil, mismatch("[]int32")
		}
		return asBytes(v), nil
	case []uint32:
		if dt != U32 {
			return nil, mismatch("[]uint32")
		}
		return asBytes(v), nil
	case []float32:
		if dt != F32 {
			return nil, mismatch("[]float32")
		}
		return asBytes(v), nil
	case nil:
		return nil, mismatch("nil")
	default:
		return nil, mismatch(fmt.Sprintf("%T", data))
	}
}

// asBytes views a slice of a fixed-width scalar as the bytes underneath it. No
// copy and no conversion: the device sees exactly these bytes.
func asBytes[T int8 | uint16 | int32 | uint32 | float32](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	var zero T
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), len(s)*int(unsafe.Sizeof(zero)))
}

// goSliceFor names the Go slice type a dtype expects, for the rejection message.
func goSliceFor(d DType) string {
	switch d {
	case F32:
		return "[]float32"
	case F16, BF16:
		return "[]uint16"
	case I32:
		return "[]int32"
	case U32:
		return "[]uint32"
	case I8:
		return "[]int8"
	case U8:
		return "[]byte"
	}
	return "a slice of the buffer's dtype"
}

// fenceState is what a Fence points at, so a Fence stays a small handle that a
// caller can hold and select on while the completion path writes here.
type fenceState struct {
	once sync.Once
	done chan struct{}
	err  error
}

func newFence() *Fence {
	return &Fence{state: &fenceState{done: make(chan struct{})}}
}

func (f *Fence) signal() { f.state.once.Do(func() { close(f.state.done) }) }

// Wait blocks until the submission completes, reporting its error if it failed.
func (f *Fence) Wait() error {
	<-f.state.done
	return f.state.err
}

// Done reports whether the submission has completed, without blocking.
func (f *Fence) Done() bool {
	select {
	case <-f.state.done:
		return true
	default:
		return false
	}
}

// C returns a channel closed when the submission completes, for selecting on it.
func (f *Fence) C() <-chan struct{} { return f.state.done }
