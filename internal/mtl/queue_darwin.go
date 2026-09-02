// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl

import (
	"fmt"
	"runtime"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/ebitengine/purego/objc"
)

// Queue is an MTLCommandQueue, retained.
type Queue struct{ id objc.ID }

// CommandBuffer is one submission.
//
// A MTLCommandBuffer is single-submit, which specs/006-backends.md section 2.2
// names as Metal's one real "cannot": there is no reusable command object
// without indirect command buffers, so a graph is re-encoded per submission
// and this type is created and destroyed per submission with it.
type CommandBuffer struct{ id objc.ID }

// ComputeEncoder encodes compute work into a command buffer.
type ComputeEncoder struct {
	id objc.ID

	// pin holds the byte slice SetBytes is passing while the call runs. See
	// SetBytes: it is reused across calls so that the fast message send costs
	// no allocation, and a Pinner must not be copied, which is why every method
	// here takes a pointer receiver.
	pin runtime.Pinner
}

// BlitEncoder encodes copies into a command buffer.
type BlitEncoder struct{ id objc.ID }

// Status values from MTLCommandBufferStatus.
const (
	StatusCompleted = 4
	StatusError     = 5
)

var (
	selNewCommandQueue       = objc.RegisterName("newCommandQueue")
	selCommandBuffer         = objc.RegisterName("commandBuffer")
	selComputeCommandEncoder = objc.RegisterName("computeCommandEncoder")
	selBlitCommandEncoder    = objc.RegisterName("blitCommandEncoder")
	selSetComputePipeline    = objc.RegisterName("setComputePipelineState:")
	selSetBuffer             = objc.RegisterName("setBuffer:offset:atIndex:")
	selSetBytes              = objc.RegisterName("setBytes:length:atIndex:")
	selDispatchThreadgroups  = objc.RegisterName("dispatchThreadgroups:threadsPerThreadgroup:")
	selDispatchIndirect      = objc.RegisterName(
		"dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:")
	selEndEncoding        = objc.RegisterName("endEncoding")
	selCommit             = objc.RegisterName("commit")
	selWaitUntilCompleted = objc.RegisterName("waitUntilCompleted")
	selStatus             = objc.RegisterName("status")
	selError              = objc.RegisterName("error")
	selCopyFromBuffer     = objc.RegisterName("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:")
)

// NewQueue makes a command queue, +1 from a new* selector.
func (d *Device) NewQueue() *Queue {
	q := &Queue{}
	withPool(func() { q.id = send(d.id, selNewCommandQueue) })
	return q
}

// Close releases the queue.
func (q *Queue) Close() {
	release(q.id)
	q.id = 0
}

// liveCommandBuffers counts the command buffers this package holds a retain
// on: incremented by [Queue.Begin] and decremented by [CommandBuffer.Close].
//
// A test hook. A retained command buffer that is never closed is a leak Metal
// does not report and Go's heap does not show, because the object lives on the
// Objective-C side; the only way to see one is to count what was retained
// against what was released.
var liveCommandBuffers atomic.Int64

// LiveCommandBuffers reports how many command buffers are retained and not yet
// closed. It exists for tests that check a submission path releases what it
// began.
func LiveCommandBuffers() int64 { return liveCommandBuffers.Load() }

// Begin starts a submission.
//
// -commandBuffer returns an autoreleased object, so this retains it: the buffer
// outlives the pool that produced it by construction, since a caller commits it
// and then waits on it. Releasing it is [CommandBuffer.Close].
func (q *Queue) Begin() *CommandBuffer {
	cb := &CommandBuffer{}
	withPool(func() { cb.id = retain(send(q.id, selCommandBuffer)) })
	if cb.id != 0 {
		liveCommandBuffers.Add(1)
	}
	return cb
}

// Compute begins a compute pass. Encoders are autoreleased and are retained for
// the same reason command buffers are.
func (cb *CommandBuffer) Compute() *ComputeEncoder {
	e := &ComputeEncoder{}
	withPool(func() { e.id = retain(send(cb.id, selComputeCommandEncoder)) })
	return e
}

// Blit begins a copy pass.
func (cb *CommandBuffer) Blit() *BlitEncoder {
	e := &BlitEncoder{}
	withPool(func() { e.id = retain(send(cb.id, selBlitCommandEncoder)) })
	return e
}

// SetPipeline selects the kernel this pass runs.
func (e *ComputeEncoder) SetPipeline(p *Pipeline) {
	send(e.id, selSetComputePipeline, uintptr(p.id))
}

// SetBuffer binds a range of a buffer at an argument index.
func (e *ComputeEncoder) SetBuffer(b *Buffer, offset, index int) {
	send(e.id, selSetBuffer, uintptr(b.id), uintptr(offset), uintptr(index))
}

// SetBytes binds a small value inline, without a buffer.
//
// Metal copies the bytes at encode time, so the caller's storage is free
// afterwards. This is how generated slice lengths and small uniform blocks are
// bound: allocating a buffer for sixteen bytes per submission would cost more
// than the dispatch.
func (e *ComputeEncoder) SetBytes(data []byte, index int) {
	if len(data) == 0 {
		return
	}
	// Pinned, then sent the fast way. A uintptr is not a reference the
	// collector honours, and nothing here proves the slice is on the heap
	// rather than on a stack that may grow and move under the call -- so the
	// address has to be made stable rather than assumed to be.
	//
	// runtime.Pinner is what makes it stable: Pin prevents the object being
	// moved or freed until Unpin, which covers both halves of the problem and
	// keeps it alive across the call without a KeepAlive.
	//
	// This was the reflected form, and it was the only reflected call left on
	// the per-node path -- about 3.7x the cost of a direct send, and the
	// boxing of its arguments was most of the allocation a submission did.
	// specs/006-backends.md section 4.3.
	e.pin.Pin(&data[0])
	send(e.id, selSetBytes, uintptr(unsafe.Pointer(&data[0])), uintptr(len(data)),
		uintptr(index))
	e.pin.Unpin()
}

// Dispatch launches a grid of threadgroups.
//
// Both arguments are MTLSize passed by value. This is the call
// specs/021-metal-bringup.md section 2 singles out: passing a pointer here
// compiles, runs, and launches a grid nobody asked for, so the test that covers
// it checks the ids the kernel saw rather than that the call returned.
func (e *ComputeEncoder) Dispatch(groups, threadsPerGroup Size) {
	e.id.Send(selDispatchThreadgroups, groups, threadsPerGroup)
}

// DispatchIndirect launches a grid whose count the device wrote.
//
// The buffer holds three uint32 threadgroup counts. Metal reads them on the
// GPU, so nothing here knows what the grid will be -- which is the point, and
// why specs/003-command-graph.md requires the count to have been clamped before
// this is reached rather than after.
func (e *ComputeEncoder) DispatchIndirect(count *Buffer, offset int, threadsPerGroup Size) {
	withPool(func() {
		e.id.Send(selDispatchIndirect, count.id, uintptr(offset), threadsPerGroup)
	})
}

// End closes the pass and releases the encoder.
func (e *ComputeEncoder) End() {
	send(e.id, selEndEncoding)
	release(e.id)
	e.id = 0
}

// Copy encodes a device-to-device copy.
func (e *BlitEncoder) Copy(dst *Buffer, dstOff int, src *Buffer, srcOff, size int) {
	withPool(func() {
		send(e.id, selCopyFromBuffer, uintptr(src.id), uintptr(srcOff), uintptr(dst.id), uintptr(dstOff), uintptr(size))
	})
}

// End closes the pass and releases the encoder.
func (e *BlitEncoder) End() {
	send(e.id, selEndEncoding)
	release(e.id)
	e.id = 0
}

// Commit submits the buffer and returns without waiting.
func (cb *CommandBuffer) Commit() {
	send(cb.id, selCommit)
}

// Wait blocks until the submission completes.
func (cb *CommandBuffer) Wait() {
	send(cb.id, selWaitUntilCompleted)
}

// Done reports without blocking whether the submission has finished, either by
// completing or by failing.
func (cb *CommandBuffer) Done() bool {
	s := int(send(cb.id, selStatus))
	return s == StatusCompleted || s == StatusError
}

// CommandBufferErrorDomain is the NSError domain of a failed submission.
const CommandBufferErrorDomain = "MTLCommandBufferErrorDomain"

// The MTLCommandBufferError codes, from Metal/MTLCommandBuffer.h.
//
// Written out because the code is what says whether the *device* failed or
// the *work* did, and that distinction is one a backend has to make: a
// timeout or a page fault is a kernel that misbehaved and leaves the device
// usable, while a removed device or revoked access is a device nothing will
// run on again. The message text says the same thing less reliably.
const (
	CommandBufferErrorNone            = 0
	CommandBufferErrorInternal        = 1
	CommandBufferErrorTimeout         = 2
	CommandBufferErrorPageFault       = 3
	CommandBufferErrorAccessRevoked   = 4
	CommandBufferErrorNotPermitted    = 7
	CommandBufferErrorOutOfMemory     = 8
	CommandBufferErrorInvalidResource = 9
	CommandBufferErrorMemoryless      = 10
	CommandBufferErrorDeviceRemoved   = 11
	CommandBufferErrorStackOverflow   = 12
)

// CommandBufferError is what a failed submission reported.
//
// It carries the NSError's domain and code beside its text, so a caller
// classifies by the code Metal assigned and not by matching the description,
// which is localized and has changed between releases.
type CommandBufferError struct {
	Domain  string
	Code    int
	Message string
}

func (e *CommandBufferError) Error() string {
	return fmt.Sprintf("accel/mtl: the submission failed: %s (%s code %d)",
		e.Message, e.Domain, e.Code)
}

// Err reports why the submission failed, or nil.
func (cb *CommandBuffer) Err() error {
	var err error
	withPool(func() {
		e := cb.id.Send(selError)
		if e == 0 {
			return
		}
		err = &CommandBufferError{
			Domain:  utf8(e.Send(selDomain)),
			Code:    int(e.Send(selCode)),
			Message: utf8(e.Send(selLocalizedDescription)),
		}
	})
	return err
}

// Close releases the command buffer. Closing twice releases once.
func (cb *CommandBuffer) Close() {
	if cb.id == 0 {
		return
	}
	release(cb.id)
	cb.id = 0
	liveCommandBuffers.Add(-1)
}

var (
	selGPUStartTime = objc.RegisterName("GPUStartTime")
	selGPUEndTime   = objc.RegisterName("GPUEndTime")
)

// GPUTime reports how long the device spent on this command buffer.
//
// Valid only after completion, and zero when the driver did not record it.
//
// This is device time, which is the number a caller measuring throughput wants:
// the wall clock around Commit and Wait includes queueing, driver work and
// whatever else the process was doing, and reporting that as GPU time answers
// the wrong question convincingly. Metal gives both timestamps for free on a
// completed buffer, so no timestamp pool is needed for the whole-submission
// figure.
func (cb *CommandBuffer) GPUTime() time.Duration {
	var start, end float64
	withPool(func() {
		start = objc.Send[float64](cb.id, selGPUStartTime)
		end = objc.Send[float64](cb.id, selGPUEndTime)
	})
	// Seconds as a CFTimeInterval. A non-positive span means the driver did not
	// record one, which is reported as zero rather than as a negative duration
	// a caller would have to know to discard.
	if end <= start {
		return 0
	}
	return time.Duration((end - start) * float64(time.Second))
}
