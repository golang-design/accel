// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
)

// TestClosedReceiverIsReportedEverywhere is spec 001 section 11.6's
// table-driven case, and the table is the point: it enumerates *every* entry
// point that takes a closed receiver, so a new one that forgets the check is
// caught by adding a row rather than by somebody remembering.
//
// Split across one test per type it does not do that job, because the next
// entry point lands in whichever file its author was already editing.
func TestClosedReceiverIsReportedEverywhere(t *testing.T) {
	// Each case closes exactly the receiver it is about, so a check that passes
	// only because something else was closed does not count.
	cases := []struct {
		name  string
		close func(*fixtureT) error
		call  func(*fixtureT) error
	}{
		{"Device.NewPool", closeDevice, func(f *fixtureT) error {
			_, err := f.dev.NewPool(accel.PoolDescriptor{Kind: accel.MemoryDevice, Bytes: 1 << 20})
			return err
		}},
		{"Device.NewPoolWith", closeDevice, func(f *fixtureT) error {
			_, err := f.dev.NewPool(accel.PoolDescriptor{Kind: accel.MemoryDevice, Bytes: 1 << 20})
			return err
		}},
		{"Device.NewBuffer", closeDevice, func(f *fixtureT) error {
			_, err := f.dev.NewBuffer(accel.BufferDescriptor{DType: accel.F32, Count: 4})
			return err
		}},
		{"Pool.Alloc", closePool, func(f *fixtureT) error {
			_, err := f.pool.AllocBuffer(accel.BufferDescriptor{DType: accel.F32, Count: 4})
			return err
		}},
		{"Pool.Reset", closePool, func(f *fixtureT) error { return f.pool.Reset() }},
		{"Buffer.View", closeBuffer, func(f *fixtureT) error {
			_, err := f.buffer.View(0, 1)
			return err
		}},
		{"Buffer.ViewAs", closeBuffer, func(f *fixtureT) error {
			_, err := f.buffer.ViewAs(accel.U32, 0, 1)
			return err
		}},
		{"Queue.WriteBuffer", closeBuffer, func(f *fixtureT) error {
			return f.queue.WriteBuffer(f.buffer, 0, []float32{1})
		}},
		{"Queue.ReadBuffer", closeBuffer, func(f *fixtureT) error {
			return f.queue.ReadBuffer(f.buffer, 0, make([]float32, 1))
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := newFixture(t)
			if err := tc.close(f); err != nil {
				t.Fatalf("closing the receiver: %v", err)
			}
			err := tc.call(f)
			var le *accel.LifetimeError
			if !errors.As(err, &le) {
				t.Fatalf("on a closed receiver returned %v, want *LifetimeError", err)
			}
			if le.Reason != "closed" {
				t.Errorf("reason is %q, want \"closed\"", le.Reason)
			}
			if !errors.Is(err, accel.ErrLifetime) {
				t.Error("the error does not unwrap to ErrLifetime")
			}
		})
	}
}

// fixtureT is one device with one pool and one buffer, so a case can close
// exactly the receiver it is about.
type fixtureT struct {
	dev    *accel.Device
	pool   *accel.Pool
	buffer *accel.Buffer
	queue  *accel.Queue
}

func closeDevice(f *fixtureT) error {
	// The device refuses while its pool is live, which is the rule under test
	// elsewhere; close the children first so this case is about the device.
	if err := f.buffer.Close(); err != nil {
		return err
	}
	if err := f.pool.Close(); err != nil {
		return err
	}
	return f.dev.Close()
}

func closePool(f *fixtureT) error {
	if err := f.buffer.Close(); err != nil {
		return err
	}
	return f.pool.Close()
}

func closeBuffer(f *fixtureT) error { return f.buffer.Close() }

func newFixture(t *testing.T) *fixtureT {
	t.Helper()
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatal(err)
	}
	p, err := d.NewPool(accel.PoolDescriptor{Kind: accel.MemoryDevice, Bytes: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	b, err := p.AllocBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: 16,
		Usage: accel.BufferCopyDst | accel.BufferCopySrc, Label: "subject",
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		b.Close()
		p.Close()
		d.Close()
	})
	return &fixtureT{dev: d, pool: p, buffer: b, queue: d.Queue()}
}

// TestOffsetsCannotWrapIntoTheBuffer covers the arithmetic that turns a
// rejection into a silent write.
//
// An element offset is scaled by the element size to reach a byte offset. Doing
// that before bounding the offset lets a large value wrap, land back inside the
// buffer, and address element zero. Spec 001 section 7.3 promises a
// hand-constructed view's worst outcome is a rejection, so this has to be
// rejected at every entry point that takes an element offset.
func TestOffsetsCannotWrapIntoTheBuffer(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	q := d.Queue()
	p := newPool(t, d, accel.MemoryDevice, 1<<20)
	defer p.Close()

	b := alloc(t, p, accel.BufferDescriptor{
		DType: accel.F32, Count: 16,
		Usage: accel.BufferCopyDst | accel.BufferCopySrc, Label: "wrappable",
	})
	defer b.Close()

	// Fill it, so a wrapped write would be visible as changed data rather than
	// as a value that happened to already be there.
	base := make([]float32, 16)
	for i := range base {
		base[i] = float32(i) + 100
	}
	if err := q.WriteBuffer(b, 0, base); err != nil {
		t.Fatal(err)
	}

	// Each offset is a multiple of a power of two large enough that multiplying
	// by 4 wraps to a small non-negative number.
	wrapping := []int{
		math.MaxInt/4 + 1,
		math.MaxInt/2 + 1,
		1 << 62,
		(1 << 62) + 4,
	}
	for _, offset := range wrapping {
		if err := q.WriteBuffer(b, offset, []float32{-1}); err == nil {
			t.Errorf("WriteBuffer at element %d was accepted", offset)
		}
		if err := q.ReadBuffer(b, offset, make([]float32, 1)); err == nil {
			t.Errorf("ReadBuffer at element %d was accepted", offset)
		}
		if _, err := b.View(offset, 1); err == nil {
			t.Errorf("View at element %d was accepted", offset)
		}
		if _, err := b.ViewAs(accel.U32, offset, 1); err == nil {
			t.Errorf("ViewAs at element %d was accepted", offset)
		}
		// A count that wraps is the same bug from the other side.
		if _, err := b.View(0, offset); err == nil {
			t.Errorf("View of %d elements was accepted", offset)
		}
	}

	got := make([]float32, 16)
	if err := q.ReadBuffer(b, 0, got); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if got[i] != base[i] {
			t.Fatalf("element %d is %v, want %v: a wrapped offset wrote inside the buffer",
				i, got[i], base[i])
		}
	}
}

// TestDeviceCloseCountsImplicitChildren is the same rule as the explicit case,
// for the pool the caller was never handed.
//
// A buffer from NewBuffer has a handle, so it is a live child. Counting only
// the explicit pools let Close decide it could proceed, mark the device closed,
// and only then meet a pool that refused, leaving a device that reported closed
// and never closed.
func TestDeviceCloseCountsImplicitChildren(t *testing.T) {
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatal(err)
	}

	b, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: 64, Usage: accel.BufferStorage, Label: "convenience",
	})
	if err != nil {
		t.Fatal(err)
	}

	err = d.Close()
	if err == nil {
		t.Fatal("a device with a live implicit buffer closed")
	}
	var le *accel.LifetimeError
	if !errors.As(err, &le) {
		t.Fatalf("error is %T, want *LifetimeError", err)
	}
	if le.Children != 1 {
		t.Errorf("Children = %d, want 1", le.Children)
	}

	// The refusal left the device fully open, so it still allocates.
	probe, err := d.NewPool(accel.PoolDescriptor{Kind: accel.MemoryUpload, Bytes: 1 << 20})
	if err != nil {
		t.Fatalf("the device stopped working after refusing to close: %v", err)
	}
	if err := probe.Close(); err != nil {
		t.Fatal(err)
	}
	// And a second Close still refuses rather than reporting the success an
	// already-marked handle would.
	if err := d.Close(); err == nil {
		t.Error("a second Close reported success for a device that never closed")
	}

	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("Close once the implicit buffer went away: %v", err)
	}
}

// Closing the *Buffer behind a graph transient's view is refused, not fatal.
//
// Recorder.Transient hands back a BufferView whose Buffer field is exported, so
// a caller can reach the buffer and call Close on it — and a transient's memory
// belongs to the builder, so it has no pool to return to. Before this was
// guarded, free dereferenced the nil pool and killed the process, which is
// exactly what BufferView.check's own doc says cannot happen: "The worst outcome
// is a rejection."
//
// The public-surface review of specs/036-documentation.md found it by asking
// what a tutorial would have to apologise for.
func TestClosingATransientsBufferIsRefusedRatherThanFatal(t *testing.T) {
	d := openDevice(t)
	r := d.NewRecorder()
	v := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: 64,
		Usage: accel.BufferStorage | accel.BufferCopyDst, Label: "mid",
	})

	err := v.Buffer.Close()
	if err == nil {
		t.Fatal("closing a graph transient's buffer was accepted; its memory belongs to " +
			"the builder and there is no pool to return it to")
	}
	if !errors.Is(err, accel.ErrLifetime) {
		t.Errorf("want an ErrLifetime, got %T: %v", err, err)
	}
	if !strings.Contains(err.Error(), "graph transient") {
		t.Errorf("the refusal should say what the resource is: %v", err)
	}

	// And the transient still works afterwards: a refusal changes nothing.
	if err := v.Buffer.Close(); err == nil {
		t.Error("the second refusal succeeded, so the first one had a side effect")
	}
}
