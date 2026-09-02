// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"errors"
	"testing"

	"golang.design/x/accel"
)

// Close and a concurrent allocation cannot both succeed.
//
// Pool.Close counted its children under the lock, released it, and only then
// marked the handle closed, so an AllocBuffer that passed its open check in
// the gap appended a live buffer to a pool that then closed. The check and
// the transition are one critical section now. Each iteration races the two
// and asserts the outcome is one of the two legal ones: the pool refused with
// a live child it still counts, or the allocation was refused as closed.
func TestPoolCloseAndAllocBufferCannotBothSucceed(t *testing.T) {
	d := openDevice(t)
	for i := 0; i < 500; i++ {
		p, err := d.NewPool(accel.PoolDescriptor{
			Kind: accel.MemoryDevice, Bytes: 1 << 16, Label: "raced",
		})
		if err != nil {
			t.Fatalf("pool: %v", err)
		}
		var (
			b        *accel.Buffer
			allocErr error
			done     = make(chan struct{})
		)
		go func() {
			defer close(done)
			b, allocErr = p.AllocBuffer(accel.BufferDescriptor{
				DType: accel.F32, Count: 4, Label: "b",
			})
		}()
		closeErr := p.Close()
		<-done

		var le *accel.LifetimeError
		switch {
		case allocErr == nil && closeErr == nil:
			t.Fatalf("iteration %d: the pool closed and handed out a buffer", i)
		case allocErr == nil:
			if !errors.As(closeErr, &le) || le.Reason != "has live children" {
				t.Fatalf("iteration %d: Close under a live buffer returned %v", i, closeErr)
			}
			if err := b.Close(); err != nil {
				t.Fatalf("iteration %d: buffer close: %v", i, err)
			}
			if err := p.Close(); err != nil {
				t.Fatalf("iteration %d: Close after the buffer went: %v", i, err)
			}
		default:
			if !errors.As(allocErr, &le) || le.Reason != "closed" {
				t.Fatalf("iteration %d: AllocBuffer on a closing pool returned %v", i, allocErr)
			}
			if closeErr != nil {
				t.Fatalf("iteration %d: Close with no children returned %v", i, closeErr)
			}
		}
	}
}

// Graph.Close and a concurrent submission never run a closed executable.
//
// Close checked inFlight under g.mu, released it, then marked the handle
// closed and closed the executable; run checked the handle before taking g.mu
// and set inFlight after. A submission that passed its check in the gap ran
// over an executable Close had just released, and the fence carried the
// backend's error rather than the lifetime one. Both sides now decide under
// g.mu. Each iteration races the two and asserts one of the legal outcomes:
// Close refused the graph as in flight, or the submission was refused as
// closed, or the submission finished before Close began.
//
// The submission is queued behind another one, and Close is called the
// moment that one's fence signals: the queue wakes the next unit at the same
// instant, which puts the two sides as close together as the scheduler
// allows.
func TestGraphCloseAndSubmitCannotRunAClosedGraph(t *testing.T) {
	d := openDevice(t)
	q := d.Queue()
	ahead, _ := buildInto(t, d, nil, 64)
	for i := 0; i < 200; i++ {
		g, _ := buildInto(t, d, nil, 64)
		first := q.Submit(ahead)
		f := q.Submit(g)
		if err := first.Wait(); err != nil {
			t.Fatalf("iteration %d: the submission ahead failed: %v", i, err)
		}
		closeErr := g.Close()
		ferr := f.Wait()

		var le *accel.LifetimeError
		if ferr != nil && (!errors.As(ferr, &le) || le.Reason != "closed") {
			t.Fatalf("iteration %d: the submission failed with %v, which is neither "+
				"success nor a closed graph", i, ferr)
		}
		if closeErr != nil {
			if !errors.As(closeErr, &le) || le.Reason != "in flight" {
				t.Fatalf("iteration %d: Close returned %v", i, closeErr)
			}
			if ferr != nil {
				t.Fatalf("iteration %d: Close refused an in-flight submission that then "+
					"failed as closed: %v", i, ferr)
			}
			if err := g.Close(); err != nil {
				t.Fatalf("iteration %d: Close after the submission: %v", i, err)
			}
		}
	}
}

// Device.Close and a concurrent NewPool cannot both succeed.
//
// Close counted its children, released every lock, and only then marked the
// handle closed; a NewPool that passed its open check in the gap registered a
// pool the count had not seen, the backend then refused to close over the live
// allocation, and the device was dead: closed to every caller and never
// closed. Registration and Close now share one lock, held across the count
// and the transition. Each iteration races the two and asserts one of the
// legal outcomes, and that a refusal leaves the device usable.
func TestDeviceCloseAndNewPoolCannotBothSucceed(t *testing.T) {
	for i := 0; i < 300; i++ {
		d, err := accel.OpenCPU(accel.CPUOptions{})
		if err != nil {
			t.Fatal(err)
		}
		var (
			p       *accel.Pool
			poolErr error
			done    = make(chan struct{})
		)
		go func() {
			defer close(done)
			p, poolErr = d.NewPool(accel.PoolDescriptor{
				Kind: accel.MemoryDevice, Bytes: 1 << 16, Label: "raced",
			})
		}()
		closeErr := d.Close()
		<-done

		var le *accel.LifetimeError
		switch {
		case poolErr == nil && closeErr == nil:
			t.Fatalf("iteration %d: the device closed and handed out a pool", i)
		case poolErr == nil:
			if !errors.As(closeErr, &le) || le.Reason != "has live children" {
				t.Fatalf("iteration %d: Close under a live pool returned %v", i, closeErr)
			}
			// The refusal left the device open and working.
			probe, err := d.NewPool(accel.PoolDescriptor{Kind: accel.MemoryUpload, Bytes: 1 << 16})
			if err != nil {
				t.Fatalf("iteration %d: the device stopped working after refusing to close: %v", i, err)
			}
			for _, c := range []*accel.Pool{probe, p} {
				if err := c.Close(); err != nil {
					t.Fatalf("iteration %d: pool close: %v", i, err)
				}
			}
			if err := d.Close(); err != nil {
				t.Fatalf("iteration %d: Close once the pools went: %v", i, err)
			}
		default:
			if !errors.As(poolErr, &le) || le.Reason != "closed" {
				t.Fatalf("iteration %d: NewPool on a closing device returned %v", i, poolErr)
			}
			if closeErr != nil {
				t.Fatalf("iteration %d: Close with no children returned %v", i, closeErr)
			}
		}
	}
}

// A device refuses to close over an open transient pool, and stays usable.
//
// The pool's allocation was counted among the implicit blocks, which Close
// does not count as children, so Close found nothing live, marked the handle
// dead, and then the backend refused over the live allocation: the device
// reported closed to every caller and never closed. An open pool is a child
// with a handle, exactly as an explicit pool is.
func TestDeviceCloseRefusesOverAnOpenTransientPool(t *testing.T) {
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatal(err)
	}
	pool, err := d.NewTransientPool("buckets")
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	// A graph built into it makes the pool allocate, and closing the graph
	// leaves that allocation with the pool, which is the point of sharing.
	r := d.NewRecorder()
	r.UseTransientPool(pool)
	mid := r.Transient(accel.BufferDescriptor{
		DType: accel.F32, Count: 64, Usage: accel.BufferStorage | accel.BufferCopyDst, Label: "mid",
	})
	r.UploadToBuffer(mid, make([]float32, 64))
	g, err := r.Build()
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if err := g.Close(); err != nil {
		t.Fatalf("graph close: %v", err)
	}
	if pool.Bytes() == 0 {
		t.Fatal("the pool holds nothing, so this test proves nothing")
	}

	err = d.Close()
	var le *accel.LifetimeError
	if !errors.As(err, &le) || le.Reason != "has live children" {
		t.Fatalf("Close over an open transient pool returned %v, want a live-children refusal", err)
	}
	if le.Children != 1 {
		t.Errorf("Children = %d, want 1 for the one pool", le.Children)
	}
	// The refusal left the device fully open.
	probe, err := d.NewPool(accel.PoolDescriptor{Kind: accel.MemoryUpload, Bytes: 1 << 16})
	if err != nil {
		t.Fatalf("the device stopped working after refusing to close: %v", err)
	}
	if err := probe.Close(); err != nil {
		t.Fatal(err)
	}
	if err := pool.Close(); err != nil {
		t.Fatalf("pool close: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("Close once the pool went: %v", err)
	}
}
