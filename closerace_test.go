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
