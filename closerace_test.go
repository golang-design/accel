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
