// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import "sync/atomic"

// resourceState is the lifetime machinery every device-backed resource carries.
//
// Resources have an explicit Close and are not finalizer-managed: device memory
// is scarce and driver-capped, and leaving its release to the collector means
// release happens at an unpredictable time under memory pressure the collector
// cannot see.
//
// The count starts at 1, which is the caller's handle, and gains one per
// outstanding hold: a queue's pending transfer batch at M1, and a submission's
// retain set once submissions exist. Reaching zero is what frees the memory,
// which happens either inside Close when nothing holds the resource or inside
// the completion path when something did. The count is never exposed and
// callers never manipulate it; the API is Close, and the count is what makes
// Close safe. See specs/001-device-resources.md section 7.1.
type resourceState struct {
	refs   atomic.Int32
	closed atomic.Bool
	label  string
}

func (r *resourceState) init(label string) {
	r.refs.Store(1)
	r.label = label
}

// retain adds a hold. It reports false if the resource has already reached
// zero, which cannot happen while a caller holds a live handle and is checked
// so that a bug in the holder is not a use-after-free.
func (r *resourceState) retain() bool {
	for {
		n := r.refs.Load()
		if n <= 0 {
			return false
		}
		if r.refs.CompareAndSwap(n, n+1) {
			return true
		}
	}
}

// release drops a hold and reports whether this was the last one, in which case
// the caller frees the underlying memory.
func (r *resourceState) release() bool { return r.refs.Add(-1) == 0 }

// isClosed reports whether the caller has called Close. Every entry point
// checks this before touching device state, which is one atomic load on paths
// that already do more work than that, and it is what turns a use-after-close
// into a named error rather than undefined behaviour.
func (r *resourceState) isClosed() bool { return r.closed.Load() }

// checkOpen is the guard every entry point starts with.
func (r *resourceState) checkOpen(op string) error {
	if r.isClosed() {
		return &LifetimeError{Op: op, Resource: r.label, Reason: reasonClosed}
	}
	return nil
}

// beginClose marks the handle dead and reports whether this call is the one
// that did it, so a second Close is a no-op rather than a double release.
func (r *resourceState) beginClose() bool { return r.closed.CompareAndSwap(false, true) }

// holds reports how many holders beyond the caller's handle remain. It is only
// meaningful after beginClose has succeeded and the caller's reference has been
// dropped.
func (r *resourceState) holds() int { return int(r.refs.Load()) }
