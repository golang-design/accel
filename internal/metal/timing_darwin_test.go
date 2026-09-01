// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal

import (
	"testing"
	"time"

	"golang.design/x/accel/internal/driver"
)

// The backend honours the timing opt-in itself, rather than relying on the
// layer above to ask only when it should.
//
// The public path gates before calling, so these two returns are unreachable
// through it. They are the driver.Timer contract — "zero when the submission
// did not ask for timing" — and a backend that only worked because its caller
// was careful would break the first time a different caller appeared.
func TestElapsedHonoursTheOptInAndTheAbsentSubmission(t *testing.T) {
	untimed := &executable{plan: &driver.Plan{}}
	if got := untimed.Elapsed(); got != 0 {
		t.Errorf("a plan that did not ask for timings reported %v", got)
	}

	// Asked for, but nothing submitted yet: there is no command buffer to read
	// a timestamp from, and zero is the honest answer rather than a panic.
	never := &executable{plan: &driver.Plan{CollectTimings: true}}
	if got := never.Elapsed(); got != 0 {
		t.Errorf("an executable with no submission reported %v", got)
	}
}

// A fence whose command buffer is gone answers from what close cached, rather
// than messaging a released object.
//
// The timing is read from the command buffer, which the executable owns, so a
// caller holding a fence past Close would otherwise send a message to freed
// memory -- a crash inside objc_msgSend with a stack pointing nowhere useful,
// which is the failure mode internal/mtl's ownership rule exists to prevent.
func TestGPUTimeAfterTheCommandBufferIsGone(t *testing.T) {
	if got := (&fence{}).gpuTime(); got != 0 {
		t.Errorf("a fence with no command buffer reported %v", got)
	}
	if got := (&fence{gpu: 3 * time.Millisecond}).gpuTime(); got != 3*time.Millisecond {
		t.Errorf("a closed fence reported %v, not what close cached", got)
	}
}
