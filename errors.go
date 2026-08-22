// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"errors"
	"fmt"
	"strings"

	"golang.design/x/accel/internal/driver"
)

// Sentinels for [errors.Is]. Each typed error in this package unwraps to one of
// these, so a caller can branch on the class without depending on the struct.
//
// Every failure carries the numbers needed to fix it. An error saying only that
// an allocation failed, or only that a type did not match, cannot be acted on,
// and one of those in this package is a defect rather than a terse style. See
// specs/001-device-resources.md section 9.
var (
	ErrOutOfDeviceMemory = errors.New("accel: out of device memory")
	ErrFragmented        = errors.New("accel: pool has space but not contiguous space")
	ErrAlignment         = errors.New("accel: alignment violation")
	ErrUsage             = errors.New("accel: usage violation")
	ErrLifetime          = errors.New("accel: lifetime violation")
	ErrFormat            = errors.New("accel: format not usable this way")

	// ErrGraphInFlight reports an attempt to rebind or resubmit a graph while a
	// submission of it is running. A graph's transients are one pool, so two
	// overlapping submissions would write each other's intermediates, and a
	// rebind between them races on which submission sees it. Build one graph per
	// concurrent user. See specs/003-command-graph.md.
	ErrGraphInFlight = errors.New("accel: a submission of this graph is already in flight")

	// ErrRebindOverlap reports a binding update in which two resources supplied
	// to one graph name overlapping bytes and at least one of them is written
	// somewhere in the graph.
	//
	// It cannot be a build-time error: hazards are inferred against the slot,
	// because a slot's eventual resource is unknown then. Two slots resolving to
	// the same bytes means the builder may have omitted an edge between nodes far
	// apart in the graph, which is a missing barrier and therefore a race. See
	// specs/003-command-graph.md, check V24.
	ErrRebindOverlap = errors.New("accel: bound resources overlap where the graph assumed they did not")

	// ErrDeviceLost reports terminal device loss. It is not recoverable: every
	// subsequent call on the device and on every resource under it reports it,
	// and every outstanding fence is signalled with it so that nothing waits
	// forever. Recovery is a full rebuild from Enumerate. See
	// specs/001-device-resources.md section 7.4.
	ErrDeviceLost = driver.ErrDeviceLost
)

// AllocError reports a failed suballocation.
//
// Free and LargestFree together are what distinguish exhaustion from
// fragmentation. They are different problems with different fixes and a bare
// failure tells the caller nothing about which one they have: a pool with
// plenty of free space and no contiguous run of it is behaving exactly as
// specs/001-device-resources.md section 5.3 says a non-compacting allocator
// does, and the fix is separating pools by lifetime class rather than retrying.
type AllocError struct {
	Label       string
	Pool        string
	Kind        MemoryKind
	Requested   int
	Alignment   int
	Free        int
	LargestFree int
	PoolSize    int
}

func (e *AllocError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "accel: alloc %q (%s, align %d) failed in pool %q (%v, %s): ",
		e.Label, humanBytes(e.Requested), e.Alignment, e.Pool, e.Kind, humanBytes(e.PoolSize))

	switch {
	case e.Requested > e.PoolSize:
		fmt.Fprintf(&b, "request exceeds pool size. Pools do not grow (spec 001 section 5.5); "+
			"size the pool from Plan.Memory() before allocating.")
	case e.Free >= e.Requested:
		fmt.Fprintf(&b, "%s free, largest %s. The pool has space but not contiguous space; "+
			"see spec 001 section 5.3.", humanBytes(e.Free), humanBytes(e.LargestFree))
	default:
		fmt.Fprintf(&b, "%s free, largest %s. Exhausted, not fragmented.",
			humanBytes(e.Free), humanBytes(e.LargestFree))
	}
	return b.String()
}

// Unwrap reports fragmentation and exhaustion as distinct classes, since a
// caller who can react at all reacts to them differently.
func (e *AllocError) Unwrap() error {
	if e.Free >= e.Requested && e.Requested <= e.PoolSize {
		return ErrFragmented
	}
	return ErrOutOfDeviceMemory
}

// AlignmentError reports an offset or pitch that does not meet a device
// requirement.
//
// Required always comes from [Limits] and Source names the field it came from,
// never a constant, because the number differs by two orders of magnitude
// across devices of one backend and a caller who hard-codes what worked locally
// has written a bug that only appears elsewhere.
type AlignmentError struct {
	What     string // "view offset", "copy offset", "row pitch"
	Resource string
	Offset   int
	Required int
	Source   string // the Limits field that imposed it
}

func (e *AlignmentError) Error() string {
	return fmt.Sprintf("accel: %s of %q (byte %d) is not a multiple of %d required on this device (Limits.%s)",
		e.What, e.Resource, e.Offset, e.Required, e.Source)
}

func (e *AlignmentError) Unwrap() error { return ErrAlignment }

// UsageError reports a resource used in a way it did not declare.
//
// Over-declaring a usage costs alignment and possibly a stricter memory type;
// under-declaring is a bug. The error therefore lands on the side that is
// wrong, and it names the recording call site so the fix is at the declaration
// rather than at the use.
type UsageError struct {
	Resource string
	Node     NodeID
	Slot     int
	Declared BufferUsage
	Needed   BufferUsage
	Site     string // the recording call site, per spec 003
}

func (e *UsageError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "accel: ")
	if e.Site != "" || e.Slot != 0 {
		fmt.Fprintf(&b, "node %d slot %d ", e.Node, e.Slot)
	}
	fmt.Fprintf(&b, "binds buffer %q declared %v but needs %v", e.Resource, e.Declared, e.Needed)
	if e.Site != "" {
		fmt.Fprintf(&b, ". Recorded at %s", e.Site)
	}
	return b.String()
}

func (e *UsageError) Unwrap() error { return ErrUsage }

// FormatError reports a format the device cannot use as asked.
type FormatError struct {
	Format Format
	Want   string // "renderable", "storage", "host copyable", "filterable"
	Device string
}

func (e *FormatError) Error() string {
	return fmt.Sprintf("accel: format %v is not %s on this device (%s)", e.Format, e.Want, e.Device)
}

func (e *FormatError) Unwrap() error { return ErrFormat }

// LifetimeError reports a resource used or released at the wrong time.
//
// A resource freed while something using it is still outstanding is a
// use-after-free, so the implementation keeps it alive and reports instead: the
// caller's handle is gone and the memory will come back, and the caller learns
// their teardown ordering was wrong. Nothing crashes and nothing leaks. See
// specs/001-device-resources.md section 7.2.
type LifetimeError struct {
	Op       string // "Close", "WriteBuffer", "Bind", ...
	Resource string // the resource's Label
	Reason   string // "in flight", "closed", "has live children", "pending transfer"
	InFlight int    // submissions still holding it, when Reason is "in flight"
	Children int    // live children, when Reason is "has live children"
}

// Lifetime reasons. They are compared as strings by callers who log them, so
// they are named here rather than spelled at each construction site.
const (
	reasonClosed   = "closed"
	reasonInFlight = "in flight"
	reasonChildren = "has live children"
	reasonPending  = "pending transfer"
)

func (e *LifetimeError) Error() string {
	switch e.Reason {
	case reasonInFlight:
		return fmt.Sprintf("accel: %s %q: %d submissions still in flight. The resource stays "+
			"valid until they complete and its memory is released then. Wait on the fence "+
			"before %s to avoid this.", e.Op, e.Resource, e.InFlight, e.Op)
	case reasonPending:
		return fmt.Sprintf("accel: %s %q: a queue write to this buffer has not been flushed. "+
			"The handle is retired and the memory is released when the batch completes; "+
			"call Queue.Flush().Wait() before %s to avoid this.", e.Op, e.Resource, e.Op)
	case reasonChildren:
		return fmt.Sprintf("accel: %s %q: %d live children. Close them first; "+
			"closing is ordered rather than recursive (spec 001 section 7.2).",
			e.Op, e.Resource, e.Children)
	default:
		return fmt.Sprintf("accel: %s %q: resource is %s.", e.Op, e.Resource, e.Reason)
	}
}

func (e *LifetimeError) Unwrap() error { return ErrLifetime }

// humanBytes formats a byte count the way the messages in spec 001 sections 5.4
// and 9 do, because a pool size in raw bytes is not a number anyone compares
// against a model's size by eye.
func humanBytes(n int) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%d B", n)
	}
	div, exp := int64(unit), 0
	for v := int64(n) / unit; v >= unit && exp < 3; v /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(n)/float64(div), "KMGT"[exp])
}
