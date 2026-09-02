// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"errors"
	"fmt"
	"runtime"
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

	// ErrNoAdapter reports that nothing was enumerated at all.
	//
	// Distinct from ErrPolicy because the two need different fixes and a caller
	// branches on which: nothing enumerated means no driver, no device, or a
	// build without the backend, and no policy change helps.
	ErrNoAdapter = errors.New("accel: no adapter was enumerated")

	// ErrPolicy reports that adapters were enumerated and the policy excluded
	// every one.
	//
	// The fix is the policy, and [SelectionReport] says which clause rejected
	// which adapter -- so a caller can widen exactly the one that cost them the
	// device rather than dropping the policy wholesale.
	ErrPolicy = errors.New("accel: no adapter satisfied the policy")

	// The binding sentinels of specs/003-command-graph.md's error taxonomy.
	//
	// They exist so a caller can branch on *which* declaration a bound resource
	// failed rather than matching a string. The spec's example is the one that
	// motivated them: a caller trying an f16 path and falling back to f32 needs
	// to tell a dtype disagreement from a size one, and `errors.Is` is how.
	//
	// Added 2026-08-27 for the checks in checkBinding, which is where a caller's
	// own resource meets the graph's declaration. The rest of that taxonomy --
	// BuildError, NodeError and the recorded call site -- is not built; see
	// specs/STATUS.md.

	// ErrDTypeMismatch reports a view whose dtype is not the slot's. A
	// reinterpreting view is legal on a buffer and not on a slot: the recorded
	// nodes computed their byte offsets from the descriptor's dtype (V3).
	ErrDTypeMismatch = errors.New("accel: dtype mismatch")

	// ErrKindMismatch reports a resource bound to a slot of another kind, such
	// as a buffer bound where a texture was declared (V2).
	ErrKindMismatch = errors.New("accel: binding kind mismatch")

	// ErrTooSmall reports a view with fewer elements than the recorded nodes
	// declared they would read or write (V5).
	ErrTooSmall = errors.New("accel: resource too small for declared access")

	// ErrUsageMissing reports a resource created without a usage flag that the
	// access recorded against it needs (V6).
	ErrUsageMissing = errors.New("accel: resource lacks a required usage flag")

	// ErrSlotUnbound reports a slot with no resource, or an index that is not
	// one of the graph's slots (V1).
	ErrSlotUnbound = errors.New("accel: binding slot has no resource")

	// ErrForeignResource reports a resource belonging to another device. It is
	// separate from ErrUsage because the fix is different: a usage violation is
	// a flag at creation, and this is the wrong object entirely (V19).
	ErrForeignResource = errors.New("accel: resource belongs to another device")

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

	// ErrUniformLoadAliased reports a dispatch that binds a write binding over
	// bytes a binding declared with //accel:uniform reads. The compiler
	// accepted the kernel's barriers on the promise that no invocation of the
	// dispatch writes those bytes, so the alias would make a barrier's control
	// flow diverge on a device. See specs/063-uniform-loads.md.
	ErrUniformLoadAliased = errors.New("accel: a dispatch writes bytes one of its bindings declared uniform")

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
	// A node is named only with the call site that recorded it: a bind-time
	// error knows the slot and not the node, and node 0 is a real node rather
	// than "none".
	if e.Site != "" {
		fmt.Fprintf(&b, "node %d ", e.Node)
	}
	if e.Slot != 0 {
		fmt.Fprintf(&b, "slot %d ", e.Slot)
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
	Format   Format
	Want     string // "renderable", "sampleable", "host copyable", "usable as a storage image"
	Device   string
	Resource string // the texture's label, when the format reached the device on one
}

func (e *FormatError) Error() string {
	if e.Resource != "" {
		return fmt.Sprintf("accel: texture %q: format %v is not %s on this device (%s)",
			e.Resource, e.Format, e.Want, e.Device)
	}
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
	reasonClosed    = "closed"
	reasonInFlight  = "in flight"
	reasonChildren  = "has live children"
	reasonPending   = "pending transfer"
	reasonTransient = "a graph transient"
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
	case reasonTransient:
		return fmt.Sprintf("accel: %s %q: it is a graph transient. Its memory belongs to the "+
			"builder, which may reuse it between nodes, so only the graph that declared it "+
			"may release it — closing the graph is what returns it (spec 001 section 7.3).",
			e.Op, e.Resource)
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

// BuildError is what [Recorder.Build] returns when recording found problems.
// It is a collection: every recorded error is reported together, one
// [NodeError] each, so a caller learns about every mistake in one build rather
// than one per rebuild. specs/003-command-graph.md's error taxonomy.
//
// errors.Is on a BuildError reaches every entry's sentinel through Unwrap, and
// errors.As with a *NodeError finds the first entry.
type BuildError struct {
	Errs []*NodeError
}

func (e *BuildError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "accel: graph build failed: %d error", len(e.Errs))
	if len(e.Errs) != 1 {
		b.WriteString("s")
	}
	for _, n := range e.Errs {
		b.WriteString("\n  ")
		b.WriteString(n.Error())
	}
	return b.String()
}

// Unwrap exposes every entry, so errors.Is and errors.As look through the
// collection.
func (e *BuildError) Unwrap() []error {
	out := make([]error, len(e.Errs))
	for i, n := range e.Errs {
		out[i] = n
	}
	return out
}

// NodeError is one problem with one recorded node.
//
// Site is the caller's file, line and column -- the first frame outside this
// module when the recording call was made, so a caller sees their own line
// rather than the recorder's. The column is 0, which runtime.Caller does not
// report; the field is in the format because the kernel compiler's
// diagnostics do have columns and one tool should parse both. Cause is the
// error the check produced, which wraps one of the sentinels above where the
// check has one, so errors.Is works through it; Detail is the human half.
type NodeError struct {
	Node   NodeID
	Label  string
	Kind   NodeKind
	Site   string
	Cause  error
	Detail string

	// atNode is whether an entry point had declared the node when the
	// error was raised. A slot or transient declaration raises errors
	// between nodes, and those carry the site and the detail alone.
	atNode bool
}

func (e *NodeError) Error() string {
	var b strings.Builder
	if e.Site != "" {
		b.WriteString(e.Site)
		b.WriteString(": ")
	}
	if e.atNode {
		fmt.Fprintf(&b, "node %d", int(e.Node))
		if e.Label != "" {
			fmt.Fprintf(&b, " %q", e.Label)
		}
		fmt.Fprintf(&b, " (%s): ", e.Kind)
	}
	b.WriteString(e.Detail)
	return b.String()
}

func (e *NodeError) Unwrap() error { return e.Cause }

// callSite is the first frame outside this module on the current stack, as
// file:line:0, or "" when the whole stack is the module's (a test in it).
func callSite() string {
	pcs := make([]uintptr, 32)
	n := runtime.Callers(2, pcs)
	frames := runtime.CallersFrames(pcs[:n])
	for {
		f, more := frames.Next()
		if f.Function != "" && !insideModule(f.Function) {
			return fmt.Sprintf("%s:%d:0", f.File, f.Line)
		}
		if !more {
			return ""
		}
	}
}

// insideModule reports whether a function belongs to this module's own
// packages. An external test package (accel_test) is a caller, not the
// module, which is what lets a test see its own line.
func insideModule(fn string) bool {
	pkg := fn
	if i := strings.LastIndex(pkg, "/"); i >= 0 {
		if j := strings.Index(pkg[i:], "."); j >= 0 {
			pkg = pkg[:i+j]
		}
	} else if j := strings.Index(pkg, "."); j >= 0 {
		pkg = pkg[:j]
	}
	return pkg == "golang.design/x/accel" || strings.HasPrefix(pkg, "golang.design/x/accel/")
}
