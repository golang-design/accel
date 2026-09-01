// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel

import (
	"fmt"
	"sort"
	"strings"
)

// Diagnostic is one thing a cooperative kernel did that no target defines.
//
// It carries where and who as well as what, because a report saying only that
// something raced tells a reader nothing they can act on: which invocation, at
// which line, over which element is the whole content.
type Diagnostic struct {
	Kind DiagKind

	// Kernel and Workgroup locate the failure in the dispatch.
	Kernel    string
	Workgroup ID3

	// Invocation is the offending invocation, and Other the one it conflicts
	// with, where there is one.
	Invocation ID3
	Other      ID3
	HasOther   bool

	// Element is the shared index or byte offset involved, or -1.
	Element int

	// Detail is the rest, already phrased for a reader.
	Detail string
}

// DiagKind is which failure this is.
type DiagKind int

const (
	// DiagUndefinedRead is a read of shared memory nothing wrote.
	DiagUndefinedRead DiagKind = iota

	// DiagArrival is invocations reaching different barriers, or not arriving.
	DiagArrival

	// DiagConflict is two invocations touching one location with nothing
	// ordering them and at least one writing.
	DiagConflict

	// DiagUndefinedLane is a broadcast or shuffle reading a lane of its own
	// subgroup that is not taking part in the operation, whose result
	// specs/002-compute-model.md section 5.2 rule 3 leaves undefined. It also
	// covers a broadcast whose lane operand is not dynamically uniform.
	DiagUndefinedLane
)

func (k DiagKind) String() string {
	switch k {
	case DiagUndefinedRead:
		return "read of undefined shared memory"
	case DiagArrival:
		return "barrier arrival mismatch"
	case DiagConflict:
		return "conflicting access"
	case DiagUndefinedLane:
		return "read of an inactive subgroup lane"
	}
	return "unknown"
}

func (d Diagnostic) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "accel: kernel %q: %v", d.Kernel, d.Kind)
	fmt.Fprintf(&b, " in workgroup %d,%d,%d", d.Workgroup.X, d.Workgroup.Y, d.Workgroup.Z)
	fmt.Fprintf(&b, " by invocation %d,%d,%d", d.Invocation.X, d.Invocation.Y, d.Invocation.Z)
	if d.HasOther {
		fmt.Fprintf(&b, " against invocation %d,%d,%d", d.Other.X, d.Other.Y, d.Other.Z)
	}
	if d.Element >= 0 {
		fmt.Fprintf(&b, " at element %d", d.Element)
	}
	if d.Detail != "" {
		fmt.Fprintf(&b, ": %s", d.Detail)
	}
	return b.String()
}

// Diagnostics is a set of them, ordered so a report reads the same every run.
type Diagnostics []Diagnostic

func (ds Diagnostics) Error() string {
	parts := make([]string, len(ds))
	for i, d := range ds {
		parts[i] = d.Error()
	}
	return strings.Join(parts, "\n")
}

// sortStable orders diagnostics by invocation and then element.
//
// Deterministically, because specs/019-cooperative-diagnostics.md requires a
// report on the first offending run rather than on an unlucky interleaving, and
// a report whose order changes between runs is one a developer learns to
// re-run past.
func (ds Diagnostics) sortStable() {
	sort.SliceStable(ds, func(i, j int) bool {
		a, b := ds[i], ds[j]
		if a.Invocation != b.Invocation {
			return linearOf(a.Invocation) < linearOf(b.Invocation)
		}
		return a.Element < b.Element
	})
}

func linearOf(id ID3) uint64 {
	return uint64(id.Z)<<42 | uint64(id.Y)<<21 | uint64(id.X)
}

// Describe names a barrier for a report: its index, and its source position
// where one is known.
func (b BarrierID) Describe() string {
	if b.Pos != "" {
		return fmt.Sprintf("barrier %d (%s)", b.Index, b.Pos)
	}
	return fmt.Sprintf("barrier %d", b.Index)
}

// definedBits tracks which shared elements have been written.
//
// # Why a shadow bit and not a sentinel value
//
// A sentinel is a value the kernel could legitimately compute. A check
// comparing against one therefore either misses the read, because the kernel
// wrote the sentinel itself, or fires on a correct kernel that happened to
// compute it. Neither is acceptable in something whose job is to be believed.
//
// specs/019-cooperative-diagnostics.md asks for the strong form: a kernel
// reading shared memory it never wrote fails for *every* stored bit pattern.
// A shadow bit meets that by construction, because it does not look at the
// value at all. A sentinel implementation fails the sweep, which is the point
// of writing the test that way.
type definedBits struct {
	bits []uint64
}

func newDefinedBits(n int) *definedBits {
	return &definedBits{bits: make([]uint64, (n+63)/64)}
}

func (d *definedBits) reset() {
	for i := range d.bits {
		d.bits[i] = 0
	}
}

func (d *definedBits) markWritten(i int) {
	if i < 0 || i/64 >= len(d.bits) {
		return
	}
	d.bits[i/64] |= 1 << (uint(i) % 64)
}

func (d *definedBits) written(i int) bool {
	if i < 0 || i/64 >= len(d.bits) {
		return false
	}
	return d.bits[i/64]&(1<<(uint(i)%64)) != 0
}

// SharedTracker records what a workgroup's invocations did to shared memory.
//
// # Why the generated code calls it rather than the runtime watching
//
// Nothing outside the kernel can see a shared-memory access: it is an ordinary
// slice index in generated Go. So the lowering is instrumented — every shared
// read and write is preceded by a call here — which is what
// specs/004-kernel-authoring.md means by calling the CPU lowering instrumented
// rather than merely generated.
//
// A nil tracker makes every call a no-op that the compiler removes, which is
// how a dispatch with diagnostics off pays nothing for a check the default
// wants. The instrumentation is always emitted; whether it does anything is a
// runtime decision, because a second lowering would be a second thing to keep
// correct.
type SharedTracker struct {
	kernel    string
	workgroup ID3

	// defined is one shadow-bit set per declared shared array.
	defined []*definedBits

	// epoch is the current rendezvous epoch. Accesses are compared within one,
	// because a barrier is exactly what orders accesses in different ones.
	epoch int

	// touched records this epoch's accesses per array, for the conflict check.
	touched []map[int]touch

	diags Diagnostics

	// current is the invocation being advanced, set by the scheduler.
	current ID3
}

// touch is one invocation's access to one element within an epoch.
type touch struct {
	by    ID3
	wrote bool
}

// NewSharedTracker makes a tracker for one workgroup's shared arrays.
func NewSharedTracker(kernelName string, workgroup ID3, sizes []int) *SharedTracker {
	t := &SharedTracker{kernel: kernelName, workgroup: workgroup}
	for _, n := range sizes {
		t.defined = append(t.defined, newDefinedBits(n))
		t.touched = append(t.touched, map[int]touch{})
	}
	return t
}

// Begin marks which invocation is running, so a report can name it.
func (t *SharedTracker) Begin(inv ID3) {
	if t == nil {
		return
	}
	t.current = inv
}

// Write records a store.
func (t *SharedTracker) Write(array, index int) {
	if t == nil || array < 0 || array >= len(t.defined) {
		return
	}
	t.defined[array].markWritten(index)
	t.conflict(array, index, true)
}

// ReadAt is Read returning the index, so the instrumentation sits inside an
// expression: a load appears wherever a value does, and hoisting it into a
// statement would change the order of evaluation.
func (t *SharedTracker) ReadAt(array, index int) int {
	t.Read(array, index)
	return index
}

// Read records a load, and reports one of memory nothing wrote.
//
// The report does not look at the value, which is the whole point: a check
// comparing against a sentinel either misses a read whose kernel wrote the
// sentinel itself, or fires on a correct kernel that computed it.
func (t *SharedTracker) Read(array, index int) {
	if t == nil || array < 0 || array >= len(t.defined) {
		return
	}
	if !t.defined[array].written(index) {
		t.diags = append(t.diags, Diagnostic{
			Kind: DiagUndefinedRead, Kernel: t.kernel, Workgroup: t.workgroup,
			Invocation: t.current, Element: index,
			Detail: "nothing in this workgroup wrote it, so its contents are whatever " +
				"the last workgroup left there",
		})
	}
	t.conflict(array, index, false)
}

// conflict records an access and reports an unordered conflicting pair.
//
// Within an epoch, because a barrier is what orders accesses in different ones.
// The comparison is after the fact, from records, so the report does not depend
// on the two accesses actually interleaving — which is what makes it fire on
// the first offending run rather than on an unlucky schedule.
func (t *SharedTracker) conflict(array, index int, wrote bool) {
	prev, seen := t.touched[array][index]
	if seen && prev.by != t.current && (prev.wrote || wrote) {
		t.diags = append(t.diags, Diagnostic{
			Kind: DiagConflict, Kernel: t.kernel, Workgroup: t.workgroup,
			Invocation: t.current, Other: prev.by, HasOther: true, Element: index,
			Detail: "both touched it between the same pair of barriers and at least one " +
				"wrote, so nothing orders them",
		})
	}
	// The writer is remembered in preference to a reader, since a later reader
	// conflicting with an earlier writer is the pair worth naming.
	if !seen || wrote {
		t.touched[array][index] = touch{by: t.current, wrote: wrote || prev.wrote}
	}
}

// Epoch advances past a barrier, which ends the window conflicts are compared
// in. Definedness carries across: a barrier makes a write visible, it does not
// undo it.
func (t *SharedTracker) Epoch() {
	if t == nil {
		return
	}
	t.epoch++
	for i := range t.touched {
		clear(t.touched[i])
	}
}

// Reset starts a new workgroup. Shared storage is fresh, so nothing is defined
// and nothing has been touched.
func (t *SharedTracker) Reset(workgroup ID3) {
	if t == nil {
		return
	}
	t.workgroup = workgroup
	t.epoch = 0
	t.diags = nil
	for i := range t.defined {
		t.defined[i].reset()
		clear(t.touched[i])
	}
}

// Diagnostics reports what this workgroup did, in a stable order.
func (t *SharedTracker) Diagnostics() Diagnostics {
	if t == nil || len(t.diags) == 0 {
		return nil
	}
	out := make(Diagnostics, len(t.diags))
	copy(out, t.diags)
	out.sortStable()
	return out
}
