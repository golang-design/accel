// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"fmt"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
)

// A recorder accumulates declarations. Nothing is checked as it goes, by
// design: [Recorder.Build] is where every error surfaces, so that a caller sees
// all of them at once with node identities attached rather than one per call
// with none. See specs/003-command-graph.md.
//
// The cost of that choice is that a recording call cannot return an error, so
// it returns a NodeID and the recorder remembers what went wrong. A NodeID is
// still handed back for a call that failed, so that later calls referring to it
// produce their own errors instead of collapsing into a cascade.

// NewRecorder returns a recorder for building a [Graph].
//
// A Recorder belongs to one goroutine and is used once: Build consumes it. The
// [Graph] it produces is immutable but permits only one in-flight submission;
// build one graph per concurrent user.
func (d *Device) NewRecorder() *Recorder {
	return &Recorder{state: recorderState{dev: d}}
}

// recNode is one recorded operation.
type recNode struct {
	id    NodeID
	kind  NodeKind
	label string

	// data is a host write's payload, copied at record time. The graph owns
	// these bytes: holding the caller's slice would make what a submission
	// writes depend on when the caller last touched it, which an immutable
	// graph cannot promise.
	data []byte

	// accesses is what this node touches, in declaration order. Declaration
	// order rather than a map, because the barrier plan and the diagnostics
	// derived from it must not depend on iteration order.
	accesses []access

	// stage is the part of the pipeline this node runs in, which is what a
	// barrier names on each side.
	stage stage

	// pipeline, count and uniforms are a dispatch node's payload.
	pipeline *ComputePipeline
	count    kernel.ID3
	uniforms []any

	// texture is a texture-copy node's image.
	texture *Texture

	// indirect marks a node whose workgroup count the device supplies, in which
	// case count is the build-time maximum and the last access is the count
	// buffer rather than a binding.
	indirect bool
}

// access is one resource range one node touches, in bytes.
type access struct {
	res  resourceRef
	off  int // bytes from the start of the buffer, or of the slot's bound window
	size int
	mode Access
}

func (a access) writes() bool { return a.mode == AccessWrite || a.mode == AccessReadWrite }

// resourceRef names either a resource known now or one supplied before
// submission. Exactly one field is set, and the two constructors that build one
// -- [Recorder.declare] and [Recorder.slotAccess] -- each set exactly one.
//
// It is not the closed shape [driver.Operand] uses, because unlike an operand a
// resourceRef never leaves this package and never crosses a build boundary: it
// is built and consumed within one Build. The operand is the one that has to
// defend itself.
type resourceRef struct {
	buf  *Buffer
	tex  *Texture
	slot Slot
}

func (r resourceRef) String() string {
	switch {
	case r.buf != nil:
		return fmt.Sprintf("buffer %q", r.buf.desc.Label)
	case r.tex != nil:
		return fmt.Sprintf("texture %q", r.tex.desc.Label)
	case r.slot != 0:
		return fmt.Sprintf("slot %d", int(r.slot))
	}
	return "an unset resource"
}

// transient is a builder-owned intermediate.
//
// Its Buffer exists from the moment it is declared, so a caller can build views
// of it while recording, and its memory is assigned at Build. That split is
// what lets [Graph.Memory] report a requirement before anything is allocated,
// which is the number a caller sizing a KV cache needs.
type transient struct {
	owner *Recorder
	buf   *Buffer
	bytes int

	// pool is the driver block backing every transient of this graph, assigned
	// at Build.
	pool driver.Block

	// offset within the graph's transient pool, assigned at Build, and placed
	// once packing has run.
	offset int
	placed bool

	// first and last are the record-order positions of this transient's first
	// and last user, which is what PeakBytes is defined over. They are -1 until
	// a node uses it.
	//
	// They are the *interval*, and the interval is not what aliasing may use:
	// see [Graph.compatible]. They are kept because PeakBytes is defined over
	// the record-order linearization, and the gap between it and the packed
	// pool is exactly what DAG-safe aliasing costs.
	first, last int

	// users are the nodes that touch this transient, in record order and
	// without duplicates. This is what the interference relation quantifies
	// over, and record order is what makes the packing deterministic.
	users []NodeID

	// writer is the first node to write this transient, or -1. It is the
	// packing order's tie break.
	writer int
}

// Recorder state.
type recorderState struct {
	dev *Device

	nodes      []recNode
	slots      []SlotDescriptor
	transients []*transient

	// errs are the reasons Build will fail, accumulated in record order so the
	// report reads in the order the caller wrote the calls.
	errs []error

	built bool

	// shared is the caller-owned transient pool this graph will plan into, or
	// nil when it owns its transients. See Recorder.UseTransientPool.
	shared *TransientPool

	// collectStats is whether the graph carries back the counters only the
	// device knows. Off by default, because they cost a readback.
	collectStats bool
}

func (r *Recorder) fail(format string, args ...any) {
	r.state.errs = append(r.state.errs, fmt.Errorf("accel: "+format, args...))
}

// node appends a recorded node and returns its id.
func (r *Recorder) node(kind NodeKind, label string, accesses []access, data []byte) NodeID {
	id := NodeID(len(r.state.nodes))
	r.state.nodes = append(r.state.nodes, recNode{
		id: id, kind: kind, label: label, accesses: accesses, data: data,
		stage: stageFor(kind),
	})
	return id
}

// stageFor is which part of the pipeline a node kind runs in.
func stageFor(k NodeKind) stage {
	switch k {
	case NodeDispatch, NodeDispatchIndirect:
		return stageCompute
	}
	return stageTransfer
}

// declare turns a view into an access declaration, reporting why it cannot.
//
// The range is checked here rather than at Build because this is where the
// caller's call site is: specs/001-device-resources.md section 7.3 requires a
// view's range to be re-checked at every use, and a recording call is a use.
func (r *Recorder) declare(op string, v BufferView, mode Access) (access, bool) {
	if v.Buffer == nil {
		r.fail("%s: the view names no buffer", op)
		return access{}, false
	}
	if err := v.checkRange(op); err != nil {
		r.state.errs = append(r.state.errs, err)
		return access{}, false
	}
	if v.Buffer.transient != nil && v.Buffer.transient.owner != r {
		r.fail("%s: %q is another graph's transient", op, v.Buffer.desc.Label)
		return access{}, false
	}
	if v.Buffer.transient == nil {
		if err := v.Buffer.state.checkOpen(op); err != nil {
			r.state.errs = append(r.state.errs, err)
			return access{}, false
		}
		if v.Buffer.pool.dev != r.state.dev {
			r.fail("%s: %q belongs to a different device", op, v.Buffer.desc.Label)
			return access{}, false
		}
	}
	off, size := v.byteRange()
	return access{res: resourceRef{buf: v.Buffer}, off: off, size: size, mode: mode}, true
}

// slotAccess declares an access relative to a slot's eventual resource.
func (r *Recorder) slotAccess(op string, s Slot, off, size int, mode Access) (access, bool) {
	d, ok := r.slotDescriptor(s)
	if !ok {
		r.fail("%s: slot %d was not declared by this recorder", op, int(s))
		return access{}, false
	}
	// V21: a recorded use must be covered by the descriptor, so that the
	// graph-wide overlap check at Rebind can reason about the slot without
	// consulting every node. The access mode is checked the same way.
	if limit := d.MinCount * d.DType.Size(); off < 0 || size < 0 || off > limit || size > limit-off {
		r.fail("%s: slot %q is declared to hold at least %d bytes and this use is [%d, %d)",
			op, d.Name, limit, off, off+size)
		return access{}, false
	}
	if !coveredBy(mode, d.Access) {
		r.fail("%s: slot %q is declared %v and this use is %v", op, d.Name, d.Access, mode)
		return access{}, false
	}
	return access{res: resourceRef{slot: s}, off: off, size: size, mode: mode}, true
}

func (r *Recorder) slotDescriptor(s Slot) (SlotDescriptor, bool) {
	if s < 1 || int(s) > len(r.state.slots) {
		return SlotDescriptor{}, false
	}
	return r.state.slots[s-1], true
}

// coveredBy reports whether a use is permitted by a declared access mode.
func coveredBy(use, declared Access) bool {
	if declared == AccessReadWrite {
		return true
	}
	return use == declared
}

// Slot declares a rebindable binding point. See the doc on the public wrapper.
func (r *Recorder) slotImpl(desc SlotDescriptor) Slot {
	if r.state.built {
		r.fail("Slot: this recorder has already been built")
		return 0
	}
	if desc.Name == "" {
		desc.Name = fmt.Sprintf("slot %d", len(r.state.slots)+1)
	}
	if desc.MinCount < 0 {
		r.fail("Slot %q: MinCount is %d", desc.Name, desc.MinCount)
		desc.MinCount = 0
	}
	if desc.DType.Size() == 0 {
		r.fail("Slot %q: %v is not a dtype", desc.Name, desc.DType)
	}
	r.state.slots = append(r.state.slots, desc)
	return Slot(len(r.state.slots))
}

// transientImpl declares a builder-owned buffer.
func (r *Recorder) transientImpl(desc BufferDescriptor) BufferView {
	if r.state.built {
		r.fail("Transient: this recorder has already been built")
		return BufferView{}
	}
	if desc.Label == "" {
		desc.Label = fmt.Sprintf("transient %d", len(r.state.transients))
	}
	bytes, err := r.state.dev.bufferBytes(desc)
	if err != nil {
		r.state.errs = append(r.state.errs, err)
		return BufferView{}
	}

	t := &transient{bytes: bytes, first: -1, last: -1, writer: -1, owner: r}
	// The Buffer exists now and its memory arrives at Build. Nothing else in
	// this package builds a Buffer without a pool, so the transient pointer is
	// what every path keys on to know the difference.
	t.buf = &Buffer{desc: desc, bytes: bytes, transient: t}
	t.buf.state.init(desc.Label)
	r.state.transients = append(r.state.transients, t)
	return BufferView{Buffer: t.buf, DType: desc.DType, Offset: 0, Count: desc.Count}
}

// blockFor resolves a concrete buffer to the bytes a plan operand names.
func blockFor(b *Buffer) (driver.Block, int) {
	if b.transient != nil {
		return b.transient.pool, b.transient.offset
	}
	return b.pool.block, b.alloc.Offset
}

// touch records that a node uses a transient, which is what PeakBytes is
// computed over.
//
// Record-order positions rather than a liveness set, because PeakBytes is
// defined over the record-order linearization. specs/017-graph-aliasing.md adds
// the reachability-based interference relation alongside this; it does not
// replace it, because the two answer different questions and the gap between
// their answers is the number a caller needs.
func (r *Recorder) touch(id NodeID, a access) {
	t := a.res.buf.transientOf()
	if t == nil {
		return
	}
	if t.first < 0 {
		t.first = int(id)
	}
	t.last = int(id)
	if a.writes() && t.writer < 0 {
		t.writer = int(id)
	}
	// One entry per node, not per access: a node reading and writing one
	// transient is one user, and counting it twice would make the interference
	// quantifier do redundant work without changing its answer.
	if n := len(t.users); n == 0 || t.users[n-1] != id {
		t.users = append(t.users, id)
	}
}

// transientOf reports the transient a buffer belongs to, or nil.
func (b *Buffer) transientOf() *transient {
	if b == nil {
		return nil
	}
	return b.transient
}
