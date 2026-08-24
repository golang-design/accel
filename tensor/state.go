// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"fmt"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/testkernels"
)

// Persistent state, and why mutation is explicit in the graph.
//
// A KV cache is the one thing in a decode step that is genuinely mutable: every
// token appends to it and the next token reads what the last one wrote. A graph
// of immutable values has to say that somehow, and the choice
// specs/007-tensor-layer.md makes is SSA-style versions -- a write returns the
// *next version* of the same binding, so the dependency between the write and
// the read after it is an ordinary edge in the DAG rather than a rule the
// planner has to be told.
//
// That is what makes the graph's hazard inference cover it. A scatter into a
// state buffer followed by a gather from it produces a read-after-write edge
// and a barrier, because the two nodes declare overlapping byte ranges -- which
// is exactly what specs/009-sequencing.md's risk table asked to be checked
// before this milestone, and was.

// StateDesc declares caller-owned mutable storage.
type StateDesc struct {
	Name  string
	DType DType

	// Shape is the whole extent, including the sequence capacity. A KV cache is
	// [capacity, heads, headDim].
	Shape Shape
}

// State is a version of caller-owned mutable storage.
//
// A value rather than a handle: [ScatterRows] returns the next version, and an
// operator reading an *earlier* version is reading what was there before the
// write. Holding on to an old version is therefore meaningful rather than a
// mistake, and the DAG records which one each reader meant.
type State struct {
	b    *Builder
	desc StateDesc

	// version counts writes. It exists so a reader of an older version is a
	// different node from a reader of a newer one, which is what makes the
	// ordering explicit rather than positional.
	version int

	// producer is the node whose write made this version, or -1 for the
	// version that was there when the plan started.
	producer int

	// offset and shape describe a layer's slice of the parent, for LayerState.
	offset int
	shape  Shape

	poison bool
}

// Persistent declares caller-owned mutable storage.
//
// Never transient and never aliased by the planner: the caller owns the buffer
// and its contents outlive the submission, which is the whole point of a cache.
func NewState(b *Builder, d StateDesc) *State {
	t := b.declare(1, "Persistent", ValueDesc{
		Name: d.Name, DType: d.DType, Shape: d.Shape,
	}, PortState)
	if t.poison {
		return &State{b: b, poison: true, producer: -1}
	}
	return &State{
		b: b, desc: d, producer: -1, shape: d.Shape,
	}
}

// ReadState returns the tensor a state version holds.
//
// # Why an older version is refused
//
// specs/007-tensor-layer.md makes a write return the *next version*, so holding
// an older one is meaningful. Expressing that on the device would mean copying
// the previous contents aside before the write, because both versions live in
// one caller-owned buffer -- and v0 does not do that.
//
// The alternative was to let an old version read the new contents, which is
// what the first implementation did: the version chain compiled, ordered
// nothing, and a test that deliberately read the stale version still passed.
// A distinction that cannot be violated is not a distinction, so this refuses.
func ReadState(b *Builder, s *State) *Tensor {
	if s == nil || s.poison {
		return b.poison()
	}
	if !stale(b, s) {
		return readState(b, s)
	}
	return b.fail(1, "ReadState", "%s", staleMessage(b, s))
}

// stale reports whether a version has been superseded.
//
// Separate from the message so an operator reading a state on a caller's behalf
// can check it at its own frame depth. Attributing the diagnostic to the line
// inside this package that called ReadState would send a reader to the wrong
// file, and frame counting through a helper is fragile because the compiler may
// inline one away.
func stale(b *Builder, s *State) bool {
	return supersededBy(b, s) > s.version
}

// supersededBy is the highest version written to any range overlapping this
// state's own, or its own version when nothing has.
//
// Overlap rather than name equality: a per-layer cache is one buffer whose
// layers are disjoint, so a write to one layer says nothing about a reader of
// another. A write to the whole cache does overlap every layer, and a write to
// a layer overlaps the whole cache, so mixing the two is still caught.
func supersededBy(b *Builder, s *State) int {
	mine := s.window()
	latest := s.version
	for w, v := range b.stateVersion {
		if w.port != mine.port || v <= latest {
			continue
		}
		if w.off >= mine.off+mine.count || mine.off >= w.off+w.count {
			continue
		}
		latest = v
	}
	return latest
}

// window is the range of the caller's buffer this state names.
func (s *State) window() window {
	return window{s.desc.Name, s.offset, s.shape.Elements()}
}

func staleMessage(b *Builder, s *State) string {
	return fmt.Sprintf("version %d of %q, which has been written %d time(s) since; both "+
		"versions live in one caller-owned buffer, so reading the older one would need its "+
		"contents copied aside first, which v0 does not do",
		s.version, s.desc.Name, supersededBy(b, s)-s.version)
}

// readState builds the tensor without checking the version, for a caller that
// has already checked it.
func readState(b *Builder, s *State) *Tensor {
	if s == nil || s.poison {
		return b.poison()
	}
	t := &Tensor{
		b: b, dtype: s.desc.DType, shape: s.shape,
		strides: contiguous(s.shape),
		node:    s.producer, port: s.desc.Name,
		// The offset is the *window's*, not the view's: the binding starts at
		// this layer, so the value is contiguous from element zero of what the
		// kernel is given. Carrying it as a view offset instead would make
		// every layer read look like a strided operand and be refused as one.
		win: &window{s.desc.Name, s.offset, s.shape.Elements()},
	}
	// A version produced by a write reads that node's output; the initial
	// version reads the bound buffer.
	if s.producer >= 0 {
		t.port = s.desc.Name
	}
	return t
}

// LayerState is a compile-time view of one layer's slice of a state.
//
// The version chain and the binding identity are the parent's, which is what
// makes a per-layer cache one buffer rather than N: a model with thirty-two
// layers binds one KV tensor and each layer addresses its own slice.
func LayerState(b *Builder, s *State, layer int) *State {
	if s == nil || s.poison {
		return &State{b: b, poison: true, producer: -1}
	}
	if len(s.shape) < 1 {
		b.fail(1, "LayerState", "%q has no shape", s.desc.Name)
		return &State{b: b, poison: true, producer: -1}
	}
	layers := s.shape[0]
	if layer < 0 || layer >= layers {
		b.fail(1, "LayerState", "layer %d of %q, which has %d", layer, s.desc.Name, layers)
		return &State{b: b, poison: true, producer: -1}
	}
	inner := s.shape[1:]
	next := *s
	next.shape = inner
	next.offset = s.offset + layer*inner.Elements()
	return &next
}

// ScatterRows writes rows into a state at runtime indices and returns the next
// version.
//
// The next version rather than nothing, so a reader downstream names *which*
// state it meant. An operator holding the version before this write reads what
// was there, and the graph orders the two because their byte ranges overlap.
func ScatterRows(b *Builder, s *State, rows *Tensor, ids *Tensor) *State {
	if s == nil || s.poison || poisoned(rows, ids) {
		return &State{b: b, poison: true, producer: -1}
	}
	if s.desc.DType != accel.F32 || rows.dtype != accel.F32 {
		b.fail(1, "ScatterRows", "state is %v and rows are %v; the registered kernel writes "+
			"f32", s.desc.DType, rows.dtype)
		return &State{b: b, poison: true, producer: -1}
	}
	if ids.dtype != accel.U32 {
		b.fail(1, "ScatterRows", "ids are %v and must be u32", ids.dtype)
		return &State{b: b, poison: true, producer: -1}
	}
	if len(s.shape) < 1 || len(rows.shape) < 1 {
		b.fail(1, "ScatterRows", "state is %v and rows are %v", s.shape, rows.shape)
		return &State{b: b, poison: true, producer: -1}
	}
	width := s.shape.Elements() / s.shape[0]
	if rows.shape[len(rows.shape)-1] != width {
		b.fail(1, "ScatterRows", "rows are %v and one row of %q is %d wide",
			rows.shape, s.desc.Name, width)
		return &State{b: b, poison: true, producer: -1}
	}
	count := rows.shape.Elements() / width
	if ids.shape.Elements() != count {
		b.fail(1, "ScatterRows", "%d rows and %d ids", count, ids.shape.Elements())
		return &State{b: b, poison: true, producer: -1}
	}

	if stale(b, s) {
		b.fail(1, "ScatterRows", "%s", staleMessage(b, s))
		return &State{b: b, poison: true, producer: -1}
	}
	out := b.record(node{
		op: "ScatterRows", inputs: []*Tensor{rows, ids},
		kernel:  &testkernels.ScatterRowsKernel,
		outPort: s.desc.Name,
		outOff:  s.offset,
		uniform: func(map[string]ScalarValue) any {
			return testkernels.RowParams{
				Rows: uint32(count), Width: uint32(width), Capacity: uint32(s.shape[0]),
			}
		},
		grid: func(*Tensor) accel.WorkgroupCount {
			wg := int(testkernels.ScatterRowsKernel.WorkgroupSize.X)
			n := count * width
			return accel.WorkgroupCount{X: (n + wg - 1) / wg}
		},
		reason: "the scatter variant; an index at or above capacity writes nothing, " +
			"because a GPU cannot report one",
	}, s.desc.DType, s.shape)

	next := *s
	next.version = s.version + 1
	next.producer = out.node
	if b.stateVersion == nil {
		b.stateVersion = map[window]int{}
	}
	b.stateVersion[s.window()] = next.version
	return &next
}
