// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

// Recorder accumulates nodes for a [Graph]. It executes nothing.
//
// A Recorder belongs to one goroutine. The [Graph] it builds is immutable and may
// be submitted from several goroutines at once.
//
// Each recorded node declares the resources it reads and writes. Dependency edges
// are inferred from those declarations rather than from the order calls were
// made, which is what lets the builder compute barriers correctly and overlap
// independent work, and what makes a missing dependency a validation error
// instead of a race.
type Recorder struct{ _ noCopy }

// Dispatch records a compute dispatch.
func (r *Recorder) Dispatch(p *ComputePipeline, b []Binding, count WorkgroupCount) NodeID {
	panic(ErrNotImplemented)
}

// DispatchIndirect records a dispatch whose workgroup count is read from a buffer
// written on the device.
//
// This is how anything data-dependent is expressed, and it sits awkwardly with an
// immutable graph. See specs/003-command-graph.md, which lists it as unresolved.
func (r *Recorder) DispatchIndirect(p *ComputePipeline, b []Binding, count BufferView) NodeID {
	panic(ErrNotImplemented)
}

// CopyToBuffer records a host-to-device transfer.
func (r *Recorder) CopyToBuffer(dst BufferView, src any) NodeID { panic(ErrNotImplemented) }

// CopyBuffer records a device-to-device buffer copy.
func (r *Recorder) CopyBuffer(dst, src BufferView) NodeID { panic(ErrNotImplemented) }

// CopyTextureToBuffer records an on-device copy from a texture into a buffer.
//
// This is what lets a rasterized G-buffer feed a compute pass without going out
// to the host and back. Readback follows caller row order regardless of the
// backend's native origin; see docs/conventions.md.
func (r *Recorder) CopyTextureToBuffer(dst BufferView, src *Texture) NodeID {
	panic(ErrNotImplemented)
}

// CopyBufferToTexture records an on-device copy from a buffer into a texture.
func (r *Recorder) CopyBufferToTexture(dst *Texture, src BufferView) NodeID {
	panic(ErrNotImplemented)
}

// Transient reserves a buffer whose lifetime the builder owns.
//
// Transients are the memory the builder may alias: it computes each one's live
// range across the graph and packs those that do not overlap into shared storage.
// Buffers the caller created are never aliased. This is why a model can run
// without allocating per operation.
func (r *Recorder) Transient(desc BufferDescriptor) BufferView { panic(ErrNotImplemented) }

// Build validates the recorded nodes, plans transient memory, computes barriers,
// and lowers the result for the target device.
//
// Every check lives here: type and dtype agreement at every binding, workgroup
// sizes within device limits, resources large enough for declared access,
// capabilities present, no unresolvable hazard, no cycle.
//
// This is the cost the recording model accepts: errors surface at Build rather
// than at the call that caused them. An error therefore names the node, the
// binding slot, and the source position of the recording call. An error that says
// only "type mismatch" is a defect.
func (r *Recorder) Build() (*Graph, error) { panic(ErrNotImplemented) }

// NodeID identifies a recorded node, for referring to it in errors.
type NodeID int

// Graph is validated, planned work that can be submitted many times.
//
// A Graph is immutable. Immutability is the point: it means validation, memory
// planning, barrier computation, and lowering happen once rather than per
// submission. Between submissions only three things may vary: buffer contents,
// which resource is bound to a declared slot, and dynamic dispatch counts.
// Anything else is a different graph.
//
// A Graph is safe for concurrent submission. Submissions are not implicitly
// ordered with respect to each other; wait on a [Fence] if ordering matters.
type Graph struct{ _ noCopy }

// Rebind points a declared binding slot at a different resource. The new resource
// must match the slot's type, dtype, and access.
func (g *Graph) Rebind(b []Binding) error { panic(ErrNotImplemented) }

// Memory reports what the graph needs to run: its transient pool size and its
// peak usage.
//
// Callers need this before submitting, to size a KV cache or decide how many
// layers fit in device memory.
func (g *Graph) Memory() GraphMemory { panic(ErrNotImplemented) }

// Close releases the graph.
func (g *Graph) Close() error { panic(ErrNotImplemented) }

// GraphMemory reports a graph's memory requirement.
type GraphMemory struct {
	// TransientBytes is what the builder needs for transients after aliasing.
	TransientBytes int

	// UnaliasedBytes is what they would have needed without it. The gap is what
	// planning bought.
	UnaliasedBytes int

	PeakBytes int
}

// Queue accepts submitted work.
type Queue struct{ _ noCopy }

// Submit submits a graph and returns immediately with a [Fence]. Nothing in this
// API blocks implicitly.
func (q *Queue) Submit(g *Graph) *Fence { panic(ErrNotImplemented) }

// SubmitAfter submits a graph that begins only once every given fence has
// signalled.
func (q *Queue) SubmitAfter(g *Graph, after ...*Fence) *Fence { panic(ErrNotImplemented) }

// Run records a one-use graph, submits it, and waits.
//
// It exists for readability in simple cases and carries the full cost of building
// a graph every call, so it is the wrong choice in a hot loop.
func (q *Queue) Run(record func(*Recorder)) error { panic(ErrNotImplemented) }

// Fence reports the completion of a submission.
type Fence struct{ _ noCopy }

// Wait blocks until the submission completes, reporting its error if it failed.
func (f *Fence) Wait() error { panic(ErrNotImplemented) }

// Done reports whether the submission has completed, without blocking.
func (f *Fence) Done() bool { panic(ErrNotImplemented) }

// C returns a channel closed when the submission completes, for selecting on it.
func (f *Fence) C() <-chan struct{} { panic(ErrNotImplemented) }
