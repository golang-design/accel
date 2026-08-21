// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

// Recorder accumulates nodes for a [Graph]. It executes nothing.
//
// A Recorder belongs to one goroutine. The [Graph] it builds is immutable, but
// immutability does not make it concurrently submittable: see [Graph] for the
// one-submission-in-flight rule and why its transient pool requires it.
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
// This is how anything data-dependent is expressed, and a wholly device-decided
// count would leave an immutable graph with nothing to validate and nothing to
// size transients against. So the node also records max, a build-time upper bound
// checked against the device's workgroup count limit. The device supplies the
// actual count and it is clamped to max: on device in strict mode, and as a
// documented caller obligation otherwise.
func (r *Recorder) DispatchIndirect(p *ComputePipeline, b []Binding, count BufferView, max WorkgroupCount) NodeID {
	panic(ErrNotImplemented)
}

// CopyToBuffer records a host-to-device transfer, copying src at record time.
//
// The copy happens here rather than at submit because a Graph is immutable, and
// holding the caller's slice would make the bytes a submission writes depend on
// when the caller last touched them. The graph therefore owns those bytes, its
// build-time footprint includes them, and every submission rewrites the same
// values. That makes this the wrong entry point for bulk upload, which wants a
// staging buffer and [Recorder.CopyBuffer], and for anything varying per
// submission, which wants an Upload buffer written between submissions. It is
// here for small constants baked into a graph.
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

// Slot declares a binding point whose resource is supplied before submission
// rather than at record time.
//
// Slots are what the three-kinds-of-variation rule means by a bound resource
// changing: a swapchain image that does not exist until it is acquired, one
// sequence's KV cache selected per submission, an adapter swapped between runs.
// The descriptor carries everything the builder needs in order to validate and to
// infer hazards without a resource in hand, which is why MinCount and Access are
// there rather than being discovered at bind.
func (r *Recorder) Slot(desc SlotDescriptor) Slot { panic(ErrNotImplemented) }

// SlotDescriptor declares a rebindable binding point.
type SlotDescriptor struct {
	Name   string // appears in every error about this slot
	Kind   BindingKind
	DType  DType // for buffer kinds; a bound view must match exactly
	Access Access

	// MinCount is the smallest bound range the recorded nodes can be given, in
	// elements of DType. It is the size check moved from build to bind, because at
	// build there is no buffer to measure.
	MinCount int

	// Format constrains a texture slot. The zero value accepts any format the
	// recorded nodes accept.
	Format Format
}

// Slot names a rebindable binding point within one graph.
//
// The zero value is not a slot: ids start at one, so a [Binding] that forgot to
// set Slot is a validation error rather than a silent reference to the first one.
type Slot int

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

// CollectRunStats makes the graph carry back the counters only the device knows:
// an indirect node's actual count and whether it was clamped. It is off by
// default because it adds a readback buffer, a transfer, and a barrier to every
// submission.
func (r *Recorder) CollectRunStats(on bool) { panic(ErrNotImplemented) }

// NodeID identifies a recorded node, for referring to it in errors.
type NodeID int

// Graph is validated, planned work that can be submitted many times.
//
// A Graph is immutable. Immutability is the point: it means validation, memory
// planning, barrier computation, and lowering happen once rather than per
// submission. Between submissions only three things may vary: buffer contents,
// which resource is bound to a declared slot, and dynamic dispatch or draw
// counts. Anything else is a different graph.
//
// Note that a per-step address is none of those three and travels as buffer
// contents. A KV cache write offset, for example, is passed as a value a kernel
// reads rather than by rebinding a view, because rebinding would cost a binding
// update per layer per step.
//
// A Graph may have only one submission in flight at a time. That is narrower
// than immutability suggests, and the reason is memory planning: its transients
// are aliased into a single pool, so two overlapping submissions would write
// each other's intermediates, and a rebind between two in-flight submissions
// races on which one sees it. To run the same work concurrently, build a graph
// per concurrent user: they share pipelines and caller-owned buffers, and only
// the transient pool is duplicated. Wait on a [Fence] if you need ordering.
type Graph struct{ _ noCopy }

// Bind points one slot at a resource. Rebind does several at once and is the
// hot-path form.
//
// Both validate kind, dtype, access, size, device ownership and liveness, and
// then check that no two slots have been bound to overlapping ranges unless both
// are read-only. That last check cannot happen at build, because hazards are
// inferred against the slot rather than against whatever will occupy it: bind one
// buffer to two slots the builder treated as independent and the inferred edge set
// is wrong, which is a missing barrier and therefore a race. A batch is rejected
// as a batch rather than half applied.
//
// Binding while a submission is in flight reports ErrGraphInFlight, for the same
// reason submitting twice does.
func (g *Graph) Bind(b Binding) error     { panic(ErrNotImplemented) }
func (g *Graph) Rebind(b []Binding) error { panic(ErrNotImplemented) }

// Slots reports what a graph expects, so a caller holding a graph they did not
// record can discover its inputs.
func (g *Graph) Slots() []SlotDescriptor { panic(ErrNotImplemented) }

// NodeStats reports what the builder decided about one node, and Nodes reports
// all of them. Both are valid as soon as Build returns and are identical for
// every submission: these are the plan, not a measurement, so they cost nothing.
func (g *Graph) NodeStats(id NodeID) NodeStats { panic(ErrNotImplemented) }
func (g *Graph) Nodes() []NodeStats            { panic(ErrNotImplemented) }

// NodeStats is one node's plan-time facts.
type NodeStats struct {
	Node  NodeID
	Label string

	// Copy is non-nil for copy nodes. Whether a texture copy repacks is decided at
	// build, not observed at run time, because the backend knows its own pitch
	// rules before anything executes.
	Copy *CopyStats

	// BarriersBefore is how many barriers the builder emits immediately before this
	// node. It is here so a caller can ask why a graph does not overlap, and so the
	// builder's own tests can assert on the plan rather than on results.
	BarriersBefore int
}

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

// Stats reports cumulative queue counters since device open. They are counters,
// not a profiler: nothing here is per node and nothing here costs a readback.
func (q *Queue) Stats() QueueStats { panic(ErrNotImplemented) }

// QueueStats are cumulative since device open.
type QueueStats struct {
	Submissions    int64
	BytesStaged    int64
	StagingWaits   int64 // times a Buffer.Write blocked waiting for a recycled block
	ImmediateReads int64
	Repacks        int64 // immediate-path texture copies that needed a padded pitch
}

// Fence reports the completion of a submission.
type Fence struct{ _ noCopy }

// Wait blocks until the submission completes, reporting its error if it failed.
func (f *Fence) Wait() error { panic(ErrNotImplemented) }

// Done reports whether the submission has completed, without blocking.
func (f *Fence) Done() bool { panic(ErrNotImplemented) }

// C returns a channel closed when the submission completes, for selecting on it.
func (f *Fence) C() <-chan struct{} { panic(ErrNotImplemented) }

// Stats reports what the device counted during this submission. It is valid only
// after the fence has signalled, and calling it earlier is an error rather than a
// stale read.
//
// Collection is off unless the graph asked for it with [Recorder.CollectRunStats],
// because the numbers are written by the device and reading them back costs a
// transfer, a barrier, and a Readback allocation. A graph that did not ask still
// clamps an indirect count against its recorded maximum; what it loses is being
// told that it did.
func (f *Fence) Stats() (SubmissionStats, error) { panic(ErrNotImplemented) }

// SubmissionStats is what one submission's device-written counters reported.
type SubmissionStats struct {
	Indirect []IndirectStats
}

// IndirectStats is one indirect node's actual count and whether it was clamped.
type IndirectStats struct {
	Node    NodeID
	Actual  [3]uint32
	Max     [3]uint32
	Clamped bool
}
