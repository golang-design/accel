// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"errors"
	"fmt"
	"reflect"
	"sync"

	"golang.design/x/accel/internal/driver"
)

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
type Recorder struct {
	_ noCopy

	state recorderState
}

// Dispatch records a compute dispatch.
func (r *Recorder) Dispatch(p *ComputePipeline, b []Binding, u []UniformValue, count WorkgroupCount) NodeID {
	return r.dispatchImpl(p, b, u, count)
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
func (r *Recorder) DispatchIndirect(p *ComputePipeline, b []Binding, u []UniformValue, count BufferView, max WorkgroupCount) NodeID {
	return r.indirectImpl(p, b, u, count, max)
}

// UploadToBuffer records a host-to-device transfer, copying src at record time.
//
// The copy happens here rather than at submit because a Graph is immutable, and
// holding the caller's slice would make the bytes a submission writes depend on
// when the caller last touched them. The graph therefore owns those bytes, its
// build-time footprint includes them, and every submission rewrites the same
// values. That makes this the wrong entry point for bulk upload, which wants a
// staging buffer and [Recorder.CopyBuffer], and for anything varying per
// submission, which wants an Upload buffer written between submissions. It is
// here for small constants baked into a graph.
func (r *Recorder) UploadToBuffer(dst BufferView, src any) NodeID {
	a, ok := r.declare("UploadToBuffer", dst, AccessWrite)
	if !ok {
		return r.node(NodeHostWrite, "UploadToBuffer", nil, nil)
	}
	data, err := hostBytes("UploadToBuffer", dst.Buffer.desc.Label, dst.DType, src)
	if err != nil {
		r.state.errs = append(r.state.errs, err)
		return r.node(NodeHostWrite, "UploadToBuffer", nil, nil)
	}
	if len(data) != a.size {
		r.fail("UploadToBuffer on %q: the view holds %d bytes and src has %d",
			dst.Buffer.desc.Label, a.size, len(data))
		return r.node(NodeHostWrite, "UploadToBuffer", nil, nil)
	}
	// Copied now, because a Graph is immutable and holding the caller's slice
	// would make what a submission writes depend on when they last touched it.
	owned := make([]byte, len(data))
	copy(owned, data)

	id := r.node(NodeHostWrite, "UploadToBuffer", []access{a}, owned)
	return id
}

// UploadToSlot records a host-to-device transfer into a slot's eventual
// resource, over its first count elements from offset.
//
// It exists because [Recorder.UploadToBuffer] takes a view and a slot has no
// resource to make a view of. Splitting it out rather than overloading a
// BufferView with an optional slot keeps a view a thing that names bytes.
func (r *Recorder) UploadToSlot(dst Slot, offset, count int, src any) NodeID {
	d, ok := r.slotDescriptor(dst)
	if !ok {
		r.fail("UploadToSlot: slot %d was not declared by this recorder", int(dst))
		return r.node(NodeHostWrite, "UploadToSlot", nil, nil)
	}
	data, err := hostBytes("UploadToSlot", d.Name, d.DType, src)
	if err != nil {
		r.state.errs = append(r.state.errs, err)
		return r.node(NodeHostWrite, "UploadToSlot", nil, nil)
	}
	a, ok := r.slotAccess("UploadToSlot", dst, offset*d.DType.Size(), count*d.DType.Size(), AccessWrite)
	if !ok {
		return r.node(NodeHostWrite, "UploadToSlot", nil, nil)
	}
	if len(data) != a.size {
		r.fail("UploadToSlot on %q: the range holds %d bytes and src has %d",
			d.Name, a.size, len(data))
		return r.node(NodeHostWrite, "UploadToSlot", nil, nil)
	}
	owned := make([]byte, len(data))
	copy(owned, data)

	id := r.node(NodeHostWrite, "UploadToSlot", []access{a}, owned)
	return id
}

// CopyBuffer records a device-to-device buffer copy.
func (r *Recorder) CopyBuffer(dst, src BufferView) NodeID {
	return r.copy("CopyBuffer", r.operandOf("CopyBuffer", dst, AccessWrite), r.operandOf("CopyBuffer", src, AccessRead))
}

// CopyFromSlot records a device-to-device copy whose source arrives before
// submission.
func (r *Recorder) CopyFromSlot(dst BufferView, src Slot, offset, count int) NodeID {
	d, _ := r.slotDescriptor(src)
	return r.copy("CopyFromSlot",
		r.operandOf("CopyFromSlot", dst, AccessWrite),
		r.slotOperand("CopyFromSlot", src, offset*d.DType.Size(), count*d.DType.Size(), AccessRead))
}

// CopyToSlot records a device-to-device copy whose destination arrives before
// submission.
func (r *Recorder) CopyToSlot(dst Slot, offset, count int, src BufferView) NodeID {
	d, _ := r.slotDescriptor(dst)
	return r.copy("CopyToSlot",
		r.slotOperand("CopyToSlot", dst, offset*d.DType.Size(), count*d.DType.Size(), AccessWrite),
		r.operandOf("CopyToSlot", src, AccessRead))
}

// copy records a two-operand copy once both ends have been declared.
//
// Both ends are declared before either is checked for success, so a call with
// two bad operands reports two errors rather than the first one.
func (r *Recorder) copy(op string, dst, src declared) NodeID {
	if !dst.ok || !src.ok {
		return r.node(NodeCopyBuffer, op, nil, nil)
	}
	if dst.a.size != src.a.size {
		r.fail("%s: the destination holds %d bytes and the source %d", op, dst.a.size, src.a.size)
		return r.node(NodeCopyBuffer, op, nil, nil)
	}
	// Destination first, so that node.accesses[0] is always the write. Build
	// reads them positionally and a reordering here would silently swap a copy.
	id := r.node(NodeCopyBuffer, op, []access{dst.a, src.a}, nil)
	return id
}

// declared is one operand plus whether declaring it succeeded, so that a caller
// can declare both ends before deciding.
type declared struct {
	a  access
	ok bool
}

func (r *Recorder) operandOf(op string, v BufferView, mode Access) declared {
	a, ok := r.declare(op, v, mode)
	return declared{a: a, ok: ok}
}

func (r *Recorder) slotOperand(op string, s Slot, off, size int, mode Access) declared {
	a, ok := r.slotAccess(op, s, off, size, mode)
	return declared{a: a, ok: ok}
}

// CopyTextureToBuffer records an on-device copy from a texture into a buffer.
//
// This is what lets a rasterized G-buffer feed a compute pass without going out
// to the host and back. Readback follows caller row order regardless of the
// backend's native origin; see docs/conventions.md.
func (r *Recorder) CopyTextureToBuffer(dst BufferView, src *Texture) NodeID {
	return r.textureCopy("CopyTextureToBuffer", dst, src, true)
}

// CopyBufferToTexture records an on-device copy from a buffer into a texture.
func (r *Recorder) CopyBufferToTexture(dst *Texture, src BufferView) NodeID {
	return r.textureCopy("CopyBufferToTexture", src, dst, false)
}

// Transient reserves a buffer whose lifetime the builder owns.
//
// Transients are the memory the builder may alias: it computes each one's live
// range across the graph and packs those that do not overlap into shared storage.
// Buffers the caller created are never aliased. This is why a model can run
// without allocating per operation.
func (r *Recorder) Transient(desc BufferDescriptor) BufferView {
	return r.transientImpl(desc)
}

// Slot declares a binding point whose resource is supplied before submission
// rather than at record time.
//
// Slots are what the three-kinds-of-variation rule means by a bound resource
// changing: a swapchain image that does not exist until it is acquired, one
// sequence's KV cache selected per submission, an adapter swapped between runs.
// The descriptor carries everything the builder needs in order to validate and to
// infer hazards without a resource in hand, which is why MinCount and Access are
// there rather than being discovered at bind.
func (r *Recorder) Slot(desc SlotDescriptor) Slot { return r.slotImpl(desc) }

// UseTransientPool makes this graph plan its intermediates into a caller-owned
// pool rather than allocating its own.
//
// Several graphs may share one pool, which is what makes a set of plans over one
// model cost one plan's transients rather than all of them: five prefill buckets
// at 200 MiB is a gigabyte of device memory of which 800 MiB is idle.
//
// The rule that makes it safe is one a graph already has, widened to the set:
// graphs sharing a pool cannot be in flight together, and a second submission is
// refused rather than queued -- a pool that queued silently would turn a design
// mistake into a latency mystery.
//
// See [TransientPool] and specs/031-shared-transients.md.
func (r *Recorder) UseTransientPool(p *TransientPool) { r.state.shared = p }

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

	// Format constrains a texture slot. FormatInvalid accepts any format the
	// recorded nodes accept.
	Format Format
}

// Slot names a rebindable binding point within one graph.
//
// The zero value is not a slot: ids start at one, so a [Binding] that forgot to
// set Slot is a validation error rather than a silent reference to the first one.
type Slot int

// CollectRunStats makes the graph carry back the counters only the device knows:
// an indirect node's actual count and whether it was clamped. It is off by
// default because it adds a readback buffer, a transfer, and a barrier to every
// submission.
func (r *Recorder) CollectRunStats(on bool) { r.state.collectStats = on }

// NodeID identifies a recorded node, for referring to it in errors.
type NodeID int

// NodeKind identifies the public operation family represented by a graph node.
type NodeKind uint8

const (
	NodeDispatch NodeKind = iota
	NodeDispatchIndirect
	NodeRenderPass
	NodeCopyBuffer
	NodeCopyTextureToBuffer
	NodeCopyBufferToTexture
	NodeHostWrite
)

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
type Graph struct {
	_ noCopy

	dev *Device

	nodes      []recNode
	slots      []SlotDescriptor
	transients []*transient

	// present carries what a present slot records beyond a plain one: the
	// surface a frame must come from, the generation, and the extent the graph
	// was built against. specs/034-surface-present.md section 2.
	present map[Slot]presentSlot

	pool driver.Block

	// shared is the caller-owned pool this graph planned into, or nil when it
	// owns its transients. specs/031-shared-transients.md.
	shared *TransientPool

	plan *driver.Plan
	exe  driver.Executable

	memory   GraphMemory
	barriers int
	hazards  int

	// collectStats is whether this graph carries back the counters only the
	// device knows, which cost a readback and are therefore opt-in.
	collectStats bool

	// succ and pred are the inferred dependency DAG, and reach is its transitive
	// closure packed as reachWords uint64s per node.
	succ, pred [][]NodeID
	reach      []uint64
	reachWords int

	// barriersBefore is the plan: entry i is the barrier emitted immediately
	// before node i, or nil.
	barriersBefore []*barrier

	// poolAlign is the alignment every transient placement respects.
	poolAlign int

	// naive reports that this graph was built under the conservative plan. See
	// [Recorder.BuildNaive].
	naive bool

	state resourceState

	// concrete, slotWriter and spans are V24's inputs, computed once at Build
	// because nothing in them varies per rebind. See [Graph.checkOverlaps].
	concrete   []span
	slotWriter []bool // by one-based slot: does any node write through it
	spans      []span // scratch, reused per rebind; guarded by mu

	mu       sync.Mutex
	bound    []SlotBinding // by one-based slot; index 0 unused
	inFlight bool
}

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
func (g *Graph) Bind(b ...SlotBinding) error { return g.bindAll(b) }

// SlotBinding supplies one graph slot with the resource it names.
//
// Its own type, because the bind path and the dispatch path check different
// things: a slot's descriptor is what a bound resource has to satisfy, and a
// pipeline's binding layout is what a dispatch has to fill. [Binding] used to
// serve both, which meant Bind required Slot *and* Buffer and never read Index
// — the exact opposite of the one-of-three rule Binding's own documentation
// states. See specs/036-documentation.md's freeze record.
type SlotBinding struct {
	Slot   Slot
	Buffer BufferView
}

// GraphSlot pairs a discoverable graph slot ID with its descriptor.
type GraphSlot struct {
	Slot       Slot
	Descriptor SlotDescriptor
}

// Slots reports the stable IDs and descriptors a graph expects, so a caller
// holding a graph they did not record can bind its inputs.
func (g *Graph) Slots() []GraphSlot {
	out := make([]GraphSlot, len(g.slots))
	for i, d := range g.slots {
		out[i] = GraphSlot{Slot: Slot(i + 1), Descriptor: d}
	}
	return out
}

// NodeStats reports what the builder decided about one node, and Nodes reports
// all of them. Both are valid as soon as Build returns and are identical for
// every submission: these are the plan, not a measurement, so they cost nothing.
func (g *Graph) NodeStats(id NodeID) NodeStats {
	if id < 0 || int(id) >= len(g.nodes) {
		return NodeStats{Node: id}
	}
	n := &g.nodes[id]
	before := 0
	if g.barriersBefore[id] != nil {
		before = 1
	}
	return NodeStats{Node: n.id, Kind: n.kind, Label: n.label, BarriersBefore: before}
}

func (g *Graph) Nodes() []NodeStats {
	out := make([]NodeStats, len(g.nodes))
	for i := range g.nodes {
		out[i] = g.NodeStats(NodeID(i))
	}
	return out
}

// Barriers is how many barriers the plan emits in total.
//
// It is reported separately from the per-node count because the number a reader
// wants first is the whole-graph one. It is far below [Graph.Hazards] because a
// barrier is queue-wide: one emitted for a hazard on one resource also orders
// every earlier write on every other.
func (g *Graph) Barriers() int { return g.barriers }

// Hazards is how many read-after-write, write-after-write, and write-after-read
// dependencies the declared accesses imply.
//
// Reported alongside [Graph.Barriers] because the gap between them is what
// batching bought, and a caller asking why a graph does not overlap wants both
// numbers rather than either alone.
func (g *Graph) Hazards() int { return g.hazards }

// Edges reports the inferred dependency DAG as one successor list per node, in
// record order within each list.
//
// It is exposed because a plan is the thing worth asserting on: a test that
// only compares results cannot tell a graph that overlapped correctly from one
// that serialized and got the same answer.
func (g *Graph) Edges() [][]NodeID {
	out := make([][]NodeID, len(g.succ))
	for i, s := range g.succ {
		out[i] = append([]NodeID(nil), s...)
	}
	return out
}

// NodeStats is one node's plan-time facts.
type NodeStats struct {
	Node  NodeID
	Kind  NodeKind
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

// TransientPlacement is where one builder-owned intermediate landed in the
// graph's transient pool.
type TransientPlacement struct {
	Label string

	// Offset and Bytes are its extent in the pool. Two placements sharing bytes
	// is aliasing, and it is only sound when every user of one is ordered
	// against every user of the other.
	Offset int
	Bytes  int

	// Users are the nodes that touch it, in record order. Reported because the
	// interference relation quantifies over exactly this set, so a caller
	// checking a placement by hand needs it.
	Users []NodeID
}

// TransientPlacement reports the pool layout the builder chose.
//
// It is exposed because a placement is the thing worth asserting on: an output
// comparison can pass on a backend that executes serially while the layout is
// unsound, since such a backend cannot observe the race the layout would create
// on one that overlaps. See specs/017-graph-aliasing.md.
func (g *Graph) TransientPlacement() []TransientPlacement {
	out := make([]TransientPlacement, len(g.transients))
	for i, t := range g.transients {
		out[i] = TransientPlacement{
			Label:  t.buf.desc.Label,
			Offset: t.offset,
			Bytes:  alignUp(t.bytes, g.poolAlign),
			Users:  append([]NodeID(nil), t.users...),
		}
	}
	return out
}

// Memory reports what the graph needs to run: its transient pool size and its
// peak usage.
//
// Callers need this before submitting, to size a KV cache or decide how many
// layers fit in device memory.
func (g *Graph) Memory() GraphMemory { return g.memory }

// Close releases the graph, including the transient memory it owns.
//
// It fails while a submission is in flight rather than freeing memory the
// device is reading.
func (g *Graph) Close() error {
	g.mu.Lock()
	if g.inFlight {
		g.mu.Unlock()
		return &LifetimeError{Op: "Close", Resource: "graph", Reason: reasonInFlight}
	}
	g.mu.Unlock()

	if !g.state.beginClose() {
		return nil
	}
	if g.exe != nil {
		if err := g.exe.Close(); err != nil {
			return err
		}
	}
	g.releaseTransients()
	g.dev.countGraphs(-1)
	return nil
}

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
type Queue struct {
	_ noCopy

	dev  *Device
	info QueueInfo

	mu      sync.Mutex
	pending []pendingWrite
	stats   QueueStats

	// prev is closed when the last unit of work put on this queue finishes. It
	// is what makes the queue a serial stream: see [Queue.enqueue].
	prev chan struct{}
}

// ReadTexture flushes this queue's pending writes, waits for prior work, and
// returns the base mip and sole array layer as tightly packed top-origin rows.
func (q *Queue) ReadTexture(src *Texture, into []byte) error {
	return q.readTexture(src, into)
}

// Submit submits a graph and returns immediately with a [Fence]. Nothing in this
// API blocks implicitly.
func (q *Queue) Submit(g *Graph) *Fence {
	if fail := q.reject(g); fail != nil {
		return fail
	}
	// The pending host writes join the same stream rather than being flushed
	// alongside it. Flushing here would let a write race a submission already
	// running, which is exactly what the queue's ordering guarantee forbids.
	q.mu.Lock()
	batch := q.pending
	q.pending = nil
	q.stats.Submissions++
	q.mu.Unlock()

	// The counters travel on the fence rather than on the graph, because a
	// caller holding a fence is the one who knows which submission produced
	// them: a graph submitted twice would otherwise report the second's. They
	// are written before the fence signals, since Done becoming true is exactly
	// the promise that everything the fence carries is complete.
	return q.enqueue(func(f *Fence) error {
		if err := runWrites(batch); err != nil {
			return err
		}
		if err := g.run(); err != nil {
			return err
		}
		f.state.stats = g.readStats()
		return nil
	})
}

// FailedFence returns a fence that has already failed with err.
//
// It exists for layers above this one. A tensor plan validates its bindings
// before submitting, and specs/007-tensor-layer.md requires that failure to
// arrive the same way every other submission failure does -- through the fence
// -- so that a caller checks one thing rather than a second return value they
// will forget. Without this, every layer above would either invent its own
// two-value convention or reach into this package.
//
// A nil err is a programming mistake rather than a success, and says so.
func FailedFence(err error) *Fence {
	if err == nil {
		err = errors.New("accel: FailedFence with no error")
	}
	f := newFence()
	f.state.err = err
	f.signal()
	return f
}

// reject reports a submission that cannot be attempted at all, as a fence that
// has already failed. Returning an error instead would make Submit the one
// entry point a caller has to check twice.
func (q *Queue) reject(g *Graph) *Fence {
	var err error
	switch {
	case g == nil:
		err = errors.New("accel: Submit: nil graph")
	case g.dev != q.dev:
		err = errors.New("accel: Submit: the graph belongs to a different device")
	default:
		return nil
	}
	f := newFence()
	f.state.err = err
	f.signal()
	return f
}

// SubmitAfter submits a graph that begins only once every given fence has
// signalled.
func (q *Queue) SubmitAfter(g *Graph, after ...*Fence) *Fence {
	if fail := q.reject(g); fail != nil {
		return fail
	}
	// The wait happens on the queue's own stream rather than before enqueuing,
	// so that submission order still follows call order: waiting first would
	// let a later call to Submit overtake this one, which is the guarantee spec
	// 003 makes about a single queue.
	//
	// Two submissions to *one* queue are already ordered, so this is for
	// ordering against another queue's work — which at v0 is a shape no device
	// reports, since every one reports a single queue. It is built because the
	// API promises it and an unbuilt promise is worse than an absent one.
	q.mu.Lock()
	batch := q.pending
	q.pending = nil
	q.stats.Submissions++
	q.mu.Unlock()

	return q.enqueue(func(f *Fence) error {
		for _, w := range after {
			if w == nil {
				continue
			}
			if err := w.Wait(); err != nil {
				return err
			}
		}
		if err := runWrites(batch); err != nil {
			return err
		}
		if err := g.run(); err != nil {
			return err
		}
		f.state.stats = g.readStats()
		return nil
	})
}

// Run records a one-use graph, submits it, and waits.
//
// It exists for readability in simple cases and carries the full cost of building
// a graph every call, so it is the wrong choice in a hot loop.
func (q *Queue) Run(record func(*Recorder)) error {
	r := q.dev.NewRecorder()
	record(r)
	g, err := r.Build()
	if err != nil {
		return err
	}
	defer g.Close()
	return q.Submit(g).Wait()
}

// QueueStats are cumulative since device open.
type QueueStats struct {
	Submissions    int64
	BytesStaged    int64
	StagingWaits   int64 // times WriteBuffer blocked waiting for a recycled block
	ImmediateReads int64
	Repacks        int64 // immediate-path texture copies that needed a padded pitch
}

// Fence reports the completion of a submission.
type Fence struct {
	_ noCopy

	state *fenceState
}

// Stats reports what the device counted during this submission. It is valid only
// after the fence has signalled, and calling it earlier is an error rather than a
// stale read.
//
// Collection is off unless the graph asked for it with [Recorder.CollectRunStats],
// because the numbers are written by the device and reading them back costs a
// transfer, a barrier, and a Readback allocation. A graph that did not ask still
// clamps an indirect count against its recorded maximum; what it loses is being
// told that it did.
func (f *Fence) Stats() (SubmissionStats, error) {
	if !f.Done() {
		// An error rather than a stale read. The counters are written by the
		// device during execution, so reading them early does not return an
		// approximation, it returns whatever the last submission left there.
		return SubmissionStats{}, errors.New("accel: Fence.Stats: the submission has not " +
			"completed; the counters are written during execution, so reading them now " +
			"would report the previous submission's")
	}
	if f.state.err != nil {
		return SubmissionStats{}, f.state.err
	}
	return f.state.stats, nil
}

// SubmissionStats is what one submission's device-written counters reported.
type SubmissionStats struct {
	Indirect []IndirectStats
}

// IndirectStats is one indirect node's actual count and whether it was clamped.
type IndirectStats struct {
	Node    NodeID
	Actual  [3]uint32 // device-supplied count before clamping
	Max     [3]uint32
	Clamped bool // Actual exceeded Max on at least one axis
}

// SetUniform replaces one by-value parameter of a recorded dispatch.
//
// # Why a graph can be told this after it is built
//
// A kernel's by-value parameters are compiled into the plan when the graph is
// built, which makes them fast and makes them fixed. Most of them should be:
// specs/007-tensor-layer.md draws the line at whether the value changes the
// *shape* of the work, and one that does needs another plan because the
// barriers and the transient layout were computed from it.
//
// A value that changes nothing structural is different. A softmax scale, a RoPE
// base, a current sequence length: these vary every step and rebuilding a graph
// for each would defeat the point of building one. So they are set here, between
// submissions, exactly as a slot is rebound.
//
// It is refused while a submission is in flight, for the reason [Graph.Bind] is:
// a value changing under a running graph would give the first half of it one
// number and the second half another, and no caller could tell which they got.
//
// The type must be the one the kernel declares. A struct of the same shape and a
// different name would encode identically and read correctly, which is why the
// check is on the type rather than on the size: the pair that encodes the same
// today diverges the first time either gains a field.
func (g *Graph) SetUniform(n NodeID, index int, v any) error {
	if err := g.state.checkOpen("SetUniform"); err != nil {
		return err
	}
	if n < 0 || int(n) >= len(g.nodes) {
		return fmt.Errorf("accel: SetUniform: node %d of %d", n, len(g.nodes))
	}
	node := &g.nodes[n]
	if node.pipeline == nil {
		return fmt.Errorf("accel: SetUniform: node %d is a %v, and only a dispatch has "+
			"by-value parameters", n, node.kind)
	}
	if index < 0 || index >= len(node.uniforms) {
		return fmt.Errorf("accel: SetUniform: node %d takes %d by-value parameters and %d "+
			"was named", n, len(node.uniforms), index)
	}
	if have, want := reflect.TypeOf(v), reflect.TypeOf(node.uniforms[index]); have != want {
		return fmt.Errorf("accel: SetUniform: node %d parameter %d is %v and %v was given; "+
			"a different type of the same shape would encode identically and diverge the "+
			"first time either gained a field", n, index, want, have)
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	if g.inFlight {
		return &LifetimeError{Op: "SetUniform", Resource: "graph", Reason: reasonInFlight}
	}
	// The plan's dispatch shares this slice, so writing here is what the next
	// submission encodes. That sharing is deliberate and load-bearing; a copy
	// would make this silently do nothing.
	node.uniforms[index] = v
	return nil
}
