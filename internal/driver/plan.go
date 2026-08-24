// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package driver

import (
	"errors"
	"fmt"

	"golang.design/x/accel/internal/kernel"
)

// A Plan is a built graph, resolved down to bytes and handed to a backend.
//
// # Why the plan crosses the seam at all
//
// The alternative is for the layer above to walk its own node list per
// submission and call backend primitives one at a time. That works on the CPU
// and forecloses every backend that has a resubmittable command object: a
// Vulkan primary command buffer, a D3D12 closed command list, and a Metal
// indirect command buffer are all built once from a whole plan and cannot be
// assembled from a stream of unrelated calls. See specs/006-backends.md R7.
//
// What a backend receives is therefore already validated, already ordered,
// already barriered, and already assigned offsets. It decides how to replay it
// and nothing else. The CPU backend replays the node list directly, which
// specs/006-backends.md section 4.5 states is enough: category (1), plan-once,
// is backend-independent and is most of the value.
type Plan struct {
	// CollectStats makes the executable record each indirect node's actual
	// count. Off by default: see [IndirectStat].
	CollectStats bool

	// Nodes are in execution order. That order is a topological order of the
	// dependency DAG the layer above inferred, so a backend that executes them
	// in sequence is always correct and a backend that overlaps consecutive
	// nodes is correct exactly where no barrier separates them.
	Nodes []PlanNode

	// Slots is how many rebindable operands the plan references, so an
	// implementation can size its resolution table without scanning.
	Slots int

	// Transients is the block backing the plan's builder-owned intermediates.
	// It is nil when the graph declared none.
	Transients Block

	Label string
}

// PlanOp is what one node does.
type PlanOp uint8

const (
	// OpInvalid is the zero value, and is rejected. A node whose op was never
	// set would otherwise be a copy of nothing that a backend executes happily.
	OpInvalid PlanOp = iota

	// OpCopy moves bytes from Src to Dst, both on the device.
	OpCopy

	// OpHostWrite writes Data into Dst. The bytes belong to the plan and are
	// rewritten on every submission, which is what makes a graph carrying a
	// small constant replayable.
	OpHostWrite

	// OpDispatch runs a compiled kernel over a grid of workgroups.
	OpDispatch

	// OpRenderPass rasterizes a set of draws into a set of attachments.
	//
	// One op for the whole pass rather than one per draw, because the pass is
	// the unit at which synchronisation is expressible: no backend can barrier
	// inside one, and a per-draw op would let a planner promise an ordering the
	// hardware cannot provide. See specs/033-render-api.md.
	OpRenderPass

	// OpCopyRows moves a rectangle whose two sides have different row pitches,
	// which is what a texture-buffer copy is: the device pads rows to its own
	// alignment and the accel API boundary does not.
	//
	// A separate op rather than a flag on OpCopy, because the sizes mean
	// different things: a plain copy's operands are equal-length ranges, and
	// these two are equal *rectangles* whose byte lengths differ.
	OpCopyRows
)

func (o PlanOp) String() string {
	switch o {
	case OpCopy:
		return "copy"
	case OpHostWrite:
		return "host write"
	case OpDispatch:
		return "dispatch"
	case OpCopyRows:
		return "row copy"
	}
	return "invalid"
}

// PlanNode is one unit of work.
type PlanNode struct {
	Op PlanOp

	Dst Operand
	Src Operand // unset for OpHostWrite

	// Data is OpHostWrite's payload, owned by the plan.
	Data []byte

	// Dispatch is OpDispatch's payload.
	Dispatch *Dispatch

	// Rows is OpCopyRows's payload.
	Rows *RowCopy

	// Render is OpRenderPass's payload.
	Render *RenderPass

	// BarrierBefore asks for every prior write to be visible before this node
	// runs. A backend with real barriers emits one; a backend that executes
	// serially may ignore it, which is why the CPU backend is not a check on
	// barrier placement and specs/006-backends.md section 5 says so.
	BarrierBefore bool

	// ID is the node's identity in the graph it came from, carried so a backend
	// diagnostic can name the node a caller recorded rather than an index into
	// this slice.
	ID int
}

// Indirect is a dispatch whose workgroup count the device supplies.
//
// # Why a build-time maximum
//
// A device-written count sits awkwardly with an immutable graph: without a
// bound there is nothing to validate at build and nothing to size transients
// against, and exceeding a backend's workgroup count limit is undefined
// behaviour on Vulkan rather than a clean error. So the node records a maximum,
// the device supplies the actual, and the backend clamps.
//
// **Every build mode clamps**, not only a debug one. Correctness cannot depend
// on a flag, and no backend may submit an out-of-range indirect count. What a
// debug mode adds is being *told* that a clamp happened, which costs a
// readback. See specs/003-command-graph.md.
type Indirect struct {
	// Count names the three uint32 workgroup counts in a device buffer.
	Count Operand

	// Max is the build-time upper bound, already normalized: omitted Y and Z
	// are one, and X is positive.
	Max kernel.ID3
}

// IndirectStat is what one indirect node's count turned out to be.
//
// Reported only when a caller asked, because reading it back costs a transfer,
// a barrier and an allocation. The *clamp* happens either way: what a caller
// gives up by not asking is being told, not being protected.
type IndirectStat struct {
	Node    int
	Actual  kernel.ID3 // what the device supplied, before clamping
	Max     kernel.ID3
	Clamped bool
}

// Dispatch is what one compute node runs.
//
// The kernel is a generated record rather than source: specs/006-backends.md R5
// says a backend does not compile Go, it consumes what the kernel compiler
// produced for its target. For this backend that artifact is the generated Go
// lowering the record carries.
type Dispatch struct {
	Kernel *kernel.Kernel

	// Count is the workgroup count, not the invocation count.
	Count kernel.ID3

	// Bindings are the resource ranges, one per entry of the kernel's binding
	// layout and in that order. Position is the contract: the kernel indexes its
	// arguments by layout index, so a reordering here silently swaps two
	// buffers.
	Bindings []Operand

	// Uniforms are the by-value parameters, in signature order.
	Uniforms []any

	// Indirect is non-nil when the workgroup count comes from a buffer. Count
	// on the Dispatch is then the maximum rather than the actual.
	Indirect *Indirect
}

// RowCopy describes a rectangle whose two sides are pitched differently.
//
// The row length is what actually moves; the two pitches are how far to step on
// each side. Where they are equal this degenerates to one contiguous copy,
// which is the common case and costs nothing extra to express this way.
type RowCopy struct {
	Rows     int
	RowBytes int
	DstPitch int
	SrcPitch int
}

// OperandKind distinguishes bytes known at build from bytes supplied later.
type OperandKind uint8

const (
	// OperandUnset is the zero value and is never valid. It exists so that a
	// half-built operand is a rejected plan rather than a node that reads zero
	// bytes from a nil block and reports success.
	OperandUnset OperandKind = iota

	// OperandBlock names bytes inside a block known when the graph was built.
	OperandBlock

	// OperandSlot names bytes inside whatever is bound to a slot before
	// submission. The block is resolved by [Executable.Rebind].
	OperandSlot
)

// Operand names the bytes one node touches.
//
// Its fields are unexported and it is built only through [BlockOperand] and
// [SlotOperand], because the two cases are mutually exclusive and nothing in a
// struct with both a block field and a slot field says so. A plan node holding
// an operand that set neither would be a copy that silently moves nothing, and
// silently moving nothing is the failure this design spends most of its
// validation budget avoiding.
type Operand struct {
	kind   OperandKind
	block  Block
	slot   int
	offset int
	size   int
}

// BlockOperand names a concrete range. It fails rather than returning a
// half-formed operand, so an invalid one cannot reach a plan.
func BlockOperand(b Block, offset, size int) (Operand, error) {
	if b == nil {
		return Operand{}, errors.New("accel: a block operand needs a block")
	}
	if err := checkRange(offset, size, b.Size()); err != nil {
		return Operand{}, err
	}
	return Operand{kind: OperandBlock, block: b, offset: offset, size: size}, nil
}

// SlotOperand names a range within a slot's eventual resource. Slot indices
// start at one, so the zero value is not a slot.
func SlotOperand(slot, offset, size int) (Operand, error) {
	if slot < 1 {
		return Operand{}, fmt.Errorf("accel: slot indices start at 1, got %d", slot)
	}
	// The bound resource is unknown, so only the shape is checkable here. The
	// range against the actual block is checked at Rebind, which is the earliest
	// point it can be.
	if err := checkRange(offset, size, -1); err != nil {
		return Operand{}, err
	}
	return Operand{kind: OperandSlot, slot: slot, offset: offset, size: size}, nil
}

func checkRange(offset, size, limit int) error {
	if offset < 0 || size < 0 {
		return fmt.Errorf("accel: operand range [%d, %d) is negative", offset, size)
	}
	if limit >= 0 && (offset > limit || size > limit-offset) {
		return fmt.Errorf("accel: operand range [%d, %d) is outside a %d-byte block", offset, offset+size, limit)
	}
	return nil
}

// Kind reports which case this operand is.
func (o Operand) Kind() OperandKind { return o.kind }

// Block is the concrete block, or nil for a slot operand.
func (o Operand) Block() Block { return o.block }

// Slot is the one-based slot index, or zero for a block operand.
func (o Operand) Slot() int { return o.slot }

// Offset is the byte offset within the block or the bound resource.
func (o Operand) Offset() int { return o.offset }

// Size is how many bytes the node touches.
func (o Operand) Size() int { return o.size }

func (o Operand) String() string {
	switch o.kind {
	case OperandBlock:
		return fmt.Sprintf("block[%d:%d]", o.offset, o.offset+o.size)
	case OperandSlot:
		return fmt.Sprintf("slot %d[%d:%d]", o.slot, o.offset, o.offset+o.size)
	}
	return "unset operand"
}

// SlotBinding resolves one slot to a concrete range.
type SlotBinding struct {
	// Slot is one-based, matching [SlotOperand].
	Slot int

	// Block is the resource the slot now names, and Offset is where the slot's
	// operand offsets are measured from.
	Block  Block
	Offset int

	// Size is how many bytes are available from Offset, which is what a slot
	// operand's range is checked against.
	Size int
}

// Executable is a compiled plan a backend can resubmit.
//
// It is separate from [Plan] because the plan is what the layer above computed
// and the executable is what a backend made of it, and on Vulkan those are a
// struct and a command buffer. See specs/006-backends.md section 4.
// StatsReporter is the optional interface an executable implements when its
// plan asked for run-time counters.
//
// Optional and discovered by assertion, the way specs/006-backends.md section 1
// requires absence to be discoverable rather than expressed as a method that
// fails when called.
type StatsReporter interface {
	// IndirectStats reports the last completed submission's counts, in node
	// order.
	IndirectStats() []IndirectStat
}

type Executable interface {
	// Rebind resolves slots. It applies the whole batch or none of it: a
	// partially applied rebind leaves an executable whose slots disagree about
	// which submission they belong to, and no caller can recover from that
	// because they cannot see which half landed.
	Rebind(binds []SlotBinding) error

	// Submit begins the work and returns without waiting.
	Submit() (Fence, error)

	Close() error
}

// Fence reports one submission's completion.
type Fence interface {
	// Wait blocks until the submission completes. It returns [ErrDeviceLost] if
	// the device died, which is the only way a caller learns: a Wait that could
	// report only completion would turn a lost device into a hang. See
	// specs/001-device-resources.md section 7.4.
	Wait() error

	// Done reports without blocking. A lost device is done.
	Done() bool
}

// ErrDeviceLost is returned once a device has stopped being usable.
//
// Loss is sticky and it is reported, never inferred: every subsequent submit
// and every outstanding fence reports it, because a driver reset that produced
// one failure and then appeared to recover would leave a caller running on
// resources whose contents are undefined.
var ErrDeviceLost = errors.New("accel: device lost")

// GraphCompiler is the optional interface a backend implements to turn a plan
// into something resubmittable. Every backend implements it today; it is an
// interface rather than a method on [Device] because specs/006-backends.md
// section 1 requires absence to be discoverable by assertion rather than by a
// method that fails when called.
type GraphCompiler interface {
	Compile(p *Plan) (Executable, error)
}

// HasDestination reports whether the op writes through PlanNode.Dst.
//
// Stated as a list of the ops that have one rather than as an exemption for the
// ones that do not. The exemption form -- "every op but a dispatch" -- was
// correct while a dispatch was the only many-operand op, and it silently
// demanded a Dst of the next such op added. A render pass writes through its
// attachments and a dispatch through its bindings; adding either to this list
// is what a reader has to decide, and forgetting to is now a refusal at the
// list rather than a nil dereference in a backend.
func (o PlanOp) HasDestination() bool {
	switch o {
	case OpCopy, OpHostWrite, OpCopyRows:
		return true
	}
	return false
}

// Validate reports why a plan cannot be compiled.
//
// A backend calls it before doing anything expensive. It checks the invariants
// the type system does not: that every node has an op, that every operand it
// needs is set, and that slot references are within the count the plan
// declared. It is not a re-validation of the graph, which the layer above
// already did against the device's limits.
func (p *Plan) Validate() error {
	if p == nil {
		return errors.New("accel: nil plan")
	}
	if p.Slots < 0 {
		return fmt.Errorf("accel: plan declares %d slots", p.Slots)
	}
	for i := range p.Nodes {
		n := &p.Nodes[i]
		if n.Op == OpInvalid {
			return fmt.Errorf("accel: plan node %d has no operation", i)
		}
		if n.Op.HasDestination() {
			if err := p.checkOperand(i, "destination", n.Dst); err != nil {
				return err
			}
		}
		switch n.Op {
		case OpCopy:
			if err := p.checkOperand(i, "source", n.Src); err != nil {
				return err
			}
			if n.Src.size != n.Dst.size {
				return fmt.Errorf("accel: plan node %d copies %d bytes into %d",
					i, n.Src.size, n.Dst.size)
			}
		case OpHostWrite:
			if n.Src.kind != OperandUnset {
				return fmt.Errorf("accel: plan node %d is a host write and has a source operand", i)
			}
			if len(n.Data) != n.Dst.size {
				return fmt.Errorf("accel: plan node %d writes %d bytes into a %d-byte destination",
					i, len(n.Data), n.Dst.size)
			}
		case OpDispatch:
			if err := p.checkDispatch(i, n); err != nil {
				return err
			}
		case OpCopyRows:
			if err := p.checkOperand(i, "source", n.Src); err != nil {
				return err
			}
			if err := p.checkRows(i, n); err != nil {
				return err
			}
		case OpRenderPass:
			if err := p.checkRenderPass(i, n); err != nil {
				return err
			}
		}
	}
	return nil
}

func (p *Plan) checkRows(node int, n *PlanNode) error {
	r := n.Rows
	if r == nil {
		return fmt.Errorf("accel: plan node %d is a row copy with no payload", node)
	}
	if r.Rows <= 0 || r.RowBytes <= 0 {
		return fmt.Errorf("accel: plan node %d copies %d rows of %d bytes",
			node, r.Rows, r.RowBytes)
	}
	if r.DstPitch < r.RowBytes || r.SrcPitch < r.RowBytes {
		return fmt.Errorf("accel: plan node %d has a %d-byte row and pitches of %d and "+
			"%d: a pitch is the distance between rows and cannot be shorter than one",
			node, r.RowBytes, r.DstPitch, r.SrcPitch)
	}
	for _, c := range []struct {
		what  string
		o     Operand
		pitch int
	}{{"destination", n.Dst, r.DstPitch}, {"source", n.Src, r.SrcPitch}} {
		// The last row needs only its own bytes, not a full pitch: a tightly
		// packed image's final row has no padding after it, and requiring one
		// would refuse a correctly sized buffer.
		need := (r.Rows-1)*c.pitch + r.RowBytes
		if c.o.size < need {
			return fmt.Errorf("accel: plan node %d's %s is %d bytes and %d rows at a "+
				"pitch of %d need %d", node, c.what, c.o.size, r.Rows, c.pitch, need)
		}
	}
	return nil
}

func (p *Plan) checkDispatch(node int, n *PlanNode) error {
	d := n.Dispatch
	if d == nil {
		return fmt.Errorf("accel: plan node %d is a dispatch with no payload", node)
	}
	if d.Kernel == nil {
		return fmt.Errorf("accel: plan node %d dispatches no kernel", node)
	}
	// A kernel has exactly one entry point, chosen by whether its body reaches
	// a barrier, shared memory, or a subgroup operation. Neither being present
	// is an incomplete generated file rather than a kernel of some third kind.
	if d.Kernel.Flat == nil && d.Kernel.Cooperative == nil {
		return fmt.Errorf("accel: plan node %d dispatches %q, which has neither a flat nor "+
			"a cooperative entry point", node, d.Kernel.Name)
	}
	if len(d.Bindings) != len(d.Kernel.Bindings) {
		return fmt.Errorf("accel: plan node %d binds %d resources to %q, which declares %d",
			node, len(d.Bindings), d.Kernel.Name, len(d.Kernel.Bindings))
	}
	for i, o := range d.Bindings {
		// Quoted, because a binding name lands inside prose: "has no in operand"
		// reads as a typo where "has no \"in\" operand" reads as a name.
		if err := p.checkOperand(node, fmt.Sprintf("%q", d.Kernel.Bindings[i].Name), o); err != nil {
			return err
		}
	}
	if ind := d.Indirect; ind != nil {
		if err := p.checkOperand(node, "indirect count", ind.Count); err != nil {
			return err
		}
		// Three uint32s. A shorter range would read past the buffer, and a
		// longer one means the caller believes the format is something else.
		if ind.Count.size != 12 {
			return fmt.Errorf("accel: plan node %d's indirect count is %d bytes, and a "+
				"workgroup count is three uint32s", node, ind.Count.size)
		}
	}
	return nil
}

func (p *Plan) checkOperand(node int, which string, o Operand) error {
	switch o.kind {
	case OperandUnset:
		return fmt.Errorf("accel: plan node %d has no %s operand", node, which)
	case OperandBlock:
		if o.block == nil {
			return fmt.Errorf("accel: plan node %d's %s operand names no block", node, which)
		}
	case OperandSlot:
		if o.slot > p.Slots {
			return fmt.Errorf("accel: plan node %d's %s operand names slot %d of %d",
				node, which, o.slot, p.Slots)
		}
	default:
		return fmt.Errorf("accel: plan node %d's %s operand has kind %d", node, which, o.kind)
	}
	return nil
}

// RenderPass is what a backend needs to rasterize one pass.
//
// The attachments are operands like any other, so the planner places and
// barriers them without knowing they are attachments. What a backend has to
// understand is the draw list and the stage functions, and those are opaque to
// the planner in the same way a kernel's entry point is.
type RenderPass struct {
	Label string

	// Color and Depth are the attachments, as operands.
	Color []Operand
	Depth *Operand

	// ColorLoad and DepthLoad are the load actions. The backend needs the
	// distinction because clear is free on a tiler and a full-screen clear draw
	// is not.
	ColorLoad  []LoadOp
	ColorClear [][4]float32
	DepthLoad  LoadOp
	DepthClear float32

	Width, Height int

	// Draws is the recorded draw list, in order. A backend never reorders it,
	// because blending is order dependent.
	Draws []RenderDraw
}

// RenderDraw is one recorded draw.
type RenderDraw struct {
	// Stage is the compiled pair, opaque to the planner.
	Vertex, Fragment any

	// Fixed-function state, as the public enums encode it.
	Topology  uint8
	FrontFace uint8
	Cull      uint8

	DepthTest    bool
	DepthWrite   bool
	DepthCompare uint8

	Masks []uint8

	VertexCount   int
	InstanceCount int
	FirstVertex   int
	FirstInstance int

	// VertexBuffers is the attribute source, one operand per bound slot, and
	// VertexLayouts says what is packed inside each. They are the same length.
	VertexBuffers []Operand
	VertexLayouts []VertexLayout

	// VertexUniforms and FragmentUniforms are the stages' by-value parameters,
	// one slice per stage because each stage indexes its own from zero.
	VertexUniforms   []any
	FragmentUniforms []any
}

// VertexLayout is what one bound vertex buffer holds.
//
// It is the lowered form of the public VertexBufferLayout: formats become
// component counts, because a backend fetches floats and the format's only
// other job -- validating against the stage -- is done by the time a plan
// exists.
type VertexLayout struct {
	Stride      int
	PerInstance bool
	Attributes  []VertexAttribute
}

// VertexAttribute is one attribute inside a bound vertex buffer.
type VertexAttribute struct {
	Location   int
	Offset     int
	Components int
}

// LoadOp is what happens to an attachment at the start of a render pass.
//
// It lives here rather than in the public package because a backend acts on the
// value and cannot import that package. accel aliases it.
type LoadOp uint8

const (
	// LoadClear clears to a stated value. On a tiler this costs nothing, where
	// a full-screen clear draw costs a full write of tile memory.
	LoadClear LoadOp = iota

	// LoadKeep preserves existing contents, which makes the attachment a read
	// as well as a write.
	LoadKeep

	// LoadDontCare leaves the contents undefined, and carries no data
	// dependency on whatever wrote the attachment last — so the
	// read-after-write edge disappears. The write-after-write edge does not.
	//
	// A backend cannot tell it from LoadKeep by its effect: both leave the
	// bytes alone. What it buys is the missing edge, and that is spent at graph
	// build. This is why the constant is defined once rather than mirrored --
	// a swap between the two is silent everywhere a test could look.
	LoadDontCare
)

// StoreOp is what happens to an attachment at the end of a render pass.
type StoreOp uint8

const (
	// StoreKeep makes the contents readable after the pass.
	StoreKeep StoreOp = iota

	// StoreDiscard leaves them undefined. On a tiler this saves writing a whole
	// depth buffer out to memory every frame.
	StoreDiscard
)

// checkRenderPass reports why a render pass node cannot be compiled.
//
// The attachments are ordinary operands and are checked as such. What is
// specific to a pass is that every draw needs both stages and a positive
// vertex count: a backend that reached a nil stage would have no honest error
// to give, because by then the plan is already accepted.
func (p *Plan) checkRenderPass(node int, n *PlanNode) error {
	r := n.Render
	if r == nil {
		return fmt.Errorf("accel: plan node %d is a render pass with no payload", node)
	}
	if r.Width <= 0 || r.Height <= 0 {
		return fmt.Errorf("accel: plan node %d renders a %dx%d area",
			node, r.Width, r.Height)
	}
	if len(r.Color) == 0 {
		return fmt.Errorf("accel: plan node %d is a render pass with no colour attachments",
			node)
	}
	for i, c := range r.Color {
		if err := p.checkOperand(node, fmt.Sprintf("colour attachment %d", i), c); err != nil {
			return err
		}
	}
	if r.Depth != nil {
		if err := p.checkOperand(node, "depth attachment", *r.Depth); err != nil {
			return err
		}
	}
	if len(r.Draws) == 0 {
		return fmt.Errorf("accel: plan node %d is a render pass with no draws", node)
	}
	for i, d := range r.Draws {
		if d.Vertex == nil || d.Fragment == nil {
			return fmt.Errorf("accel: plan node %d draw %d is missing a stage", node, i)
		}
		if d.VertexCount <= 0 || d.InstanceCount <= 0 {
			return fmt.Errorf("accel: plan node %d draw %d draws %d vertices in %d instances",
				node, i, d.VertexCount, d.InstanceCount)
		}
	}
	return nil
}
