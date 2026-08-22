// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package driver

import (
	"errors"
	"fmt"
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
)

func (o PlanOp) String() string {
	switch o {
	case OpCopy:
		return "copy"
	case OpHostWrite:
		return "host write"
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
		if err := p.checkOperand(i, "destination", n.Dst); err != nil {
			return err
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
