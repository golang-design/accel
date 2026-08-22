// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package cpu

import (
	"fmt"
	"sync"

	"golang.design/x/accel/internal/driver"
)

// Compile turns a plan into something this backend can resubmit.
//
// There is no native command object to build, so the executable is the plan
// plus a slot resolution table. specs/006-backends.md section 4.5 says this is
// enough: the saving that matters is plan-once, and it was already paid by the
// time a plan reaches here.
func (d *device) Compile(p *driver.Plan) (driver.Executable, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.lost != nil {
		return nil, fmt.Errorf("accel: compile %q: %w", p.Label, d.lost)
	}
	if d.closed {
		return nil, fmt.Errorf("accel: compile %q: the device is closed", p.Label)
	}
	return &executable{dev: d, plan: p, bound: make([]driver.SlotBinding, p.Slots+1)}, nil
}

// executable is a compiled plan.
//
// Its mutex covers both the slot table and the in-flight flag, together rather
// than separately, because the pair is exactly what must not change underneath
// a submission: a rebind landing between two nodes of one submission would give
// the first half of a graph one resource and the second half another.
type executable struct {
	dev  *device
	plan *driver.Plan

	mu     sync.Mutex
	bound  []driver.SlotBinding // indexed by one-based slot; index 0 unused
	closed bool

	// cur is the most recent submission, and "in flight" is derived from it
	// rather than tracked alongside it.
	//
	// The two-piece version -- a bool cleared by the completion goroutine and a
	// fence signalled by it -- has to keep them in agreement, and it did not: it
	// signalled first, so a caller who waited on the fence and resubmitted was
	// refused by a flag nobody had got round to clearing. Deriving the answer
	// from the fence removes the disagreement rather than ordering it, which is
	// the difference between a fixed race and a narrower one.
	cur *fence
}

// busy reports whether a submission is still running. e.mu is held.
func (e *executable) busy() bool { return e.cur != nil && !e.cur.Done() }

func (e *executable) Rebind(binds []driver.SlotBinding) error {
	// Validated in full before anything is written, so a batch containing one
	// bad entry leaves the previous bindings intact. A half-applied rebind is
	// unrecoverable for a caller, because they cannot see which half landed.
	staged := make([]driver.SlotBinding, len(binds))
	for i, b := range binds {
		if b.Slot < 1 || b.Slot > e.plan.Slots {
			return fmt.Errorf("accel: rebind names slot %d of %d", b.Slot, e.plan.Slots)
		}
		if b.Block == nil {
			return fmt.Errorf("accel: rebind of slot %d names no block", b.Slot)
		}
		if b.Offset < 0 || b.Size < 0 || b.Offset > b.Block.Size() || b.Size > b.Block.Size()-b.Offset {
			return fmt.Errorf("accel: rebind of slot %d names [%d, %d) of a %d-byte block",
				b.Slot, b.Offset, b.Offset+b.Size, b.Block.Size())
		}
		staged[i] = b
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return fmt.Errorf("accel: rebind: the executable is closed")
	}
	if e.busy() {
		return fmt.Errorf("accel: rebind while a submission is in flight")
	}
	for _, b := range staged {
		e.bound[b.Slot] = b
	}
	return nil
}

func (e *executable) Submit() (driver.Fence, error) {
	if err := e.dev.beginSubmission(); err != nil {
		// A lost device is reported here rather than through a fence that never
		// signals. Spec 001 section 7.4: every subsequent call returns the loss,
		// and every outstanding fence is signalled with it, so nothing waits
		// forever.
		return nil, err
	}

	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil, fmt.Errorf("accel: submit: the executable is closed")
	}
	if e.busy() {
		e.mu.Unlock()
		return nil, fmt.Errorf("accel: submit while a submission is in flight")
	}
	// Resolve every operand under the same lock that records the fence, so a
	// rebind racing a submit either fully precedes it or is rejected. There is
	// no window in which half the plan sees the new binding.
	nodes, err := e.resolve()
	if err != nil {
		e.mu.Unlock()
		return nil, err
	}
	f := &fence{done: make(chan struct{})}
	e.cur = f
	e.mu.Unlock()

	go func() {
		f.err = run(nodes)
		if lost := e.dev.Lost(); lost != nil {
			f.err = lost
		}
		close(f.done)
	}()
	return f, nil
}

// resolvedNode is a plan node with every operand reduced to bytes.
type resolvedNode struct {
	op       driver.PlanOp
	dst, src []byte
	data     []byte
	id       int
}

func (e *executable) resolve() ([]resolvedNode, error) {
	out := make([]resolvedNode, len(e.plan.Nodes))
	for i := range e.plan.Nodes {
		n := &e.plan.Nodes[i]
		dst, err := e.bytes(n.Dst)
		if err != nil {
			return nil, fmt.Errorf("accel: node %d destination: %w", n.ID, err)
		}
		r := resolvedNode{op: n.Op, dst: dst, data: n.Data, id: n.ID}
		if n.Op == driver.OpCopy {
			src, err := e.bytes(n.Src)
			if err != nil {
				return nil, fmt.Errorf("accel: node %d source: %w", n.ID, err)
			}
			r.src = src
		}
		out[i] = r
	}
	return out, nil
}

// bytes reduces one operand to the slice it names.
//
// A slot operand's offset is relative to the bound range rather than to the
// block, which is what lets one graph run over a window of a larger allocation
// without the recorded nodes knowing where that window is.
func (e *executable) bytes(o driver.Operand) ([]byte, error) {
	switch o.Kind() {
	case driver.OperandBlock:
		mem, err := backing(o.Block())
		if err != nil {
			return nil, fmt.Errorf("%s: %w", o, err)
		}
		return mem[o.Offset() : o.Offset()+o.Size()], nil
	case driver.OperandSlot:
		b := e.bound[o.Slot()]
		if b.Block == nil {
			return nil, fmt.Errorf("slot %d has no resource bound", o.Slot())
		}
		if o.Offset() > b.Size || o.Size() > b.Size-o.Offset() {
			return nil, fmt.Errorf("%s is outside the %d bytes bound to it", o, b.Size)
		}
		mem, err := backing(b.Block)
		if err != nil {
			return nil, fmt.Errorf("slot %d: %w", o.Slot(), err)
		}
		off := b.Offset + o.Offset()
		return mem[off : off+o.Size()], nil
	}
	return nil, fmt.Errorf("%s", o)
}

// backing is this backend reaching into its own allocation.
//
// Deliberately not [driver.Block.Bytes], which is the *host* mapping and is nil
// for MemoryDevice on every backend including this one: a device-local pool is
// unmappable by contract (specs/006-backends.md section 1), and a transient
// pool is device-local. A backend executing a copy between its own allocations
// is not mapping them, so it uses the concrete type it created.
func backing(b driver.Block) ([]byte, error) {
	blk, ok := b.(*block)
	if !ok {
		return nil, fmt.Errorf("a %T was not allocated by the CPU backend", b)
	}
	if blk.mem == nil {
		return nil, fmt.Errorf("the block has been freed")
	}
	return blk.mem, nil
}

// run executes resolved nodes in order.
//
// In order, and serially: this backend does not overlap independent nodes, so
// it cannot observe a missing barrier. specs/006-backends.md section 5 says so
// explicitly, and it is why the whole-plan oracle compares two *plans* rather
// than trusting execution here to find a barrier bug.
func run(nodes []resolvedNode) error {
	for i := range nodes {
		n := &nodes[i]
		switch n.op {
		case driver.OpCopy:
			copy(n.dst, n.src)
		case driver.OpHostWrite:
			copy(n.dst, n.data)
		default:
			return fmt.Errorf("accel: node %d has operation %v", n.id, n.op)
		}
	}
	return nil
}

func (e *executable) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	if e.busy() {
		return fmt.Errorf("accel: close: a submission is in flight")
	}
	e.closed = true
	return nil
}

// fence reports one submission's completion.
type fence struct {
	done chan struct{}
	err  error
}

func (f *fence) Wait() error {
	<-f.done
	return f.err
}

func (f *fence) Done() bool {
	select {
	case <-f.done:
		return true
	default:
		return false
	}
}
