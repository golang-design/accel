// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package cpu

import (
	"fmt"
	"sync"
	"unsafe"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
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
	// stats is the last submission's indirect counters, when the plan asked for
	// them. One slice reused, because only one submission runs at a time.
	stats []driver.IndirectStat

	cur *fence

	// scratch is the resolved node list, reused across submissions.
	//
	// specs/003-command-graph.md is explicit that no backend validates, plans,
	// or *allocates* per submission -- that is the plan-once saving the whole
	// model exists for -- and a fresh slice per Submit is exactly that
	// allocation. Only one submission runs at a time, so one buffer suffices.
	scratch []resolvedNode
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
	e.stats = e.stats[:0]
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

	// dispatch and args are OpDispatch's payload and the argument set built
	// from its bindings. The slices in args alias device memory rather than
	// copying it, which is what makes a kernel write where the graph said it
	// would.
	dispatch *driver.Dispatch
	args     kernel.Args

	// subgroupSize and diagnostics come from the device's options, so a graph
	// submitted in developer mode is checked and one in strict mode is not.
	subgroupSize uint32
	diagnostics  bool

	// indirect is the count read from the device at submission, already
	// clamped, or nil for a direct dispatch. indirectStats is where the actual
	// and the clamp flag are recorded when a caller asked for them.
	indirect      *kernel.ID3
	indirectSrc   []byte
	indirectMax   kernel.ID3
	indirectStats *driver.IndirectStat
}

func (e *executable) resolve() ([]resolvedNode, error) {
	if cap(e.scratch) < len(e.plan.Nodes) {
		e.scratch = make([]resolvedNode, len(e.plan.Nodes))
	}
	out := e.scratch[:len(e.plan.Nodes)]
	for i := range e.plan.Nodes {
		n := &e.plan.Nodes[i]
		var dst []byte
		if n.Op != driver.OpDispatch {
			var err error
			dst, err = e.bytes(n.Dst)
			if err != nil {
				return nil, fmt.Errorf("accel: node %d destination: %w", n.ID, err)
			}
		}
		r := resolvedNode{op: n.Op, dst: dst, data: n.Data, id: n.ID}
		if n.Op == driver.OpCopy {
			src, err := e.bytes(n.Src)
			if err != nil {
				return nil, fmt.Errorf("accel: node %d source: %w", n.ID, err)
			}
			r.src = src
		}
		if n.Op == driver.OpDispatch {
			if err := e.resolveDispatch(&r, n); err != nil {
				return nil, err
			}
			if ind := n.Dispatch.Indirect; ind != nil {
				raw, err := e.bytes(ind.Count)
				if err != nil {
					return nil, fmt.Errorf("accel: node %d indirect count: %w", n.ID, err)
				}
				r.indirectSrc = raw
				r.indirectMax = ind.Max
				if e.plan.CollectStats {
					e.stats = append(e.stats, driver.IndirectStat{Node: n.ID})
					r.indirectStats = &e.stats[len(e.stats)-1]
				}
			}
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

// resolveDispatch turns a dispatch's binding operands into the typed slices its
// generated entry point takes.
//
// The slices alias device memory. They are not copies: a kernel writing its
// output must write where the graph said it would, and a copy back afterwards
// would be a second definition of what a binding means.
func (e *executable) resolveDispatch(r *resolvedNode, n *driver.PlanNode) error {
	d := n.Dispatch
	r.dispatch = d
	slices := make([]any, len(d.Bindings))
	for i, o := range d.Bindings {
		raw, err := e.bytes(o)
		if err != nil {
			return fmt.Errorf("accel: node %d binding %q: %w", n.ID, d.Kernel.Bindings[i].Name, err)
		}
		s, err := typedSlice(d.Kernel.Bindings[i].DType, raw)
		if err != nil {
			return fmt.Errorf("accel: node %d binding %q: %w", n.ID, d.Kernel.Bindings[i].Name, err)
		}
		slices[i] = s
	}
	r.args = kernel.Args{Slices: slices, Uniforms: d.Uniforms}
	r.subgroupSize = e.dev.subgroupSizeU32()
	r.diagnostics = e.dev.diagnostics
	return nil
}

// typedSlice reinterprets device bytes as the element type a binding declares.
//
// A reinterpretation rather than a conversion: the bytes are the device's
// representation already, and spec 001 section 3.5 requires host and device to
// share byte order, which is checked where a transfer enters rather than per
// binding here. Alignment holds because a pool's suballocations are aligned to
// at least the binding alignment, which is far above any element's.
func typedSlice(dt kernel.DType, b []byte) (any, error) {
	switch dt {
	case kernel.U8:
		return b, nil
	case kernel.I8:
		return reinterpret[int8](b)
	case kernel.F16:
		return reinterpret[kernel.Float16](b)
	case kernel.BF16:
		return reinterpret[kernel.BFloat16](b)
	case kernel.I32:
		return reinterpret[int32](b)
	case kernel.U32:
		return reinterpret[uint32](b)
	case kernel.F32:
		return reinterpret[float32](b)
	}
	return nil, fmt.Errorf("%v is not a binding element type", dt)
}

// reinterpret views bytes as elements of T.
//
// The narrow storage types go through here too: each is a struct wrapping a
// uint16 and therefore has that layout exactly, which is the property that lets
// a device store them without the host converting.
//
// A byte length that is not a whole number of elements is refused rather than
// truncated. Truncation would hide a binding whose range was computed with the
// wrong element size, and the kernel would run happily over one element fewer
// than the caller believes it bound.
func reinterpret[T any](b []byte) (any, error) {
	var zero T
	size := int(unsafe.Sizeof(zero))
	if len(b)%size != 0 {
		return nil, fmt.Errorf("%d bytes is not a whole number of %d-byte elements", len(b), size)
	}
	if len(b) == 0 {
		return []T(nil), nil
	}
	return unsafe.Slice((*T)(unsafe.Pointer(&b[0])), len(b)/size), nil
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
		case driver.OpDispatch:
			if err := dispatch(n); err != nil {
				return err
			}
		default:
			return fmt.Errorf("accel: node %d has operation %v", n.id, n.op)
		}
	}
	return nil
}

// readIndirect reads a device-supplied workgroup count and clamps it.
//
// # Why the clamp is unconditional
//
// specs/003-command-graph.md is explicit: every build mode clamps on device
// before the indirect fetch, because correctness cannot depend on a debug flag
// and no backend may submit an out-of-range count -- which on Vulkan is
// undefined behaviour rather than a clean error. What a caller gives up by not
// collecting statistics is being *told* that a clamp happened, not being
// protected from one.
//
// # Why zero is not normalized
//
// A host-authored maximum normalizes an omitted Y or Z to one, because a caller
// writing WorkgroupCount{X: n} means one of each. The three values read from a
// buffer are the device's, and specs/003-command-graph.md says a zero in any of
// them skips the dispatch. Normalizing here would turn a deliberate skip into a
// single workgroup.
func readIndirect(src []byte, max kernel.ID3) (count kernel.ID3, clamped bool) {
	raw := kernel.ID3{
		X: le32(src[0:4]),
		Y: le32(src[4:8]),
		Z: le32(src[8:12]),
	}
	count = raw
	for _, c := range []struct {
		v   *uint32
		lim uint32
	}{{&count.X, max.X}, {&count.Y, max.Y}, {&count.Z, max.Z}} {
		if *c.v > c.lim {
			*c.v = c.lim
			clamped = true
		}
	}
	return count, clamped
}

// le32 reads a little-endian uint32, which is the layout every target writes an
// indirect count in and which specs/001-device-resources.md section 3.5 makes a
// requirement on the host rather than a conversion here.
func le32(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

// skipsDispatch reports a count with a zero axis, which specs/003-command-graph.md
// defines as a skip rather than as a single workgroup.
func skipsDispatch(c kernel.ID3) bool { return c.X == 0 || c.Y == 0 || c.Z == 0 }

// dispatch runs one kernel, turning a panic inside it into an error.
//
// A kernel that indexes past a binding is a kernel bug, and on a GPU it is an
// out-of-bounds access the hardware clamps or leaves undefined. On this backend
// it is a Go panic, which without this would take the caller's process down
// from inside a goroutine they did not start and cannot recover in.
//
// specs/006-backends.md section 5 makes this backend the oracle: its job is to
// fail loudly where another backend would silently do something else. Loudly
// means a reported error naming the kernel, not an abort.
func dispatch(n *resolvedNode) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("accel: kernel %q panicked at node %d: %v; on a GPU backend this "+
				"would be an out-of-bounds access with undefined results rather than a "+
				"crash, so it is a kernel bug either way",
				n.dispatch.Kernel.Name, n.id, r)
		}
	}()
	// A kernel has exactly one entry point, chosen by whether its body reaches
	// a barrier, shared memory, or a subgroup operation. Dispatching a
	// cooperative kernel through the flat path would run its invocations one
	// after another, which is a different program rather than a slower one.
	k := n.dispatch.Kernel
	count := n.dispatch.Count
	if n.indirectSrc != nil {
		var clamped bool
		raw := kernel.ID3{
			X: le32(n.indirectSrc[0:4]),
			Y: le32(n.indirectSrc[4:8]),
			Z: le32(n.indirectSrc[8:12]),
		}
		count, clamped = readIndirect(n.indirectSrc, n.indirectMax)
		if n.indirectStats != nil {
			n.indirectStats.Actual = raw
			n.indirectStats.Max = n.indirectMax
			n.indirectStats.Clamped = clamped
		}
		if skipsDispatch(count) {
			return nil
		}
	}
	if k.Cooperative != nil {
		return kernel.DispatchCooperativeWith(k, count, n.args,
			kernel.Options{SubgroupSize: n.subgroupSize, Diagnostics: n.diagnostics})
	}
	return kernel.Dispatch(k, count, n.args)
}

// IndirectStats reports the last completed submission's counts.
//
// It is a separate interface rather than a method every executable carries,
// because a backend that cannot produce them should be discovered by assertion
// rather than by a method that fails when called
// (specs/006-backends.md section 1).
func (e *executable) IndirectStats() []driver.IndirectStat {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]driver.IndirectStat, len(e.stats))
	copy(out, e.stats)
	return out
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
