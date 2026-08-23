// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal

import (
	"fmt"
	"sync"
	"unsafe"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
	"golang.design/x/accel/internal/kernelc/emit"
	"golang.design/x/accel/internal/mtl"
)

// Compile turns a plan into something this backend can resubmit.
//
// "Resubmit" means something narrower on Metal than elsewhere, and
// specs/006-backends.md section 4.3 says why: a MTLCommandBuffer is
// single-submit, so there is no command object to build once. What this
// executable holds is the compiled pipelines and the staging buffers, and the
// graph is re-encoded per submission. Encoding is a cheap CPU-side call per
// command, and the plan-once saving was already paid before a plan arrived here.
//
// Everything a submission could fail on that is knowable now fails now: an
// unsupported op, a kernel with no MSL artifact, a workgroup above the
// pipeline's ceiling. A backend that deferred those to Submit would report them
// through a fence, which is the hardest place for a caller to act on them.
func (d *device) Compile(p *driver.Plan) (driver.Executable, error) {
	if err := p.Validate(); err != nil {
		return nil, err
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil, fmt.Errorf("accel: compile %q: the device is closed", p.Label)
	}

	e := &executable{dev: d, plan: p, bound: make([]driver.SlotBinding, p.Slots+1)}
	for i := range p.Nodes {
		n := &p.Nodes[i]
		switch n.Op {
		case driver.OpCopy:
		case driver.OpHostWrite:
			// Staged rather than written into the mapping. A host write has a
			// place in the plan's order, and a memcpy at encode time happens
			// before any GPU work in the same submission -- so a write ordered
			// after a dispatch that produced its destination would land first
			// and be overwritten. Going through the command stream makes the
			// order the plan said it was, and costs one copy on a backend where
			// the copy is a memcpy between two mapped pages.
			stage, err := d.dev.NewBuffer(max(len(n.Data), 1), mtl.StorageShared)
			if err != nil {
				return nil, fmt.Errorf("accel: node %d staging buffer: %w", n.ID, err)
			}
			e.staging = append(e.staging, stage)
			e.stageAt = append(e.stageAt, stagedWrite{node: i, buf: stage})
		case driver.OpDispatch:
			if ind := n.Dispatch.Indirect; ind != nil {
				s, err := d.newIndirectSlot(i, ind.Max)
				if err != nil {
					return nil, fmt.Errorf("accel: node %d indirect clamp: %w", n.ID, err)
				}
				e.indirect = append(e.indirect, s)
			}
			if err := checkUniforms(n.Dispatch); err != nil {
				return nil, fmt.Errorf("accel: node %d: %w", n.ID, err)
			}
			if _, err := d.pipelineFor(n.Dispatch.Kernel); err != nil {
				return nil, fmt.Errorf("accel: node %d: %w", n.ID, err)
			}
		default:
			return nil, fmt.Errorf("accel: node %d is a %v, which the Metal backend does not "+
				"lower at specs/021-metal-bringup.md", n.ID, n.Op)
		}
	}
	return e, nil
}

// pipelineFor compiles a kernel's MSL, once per device.
//
// Keyed by digest, which is what identifies the generated source: two records
// with the same digest were generated from the same input, and compiling the
// text twice would cost milliseconds inside the device compiler for an
// identical result.
//
// The caller holds d.mu.
func (d *device) pipelineFor(k *kernel.Kernel) (*mtl.Pipeline, error) {
	if k.MSL == "" {
		// Never a fallback to the Go lowering. Running the CPU kernel on a
		// device the caller selected specifically would be correct, fast enough
		// to miss, and would mean the GPU was never exercised.
		return nil, fmt.Errorf("kernel %s carries no MSL artifact, so it cannot run on Metal; "+
			"it is outside the subset specs/021-metal-bringup.md section 5 lowers", k.Name)
	}
	if p, ok := d.pipelines[k.Digest]; ok {
		return p, nil
	}
	p, err := d.dev.Compile(k.MSL, k.Name)
	if err != nil {
		return nil, fmt.Errorf("kernel %s: %w", k.Name, err)
	}
	invocations := int(k.WorkgroupSize.X * k.WorkgroupSize.Y * k.WorkgroupSize.Z)
	if invocations > p.MaxTotalThreadsPerThreadgroup {
		p.Close()
		return nil, fmt.Errorf("kernel %s declares a %v workgroup, which is %d invocations, "+
			"above this pipeline's ceiling of %d", k.Name, k.WorkgroupSize,
			invocations, p.MaxTotalThreadsPerThreadgroup)
	}
	if d.pipelines == nil {
		d.pipelines = map[string]*mtl.Pipeline{}
	}
	d.pipelines[k.Digest] = p
	return p, nil
}

type stagedWrite struct {
	node int
	buf  *mtl.Buffer
}

type executable struct {
	dev  *device
	plan *driver.Plan

	staging []*mtl.Buffer
	stageAt []stagedWrite

	// uniformBufs is scratch for one dispatch's encoded blocks, reused across
	// submissions. Guarded by mu with everything else: it is written during
	// encoding, which happens under the same lock that records the fence.
	uniformBufs [][]byte

	// indirect is one slot per indirect node, in plan order.
	indirect []*indirectSlot

	// stats is the last completed submission's indirect counters, rebuilt on
	// each read rather than cached: the buffers are the truth and a cache would
	// be a second one.
	stats []driver.IndirectStat

	mu     sync.Mutex
	bound  []driver.SlotBinding
	closed bool

	// cur is the most recent submission, and "in flight" is derived from it
	// rather than tracked beside it, for the reason the CPU backend records:
	// two pieces that must agree eventually disagree.
	cur *fence
}

func (e *executable) busy() bool { return e.cur != nil && !e.cur.Done() }

// Rebind applies the whole batch or none of it, so a batch with one bad entry
// leaves the previous bindings intact. A caller cannot see which half of a
// partial rebind landed, which makes a partial one unrecoverable.
func (e *executable) Rebind(binds []driver.SlotBinding) error {
	staged := make([]driver.SlotBinding, len(binds))
	for i, b := range binds {
		if b.Slot < 1 || b.Slot > e.plan.Slots {
			return fmt.Errorf("accel: rebind names slot %d of %d", b.Slot, e.plan.Slots)
		}
		if b.Block == nil {
			return fmt.Errorf("accel: rebind of slot %d names no block", b.Slot)
		}
		if _, ok := driver.Unwrap(b.Block).(*block); !ok {
			return fmt.Errorf("accel: rebind of slot %d names a %T, which is not Metal memory",
				b.Slot, b.Block)
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

// resolved names the buffer and offset one operand reduces to.
type resolved struct {
	buf  *mtl.Buffer
	off  int
	size int
}

func (e *executable) operand(o driver.Operand) (resolved, error) {
	switch o.Kind() {
	case driver.OperandBlock:
		// Unwrapped first: a Block may be a handle to another, which is how
		// accel's shared transient pool grows without invalidating operands.
		b, ok := driver.Unwrap(o.Block()).(*block)
		if !ok {
			return resolved{}, fmt.Errorf("%s names a %T, which is not Metal memory", o, o.Block())
		}
		return resolved{buf: b.buf, off: o.Offset(), size: o.Size()}, nil
	case driver.OperandSlot:
		bind := e.bound[o.Slot()]
		if bind.Block == nil {
			return resolved{}, fmt.Errorf("slot %d has no resource bound", o.Slot())
		}
		if o.Offset() > bind.Size || o.Size() > bind.Size-o.Offset() {
			return resolved{}, fmt.Errorf("%s is outside the %d bytes bound to it", o, bind.Size)
		}
		b := driver.Unwrap(bind.Block).(*block)
		return resolved{buf: b.buf, off: bind.Offset + o.Offset(), size: o.Size()}, nil
	}
	return resolved{}, fmt.Errorf("%s", o)
}

// Submit encodes the plan into a fresh command buffer and commits it.
func (e *executable) Submit() (driver.Fence, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, fmt.Errorf("accel: submit: the executable is closed")
	}
	if e.busy() {
		return nil, fmt.Errorf("accel: submit while a submission is in flight")
	}

	// A lost device is reported here rather than through a fence that never
	// signals. specs/001-device-resources.md section 7.4: every subsequent call
	// returns the loss, so nothing waits forever.
	if err := e.dev.Lost(); err != nil {
		return nil, fmt.Errorf("accel: submit %q: %w", e.plan.Label, err)
	}

	cb := e.dev.queue.Begin()
	enc := &pass{cb: cb}
	if err := e.encode(enc); err != nil {
		enc.end()
		cb.Close()
		return nil, err
	}
	enc.end()
	cb.Commit()
	f := &fence{cb: cb, dev: e.dev}
	e.cur = f
	return f, nil
}

func (e *executable) encode(p *pass) error {
	stage := 0
	for i := range e.plan.Nodes {
		n := &e.plan.Nodes[i]

		// A barrier is an encoder boundary. Metal orders one encoder's work
		// against the next one's, so ending the current pass is the barrier
		// Metal has between dispatches. Whether a memory barrier inside one
		// encoder would be enough is specs/023-metal-graph.md's question, and
		// answering it here would be an optimisation with no test behind it.
		if n.BarrierBefore {
			p.end()
		}

		switch n.Op {
		case driver.OpHostWrite:
			for stage < len(e.stageAt) && e.stageAt[stage].node < i {
				stage++
			}
			if stage >= len(e.stageAt) || e.stageAt[stage].node != i {
				return fmt.Errorf("accel: node %d has no staging buffer", n.ID)
			}
			buf := e.stageAt[stage].buf
			dst, err := e.operand(n.Dst)
			if err != nil {
				return fmt.Errorf("accel: node %d destination: %w", n.ID, err)
			}
			copy(buf.Bytes(), n.Data)
			p.blit().Copy(dst.buf, dst.off, buf, 0, len(n.Data))

		case driver.OpCopy:
			// The sizes are not compared here. Plan.Validate already rejects a
			// copy between unequal ranges and a host write larger than its
			// destination, and Compile runs it before anything else, so a check
			// repeated here would be unreachable -- a second statement of a rule
			// that can then drift from the first.
			dst, err := e.operand(n.Dst)
			if err != nil {
				return fmt.Errorf("accel: node %d destination: %w", n.ID, err)
			}
			src, err := e.operand(n.Src)
			if err != nil {
				return fmt.Errorf("accel: node %d source: %w", n.ID, err)
			}
			p.blit().Copy(dst.buf, dst.off, src.buf, src.off, src.size)

		case driver.OpDispatch:
			if err := e.dispatch(p, n); err != nil {
				return err
			}
		}
	}
	return nil
}

func (e *executable) dispatch(p *pass, n *driver.PlanNode) error {
	d := n.Dispatch
	k := d.Kernel

	// A zero in any dimension is a grid with no invocations. Metal rejects it
	// rather than treating it as a no-op, so the node is skipped here, which is
	// what the CPU backend does and therefore what the oracle expects.
	//
	// An indirect node is exempt: its Count is the *maximum* rather than the
	// actual, and the actual is not known on this side of the submission.
	if d.Indirect == nil && (d.Count.X == 0 || d.Count.Y == 0 || d.Count.Z == 0) {
		return nil
	}
	pipe, err := e.dev.pipelineFor(k)
	if err != nil {
		return fmt.Errorf("accel: node %d: %w", n.ID, err)
	}
	if len(d.Bindings) != len(k.Bindings) {
		return fmt.Errorf("accel: node %d supplies %d bindings for a kernel declaring %d",
			n.ID, len(d.Bindings), len(k.Bindings))
	}

	// The clamp runs first and in its own pass, so the indirect buffer the
	// dispatch reads was written by work Metal has ordered before it.
	var slot *indirectSlot
	if d.Indirect != nil {
		for _, s := range e.indirect {
			if e.plan.Nodes[s.node].ID == n.ID {
				slot = s
				break
			}
		}
		if slot == nil {
			return fmt.Errorf("accel: node %d is indirect and has no clamp slot", n.ID)
		}
		// The count's size is not checked here. Plan.Validate already rejects
		// an indirect count that cannot hold three uint32s, and Compile runs it
		// first, so a check repeated here would be unreachable -- a second
		// statement of a rule that can then drift from the first.
		count, err := e.operand(d.Indirect.Count)
		if err != nil {
			return fmt.Errorf("accel: node %d indirect count: %w", n.ID, err)
		}
		if err := e.encodeClamp(p, slot, count); err != nil {
			return err
		}
	}

	enc := p.compute()
	enc.SetPipeline(pipe)
	lens := make([]uint32, len(d.Bindings))
	for i, o := range d.Bindings {
		r, err := e.operand(o)
		if err != nil {
			return fmt.Errorf("accel: node %d binding %d (%s): %w", n.ID, i, k.Bindings[i].Name, err)
		}
		if r.off%elemBytes(k.Bindings[i].DType) != 0 {
			return fmt.Errorf("accel: node %d binding %d (%s) starts at byte %d, which is not a "+
				"multiple of its %s element size", n.ID, i, k.Bindings[i].Name, r.off,
				k.Bindings[i].DType)
		}
		enc.SetBuffer(r.buf, r.off, i)
		// The floor is what makes a binding whose length is not a whole number
		// of elements safe rather than undefined. The layer above rejects one,
		// so this is the second line rather than the first.
		lens[i] = uint32(r.size / elemBytes(k.Bindings[i].DType))
	}
	enc.SetBytes(u32Bytes(lens), emit.MSLLengthsIndex(len(d.Bindings)))

	// The uniform blocks follow the lengths slot, in signature order, which is
	// the layout the emitter fixed and exported so there is one copy of it.
	if len(k.Uniforms) > 0 {
		if cap(e.uniformBufs) < len(k.Uniforms) {
			e.uniformBufs = make([][]byte, len(k.Uniforms))
		}
		bufs := e.uniformBufs[:len(k.Uniforms)]
		if err := e.encodeUniforms(n, bufs); err != nil {
			return err
		}
		for i := range bufs {
			enc.SetBytes(bufs[i], emit.MSLUniformIndex(len(d.Bindings), i))
		}
	}
	threads := mtl.Size{
		Width:  uint64(k.WorkgroupSize.X),
		Height: uint64(k.WorkgroupSize.Y),
		Depth:  uint64(k.WorkgroupSize.Z),
	}
	if slot != nil {
		enc.DispatchIndirect(slot.clamped, 0, threads)
		return nil
	}
	enc.Dispatch(
		mtl.Size{Width: uint64(d.Count.X), Height: uint64(d.Count.Y), Depth: uint64(d.Count.Z)},
		threads)
	return nil
}

// checkUniforms rejects at compile what would otherwise fail per submission.
//
// A record generated before the encoder field existed carries a nil Encode, and
// the answer is to say so by name rather than to bind zeros: a uniform block of
// zeros is a plausible set of parameters, so the kernel would run and compute
// something wrong.
func checkUniforms(d *driver.Dispatch) error {
	k := d.Kernel
	if len(d.Uniforms) != len(k.Uniforms) {
		return fmt.Errorf("%s declares %d by-value parameters and the dispatch supplies %d",
			k.Name, len(k.Uniforms), len(d.Uniforms))
	}
	for i, u := range k.Uniforms {
		if u.Encode == nil {
			return fmt.Errorf("%s parameter %d (%s %s) has no std140 encoder, so this "+
				"kernel was generated before the record carried one; regenerate it",
				k.Name, i, u.Name, u.Type)
		}
		if u.Size <= 0 {
			return fmt.Errorf("%s parameter %d (%s) has an encoded size of %d",
				k.Name, i, u.Name, u.Size)
		}
	}
	return nil
}

// encodeUniforms produces each block's std140 bytes for one submission.
//
// Per submission rather than per compile, because a plan's uniform values are
// part of the dispatch and a caller may rebuild a plan with different ones. The
// buffers are reused across submissions, since the sizes are fixed by the
// layout and reallocating them per submit would allocate on the hot path.
func (e *executable) encodeUniforms(n *driver.PlanNode, into [][]byte) error {
	for i, u := range n.Dispatch.Kernel.Uniforms {
		if len(into[i]) != u.Size {
			into[i] = make([]byte, u.Size)
		}
		if err := u.Encode(into[i], n.Dispatch.Uniforms[i]); err != nil {
			return fmt.Errorf("accel: node %d parameter %d (%s): %w", n.ID, i, u.Name, err)
		}
	}
	return nil
}

func elemBytes(d kernel.DType) int {
	switch d {
	case kernel.F16, kernel.BF16:
		return 2
	case kernel.I8, kernel.U8:
		return 1
	}
	return 4
}

// unsafeU32 views bytes as u32, for reading a statistics buffer back.
func unsafeU32(b []byte) []uint32 {
	if len(b) < 4 {
		return nil
	}
	return unsafe.Slice((*uint32)(unsafe.Pointer(&b[0])), len(b)/4)
}

func u32Bytes(v []uint32) []byte {
	if len(v) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(v)*4)
}

// pass holds whichever encoder is open.
//
// Metal permits one encoder at a time on a command buffer, so switching between
// compute and copy work closes the current one. That closing is also the
// barrier between them, which is why nothing here needs to ask whether a
// barrier is required between a copy and a dispatch: there is always one.
type pass struct {
	cb      *mtl.CommandBuffer
	comp    *mtl.ComputeEncoder
	blitEnc *mtl.BlitEncoder
}

func (p *pass) compute() *mtl.ComputeEncoder {
	if p.blitEnc != nil {
		p.blitEnc.End()
		p.blitEnc = nil
	}
	if p.comp == nil {
		p.comp = p.cb.Compute()
	}
	return p.comp
}

func (p *pass) blit() *mtl.BlitEncoder {
	if p.comp != nil {
		p.comp.End()
		p.comp = nil
	}
	if p.blitEnc == nil {
		p.blitEnc = p.cb.Blit()
	}
	return p.blitEnc
}

func (p *pass) end() {
	if p.comp != nil {
		p.comp.End()
		p.comp = nil
	}
	if p.blitEnc != nil {
		p.blitEnc.End()
		p.blitEnc = nil
	}
}

func (e *executable) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil
	}
	if e.busy() {
		return fmt.Errorf("accel: close while a submission is in flight")
	}
	e.closed = true
	if e.cur != nil {
		e.cur.close()
		e.cur = nil
	}
	for _, s := range e.staging {
		s.Close()
	}
	e.staging = nil
	for _, s := range e.indirect {
		s.close()
	}
	e.indirect = nil
	return nil
}

// IndirectStats reports the last completed submission's counts, in node order.
//
// Read from the device buffers rather than from anything recorded during
// encoding, because the counts are the device's: the host never saw them. A
// caller who has not waited on the fence gets whatever the buffers hold, which
// is why this is documented on driver.StatsReporter as being about the last
// *completed* submission.
func (e *executable) IndirectStats() []driver.IndirectStat {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.plan.CollectStats || len(e.indirect) == 0 {
		return nil
	}
	e.stats = e.stats[:0]
	for _, s := range e.indirect {
		raw := s.stats.Bytes()
		if len(raw) < 16 {
			continue
		}
		v := unsafeU32(raw)
		e.stats = append(e.stats, driver.IndirectStat{
			Node:    e.plan.Nodes[s.node].ID,
			Actual:  kernel.ID3{X: v[0], Y: v[1], Z: v[2]},
			Max:     s.max,
			Clamped: v[3] != 0,
		})
	}
	return e.stats
}

// fence reports one submission's completion.
type fence struct {
	mu     sync.Mutex
	cb     *mtl.CommandBuffer
	dev    *device
	closed bool
}

// Wait blocks until the submission completes, and reports device loss.
//
// The device learns about loss here, because a command buffer's error is the
// only place Metal reports it and this is where that error is read. A caller
// who never waits never finds out, which is the same as everywhere else: an
// unobserved failure is indistinguishable from no failure.
func (f *fence) Wait() error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closed {
		return nil
	}
	f.cb.Wait()
	err := f.cb.Err()
	f.dev.noteSubmissionError(err)
	if lost := f.dev.Lost(); lost != nil {
		return lost
	}
	return err
}

func (f *fence) Done() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.closed || f.cb.Done()
}

func (f *fence) close() {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.closed {
		return
	}
	f.cb.Wait()
	f.cb.Close()
	f.closed = true
}
