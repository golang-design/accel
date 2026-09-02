// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package cpu

import (
	"fmt"
	"sync"
	"time"
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

	// elapsed is the last timed submission's duration.
	elapsed time.Duration

	// stats is the last submission's indirect counters, when the plan asked for
	// them. One slice reused, because only one submission runs at a time.
	stats []driver.IndirectStat

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
	// The device counts this submission only once every refusal above is past:
	// LoseAtSubmission names the n-th submission that ran, and a refused one
	// never reached the device.
	//
	// A lost device is reported here rather than through a fence that never
	// signals. Spec 001 section 7.4: every subsequent call returns the loss,
	// and every outstanding fence is signalled with it, so nothing waits
	// forever.
	if err := e.dev.beginSubmission(); err != nil {
		e.mu.Unlock()
		return nil, err
	}
	f := &fence{done: make(chan struct{})}
	e.cur = f
	e.mu.Unlock()

	go func() {
		start := time.Now()
		f.err = run(nodes)
		if lost := e.dev.Lost(); lost != nil {
			f.err = lost
		}
		// The wall clock around the run, and this backend is the one place that
		// is honest rather than a substitute: its "device" is this goroutine,
		// so the time the work took and the time the device took are the same
		// number. A GPU backend must not do this -- there the two differ by
		// queueing and driver work, and reporting one as the other answers the
		// wrong question convincingly.
		if e.plan.CollectTimings {
			e.mu.Lock()
			e.elapsed = time.Since(start)
			e.mu.Unlock()
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

	// render is OpRenderPass's payload, with its attachments already resolved
	// to the device memory they name. The bytes are kept as bytes and paired
	// with the codec their format asks for: the rasterizer works in float32
	// components and the attachment holds whatever its format holds, so the
	// pass converts on the way in and on the way out.
	render      *driver.RenderPass
	colorAttach [][]byte
	colorCodec  []texelCodec

	// vertexTextures and fragmentTextures are each stage's bound textures,
	// per draw and then per slot, decoded to the four floats a fetch returns.
	// vertexUniformBytes and fragmentUniformBytes are each stage's
	// buffer-bound by-value parameters, resolved to bytes here and decoded when
	// the draw runs -- the same split the textures take, for the same reason.
	vertexUniformBytes   [][][]byte
	fragmentUniformBytes [][][]byte

	vertexTextures   [][]boundTexture
	fragmentTextures [][]boundTexture

	// vertexBytes is the attribute source, per draw and then per bound slot.
	// indexBytes is one per draw, nil for a non-indexed one.
	vertexBytes [][][]byte
	indexBytes  [][]byte

	// indirectArgs is one per draw, nil for a direct one.
	indirectArgs [][]byte
	depthAttach  []byte
	depthCodec   texelCodec
	args         kernel.Args
	rows         *driver.RowCopy

	// subgroupSize and diagnostics come from the device's options, so a graph
	// submitted to a device opened with NoDiagnostics runs unchecked and one
	// submitted to any other is checked.
	subgroupSize uint32
	diagnostics  bool

	// shuffleSeed varies the order invocations advance in within an epoch. It
	// comes from the device's options for the same reason the two above do: a
	// graph records what to run, not how the backend chooses to run it.
	shuffleSeed uint64

	// indirectSrc is the device memory the workgroup count is read from when
	// the node runs, or nil for a direct dispatch. It is read then rather than
	// at resolution because an earlier node of the same submission may write
	// it. indirectStats is where the actual count and the clamp flag are
	// recorded when a caller asked for them.
	indirectSrc   []byte
	indirectMax   kernel.ID3
	indirectStats *driver.IndirectStat
}

func (e *executable) resolve() ([]resolvedNode, error) {
	if cap(e.scratch) < len(e.plan.Nodes) {
		e.scratch = make([]resolvedNode, len(e.plan.Nodes))
	}
	out := e.scratch[:len(e.plan.Nodes)]

	// The counters are sized before any node takes a pointer into them. Each
	// indirect node records through a pointer, and a slice that grows after the
	// first pointer is taken leaves that pointer addressing the array the growth
	// abandoned: the node then writes its counters where nothing reads them and
	// the report shows zeros for every node but the last.
	e.stats = e.stats[:0]
	if e.plan.CollectStats {
		indirect := 0
		for i := range e.plan.Nodes {
			if n := &e.plan.Nodes[i]; n.Op == driver.OpDispatch && n.Dispatch.Indirect != nil {
				indirect++
			}
		}
		if cap(e.stats) < indirect {
			e.stats = make([]driver.IndirectStat, 0, indirect)
		}
	}

	for i := range e.plan.Nodes {
		n := &e.plan.Nodes[i]
		var dst []byte
		if n.Op.HasDestination() {
			var err error
			dst, err = e.bytes(n.Dst)
			if err != nil {
				return nil, fmt.Errorf("accel: node %d destination: %w", n.ID, err)
			}
		}
		r := resolvedNode{op: n.Op, dst: dst, data: n.Data, id: n.ID, rows: n.Rows}
		if n.Op == driver.OpCopy || n.Op == driver.OpCopyRows {
			src, err := e.bytes(n.Src)
			if err != nil {
				return nil, fmt.Errorf("accel: node %d source: %w", n.ID, err)
			}
			r.src = src
		}
		if n.Op == driver.OpRenderPass {
			if err := e.resolveRender(&r, n); err != nil {
				return nil, err
			}
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
	r.shuffleSeed = e.dev.shuffleSeed
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
	// A Block may be a handle to another Block -- accel's shared transient pool
	// hands one out so it can grow without invalidating what captured it.
	b = driver.Unwrap(b)
	blk, ok := b.(*block)
	if !ok {
		return nil, fmt.Errorf("a %T was not allocated by the CPU backend", b)
	}
	mem := blk.contents()
	if mem == nil {
		return nil, fmt.Errorf("the block has been freed")
	}
	return mem, nil
}

// run executes resolved nodes in order.
//
// In order, and serially: this backend does not overlap independent nodes, so
// it cannot observe a missing barrier. specs/006-backends.md section 5 says so
// explicitly, and it is why the whole-plan oracle compares two *plans* rather
// than trusting execution here to find a barrier bug.
//
// Nodes, not workgroups. One node's workgroups do run at once, because the
// compute model defines them not to depend on each other and a missing barrier
// between two of them is already undefined everywhere. Two nodes are the case
// a barrier exists to order, so overlapping them here would hide exactly the
// bug this backend is kept serial to expose.
func run(nodes []resolvedNode) error {
	for i := range nodes {
		n := &nodes[i]
		switch n.op {
		case driver.OpCopy:
			copy(n.dst, n.src)
		case driver.OpHostWrite:
			copy(n.dst, n.data)
		case driver.OpCopyRows:
			r := n.rows
			for row := range r.Rows {
				copy(n.dst[row*r.DstPitch:row*r.DstPitch+r.RowBytes],
					n.src[row*r.SrcPitch:row*r.SrcPitch+r.RowBytes])
			}
		case driver.OpDispatch:
			if err := dispatch(n); err != nil {
				return err
			}
		case driver.OpRenderPass:
			if err := renderPass(n); err != nil {
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
//
// It returns the device's count as well as the clamped one, because the
// statistics report both and decoding the bytes a second time at the call site
// would be a second definition of the layout.
func readIndirect(src []byte, max kernel.ID3) (raw, count kernel.ID3, clamped bool) {
	raw = kernel.ID3{
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
	return raw, count, clamped
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
// means a reported error naming the kernel and the line that failed, not an
// abort: the site is captured where the panic is first recovered, which is
// here on the serial path and in the worker on the parallel one.
func dispatch(n *resolvedNode) (err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("accel: kernel %q panicked at node %d: %v; on a GPU backend this "+
				"would be an out-of-bounds access with undefined results rather than a "+
				"crash, so it is a kernel bug either way",
				n.dispatch.Kernel.Name, n.id, kernel.Recovered(r))
		}
	}()
	// A kernel has exactly one entry point, chosen by whether its body reaches
	// a barrier, shared memory, or a subgroup operation. Dispatching a
	// cooperative kernel through the flat path would run its invocations one
	// after another, which is a different program rather than a slower one.
	k := n.dispatch.Kernel
	count := n.dispatch.Count
	if n.indirectSrc != nil {
		var raw kernel.ID3
		var clamped bool
		raw, count, clamped = readIndirect(n.indirectSrc, n.indirectMax)
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
			kernel.Options{SubgroupSize: n.subgroupSize, Diagnostics: n.diagnostics, ShuffleSeed: n.shuffleSeed})
	}
	return kernel.Dispatch(k, count, n.args)
}

// Elapsed reports how long the last timed submission took.
//
// See the note where it is measured: on this backend the device is a goroutine,
// so wall clock is device time rather than a stand-in for it.
func (e *executable) Elapsed() time.Duration {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.elapsed
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

// resolveRender resolves a render pass's attachments to the memory they name.
//
// The slices alias device memory rather than copying it, for the same reason a
// dispatch's bindings do: an attachment written anywhere else is not the
// attachment the graph ordered against.
func (e *executable) resolveRender(r *resolvedNode, n *driver.PlanNode) error {
	rp := n.Render
	if rp == nil {
		return fmt.Errorf("accel: node %d is a render pass with no payload", n.ID)
	}
	r.render = rp
	for i, o := range rp.Color {
		raw, err := e.bytes(o)
		if err != nil {
			return fmt.Errorf("accel: node %d colour attachment %d: %w", n.ID, i, err)
		}
		c, err := codecFor(rp.ColorFormat[i])
		if err != nil {
			return fmt.Errorf("accel: node %d colour attachment %d: %w", n.ID, i, err)
		}
		r.colorAttach = append(r.colorAttach, raw)
		r.colorCodec = append(r.colorCodec, c)
	}
	if rp.Depth != nil {
		raw, err := e.bytes(*rp.Depth)
		if err != nil {
			return fmt.Errorf("accel: node %d depth attachment: %w", n.ID, err)
		}
		c, err := codecFor(rp.DepthFormat)
		if err != nil {
			return fmt.Errorf("accel: node %d depth attachment: %w", n.ID, err)
		}
		r.depthAttach, r.depthCodec = raw, c
	}

	// The attribute bytes, resolved here for the same reason the attachments
	// are: run walks resolved nodes and has no device left to ask.
	for di, d := range rp.Draws {
		var bufs [][]byte
		for bi, o := range d.VertexBuffers {
			raw, err := e.bytes(o)
			if err != nil {
				return fmt.Errorf("accel: node %d draw %d vertex buffer %d: %w",
					n.ID, di, bi, err)
			}
			bufs = append(bufs, raw)
		}
		r.vertexBytes = append(r.vertexBytes, bufs)

		// A stage's texture *bytes*, resolved here and decoded when the draw
		// runs. Decoding here would be decoding too early: resolution happens
		// before any node executes, so a pass fetching what an earlier pass
		// drew would capture the texture as it was before the draw -- black,
		// and silently so.
		vt, err := e.textureBytes(d.VertexTextures)
		if err != nil {
			return fmt.Errorf("accel: node %d draw %d vertex texture: %w", n.ID, di, err)
		}
		ft, err := e.textureBytes(d.FragmentTextures)
		if err != nil {
			return fmt.Errorf("accel: node %d draw %d fragment texture: %w", n.ID, di, err)
		}
		vub, err := e.uniformBytes2(d.VertexUniformBuffers)
		if err != nil {
			return fmt.Errorf("accel: node %d draw %d vertex uniform buffer: %w", n.ID, di, err)
		}
		fub, err := e.uniformBytes2(d.FragmentUniformBuffers)
		if err != nil {
			return fmt.Errorf("accel: node %d draw %d fragment uniform buffer: %w", n.ID, di, err)
		}
		r.vertexUniformBytes = append(r.vertexUniformBytes, vub)
		r.fragmentUniformBytes = append(r.fragmentUniformBytes, fub)

		r.vertexTextures = append(r.vertexTextures, vt)
		r.fragmentTextures = append(r.fragmentTextures, ft)

		var idx []byte
		if d.Indexed {
			raw, err := e.bytes(d.Index)
			if err != nil {
				return fmt.Errorf("accel: node %d draw %d index buffer: %w", n.ID, di, err)
			}
			idx = raw
		}
		r.indexBytes = append(r.indexBytes, idx)

		var args []byte
		if d.Indirect {
			raw, err := e.bytes(d.IndirectArgs)
			if err != nil {
				return fmt.Errorf("accel: node %d draw %d indirect arguments: %w",
					n.ID, di, err)
			}
			args = raw
		}
		r.indirectArgs = append(r.indirectArgs, args)
	}
	return nil
}

// stageTextures decodes one stage's bound textures into the form a fetch reads.
//
// Once per draw rather than once per fetch: a fragment stage fetching per pixel
// would otherwise decode the same texel a thousand times, and the decode is the
// same work every time. The cost is one image-sized slice per bound texture,
// which is what the GPU has resident anyway.
func (e *executable) textureBytes(ts []driver.RenderTexture) ([]boundTexture, error) {
	if len(ts) == 0 {
		return nil, nil
	}
	out := make([]boundTexture, len(ts))
	for i, t := range ts {
		raw, err := e.bytes(t.Operand)
		if err != nil {
			return nil, fmt.Errorf("slot %d: %w", i, err)
		}
		if _, err := codecFor(t.Format); err != nil {
			return nil, fmt.Errorf("slot %d: %w", i, err)
		}
		out[i] = boundTexture{desc: t, raw: raw}
	}
	return out, nil
}

// boundTexture is a bound texture's bytes and the shape they decode with.
//
// The bytes are a live view of device memory rather than a copy, which is what
// lets the decode happen when the draw runs: a pass that fetches what an
// earlier pass wrote sees the write.
type boundTexture struct {
	desc driver.RenderTexture
	raw  []byte
}

// textureCache decodes each bound texture once per pass.
//
// Once per pass rather than once per draw: a pass of N draws sharing one
// texture decoded it N times, and a decode is an image-sized allocation and a
// conversion of every texel. Once is sound because feedback is rejected at
// build -- a texture bound to a stage in a pass cannot be an attachment of the
// same pass -- so its bytes cannot change between two draws of one pass. It is
// still per pass rather than per submission, because an earlier pass of the
// same submission may have drawn it, which is the case decoding at resolution
// got wrong.
//
// Once per draw rather than once per fetch was the earlier bound, and it still
// holds: a fragment stage fetching per pixel would otherwise decode the same
// texel a thousand times for the same answer.
type textureCache struct {
	decoded map[textureKey]kernel.Texture2D

	// decodes counts images decoded, so a test can say "once" rather than
	// infer it from allocations.
	decodes int
}

// textureKey is what makes two bindings the same texture: the same bytes, the
// same extent and pitch, and the same format. The format is part of it because
// the codec is the *view's*: a texture written through a linear view and
// fetched through an sRGB one decodes the way the fetch asked -- which is what a
// texture unit does in fixed function on every target -- and two views of one
// texture with different formats are two decodes.
type textureKey struct {
	data          *byte
	n             int
	format        driver.Format
	width, height int
	pitch         int
}

func (c *textureCache) decode(ts []boundTexture) ([]kernel.Texture2D, error) {
	if len(ts) == 0 {
		return nil, nil
	}
	out := make([]kernel.Texture2D, len(ts))
	for i, t := range ts {
		key := textureKey{
			data: unsafe.SliceData(t.raw), n: len(t.raw), format: t.desc.Format,
			width: t.desc.Width, height: t.desc.Height, pitch: t.desc.Pitch,
		}
		if tex, ok := c.decoded[key]; ok {
			out[i] = tex
			continue
		}
		codec, err := codecFor(t.desc.Format)
		if err != nil {
			return nil, fmt.Errorf("slot %d: %w", i, err)
		}
		texels := make([]float32, t.desc.Width*t.desc.Height*4)
		if err := codec.decodeImage(texels, t.raw, t.desc.Width, t.desc.Height, t.desc.Pitch); err != nil {
			return nil, fmt.Errorf("slot %d: %w", i, err)
		}
		tex := kernel.NewTexture2D(t.desc.Width, t.desc.Height, texels)
		if c.decoded == nil {
			c.decoded = map[textureKey]kernel.Texture2D{}
		}
		c.decoded[key] = tex
		c.decodes++
		out[i] = tex
	}
	return out, nil
}

// uniformBytes2 resolves each bound uniform block to bytes.
//
// Resolved with the rest of the node rather than read when the draw runs, the
// way every other operand is: the bytes a submission reads are the ones bound
// to the graph, and resolving them together is what keeps one submission
// consistent.
func (e *executable) uniformBytes2(ops []driver.Operand) ([][]byte, error) {
	if len(ops) == 0 {
		return nil, nil
	}
	out := make([][]byte, len(ops))
	for i, o := range ops {
		if o.Kind() == driver.OperandUnset {
			continue
		}
		raw, err := e.bytes(o)
		if err != nil {
			return nil, err
		}
		out[i] = raw
	}
	return out, nil
}
