// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"errors"
	"fmt"
	"reflect"

	"golang.design/x/accel/internal/driver"
)

// BuildNaive builds the graph under the conservative plan of
// specs/015-graph-recording.md: nodes in record order, a barrier before each,
// and no transient aliasing.
//
// Nodes in record order with a full barrier between them is correct rather than
// merely safe, and the reason is worth stating: every dependency edge a hazard
// analysis could infer runs from a lower node id to a higher one, because an
// edge exists only where a later node's declared access conflicts with an
// earlier one's. Record order is therefore a topological order of that DAG, and
// a barrier between consecutive nodes covers every read-after-write,
// write-after-read and write-after-write it could classify.
//
// It exists so that an optimized plan has something to be compared against.
// specs/003-command-graph.md defines the whole-plan oracle as executing a graph
// a second time under exactly this plan and comparing results, and any
// disagreement is a planner or barrier bug localized to the builder rather than
// to a kernel, because both sides ran the same kernels over the same inputs.
//
// This is not scaffolding written alongside the optimizer. It is what
// [Recorder.Build] produced before edge inference existed, retained: an oracle
// written after the thing it checks, by whoever just wrote it, is under
// constant pressure to share its reachability code and therefore its mistakes.
//
// It is exported for testing and for a caller who suspects a planning bug. It
// is slower by construction and never the right choice otherwise.
func (r *Recorder) BuildNaive() (*Graph, error) { return r.build(true) }

// Build validates the recorded nodes, infers the dependency edges between them,
// computes the barriers those edges require, packs the transients into
// overlapping memory where their live ranges allow, and lowers the result for
// the device.
//
// The graph it returns is immutable and replayable: what varies between
// submissions is what slots are bound to, the contents of buffers, and
// device-supplied dispatch and draw counts. Nothing else, which is what lets a
// plan be built once and submitted every frame.
//
// A build error is a statement about the recording rather than about the
// device: an operand outside its resource, a slot smaller than the use it is
// bound to, a transient nothing wrote. Every one of them is cheaper here than
// at submission, which is why they are here.
//
// [Recorder.BuildNaive] produces the conservative plan this one is checked
// against.
func (r *Recorder) Build() (*Graph, error) { return r.build(false) }

func (r *Recorder) build(naive bool) (*Graph, error) {
	if r.state.dev == nil {
		return nil, errors.New("accel: Build: this Recorder was not created by Device.NewRecorder")
	}
	if r.state.built {
		return nil, errors.New("accel: Build: this recorder has already been built")
	}
	r.state.built = true

	if err := r.state.dev.state.checkOpen("Build"); err != nil {
		return nil, err
	}
	if lost := r.state.dev.dev.Lost(); lost != nil {
		return nil, lost
	}

	// Every recorded error is reported together. One per call would tell a
	// caller about their first mistake and hide the rest behind a rebuild.
	if err := errors.Join(r.state.errs...); err != nil {
		return nil, err
	}

	g := &Graph{
		dev:            r.state.dev,
		nodes:          r.state.nodes,
		slots:          r.state.slots,
		present:        r.state.present,
		transients:     r.state.transients,
		collectStats:   r.state.collectStats,
		collectTimings: r.state.collectTimings,
		shared:         r.state.shared,
	}
	g.naive = naive
	g.inferEdges()
	if err := g.assertAcyclic(); err != nil {
		return nil, err
	}
	g.reachability()
	if err := g.checkTransientsAreWritten(); err != nil {
		return nil, err
	}
	if naive {
		g.planSerialBarriers()
	} else {
		g.planBarriers()
	}
	// Packing needs the DAG, because compatibility is reachability rather than
	// record-order position, so transients are placed after inference rather
	// than before it.
	if err := g.placeTransients(); err != nil {
		return nil, err
	}
	if !naive {
		g.planHandovers()
	}
	if err := g.lower(); err != nil {
		g.releaseTransients()
		return nil, err
	}
	g.state.init("graph")
	g.bound = make([]SlotBinding, len(g.slots)+1)
	// Computed once. The ranges a graph names through resources it already holds
	// cannot change, and rebuilding them per rebind would put a scan of every
	// node on the path specs/003-command-graph.md promises stays cheap.
	g.concrete = g.concreteSpans()
	g.slotWriter = make([]bool, len(g.slots)+1)
	for s := 1; s <= len(g.slots); s++ {
		g.slotWriter[s] = g.slotWrites(Slot(s))
	}
	g.spans = make([]span, 0, len(g.slots))
	g.dev.countGraphs(1)
	return g, nil
}

// placeTransients assigns each transient a range and allocates the pool.
//
// Consecutive placement, one transient per range, is this milestone's whole
// memory plan. Offsets are still computed and stored rather than implied,
// because specs/017-graph-aliasing.md replaces the assignment rule and nothing
// else, and a version that skipped offsets entirely would make that a larger
// change than it is.
func (g *Graph) placeTransients() error {
	align := g.dev.allocAlignment(BufferStorage | BufferCopySrc | BufferCopyDst)
	g.poolAlign = align
	for _, t := range g.transients {
		g.memory.UnaliasedBytes += alignUp(t.bytes, align)
	}
	total := g.memory.UnaliasedBytes
	if g.naive {
		// No aliasing: consecutive placement, each transient its own bytes. This
		// is half of what makes the naive plan an oracle, the other half being
		// the barrier before every node.
		at := 0
		for _, t := range g.transients {
			t.offset = at
			t.placed = true
			at += alignUp(t.bytes, align)
		}
		total = alignUp(at, align)
	} else {
		total = g.packTransients(align)
	}
	g.memory.TransientBytes = total
	g.memory.PeakBytes = g.peakBytes(align)

	// V20: the planned pool against the device's reported budget. It is checked
	// after packing rather than before, because packing is what decides the
	// number, and reporting the unaliased total here would refuse graphs that
	// fit.
	if budget := g.dev.info.Limits.MaxPoolBytes; budget > 0 && total > budget {
		return fmt.Errorf("accel: Build: the graph's transients need %s after aliasing and %q "+
			"reports a %s pool budget", humanBytes(total), g.dev.info.Name, humanBytes(budget))
	}
	if total == 0 {
		// A graph with no transients takes nothing from a shared pool, so it
		// does not hold one either: it must not claim the pool when it runs,
		// and it must not release a count it never took. Dropping the pool here
		// is what makes "g.pool is non-nil" and "this graph holds the pool" the
		// same fact, which is what releaseTransients keys on.
		g.shared = nil
		return nil
	}

	// A caller-owned pool is grown to fit rather than allocated per graph, and
	// the offsets computed above are relative to it. Two graphs sharing one
	// overlap completely, which is sound because they never run together --
	// see specs/031-shared-transients.md.
	var blk driver.Block
	if g.shared != nil {
		var err error
		blk, err = g.shared.reserve(total, "graph")
		if err != nil {
			return err
		}
	} else {
		var err error
		blk, err = g.dev.dev.Alloc(driver.MemoryDevice, total, "graph transients")
		if err != nil {
			return fmt.Errorf("accel: Build: transient pool: %w", err)
		}
		g.dev.countImplicit(1)
	}
	g.pool = blk
	for _, t := range g.transients {
		t.pool = blk
	}
	return nil
}

// peakBytes is the record-order interval peak: the most transient bytes live at
// any one point if the graph ran strictly in record order.
//
// It is a lower bound on what any planner can achieve and it needs no
// reachability, which is why it is computable here rather than waiting for
// specs/017-graph-aliasing.md. The gap between it and the pool size is what
// DAG-safe aliasing costs, and reporting both is what stops the larger number
// looking like a planner failure.
func (g *Graph) peakBytes(align int) int {
	if len(g.nodes) == 0 {
		return 0
	}
	peak := 0
	for at := range g.nodes {
		live := 0
		for _, t := range g.transients {
			if t.first >= 0 && t.first <= at && at <= t.last {
				live += alignUp(t.bytes, align)
			}
		}
		peak = max(peak, live)
	}
	return peak
}

// lower turns recorded nodes into a driver plan and compiles it.
func (g *Graph) lower() error {
	c, ok := g.dev.dev.(driver.GraphCompiler)
	if !ok {
		return fmt.Errorf("accel: Build: the %v backend cannot compile graphs",
			g.dev.info.Backend)
	}

	plan := &driver.Plan{
		Slots: len(g.slots), Transients: g.pool, Label: "graph",
		CollectStats:   g.collectStats,
		CollectTimings: g.collectTimings,
	}
	for i := range g.nodes {
		n := &g.nodes[i]
		node := driver.PlanNode{ID: int(n.id), BarrierBefore: g.barriersBefore[i] != nil}
		switch n.kind {
		case NodeHostWrite:
			node.Op = driver.OpHostWrite
			node.Data = n.data
			op, err := g.operand(n, n.accesses[0])
			if err != nil {
				return err
			}
			node.Dst = op
		case NodeCopyBuffer:
			node.Op = driver.OpCopy
			dst, err := g.operand(n, n.accesses[0])
			if err != nil {
				return err
			}
			src, err := g.operand(n, n.accesses[1])
			if err != nil {
				return err
			}
			node.Dst, node.Src = dst, src
		case NodeCopyTextureToBuffer, NodeCopyBufferToTexture:
			node.Op = driver.OpCopyRows
			dst, err := g.operand(n, n.accesses[0])
			if err != nil {
				return err
			}
			src, err := g.operand(n, n.accesses[1])
			if err != nil {
				return err
			}
			node.Dst, node.Src = dst, src
			tex := n.texture
			tight := tightRowPitch(tex.desc.Format, tex.desc.Size.Width)
			padded := g.dev.AlignedRowPitch(tex.desc.Format, tex.desc.Size.Width)
			// The texture side steps by the device's pitch and the buffer side
			// by the tight one, whichever direction the copy runs. Where the
			// two agree this is one contiguous copy.
			rows := &driver.RowCopy{
				Rows: tex.desc.Size.Height, RowBytes: tight,
				DstPitch: tight, SrcPitch: padded,
			}
			if n.kind == NodeCopyBufferToTexture {
				rows.DstPitch, rows.SrcPitch = padded, tight
			}
			node.Rows = rows

		case NodeDispatch, NodeDispatchIndirect:
			node.Op = driver.OpDispatch
			d, err := g.dispatchOperands(n)
			if err != nil {
				return err
			}
			node.Dispatch = d
		case NodeRenderPass:
			node.Op = driver.OpRenderPass
			rp, err := g.renderOperands(n)
			if err != nil {
				return err
			}
			node.Render = rp

		default:
			return fmt.Errorf("accel: Build: node %d is a %v, which is not yet lowered "+
				"(specs/009-sequencing.md)", n.id, n.kind)
		}
		plan.Nodes = append(plan.Nodes, node)
	}

	exe, err := c.Compile(plan)
	if err != nil {
		return err
	}
	g.plan, g.exe = plan, exe
	return nil
}

// operand turns one declared access into the bytes a backend touches.
func (g *Graph) operand(n *recNode, a access) (driver.Operand, error) {
	switch {
	case a.res.buf != nil:
		blk, base := blockFor(a.res.buf)
		o, err := driver.BlockOperand(blk, base+a.off, a.size)
		if err != nil {
			return driver.Operand{}, fmt.Errorf("accel: Build: node %d on %s: %w", n.id, a.res, err)
		}
		return o, nil
	case a.res.tex != nil:
		t := a.res.tex
		o, err := driver.BlockOperand(t.pool.block, t.alloc.Offset+a.off, a.size)
		if err != nil {
			return driver.Operand{}, fmt.Errorf("accel: Build: node %d on %s: %w", n.id, a.res, err)
		}
		return o, nil
	case a.res.slot != 0:
		o, err := driver.SlotOperand(int(a.res.slot), a.off, a.size)
		if err != nil {
			return driver.Operand{}, fmt.Errorf("accel: Build: node %d on %s: %w", n.id, a.res, err)
		}
		return o, nil
	}
	return driver.Operand{}, fmt.Errorf("accel: Build: node %d declares an unset resource", n.id)
}

func alignUp(n, to int) int {
	if to <= 1 {
		return n
	}
	return (n + to - 1) / to * to
}

// countGraphs adjusts the device's live graph count.
func (d *Device) countGraphs(delta int) {
	d.mu.Lock()
	d.graphs += delta
	d.mu.Unlock()
}

// releaseTransients gives up whatever this graph holds of its transient memory.
//
// It is the single place that does so, which is what keeps the accounting
// right: Build calls it when a later step fails, and Close calls it, and those
// are the only two ways a graph stops holding memory. A second release path --
// Close doing this itself for a shared pool -- left a Build that failed after
// reserving still counted against the pool, which then refused to close for the
// rest of the program.
//
// g.pool is the marker for "this graph took something", and it is nil for a
// graph with no transients at all. Releasing on the strength of g.shared
// instead would decrement a pool such a graph never incremented, and the count
// would then let the pool close while a live graph held offsets into it.
func (g *Graph) releaseTransients() {
	pool := g.pool
	if pool == nil {
		return
	}
	g.pool = nil

	// Under the lock, because Graph.run reads g.shared inside the critical
	// section that marks the graph in flight.
	g.mu.Lock()
	shared := g.shared
	g.shared = nil
	g.mu.Unlock()

	if shared != nil {
		// A shared pool belongs to the caller and outlives this graph. Freeing
		// it here would pull the memory out from under every other graph
		// planned into it, and the symptom would be one graph's results
		// appearing in another's buffers rather than a crash.
		shared.release()
		return
	}
	pool.Free()
	g.dev.countImplicit(-1)
}

// renderOperands lowers a render pass node.
//
// The attachments become ordinary operands, so the planner places and barriers
// them without knowing they are attachments — which is the point of declaring
// access rather than kind. What the backend has to understand is the draw list
// and the stage adapters, and those are as opaque to the planner as a kernel's
// entry point is.
func (g *Graph) renderOperands(n *recNode) (*driver.RenderPass, error) {
	p := n.pass
	if p == nil {
		return nil, fmt.Errorf("accel: Build: node %d is a render pass with nothing recorded",
			n.id)
	}
	if len(p.draws) == 0 {
		return nil, fmt.Errorf("accel: Build: render pass %q records no draws", p.desc.Label)
	}

	out := &driver.RenderPass{
		Label: p.desc.Label, Width: p.desc.Width, Height: p.desc.Height,
	}
	// colorFormats and depthFormat are what check V13 compares a pipeline
	// against. Kept in the public spelling rather than read back out of the
	// plan, because the message a caller reads names the format they wrote.
	var colorFormats []Format
	depthFormat := FormatInvalid
	for i, c := range p.desc.Color {
		op, err := g.operand(n, n.accesses[i])
		if err != nil {
			return nil, err
		}
		format, pitch, err := g.checkAttachment(p, fmt.Sprintf("colour attachment %d", i),
			c.View, c.Slot, 4)
		if err != nil {
			return nil, err
		}
		colorFormats = append(colorFormats, format)
		out.Color = append(out.Color, op)
		out.ColorFormat = append(out.ColorFormat, format.plan())
		out.ColorPitch = append(out.ColorPitch, pitch)
		out.ColorLoad = append(out.ColorLoad, c.Load)
		out.ColorStore = append(out.ColorStore, c.Store)
		out.ColorClear = append(out.ColorClear, c.Clear)
	}
	if p.desc.Depth != nil {
		op, err := g.operand(n, n.accesses[len(p.desc.Color)])
		if err != nil {
			return nil, err
		}
		format, pitch, err := g.checkAttachment(p, "depth attachment", p.desc.Depth.View,
			p.desc.Depth.Slot, 1)
		if err != nil {
			return nil, err
		}
		depthFormat = format
		out.Depth = &op
		out.DepthFormat = format.plan()
		out.DepthPitch = pitch
		out.DepthLoad = p.desc.Depth.Load
		out.DepthStore = p.desc.Depth.Store
		out.DepthClear = p.desc.Depth.Clear
		if c := p.desc.Depth.Clear; c < 0 || c > 1 {
			return nil, fmt.Errorf("accel: Build: render pass %q clears depth to %v, and "+
				"stored window depth is in [0, 1] — clip space is [-1, 1] and they are "+
				"different ranges", p.desc.Label, c)
		}
	}

	for i, d := range p.draws {
		pipe := d.pipeline
		if len(pipe.desc.Targets) != len(p.desc.Color) {
			return nil, fmt.Errorf("accel: Build: render pass %q draw %d: the pipeline %q "+
				"has %d colour targets and the pass has %d attachments",
				p.desc.Label, i, pipe.label, len(pipe.desc.Targets), len(p.desc.Color))
		}
		if (pipe.desc.DepthStencil != nil) != (p.desc.Depth != nil) {
			return nil, fmt.Errorf("accel: Build: render pass %q draw %d: the pipeline %q "+
				"%s depth state and the pass %s depth attachment", p.desc.Label, i,
				pipe.label, has(pipe.desc.DepthStencil != nil), has(p.desc.Depth != nil))
		}
		if err := g.checkTargetFormats(n, p, i, pipe, colorFormats, depthFormat); err != nil {
			return nil, err
		}
		rd := driver.RenderDraw{
			Vertex: pipe.desc.Vertex, Fragment: pipe.desc.Fragment,
			Topology:    uint8(pipe.desc.Primitive.Topology),
			FrontFace:   uint8(pipe.desc.Primitive.FrontFace),
			Cull:        uint8(pipe.desc.Primitive.Cull),
			VertexCount: d.vertices, InstanceCount: d.instances,
			FirstVertex: d.first, FirstInstance: d.firstInst,
			Indexed: d.indexed, BaseVertex: d.baseVertex,
		}
		if d.indirect {
			op, err := g.operand(n, n.accesses[d.indirectAccess])
			if err != nil {
				return nil, fmt.Errorf("accel: Build: render pass %q draw %d indirect "+
					"arguments: %w", p.desc.Label, i, err)
			}
			rd.Indirect, rd.IndirectArgs = true, op
		}
		if d.indexed {
			op, err := g.indexOperand(n, p, i, d)
			if err != nil {
				return nil, err
			}
			rd.Index, rd.IndexWidth = op, d.indexFmt.size()
		}
		if ds := pipe.desc.DepthStencil; ds != nil {
			rd.DepthTest, rd.DepthWrite = ds.Test, ds.Write
			rd.DepthCompare = uint8(ds.Compare)
		}
		for _, t := range pipe.desc.Targets {
			rd.Masks = append(rd.Masks, uint8(t.Mask.resolved()))
			rd.Blends = append(rd.Blends, driver.Blend(t.Blend))
		}
		vu, err := stageUniforms(p.desc.Label, i, pipe.desc.Vertex, d.vertexU, "SetVertexUniform")
		if err != nil {
			return nil, err
		}
		fu, err := stageUniforms(p.desc.Label, i, pipe.desc.Fragment, d.fragmentU, "SetFragmentUniform")
		if err != nil {
			return nil, err
		}
		rd.VertexUniforms, rd.FragmentUniforms = vu, fu
		if err := g.textureOperands(n, p, i, pipe, d, &rd); err != nil {
			return nil, err
		}
		if err := g.vertexOperands(n, p, i, pipe, d, &rd); err != nil {
			return nil, err
		}
		out.Draws = append(out.Draws, rd)
	}
	return out, nil
}

// stageUniforms checks one stage's by-value parameters against what the pass set.
//
// One slice per stage, and the index is the stage's own: a vertex stage's
// parameter 0 and a fragment stage's parameter 0 are different parameters that
// share a number. Checked here rather than left to the generated adapter, which
// would index past the end of a short slice or assert on the wrong type -- a
// panic inside a backend, where the caller cannot see which parameter was wrong.
//
// The type is compared by name, which is what the stage record carries. Two
// identically named types in different packages would pass; the adapter's own
// assertion catches that, and this catches the mistake a caller actually makes.
func stageUniforms(label string, draw int, s *Stage, set []any, call string) ([]any, error) {
	if len(s.Uniforms) == 0 {
		for i, v := range set {
			if v != nil {
				return nil, fmt.Errorf("accel: Build: render pass %q draw %d: %s(%d, ...) "+
					"was called and %s declares no by-value parameters", label, draw,
					call, i, s.Name)
			}
		}
		return nil, nil
	}
	out := make([]any, len(s.Uniforms))
	for _, u := range s.Uniforms {
		if u.Index >= len(set) || set[u.Index] == nil {
			return nil, fmt.Errorf("accel: Build: render pass %q draw %d: %s takes %q at "+
				"index %d and no value was set; call %s(%d, ...) before the draw",
				label, draw, s.Name, u.Name, u.Index, call, u.Index)
		}
		v := set[u.Index]
		if got := reflect.TypeOf(v).Name(); got != u.Type {
			return nil, fmt.Errorf("accel: Build: render pass %q draw %d: %s takes %s as "+
				"%q at index %d and a %s was set", label, draw, s.Name, u.Type,
				u.Name, u.Index, got)
		}
		out[u.Index] = v
	}
	for i, v := range set {
		if v != nil && i >= len(out) {
			return nil, fmt.Errorf("accel: Build: render pass %q draw %d: %s(%d, ...) was "+
				"called and %s declares %d by-value parameters", label, draw, call, i,
				s.Name, len(out))
		}
	}
	return out, nil
}

// indexOperand lowers an indexed draw's index buffer.
//
// The range is checked against the draw rather than against the buffer, for the
// reason the vertex range is: a buffer sized for the mesh with a draw that reads
// past the end of it is the shape that reads as corruption rather than as a
// mistake in the call.
func (g *Graph) indexOperand(n *recNode, p *RenderPass, draw int, d drawCall) (driver.Operand, error) {
	v := d.indexBuf
	op, err := g.operand(n, n.accesses[d.indexAccess])
	if err != nil {
		return driver.Operand{}, fmt.Errorf("accel: Build: render pass %q draw %d index "+
			"buffer: %w", p.desc.Label, draw, err)
	}
	size := v.Count * v.Buffer.DType().Size()
	if need := (d.first + d.vertices) * d.indexFmt.size(); need > size {
		return driver.Operand{}, fmt.Errorf("accel: Build: render pass %q draw %d: the "+
			"index buffer is %d bytes and the draw reads %d %s starting at %d, which "+
			"needs %d", p.desc.Label, draw, size, d.vertices, d.indexFmt, d.first, need)
	}
	return op, nil
}

// vertexOperands lowers a draw's bound vertex buffers.
//
// The layout is the pipeline's and the buffers are the pass's, so this is where
// the two meet and the only place that can check they agree. A slot the layout
// reads and no draw bound is refused here rather than fetched as zeros: zeroed
// attributes put every vertex at the origin, which reads as a broken transform
// rather than as a missing binding.
func (g *Graph) vertexOperands(n *recNode, p *RenderPass, draw int, pipe *RenderPipeline, d drawCall, rd *driver.RenderDraw) error {
	for slot, layout := range pipe.desc.VertexBuffers {
		if slot >= len(d.vertexBuf) || d.vertexBuf[slot].Buffer == nil {
			return fmt.Errorf("accel: Build: render pass %q draw %d: the pipeline %q reads "+
				"vertex buffer %d and no buffer is bound there; call SetVertexBuffer "+
				"before the draw", p.desc.Label, draw, pipe.label, slot)
		}
		v := d.vertexBuf[slot]
		op, err := g.operand(n, n.accesses[d.vertexAccess[slot]])
		if err != nil {
			return fmt.Errorf("accel: Build: render pass %q draw %d vertex buffer %d: %w",
				p.desc.Label, draw, slot, err)
		}
		size := v.Count * v.Buffer.DType().Size()

		// The elements the draw reaches must be inside the view. Checked
		// against the draw's own counts rather than against the buffer, because
		// a buffer big enough for the geometry and a draw that walks past the
		// end of it is the common shape and the one that reads as corruption.
		//
		// An indexed draw's per-vertex range is decided by the index *values*,
		// which are data and not structure, so build cannot know it. The
		// backend checks that range once per draw, against the indices it has
		// already decoded. The per-instance range is still structural.
		last := d.first + d.vertices
		if layout.StepMode == StepInstance {
			last = d.firstInst + d.instances
		} else if d.indexed {
			rd.VertexStrides = append(rd.VertexStrides, layout.Stride)
			rd.VertexBuffers = append(rd.VertexBuffers, op)
			rd.VertexLayouts = append(rd.VertexLayouts, lowerLayout(layout))
			continue
		}
		if need := last * layout.Stride; need > size {
			return fmt.Errorf("accel: Build: render pass %q draw %d: vertex buffer %d is "+
				"%d bytes and the draw reads %d elements %s at a stride of %d, which "+
				"needs %d", p.desc.Label, draw, slot, size, last, layout.StepMode,
				layout.Stride, need)
		}

		rd.VertexStrides = append(rd.VertexStrides, layout.Stride)
		rd.VertexBuffers = append(rd.VertexBuffers, op)
		rd.VertexLayouts = append(rd.VertexLayouts, lowerLayout(layout))
	}
	return nil
}

// lowerLayout drops the formats, which have done their job by the time a plan
// exists: a backend fetches floats, and validating against the stage happened at
// pipeline creation.
func lowerLayout(l VertexBufferLayout) driver.VertexLayout {
	out := driver.VertexLayout{Stride: l.Stride, PerInstance: l.StepMode == StepInstance}
	for _, a := range l.Attributes {
		out.Attributes = append(out.Attributes, driver.VertexAttribute{
			Location: a.Location, Offset: a.Offset, Components: a.Format.Components(),
		})
	}
	return out
}

// has renders one half of the depth-agreement message.
//
// It carries the article, because "has no" and "has a" take different ones and
// a shared "a" in the format string produced "has no a depth attachment".
func has(b bool) string {
	if b {
		return "has a"
	}
	return "has no"
}

// checkAttachment validates one attachment against the render area and reports
// the format its bytes are in and the distance between its rows.
//
// components is what one pixel occupies in the rasterizer's framebuffer: four
// for a colour attachment and one for depth. It is also what distinguishes the
// two aspects here, which is why a colour attachment of a depth format is
// caught in this one place rather than twice.
//
// Checked here rather than only in a backend, because an undersized attachment
// is a recording mistake and every backend would report it in its own words at
// its own moment -- and a backend that did not check it would read past the end
// of an allocation instead.
//
// # The two cases, and why a slot is not exempt
//
// A texture attachment is measured against the texture's extent, and its
// format is the view's rather than the texture's: a view may reinterpret
// within a compatible family, and the whole point of specs/045-texture-attachments.md
// section 2.1 is that writing through an sRGB view of a linear texture is a
// different operation from writing through the texture's own format.
//
// A slot is checked through its MinCount, which is the size check moved from
// build to bind: whatever is bound is at least that large, so a MinCount big
// enough for the area is a promise that every binding will be. Checking the
// slot rather than skipping it is the point -- "check it unless it is a slot"
// is the exemption shape that has already been wrong twice here, and a slot
// declared four elements wide and used as a 64x64 attachment would otherwise
// reach a backend.
func (g *Graph) checkAttachment(p *RenderPass, what string, v TextureView, s Slot, components int) (Format, int, error) {
	if s != 0 {
		return g.checkSlotAttachment(p, what, s, components)
	}
	t := v.Texture
	if t.desc.Usage&TextureRenderTarget == 0 {
		return 0, 0, fmt.Errorf("%w: Build: render pass %q %s is texture %q, which needs "+
			"%v and was created with %v", ErrUsage, p.desc.Label, what, t.desc.Label,
			TextureRenderTarget, t.desc.Usage)
	}
	if t.desc.Size.Width < p.desc.Width || t.desc.Size.Height < p.desc.Height {
		return 0, 0, fmt.Errorf("accel: Build: render pass %q %s is texture %q, which is "+
			"%dx%d, and the render area is %dx%d", p.desc.Label, what, t.desc.Label,
			t.desc.Size.Width, t.desc.Size.Height, p.desc.Width, p.desc.Height)
	}

	// The aspect check, and what it is worth given V13 exists.
	//
	// V13 would refuse every input this refuses, because NewRenderPipeline
	// already rejects a depth format as a colour target and a colour format as
	// a depth one -- so a pipeline that *agreed* with a wrong-aspect attachment
	// cannot be constructed, and the format comparison would catch the
	// disagreement. This runs first and says something better: "a colour
	// attachment does not take a depth format" names the mistake, where
	// "colour target 0 is RGBA8Unorm and attachment 0 is Depth32Float" leaves a
	// caller to work out which of the two they got wrong.
	//
	// The ordering is therefore load-bearing rather than incidental. Moving
	// V13 ahead of this would leave the check dead with its tests still
	// passing on V13's message, which is the shape a rule ends up withdrawn in.
	info := g.dev.FormatInfo(v.Format)
	if info.IsDepth != (components == 1) {
		return 0, 0, fmt.Errorf("%w: Build: render pass %q %s is viewed as %v, and %s",
			ErrFormat, p.desc.Label, what, v.Format, aspectMismatch(components == 1))
	}
	pitch := g.dev.AlignedRowPitch(v.Format, t.desc.Size.Width)
	if pitch == 0 {
		return 0, 0, fmt.Errorf("%w: Build: render pass %q %s is %v, whose layout is "+
			"device-defined -- there is no row pitch to give a backend and no one "+
			"encoding for the reference rasterizer to check against",
			ErrFormat, p.desc.Label, what, v.Format)
	}

	// Both backends lower a texture attachment. The gate that stood here
	// refused every non-CPU device while the Metal path was unwritten, which
	// was right at the time and is recorded in
	// specs/045-texture-attachments.md section 6.1 along with what it cost: the
	// render differential skipped, so graphics was verified on one backend
	// while the commit before it verified two.
	return v.Format, pitch, nil
}

// checkSlotAttachment is the rebindable half.
//
// A slot's resource is a buffer -- specs/034-surface-present.md makes a
// presented image one, and [Recorder.PresentSlot] declares it F32 with four
// elements per pixel -- so its format is what those bytes already are rather
// than anything the descriptor states. Saying it here is what lets the backend
// stop assuming it, and it is byte for byte what a slot attachment has always
// been.
func (g *Graph) checkSlotAttachment(p *RenderPass, what string, s Slot, components int) (Format, int, error) {
	if int(s) < 1 || int(s) > len(g.slots) {
		return 0, 0, fmt.Errorf("accel: Build: render pass %q %s names slot %d of %d",
			p.desc.Label, what, s, len(g.slots))
	}
	want := p.desc.Width * p.desc.Height * components
	if have := g.slots[s-1].MinCount; have < want {
		return 0, 0, fmt.Errorf("accel: Build: render pass %q %s declares a MinCount of "+
			"%d elements and a %dx%d area at %d components per pixel needs %d",
			p.desc.Label, what, have, p.desc.Width, p.desc.Height, components, want)
	}
	if components == 1 {
		return Depth32Float, p.desc.Width * Depth32Float.BytesPerPixel(), nil
	}
	return RGBA32Float, p.desc.Width * RGBA32Float.BytesPerPixel(), nil
}

// checkTargetFormats is check V13: a pipeline's declared attachment formats are
// the pass's.
//
// # Why this is checked at all, and why it could not be before
//
// Every target compiles a render pipeline against its attachment formats --
// Vulkan through the render pass or the dynamic-rendering format list, Metal
// through the render pipeline's colour attachment descriptors, D3D12 through
// RTVFormats. Binding a pipeline to attachments it was not compiled for is
// undefined on some of them and a validation error on others, and neither
// tells a caller which attachment.
//
// specs/003-command-graph.md's table has carried V13 since it was written, and
// specs/042-surface-completion.md section 5.2 found it unimplementable rather
// than merely unimplemented: an attachment was a buffer view and a buffer view
// has no format, so there was nothing on one side to compare. That is the
// same category as the withdrawn V23, with the difference that V23 was marked.
//
// # It compares the view's format and not the texture's
//
// A view may reinterpret within a compatible family, and a pass writing an
// RGBA8Unorm texture through an RGBA8UnormSRGB view is doing sRGB. The
// pipeline is compiled for what the writes go through, which is the view.
// Comparing the texture's own format would refuse exactly the case
// specs/045-texture-attachments.md section 2.1 exists for.
func (g *Graph) checkTargetFormats(n *recNode, p *RenderPass, draw int,
	pipe *RenderPipeline, colour []Format, depth Format) error {
	for i, tgt := range pipe.desc.Targets {
		if tgt.Format == colour[i] {
			continue
		}
		return fmt.Errorf("%w: Build: render pass %q (node %d) draw %d: the pipeline %q "+
			"declares colour target %d as %v and attachment %d is %v. A pipeline is "+
			"compiled against its attachment formats on every backend, so the two are "+
			"the same format (check V13)",
			ErrFormat, p.desc.Label, n.id, draw, pipe.label, i, tgt.Format, i, colour[i])
	}
	if ds := pipe.desc.DepthStencil; ds != nil && ds.Format != depth {
		return fmt.Errorf("%w: Build: render pass %q (node %d) draw %d: the pipeline %q "+
			"declares depth as %v and the depth attachment is %v (check V13)",
			ErrFormat, p.desc.Label, n.id, draw, pipe.label, ds.Format, depth)
	}
	return nil
}

// aspectMismatch names which way round an attachment's format is wrong.
func aspectMismatch(wantDepth bool) string {
	if wantDepth {
		return "a depth attachment takes a depth format"
	}
	return "a colour attachment does not take a depth format"
}

// textureOperands lowers a draw's bound stage textures.
//
// A stage declares which textures it fetches and a pass binds them, so this is
// where the two meet -- the same shape vertexOperands has, and the same
// refusal: a slot the stage reads and no draw bound is refused here rather than
// fetched as zeros, because a stage sampling a black texture looks like a
// lighting bug rather than a missing binding.
func (g *Graph) textureOperands(n *recNode, p *RenderPass, draw int, pipe *RenderPipeline, d drawCall, rd *driver.RenderDraw) error {
	for _, s := range []struct {
		stage  *Stage
		access []int
		out    *[]driver.RenderTexture
		what   string
	}{
		{pipe.desc.Vertex, d.vertexTexAccess, &rd.VertexTextures, "vertex"},
		{pipe.desc.Fragment, d.fragmentTexAccess, &rd.FragmentTextures, "fragment"},
	} {
		if s.stage == nil {
			continue
		}
		for _, tx := range s.stage.Textures {
			if !tx.Reads {
				continue
			}
			if tx.Index >= len(s.access) || s.access[tx.Index] < 0 {
				return fmt.Errorf("accel: Build: render pass %q draw %d: the %s stage %q "+
					"fetches texture %d and no texture is bound there; call SetTexture "+
					"before the draw", p.desc.Label, draw, s.what, s.stage.Name, tx.Index)
			}
			v := d.textures[tx.Index]
			op, err := g.operand(n, n.accesses[s.access[tx.Index]])
			if err != nil {
				return fmt.Errorf("accel: Build: render pass %q draw %d %s texture %d: %w",
					p.desc.Label, draw, s.what, tx.Index, err)
			}
			sz := v.Texture.desc.Size
			for len(*s.out) <= tx.Index {
				*s.out = append(*s.out, driver.RenderTexture{})
			}
			(*s.out)[tx.Index] = driver.RenderTexture{
				Operand: op,
				Format:  v.Format.plan(),
				Width:   sz.Width, Height: sz.Height,
				Pitch: g.dev.AlignedRowPitch(v.Format, sz.Width),
			}
		}
	}
	return nil
}
