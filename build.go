// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"errors"
	"fmt"

	"golang.design/x/accel/internal/driver"
)

// Build validates the recorded nodes, assigns transient memory, plans barriers,
// and lowers the result for the device.
//
// # The plan this milestone produces
//
// Nodes execute in record order with a barrier before each one, and every
// transient gets its own bytes. That is deliberately the most conservative plan
// there is, and it is correct rather than merely safe: every dependency edge a
// hazard analysis could infer runs from a lower node id to a higher one, since
// an edge exists only because a later node's declared access conflicts with an
// earlier one's. Record order is therefore a topological order of that DAG, and
// a full barrier between consecutive nodes covers every read-after-write,
// write-after-read, and write-after-write it could classify.
//
// specs/016-graph-execution.md computes the edges instead of assuming them all,
// and specs/017-graph-aliasing.md packs the transients. This plan is not thrown
// away when they land: it is what 017's differential fuzz compares against, and
// an oracle that already existed before the optimizer cannot have inherited the
// optimizer's bugs.
// BuildNaive builds the graph under the conservative plan of
// specs/015-graph-recording.md: nodes in record order, a barrier before each,
// and no transient aliasing.
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
		dev:          r.state.dev,
		nodes:        r.state.nodes,
		slots:        r.state.slots,
		transients:   r.state.transients,
		collectStats: r.state.collectStats,
		shared:       r.state.shared,
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
		CollectStats: g.collectStats,
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
	for i, c := range p.desc.Color {
		op, err := g.operand(n, n.accesses[i])
		if err != nil {
			return nil, err
		}
		if err := g.checkAttachment(p, fmt.Sprintf("colour attachment %d", i), c.View, 4); err != nil {
			return nil, err
		}
		out.Color = append(out.Color, op)
		out.ColorLoad = append(out.ColorLoad, c.Load)
		out.ColorClear = append(out.ColorClear, c.Clear)
	}
	if p.desc.Depth != nil {
		op, err := g.operand(n, n.accesses[len(p.desc.Color)])
		if err != nil {
			return nil, err
		}
		if err := g.checkAttachment(p, "depth attachment", p.desc.Depth.View, 1); err != nil {
			return nil, err
		}
		out.Depth = &op
		out.DepthLoad = p.desc.Depth.Load
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
		rd := driver.RenderDraw{
			Vertex: pipe.desc.Vertex.RunVertex, Fragment: pipe.desc.Fragment.RunFragment,
			Topology:    uint8(pipe.desc.Primitive.Topology),
			FrontFace:   uint8(pipe.desc.Primitive.FrontFace),
			Cull:        uint8(pipe.desc.Primitive.Cull),
			VertexCount: d.vertices, InstanceCount: d.instances,
			FirstVertex: d.first, FirstInstance: d.firstInst,
		}
		if ds := pipe.desc.DepthStencil; ds != nil {
			rd.DepthTest, rd.DepthWrite = ds.Test, ds.Write
			rd.DepthCompare = uint8(ds.Compare)
		}
		for _, t := range pipe.desc.Targets {
			rd.Masks = append(rd.Masks, uint8(t.Mask.resolved()))
		}
		// A stage that declares a by-value parameter cannot be drawn, and this
		// is where the value would have been needed. specs/033-render-api.md
		// deviation 1: the draw-time uniform channel was removed, and the
		// mechanism the spec does describe -- a uniform buffer at a recorded
		// offset, section 6 -- is unbuilt. Refused rather than passed an empty
		// slice, because the generated adapter would then index past its end
		// and the diagnostic would come from the backend.
		for _, s := range []*Stage{pipe.desc.Vertex, pipe.desc.Fragment} {
			if len(s.Uniforms) > 0 {
				return nil, fmt.Errorf("accel: Build: render pass %q draw %d: %s declares "+
					"the by-value parameter %q, and no render path supplies one yet "+
					"(specs/033-render-api.md deviation 1)",
					p.desc.Label, i, s.Name, s.Uniforms[0].Name)
			}
		}
		if len(pipe.desc.Vertex.Attributes) > 0 {
			return nil, fmt.Errorf("accel: Build: render pass %q draw %d: %s reads the "+
				"vertex attribute %q, and the vertex layout that would fetch it is "+
				"unbuilt (specs/033-render-api.md deviation 1)",
				p.desc.Label, i, pipe.desc.Vertex.Name,
				pipe.desc.Vertex.Attributes[0].Name)
		}
		out.Draws = append(out.Draws, rd)
	}
	return out, nil
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

// checkAttachment validates one attachment against the render area.
//
// components is what one pixel occupies: four for a colour attachment and one
// for depth. Checked here rather than only in a backend, because an undersized
// attachment is a recording mistake and every backend would report it in its
// own words at its own moment -- and a backend that did not check it would read
// past the end of a buffer instead.
func (g *Graph) checkAttachment(p *RenderPass, what string, v BufferView, components int) error {
	want := p.desc.Width * p.desc.Height * components
	if v.Count < want {
		return fmt.Errorf("accel: Build: render pass %q %s holds %d elements and a %dx%d "+
			"area at %d components per pixel needs %d",
			p.desc.Label, what, v.Count, p.desc.Width, p.desc.Height, components, want)
	}
	return nil
}
