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
func (r *Recorder) Build() (*Graph, error) {
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
		dev:        r.state.dev,
		nodes:      r.state.nodes,
		slots:      r.state.slots,
		transients: r.state.transients,
	}
	if err := g.placeTransients(); err != nil {
		return nil, err
	}
	g.inferEdges()
	if err := g.assertAcyclic(); err != nil {
		g.releaseTransients()
		return nil, err
	}
	g.reachability()
	g.planBarriers()
	if err := g.lower(); err != nil {
		g.releaseTransients()
		return nil, err
	}
	g.state.init("graph")
	g.bound = make([]Binding, len(g.slots)+1)
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
	align := g.dev.allocAlignment(UsageStorage | UsageCopySrc | UsageCopyDst)
	total := 0
	for _, t := range g.transients {
		t.offset = alignUp(total, align)
		next := t.offset + t.bytes
		if next < t.offset {
			return fmt.Errorf("accel: Build: the transient pool overflows at %q", t.buf.desc.Label)
		}
		total = next
		g.memory.UnaliasedBytes += alignUp(t.bytes, align)
	}
	// The pool is rounded up like every transient in it. Without that the last
	// transient is the one that is not padded, so the pool disagrees with the
	// unaliased total it is supposed to equal at this milestone -- which is
	// exactly what the build fuzzer reported.
	total = alignUp(total, align)
	g.memory.TransientBytes = total
	g.memory.PeakBytes = g.peakBytes(align)
	if total == 0 {
		return nil
	}

	blk, err := g.dev.dev.Alloc(driver.MemoryDevice, total, "graph transients")
	if err != nil {
		return fmt.Errorf("accel: Build: transient pool: %w", err)
	}
	g.dev.countImplicit(1)
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

	plan := &driver.Plan{Slots: len(g.slots), Transients: g.pool, Label: "graph"}
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
		case NodeDispatch:
			node.Op = driver.OpDispatch
			d, err := g.dispatchOperands(n)
			if err != nil {
				return err
			}
			node.Dispatch = d
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

func (g *Graph) releaseTransients() {
	if g.pool != nil {
		g.pool.Free()
		g.dev.countImplicit(-1)
		g.pool = nil
	}
}
