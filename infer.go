// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"fmt"
	"math/bits"
)

// hazard is why one node must be ordered after another.
type hazard uint8

const (
	hazardRAW hazard = iota // the read must see the write
	hazardWAW               // the earlier write must not land last
	hazardWAR               // the read must not observe the new value
)

func (h hazard) String() string {
	switch h {
	case hazardRAW:
		return "read after write"
	case hazardWAW:
		return "write after write"
	case hazardWAR:
		return "write after read"
	}
	return "unknown hazard"
}

// stage is the part of the pipeline an access happens in.
//
// It is a mask rather than an enum because a barrier names a set on each side,
// and a node reading one resource in one stage and another in a second is
// ordinary. Only two exist while the graph holds transfers and flat dispatches;
// specs/005-graphics.md adds the rest.
type stage uint8

const (
	stageTransfer stage = 1 << iota
	stageCompute
)

func (s stage) String() string {
	switch s {
	case 0:
		return "none"
	case stageTransfer:
		return "transfer"
	case stageCompute:
		return "compute"
	}
	return "transfer and compute"
}

// span is one resource range an access covers, plus who made it.
type accessSpan struct {
	res   resourceRef
	off   int
	size  int
	node  NodeID
	mode  Access
	stage stage
}

func (a accessSpan) overlaps(b accessSpan) bool {
	if !sameResource(a.res, b.res) {
		return false
	}
	// Half-open intervals. Comparing whole resources instead is not a missed
	// optimization, it is the optimization: a tiled workload is a stream of
	// nodes touching disjoint slices of one allocation, and whole-resource
	// comparison serializes exactly the parallelism the graph exists to express.
	return a.off < b.off+b.size && b.off < a.off+a.size
}

// covers reports whether a fully contains b. Trimming state on full cover
// rather than partial cover is deliberate: partial-cover trimming needs
// interval splitting, and the ranges that occur are either identical or
// disjoint.
func (a accessSpan) covers(b accessSpan) bool {
	return sameResource(a.res, b.res) && a.off <= b.off && b.off+b.size <= a.off+a.size
}

func sameResource(a, b resourceRef) bool {
	if a.buf != nil || b.buf != nil {
		return a.buf == b.buf
	}
	return a.slot != 0 && a.slot == b.slot
}

// resState is one resource's pending hazards, updated as nodes are visited in
// record order.
type resState struct {
	writers []accessSpan // writes not yet fully overwritten
	readers []accessSpan // reads since the last write covering their range
}

// inferEdges builds the dependency DAG from declared accesses.
//
// # Why edges are inferred and not declared
//
// A caller writing edges by hand would be writing the thing this design exists
// to remove, and would get it wrong in the direction that does not fail: a
// missing edge is a race, and a race on a backend that happens to serialize is
// invisible until one that does not runs the two arms at once. Declared
// accesses are the input a caller can be trusted with, because getting one
// wrong makes their own node wrong rather than a distant one.
//
// # Why no cycle check
//
// Edges run only from a node to a later-recorded one, because this walks nodes
// in record order and links each access against state left by earlier nodes. So
// record order is always a topological order of the result and a cycle is
// impossible by construction. [Graph.assertAcyclic] keeps that as an assertion
// against a builder defect rather than advertising a caller-facing check that
// can never fire.
//
// See specs/003-command-graph.md.
func (g *Graph) inferEdges() {
	st := make(map[resourceRef]*resState, len(g.nodes))
	g.succ = make([][]NodeID, len(g.nodes))
	g.pred = make([][]NodeID, len(g.nodes))

	for i := range g.nodes {
		n := &g.nodes[i]
		spans := g.spansOf(n)

		// Classify all of a node's accesses before committing any of them.
		// Committing during classification would make a node that reads and
		// writes one range hazard against itself.
		for _, a := range spans {
			s := st[key(a.res)]
			if s == nil {
				continue
			}
			if a.mode == AccessRead || a.mode == AccessReadWrite {
				for _, w := range s.writers {
					if w.overlaps(a) {
						g.addEdge(w.node, n.id, hazardRAW)
					}
				}
			}
			if a.writes() {
				for _, w := range s.writers {
					if w.overlaps(a) {
						g.addEdge(w.node, n.id, hazardWAW)
					}
				}
				for _, r := range s.readers {
					if r.overlaps(a) {
						g.addEdge(r.node, n.id, hazardWAR)
					}
				}
			}
		}
		for _, a := range spans {
			commit(st, a)
		}
	}
}

// key normalizes a resourceRef so that two references to one resource hash
// alike. A slot is its own identity, because its eventual resource is unknown
// and V24 is what checks that two slots did not land on one buffer.
func key(r resourceRef) resourceRef {
	if r.buf != nil {
		return resourceRef{buf: r.buf}
	}
	return resourceRef{slot: r.slot}
}

func commit(st map[resourceRef]*resState, a accessSpan) {
	k := key(a.res)
	s := st[k]
	if s == nil {
		s = &resState{}
		st[k] = s
	}
	if a.mode == AccessRead || a.mode == AccessReadWrite {
		s.readers = append(s.readers, a)
	}
	if !a.writes() {
		return
	}
	// A write drops every pending span its range fully covers. The dropped
	// readers are already ordered before it by the write-after-read edge just
	// added, so no later node needs them.
	s.writers = keepUncovered(s.writers, a)
	s.readers = keepUncovered(s.readers, a)
	s.writers = append(s.writers, a)
}

func keepUncovered(spans []accessSpan, by accessSpan) []accessSpan {
	out := spans[:0]
	for _, s := range spans {
		if !by.covers(s) {
			out = append(out, s)
		}
	}
	return out
}

// addEdge records one dependency, ignoring a duplicate.
//
// Duplicates are common and are not a defect: one node reading and writing
// overlapping ranges of a resource an earlier node wrote produces both a RAW
// and a WAW against it, and the ordering they need is the same edge.
func (g *Graph) addEdge(from, to NodeID, h hazard) {
	if from == to {
		return
	}
	g.hazards++
	for _, e := range g.succ[from] {
		if e == to {
			return
		}
	}
	g.succ[from] = append(g.succ[from], to)
	g.pred[to] = append(g.pred[to], from)
}

// spansOf turns a node's declarations into access spans.
//
// Declaration order, not a map: the barrier plan and every diagnostic derived
// from it must not depend on iteration order, or the plan goldens flap and a
// planning regression shows up as a flaky test rather than as a bug.
func (g *Graph) spansOf(n *recNode) []accessSpan {
	out := make([]accessSpan, len(n.accesses))
	for i, a := range n.accesses {
		out[i] = accessSpan{
			res: a.res, off: a.off, size: a.size,
			node: n.id, mode: a.mode, stage: n.stage,
		}
	}
	return out
}

func (a accessSpan) writes() bool {
	return a.mode == AccessWrite || a.mode == AccessReadWrite
}

// assertAcyclic is V22: an internal check against a builder defect.
//
// It can only fire if inference stopped running in record order, which is why
// it is an assertion and not a caller-facing validation. It costs one pass.
func (g *Graph) assertAcyclic() error {
	for from := range g.succ {
		for _, to := range g.succ[from] {
			if int(to) <= from {
				return fmt.Errorf("accel: Build: the inferred edge set has a cycle: "+
					"node %d depends on node %d, which does not follow it in record order",
					int(to), from)
			}
		}
	}
	return nil
}

// reachability computes the transitive closure of the DAG as one bitset per
// node.
//
// It is here rather than with the interference relation that consumes it
// because it is a property of the edge set. Computed backwards over record
// order, which is a reverse topological order by construction, so one pass
// suffices: a node's reach is the union of its successors' reaches plus the
// successors themselves.
//
// Cost is O(N²/64) words for N nodes, which for the 200-node graphs
// specs/003-command-graph.md sizes against is under a thousand words.
func (g *Graph) reachability() {
	n := len(g.nodes)
	if n == 0 {
		return
	}
	words := (n + 63) / 64
	g.reach = make([]uint64, n*words)
	g.reachWords = words

	for i := n - 1; i >= 0; i-- {
		row := g.reach[i*words : (i+1)*words]
		for _, to := range g.succ[i] {
			row[int(to)/64] |= 1 << (uint(to) % 64)
			other := g.reach[int(to)*words : (int(to)+1)*words]
			for w := range row {
				row[w] |= other[w]
			}
		}
	}
}

// reaches reports whether b must run after a.
func (g *Graph) reaches(a, b NodeID) bool {
	if g.reachWords == 0 {
		return false
	}
	row := g.reach[int(a)*g.reachWords : (int(a)+1)*g.reachWords]
	return row[int(b)/64]&(1<<(uint(b)%64)) != 0
}

// popcount reports how many nodes a node reaches, for statistics and tests.
func (g *Graph) reachCount(a NodeID) int {
	if g.reachWords == 0 {
		return 0
	}
	n := 0
	for _, w := range g.reach[int(a)*g.reachWords : (int(a)+1)*g.reachWords] {
		n += bits.OnesCount64(w)
	}
	return n
}
