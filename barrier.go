// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

// barrier is one ordering point emitted before a node.
//
// A barrier is queue-wide. That single fact is what makes the emitted count far
// smaller than the hazard count: one emitted before node j for a hazard on
// resource R also makes visible every write before it on every other resource,
// so it covers every hazard whose source precedes it and whose destination
// follows.
type barrier struct {
	// Src and Dst are the stages being ordered. A backend lowers them to its
	// native primitive; the CPU backend executes serially and so cannot observe
	// a missing one, which specs/006-backends.md section 5 states and which is
	// why specs/017-graph-aliasing.md compares plans rather than trusting
	// execution here to find a barrier bug.
	src, dst stage

	// memory is false for a write-after-read, which needs execution ordering
	// only. A read dirties nothing, so nothing needs flushing, and a memory
	// barrier there is a needless cache operation. Stating it explicitly matters
	// because emitting one anyway is the standard mistake.
	memory bool

	// reasons is what this barrier was emitted for, kept so a caller asking why
	// a graph does not overlap gets an answer naming resources.
	reasons []barrierReason
}

type barrierReason struct {
	hazard hazard
	from   NodeID
	label  string
}

// rangeState is the current state of one resource range, as opposed to the
// pending hazards edge inference tracks. Barrier insertion is a second walk in
// record order over the same resources.
type rangeState struct {
	span accessSpan

	lastWriteStage stage
	lastWriteNode  NodeID
	written        bool

	readStages stage
	readNodes  []NodeID
}

// planBarriers is the second walk. It replaces the record-order plan's blanket
// barrier before every node with the ones the declared accesses actually need.
//
// Batching is not only an efficiency measure, it changes the count: all the
// barriers one node's accesses require are accumulated and issued as one
// barrier immediately before it, and because that barrier is queue-wide it also
// satisfies every pending hazard whose source precedes it. The state machine
// exploits that by clearing satisfied state when a barrier is emitted rather
// than re-emitting per hazard.
//
// The honest cost: a barrier naming compute on both sides is a full pipeline
// drain, so it stops independent work either side from overlapping even though
// the hazard concerns one buffer. Finer primitives exist on several backends
// and none is used at v0. See specs/003-command-graph.md.
func (g *Graph) planBarriers() {
	g.barriersBefore = make([]*barrier, len(g.nodes))
	states := map[resourceRef][]*rangeState{}

	// The head-of-submission barrier. Two submissions to one queue are fully
	// ordered and every write by the first is visible to the second with no
	// caller action, and this is what implements that: a global acquire covering
	// the graph's external reads, and the cross-submission write-after-write on
	// a transient pool whose bytes the last submission also used.
	if len(g.nodes) > 0 {
		g.barriersBefore[0] = &barrier{
			src: stageAll, dst: stageAll, memory: true,
			reasons: []barrierReason{{from: -1, label: "head of submission"}},
		}
	}

	for i := range g.nodes {
		n := &g.nodes[i]
		var b *barrier
		// dst is the stage of the access that needs the ordering, not the
		// node's. A pass reading a vertex buffer an earlier node wrote must
		// wait at vertex input, and naming the whole pass instead would stall
		// the colour and depth stages on a hazard neither one has.
		need := func(h hazard, src, dst stage, from NodeID, label string) {
			if b == nil {
				b = &barrier{}
			}
			b.src |= src
			b.dst |= dst
			// A write-after-read needs ordering only. Any other hazard in the
			// same batch upgrades the whole barrier, which is correct: the union
			// of what the accesses need.
			if h != hazardWAR {
				b.memory = true
			}
			b.reasons = append(b.reasons, barrierReason{hazard: h, from: from, label: label})
		}

		spans := g.spansOf(n)
		for _, a := range spans {
			for _, rs := range states[key(a.res)] {
				if !rs.span.overlaps(a) {
					continue
				}
				if a.mode == AccessRead || a.mode == AccessReadWrite {
					if rs.written {
						need(hazardRAW, rs.lastWriteStage, a.stage, rs.lastWriteNode, g.labelOf(a.res))
					}
				}
				if a.writes() {
					if rs.written {
						need(hazardWAW, rs.lastWriteStage, a.stage, rs.lastWriteNode, g.labelOf(a.res))
					}
					for _, r := range rs.readNodes {
						need(hazardWAR, rs.readStages, a.stage, r, g.labelOf(a.res))
					}
				}
			}
		}

		if b != nil || g.barriersBefore[i] != nil {
			if g.barriersBefore[i] != nil && b != nil {
				// The head barrier already orders everything; fold this node's
				// reasons into it rather than emitting two before one node.
				g.barriersBefore[i].reasons = append(g.barriersBefore[i].reasons, b.reasons...)
			} else if b != nil {
				g.barriersBefore[i] = b
			}
			// A queue-wide barrier satisfies every hazard whose source precedes
			// it, so every range's pending state is cleared. Without this the
			// next node re-emits a barrier for a hazard this one already covered,
			// which is where the count doubles.
			clearPending(states)
		}

		for _, a := range spans {
			commitRange(states, a)
		}
	}

	g.barriers = 0
	for _, b := range g.barriersBefore {
		if b != nil {
			g.barriers++
		}
	}
}

func clearPending(states map[resourceRef][]*rangeState) {
	for _, rs := range states {
		for _, r := range rs {
			r.written = false
			r.readStages = 0
			r.readNodes = r.readNodes[:0]
		}
	}
}

func commitRange(states map[resourceRef][]*rangeState, a accessSpan) {
	k := key(a.res)
	list := states[k]
	for _, rs := range list {
		if rs.span.off == a.off && rs.span.size == a.size {
			update(rs, a)
			states[k] = list
			return
		}
	}
	rs := &rangeState{span: a}
	update(rs, a)
	states[k] = append(list, rs)
}

func update(rs *rangeState, a accessSpan) {
	if a.mode == AccessRead || a.mode == AccessReadWrite {
		rs.readStages |= a.stage
		rs.readNodes = append(rs.readNodes, a.node)
	}
	if a.writes() {
		rs.written = true
		rs.lastWriteStage = a.stage
		rs.lastWriteNode = a.node
		// A write ends the read set it was ordered against: the write-after-read
		// edge already covers those readers, so keeping them would re-emit a
		// barrier for a hazard that no longer exists.
		rs.readStages = 0
		rs.readNodes = rs.readNodes[:0]
	}
}

func (g *Graph) labelOf(r resourceRef) string {
	if r.buf != nil {
		return r.buf.desc.Label
	}
	if r.tex != nil {
		return r.tex.desc.Label
	}
	if int(r.slot) <= len(g.slots) && r.slot >= 1 {
		return g.slots[r.slot-1].Name
	}
	return "an unknown resource"
}

// planSerialBarriers is the conservative plan: a barrier before every node.
//
// Correct rather than merely safe. Every inferred edge runs from a lower node
// id to a strictly higher one, because inference walks nodes in record order
// and links each access against state left by earlier ones. Record order is
// therefore a topological order of the DAG, so a serial execution respects
// every edge, and a full barrier between consecutive nodes covers every
// read-after-write, write-after-write, and write-after-read the inference could
// classify. See specs/015-graph-recording.md section 3.
func (g *Graph) planSerialBarriers() {
	g.barriersBefore = make([]*barrier, len(g.nodes))
	for i := range g.nodes {
		g.barriersBefore[i] = &barrier{
			src: stageAll, dst: stageAll, memory: true,
			reasons: []barrierReason{{from: -1, label: "the conservative plan"}},
		}
	}
	g.barriers = len(g.nodes)
}
