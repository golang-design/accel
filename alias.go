// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import "sort"

// compatible reports whether two transients may share bytes.
//
// # Why this is an interference relation and not an interval
//
// An interval planner gives each transient the record-order span from its first
// user to its last, and aliases those whose spans are disjoint. On a DAG that is
// unsound, and the counterexample is a diamond: node 1 fans out to 2 and 3,
// which join at 4. Transient T is written by 1 and read by 2 and 3; U is written
// by 4. In record order T occupies [1, 3] and U occupies [4, ...], disjoint, so
// an interval planner aliases them. But node 3 reads T and node 4 writes U, and
// nothing orders 3 before 4 — so a backend that runs the two arms at once, which
// is every backend doing what the DAG permits, corrupts T.
//
// The sound test is reachability, not position: every user of one must be
// ordered against every user of the other.
//
//	compatible(T, U)  ⟺  ∀ t ∈ users(T), ∀ u ∈ users(U) : t → u ∨ u → t
//
// # Why the reachability is strict
//
// A node does not reach itself, so a node using *both* transients makes them
// incompatible — which is right, and is spec 003's "t0, t1: no, n2 uses both"
// row. A relation that counted a shared user as ordered would alias the input
// and output of one node, and every in-place-looking chain would corrupt
// itself. That is not a subtle failure and it is not a rare one: it is what a
// chain of dispatches is made of.
//
// See specs/017-graph-aliasing.md.
func (g *Graph) compatible(a, b *transient) bool {
	for _, ta := range a.users {
		for _, tb := range b.users {
			if !g.reaches(ta, tb) && !g.reaches(tb, ta) {
				return false
			}
		}
	}
	return true
}

// packTransients assigns each transient an offset in one pool.
//
// # Why this is dynamic storage allocation and not graph colouring
//
// Colouring assigns transients to a fixed set of equal slots, and transients
// have different sizes. The right formulation is to give each an offset such
// that interfering transients do not overlap in bytes, which is NP-hard, so
// this is a heuristic and says so rather than implying optimality.
//
// Greedy by size descending beats greedy by first use, by live length, and best
// fit here, because the large transients constrain everything else and placing
// them first avoids the fragmentation a small-first order creates. The
// deterministic tie break is not cosmetic: without it the layout depends on
// sort stability and the plan goldens flap.
//
// Cost is O(n² log n) for n transients, dominated by the scan of placed
// transients per placement. For the 200 transients spec 003 sizes against, that
// is tens of thousands of operations, once, at build.
func (g *Graph) packTransients(align int) int {
	order := make([]*transient, len(g.transients))
	copy(order, g.transients)
	sort.SliceStable(order, func(i, j int) bool {
		if order[i].bytes != order[j].bytes {
			return order[i].bytes > order[j].bytes
		}
		return order[i].firstWriter() < order[j].firstWriter()
	})

	placed := make([]*transient, 0, len(order))
	total := 0
	for _, t := range order {
		size := alignUp(t.bytes, align)
		t.offset = lowestFreeOffset(t, placed, g, align, size)
		t.placed = true
		placed = append(placed, t)
		total = max(total, t.offset+size)
	}
	return alignUp(total, align)
}

// lowestFreeOffset finds the first offset at which t misses every interfering
// placement.
func lowestFreeOffset(t *transient, placed []*transient, g *Graph, align, size int) int {
	type span struct{ lo, hi int }
	var occupied []span
	for _, u := range placed {
		if g.compatible(t, u) {
			continue
		}
		occupied = append(occupied, span{u.offset, u.offset + alignUp(u.bytes, align)})
	}
	sort.Slice(occupied, func(i, j int) bool { return occupied[i].lo < occupied[j].lo })

	at := 0
	for _, o := range occupied {
		if at+size <= o.lo {
			break
		}
		at = max(at, alignUp(o.hi, align))
	}
	return at
}

// firstWriter is the record-order id of the node that first writes this
// transient, or its first user if nothing writes it. It is the packing order's
// tie break, so it must be defined for every transient.
func (t *transient) firstWriter() int {
	if t.writer >= 0 {
		return t.writer
	}
	if len(t.users) > 0 {
		return int(t.users[0])
	}
	return 0
}

// handovers are the aliasing barriers.
//
// When transient U is placed at bytes T also occupies, U's first write must be
// ordered after T's last use: without it a backend may reorder U's write before
// T's read and corrupt the value T's reader expects.
//
// On the worked graph both handovers ride on a barrier the data flow required
// anyway, so the count does not change. That is an assertion in the tests
// rather than a hope, because a planner that added one per handover would be
// paying for aliasing twice.
func (g *Graph) planHandovers() {
	for i, a := range g.transients {
		if !a.placed || len(a.users) == 0 {
			continue
		}
		for _, b := range g.transients[:i] {
			if !b.placed || len(b.users) == 0 || !overlapBytes(a, b, g.poolAlign) {
				continue
			}
			// The two are ordered — packing placed them together only because
			// every user of one precedes every user of the other — so the
			// handover is a barrier before the later one's first user.
			later, earlier := a, b
			if earlier.users[0] > later.users[0] {
				later, earlier = b, a
			}
			g.requireBarrier(later.users[0], earlier.lastUser())
		}
	}
	g.barriers = 0
	for _, b := range g.barriersBefore {
		if b != nil {
			g.barriers++
		}
	}
}

func overlapBytes(a, b *transient, align int) bool {
	ah := a.offset + alignUp(a.bytes, align)
	bh := b.offset + alignUp(b.bytes, align)
	return a.offset < bh && b.offset < ah
}

func (t *transient) lastUser() NodeID {
	if len(t.users) == 0 {
		return -1
	}
	return t.users[len(t.users)-1]
}

// requireBarrier makes sure a barrier exists before a node, adding the aliasing
// reason to one already there rather than emitting a second.
func (g *Graph) requireBarrier(before NodeID, from NodeID) {
	b := g.barriersBefore[before]
	if b == nil {
		b = &barrier{src: stageTransfer | stageCompute, dst: g.nodes[before].stage, memory: true}
		g.barriersBefore[before] = b
	}
	b.memory = true
	b.reasons = append(b.reasons, barrierReason{from: from, label: "aliasing handover"})
}
