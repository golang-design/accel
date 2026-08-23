// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"errors"
	"fmt"

	"golang.design/x/accel/internal/driver"
)

// bind resolves one slot, or reports why it cannot.
//
// The checks here are validation rows V2, V3, V4, V5, V6 and V19 against the
// [SlotDescriptor], which is one of the two declarations a bound resource has
// to satisfy. The other is the pipeline's binding layout, checked where a node
// uses one; specs/015-graph-recording.md section 4 records why collapsing the
// two would drop whichever declaration the survivor did not consult.
func (g *Graph) checkBinding(b SlotBinding) (driver.SlotBinding, error) {
	d, ok := g.descriptor(b.Slot)
	if !ok {
		return driver.SlotBinding{}, fmt.Errorf("accel: Bind: slot %d is not one of this graph's %d",
			int(b.Slot), len(g.slots))
	}
	v := b.Buffer
	if v.Buffer == nil {
		return driver.SlotBinding{}, fmt.Errorf("accel: Bind %q: no resource bound", d.Name)
	}

	// V2: the kind the descriptor declared.
	switch d.Kind {
	case BindingStorageBuffer, BindingUniformBuffer:
	default:
		return driver.SlotBinding{}, fmt.Errorf("accel: Bind %q: a buffer was bound to a %v slot",
			d.Name, d.Kind)
	}
	// V3: dtype agreement, exactly. A reinterpreting view is legal on a buffer
	// and not on a slot, because the recorded nodes computed their byte offsets
	// from the descriptor's dtype.
	if v.DType != d.DType {
		return driver.SlotBinding{}, fmt.Errorf("accel: Bind %q: the slot is %v and the view is %v",
			d.Name, d.DType, v.DType)
	}
	// V19: the resource is this device's and open. Repeated at submit, because a
	// caller may close it after binding.
	if err := v.check("Bind " + d.Name); err != nil {
		return driver.SlotBinding{}, err
	}
	if v.Buffer.pool.dev != g.dev {
		return driver.SlotBinding{}, fmt.Errorf("accel: Bind %q: %q belongs to a different device",
			d.Name, v.Buffer.desc.Label)
	}
	// V5: large enough for what the recorded nodes declared.
	if v.Count < d.MinCount {
		return driver.SlotBinding{}, fmt.Errorf("accel: Bind %q: the recorded nodes need %d "+
			"elements and the view has %d", d.Name, d.MinCount, v.Count)
	}
	// V6: the usage the access needs was declared when the buffer was created.
	if err := g.checkUsage(d, v.Buffer); err != nil {
		return driver.SlotBinding{}, err
	}

	off, size := v.byteRange()
	return driver.SlotBinding{
		Slot:   int(b.Slot),
		Block:  v.Buffer.pool.block,
		Offset: v.Buffer.alloc.Offset + off,
		Size:   size,
	}, nil
}

func (g *Graph) checkUsage(d SlotDescriptor, b *Buffer) error {
	var need BufferUsage
	switch d.Kind {
	case BindingUniformBuffer:
		need = UsageUniform
	default:
		need = UsageStorage
	}
	if b.desc.Usage&need == 0 {
		return fmt.Errorf("accel: Bind %q: this slot needs %v and %q was created with %v",
			d.Name, need, b.desc.Label, b.desc.Usage)
	}
	return nil
}

func (g *Graph) descriptor(s Slot) (SlotDescriptor, bool) {
	if s < 1 || int(s) > len(g.slots) {
		return SlotDescriptor{}, false
	}
	return g.slots[s-1], true
}

// bindAll validates a whole batch, then applies it or none of it.
func (g *Graph) bindAll(bs []SlotBinding) error {
	if err := g.state.checkOpen("Bind"); err != nil {
		return err
	}
	if lost := g.dev.dev.Lost(); lost != nil {
		return lost
	}

	staged := make([]driver.SlotBinding, 0, len(bs))
	var errs []error
	for _, b := range bs {
		sb, err := g.checkBinding(b)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		staged = append(staged, sb)
	}
	if err := errors.Join(errs...); err != nil {
		return err
	}

	g.mu.Lock()
	defer g.mu.Unlock()
	if g.inFlight {
		return ErrGraphInFlight
	}
	// V24, dynamic against dynamic and dynamic against concrete. The transient
	// term lands with specs/017-graph-aliasing.md, when transients have
	// placements to collide over.
	if err := g.checkOverlaps(bs); err != nil {
		return err
	}
	if err := g.exe.Rebind(staged); err != nil {
		return err
	}
	for _, b := range bs {
		g.bound[b.Slot] = b
	}
	return nil
}

// checkOverlaps is V24.
//
// It is graph-wide rather than per node, and that is the point. Hazards were
// inferred against the *slot*, because a slot's eventual resource is unknown at
// build. If two slots later resolve to overlapping bytes, or one resolves over
// a concrete resource the graph already names, then the builder may have
// omitted a read-after-write edge between nodes far apart in the graph, and
// checking only two bindings of one node would miss exactly that.
//
// **Only pairs involving a slot are compared.** Two concrete resources
// overlapping is not a violation and never was: their identity and their exact
// ranges were known when the edges were inferred, so whatever hazards exist
// between them were already expressed. Comparing them here would reject a graph
// that writes one buffer from two nodes, which is ordinary.
//
// An overlap is accepted only when both sides' graph-wide access unions are
// read-only. g.mu is held.
func (g *Graph) checkOverlaps(updates []SlotBinding) error {
	// The prospective binding set: what is already bound, with this batch
	// applied. Checking the batch against the old state would let a batch that
	// swaps two slots pass while leaving them overlapping.
	slots := g.spans[:0]
	for s := 1; s <= len(g.slots); s++ {
		v := g.bound[s].Buffer
		for _, u := range updates {
			if u.Slot == Slot(s) {
				v = u.Buffer
			}
		}
		if v.Buffer == nil {
			continue
		}
		off, size := v.byteRange()
		base := v.Buffer.alloc.Offset + off
		slots = append(slots, span{
			slot:  Slot(s),
			buf:   v.Buffer,
			lo:    base,
			hi:    base + size,
			write: g.slotWriter[s],
		})
	}
	g.spans = slots

	for i := range slots {
		for j := i + 1; j < len(slots); j++ {
			if err := g.conflict(slots[i], slots[j]); err != nil {
				return err
			}
		}
		for j := range g.concrete {
			if err := g.conflict(slots[i], g.concrete[j]); err != nil {
				return err
			}
		}
	}
	return nil
}

// span is one byte range a graph reaches, with its graph-wide access union.
//
// It names its origin by slot or by node rather than carrying a formatted
// string, because a span is built on the rebind path and formatting one costs
// an allocation per binding whether or not anything is wrong.
type span struct {
	slot  Slot   // non-zero for a dynamic span
	node  NodeID // meaningful when slot is zero
	buf   *Buffer
	lo    int
	hi    int
	write bool
}

func (g *Graph) describe(s span) string {
	if s.slot != 0 {
		return fmt.Sprintf("slot %q", g.slots[s.slot-1].Name)
	}
	return fmt.Sprintf("node %d's use of %q", int(s.node), s.buf.desc.Label)
}

// conflict reports an overlap that at least one side writes.
func (g *Graph) conflict(a, b span) error {
	if a.buf != b.buf || !(a.write || b.write) {
		return nil
	}
	if a.lo >= b.hi || b.lo >= a.hi {
		return nil
	}
	return fmt.Errorf("%w: %s and %s both name bytes [%d, %d) of %q, and at least "+
		"one of them writes: the inferred hazards treated them as independent",
		ErrRebindOverlap, g.describe(a), g.describe(b),
		max(a.lo, b.lo), min(a.hi, b.hi), a.buf.desc.Label)
}

// slotWrites reports whether any node writes through a slot anywhere in the
// graph. Graph-wide, because that is the union V24 compares.
func (g *Graph) slotWrites(s Slot) bool {
	for i := range g.nodes {
		for _, a := range g.nodes[i].accesses {
			if a.res.slot == s && a.writes() {
				return true
			}
		}
	}
	return false
}

// concreteSpans is every byte range the graph names through a resource it
// already holds, with each range's graph-wide access union.
//
// Transients are excluded, and stay excluded now that they have placements.
// Nothing a caller can bind reaches them: BufferView.check refuses a transient
// offered through a slot outright, which is stricter than an overlap test, and
// the builder's memory has no public allocator. specs/017-graph-aliasing.md §5
// records why the term 015 §4 expected here is unreachable rather than missing.
func (g *Graph) concreteSpans() []span {
	var out []span
	for i := range g.nodes {
		n := &g.nodes[i]
		for _, a := range n.accesses {
			b := a.res.buf
			if b == nil || b.transient != nil {
				continue
			}
			base := b.alloc.Offset + a.off
			out = append(out, span{
				node: n.id, buf: b, lo: base, hi: base + a.size, write: a.writes(),
			})
		}
	}
	return out
}

// run executes the graph once, on the queue's stream, and returns when it is
// done. It is called from inside [Queue.enqueue], which is what orders it
// against everything else on that queue.
func (g *Graph) run() error {
	if err := g.state.checkOpen("Submit"); err != nil {
		return err
	}
	if lost := g.dev.dev.Lost(); lost != nil {
		return lost
	}

	g.mu.Lock()
	if g.inFlight {
		g.mu.Unlock()
		return ErrGraphInFlight
	}
	// V19 again, and V1 for slots: a caller may have closed a bound resource
	// since binding it, and a rebindable slot may legally have been empty until
	// now. Both are checked inside the same critical section that marks the
	// graph in flight, so there is no validate-then-submit gap.
	if err := g.checkBoundAtSubmit(); err != nil {
		g.mu.Unlock()
		return err
	}
	g.inFlight = true
	shared := g.shared
	g.mu.Unlock()

	defer func() {
		g.mu.Lock()
		g.inFlight = false
		g.mu.Unlock()
	}()

	// A shared transient pool widens the rule above from this graph to every
	// graph planned into the same bytes: see specs/031-shared-transients.md.
	//
	// The claim is taken here, and not in Queue.Submit, for two reasons. This
	// function is the only one that executes a graph, so Queue.SubmitAfter is
	// covered by the same three lines rather than by a second copy somebody has
	// to remember. And the claim then spans execution and nothing else: taken
	// at call time it would also span the wait for everything already on the
	// queue, so a back-to-back pair of submissions that a serial queue could
	// never overlap would be refused, or not, depending on goroutine
	// scheduling. The refusal reaches the caller through the fence, which is
	// how every other submission failure arrives.
	if shared != nil {
		if err := shared.begin(); err != nil {
			return err
		}
		defer shared.end()
	}

	df, err := g.exe.Submit()
	if err != nil {
		return err
	}
	return df.Wait()
}

// readStats collects the counters a completed submission wrote, when the graph
// asked for them.
//
// Through an optional interface, because a backend that cannot produce them
// should be discovered by assertion rather than by a method that fails when
// called. See specs/006-backends.md section 1.
func (g *Graph) readStats() SubmissionStats {
	if !g.collectStats {
		return SubmissionStats{}
	}
	r, ok := g.exe.(driver.StatsReporter)
	if !ok {
		return SubmissionStats{}
	}
	raw := r.IndirectStats()
	out := SubmissionStats{Indirect: make([]IndirectStats, len(raw))}
	for i, s := range raw {
		out.Indirect[i] = IndirectStats{
			Node:    NodeID(s.Node),
			Actual:  [3]uint32{s.Actual.X, s.Actual.Y, s.Actual.Z},
			Max:     [3]uint32{s.Max.X, s.Max.Y, s.Max.Z},
			Clamped: s.Clamped,
		}
	}
	return out
}

func (g *Graph) checkBoundAtSubmit() error {
	var errs []error
	for s := 1; s <= len(g.slots); s++ {
		b := g.bound[s]
		if b.Buffer.Buffer == nil {
			errs = append(errs, fmt.Errorf("accel: Submit: slot %q has no resource bound",
				g.slots[s-1].Name))
			continue
		}
		if err := b.Buffer.check("Submit " + g.slots[s-1].Name); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}
