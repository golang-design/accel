// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"errors"
	"fmt"
	"sort"

	"golang.design/x/accel"
)

// Ports reports every external buffer this plan expects, in declaration order.
//
// A caller binds by name, so this is how they discover what to bind without
// having kept the builder. Sorted by declaration rather than by name, because
// declaration order is the order a reader of the model saw them.
func (p *Plan) Ports() []PortDesc {
	out := make([]PortDesc, len(p.ports))
	copy(out, p.ports)
	return out
}

// Scalars reports every named per-step value this plan reads.
func (p *Plan) Scalars() []ScalarDesc {
	out := make([]ScalarDesc, len(p.scalars))
	copy(out, p.scalars)
	return out
}

// Selections reports which kernel each operator became, and why.
func (p *Plan) Selections() []KernelSelection {
	out := make([]KernelSelection, len(p.selections))
	copy(out, p.selections)
	return out
}

// Memory reports what the plan's intermediates cost after the graph planner
// aliased them.
//
// Straight from the device graph rather than counted here, which is the point:
// the intermediates went through the recorder, so they got specs/017's aliasing
// and this number is evidence of it rather than a second accounting.
func (p *Plan) Memory() accel.GraphMemory { return p.graph.Memory() }

// Submit binds and runs the plan, returning without waiting.
//
// The whole binding set is validated before anything is bound, so a submission
// with one bad name leaves the plan exactly as it was. A failure comes back
// through an already-signalled fence rather than as a second return value,
// which matches specs/003-command-graph.md and means a caller checks one thing.
//
// # One submission at a time
//
// specs/007-tensor-layer.md gives a plan "the Graph's one-submission-in-flight
// restriction", and this is where it has to be enforced rather than inherited.
// Binding happens here and synchronously; the submission is handed to the
// queue's serial stream and runs later. So a second Submit before the first has
// run would rebind the slots underneath it -- and the graph's own in-flight
// check does not catch that, because the graph is not marked in flight until
// its worker reaches it.
//
// That was not theoretical. Two submissions with different inputs and different
// outputs, back to back, produced *one* result: the first submission wrote into
// the second's output buffer, and both fences reported success. A silently lost
// result is the worst failure available here, so this refuses instead.
func (p *Plan) Submit(q *accel.Queue, bindings Bindings) *accel.Fence {
	// Held across the bind and the submit together, because that pair is
	// exactly what must not interleave: two callers binding and then submitting
	// would each pass the in-flight check and the second would rebind the
	// first's resources, which is the bug this check exists for arriving by a
	// different route.
	p.mu.Lock()
	defer p.mu.Unlock()

	if f := p.inFlight; f != nil && !f.Done() {
		return accel.FailedFence(fmt.Errorf("accel/tensor: Submit %q while a submission is "+
			"in flight; a plan binds when you submit and runs when the queue reaches it, so "+
			"a second submission would rebind the first one's resources. Wait on the fence, "+
			"or compile a second plan", p.label))
	}
	if err := p.bind(bindings); err != nil {
		return accel.FailedFence(err)
	}
	f := q.Submit(p.graph)
	p.inFlight = f
	return f
}

// bind validates and applies the whole binding set.
//
// The caller holds p.mu.
func (p *Plan) bind(bindings Bindings) error {
	if p.closed {
		return errors.New("accel/tensor: Submit: the plan is closed")
	}

	var errs []error

	// Every declared port must be supplied, and nothing else may be. An extra
	// name is almost always a typo in the name of a port that is therefore also
	// missing, so reporting both is what makes the mistake obvious.
	declared := map[string]PortDesc{}
	for _, d := range p.ports {
		declared[d.Name] = d
		v, ok := bindings.Buffers[d.Name]
		if !ok {
			errs = append(errs, fmt.Errorf("accel/tensor: %s %q is not bound", d.Kind, d.Name))
			continue
		}
		if v.DType != d.DType {
			errs = append(errs, fmt.Errorf("accel/tensor: %q is declared %v and the bound "+
				"view is %v", d.Name, d.DType, v.DType))
			continue
		}
		if want := d.Shape.Elements(); v.Count < want {
			errs = append(errs, fmt.Errorf("accel/tensor: %q has shape %v, which is %d "+
				"elements, and the bound view has %d", d.Name, d.Shape, want, v.Count))
		}
	}
	for name := range bindings.Buffers {
		if _, ok := declared[name]; !ok {
			errs = append(errs, fmt.Errorf("accel/tensor: %q is bound and this plan declares "+
				"no such port", name))
		}
	}

	// Scalars, the same way. The wrong *kind* is the case worth catching: the
	// bytes pack either way and the kernel reads a float as an integer or the
	// reverse, then computes something plausible.
	for _, s := range p.scalars {
		v, ok := bindings.Scalars[s.Name]
		if !ok {
			errs = append(errs, fmt.Errorf("accel/tensor: the scalar %q is not bound", s.Name))
			continue
		}
		if v.Kind != s.Kind {
			errs = append(errs, fmt.Errorf("accel/tensor: the scalar %q is declared %v and "+
				"the bound value is %v", s.Name, s.Kind, v.Kind))
		}
	}
	for name := range bindings.Scalars {
		if _, ok := p.scalarPos[name]; !ok {
			errs = append(errs, fmt.Errorf("accel/tensor: the scalar %q is bound and this "+
				"plan declares no such scalar", name))
		}
	}

	if err := errors.Join(sorted(errs)...); err != nil {
		return err
	}

	// Applied only now that everything checked out.
	// One binding per window rather than per port. A plan using no LayerState
	// has exactly one window per port and this is the loop it always was; a
	// per-layer cache has one per (port, layer) and the caller still binds the
	// port once, which is the whole point of the view.
	batch := make([]accel.SlotBinding, 0, len(p.windows))
	for _, w := range p.windows {
		parent := bindings.Buffers[w.port]
		v := parent
		if w.off != 0 || w.count != parent.Count {
			sub, err := parent.Buffer.View(parent.Offset+w.off, w.count)
			if err != nil {
				return fmt.Errorf("accel/tensor: %q elements [%d,%d) of the %d bound: %w",
					w.port, w.off, w.off+w.count, parent.Count, err)
			}
			v = sub
		}
		batch = append(batch, accel.SlotBinding{Slot: p.slots[w], Buffer: v})
	}
	if err := p.graph.Bind(batch...); err != nil {
		return err
	}

	// The by-value parameters last, and unconditionally: a plan that rewrote
	// only what changed would compute with the placeholder on its first
	// submission.
	for _, u := range p.uniformNodes {
		if err := p.graph.SetUniform(u.node, 0, u.build(bindings.Scalars)); err != nil {
			return err
		}
	}
	return nil
}

// sorted orders diagnostics so a report is stable between runs.
func sorted(errs []error) []error {
	msgs := make([]string, 0, len(errs))
	for _, e := range errs {
		if e != nil {
			msgs = append(msgs, e.Error())
		}
	}
	sort.Strings(msgs)
	out := make([]error, len(msgs))
	for i, m := range msgs {
		out[i] = errors.New(m)
	}
	return out
}

// Close releases the plan's graph and transient memory.
func (p *Plan) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return nil
	}
	p.closed = true
	p.rt.planClosed()
	return p.graph.Close()
}
