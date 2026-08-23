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
func (p *Plan) Submit(q *accel.Queue, bindings Bindings) *accel.Fence {
	if err := p.bind(bindings); err != nil {
		return accel.FailedFence(err)
	}
	return q.Submit(p.graph)
}

// bind validates and applies the whole binding set.
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

	if err := errors.Join(sorted(errs)...); err != nil {
		return err
	}

	// Applied only now that everything checked out.
	batch := make([]accel.Binding, 0, len(p.ports))
	for _, d := range p.ports {
		batch = append(batch, accel.Binding{
			Slot: p.slots[d.Name], Buffer: bindings.Buffers[d.Name],
		})
	}
	return p.graph.Rebind(batch)
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
	if p.closed {
		return nil
	}
	p.closed = true
	p.rt.plans--
	return p.graph.Close()
}
