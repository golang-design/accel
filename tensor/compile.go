// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"errors"
	"fmt"

	"golang.design/x/accel"
)

// CompileOptions carries what a plan needs beyond the graph.
type CompileOptions struct {
	Label string
}

// KernelSelection reports which kernel an operator became, and why.
//
// Reported rather than merely decided, because specs/007-tensor-layer.md makes
// fused attention a *selection* rather than a capability: a caller who cannot
// see which they got cannot explain a performance cliff or a numeric
// difference, and would have to guess from timings.
type KernelSelection struct {
	Op       string
	Kernel   string
	Reason   string
	Rejected []string
}

// uniformNode remembers which recorded dispatch a scalar feeds.
type uniformNode struct {
	node  accel.NodeID
	build func(map[string]ScalarValue) any
}

// Plan is a compiled tensor graph, ready to submit.
//
// It owns one device graph and the transient memory the planner chose; the
// caller owns every buffer they named. Immutable, and with the graph's
// one-submission-in-flight restriction.
type Plan struct {
	rt         *Runtime
	graph      *accel.Graph
	label      string
	ports      []PortDesc
	slots      map[string]accel.Slot
	selections []KernelSelection

	scalars   []ScalarDesc
	scalarPos map[string]int

	// uniformNodes is every dispatch that reads a scalar, so a submission can
	// rewrite its by-value parameter before the graph runs.
	uniformNodes []uniformNode

	closed bool
}

// Bindings is everything a submission needs from the caller.
type Bindings struct {
	Buffers map[string]accel.BufferView
	Scalars map[string]ScalarValue
}

// Compile turns the recorded graph into a plan.
//
// Every collected error is returned together rather than the first one, because
// a model with three mistakes should take one compile to find all three. Each
// names the operator, the operand, and the line that recorded it.
func (b *Builder) Compile(rt *Runtime, opts CompileOptions) (*Plan, error) {
	if rt == nil {
		return nil, errors.New("accel/tensor: Compile needs a runtime")
	}
	if b.rt != nil && b.rt != rt {
		return nil, errors.New("accel/tensor: this builder belongs to another runtime")
	}
	// Reported only when nothing else went wrong. Output ignores a poisoned
	// value, so a graph whose only mistake is upstream has no outputs *because*
	// of that mistake -- and saying so as well would be the second diagnostic
	// the poisoned-tensor rule exists to prevent, arriving from a different
	// direction.
	if err := b.Err(); err != nil {
		return nil, err
	}
	if len(b.outputs) == 0 {
		return nil, errors.New("accel/tensor: Compile: the graph declares no output, so " +
			"nothing it computes would be readable")
	}

	label := opts.Label
	if label == "" {
		label = b.label
	}
	p := &Plan{
		rt: rt, label: label, ports: b.ports, slots: map[string]accel.Slot{},
		scalars: b.scalars, scalarPos: b.scalarPos,
	}

	r := rt.dev.NewRecorder()

	// Every external port is a slot, so one plan serves many submissions with
	// different buffers. specs/003-command-graph.md's slots are exactly this
	// and there is no reason to invent a second mechanism.
	// An output that something else also reads is read-write, not write-only.
	// A model that names an intermediate as an output and keeps using it is
	// ordinary, and the alternative is refusing it or writing it twice.
	readAgain := map[*Tensor]bool{}
	for i := range b.nodes {
		for _, in := range b.nodes[i].inputs {
			readAgain[in] = true
		}
	}
	consumed := map[string]bool{}
	for _, o := range b.outputs {
		if readAgain[o.t] {
			consumed[o.name] = true
		}
	}

	for _, d := range b.ports {
		access := accel.AccessRead
		switch {
		case d.Kind == PortOutput && consumed[d.Name]:
			access = accel.AccessReadWrite
		case d.Kind == PortOutput:
			access = accel.AccessWrite
		case d.Kind == PortState:
			access = accel.AccessReadWrite
		}
		p.slots[d.Name] = r.Slot(accel.SlotDescriptor{
			Name: d.Name, Kind: accel.BindingStorageBuffer,
			DType: d.DType, Access: access, MinCount: d.Shape.Elements(),
		})
	}

	// Intermediates are transients, so the graph's aliasing and barrier
	// planning apply unchanged. A tensor layer allocating its own would
	// reimplement specs/017-graph-aliasing.md, worse.
	views := make([]accel.BufferView, len(b.nodes))
	// A node that wrote into a caller-owned port has no transient, so its
	// consumers bind that port's slot instead. Parallel to views because a node
	// has exactly one result and it is one or the other.
	wroteSlot := make([]accel.Slot, len(b.nodes))
	outputOf := map[*Tensor]string{}
	for _, o := range b.outputs {
		outputOf[o.t] = o.name
	}

	var errs []error
	for i := range b.nodes {
		n := &b.nodes[i]
		if err := p.lowerNode(r, n, views, wroteSlot, outputOf, i); err != nil {
			errs = append(errs, err)
		}
	}
	if err := errors.Join(errs...); err != nil {
		return nil, err
	}

	// An output that names an external port rather than a computed value would
	// be a copy between two caller buffers. Refused rather than emitted,
	// because a caller who wanted that wrote a graph that computes nothing and
	// almost certainly meant something else -- and the copy they can write
	// themselves is one line.
	for _, o := range b.outputs {
		if o.t.node < 0 {
			return nil, fmt.Errorf("accel/tensor: output %q is an input or weight rather "+
				"than a computed value; a plan that only copies is a copy, not a plan",
				o.name)
		}
	}

	g, err := r.Build()
	if err != nil {
		return nil, fmt.Errorf("accel/tensor: compiling %q: %w", label, err)
	}
	p.graph = g
	rt.plans++
	return p, nil
}

// lowerNode turns one operator into a dispatch.
func (p *Plan) lowerNode(r *accel.Recorder, n *node, views []accel.BufferView,
	wroteSlot []accel.Slot, outputOf map[*Tensor]string, i int) error {

	pipe, err := p.rt.pipeline(n.kernel, fmt.Sprintf("%s.%s", p.label, n.op))
	if err != nil {
		return fmt.Errorf("accel/tensor: %s: %w", n.op, err)
	}
	p.selections = append(p.selections, KernelSelection{
		Op: n.op, Kernel: n.kernel.Name, Reason: n.reason, Rejected: n.rejected,
	})

	var inPlaceResult *accel.Binding
	var inPlaceScratch accel.BufferView
	inPlaceCount := 0

	binds := make([]accel.Binding, 0, len(n.inputs)+2)
	if n.uniform != nil {
		// A placeholder, rewritten before every submission. Recording the zero
		// and never rewriting it would be a plan that ran and computed with a
		// factor of nothing, which is why Submit rewrites unconditionally
		// rather than only when a value changed.
		binds = append(binds, accel.Binding{Index: 0, Uniform: n.uniform(nil)})
	}

	// The result: a transient unless the caller asked for it by name, in which
	// case it is written straight into their buffer and no copy exists.
	//
	// Resolved before the operands because an in-place kernel reads and writes
	// the same binding, so its operand *is* the result.
	outIndex := len(n.inputs)
	var result accel.Binding
	if n.outPort != "" {
		// A mutation of caller-owned state: the node writes into the bound
		// buffer rather than into a transient, which is what makes the write
		// visible after the submission and what lets the graph order a later
		// read against it.
		result = accel.Binding{Slot: p.slots[n.outPort]}
		wroteSlot[i] = result.Slot
	} else if name, wanted := outputOf[n.out]; wanted {
		wroteSlot[i] = p.slots[name]
		result = accel.Binding{Slot: p.slots[name]}
	} else {
		v := r.Transient(accel.BufferDescriptor{
			DType: n.out.dtype, Count: n.out.shape.Elements(),
			Usage: accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst,
			Label: fmt.Sprintf("%s.%s.%d", p.label, n.op, i),
		})
		views[i] = v
		result = accel.Binding{Buffer: v}
	}

	for j, in := range n.inputs {
		bind, err := p.operand(in, views, wroteSlot)
		if err != nil {
			return fmt.Errorf("accel/tensor: %s operand %d: %w", n.op, j, err)
		}
		if !in.contiguousLayout() && !n.bcast {
			return fmt.Errorf("accel/tensor: %s operand %d is a strided view, and the "+
				"kernel indexes contiguously; the operators that pack an operand for you "+
				"are the elementwise ones, because specs/007-tensor-layer.md keeps a copy "+
				"of a matrix something a caller asks for", n.op, j)
		}
		// An elementwise operand that is not already the result's shape is
		// packed into a transient first. Reported in Selections rather than
		// done quietly: a copy nobody can see is a cost nobody can explain.
		if n.bcast && (!in.shape.Equal(n.out.shape) || !in.contiguousLayout()) {
			bind, err = p.materialize(r, n.op, in, n.out.shape, bind,
				fmt.Sprintf("%s.%s.%d.broadcast", p.label, n.op, i))
			if err != nil {
				return fmt.Errorf("accel/tensor: %s operand %d: %w", n.op, j, err)
			}
		}
		bind.Index = j
		binds = append(binds, bind)
	}

	if n.inPlace {
		// The kernel rewrites what it reads and a tensor is an immutable value,
		// so the work happens in a transient: the operand is copied in, the
		// kernel rewrites it, and the result is copied out if the caller named
		// it. A transient always, rather than the result buffer directly,
		// because the operand and the result may both be caller buffers and
		// the recorder moves bytes between a slot and a view rather than
		// between two slots.
		if len(binds) != outIndex+1 || outIndex != 1 {
			return fmt.Errorf("accel/tensor: %s is in place and takes %d operands; an "+
				"in-place kernel has exactly one", n.op, outIndex)
		}
		count := n.out.shape.Elements()
		scratch := r.Transient(accel.BufferDescriptor{
			DType: n.out.dtype, Count: count,
			Usage: accel.UsageStorage | accel.UsageCopySrc | accel.UsageCopyDst,
			Label: fmt.Sprintf("%s.%s.%d.inplace", p.label, n.op, i),
		})
		src := binds[len(binds)-1]
		if err := p.copyInto(r, accel.Binding{Buffer: scratch}, src, count); err != nil {
			return fmt.Errorf("accel/tensor: %s: %w", n.op, err)
		}
		p.selections = append(p.selections, KernelSelection{
			Op: n.op, Kernel: "copy",
			Reason: "copying the operand into scratch, because the kernel rewrites what it " +
				"reads and a tensor is a value",
		})
		binds = binds[:len(binds)-1]
		binds = append(binds, accel.Binding{Index: 0, Buffer: scratch})
		inPlaceResult = &result
		inPlaceScratch = scratch
		inPlaceCount = count
	} else {
		result.Index = outIndex
		binds = append(binds, result)
	}

	g := n.grid
	if g == nil {
		g = perElement(int(n.kernel.WorkgroupSize.X))
	}
	id := r.Dispatch(pipe, binds, g(n.out))
	if n.uniform != nil {
		p.uniformNodes = append(p.uniformNodes, uniformNode{node: id, build: n.uniform})
	}

	if inPlaceResult != nil {
		// The scratch buffer holds the answer. If the caller named this value
		// it has to reach their buffer; if not, the scratch *is* the value and
		// the consumers read it.
		if inPlaceResult.Buffer.Buffer == nil && inPlaceResult.Slot != 0 {
			r.CopyToSlot(inPlaceResult.Slot, 0, inPlaceCount, inPlaceScratch)
		}
		views[i] = inPlaceScratch
	}
	return nil
}

// copyInto records a copy from one binding into another.
func (p *Plan) copyInto(r *accel.Recorder, dst, src accel.Binding, count int) error {
	switch {
	case dst.Buffer.Buffer != nil && src.Buffer.Buffer != nil:
		r.CopyBuffer(dst.Buffer, src.Buffer)
	case dst.Buffer.Buffer != nil:
		r.CopyFromSlot(dst.Buffer, src.Slot, 0, count)
	case src.Buffer.Buffer != nil:
		r.CopyToSlot(dst.Slot, 0, count, src.Buffer)
	default:
		return fmt.Errorf("a copy between two bound resources is not lowered; make the " +
			"source an intermediate")
	}
	return nil
}

// operand resolves one operand to a binding.
//
// An external port becomes a slot, so the same plan serves many submissions; a
// computed value becomes the transient its producer wrote.
func (p *Plan) operand(t *Tensor, views []accel.BufferView,
	wroteSlot []accel.Slot) (accel.Binding, error) {
	if t.node < 0 {
		slot, ok := p.slots[t.port]
		if !ok {
			return accel.Binding{}, fmt.Errorf("the port %q has no slot", t.port)
		}
		return accel.Binding{Slot: slot}, nil
	}
	if v := views[t.node]; v.Buffer != nil {
		return accel.Binding{Buffer: v}, nil
	}
	// The producing node wrote into a caller-owned buffer rather than a
	// transient -- a mutation of state, or a value the caller named as an
	// output. Either way the consumer binds the same slot, which is also what
	// makes the graph order the two: they declare overlapping byte ranges.
	if slot := wroteSlot[t.node]; slot != 0 {
		return accel.Binding{Slot: slot}, nil
	}
	return accel.Binding{}, fmt.Errorf("it reads a value that was written nowhere")
}
