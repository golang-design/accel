// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"errors"
	"fmt"
	"golang.design/x/accel/kernelabi"
	"strings"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
)

// newComputePipeline validates a compiled kernel against the device and bakes
// its metadata in.
//
// The checks here are validation rows V10, V11, and V17, and they happen at
// pipeline creation rather than at dispatch because that is where a caller can
// still do something about them: a graph is built once and submitted many
// times, so a limit violation discovered at submit would be discovered on the
// hot path.
func (d *Device) newComputePipeline(desc ComputePipelineDescriptor) (*ComputePipeline, error) {
	if err := d.state.checkOpen("NewComputePipeline"); err != nil {
		return nil, err
	}
	if lost := d.dev.Lost(); lost != nil {
		return nil, lost
	}
	k := desc.Kernel
	if k == nil {
		return nil, errors.New("accel: NewComputePipeline: the descriptor names no kernel")
	}
	label := desc.Label
	if label == "" {
		label = k.Name
	}

	// The generated record is the source of truth for everything static about a
	// kernel, so this is a check on it rather than on a second declaration the
	// descriptor might have carried. See specs/006-backends.md R6.
	if k.Generator != kernelabi.Version {
		return nil, fmt.Errorf("accel: NewComputePipeline %q: kernel %q was generated against "+
			"ABI %d and this runtime is ABI %d; re-run go generate",
			label, k.Name, k.Generator, kernelabi.Version)
	}
	// A kernel has exactly one entry point: flat if its body reaches no
	// barrier, shared memory, or subgroup operation, and cooperative otherwise.
	// Neither is a kernel the generator did not finish.
	if k.Flat == nil && k.Cooperative == nil {
		return nil, fmt.Errorf("accel: NewComputePipeline %q: kernel %q has neither a flat "+
			"nor a cooperative entry point, so the generated file is incomplete; re-run "+
			"go generate", label, k.Name)
	}

	size := k.WorkgroupSize
	for _, c := range [3]struct {
		axis string
		v    uint32
	}{{"X", size.X}, {"Y", size.Y}, {"Z", size.Z}} {
		if c.v == 0 {
			return nil, fmt.Errorf("accel: NewComputePipeline %q: workgroup extent %s is zero, "+
				"and an axis of zero dispatches nothing", label, c.axis)
		}
	}

	// V10, V11, and V17 in one call. Requirements are what the kernel body
	// implies, derived by the compiler and never declared by hand, so this
	// compares the record against what the device reports rather than trusting
	// a second declaration the descriptor might have carried.
	if unmet := d.Missing(requirementsOf(k)); len(unmet) > 0 {
		return nil, fmt.Errorf("accel: NewComputePipeline %q: %q does not meet what kernel %q "+
			"requires: %s", label, d.info.Name, k.Name, describeUnmet(unmet))
	}

	p := &ComputePipeline{dev: d, kernel: k, label: label}
	p.state.init(label)
	d.countPipelines(1)
	return p, nil
}

// requirementsOf reads what a compiled kernel needs out of its record.
//
// Every field comes from the record, including the capabilities: they are
// inferred by the compiler from what the body reaches, never declared by the
// kernel's author. A declaration can be forgotten, and the failure is silent --
// a kernel using a feature the device lacks would produce wrong results rather
// than an error, because nothing checked. See
// specs/020-cooperative-atomics.md section 3.
func requirementsOf(k *Kernel) Requirements {
	s := k.WorkgroupSize
	return Requirements{
		Caps:                 Capability(k.Caps),
		WorkgroupSize:        [3]uint32{s.X, s.Y, s.Z},
		WorkgroupInvocations: s.X * s.Y * s.Z,
	}
}

func describeUnmet(u []Unmet) string {
	parts := make([]string, len(u))
	for i, m := range u {
		if m.Limit != "" {
			parts[i] = fmt.Sprintf("%s needs %d and the device reports %d", m.Limit, m.Required, m.Available)
		} else {
			parts[i] = fmt.Sprintf("the capability %v is absent", m.Cap)
		}
	}
	return strings.Join(parts, "; ")
}

// countPipelines adjusts the device's live pipeline count, so that closing a
// device under one is refused rather than stranding it.
func (d *Device) countPipelines(delta int) {
	d.mu.Lock()
	d.pipelines += delta
	d.mu.Unlock()
}

// dispatchImpl records a compute dispatch.
//
// Accesses come from the kernel's binding layout, not from the caller: the
// access mode was inferred from the kernel body by the compiler, and letting a
// caller restate it would let them under-declare, which is how a missing
// dependency becomes a race.
func (r *Recorder) dispatchImpl(p *ComputePipeline, bs []Binding, us []UniformValue, count WorkgroupCount) NodeID {
	if p == nil {
		r.fail("Dispatch: no pipeline")
		return r.node(NodeDispatch, "Dispatch", nil, nil)
	}
	if err := p.state.checkOpen("Dispatch"); err != nil {
		r.state.errs = append(r.state.errs, err)
		return r.node(NodeDispatch, p.label, nil, nil)
	}
	if p.dev != r.state.dev {
		r.fail("Dispatch %q: the pipeline belongs to a different device", p.label)
		return r.node(NodeDispatch, p.label, nil, nil)
	}

	// V8: the recorded count normalizes an omitted Y or Z to one, needs a
	// positive X, and stays within the per-dimension limit.
	c, ok := r.normalizeCount(p.label, count)
	if !ok {
		return r.node(NodeDispatch, p.label, nil, nil)
	}

	accesses, uniforms, ok := r.bindingAccesses(p, bs, us)
	if !ok {
		return r.node(NodeDispatch, p.label, nil, nil)
	}

	id := r.node(NodeDispatch, p.label, accesses, nil)
	n := &r.state.nodes[id]
	n.pipeline = p
	n.count = c
	n.uniforms = uniforms
	for _, a := range accesses {
		r.touch(id, a)
	}
	return id
}

func (r *Recorder) normalizeCount(label string, count WorkgroupCount) (kernel.ID3, bool) {
	c := WorkgroupCount{X: count.X, Y: max(count.Y, 1), Z: max(count.Z, 1)}
	if count.X <= 0 {
		r.fail("Dispatch %q: workgroup count X is %d, and a direct dispatch of zero "+
			"workgroups is a recording mistake rather than a skip", label, count.X)
		return kernel.ID3{}, false
	}
	lim := r.state.dev.info.Limits.MaxWorkgroupCount
	for i, v := range [3]int{c.X, c.Y, c.Z} {
		if v > lim[i] {
			r.fail("Dispatch %q: workgroup count %s is %d and %q allows %d",
				label, [3]string{"X", "Y", "Z"}[i], v, r.state.dev.info.Name, lim[i])
			return kernel.ID3{}, false
		}
	}
	return kernel.ID3{X: uint32(c.X), Y: uint32(c.Y), Z: uint32(c.Z)}, true
}

// bindingAccesses turns a caller's bindings into declarations, one per entry of
// the kernel's layout and in that order.
func (r *Recorder) bindingAccesses(p *ComputePipeline, bs []Binding, us []UniformValue) ([]access, []any, bool) {
	layout := p.kernel.Bindings
	out := make([]access, len(layout))
	filled := make([]bool, len(layout))
	uniforms := make([]any, len(p.kernel.Uniforms))
	ok := true

	// A by-value parameter travels with the node rather than through a binding:
	// a caller supplies one as a value and the other as a slice, and a record
	// that conflated them would reinterpret a mismatched argument set rather
	// than refusing it. See specs/014-kernel-uniforms.md.
	for _, u := range us {
		if u.Index < 0 || u.Index >= len(uniforms) {
			r.fail("Dispatch %q: uniform index %d is outside the kernel's %d by-value "+
				"parameters", p.label, u.Index, len(uniforms))
			ok = false
			continue
		}
		if u.Value == nil {
			r.fail("Dispatch %q: by-value parameter %d has a nil value", p.label, u.Index)
			ok = false
			continue
		}
		uniforms[u.Index] = u.Value
	}
	for i, u := range p.kernel.Uniforms {
		if uniforms[i] == nil {
			r.fail("Dispatch %q: by-value parameter %q at index %d has no value",
				p.label, u.Name, i)
			ok = false
		}
	}

	for _, b := range bs {
		// V1's index half: an entry outside the layout names nothing.
		if b.Index < 0 || b.Index >= len(layout) {
			r.fail("Dispatch %q: binding index %d is outside the kernel's %d entries",
				p.label, b.Index, len(layout))
			ok = false
			continue
		}
		slot := layout[b.Index]
		if filled[b.Index] {
			r.fail("Dispatch %q: binding %q is bound twice", p.label, slot.Name)
			ok = false
			continue
		}
		a, good := r.bindingAccess(p, b, slot)
		if !good {
			ok = false
			continue
		}
		out[b.Index] = a
		filled[b.Index] = true
	}

	// V1: every entry has a binding. A slot left empty is not deferred to submit
	// the way a graph slot is, because a pipeline's layout is fixed.
	for i, f := range filled {
		if !f {
			r.fail("Dispatch %q: binding %q at index %d has no resource bound",
				p.label, layout[i].Name, i)
			ok = false
		}
	}
	return out, uniforms, ok
}

func (r *Recorder) bindingAccess(p *ComputePipeline, b Binding, slot kernel.Binding) (access, bool) {
	mode := publicAccess(slot.Access)
	if b.Texture != nil {
		r.fail("Dispatch %q: binding %q is a buffer and textures arrive with "+
			"specs/001-device-resources.md section 4", p.label, slot.Name)
		return access{}, false
	}

	if b.Slot != 0 {
		d, found := r.slotDescriptor(b.Slot)
		if !found {
			r.fail("Dispatch %q: binding %q names slot %d, which this recorder did not declare",
				p.label, slot.Name, int(b.Slot))
			return access{}, false
		}
		// V3 against the pipeline's layout, which is the second of the two
		// declarations a bound resource satisfies. The first is the slot
		// descriptor, checked at Bind. See specs/015-graph-recording.md section 4.
		if publicDType(slot.DType) != d.DType {
			r.fail("Dispatch %q: binding %q is %v and slot %q is %v",
				p.label, slot.Name, publicDType(slot.DType), d.Name, d.DType)
			return access{}, false
		}
		size := d.MinCount * d.DType.Size()
		return r.slotAccess("Dispatch "+p.label, b.Slot, 0, size, mode)
	}

	v := b.Buffer
	if v.Buffer == nil {
		r.fail("Dispatch %q: binding %q has no resource bound", p.label, slot.Name)
		return access{}, false
	}
	if want := publicDType(slot.DType); v.DType != want {
		r.fail("Dispatch %q: binding %q is %v and the view is %v",
			p.label, slot.Name, want, v.DType)
		return access{}, false
	}
	if v.Buffer.transient == nil && v.Buffer.desc.Usage&UsageStorage == 0 {
		r.fail("Dispatch %q: binding %q needs %v and %q was created with %v",
			p.label, slot.Name, UsageStorage, v.Buffer.desc.Label, v.Buffer.desc.Usage)
		return access{}, false
	}
	return r.declare("Dispatch "+p.label, v, mode)
}

// publicAccess maps a kernel's inferred access onto the graph's vocabulary.
func publicAccess(a kernel.Access) Access {
	switch a {
	case kernel.Read:
		return AccessRead
	case kernel.Write:
		return AccessWrite
	}
	return AccessReadWrite
}

// publicDType maps a binding's element type onto the public dtype. The two
// enums are mirrors, and a test asserts they stay in step.
func publicDType(d kernel.DType) DType {
	switch d {
	case kernel.F32:
		return F32
	case kernel.F16:
		return F16
	case kernel.BF16:
		return BF16
	case kernel.I32:
		return I32
	case kernel.U32:
		return U32
	case kernel.I8:
		return I8
	case kernel.U8:
		return U8
	}
	return DType(-1)
}

// indirectImpl records a dispatch whose workgroup count the device supplies.
//
// The count buffer is declared read, and read in the transfer stage rather than
// the compute one: the indirect fetch happens before the dispatch does, so a
// node that writes it must be ordered before this one by the fetch rather than
// by the kernel. Getting that wrong would be a hazard the barrier plan does not
// see.
func (r *Recorder) indirectImpl(p *ComputePipeline, bs []Binding, us []UniformValue,
	countBuf BufferView, max WorkgroupCount) NodeID {

	if p == nil {
		r.fail("DispatchIndirect: no pipeline")
		return r.node(NodeDispatchIndirect, "DispatchIndirect", nil, nil)
	}
	if err := p.state.checkOpen("DispatchIndirect"); err != nil {
		r.state.errs = append(r.state.errs, err)
		return r.node(NodeDispatchIndirect, p.label, nil, nil)
	}
	if p.dev != r.state.dev {
		r.fail("DispatchIndirect %q: the pipeline belongs to a different device", p.label)
		return r.node(NodeDispatchIndirect, p.label, nil, nil)
	}
	if !r.state.dev.info.Capabilities.IndirectDispatch {
		r.fail("DispatchIndirect %q: %q does not support indirect dispatch",
			p.label, r.state.dev.info.Name)
		return r.node(NodeDispatchIndirect, p.label, nil, nil)
	}

	// V9: the host-authored maximum follows the same normalization and limit
	// checks a direct count does. Without a maximum there is nothing to
	// validate at build and nothing to size transients against, and exceeding
	// the limit is undefined behaviour on Vulkan rather than a clean error.
	c, ok := r.normalizeCount("DispatchIndirect "+p.label, max)
	if !ok {
		return r.node(NodeDispatchIndirect, p.label, nil, nil)
	}

	// The count is three uint32s. A view of any other length means the caller
	// believes the format is something else, which is worth refusing rather
	// than reading past.
	if countBuf.DType != U32 || countBuf.Count != 3 {
		r.fail("DispatchIndirect %q: the count is three uint32 values and this view is "+
			"%d of %v", p.label, countBuf.Count, countBuf.DType)
		return r.node(NodeDispatchIndirect, p.label, nil, nil)
	}
	if countBuf.Buffer != nil && countBuf.Buffer.transient == nil &&
		countBuf.Buffer.desc.Usage&UsageIndirect == 0 {
		r.fail("DispatchIndirect %q: the count buffer needs %v and %q was created with %v",
			p.label, UsageIndirect, countBuf.Buffer.desc.Label, countBuf.Buffer.desc.Usage)
		return r.node(NodeDispatchIndirect, p.label, nil, nil)
	}
	countAccess, ok := r.declare("DispatchIndirect "+p.label, countBuf, AccessRead)
	if !ok {
		return r.node(NodeDispatchIndirect, p.label, nil, nil)
	}

	accesses, uniforms, ok := r.bindingAccesses(p, bs, us)
	if !ok {
		return r.node(NodeDispatchIndirect, p.label, nil, nil)
	}

	// The count's access is recorded last, so the binding list stays aligned
	// with the kernel's layout: dispatchOperands reads the first len(bindings)
	// positionally.
	accesses = append(accesses, countAccess)

	id := r.node(NodeDispatchIndirect, p.label, accesses, nil)
	n := &r.state.nodes[id]
	n.pipeline = p
	n.count = c
	n.uniforms = uniforms
	n.indirect = true
	for _, a := range accesses {
		r.touch(id, a)
	}
	return id
}

// dispatchOperands resolves a dispatch node's bindings for the plan.
func (g *Graph) dispatchOperands(n *recNode) (*driver.Dispatch, error) {
	// An indirect node's last access is the count buffer rather than a binding,
	// so the binding list is the prefix. Positional, because the kernel indexes
	// its arguments by layout index.
	bindings := n.accesses
	var countAccess *access
	if n.indirect {
		bindings = n.accesses[:len(n.accesses)-1]
		countAccess = &n.accesses[len(n.accesses)-1]
	}

	ops := make([]driver.Operand, len(bindings))
	for i, a := range bindings {
		o, err := g.operand(n, a)
		if err != nil {
			return nil, err
		}
		ops[i] = o
	}
	d := &driver.Dispatch{
		Kernel: n.pipeline.kernel, Count: n.count, Bindings: ops,
		Uniforms: n.uniforms,
	}
	if countAccess != nil {
		o, err := g.operand(n, *countAccess)
		if err != nil {
			return nil, err
		}
		d.Indirect = &driver.Indirect{Count: o, Max: n.count}
	}
	return d, nil
}
