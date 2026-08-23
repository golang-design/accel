// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import (
	"errors"
	"fmt"
	"path/filepath"
	"runtime"
	"sort"

	"golang.design/x/accel"
)

// Runtime owns one device and the pipelines compiled for it.
//
// Pipelines are cached here rather than on a plan because two plans over the
// same model share nearly all of them, and compiling MSL is a call into the
// device compiler that takes milliseconds. There is deliberately no plan cache:
// specs/007-tensor-layer.md puts that above this package, because a key that
// looked only at shapes would be wrong in ways nobody could see.
type Runtime struct {
	dev   *accel.Device
	pipes map[*accel.Kernel]*accel.ComputePipeline
	plans int
}

// NewRuntime prepares a device for tensor work.
func NewRuntime(dev *accel.Device) (*Runtime, error) {
	if dev == nil {
		return nil, errors.New("accel/tensor: NewRuntime needs a device")
	}
	return &Runtime{dev: dev, pipes: map[*accel.Kernel]*accel.ComputePipeline{}}, nil
}

// Device reports the device this runtime lowers to.
func (r *Runtime) Device() *accel.Device { return r.dev }

// NewBuilder starts recording one tensor graph.
//
// A builder belongs to one goroutine. It records rather than executes, so
// nothing here touches the device until Compile.
func (r *Runtime) NewBuilder(label string) *Builder {
	return &Builder{rt: r, label: label}
}

// pipeline returns the compiled form of a kernel, once per runtime.
func (r *Runtime) pipeline(k *accel.Kernel, label string) (*accel.ComputePipeline, error) {
	if p, ok := r.pipes[k]; ok {
		return p, nil
	}
	p, err := r.dev.NewComputePipeline(accel.ComputePipelineDescriptor{Kernel: k, Label: label})
	if err != nil {
		return nil, err
	}
	r.pipes[k] = p
	return p, nil
}

// Close releases the runtime's pipelines.
//
// Every plan built from it must be closed first, which is checked rather than
// assumed: a plan outliving its runtime would hold a pipeline nobody owns.
func (r *Runtime) Close() error {
	if r.plans != 0 {
		return fmt.Errorf("accel/tensor: %d plan(s) are still open on this runtime", r.plans)
	}
	var errs []error
	for _, p := range r.pipes {
		errs = append(errs, p.Close())
	}
	r.pipes = nil
	return errors.Join(errs...)
}

// PortKind says how a caller supplies an external value.
type PortKind uint8

const (
	// PortInput varies on every submission.
	PortInput PortKind = iota
	// PortWeight may be rebound between submissions and is read-only.
	PortWeight
	// PortState is caller-owned read-write storage, never aliased by the
	// planner.
	PortState
	// PortOutput is a value the caller wants written somewhere they can read.
	PortOutput
)

func (k PortKind) String() string {
	switch k {
	case PortInput:
		return "input"
	case PortWeight:
		return "weight"
	case PortState:
		return "state"
	case PortOutput:
		return "output"
	}
	return fmt.Sprintf("PortKind(%d)", uint8(k))
}

// PortDesc is one external buffer a plan expects.
type PortDesc struct {
	Name  string
	DType DType
	Shape Shape
	Kind  PortKind
}

// ValueDesc declares an input or weight.
type ValueDesc struct {
	Name  string
	DType DType
	Shape Shape
}

// Builder records one tensor graph.
//
// It collects errors rather than returning them, so model code has no error
// branch per line. See the package doc.
type Builder struct {
	rt    *Runtime
	label string

	nodes  []node
	ports  []PortDesc
	byName map[string]int // port name to index in ports

	scalars   []ScalarDesc
	scalarPos map[string]int

	outputs []output
	errs    []error
}

// output pairs a declared name with the value that fills it.
type output struct {
	name string
	t    *Tensor
}

// node is one recorded operator.
type node struct {
	op     string
	inputs []*Tensor
	out    *Tensor

	// kernel is the corpus kernel this lowers to, chosen when the node is
	// recorded so that Selections can report it before anything runs.
	kernel *accel.Kernel

	// uniform builds this operator's by-value parameter from the scalars a
	// submission supplies, or is nil when the operator takes none. A function
	// rather than a value, because the value is not known until submission and
	// the *shape* of it is known only here.
	uniform func(map[string]ScalarValue) any

	// reads names the scalars uniform consults, so binding validation can say
	// which operator wanted a value nobody bound.
	reads []string

	// grid computes the workgroup count from the result. Nil means one
	// invocation per output element, which is what an elementwise kernel wants
	// and what most of the corpus is.
	grid grid

	// bcast marks an operator whose operands are broadcast to the result's
	// shape and packed if they are not already it. True for the elementwise
	// family and false for everything else, whose operands have shapes of their
	// own: a gather's table is not the shape of its result.
	bcast bool

	// inPlace marks a kernel that rewrites the buffer it reads. Its operand is
	// copied into the result's storage first, because a tensor is an immutable
	// value and the kernel is not.
	inPlace bool

	// reason and rejected explain the selection, for Plan.Selections.
	reason   string
	rejected []string
}

// fail records a diagnostic with the caller's source position.
//
// The position is recovered here because this is the only moment it exists: a
// tensor graph has no source of its own, and by the time Compile runs, the call
// that made the mistake is long returned. skip counts the frames between the
// caller of the operator and this function.
func (b *Builder) fail(skip int, op string, format string, args ...any) *Tensor {
	where := "unknown position"
	if _, file, line, ok := runtime.Caller(skip + 1); ok {
		where = fmt.Sprintf("%s:%d", filepath.Base(file), line)
	}
	b.errs = append(b.errs, fmt.Errorf("accel/tensor: %s at %s: %s",
		op, where, fmt.Sprintf(format, args...)))
	return &Tensor{b: b, poison: true, node: -1}
}

// poisoned reports whether any operand is poisoned.
//
// A poisoned operand produces a poisoned result and *no new error*: one mistake
// is one diagnostic. Without this rule a wrong shape near the top of a model
// produces a page of errors that all describe the same thing.
func poisoned(ts ...*Tensor) bool {
	for _, t := range ts {
		if t == nil || t.poison {
			return true
		}
	}
	return false
}

// poison returns a poisoned value without recording anything.
func (b *Builder) poison() *Tensor { return &Tensor{b: b, poison: true, node: -1} }

// declare adds an external port and returns its tensor.
func (b *Builder) declare(skip int, op string, d ValueDesc, kind PortKind) *Tensor {
	if b.byName == nil {
		b.byName = map[string]int{}
	}
	switch {
	case d.Name == "":
		return b.fail(skip+1, op, "a port needs a name")
	case len(d.Shape) == 0:
		return b.fail(skip+1, op, "%q has no shape; a scalar is a shape of [1]", d.Name)
	}
	for i, dim := range d.Shape {
		if dim <= 0 {
			return b.fail(skip+1, op, "%q has dimension %d of %d, and every dimension is "+
				"a positive concrete integer", d.Name, i, dim)
		}
	}
	if _, dup := b.byName[d.Name]; dup {
		return b.fail(skip+1, op, "%q is declared twice; port names are unique within a "+
			"builder because they are how a caller binds", d.Name)
	}
	b.byName[d.Name] = len(b.ports)
	b.ports = append(b.ports, PortDesc{Name: d.Name, DType: d.DType, Shape: d.Shape, Kind: kind})
	return &Tensor{
		b: b, dtype: d.DType, shape: d.Shape,
		strides: contiguous(d.Shape), node: -1, port: d.Name,
	}
}

// Input declares a value that varies on every submission.
func Input(b *Builder, d ValueDesc) *Tensor { return b.declare(1, "Input", d, PortInput) }

// Weight declares a read-only value that may be rebound between submissions.
func Weight(b *Builder, d ValueDesc) *Tensor { return b.declare(1, "Weight", d, PortWeight) }

// Output declares a value the caller wants written into a bound buffer.
//
// Undeclared intermediates are inaccessible after compilation, which is what
// lets the planner alias them.
func Output(b *Builder, name string, x *Tensor) {
	if poisoned(x) {
		return
	}
	if b.byName == nil {
		b.byName = map[string]int{}
	}
	if name == "" {
		b.fail(1, "Output", "an output needs a name")
		return
	}
	if _, dup := b.byName[name]; dup {
		b.fail(1, "Output", "%q is declared twice", name)
		return
	}
	b.byName[name] = len(b.ports)
	b.ports = append(b.ports, PortDesc{
		Name: name, DType: x.dtype, Shape: x.shape, Kind: PortOutput,
	})
	b.outputs = append(b.outputs, output{name: name, t: x})
}

// record appends an operator node and returns its result tensor.
func (b *Builder) record(n node, dtype DType, shape Shape) *Tensor {
	out := &Tensor{
		b: b, dtype: dtype, shape: shape,
		strides: contiguous(shape), node: len(b.nodes),
	}
	n.out = out
	b.nodes = append(b.nodes, n)
	return out
}

// Err reports every diagnostic collected so far, joined.
//
// Compile returns the same thing; this exists so a caller building a model in
// pieces can check without compiling.
func (b *Builder) Err() error {
	if len(b.errs) == 0 {
		return nil
	}
	// Sorted so the report is stable across runs. Two errors recorded from the
	// same line in different goroutines would otherwise swap places between
	// runs, which makes a diff of test output noise.
	msgs := make([]string, len(b.errs))
	for i, e := range b.errs {
		msgs[i] = e.Error()
	}
	sort.Strings(msgs)
	joined := make([]error, len(msgs))
	for i, m := range msgs {
		joined[i] = errors.New(m)
	}
	return errors.Join(joined...)
}
