// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"fmt"
	"golang.design/x/accel/kernelabi"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/kernel"
)

// Capabilities is what a device can actually do.
//
// It is queried before use so an absent feature is a typed answer rather than a
// dispatch-time failure. A graph requiring something absent fails when it is
// built, with an error naming the capability and the device. See
// specs/000-decisions.md decision 6.
type Capabilities struct {
	// Subgroups reports whether subgroup operations exist. Numeric lane bounds
	// live in [Limits].
	Subgroups bool

	// SubgroupOps reports which subgroup operations are present. Vulkan exposes
	// these as an independent feature set rather than one flag, so a device can
	// have ballot without shuffle.
	SubgroupOps SubgroupOpSet

	// F16Arithmetic and BF16Arithmetic report native narrow arithmetic. These are
	// separate from being able to *store* those dtypes, which several backends do
	// without being able to compute in them.
	F16Arithmetic  bool
	BF16Arithmetic bool
	I8DotProduct   bool

	// AtomicFloatAddStorage and AtomicFloatAddShared are separate because backends
	// genuinely differ between them: a device can have the storage form and not
	// the shared one. They are called out at all because atomic float add is what
	// people reach for when writing a reduction, and because it makes a reduction
	// non-deterministic (the hardware picks the accumulation order), so a test
	// asserting an exact total is wrong for floats even where it is right for
	// integers.
	AtomicFloatAddStorage bool
	AtomicFloatAddShared  bool

	// DenormF32Preserved and DenormF16Preserved report whether denormals survive
	// rather than being flushed to zero, and InfNaNProduced whether infinities and
	// NaNs are generated rather than being treated as undefined. All three vary by
	// backend and all three change numeric results, so they are queryable rather
	// than assumed.
	DenormF32Preserved bool
	DenormF16Preserved bool
	InfNaNProduced     bool
	ContractionControl bool

	// SharedMemoryKind reports whether MemoryShared is real on this device, which
	// is true on unified-memory hardware and removes a staging copy.
	SharedMemoryKind bool

	// Graphics reports whether this device can rasterize. A compute-only backend
	// is legitimate.
	Graphics                bool
	Presentation            bool
	Multisampling           bool
	RasterizerOrderedAccess bool
	IndirectDispatch        bool

	// NativeGraphReplay reports whether recorded graphs lower to a native
	// replayable object (Vulkan secondary command buffers, D3D12 bundles, Metal
	// indirect command buffers) rather than being replayed in software. Replay
	// works either way; this says how much it saves.
	NativeGraphReplay bool
}

// ID3 is a three-dimensional invocation identifier, with X, Y, and Z of type
// uint32.
//
// Ids are three-dimensional rather than scalar because a two-dimensional shared
// tile cannot be addressed from a scalar id without index arithmetic the
// compiler then cannot prove uniform. See specs/002-compute-model.md section 1.
type ID3 = kernel.ID3

// Thread is a kernel's first parameter. It carries the invocation's ids and,
// once cooperative kernels exist, the CPU backend's rendezvous state.
//
// A kernel author writes accel.Thread and never names an internal package. It
// is an alias because the backend that executes kernels cannot import this
// package, so the type has to be declared below it; that also makes the
// generated code's accel.Thread and the runtime's kernel.Thread one type rather
// than two that have to be converted. See specs/012-kernel-pipeline.md.
type Thread = kernel.Thread

// SubgroupOpSet reports which subgroup operations a device provides. Vulkan
// exposes these independently, so presence of one does not imply the others.
//
// It is an alias rather than its own type so that [Capabilities] and the
// backend-facing struct it is converted from have identical field types, which
// makes that conversion a compile-time check on the two staying in step. See
// internal/driver.
type SubgroupOpSet = driver.SubgroupOpSet

const (
	SubgroupBasic      = driver.SubgroupBasic
	SubgroupVote       = driver.SubgroupVote
	SubgroupBallot     = driver.SubgroupBallot
	SubgroupShuffle    = driver.SubgroupShuffle
	SubgroupArithmetic = driver.SubgroupArithmetic
)

// Set is what a device offers, expressed as the [Capability] set a kernel
// requires — so the two halves of the same question can be compared.
//
// A device reports [Capabilities], a struct of flags plus a [SubgroupOpSet]. A
// kernel requires a [Capability] bitmask, and so does [Policy].Require. Without
// this bridge a caller holding a [DeviceInfo] could not answer "does this device
// satisfy what I am about to require" except by reimplementing the mapping,
// including the part that is easy to miss: every subgroup bit is gated on
// Subgroups as well as on its own op bit.
func (c Capabilities) Set() Capability { return available(c) }

// Has reports whether this device offers every capability in want.
//
//	if !dev.Capabilities().Has(accel.CapAtomicFloatAddStorage) {
//		// pick another kernel, or another device
//	}
//
// Which one is missing is often the interesting half; [Device.Missing] answers
// that against a whole [Requirements].
func (c Capabilities) Has(want Capability) bool { return c.Set()&want == want }

// Capability names something a kernel can require and a device may lack. It is a
// requirement set, not a feature list: the values here are exactly the ones a
// kernel body can imply, and they are inferred from that body rather than
// declared by its author, because a declaration can be forgotten.
type Capability uint32

const (
	CapSubgroupBasic Capability = 1 << iota
	CapSubgroupVote
	CapSubgroupBallot
	CapSubgroupShuffle
	CapSubgroupArithmetic
	CapF16Arithmetic
	CapBF16Arithmetic
	CapAtomicFloatAddStorage
	CapAtomicFloatAddShared
	CapI8DotProduct
)

// Requirements is what a compiled kernel needs from a device. It is derived from
// the kernel body by the kernel compiler, never written by hand. The
// //accel:requires directive is an assertion checked against this, not a source
// of it: a mismatch in either direction fails generation.
type Requirements struct {
	Caps                 Capability
	WorkgroupSize        [3]uint32
	WorkgroupInvocations uint32
	SharedBytes          uint32
}

// Unmet is one requirement a device does not meet. It carries what was required
// and what the device reports, because an error saying only that a capability is
// missing does not tell a caller whether to change the kernel or the device.
type Unmet struct {
	Cap       Capability // zero when the unmet requirement is a limit
	Limit     string     // the Capabilities field that was exceeded, if any
	Required  uint64
	Available uint64
}

// Missing reports every feature or numeric requirement this device does not
// meet, in stable order. It consults both [Capabilities] and [Limits].
//
// Stable order matters: a caller printing the result should get the same text
// twice, and a test asserting on it should not depend on map iteration.
func (d *Device) Missing(r Requirements) []Unmet {
	var out []Unmet
	caps := d.info.Capabilities
	lim := d.info.Limits

	// Capabilities first, in declaration order, each with the Capabilities field
	// that answers it.
	for _, c := range []struct {
		cap  Capability
		have bool
	}{
		{CapSubgroupBasic, caps.Subgroups && caps.SubgroupOps&SubgroupBasic != 0},
		{CapSubgroupVote, caps.Subgroups && caps.SubgroupOps&SubgroupVote != 0},
		{CapSubgroupBallot, caps.Subgroups && caps.SubgroupOps&SubgroupBallot != 0},
		{CapSubgroupShuffle, caps.Subgroups && caps.SubgroupOps&SubgroupShuffle != 0},
		{CapSubgroupArithmetic, caps.Subgroups && caps.SubgroupOps&SubgroupArithmetic != 0},
		{CapF16Arithmetic, caps.F16Arithmetic},
		{CapBF16Arithmetic, caps.BF16Arithmetic},
		{CapAtomicFloatAddStorage, caps.AtomicFloatAddStorage},
		{CapAtomicFloatAddShared, caps.AtomicFloatAddShared},
		{CapI8DotProduct, caps.I8DotProduct},
	} {
		if r.Caps&c.cap != 0 && !c.have {
			out = append(out, Unmet{Cap: c.cap})
		}
	}

	for _, c := range []struct {
		name      string
		need      uint64
		available uint64
	}{
		{"MaxWorkgroupSize[0]", uint64(r.WorkgroupSize[0]), uint64(lim.MaxWorkgroupSize[0])},
		{"MaxWorkgroupSize[1]", uint64(r.WorkgroupSize[1]), uint64(lim.MaxWorkgroupSize[1])},
		{"MaxWorkgroupSize[2]", uint64(r.WorkgroupSize[2]), uint64(lim.MaxWorkgroupSize[2])},
		{"MaxWorkgroupInvocations", uint64(r.WorkgroupInvocations), uint64(lim.MaxWorkgroupInvocations)},
		{"MaxSharedMemoryBytes", uint64(r.SharedBytes), uint64(lim.MaxSharedMemoryBytes)},
	} {
		if c.need > c.available {
			out = append(out, Unmet{Limit: c.name, Required: c.need, Available: c.available})
		}
	}
	return out
}

func (c Capability) String() string {
	switch c {
	case CapSubgroupBasic:
		return "CapSubgroupBasic"
	case CapSubgroupVote:
		return "CapSubgroupVote"
	case CapSubgroupBallot:
		return "CapSubgroupBallot"
	case CapSubgroupShuffle:
		return "CapSubgroupShuffle"
	case CapSubgroupArithmetic:
		return "CapSubgroupArithmetic"
	case CapF16Arithmetic:
		return "CapF16Arithmetic"
	case CapBF16Arithmetic:
		return "CapBF16Arithmetic"
	case CapAtomicFloatAddStorage:
		return "CapAtomicFloatAddStorage"
	case CapAtomicFloatAddShared:
		return "CapAtomicFloatAddShared"
	case CapI8DotProduct:
		return "CapI8DotProduct"
	}
	return fmt.Sprintf("Capability(%d)", uint32(c))
}

// WorkgroupCount is how many workgroups a dispatch runs.
//
// This counts workgroups, not threads, deliberately. A thread count makes the
// workgroup size invisible to the caller, which is how a predecessor project
// ended up dispatching one thread per workgroup and leaving the hardware idle.
// For a direct dispatch, omitted Y or Z values normalize to one. X must be
// positive. Indirect dispatches keep zero as the specified skip mechanism.
type WorkgroupCount struct{ X, Y, Z int }

// Workgroups is how many workgroups cover n invocations of this pipeline's
// kernel along X, with Y and Z at one.
//
// # Why this is a method and not arithmetic a caller writes
//
// The arithmetic is one line -- ceiling division by the kernel's workgroup size
// -- and every caller writing that line has to know the size, which is the
// kernel's and not theirs. So the line is written wherever the size is edited,
// and the two drift silently: too few workgroups leaves a tail of the data
// untouched, which looks like a kernel bug at the boundary, and too many runs
// invocations past the end, which the kernel is supposed to guard but that
// guard is exactly what a caller forgets to check.
//
// It is deliberately not a thread count. [WorkgroupCount] counts workgroups
// because a thread count makes the workgroup size invisible, which is how a
// predecessor ended up dispatching one thread per workgroup.
//
//	r.Dispatch(pipe, binds, nil, pipe.Workgroups(len(data)))
func (p *ComputePipeline) Workgroups(n int) WorkgroupCount {
	return p.WorkgroupsFor(n, 1, 1)
}

// WorkgroupsFor is [ComputePipeline.Workgroups] over three dimensions.
//
// A zero or negative extent in any axis yields a zero count in that axis, which
// is the specified skip: specs/003-command-graph.md makes a zero in any
// dimension a dispatch of nothing rather than an error, so covering "no work"
// produces "no work" rather than one workgroup that reads past the end.
func (p *ComputePipeline) WorkgroupsFor(x, y, z int) WorkgroupCount {
	size := p.kernel.WorkgroupSize
	return WorkgroupCount{
		X: coverGroups(x, int(size.X)),
		Y: coverGroups(y, int(size.Y)),
		Z: coverGroups(z, int(size.Z)),
	}
}

// coverGroups is ceiling division that treats a non-positive extent as no work.
func coverGroups(n, size int) int {
	if n <= 0 || size <= 0 {
		return 0
	}
	return (n + size - 1) / size
}

// ComputePipelineDescriptor describes a compute pipeline to create.
type ComputePipelineDescriptor struct {
	// Kernel is the compiled kernel. See the kernel authoring package for how one
	// is produced from Go source. It owns the workgroup, shared-memory, binding,
	// access, and requirement metadata.
	Kernel *Kernel

	Label string
}

// ComputePipeline is a kernel compiled for a device with a fixed workgroup size.
type ComputePipeline struct {
	_ noCopy

	dev    *Device
	kernel *Kernel
	label  string
	state  resourceState
}

// Kernel is the compiled kernel this pipeline was created from. It is the
// source of truth for everything static: workgroup extent, binding layout,
// inferred access, and requirements.
func (p *ComputePipeline) Kernel() *Kernel { return p.kernel }

// Close releases the pipeline. It fails while a graph still names it, because
// a graph submitted afterwards would dispatch a kernel whose pipeline is gone.
func (p *ComputePipeline) Close() error {
	if !p.state.beginClose() {
		return nil
	}
	p.dev.countPipelines(-1)
	return nil
}

// Kernel is a kernel compiled to whatever form the target device consumes.
//
// An alias for [kernelabi.Kernel], which is where generated code names it and
// where its fields are documented. A caller does not construct one and does not
// read its fields: `go generate` emits one package-level variable per kernel,
// and the address is the whole of the interface —
// `ComputePipelineDescriptor{Kernel: &kernels.ScaleKernel}`.
//
// The alias stays so that spelling remains available at the point of use, while
// the thirty-odd names a *generated file* needs live in kernelabi rather than
// in this package's index. See specs/036-documentation.md's freeze record.
type Kernel = kernelabi.Kernel

// Access is how a binding is used by a kernel. The graph builder infers
// dependency edges and barriers from declared access, so this must be accurate:
// under-declaring is what turns a missing dependency into a race.
type Access int

const (
	AccessRead Access = iota
	AccessWrite
	AccessReadWrite
)

func (a Access) String() string {
	switch a {
	case AccessRead:
		return "read"
	case AccessWrite:
		return "write"
	case AccessReadWrite:
		return "read-write"
	}
	return fmt.Sprintf("Access(%d)", int(a))
}

// BindingKind is what sort of resource fills a slot.
type BindingKind int

const (
	BindingStorageBuffer BindingKind = iota
	BindingUniformBuffer
	BindingSampledTexture
	BindingStorageTexture
	BindingSampler

	// BindingAttachment is a texture a render pass writes as a colour or depth
	// attachment. An attachment is not bound to a pipeline slot the way the kinds
	// above are, and it is in this enum for one reason: a graph slot has to be able
	// to name one, which is how the swapchain image reaches a recorded pass
	// (specs/005-graphics.md).
	BindingAttachment
)

// BindingSlot declares one entry of a pipeline's binding layout.
type BindingSlot struct {
	Index  int
	Kind   BindingKind
	Access Access

	// DType constrains storage buffer slots. A resource bound to this slot must
	// match, which is checked when the graph is built.
	DType DType

	Name string
}

// Binding binds one resource to one entry of a pipeline's binding layout.
//
// A binding is what varies between submissions of the same [Graph]: pointing it
// at a different resource of the same type, dtype, and access is allowed, and
// anything more than that is a different graph.
//
// Exactly one of Buffer, Texture, Sampler and Slot is set, and setting none or
// several is a validation error naming the binding. Slot is the indirection that
// makes a graph replayable: naming a [Slot] instead of a resource says the
// resource arrives before submission rather than at record time, which is how a
// swapchain image that does not exist yet, or one sequence's cache out of many,
// reaches a recorded node.
//
// Two vocabularies meet in this struct and they are not the same thing. Index is
// an entry in the *pipeline's* binding layout, declared by [BindingSlot] and
// fixed when the pipeline is created. Slot is a *graph's* rebindable input,
// declared by [Recorder.Slot] and bound per submission. A pipeline's binding is
// where a resource is used; a graph's slot is where it comes from.
type Binding struct {
	// Index is the entry in the pipeline's binding layout this binds to.
	//
	// The pipeline's layout, and nothing else. It used to name the kernel's
	// by-value list instead whenever Uniform was set, so one dispatch could
	// carry two entries both spelled Index: 0 meaning different things — and
	// neither matched the authored parameter position a reader was looking at.
	// By-value parameters are [UniformValue] now, and have their own argument.
	Index int

	Buffer  BufferView
	Texture *Texture

	// Slot supplies the resource before submission instead of at record time. Its
	// zero value is not a slot, so a Binding that set none of the three is
	// rejected rather than silently referring to the first one.
	Slot Slot
}

// UniformValue is one by-value parameter of a dispatch.
//
// Index names the kernel's by-value list — its own space, separate from
// [Binding.Index]'s. A kernel's signature interleaves the two, so neither index
// is the parameter position; the generated record carries both lists and the
// error names the kernel when they disagree.
type UniformValue struct {
	Index int
	Value any
}
