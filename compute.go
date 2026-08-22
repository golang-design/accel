// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
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

// WorkgroupSize is the shape of one workgroup, fixed when a pipeline is created
// because backends need it at compile time.
//
// All three extents matter: a kernel addressing a 2D tile needs a 2D workgroup,
// and flattening it into X alone forces index math that the compiler cannot then
// prove uniform.
type WorkgroupSize struct{ X, Y, Z int }

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
func (d *Device) Missing(r Requirements) []Unmet { panic(ErrNotImplemented) }

// WorkgroupCount is how many workgroups a dispatch runs.
//
// This counts workgroups, not threads, deliberately. A thread count makes the
// workgroup size invisible to the caller, which is how a predecessor project
// ended up dispatching one thread per workgroup and leaving the hardware idle.
// For a direct dispatch, omitted Y or Z values normalize to one. X must be
// positive. Indirect dispatches keep zero as the specified skip mechanism.
type WorkgroupCount struct{ X, Y, Z int }

// SharedMemory declares workgroup-shared storage for a kernel. The size is fixed
// at pipeline creation because every backend needs it statically.
//
// Shared memory is uninitialised at workgroup start. The CPU backend fills it
// with a poison pattern rather than zeroes, so a kernel that reads before writing
// fails loudly on the oracle instead of working by accident on one backend.
type SharedMemory struct {
	DType DType
	Count int
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
type ComputePipeline struct{ _ noCopy }

// Close releases the pipeline.
func (p *ComputePipeline) Close() error { panic(ErrNotImplemented) }

// Kernel is a kernel compiled to whatever form the target device consumes.
//
// Kernels are authored in a subset of Go and lowered from one typed IR to
// generated, instrumented CPU code and native GPU shaders. The authored function
// is type-checking input, not the CPU executable. See specs/004-kernel-authoring.md.
type Kernel struct{ _ noCopy }

// UniformCodec is generated for one by-value kernel parameter type. It owns the
// std140 size, alignment, field offsets, and typed encoder.
type UniformCodec[T any] interface {
	EncodedSize() int
	Encode(dst []byte, value T) error
}

// UniformBuffer owns a generated-codec uniform allocation.
type UniformBuffer[T any] struct{ _ noCopy }

// NewUniformBuffer allocates storage sized and aligned by codec.
func NewUniformBuffer[T any](d *Device, codec UniformCodec[T]) (*UniformBuffer[T], error) {
	panic(ErrNotImplemented)
}

// Write encodes value and stages it through q.
func (u *UniformBuffer[T]) Write(q *Queue, value T) error { panic(ErrNotImplemented) }

// View returns the ordinary uniform-buffer view used by generated bindings.
func (u *UniformBuffer[T]) View() BufferView { panic(ErrNotImplemented) }

// Close releases the uniform buffer.
func (u *UniformBuffer[T]) Close() error { panic(ErrNotImplemented) }

// Access is how a binding is used by a kernel. The graph builder infers
// dependency edges and barriers from declared access, so this must be accurate:
// under-declaring is what turns a missing dependency into a race.
type Access int

const (
	AccessRead Access = iota
	AccessWrite
	AccessReadWrite
)

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
	Index int

	Buffer  BufferView
	Texture *Texture
	Sampler *Sampler

	// Slot supplies the resource before submission instead of at record time. Its
	// zero value is not a slot, so a Binding that set none of the four is rejected
	// rather than silently referring to the first one.
	Slot Slot
}
