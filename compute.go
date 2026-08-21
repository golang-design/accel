// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

// Capabilities is what a device can actually do.
//
// It is queried before use so an absent feature is a typed answer rather than a
// dispatch-time failure. A graph requiring something absent fails when it is
// built, with an error naming the capability and the device. See
// docs/design.md decision 6.
type Capabilities struct {
	// MaxWorkgroupSize is the largest workgroup per dimension, and
	// MaxWorkgroupInvocations the largest total across all three. Both are limits:
	// a pipeline exceeding either fails to create.
	//
	// The portable floor for MaxWorkgroupInvocations is 128, not 256. That is not
	// academic: the worked GEMM in specs/002-compute-model.md uses a 16x8 tile
	// rather than 16x16 because of it.
	MaxWorkgroupSize        [3]int
	MaxWorkgroupInvocations int

	// MaxWorkgroupCount is the largest dispatch per dimension. Real backends cap
	// this at 65535 per axis, so a large 1D dispatch has to be reshaped rather
	// than assumed to fit.
	MaxWorkgroupCount [3]int

	// SharedMemoryBytes is the shared storage budget per workgroup.
	SharedMemoryBytes int

	// Subgroups reports whether subgroup operations exist. MinSubgroupSize and
	// MaxSubgroupSize bound the number of lanes scheduled together.
	//
	// Two bounds rather than one value, because a single number cannot express
	// what backends report: D3D12 gives a minimum and a maximum that can differ,
	// and a device may vary it per pipeline. A kernel must read these rather than
	// assume, and must have a correct path for Subgroups being false.
	Subgroups       bool
	MinSubgroupSize int
	MaxSubgroupSize int

	// SubgroupOps reports which subgroup operations are present. Vulkan exposes
	// these as an independent feature set rather than one flag, so a device can
	// have ballot without shuffle.
	SubgroupOps SubgroupOpSet

	// F16Arithmetic and BF16Arithmetic report native narrow arithmetic. These are
	// separate from being able to *store* those dtypes, which several backends do
	// without being able to compute in them.
	F16Arithmetic  bool
	BF16Arithmetic bool

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

	// SharedMemoryKind reports whether MemoryShared is real on this device, which
	// is true on unified-memory hardware and removes a staging copy.
	SharedMemoryKind bool

	// Graphics reports whether this device can rasterize. A compute-only backend
	// is legitimate.
	Graphics bool

	// NativeGraphReplay reports whether recorded graphs lower to a native
	// replayable object (Vulkan secondary command buffers, D3D12 bundles, Metal
	// indirect command buffers) rather than being replayed in software. Replay
	// works either way; this says how much it saves.
	NativeGraphReplay bool

	MaxStorageBufferBytes int
	MaxBindingsPerKind    int
}

// WorkgroupSize is the shape of one workgroup, fixed when a pipeline is created
// because backends need it at compile time.
//
// All three extents matter: a kernel addressing a 2D tile needs a 2D workgroup,
// and flattening it into X alone forces index math that the compiler cannot then
// prove uniform.
type WorkgroupSize struct{ X, Y, Z int }

// SubgroupOpSet reports which subgroup operations a device provides. Vulkan
// exposes these independently, so presence of one does not imply the others.
type SubgroupOpSet uint32

const (
	SubgroupBasic SubgroupOpSet = 1 << iota
	SubgroupVote
	SubgroupBallot
	SubgroupShuffle
	SubgroupArithmetic
)

// WorkgroupCount is how many workgroups a dispatch runs.
//
// This counts workgroups, not threads, deliberately. A thread count makes the
// workgroup size invisible to the caller, which is how a predecessor project
// ended up dispatching one thread per workgroup and leaving the hardware idle.
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
	// is produced from Go source.
	Kernel *Kernel

	// WorkgroupSize is fixed here, not at dispatch.
	WorkgroupSize WorkgroupSize

	// Shared declares the kernel's workgroup-shared storage, if any.
	Shared []SharedMemory

	// Layout declares the binding slots the kernel expects. Slots are matched by
	// type, dtype, and access when a graph binds resources to them.
	Layout []BindingSlot

	Label string
}

// ComputePipeline is a kernel compiled for a device with a fixed workgroup size.
type ComputePipeline struct{ _ noCopy }

// Close releases the pipeline.
func (p *ComputePipeline) Close() error { panic(ErrNotImplemented) }

// Kernel is a kernel compiled to whatever form the target device consumes.
//
// Kernels are authored in a subset of Go: one source runs as ordinary Go on
// [BackendCPU] and compiles to native shader code on GPU backends, which is what
// makes the CPU backend an exact oracle rather than an approximation. See
// specs/004-kernel-authoring.md.
type Kernel struct{ _ noCopy }

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

// Binding binds one resource to one slot.
//
// A binding is what varies between submissions of the same [Graph]: rebinding a
// slot to a different resource of the same type, dtype, and access is allowed,
// and anything more than that is a different graph.
type Binding struct {
	Index   int
	Buffer  BufferView
	Texture *Texture
	Sampler *Sampler
}
