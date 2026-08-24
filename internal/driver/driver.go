// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package driver is the seam between the public accel API and a backend.
//
// The public package owns policy: validation, alignment, suballocation, views,
// lifetime, and retain sets are backend-independent and are implemented once.
// A backend owns what only it can answer: what the device is, what it can do,
// what its numeric bounds are, and where its memory comes from.
//
// # Why the vocabulary is duplicated here
//
// accel links its backends in, so a backend cannot import accel. Anything that
// crosses this seam must therefore be declared below both. [Limits] and
// [Capabilities] cross it because spec 001 section 1.1 requires every field to
// be queried at device open: a Metal device reports what Metal answered, not a
// remembered constant, so the seam has to carry the answer.
//
// The duplication is checked by the compiler rather than by a test. Each struct
// here has the same field list as its accel counterpart, which makes the Go
// struct conversion accel.Limits(l) legal, and that conversion stops compiling
// the moment either side gains, loses, renames, or retypes a field.
package driver

// MemoryKind is where a block's memory lives and who can reach it. It mirrors
// accel.MemoryKind and shares its underlying type, so the two convert.
type MemoryKind int

const (
	MemoryDevice MemoryKind = iota
	MemoryUpload
	MemoryReadback
	MemoryShared
)

// Backend identifies a device implementation. It mirrors accel.Backend and
// shares its underlying type, so the two convert.
type Backend int

const (
	BackendCPU Backend = iota
	BackendMetal
	BackendVulkan
	BackendD3D12
	BackendOpenGL
)

// SubgroupOpSet reports which subgroup operations a device provides.
//
// This one type is declared here rather than in accel and aliased back, because
// [Capabilities] has a field of this type and a struct conversion requires the
// field types to be identical rather than merely to share an underlying type.
type SubgroupOpSet uint32

const (
	SubgroupBasic SubgroupOpSet = 1 << iota
	SubgroupVote
	SubgroupBallot
	SubgroupShuffle
	SubgroupArithmetic
)

// Limits mirrors accel.Limits. The field documentation lives there; this
// declaration exists so a backend can report queried numbers across the seam.
type Limits struct {
	MinStorageBufferOffsetAlignment int
	MinUniformBufferOffsetAlignment int

	MinBufferCopyOffsetAlignment   int
	MinBufferCopyRowPitchAlignment int

	MinTexturePlacementAlignment int

	MaxBufferBytes int
	MaxPoolBytes   int
	MaxPools       int

	MaxTextureExtent2D    int
	MaxTextureExtent3D    int
	MaxTextureArrayLayers int

	MaxUniformBlockBytes int

	MaxWorkgroupSize             [3]int
	MaxWorkgroupInvocations      int
	MaxWorkgroupCount            [3]int
	MaxSharedMemoryBytes         int
	MaxStorageBufferBindingBytes int
	MaxBindingsPerKind           int

	// MaxColorAttachments is how many colour targets one render pass may have.
	// Zero means this backend reports none, which at v0 means it cannot draw.
	MaxColorAttachments int

	MinSubgroupSize int
	MaxSubgroupSize int
}

// Capabilities mirrors accel.Capabilities. The field documentation lives there.
type Capabilities struct {
	Subgroups bool

	SubgroupOps SubgroupOpSet

	F16Arithmetic  bool
	BF16Arithmetic bool
	I8DotProduct   bool

	AtomicFloatAddStorage bool
	AtomicFloatAddShared  bool

	DenormF32Preserved bool
	DenormF16Preserved bool
	InfNaNProduced     bool
	ContractionControl bool

	SharedMemoryKind bool

	Graphics                bool
	Presentation            bool
	Multisampling           bool
	RasterizerOrderedAccess bool
	IndirectDispatch        bool

	NativeGraphReplay bool
}

// Info is everything a backend reports about one adapter.
type Info struct {
	Backend      Backend
	Name         string
	Vendor       string
	Software     bool
	Capabilities Capabilities
	Limits       Limits
}

// Adapter is an enumerated device that has not been opened.
type Adapter interface {
	// Info reports what this adapter is, before anything is opened.
	Info() Info

	// Token is a stable, comparable identity for this adapter within the
	// process. accel wraps it in an opaque accel.AdapterID.
	Token() [16]byte

	// Open opens the adapter. opts is backend-defined and may be nil, which
	// every backend must accept as its default configuration.
	Open(opts any) (Device, error)
}

// Device is an opened adapter. It is safe for concurrent use.
type Device interface {
	// Info reports the opened device's identity, capabilities, and limits, which
	// may differ from the adapter's when the backend was opened with options
	// that constrain them.
	Info() Info

	// Supports reports whether this device can back a pool of the given kind. A
	// kind reported absent is an error at the accel layer naming the kind and the
	// device; it is never silently substituted.
	Supports(kind MemoryKind) bool

	// Alloc makes one device allocation. accel suballocates within it: a pool is
	// exactly one Block, which is why a pool cannot grow.
	Alloc(kind MemoryKind, bytes int, label string) (Block, error)

	// Lost reports whether the device has stopped being usable, and stays
	// non-nil once it is: specs/001-device-resources.md section 7.4 makes loss
	// terminal, so a backend that reported it once and then appeared to recover
	// would leave a caller running on resources whose contents are undefined.
	//
	// It is in the core interface rather than discovered by assertion because
	// every backend must be able to answer it. A backend that cannot lose a
	// device answers nil forever, which is a real answer rather than a stub.
	Lost() error

	// Close releases the device. accel guarantees every Block is freed first.
	Close() error
}

// Unwrap resolves a Block that is a handle to another Block.
//
// A Block is an interface, and the layer above may hand a backend one that
// forwards rather than one the backend allocated: accel's shared transient pool
// does exactly that, so growing it swaps the allocation underneath without
// invalidating the operands, transients and executables that captured a handle
// at build time.
//
// Every backend that type-asserts a Block to its own concrete type must call
// this first. Forgetting to is not subtle -- the assertion fails and the error
// names the wrapper -- which is why this is a free function rather than a
// method a backend could forget was there.
func Unwrap(b Block) Block {
	for {
		u, ok := b.(interface{ Unwrap() Block })
		if !ok {
			return b
		}
		next := u.Unwrap()
		if next == nil || next == b {
			return b
		}
		b = next
	}
}

// Block is one device allocation backing one pool.
type Block interface {
	// Bytes returns the host mapping, or nil when this memory is not
	// host-visible. This is the authority on mappability: a Metal private buffer
	// has no host pointer, and the CPU backend returns nil for MemoryDevice even
	// though the memory physically could be mapped, so that the one backend able
	// to enforce the rule does (spec 006 section 1).
	Bytes() []byte

	// Write and Read move bytes through the device for memory Bytes does not
	// map. They are synchronous: this is the immediate transfer path of spec 001
	// section 8.1, not the recorded one.
	Write(off int, src []byte) error
	Read(off int, dst []byte) error

	// Size reports the allocation's length in bytes.
	Size() int

	// Free releases the allocation.
	Free()
}
