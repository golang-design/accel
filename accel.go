// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package accel is a backend-selectable, cgo-free foundation for running compute
// and graphics work on a GPU.
//
// This is the device layer. Its vocabulary is buffers, textures, workgroups, and
// barriers; it knows nothing about tensors or meshes. The tensor layer is built
// on top of it and lives in a subpackage.
//
// # Status
//
// Design stage. Every function here reports [ErrNotImplemented]. The API surface
// exists so the design can be read as Go and can be checked by the compiler, and
// it will change.
//
// # The model
//
// Work is recorded into a [Graph], which is immutable once built and can be
// submitted many times with its inputs rebound between submissions. This is
// deliberately not a one-shot command encoder; see docs/design.md decision 1 for
// why, and specs/003-command-graph.md for the details.
//
//	rec := dev.NewRecorder()
//	rec.Dispatch(pipeline, bindings, WorkgroupCount{X: n})
//	g, err := rec.Build()   // validate, plan memory, compute barriers, lower
//	...
//	f := dev.Queue().Submit(g)
//	f.Wait()
//
// Memory comes from pools rather than one allocation per resource, because a
// model has thousands of tensors and drivers cap the number of allocations. See
// specs/001-device-resources.md.
package accel

import "errors"

// ErrNotImplemented is reported by every operation while this package is at the
// design stage.
var ErrNotImplemented = errors.New("accel: not implemented (design stage)")

// ErrUnsupported reports that a device cannot perform an operation because it
// lacks a capability. The error names the capability and the device: absence is
// always explicit, never a silent wrong result. See docs/design.md decision 6.
var ErrUnsupported = errors.New("accel: unsupported by this device")

// Backend identifies a device implementation.
type Backend int

const (
	// BackendCPU is a pure-Go implementation. It is a first-class backend and the
	// correctness oracle every other backend is verified against, not a fallback.
	// It is always available on every platform. See docs/design.md decision 3.
	BackendCPU Backend = iota

	BackendMetal
	BackendVulkan
	BackendD3D12
	BackendOpenGL
)

// String returns the backend's name.
func (b Backend) String() string { panic(ErrNotImplemented) }

// DeviceInfo describes a device that could be opened, without opening it. Callers
// choose on reported capability rather than by trying and catching failures.
type DeviceInfo struct {
	Backend Backend
	Name    string

	// Capabilities is what this device can actually do. Queried before use.
	Capabilities Capabilities
}

// Enumerate reports every device present on this machine.
func Enumerate() ([]DeviceInfo, error) { panic(ErrNotImplemented) }

// Open opens a device on the named backend.
//
// It never falls back to another backend. A caller asking for something
// unavailable gets an error saying so, because silent fallback turns "my GPU code
// is slow" into a mystery. Use [OpenBest] to ask for automatic selection
// explicitly.
func Open(b Backend) (*Device, error) { panic(ErrNotImplemented) }

// OpenBest opens the best available device under an explicit policy. Unlike
// [Open] this is a request to choose, and the policy is what it chooses by: it
// fails rather than descending into something the caller did not sanction.
func OpenBest(p Policy) (*Device, error) { panic(ErrNotImplemented) }

// Policy is what OpenBest is allowed to select.
//
// Two defaults are deliberate. The CPU backend is never selected unless
// AllowCPU says so: it is a first-class backend and it is not a fast path, and a
// caller who wanted a GPU and got it should hear about that as an error. Software
// GPU devices are their own class rather than being lumped in with hardware,
// because lavapipe and WARP are real devices that may well be slower than the CPU
// backend, and automatic selection has to be able to see the difference.
type Policy struct {
	Prefer        []Backend // tried in order; empty means every compiled-in backend
	AllowCPU      bool
	AllowSoftware bool
	Require       Capability // a device lacking any of these is not a candidate
}

// Device is an opened accelerator.
//
// A Device is safe for concurrent use. A [Recorder] obtained from it is not; see
// [Device.NewRecorder].
type Device struct{ _ noCopy }

// Info reports what this device is and what it can do.
func (d *Device) Info() DeviceInfo { panic(ErrNotImplemented) }

// Queue returns the device's default queue, which is always [QueueUniversal].
func (d *Device) Queue() *Queue { panic(ErrNotImplemented) }

// Queues reports every queue this device exposes, in a stable order whose first
// entry is what [Device.Queue] returns.
//
// Queue topology is reported rather than inferred from the platform, because the
// backends disagree completely: Vulkan exposes queue families with capability
// bits, D3D12 has typed command queues, and Metal, GL and the CPU backend have
// exactly one. Ordering between submissions depends on which queue they went to,
// so a caller who cannot enumerate them cannot use that rule.
func (d *Device) Queues() []QueueInfo { panic(ErrNotImplemented) }

// QueueFor returns a queue able to run kind.
//
// It never fails and never invents a queue: on a device with one universal queue
// it returns that queue, and the caller sees which one they got through
// [Device.Queues]. That is not the silent substitution [Open] refuses, because
// nothing about the result is weaker than what was asked for, only less parallel.
func (d *Device) QueueFor(kind QueueKind) *Queue { panic(ErrNotImplemented) }

// QueueKind is what a queue accepts.
type QueueKind int

const (
	// QueueUniversal accepts everything: compute, graphics, and transfer.
	QueueUniversal QueueKind = iota
	QueueCompute             // compute and transfer, no rasterization
	QueueTransfer            // transfer only
)

// QueueInfo describes one queue a device exposes.
type QueueInfo struct {
	Kind  QueueKind
	Index int
	Label string // the backend's own name for it, for logs
}

// NewPool allocates a memory pool of the given kind and size, from which buffers
// are suballocated. See specs/001-device-resources.md for why allocation is
// pooled rather than per resource.
//
// It is [Device.NewPoolWith] with the general-purpose policy. A pool never grows:
// a pool is one device allocation, no backend can resize one in place, and moving
// it would invalidate every address already handed out.
func (d *Device) NewPool(kind MemoryKind, bytes int) (*Pool, error) {
	panic(ErrNotImplemented)
}

// NewPoolWith allocates a pool with an explicit policy, which is how a caller
// asks for a linear pool or reserves one for textures.
func (d *Device) NewPoolWith(desc PoolDescriptor) (*Pool, error) {
	panic(ErrNotImplemented)
}

// NewBuffer allocates a single buffer from an implicit pool. It is a convenience
// for callers with a handful of buffers; anything allocating at scale should use
// [Device.NewPool].
func (d *Device) NewBuffer(desc BufferDescriptor) (*Buffer, error) {
	panic(ErrNotImplemented)
}

// NewTexture creates a texture. Bytes per pixel always derives from the format,
// and depth formats carry backend constraints the implementation enforces; see
// docs/conventions.md.
func (d *Device) NewTexture(desc TextureDescriptor) (*Texture, error) {
	panic(ErrNotImplemented)
}

// NewComputePipeline compiles a kernel into a pipeline.
//
// Workgroup size is fixed here rather than at dispatch, because backends need it
// at compile time: it appears in the GLSL layout qualifier and in Metal's
// threads-per-threadgroup. See specs/002-compute-model.md.
func (d *Device) NewComputePipeline(desc ComputePipelineDescriptor) (*ComputePipeline, error) {
	panic(ErrNotImplemented)
}

// NewRecorder returns a recorder for building a [Graph].
//
// A Recorder belongs to one goroutine. The [Graph] it produces is immutable and
// may be submitted from several goroutines at once.
func (d *Device) NewRecorder() *Recorder { panic(ErrNotImplemented) }

// Close releases the device. Resources created from it must be closed first.
func (d *Device) Close() error { panic(ErrNotImplemented) }

// noCopy makes `go vet` complain about copying a value that owns device state.
type noCopy struct{}

func (*noCopy) Lock()   {}
func (*noCopy) Unlock() {}
