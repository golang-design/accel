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
// Under construction, and the API will change. Device open, enumeration, and
// selection are implemented; everything past them still reports
// [ErrNotImplemented]. specs/009-sequencing.md is the order the rest arrives in.
//
// # The model
//
// Work is recorded into a [Graph], which is immutable once built and can be
// submitted many times with its inputs rebound between submissions. This is
// deliberately not a one-shot command encoder; see specs/000-decisions.md
// decision 1 for why, and specs/003-command-graph.md for the details.
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

import (
	"errors"
	"sync"

	"golang.design/x/accel/internal/driver"
)

// ErrNotImplemented is reported by every operation while this package is at the
// design stage.
var ErrNotImplemented = errors.New("accel: not implemented (design stage)")

// ErrUnsupported reports that a device cannot perform an operation because it
// lacks a capability. The error names the capability and the device: absence is
// always explicit, never a silent wrong result. See specs/000-decisions.md
// decision 6.
var ErrUnsupported = errors.New("accel: unsupported by this device")

// Backend identifies a device implementation.
type Backend int

const (
	// BackendCPU is a pure-Go implementation. It is a first-class backend and the
	// correctness oracle every other backend is verified against, not a fallback.
	// It is always available on every platform. See specs/000-decisions.md
	// decision 3.
	BackendCPU Backend = iota

	BackendMetal
	BackendVulkan
	BackendD3D12
	BackendOpenGL
)

// AdapterID identifies one enumerated adapter for this process. It is opaque,
// comparable, stable across repeated enumerations while the adapter is present,
// and intentionally not serializable.
type AdapterID struct{ token [16]byte }

// DeviceInfo describes a device that could be opened, without opening it.
// Callers choose on reported capabilities and limits rather than by trying and
// catching failures.
type DeviceInfo struct {
	ID           AdapterID
	Backend      Backend
	Name, Vendor string
	Software     bool

	// Capabilities is what this device can actually do. Queried before use.
	Capabilities Capabilities
	Limits       Limits
}

// ProbeStage identifies the native-probe phase that failed.
type ProbeStage int

const (
	ProbeLoadLibrary ProbeStage = iota
	ProbeCreateInstance
	ProbeEnumerateAdapters
	ProbeQueryDevice
)

// ProbeDiagnostic explains why a backend or adapter did not produce an openable
// device without hiding healthy adapters from other backends.
type ProbeDiagnostic struct {
	Backend Backend
	Stage   ProbeStage
	Err     error
}

// Enumeration separates openable adapters from probe failures.
type Enumeration struct {
	Devices     []DeviceInfo
	Diagnostics []ProbeDiagnostic
}

// AdapterRejection explains why automatic selection skipped one adapter.
type AdapterRejection struct {
	ID     AdapterID
	Reason string
}

// SelectionReport makes automatic selection reproducible in logs.
type SelectionReport struct {
	Selected           AdapterID
	EnvironmentBackend string
	Rejected           []AdapterRejection
}

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
	Limits        LimitConstraints
}

// LimitConstraints filters automatic selection. Zero fields are unconstrained;
// array components are compared independently.
type LimitConstraints struct {
	AtLeast Limits
	AtMost  Limits
}

// CPUMode selects the CPU backend's reported capability and limit profile.
type CPUMode int

const (
	CPUDeveloper CPUMode = iota
	CPUStrict
	CPUMimic
)

// DeviceProfile is a captured device contract used to reproduce another
// adapter's capability and numeric-limit behavior on the CPU backend.
type DeviceProfile struct {
	Info DeviceInfo
}

// CPUOptions configures the CPU oracle. StrictTargets is required in strict
// mode; Mimic is required in mimic mode.
type CPUOptions struct {
	Mode          CPUMode
	StrictTargets []Backend
	Mimic         *DeviceProfile
	SubgroupSize  int
	ShuffleSeed   uint64
}

// Device is an opened accelerator.
//
// A Device is safe for concurrent use. A [Recorder] obtained from it is not; see
// [Device.NewRecorder].
type Device struct {
	_ noCopy

	dev       driver.Device
	info      DeviceInfo
	queues    []QueueInfo
	handles   []*Queue
	queue     *Queue
	selection *SelectionReport

	mu     sync.Mutex
	closed bool
}

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
// A Recorder belongs to one goroutine. The [Graph] it produces is immutable but
// permits only one in-flight submission; build one graph per concurrent user.
func (d *Device) NewRecorder() *Recorder { panic(ErrNotImplemented) }

// noCopy makes `go vet` complain about copying a value that owns device state.
type noCopy struct{}

func (*noCopy) Lock()   {}
func (*noCopy) Unlock() {}
