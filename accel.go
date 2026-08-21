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

// OpenBest opens the most capable available device, preferring GPU backends over
// [BackendCPU]. Unlike [Open] this is an explicit request to choose.
func OpenBest() (*Device, error) { panic(ErrNotImplemented) }

// Device is an opened accelerator.
//
// A Device is safe for concurrent use. A [Recorder] obtained from it is not; see
// [Device.NewRecorder].
type Device struct{ _ noCopy }

// Info reports what this device is and what it can do.
func (d *Device) Info() DeviceInfo { panic(ErrNotImplemented) }

// Queue returns the device's default queue for submitting work.
func (d *Device) Queue() *Queue { panic(ErrNotImplemented) }

// NewPool allocates a memory pool of the given kind and size, from which buffers
// are suballocated. See specs/001-device-resources.md for why allocation is
// pooled rather than per resource.
func (d *Device) NewPool(kind MemoryKind, bytes int) (*Pool, error) {
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
