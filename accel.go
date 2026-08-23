// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package accel runs compute work on a GPU from Go, with no cgo.
//
// You write a kernel in a subset of Go. It is compiled ahead of time by
// cmd/accel-kernel under go generate, and runs on whichever backend the machine
// has: Metal, or a pure-Go CPU device that produces the same results and needs
// no GPU at all.
//
//	go get golang.design/x/accel
//
// This is the device layer. Its vocabulary is buffers, textures, workgroups and
// barriers; it knows nothing about tensors or meshes. For inference, use
// golang.design/x/accel/tensor, which is built on this package and deals in
// shapes and operators instead.
//
// # Getting started
//
// Kernels live in a package of their own, because the generator type-checks the
// package it compiles and cannot run on a package that already refers to the
// symbol it is about to define. Given a generated kernels.ScaleKernel:
//
//	dev, err := accel.OpenCPU(accel.CPUOptions{})
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer dev.Close()
//
//	pipe, err := dev.NewComputePipeline(accel.ComputePipelineDescriptor{
//		Kernel: &kernels.ScaleKernel,
//	})
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer pipe.Close()
//
//	err = dev.Queue().Run(func(r *accel.Recorder) {
//		r.Dispatch(pipe, bindings, accel.WorkgroupCount{X: 4})
//	})
//
// The README has the whole program, buffers included.
//
// To select a GPU instead, use [OpenBest]. It does not choose the CPU backend
// unless [Policy].AllowCPU is set, so a caller who asked for a GPU and has none
// gets an error rather than a silent substitution.
//
// # What runs today
//
// Two backends: Metal on darwin, and a pure-Go CPU backend everywhere.
// [Enumerate] reports what this machine has. [BackendVulkan], [BackendD3D12]
// and [BackendOpenGL] exist in the [Backend] enumeration and are not built.
//
// On both backends: buffers, textures, memory pools, uploads and readbacks,
// compute dispatch both direct and indirect, command graphs, cooperative
// kernels with workgroup-shared memory and barriers, atomics, and subgroup
// reductions. The two agree exactly where the kernel's arithmetic is exact, and
// within a stated ceiling where it reaches a bounded primitive such as exp.
//
// Uniform blocks are encoded by a generated std140 codec, so you supply a
// [UniformBuffer] and never write a padding offset. Texture data is tightly
// packed at this API's boundary — row r begins at r*width*bpp — so a readback
// sized width*height*bpp is right whatever pitch the device stores.
//
// A kernel the Metal target cannot lower is refused by name at
// [Device.NewComputePipeline] rather than run on the CPU instead, so a device
// you selected is never quietly bypassed.
//
// # What does not
//
// The API is under construction and will change.
//
// Subgroup shuffles and scans are unbuilt.
//
// Graphics is designed and being written, and none of it is in this API yet.
// The stage receivers and the sampler family were exported ahead of the code
// that gives them meaning and have been withdrawn until a stage can run; see
// specs/036-documentation.md's freeze record.
//
// # The model
//
// Work is recorded into a [Graph], which is immutable once built and can be
// submitted many times with its inputs rebound in between ([Graph.Rebind]).
// Validation, memory planning and barrier placement happen once, at
// [Recorder.Build], not on every submission.
//
//	rec := dev.NewRecorder()
//	rec.Dispatch(pipeline, bindings, WorkgroupCount{X: n})
//	g, err := rec.Build()   // validate, plan memory, compute barriers, lower
//	...
//	f := dev.Queue().Submit(g)
//	f.Wait()
//
// You do not write barriers. Each node declares what it reads and writes, and
// the builder compares those as byte ranges, ordering only the pairs that
// really conflict. [Graph.Edges], [Graph.Hazards] and [Graph.Barriers] report
// what it decided, so a graph that does not overlap can be explained rather
// than timed.
//
// Buffers the builder owns share memory when their uses cannot overlap.
// [Graph.Memory] reports what that saved and [Recorder.BuildNaive] rebuilds the
// same graph with aliasing off, to isolate a suspected planning bug. Several
// graphs can share one [TransientPool], sized to the largest rather than the
// sum, at the price that they cannot execute at the same time.
//
// Memory comes from pools rather than one allocation per resource, because a
// model has thousands of tensors and drivers cap the number of allocations. A
// pool is exactly one device allocation and never grows.
//
// The reasoning behind these choices is in specs/: 003 for the graph, 001 for
// memory, 000 for the decisions the rest follow from.
package accel

import (
	"encoding/hex"
	"errors"
	"sync"

	"golang.design/x/accel/internal/driver"
)

// ErrNotImplemented marks a declaration that exists so its shape is fixed but
// whose implementation has not arrived.
//
// It has no user today: its only one was the sampler family, withdrawn with the
// rest of the unbuilt graphics surface. It stays because
// specs/032-stage-abi.md and specs/033-render-api.md will give it users again,
// and because the rule it carries is worth keeping — a declaration that exists
// only for its shape **panics** rather than returning, so a caller cannot write
// a plausible handler for a path that can never succeed.
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

// String is the adapter's stable identity, as hex.
//
// Exported because a layer above this one needs it and cannot reach the token:
// specs/007-tensor-layer.md requires a plan cache's key to include "the device
// identity", and without this the best a caller could do is the device's name,
// which two identical GPUs in one machine share.
//
// Stable within a process and comparable across enumerations, which is what
// [driver.Adapter] promises of the token underneath. Not stable across machines
// or driver versions, and not meant to be: it is an identity, not a fingerprint.
func (id AdapterID) String() string { return hex.EncodeToString(id.token[:]) }

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
	Selected AdapterID
	Rejected []AdapterRejection
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

	// LoseAtSubmission marks the device lost when it reaches this submission,
	// counting from one. Zero never loses.
	//
	// It is fault injection, and it is here rather than in a test helper because
	// specs/001-device-resources.md section 7.4 requires it: the CPU backend
	// cannot lose a device and Metal rarely does, so without a way to ask for it
	// the whole terminal-loss path is code nobody runs until a caller's driver
	// restarts in production. It has no effect on any other backend.
	LoseAtSubmission int
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

	state resourceState

	mu             sync.Mutex
	pools          []*Pool
	implicit       map[MemoryKind]*blockSet
	implicitBlocks int

	// graphs counts built, unclosed graphs. They own transient memory and a
	// compiled executable, so a device closing under one would strand both;
	// counted here for the same reason implicit blocks are, since a graph has a
	// handle and is therefore a live child.
	graphs int

	// pipelines counts created, unclosed compute pipelines, for the same reason
	// graphs are counted: a device closing under one would strand it.
	pipelines int
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

// NewTexture creates a texture. Bytes per pixel always derives from the format,
// and depth formats carry backend constraints the implementation enforces; see
// docs/conventions.md.
func (d *Device) NewTexture(desc TextureDescriptor) (*Texture, error) {
	return d.newTexture(desc)
}

// NewComputePipeline compiles a kernel into a pipeline.
//
// Workgroup size is fixed here rather than at dispatch, because backends need it
// at compile time: it appears in the GLSL layout qualifier and in Metal's
// threads-per-threadgroup. See specs/002-compute-model.md.
func (d *Device) NewComputePipeline(desc ComputePipelineDescriptor) (*ComputePipeline, error) {
	return d.newComputePipeline(desc)
}

// noCopy makes `go vet` complain about copying a value that owns device state.
type noCopy struct{}

func (*noCopy) Lock()   {}
func (*noCopy) Unlock() {}
