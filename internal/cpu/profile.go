// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package cpu

import (
	"errors"
	"fmt"
	"math/bits"
	"sort"
	"strings"

	"golang.design/x/accel/internal/driver"
)

// Mode selects the reported capability and limit profile. It mirrors
// accel.CPUMode. See spec 006 section 5.
type Mode int

const (
	// Developer reports generous limits and every capability the CPU backend can
	// emulate. It is the zero value and is for development, not a portability
	// contract.
	Developer Mode = iota

	// Strict reports the intersection of the capabilities and the minimum
	// guaranteed bounds of an explicit target set. A kernel that builds here is
	// portable to that stated set, not to an implied future backend.
	Strict

	// Mimic reports a captured device's profile so a remote failure reproduces
	// locally.
	Mimic
)

// Options configures an opened CPU device. It mirrors accel.CPUOptions.
type Options struct {
	Mode          Mode
	StrictTargets []driver.Backend
	Mimic         *driver.Info
	SubgroupSize  int
	ShuffleSeed   uint64

	// LoseAtSubmission marks the device lost when it reaches this submission,
	// counting from one. Zero never loses.
	//
	// specs/001-device-resources.md section 7.4 calls the loss path close to
	// untestable at v0 -- the CPU backend cannot lose a device and Metal rarely
	// does -- and requires this injection mode for exactly that reason. Without
	// it, every error path behind ErrDeviceLost is code nobody runs until a
	// caller's driver restarts in production.
	LoseAtSubmission int
}

// backendNames are used in the errors this package reports. accel has its own
// Backend.String; this one exists so the seam does not need it.
var backendNames = map[driver.Backend]string{
	driver.BackendCPU:    "CPU",
	driver.BackendMetal:  "Metal",
	driver.BackendVulkan: "Vulkan",
	driver.BackendD3D12:  "D3D12",
	driver.BackendOpenGL: "OpenGL",
}

// portableFloor is the limit set every target backend is guaranteed to meet.
//
// Every number is sourced, because spec 006 rules that a confidently wrong pin
// is worse than an unknown, and spec 001 section 1.1 rules that a backend which
// cannot query a limit reports this floor rather than zero.
//
// Alignments over-report deliberately: rounding a bound offset to a larger
// multiple than the device needs wastes bytes and is never incorrect, whereas
// a maximum that over-reports admits work the device cannot run, so the maxima
// below are guaranteed minimums and the alignments are guaranteed-sufficient
// values.
var portableFloor = driver.Limits{
	// Spec 001 section 3.1: a multiple of 256 is always sufficient for a bound
	// storage or uniform range, and 16 for a copy endpoint, on every backend.
	MinStorageBufferOffsetAlignment: 256,
	MinUniformBufferOffsetAlignment: 256,
	MinBufferCopyOffsetAlignment:    16,

	// D3D12_TEXTURE_DATA_PITCH_ALIGNMENT and D3D12's default resource placement
	// alignment, which are the coarsest of the target set. Textures are deferred
	// until graphics work (spec 009 M1), so these are reported rather than used.
	MinBufferCopyRowPitchAlignment: 256,
	MinTexturePlacementAlignment:   65536,

	// WebGPU's default maxBufferSize, and Vulkan's guaranteed
	// maxMemoryAllocationCount. A pool is one device allocation, so MaxPools is
	// the number that makes pooling mandatory rather than merely efficient.
	MaxBufferBytes: 256 << 20,
	MaxPoolBytes:   256 << 20,
	MaxPools:       4096,

	// GLES 3.1 GL_MAX_TEXTURE_SIZE, GL_MAX_3D_TEXTURE_SIZE, and
	// GL_MAX_ARRAY_TEXTURE_LAYERS guaranteed minimums.
	MaxTextureExtent2D:    2048,
	MaxTextureExtent3D:    256,
	MaxTextureArrayLayers: 256,
	// Vulkan's maxVertexInputBindings required minimum, which is also what
	// Metal's buffer-index reservation leaves and what D3D12's input assembler
	// carries. Sixteen is the portable floor rather than this backend's own
	// ceiling -- the CPU rasterizer indexes buffers by slice position and has
	// none -- and the floor is what the oracle reports so a layout that passes
	// here passes everywhere.
	MaxVertexBuffers: 16,

	// Vulkan maxUniformBufferRange and GLES 3.1 GL_MAX_UNIFORM_BLOCK_SIZE
	// guaranteed minimums.
	MaxUniformBlockBytes: 16384,

	// Spec 002 section 1.5's portable floor table, which cites Vulkan, D3D12,
	// and GLES 3.1 guaranteed minimums. This is why that spec's GEMM uses a
	// 128-invocation workgroup rather than the 256 a 16x16 tile suggests.
	MaxWorkgroupSize:        [3]int{128, 128, 64},
	MaxWorkgroupInvocations: 128,
	MaxWorkgroupCount:       [3]int{65535, 65535, 65535},
	MaxSharedMemoryBytes:    16384,

	// Vulkan maxStorageBufferRange and GLES 3.1 GL_MAX_SHADER_STORAGE_BLOCK_SIZE
	// guaranteed minimums, and Vulkan's guaranteed
	// maxPerStageDescriptorStorageBuffers.
	MaxStorageBufferBindingBytes: 128 << 20,
	MaxBindingsPerKind:           4,

	// Four colour attachments is the portable floor: GLES 3.1 guarantees four,
	// and a deferred renderer's albedo, normal and position fit in it. Metal
	// and Vulkan report more, which is what the developer profile below
	// reflects.
	MaxColorAttachments: 4,

	// Spec 001 section 1.1: a device without subgroups reports 1/1 rather than
	// zero, so an opened device never has a zero-valued limit. Under the
	// conservative rule below no target's subgroup support is guaranteed, so the
	// strict intersection has no subgroups.
	MinSubgroupSize: 1,
	MaxSubgroupSize: 1,
}

// developerLimits are generous rather than portable. They are the CPU backend's
// own choice, bounded by host memory rather than by a driver, and they are not a
// portability contract: that is what [Strict] is for.
//
// The alignments are the exception and stay at the portable floor, because spec
// 001 section 3.1 already resolves the CPU backend's alignment column that way.
// Reporting a looser alignment in developer mode would let code allocate at an
// offset no GPU accepts and discover it on a different machine.

// maxDeveloperBytes is the developer profile's buffer and pool ceiling.
//
// Two gibibytes less one byte, and the "less one byte" is the point: Limits
// measures bytes in an int, which is 32 bits on 386 and arm, where a literal
// 1<<31 does not fit and this package does not compile at all. It was written
// that way, and nothing caught it, because the test matrix runs three 64-bit
// runners while the README told readers they could build for any GOOS. The
// cross-GOOS job in CI is what found it and is what keeps it found.
const maxDeveloperBytes = 1<<31 - 1

var developerLimits = driver.Limits{
	MinStorageBufferOffsetAlignment: portableFloor.MinStorageBufferOffsetAlignment,
	MinUniformBufferOffsetAlignment: portableFloor.MinUniformBufferOffsetAlignment,
	MinBufferCopyOffsetAlignment:    portableFloor.MinBufferCopyOffsetAlignment,
	MinBufferCopyRowPitchAlignment:  portableFloor.MinBufferCopyRowPitchAlignment,
	MinTexturePlacementAlignment:    portableFloor.MinTexturePlacementAlignment,

	MaxBufferBytes: maxDeveloperBytes,
	MaxPoolBytes:   maxDeveloperBytes,
	MaxPools:       1 << 20,

	MaxTextureExtent2D:    16384,
	MaxTextureExtent3D:    2048,
	MaxTextureArrayLayers: 2048,

	MaxUniformBlockBytes: 65536,

	MaxWorkgroupSize:        [3]int{1024, 1024, 64},
	MaxWorkgroupInvocations: 1024,
	MaxWorkgroupCount:       [3]int{1 << 20, 1 << 20, 1 << 20},
	MaxSharedMemoryBytes:    1 << 20,

	MaxStorageBufferBindingBytes: maxDeveloperBytes,
	MaxBindingsPerKind:           64,

	// Eight, which is Metal's and a common Vulkan figure. The CPU rasterizer
	// has no limit of its own -- a Framebuffer holds a slice -- so this is a
	// portability ceiling rather than a capacity one, and reporting no limit
	// would let a kernel that works here fail on every real device.
	// The developer profile has no ABI reservation to respect, so its ceiling
	// is generous. Strict mode reports the floor above, which is what makes a
	// layout that passes on the oracle pass on a real device.
	MaxVertexBuffers: 64,

	MaxColorAttachments: 8,

	MinSubgroupSize: defaultSubgroupSize,
	MaxSubgroupSize: defaultSubgroupSize,
}

// defaultSubgroupSize is 4 rather than 1 because a subgroup size of 1 makes
// shuffle and ballot degenerate and hides exactly the bugs the emulation exists
// to find, and small enough that a normal workgroup spans several subgroups so
// cross-subgroup errors and tail handling are exercised. See spec 006 section 5.
const defaultSubgroupSize = 4

// maxSubgroupSize bounds what [Options.SubgroupSize] may request in developer
// mode. A kernel whose result is subgroup-size independent is expected to run at
// 1, 4, 32, and 64 and match its no-subgroup fallback.
const maxSubgroupSize = 128

// developerCaps is every capability the CPU backend can emulate. The CPU column
// of spec 006's capability matrix is `yes` in every compute row.
var developerCaps = driver.Capabilities{
	Subgroups: true,
	SubgroupOps: driver.SubgroupBasic | driver.SubgroupVote | driver.SubgroupBallot |
		driver.SubgroupShuffle | driver.SubgroupArithmetic,

	F16Arithmetic:  true,
	BF16Arithmetic: true,
	I8DotProduct:   true,

	AtomicFloatAddStorage: true,
	AtomicFloatAddShared:  true,

	DenormF32Preserved: true,
	DenormF16Preserved: true,
	InfNaNProduced:     true,
	ContractionControl: true,

	// MemoryShared is real on the CPU backend: host memory is device memory.
	SharedMemoryKind: true,

	Graphics: true,
	// Presentation and Multisampling are `no` for the CPU backend in spec 006,
	// and rasterizer-ordered access is `emul`, which the reference rasterizer
	// provides by construction. Graphics itself is post-v0 (spec 005).
	RasterizerOrderedAccess: true,
	IndirectDispatch:        true,

	// Spec 006: the CPU backend replays the planned node list directly. There is
	// no native replayable object to lower a graph into.
	NativeGraphReplay: false,
}

// baselines is the published conservative baseline capability set per backend,
// derived mechanically from spec 006's capability matrix by one rule:
//
//	yes, emul  -> present
//	cap, ?, gated, no, n/a -> absent
//
// `cap` and `?` are absent because a strict profile that assumed them would
// promise portability the device has not been asked about, and `gated` is absent
// because no query resolves it: a distribution decision does. Nothing here is
// invented, and nothing is measured; a measured value replaces its row at first
// contact with that backend, per spec 006's discipline.
//
// A backend missing from this map has no published baseline and is rejected by
// [Strict]. The limit half of every baseline is [portableFloor], because spec
// 006 declines to pin per-backend numeric bounds until they are measured.
var baselines = map[driver.Backend]driver.Capabilities{
	// Metal: f16 arithmetic yes, i8/u8 and f16 storage yes, indirect dispatch
	// yes, graphics yes. Everything else in the Metal column is cap or ?.
	driver.BackendMetal: {
		F16Arithmetic:    true,
		BF16Arithmetic:   false,
		IndirectDispatch: true,
		Graphics:         true,
	},
	// Vulkan: storage rows are yes or emul; every arithmetic and subgroup row is
	// cap. Contraction control is yes.
	driver.BackendVulkan: {
		IndirectDispatch:   true,
		ContractionControl: true,
		Graphics:           true,
	},
	// D3D12: f16 arithmetic and subgroup ops are gated, atomics float are no.
	// Contraction control is yes.
	driver.BackendD3D12: {
		IndirectDispatch:   true,
		ContractionControl: true,
		Graphics:           true,
	},
	// GLES 3.1: no subgroups, no f16 arithmetic, no float atomics, no i8 dot
	// product, and contraction control is unknown.
	driver.BackendOpenGL: {
		IndirectDispatch: true,
		Graphics:         true,
	},
}

var (
	// ErrStrictTargets reports a target set that cannot produce a profile.
	ErrStrictTargets = errors.New("accel: invalid strict target set")

	// ErrSubgroupSize reports a subgroup size outside the profile's bounds.
	ErrSubgroupSize = errors.New("accel: invalid subgroup size")

	// ErrMimicProfile reports a missing mimic profile.
	ErrMimicProfile = errors.New("accel: mimic mode requires a device profile")
)

// resolve turns options into the profile an opened device reports.
func resolve(opts Options) (driver.Info, int, error) {
	var caps driver.Capabilities
	var lim driver.Limits
	var name string

	switch opts.Mode {
	case Developer:
		caps, lim, name = developerCaps, developerLimits, "CPU (pure Go, developer)"

	case Strict:
		targets, err := normalizeTargets(opts.StrictTargets)
		if err != nil {
			return driver.Info{}, 0, err
		}
		caps, lim = intersect(targets)
		names := make([]string, len(targets))
		for i, t := range targets {
			names[i] = backendNames[t]
		}
		name = fmt.Sprintf("CPU (pure Go, strict: %s)", strings.Join(names, "+"))

	case Mimic:
		if opts.Mimic == nil {
			return driver.Info{}, 0, ErrMimicProfile
		}
		caps, lim = opts.Mimic.Capabilities, opts.Mimic.Limits
		name = fmt.Sprintf("CPU (pure Go, mimicking %q)", opts.Mimic.Name)

	default:
		return driver.Info{}, 0, fmt.Errorf("accel: unknown CPU mode %d", opts.Mode)
	}

	size, err := resolveSubgroupSize(opts.SubgroupSize, caps, lim)
	if err != nil {
		return driver.Info{}, 0, err
	}
	// The emulation runs at exactly one lane count, so both bounds report it.
	// SubgroupSize controls emulation only and never changes the reported target
	// intersection, so a device without subgroups keeps the 1/1 sentinel.
	if caps.Subgroups {
		lim.MinSubgroupSize, lim.MaxSubgroupSize = size, size
	}

	return driver.Info{
		Backend:      driver.BackendCPU,
		Name:         name,
		Vendor:       "golang.design",
		Software:     false, // A CPU device is not a software *GPU*; see spec 006 section 6.
		Capabilities: caps,
		Limits:       lim,
	}, size, nil
}

// normalizeTargets validates and sorts a strict target set. Order does not
// affect the result and is normalized so the reported device name is stable.
func normalizeTargets(targets []driver.Backend) ([]driver.Backend, error) {
	if len(targets) == 0 {
		return nil, fmt.Errorf("%w: strict mode requires a non-empty target set", ErrStrictTargets)
	}
	seen := make(map[driver.Backend]bool, len(targets))
	out := make([]driver.Backend, 0, len(targets))
	for _, t := range targets {
		switch {
		case t == driver.BackendCPU:
			return nil, fmt.Errorf("%w: the CPU backend cannot be its own portability target", ErrStrictTargets)
		case seen[t]:
			return nil, fmt.Errorf("%w: %s appears twice", ErrStrictTargets, backendNames[t])
		}
		if _, ok := baselines[t]; !ok {
			name, known := backendNames[t]
			if !known {
				name = fmt.Sprintf("backend %d", int(t))
			}
			return nil, fmt.Errorf("%w: %s has no published baseline profile", ErrStrictTargets, name)
		}
		seen[t] = true
		out = append(out, t)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out, nil
}

// intersect reduces a target set to the capabilities every member guarantees and
// the limits every member guarantees to meet.
func intersect(targets []driver.Backend) (driver.Capabilities, driver.Limits) {
	caps := baselines[targets[0]]
	for _, t := range targets[1:] {
		caps = intersectCaps(caps, baselines[t])
	}
	return caps, portableFloor
}

func intersectCaps(a, b driver.Capabilities) driver.Capabilities {
	return driver.Capabilities{
		Subgroups:               a.Subgroups && b.Subgroups,
		SubgroupOps:             a.SubgroupOps & b.SubgroupOps,
		F16Arithmetic:           a.F16Arithmetic && b.F16Arithmetic,
		BF16Arithmetic:          a.BF16Arithmetic && b.BF16Arithmetic,
		I8DotProduct:            a.I8DotProduct && b.I8DotProduct,
		AtomicFloatAddStorage:   a.AtomicFloatAddStorage && b.AtomicFloatAddStorage,
		AtomicFloatAddShared:    a.AtomicFloatAddShared && b.AtomicFloatAddShared,
		DenormF32Preserved:      a.DenormF32Preserved && b.DenormF32Preserved,
		DenormF16Preserved:      a.DenormF16Preserved && b.DenormF16Preserved,
		InfNaNProduced:          a.InfNaNProduced && b.InfNaNProduced,
		ContractionControl:      a.ContractionControl && b.ContractionControl,
		SharedMemoryKind:        a.SharedMemoryKind && b.SharedMemoryKind,
		Graphics:                a.Graphics && b.Graphics,
		Presentation:            a.Presentation && b.Presentation,
		Multisampling:           a.Multisampling && b.Multisampling,
		RasterizerOrderedAccess: a.RasterizerOrderedAccess && b.RasterizerOrderedAccess,
		IndirectDispatch:        a.IndirectDispatch && b.IndirectDispatch,
		NativeGraphReplay:       a.NativeGraphReplay && b.NativeGraphReplay,
	}
}

// resolveSubgroupSize validates a requested emulation lane count against the
// resolved profile. Zero selects the profile's default.
func resolveSubgroupSize(want int, caps driver.Capabilities, lim driver.Limits) (int, error) {
	if !caps.Subgroups {
		// A device reporting no subgroups emulates none. Only the degenerate size
		// is expressible, and asking for another is a request the profile cannot
		// satisfy rather than one it silently ignores.
		if want > 1 {
			return 0, fmt.Errorf("%w: %d requested but this profile reports no subgroups", ErrSubgroupSize, want)
		}
		return 1, nil
	}
	if want == 0 {
		if lim.MaxSubgroupSize > 0 {
			return min(defaultSubgroupSize, lim.MaxSubgroupSize), nil
		}
		return defaultSubgroupSize, nil
	}
	if want < 0 || bits.OnesCount(uint(want)) != 1 {
		return 0, fmt.Errorf("%w: %d is not 1 or a power of two", ErrSubgroupSize, want)
	}
	if want > maxSubgroupSize {
		return 0, fmt.Errorf("%w: %d exceeds the %d the CPU backend emulates", ErrSubgroupSize, want, maxSubgroupSize)
	}
	return want, nil
}
