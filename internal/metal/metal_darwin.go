// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"sync"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/mslabi"
	"golang.design/x/accel/internal/mtl"
)

// ErrNoDevice reports that Metal loaded and enumerated nothing.
//
// It is a distinct error rather than an empty list, because an empty list is
// indistinguishable from a build with no Metal at all -- and
// specs/006-backends.md section 6.4 requires a caller to be able to tell "this
// machine has no Metal device" from "this binary was not built with Metal".
// The layer above turns it into a probe diagnostic, which is not an open error.
var ErrNoDevice = errors.New("accel: Metal is present and reports no device")

// Adapters reports every Metal device on this machine.
//
// A machine with no device returns [ErrNoDevice], not an empty list.
//
// Enumerated once per process. The layer above calls this on every Enumerate,
// OpenDevice and OpenBest, and each enumeration used to retain a fresh set of
// MTLDevice objects that nothing closed and compile the subgroup probe on each
// of them, because the probe is cached per object. The devices are a process
// fact, so they are found once and every caller gets the same adapters.
func Adapters() ([]driver.Adapter, error) {
	adaptersOnce.Do(func() {
		devs, err := mtl.Devices()
		if err != nil {
			adaptersErr = err
			return
		}
		adaptersList, adaptersErr = adaptersFrom(devs, infoFor)
	})
	if adaptersErr != nil {
		return nil, adaptersErr
	}
	// A copy, so a caller appending to the result appends to their own slice.
	return append([]driver.Adapter(nil), adaptersList...), nil
}

var (
	adaptersOnce sync.Once
	adaptersList []driver.Adapter
	adaptersErr  error
)

// adaptersFrom wraps each device, or releases every one of them when any
// cannot answer for itself.
//
// Every one, including those already wrapped: a device that cannot report its
// limits is not enumerated, and the enumeration is all or nothing, so the
// devices that had answered would otherwise stay retained with no adapter to
// reach them from. The refusal becomes a probe diagnostic one layer up, which
// is the difference specs/006-backends.md section 6.4 draws between a probe
// failure and an open error.
func adaptersFrom(devs []*mtl.Device, info func(*mtl.Device) (driver.Info, error)) ([]driver.Adapter, error) {
	out := make([]driver.Adapter, 0, len(devs))
	for _, d := range devs {
		i, err := info(d)
		if err != nil {
			for _, x := range devs {
				x.Close()
			}
			return nil, err
		}
		out = append(out, &adapter{dev: d, info: i})
	}
	if len(out) == 0 {
		return nil, ErrNoDevice
	}
	return out, nil
}

type adapter struct {
	dev  *mtl.Device
	info driver.Info
}

func (a *adapter) Info() driver.Info { return a.info }

// Token identifies this adapter within the process.
//
// It is seeded with the device's registryID rather than with an index, so a
// machine with two GPUs gives two stable tokens and enumerating twice gives the
// same one for the same device. An index would be stable only until something
// changed the order.
func (a *adapter) Token() [16]byte {
	var t [16]byte
	sum := sha256.Sum256(fmt.Appendf(nil, "golang.design/x/accel:metal:%d", a.dev.RegistryID()))
	copy(t[:], sum[:])
	return t
}

// Open opens the adapter. opts is nil; Metal has no options at this milestone.
func (a *adapter) Open(opts any) (driver.Device, error) {
	if opts != nil {
		return nil, fmt.Errorf("accel: the Metal backend takes no options yet, got %T", opts)
	}
	return &device{dev: a.dev, info: a.info, queue: a.dev.NewQueue()}, nil
}

type device struct {
	dev   *mtl.Device
	info  driver.Info
	queue *mtl.Queue

	mu     sync.Mutex
	closed bool

	// blocks and executables count what is open against this device, so
	// Close can refuse rather than release the queue and the pipelines out
	// from under them. The CPU backend's Close makes the same refusal for its
	// allocations, and an executable here is the thing a submission runs on.
	blocks      int
	executables int

	// lost is non-nil once a submission reported the device gone, and stays
	// non-nil. See [device.Lost].
	lost error

	// pipelines caches one compiled pipeline per kernel record. Compiling MSL
	// is a call into the device compiler and takes milliseconds, so a graph
	// resubmitted a thousand times must not pay it a thousand times. The key is
	// the record's digest, which is what identifies the generated source.
	//
	// Guarded by pipeMu rather than mu, because the cache is read from two
	// places with different locks above them: Compile, under mu, and every
	// dispatch a Submit encodes, under the executable's own mutex only. A
	// separate lock is what lets a submission on one executable run while
	// another is compiled. pipeMu is taken after mu where both are held and
	// never before it.
	pipeMu    sync.Mutex
	pipelines map[string]*mtl.Pipeline
}

func (d *device) Info() driver.Info { return d.info }

// Supports reports which memory kinds this device can back.
//
// Every kind, because unified memory makes all of them real: there is no kind
// here that would have to be emulated with a copy the caller cannot see.
func (d *device) Supports(kind driver.MemoryKind) bool {
	switch kind {
	case driver.MemoryDevice, driver.MemoryUpload, driver.MemoryReadback, driver.MemoryShared:
		return true
	}
	return false
}

// storageFor maps a memory kind to a Metal storage mode.
//
// MemoryDevice is private even though unified memory would let it be shared and
// faster. specs/006-backends.md section 1 makes Block.Bytes the authority on
// mappability, and a backend that maps device memory on an Apple GPU and not on
// an Intel one turns a portability bug into a machine-specific one. The cost is
// a blit in Block.Write and Block.Read, which is the immediate transfer path
// and is allowed to be slow.
func storageFor(kind driver.MemoryKind) (mtl.StorageMode, error) {
	switch kind {
	case driver.MemoryDevice:
		return mtl.StoragePrivate, nil
	case driver.MemoryUpload, driver.MemoryReadback, driver.MemoryShared:
		return mtl.StorageShared, nil
	}
	return 0, fmt.Errorf("accel: the Metal backend has no storage mode for memory kind %d", kind)
}

func (d *device) Alloc(kind driver.MemoryKind, bytes int, label string) (driver.Block, error) {
	mode, err := storageFor(kind)
	if err != nil {
		return nil, err
	}
	d.mu.Lock()
	if d.closed {
		d.mu.Unlock()
		return nil, fmt.Errorf("accel: alloc %q: the device is closed", label)
	}
	d.mu.Unlock()

	buf, err := d.dev.NewBuffer(bytes, mode)
	if err != nil {
		return nil, fmt.Errorf("accel: alloc %q: %w", label, err)
	}
	d.mu.Lock()
	d.blocks++
	d.mu.Unlock()
	return &block{dev: d, buf: buf, label: label}, nil
}

// Lost reports device loss, stickily.
//
// Metal has no device-level flag for this: loss surfaces as an error on a
// command buffer, which is a property of one submission. So the answer is
// *derived* from what submissions have reported, and once derived it never
// clears -- specs/001-device-resources.md section 7.4 makes loss terminal,
// because a driver reset that produced one failure and then appeared to recover
// would leave a caller running on resources whose contents are undefined.
//
// Not every command buffer error is loss. A kernel that ran off the end of a
// buffer faults one submission and leaves the device usable, and reporting that
// as loss would turn a bug into an unrecoverable device. [isDeviceLoss] draws
// the line, and draws it narrowly.
func (d *device) Lost() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.lost
}

// noteSubmissionError records what one submission reported.
//
// Called with the error from a completed command buffer, including nil.
func (d *device) noteSubmissionError(err error) {
	if err == nil || !isDeviceLoss(err) {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.lost == nil {
		d.lost = fmt.Errorf("%w: %v", driver.ErrDeviceLost, err)
	}
}

// isDeviceLoss reports whether a command buffer error means the device is gone
// rather than the work being wrong.
//
// Classified by the MTLCommandBufferError code, which is what Metal assigns
// to say exactly this. Two codes mean the device: DeviceRemoved, an eGPU
// unplugged, and AccessRevoked, the process barred from the GPU after too many
// hangs. Everything else is about one submission: Timeout is a kernel that ran
// too long, PageFault one that read off the end of a buffer, and the device
// runs the next command buffer as if nothing happened. This used to match
// message text, and matched "Caused GPU Hang" -- the timeout's usual wording
// -- so a kernel with an infinite loop made the device unusable for good,
// which is the opposite of what [device.Lost] promises.
//
// Narrow on purpose. A false positive here is worse than a false negative,
// because it turns a recoverable kernel bug into a device a caller must throw
// away, and specs/001-device-resources.md section 7.4 makes that unrecoverable
// by design. An error that is not a command buffer's at all is not loss.
func isDeviceLoss(err error) bool {
	var cbe *mtl.CommandBufferError
	if !errors.As(err, &cbe) || cbe.Domain != mtl.CommandBufferErrorDomain {
		return false
	}
	switch cbe.Code {
	case mtl.CommandBufferErrorDeviceRemoved, mtl.CommandBufferErrorAccessRevoked:
		return true
	}
	return false
}

// Close releases the queue and the compiled pipelines.
//
// Refused while anything is open against the device. An executable that is
// open may be mid-submission, and its next Submit would encode into a released
// queue; a block that is live is memory a caller still reads. executable.Close
// already refuses while a submission is in flight, so counting open
// executables covers in-flight work too. Closing twice is harmless.
func (d *device) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil
	}
	if d.blocks > 0 || d.executables > 0 {
		return fmt.Errorf("accel: close Metal device: %d allocations and %d executables "+
			"are still open", d.blocks, d.executables)
	}
	d.closed = true
	d.pipeMu.Lock()
	for _, p := range d.pipelines {
		p.Close()
	}
	d.pipelines = nil
	d.pipeMu.Unlock()
	d.queue.Close()
	return nil
}

// block is one device allocation, backing one pool.
type block struct {
	dev   *device
	buf   *mtl.Buffer
	label string
}

func (b *block) Bytes() []byte { return b.buf.Bytes() }
func (b *block) Size() int     { return b.buf.Size() }

func (b *block) Free() {
	b.buf.Close()
	b.dev.mu.Lock()
	b.dev.blocks--
	b.dev.mu.Unlock()
}

// Write and Read move bytes for memory Bytes does not map.
//
// Synchronous, per the driver interface: this is the immediate transfer path of
// specs/001-device-resources.md section 8.1, not the recorded one. The staging
// buffer is per call rather than pooled, because the recorded path is what a
// caller uses when the cost matters.
func (b *block) Write(off int, src []byte) error {
	if host := b.buf.Bytes(); host != nil {
		if err := checkRange(off, len(src), len(host)); err != nil {
			return err
		}
		copy(host[off:], src)
		return nil
	}
	if err := checkRange(off, len(src), b.buf.Size()); err != nil {
		return err
	}
	stage, err := b.dev.dev.NewBuffer(len(src), mtl.StorageShared)
	if err != nil {
		return err
	}
	defer stage.Close()
	copy(stage.Bytes(), src)
	return b.dev.blit(func(e *mtl.BlitEncoder) { e.Copy(b.buf, off, stage, 0, len(src)) })
}

func (b *block) Read(off int, dst []byte) error {
	if host := b.buf.Bytes(); host != nil {
		if err := checkRange(off, len(dst), len(host)); err != nil {
			return err
		}
		copy(dst, host[off:])
		return nil
	}
	if err := checkRange(off, len(dst), b.buf.Size()); err != nil {
		return err
	}
	stage, err := b.dev.dev.NewBuffer(len(dst), mtl.StorageShared)
	if err != nil {
		return err
	}
	defer stage.Close()
	if err := b.dev.blit(func(e *mtl.BlitEncoder) { e.Copy(stage, 0, b.buf, off, len(dst)) }); err != nil {
		return err
	}
	copy(dst, stage.Bytes())
	return nil
}

func checkRange(off, n, limit int) error {
	if off < 0 || n < 0 || off > limit || n > limit-off {
		return fmt.Errorf("accel: range [%d, %d) is outside a %d-byte block", off, off+n, limit)
	}
	return nil
}

// blit runs one synchronous copy pass.
func (d *device) blit(encode func(*mtl.BlitEncoder)) error {
	cb := d.queue.Begin()
	defer cb.Close()
	e := cb.Blit()
	encode(e)
	e.End()
	cb.Commit()
	cb.Wait()
	return cb.Err()
}

// infoFor builds the capability and limit report for a device.
//
// Every row is one of four kinds, and specs/021-metal-bringup.md section 3 says
// which: queried from the device, derived from a query, a documented constant,
// or deliberately under-reported. The last is the interesting one. A capability
// this backend cannot yet lower is reported absent, because over-reporting
// means a kernel is accepted and then produces a wrong answer, while
// under-reporting means it is refused with a name.
func infoFor(d *mtl.Device) (driver.Info, error) {
	maxThreads := d.MaxThreadsPerThreadgroup
	// Measured by compiling a trivial kernel and reading its execution width,
	// because MTLDevice has no query for it. A device that cannot answer is not
	// enumerated at all: specs/001-device-resources.md section 1.1 forbids an
	// opened device reporting a zero limit, so the alternative would be handing
	// a caller a number they cannot use.
	width := d.SubgroupSize()
	if width <= 0 {
		return driver.Info{}, fmt.Errorf("accel: %s reports no SIMD width, so its subgroup "+
			"limits would be zero and specs/001-device-resources.md section 1.1 forbids that", d.Name())
	}
	return driver.Info{
		Backend: driver.BackendMetal,
		Name:    d.Name(),
		Vendor:  "Apple",
		// Software is false even for a device that is not the fastest one
		// present: Metal exposes no software rasterizer, and LowPower means an
		// integrated GPU rather than an emulated one.
		Software: false,
		Capabilities: driver.Capabilities{
			// Reported present because the MSL target lowers them
			// (specs/022-msl-target.md section 2). A capability is about what a
			// caller can use, so it follows the emitter rather than the
			// hardware: this device has had SIMD reductions all along, and
			// reporting them before anything could emit them would have let a
			// kernel be accepted and then refused at graph build.
			Subgroups: true,
			// Basic gives the size, lane index and group index; arithmetic
			// gives simd_sum, simd_min, simd_max and the prefix sums; vote
			// gives simd_any, simd_all and simd_is_first; shuffle gives
			// simd_broadcast, simd_shuffle, simd_shuffle_xor and the relative
			// shuffles up and down.
			//
			// Ballot is absent because simd_ballot returns a simd_vote rather
			// than an integer, and the conversion is family-dependent. It is
			// the one subgroup bit this device has and cannot spell, and the
			// emitter refuses the operation by name to match.
			SubgroupOps: driver.SubgroupBasic | driver.SubgroupArithmetic |
				driver.SubgroupVote | driver.SubgroupShuffle,

			// Still absent. atomic<float> is a Metal *version* capability
			// rather than a spelling, so it needs the family query this table
			// does not make yet, and the emitter refuses an f32 atomic by name.
			AtomicFloatAddStorage: false,
			AtomicFloatAddShared:  false,

			// Present: specs/023-metal-graph.md encodes it, and the count is
			// clamped on the device before the dispatch reads it, so the
			// unconditional clamp specs/003-command-graph.md requires costs no
			// readback.
			IndirectDispatch: true,

			InfNaNProduced: true,

			// Verified rather than claimed: the emitter's pragma turns
			// contraction off and a device test checks that it does, and that
			// the default does not.
			ContractionControl: true,

			// Unified memory: a shared allocation is the memory the GPU reads,
			// not a staging copy of it.
			SharedMemoryKind: true,

			// Graphics and Presentation are true since 2026-08-24: this backend
			// runs render passes and presents to a CAMetalLayer. They read
			// false until then, which made Capabilities disagree with what
			// NewRenderPipeline and NewWindowSurface actually accept -- the
			// inverse of specs/006-backends.md decision 6, where absence is
			// reported: presence was not.
			//
			// A command buffer is single-submit, so there is still no native
			// replay (specs/006-backends.md section 4.3).
			Graphics:                true,
			Presentation:            true,
			Multisampling:           false,
			RasterizerOrderedAccess: false,
			NativeGraphReplay:       false,
		},
		Limits: driver.Limits{
			// Metal requires a buffer offset to be a multiple of 4 on Apple
			// silicon and 256 for constant buffers on macOS. 256 is the
			// coarsest of those and is therefore always sufficient; it matches
			// the portable floor the CPU backend reports, so a graph built
			// against one runs on the other. Lowering it needs a measurement,
			// not a reading.
			MinStorageBufferOffsetAlignment: 256,
			MinUniformBufferOffsetAlignment: 256,
			MinBufferCopyOffsetAlignment:    16,

			// Reported rather than used: textures are not in this milestone.
			// MTLBlitCommandEncoder requires a 256-byte row pitch for
			// buffer-to-texture copies on macOS.
			MinBufferCopyRowPitchAlignment: 256,
			MinTexturePlacementAlignment:   65536,
			MaxTextureExtent2D:             16384,
			MaxTextureExtent3D:             2048,
			MaxTextureArrayLayers:          2048,
			// A vertex stage's uniforms begin where its vertex buffers end,
			// because Metal gives the two one buffer index space. The
			// reservation is mslabi's and this is where it becomes a *device's*
			// limit rather than every device's.
			MaxVertexBuffers: mslabi.StageVertexBufferLimit,

			// Queried. maxBufferLength is the largest single allocation, and a
			// pool is exactly one allocation.
			MaxBufferBytes:               d.MaxBufferBytes,
			MaxPoolBytes:                 d.MaxBufferBytes,
			MaxStorageBufferBindingBytes: d.MaxBufferBytes,

			// Metal documents no ceiling on the number of allocations, so this
			// is a sanity bound rather than a device fact. It is far above any
			// pooling strategy this library would produce.
			MaxPools: 1 << 20,

			// A compute encoder has 31 buffer argument slots, which is a fixed
			// property of the API rather than of the device.
			MaxBindingsPerKind: 31,

			// Eight colour attachments, which every Metal family reports and
			// which the API caps at regardless of device. A constant rather
			// than a query because Metal exposes no counterpart to ask.
			MaxColorAttachments: 8,

			// The constant address space is not separately bounded on Apple
			// silicon. 64 KiB is what every backend in the target set
			// guarantees, and under-reporting a ceiling costs nothing here.
			MaxUniformBlockBytes: 65536,

			// Queried: -maxThreadsPerThreadgroup and
			// -maxThreadgroupMemoryLength.
			MaxWorkgroupSize: [3]int{
				int(maxThreads.Width), int(maxThreads.Height), int(maxThreads.Depth),
			},
			MaxWorkgroupInvocations: int(maxThreads.Width),
			MaxSharedMemoryBytes:    d.MaxThreadgroupMemoryBytes,

			// Metal documents no threadgroup-count ceiling. MaxInt32 says so
			// without pretending to a number the device would confirm.
			MaxWorkgroupCount: [3]int{math.MaxInt32, math.MaxInt32, math.MaxInt32},

			// Measured, not tabled. The width is a property of a compiled
			// pipeline rather than of the device, so it is read from one. Min
			// and Max are equal because Apple silicon has a single width; a
			// device that varied would report the range it varies over, and
			// this is the number that would have to change.
			//
			// Reported even though Capabilities.Subgroups is false. The two
			// answer different questions: the width is a fact about the
			// hardware, and the capability is whether this backend can lower a
			// kernel that uses it.
			MinSubgroupSize: width,
			MaxSubgroupSize: width,
		},
	}, nil
}
