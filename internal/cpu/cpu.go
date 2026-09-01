// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package cpu is the pure-Go backend.
//
// It is a first-class backend and the correctness oracle every other backend is
// verified against, not a fallback: it is always available, it enforces the
// intersection of what every backend allows, and it reports as absent the
// things it could physically do but no GPU can. See specs/000-decisions.md
// decision 3 and specs/006-backends.md section 5.
package cpu

import (
	"crypto/sha256"
	"fmt"
	"sync"

	"golang.design/x/accel/internal/driver"
)

// Adapter is the CPU backend's single enumerated adapter. It is always present
// on every platform.
type Adapter struct{}

// Info reports the adapter's default profile, which is what opening it with no
// options produces.
func (Adapter) Info() driver.Info {
	info, _, err := resolve(Options{})
	if err != nil {
		// The zero Options are always valid; a failure here is a programming
		// error in this package rather than anything a caller can cause.
		panic(fmt.Sprintf("accel/internal/cpu: default profile is invalid: %v", err))
	}
	return info
}

// Token is a stable identity for the CPU adapter. There is exactly one, so it is
// a constant rather than derived from anything the host might change between
// enumerations.
func (Adapter) Token() [16]byte {
	var t [16]byte
	sum := sha256.Sum256([]byte("golang.design/x/accel:cpu:0"))
	copy(t[:], sum[:])
	return t
}

// Open opens the CPU backend. opts is a *[Options] or nil, which means the zero
// Options: developer mode at the default emulated subgroup size.
func (Adapter) Open(opts any) (driver.Device, error) {
	var o Options
	switch v := opts.(type) {
	case nil:
	case *Options:
		if v != nil {
			o = *v
		}
	case Options:
		o = v
	default:
		return nil, fmt.Errorf("accel: the CPU backend cannot open with %T", opts)
	}

	info, subgroupSize, err := resolve(o)
	if err != nil {
		return nil, err
	}
	return &device{
		info:         info,
		subgroupSize: subgroupSize,
		shuffleSeed:  o.ShuffleSeed,
		diagnostics:  o.Mode != Strict,
		loseAt:       o.LoseAtSubmission,
	}, nil
}

// device is an opened CPU device. Its memory is the Go heap.
type device struct {
	info         driver.Info
	subgroupSize int
	shuffleSeed  uint64

	// diagnostics is whether cooperative kernels are instrumented, which is what
	// developer mode means: the checks are what make this backend an oracle
	// rather than an executor, so they are on unless a caller asks for the
	// speed. See specs/006-backends.md section 5.
	diagnostics bool

	// loseAt is the submission number at which this device reports itself lost,
	// or zero. See [Options].LoseAtSubmission.
	loseAt int

	mu          sync.Mutex
	blocks      int // live allocations, so Close can refuse to strand them
	closed      bool
	submissions int
	lost        error
}

func (d *device) Info() driver.Info { return d.info }

// SubgroupSize reports the emulated lane count. It is not part of the driver
// contract: the kernel executor reads it directly once it exists (M2 onward).
func (d *device) SubgroupSize() int { return d.subgroupSize }

// subgroupSizeU32 is the emulated lane count as the kernel runtime wants it.
func (d *device) subgroupSizeU32() uint32 {
	if d.subgroupSize < 0 {
		return 0
	}
	return uint32(d.subgroupSize)
}

// ShuffleSeed reports the seed for shuffled invocation advancement, or zero when
// advancement is not shuffled.
func (d *device) ShuffleSeed() uint64 { return d.shuffleSeed }

// Supports reports whether a memory kind is backed. Every kind is real on the
// CPU backend: host memory is device memory, so MemoryShared is genuine rather
// than an alias, and spec 006's memory-kind table reports `yes` in all four
// rows.
func (d *device) Supports(kind driver.MemoryKind) bool {
	switch kind {
	case driver.MemoryDevice, driver.MemoryUpload, driver.MemoryReadback, driver.MemoryShared:
		return true
	}
	return false
}

func (d *device) Alloc(kind driver.MemoryKind, bytes int, label string) (driver.Block, error) {
	if bytes <= 0 {
		return nil, fmt.Errorf("accel: allocation %q: %d bytes is not a positive size", label, bytes)
	}
	if !d.Supports(kind) {
		return nil, fmt.Errorf("accel: allocation %q: unknown memory kind %d", label, int(kind))
	}

	d.mu.Lock()
	if d.lost != nil {
		d.mu.Unlock()
		return nil, fmt.Errorf("accel: allocation %q: %w", label, d.lost)
	}
	if d.closed {
		d.mu.Unlock()
		return nil, fmt.Errorf("accel: allocation %q: the device is closed", label)
	}
	d.blocks++
	d.mu.Unlock()

	return &block{
		dev:  d,
		mem:  make([]byte, bytes),
		kind: kind,
		// MemoryDevice is not mappable even though the memory physically could
		// be. Without one backend enforcing it, nothing does until a discrete GPU
		// does, and by then the mapping is load-bearing. See spec 006 section 1.
		hostVisible: kind != driver.MemoryDevice,
		label:       label,
	}, nil
}

// Lost reports terminal device loss. It is sticky by construction: nothing
// clears d.lost once set.
func (d *device) Lost() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.lost
}

// beginSubmission counts an accepted submission and reports the device lost
// once the injected count is reached.
//
// Accepted: an executable calls it after every refusal of its own, so a submit
// turned away for a closed executable or an unbound slot does not advance the
// count. The count is taken before the check so that LoseAtSubmission of one
// loses the first submission rather than the second, which is what a test
// asking for "lose immediately" means.
func (d *device) beginSubmission() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.lost != nil {
		return d.lost
	}
	d.submissions++
	if d.loseAt > 0 && d.submissions >= d.loseAt {
		d.lost = driver.ErrDeviceLost
		return d.lost
	}
	return nil
}

func (d *device) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.closed {
		return nil
	}
	if d.blocks > 0 {
		return fmt.Errorf("accel: close CPU device: %d allocations are still live", d.blocks)
	}
	d.closed = true
	return nil
}

// block is one CPU device allocation: a Go slice, plus the rule about who may
// look at it directly.
type block struct {
	dev         *device
	mem         []byte
	kind        driver.MemoryKind
	hostVisible bool
	label       string

	mu    sync.Mutex
	freed bool
}

// Bytes returns the host mapping, or nil for memory the CPU backend reports
// non-mappable. See the Alloc comment for why the CPU backend refuses a mapping
// it could physically provide.
func (b *block) Bytes() []byte {
	if !b.hostVisible {
		return nil
	}
	return b.contents()
}

func (b *block) Size() int { return len(b.contents()) }

// contents is the allocation, or nil once freed.
//
// Every read of mem goes through here, under the mutex Free writes it under.
// The slice header is what the lock protects: Free sets it to nil, and a Size
// or a resolve racing that read would see half of a two-word write. The bytes
// the header names are the caller's to order, the way any mapping's are.
func (b *block) contents() []byte {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.mem
}

func (b *block) Write(off int, src []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.checkRange("write", off, len(src)); err != nil {
		return err
	}
	copy(b.mem[off:], src)
	return nil
}

func (b *block) Read(off int, dst []byte) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.checkRange("read", off, len(dst)); err != nil {
		return err
	}
	copy(dst, b.mem[off:])
	return nil
}

func (b *block) checkRange(op string, off, n int) error {
	if b.freed {
		return fmt.Errorf("accel: %s %q: the allocation is freed", op, b.label)
	}
	if off < 0 || n < 0 || off+n > len(b.mem) {
		return fmt.Errorf("accel: %s %q: range [%d, %d) is outside the %d-byte allocation",
			op, b.label, off, off+n, len(b.mem))
	}
	return nil
}

func (b *block) Free() {
	b.mu.Lock()
	if b.freed {
		b.mu.Unlock()
		return
	}
	b.freed = true
	b.mem = nil
	b.mu.Unlock()

	b.dev.mu.Lock()
	b.dev.blocks--
	b.dev.mu.Unlock()
}
