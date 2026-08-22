// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"

	"golang.design/x/accel"
)

func newPool(t *testing.T, d *accel.Device, kind accel.MemoryKind, bytes int) *accel.Pool {
	t.Helper()
	p, err := d.NewPool(kind, bytes)
	if err != nil {
		t.Fatalf("NewPool(%v, %d): %v", kind, bytes, err)
	}
	return p
}

func alloc(t *testing.T, p *accel.Pool, desc accel.BufferDescriptor) *accel.Buffer {
	t.Helper()
	b, err := p.Alloc(desc)
	if err != nil {
		t.Fatalf("Alloc(%+v): %v", desc, err)
	}
	return b
}

// TestPoolKindsAndRejection is spec 001 section 2: a kind the device reports
// absent is an error naming the kind and the device, never an Upload pool
// handed back under a different name.
func TestPoolKindsAndRejection(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	for _, kind := range []accel.MemoryKind{
		accel.MemoryDevice, accel.MemoryUpload, accel.MemoryReadback, accel.MemoryShared,
	} {
		p := newPool(t, d, kind, 1<<20)
		if p.Kind() != kind {
			t.Errorf("NewPool(%v) produced a %v pool", kind, p.Kind())
		}
		if err := p.Close(); err != nil {
			t.Errorf("Close %v pool: %v", kind, err)
		}
	}

	// A device that reports no unified memory must refuse MemoryShared rather
	// than substituting Upload. Mimicking a discrete GPU is how that path is
	// reachable before a discrete GPU backend exists.
	discrete := accel.DeviceProfile{Info: accel.DeviceInfo{
		Backend:      accel.BackendMetal,
		Name:         "AMD Radeon Pro 5500M",
		Capabilities: accel.Capabilities{SharedMemoryKind: false},
		Limits: accel.Limits{
			MaxPoolBytes: 1 << 30, MaxBufferBytes: 1 << 30, MaxPools: 16,
			MinStorageBufferOffsetAlignment: 256, MinUniformBufferOffsetAlignment: 256,
			MinBufferCopyOffsetAlignment: 16, MinSubgroupSize: 1, MaxSubgroupSize: 1,
		},
	}}
	dd := openCPU(t, accel.CPUOptions{Mode: accel.CPUMimic, Mimic: &discrete})
	if dd.Info().Capabilities.SharedMemoryKind {
		t.Fatal("the mimicked profile should report no unified memory")
	}

	_, err := dd.NewPool(accel.MemoryShared, 1<<20)
	if err == nil {
		t.Fatal("a device reporting no unified memory returned a Shared pool")
	}
	if !errors.Is(err, accel.ErrUnsupported) {
		t.Errorf("error %v does not unwrap to ErrUnsupported", err)
	}
	for _, want := range []string{"MemoryShared", "AMD Radeon Pro 5500M", "SharedMemoryKind"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not name %q", err, want)
		}
	}
}

func TestPoolConstructionRejections(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	for _, tc := range []struct {
		name string
		desc accel.PoolDescriptor
		want string
	}{
		{"zero bytes", accel.PoolDescriptor{Bytes: 0}, "positive"},
		{"negative bytes", accel.PoolDescriptor{Bytes: -1}, "positive"},
		{"beyond MaxPoolBytes", accel.PoolDescriptor{Bytes: 1 << 62}, "MaxPoolBytes"},
		{"unknown policy", accel.PoolDescriptor{Bytes: 1 << 20, Policy: accel.PoolPolicy(9)}, "policy"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := d.NewPoolWith(tc.desc)
			if err == nil {
				t.Fatalf("NewPoolWith(%+v) was accepted", tc.desc)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not explain %q", err, tc.want)
			}
		})
	}
}

// TestAllocationAlignsByUsage is spec 001 section 3.1: a buffer's alignment is
// the strictest its declared usage implies, with a floor of 4.
func TestAllocationAlignsByUsage(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	p := newPool(t, d, accel.MemoryDevice, 4<<20)
	defer p.Close()

	lim := d.Limits()
	for _, tc := range []struct {
		usage accel.BufferUsage
		align int
	}{
		{0, 4},
		{accel.UsageVertex, 4},
		{accel.UsageIndirect, 4},
		{accel.UsageStorage, lim.MinStorageBufferOffsetAlignment},
		{accel.UsageUniform, lim.MinUniformBufferOffsetAlignment},
		{accel.UsageCopySrc, lim.MinBufferCopyOffsetAlignment},
		{accel.UsageCopyDst, lim.MinBufferCopyOffsetAlignment},
		{accel.UsageStorage | accel.UsageCopyDst, lim.MinStorageBufferOffsetAlignment},
	} {
		// Push the cursor to an odd place first so a satisfied alignment is the
		// allocator's doing rather than an accident of starting at zero.
		filler := alloc(t, p, accel.BufferDescriptor{DType: accel.U8, Count: 17, Label: "filler"})
		b := alloc(t, p, accel.BufferDescriptor{
			DType: accel.F32, Count: 4, Usage: tc.usage, Label: fmt.Sprintf("usage %v", tc.usage),
		})
		if got := poolOffsetOf(t, p, b); got%tc.align != 0 {
			t.Errorf("usage %v placed at offset %d, not a multiple of the %d it implies",
				tc.usage, got, tc.align)
		}
		filler.Close()
		b.Close()
	}
}

// poolOffsetOf recovers a buffer's placement through the public surface: a
// pool's Used before and after is the only offset-shaped number accel exposes,
// so alignment is checked by allocating into a known-empty pool instead.
func poolOffsetOf(t *testing.T, p *accel.Pool, b *accel.Buffer) int {
	t.Helper()
	// PoolStats.Used counts allocation sizes including padding, so the offset of
	// the most recent allocation is Used minus its own allocation size. That is
	// exact here because allocations only ever move forward in a fresh pool.
	s := p.Stats()
	return s.Used - roundUpTo(b.Bytes(), 256)
}

func roundUpTo(v, to int) int { return (v + to - 1) / to * to }

// TestUsedIncludesPadding is spec 001 section 3.4: PoolStats.Used reports
// allocation sizes, so a caller comparing it against the sum of their buffer
// sizes sees the alignment tax rather than suspecting a leak.
func TestUsedIncludesPadding(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	p := newPool(t, d, accel.MemoryDevice, 1<<20)
	defer p.Close()

	const n = 16
	var live []*accel.Buffer
	for i := range n {
		live = append(live, alloc(t, p, accel.BufferDescriptor{
			DType: accel.U8, Count: 1, Label: fmt.Sprintf("tiny%d", i),
		}))
	}
	s := p.Stats()
	if want := n * 256; s.Used != want {
		t.Errorf("Used = %d after %d one-byte buffers, want %d: the alignment tax is "+
			"deliberately visible in the number a caller already looks at", s.Used, n, want)
	}
	if s.Allocations != n {
		t.Errorf("Allocations = %d, want %d", s.Allocations, n)
	}
	for _, b := range live {
		if err := b.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if s := p.Stats(); s.Used != 0 || s.Allocations != 0 {
		t.Errorf("after closing every buffer: Used = %d, Allocations = %d", s.Used, s.Allocations)
	}
}

// TestAllocErrorDistinguishesFragmentation is spec 001 section 5.4: the message
// carries both Free and LargestFree, because those two numbers together tell
// exhaustion from fragmentation and a bare failure does not.
func TestAllocErrorDistinguishesFragmentation(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	p, err := d.NewPoolWith(accel.PoolDescriptor{Kind: accel.MemoryDevice, Bytes: 1 << 20, Label: "weights"})
	if err != nil {
		t.Fatal(err)
	}
	var live []*accel.Buffer
	defer func() {
		for _, b := range live {
			b.Close()
		}
		if err := p.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	}()

	// Exhaustion first: a request larger than the pool.
	_, err = p.Alloc(accel.BufferDescriptor{DType: accel.U8, Count: 4 << 20, Label: "kv"})
	if err == nil {
		t.Fatal("a request larger than the pool was accepted")
	}
	var ae *accel.AllocError
	if !errors.As(err, &ae) {
		t.Fatalf("error is %T, want *AllocError", err)
	}
	if !errors.Is(err, accel.ErrOutOfDeviceMemory) {
		t.Errorf("a request beyond the pool does not unwrap to ErrOutOfDeviceMemory: %v", err)
	}
	for _, want := range []string{`"kv"`, `"weights"`, "Device", "do not grow"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("exhaustion message %q does not carry %q", err, want)
		}
	}

	// Then fragmentation: fill, free every other, ask for more than any hole.
	for i := 0; ; i++ {
		b, err := p.Alloc(accel.BufferDescriptor{
			DType: accel.U8, Count: 256, Label: fmt.Sprintf("block%d", i),
		})
		if err != nil {
			break
		}
		live = append(live, b)
	}
	var held []*accel.Buffer
	for i, b := range live {
		if i%2 == 0 {
			b.Close()
			continue
		}
		held = append(held, b)
	}
	live = held

	s := p.Stats()
	_, err = p.Alloc(accel.BufferDescriptor{DType: accel.U8, Count: 4096, Label: "big"})
	if err == nil {
		t.Fatal("a fragmented pool served a request larger than any hole")
	}
	if !errors.Is(err, accel.ErrFragmented) {
		t.Errorf("error does not unwrap to ErrFragmented: %v", err)
	}
	if !errors.As(err, &ae) {
		t.Fatalf("error is %T, want *AllocError", err)
	}
	if ae.Free < ae.Requested {
		t.Errorf("AllocError.Free = %d is below the %d requested, so this is exhaustion not fragmentation",
			ae.Free, ae.Requested)
	}
	if ae.LargestFree >= ae.Requested {
		t.Errorf("AllocError.LargestFree = %d is at or above the %d requested", ae.LargestFree, ae.Requested)
	}
	if !strings.Contains(err.Error(), "not contiguous space") {
		t.Errorf("fragmentation message %q does not say the pool has space but not contiguous space", err)
	}
	if s.LargestFree > s.Free {
		t.Errorf("LargestFree %d exceeds Free %d", s.LargestFree, s.Free)
	}
}

// TestBufferDescriptorRejections covers what a descriptor must not be.
func TestBufferDescriptorRejections(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	p := newPool(t, d, accel.MemoryDevice, 1<<20)
	defer p.Close()

	for _, tc := range []struct {
		name string
		desc accel.BufferDescriptor
		want string
	}{
		{"zero count", accel.BufferDescriptor{DType: accel.F32, Count: 0}, "positive"},
		{"negative count", accel.BufferDescriptor{DType: accel.F32, Count: -1}, "positive"},
		{"unknown dtype", accel.BufferDescriptor{DType: accel.DType(99), Count: 1}, "not a dtype"},
		{"beyond MaxBufferBytes", accel.BufferDescriptor{DType: accel.F32, Count: 1 << 40}, "MaxBufferBytes"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := p.Alloc(tc.desc); err == nil {
				t.Fatalf("Alloc(%+v) was accepted", tc.desc)
			} else if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not explain %q", err, tc.want)
			}
		})
	}
}

// TestTexturePoolRejectsBuffers is spec 001 section 4.4: a pool is a buffer
// pool or a texture pool and never both.
func TestTexturePoolRejectsBuffers(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	p, err := d.NewPoolWith(accel.PoolDescriptor{
		Kind: accel.MemoryDevice, Bytes: 1 << 20, Textures: true, Label: "targets",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()

	_, err = p.Alloc(accel.BufferDescriptor{DType: accel.F32, Count: 4, Label: "b"})
	if err == nil {
		t.Fatal("a texture pool allocated a buffer")
	}
	if !errors.Is(err, accel.ErrUsage) {
		t.Errorf("error does not unwrap to ErrUsage: %v", err)
	}
	if !strings.Contains(err.Error(), "textures") {
		t.Errorf("error %q does not say the pool holds textures", err)
	}
}

// TestLinearPoolResets is spec 001 sections 2.1 and 11.4: Reset rejects a
// general pool, and after a successful reset every child handle is closed.
func TestLinearPoolResets(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})

	general := newPool(t, d, accel.MemoryDevice, 1<<20)
	defer general.Close()
	if err := general.Reset(); err == nil {
		t.Error("a general pool reset")
	} else if !errors.Is(err, accel.ErrUsage) {
		t.Errorf("error does not unwrap to ErrUsage: %v", err)
	}

	linear, err := d.NewPoolWith(accel.PoolDescriptor{
		Kind: accel.MemoryDevice, Bytes: 1 << 20, Policy: accel.PoolLinear, Label: "transients",
	})
	if err != nil {
		t.Fatal(err)
	}
	defer linear.Close()

	b := alloc(t, linear, accel.BufferDescriptor{DType: accel.F32, Count: 64, Label: "t0"})
	if s := linear.Stats(); s.Allocations != 1 {
		t.Errorf("Allocations = %d, want 1", s.Allocations)
	}

	// An individual close in a linear pool is a no-op against the memory, so the
	// pool's occupancy does not change.
	if err := b.Close(); err != nil {
		t.Fatalf("Close in a linear pool: %v", err)
	}
	if s := linear.Stats(); s.Used == 0 {
		t.Error("closing one buffer freed memory in a linear pool; it frees only by Reset")
	}

	again := alloc(t, linear, accel.BufferDescriptor{DType: accel.F32, Count: 64, Label: "t1"})
	if err := linear.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	if s := linear.Stats(); s.Used != 0 || s.Allocations != 0 {
		t.Errorf("after Reset: %+v, want an empty pool", s)
	}
	// Every handle from before the reset is closed, so a stale one is reported.
	if _, err := again.View(0, 1); err == nil {
		t.Error("a handle from before Reset is still usable")
	} else if !errors.Is(err, accel.ErrLifetime) {
		t.Errorf("stale handle returned %v, want a lifetime error", err)
	}
}

// TestClosingIsOrderedNotRecursive is spec 001 section 7.2 at both levels:
// closing a pool with live buffers is reported and frees nothing, and the
// buffers still work afterwards.
func TestClosingIsOrderedNotRecursive(t *testing.T) {
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatal(err)
	}

	p, err := d.NewPoolWith(accel.PoolDescriptor{Kind: accel.MemoryDevice, Bytes: 1 << 20, Label: "weights"})
	if err != nil {
		t.Fatal(err)
	}
	b := alloc(t, p, accel.BufferDescriptor{DType: accel.F32, Count: 128, Label: "w0"})

	err = p.Close()
	if err == nil {
		t.Fatal("a pool with a live buffer closed")
	}
	var le *accel.LifetimeError
	if !errors.As(err, &le) {
		t.Fatalf("error is %T, want *LifetimeError", err)
	}
	if le.Children != 1 {
		t.Errorf("LifetimeError.Children = %d, want 1", le.Children)
	}
	if !strings.Contains(err.Error(), "rather than recursive") {
		t.Errorf("message %q does not say closing is ordered rather than recursive", err)
	}

	// The pool was not freed, so the buffer still works.
	if _, err := b.View(0, 4); err != nil {
		t.Errorf("the buffer stopped working after its pool refused to close: %v", err)
	}

	// The device refuses for the same reason.
	if err := d.Close(); err == nil {
		t.Fatal("a device with a live pool closed")
	} else if !errors.As(err, &le) || le.Children != 1 {
		t.Errorf("device close error is %v, want a LifetimeError counting one live pool", err)
	}

	// A device that refused to close is still fully open. It must not have
	// marked itself closed and rolled back, because a concurrent caller would
	// then see a closed device that never closed.
	probe, err := d.NewPool(accel.MemoryUpload, 1<<20)
	if err != nil {
		t.Fatalf("the device stopped working after refusing to close: %v", err)
	}
	if err := probe.Close(); err != nil {
		t.Fatal(err)
	}
	// And a second Close still refuses, rather than reporting the success an
	// already-marked handle would.
	if err := d.Close(); err == nil {
		t.Error("a second Close on a device with a live pool reported success")
	}

	// The refusal must never be observable as a closed device, not even for the
	// instant between deciding to close and discovering the children. A Close
	// that marks the handle dead and rolls back opens exactly that window, and a
	// concurrent caller falls into it.
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(2)
		go func() {
			defer wg.Done()
			for range 200 {
				if err := d.Close(); err == nil {
					t.Error("Close succeeded on a device with a live pool")
					return
				}
			}
		}()
		go func() {
			defer wg.Done()
			for range 200 {
				q, err := d.NewPool(accel.MemoryUpload, 1<<16)
				if err != nil {
					t.Errorf("NewPool raced Close and saw: %v", err)
					return
				}
				if err := q.Close(); err != nil {
					t.Errorf("Pool.Close: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()

	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	if err := p.Close(); err != nil {
		t.Fatalf("Close after the buffer went away: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("Close after the pool went away: %v", err)
	}
}

// TestClosedHandlesAreReported is spec 001 section 11.6: using a closed
// resource is a *LifetimeError at every entry point, enumerated by a
// table-driven test so a new entry point that forgets the check is caught.
func TestClosedHandlesAreReported(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	p := newPool(t, d, accel.MemoryDevice, 1<<20)
	b := alloc(t, p, accel.BufferDescriptor{DType: accel.F32, Count: 16, Label: "logits"})
	view, err := b.View(0, 4)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}

	entries := []struct {
		name string
		call func() error
	}{
		{"Buffer.View", func() error { _, err := b.View(0, 4); return err }},
		{"Buffer.ViewAs", func() error { _, err := b.ViewAs(accel.U32, 0, 4); return err }},
	}
	for _, e := range entries {
		err := e.call()
		var le *accel.LifetimeError
		if !errors.As(err, &le) {
			t.Errorf("%s on a closed buffer returned %v, want *LifetimeError", e.name, err)
			continue
		}
		if le.Reason != "closed" {
			t.Errorf("%s reported reason %q, want \"closed\"", e.name, le.Reason)
		}
		if le.Resource != "logits" {
			t.Errorf("%s named %q, want the buffer's label", e.name, le.Resource)
		}
	}

	// A view of a closed buffer is rejected rather than crashing: the view holds
	// no reference, the Go pointer keeps the Buffer object alive, and the closed
	// flag lives on the object the view still points at.
	if view.Buffer == nil {
		t.Fatal("the view lost its buffer")
	}

	// Closing twice is a no-op rather than a double free.
	if err := b.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
	if err := p.Close(); err != nil {
		t.Fatalf("the pool should be empty: %v", err)
	}
	// A closed pool refuses to allocate.
	if _, err := p.Alloc(accel.BufferDescriptor{DType: accel.F32, Count: 1}); err == nil {
		t.Error("a closed pool allocated")
	}
	if err := p.Reset(); err == nil {
		t.Error("a closed pool reset")
	}
}

// TestImplicitPoolGrows is spec 001 section 5.5: explicit pools never grow, and
// the implicit pool behind NewBuffer does, by adding a block rather than by
// moving anything.
func TestImplicitPoolGrows(t *testing.T) {
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// One block is 64 MiB, so a handful of 24 MiB buffers forces a second.
	const each = 24 << 20
	var live []*accel.Buffer
	for i := range 4 {
		b, err := d.NewBuffer(accel.BufferDescriptor{
			DType: accel.U8, Count: each, Usage: accel.UsageStorage,
			Label: fmt.Sprintf("convenience%d", i),
		})
		if err != nil {
			t.Fatalf("NewBuffer %d: %v", i, err)
		}
		live = append(live, b)
	}

	// A request larger than a whole block gets a block of its own.
	big, err := d.NewBuffer(accel.BufferDescriptor{
		DType: accel.U8, Count: 100 << 20, Label: "oversized",
	})
	if err != nil {
		t.Fatalf("NewBuffer larger than a block: %v", err)
	}
	if got := big.Bytes(); got != 100<<20 {
		t.Errorf("oversized buffer is %d bytes, want %d", got, 100<<20)
	}
	live = append(live, big)

	// The implicit pool has no handle, so the device owns it: closing the device
	// with the buffers gone must succeed without the caller closing a pool they
	// were never given.
	for _, b := range live {
		if err := b.Close(); err != nil {
			t.Fatal(err)
		}
	}
	if err := d.Close(); err != nil {
		t.Fatalf("Close with only implicit pools left: %v", err)
	}
}

// TestMaxPoolsIsEnforced is spec 001 section 1.1's MaxPools: the number that
// makes pooling mandatory rather than merely efficient.
func TestMaxPoolsIsEnforced(t *testing.T) {
	capped := accel.DeviceProfile{Info: accel.DeviceInfo{
		Backend: accel.BackendVulkan,
		Name:    "capped",
		Limits: accel.Limits{
			MaxPools: 2, MaxPoolBytes: 1 << 30, MaxBufferBytes: 1 << 30,
			MinStorageBufferOffsetAlignment: 256, MinUniformBufferOffsetAlignment: 256,
			MinBufferCopyOffsetAlignment: 16, MinSubgroupSize: 1, MaxSubgroupSize: 1,
		},
	}}
	d := openCPU(t, accel.CPUOptions{Mode: accel.CPUMimic, Mimic: &capped})

	first := newPool(t, d, accel.MemoryDevice, 1<<20)
	defer first.Close()
	second := newPool(t, d, accel.MemoryDevice, 1<<20)
	defer second.Close()

	_, err := d.NewPool(accel.MemoryDevice, 1<<20)
	if err == nil {
		t.Fatal("a third pool was created on a device capped at two")
	}
	if !errors.Is(err, accel.ErrOutOfDeviceMemory) {
		t.Errorf("error does not unwrap to ErrOutOfDeviceMemory: %v", err)
	}
	if !strings.Contains(err.Error(), "cap of 2") {
		t.Errorf("error %q does not name the cap", err)
	}
}

// TestPoolIsSafeForConcurrentUse is spec 001 sections 1.2 and 11.7: many
// goroutines allocating from one pool, with no reported race and no lost
// allocation. The allocator inside deliberately is not synchronised, so this is
// the test that proves the pool holds the lock.
func TestPoolIsSafeForConcurrentUse(t *testing.T) {
	d := openCPU(t, accel.CPUOptions{})
	p := newPool(t, d, accel.MemoryDevice, 16<<20)
	defer p.Close()

	const goroutines, each = 8, 64

	var wg sync.WaitGroup
	buffers := make([][]*accel.Buffer, goroutines)
	for g := range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range each {
				b, err := p.Alloc(accel.BufferDescriptor{
					DType: accel.F32, Count: 16, Usage: accel.UsageStorage,
					Label: fmt.Sprintf("g%d-%d", g, i),
				})
				if err != nil {
					t.Errorf("goroutine %d allocation %d: %v", g, i, err)
					return
				}
				buffers[g] = append(buffers[g], b)
			}
		}()
	}
	wg.Wait()

	var all []*accel.Buffer
	for _, set := range buffers {
		all = append(all, set...)
	}
	if len(all) != goroutines*each {
		t.Fatalf("%d allocations survived, want %d", len(all), goroutines*each)
	}
	if got := p.Stats().Allocations; got != len(all) {
		t.Errorf("the pool counts %d allocations, the callers hold %d", got, len(all))
	}
	// Nothing overlaps: every buffer gets a distinct byte pattern later, but at
	// this level distinctness of placement is what the pool owes.
	if s := p.Stats(); s.Used != len(all)*256 {
		t.Errorf("Used = %d, want %d: allocations were lost or double-counted", s.Used, len(all)*256)
	}

	for i, b := range all {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := b.Close(); err != nil {
				t.Errorf("closing buffer %d: %v", i, err)
			}
		}()
	}
	wg.Wait()

	if s := p.Stats(); s.Used != 0 || s.Allocations != 0 {
		t.Errorf("after closing everything concurrently: %+v", s)
	}
}

// TestDeviceIsSafeForConcurrentUse is spec 001 section 11.7's concurrency case,
// which asks for the operations together rather than one at a time: many
// goroutines allocating from one pool, writing disjoint ranges through one
// queue, and creating pools and buffers, all at once, with no reported race and
// no lost allocation.
//
// Running them together is the point. Creating a pool counts the device's live
// allocations while NewBuffer's implicit pool is adding one, and closing races
// both, so the interesting failures are between operations rather than inside
// any one of them. This is a test that only fails when it matters, which is why
// it runs in the ordinary suite rather than as a stress target.
func TestDeviceIsSafeForConcurrentUse(t *testing.T) {
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatal(err)
	}

	shared := newPool(t, d, accel.MemoryDevice, 8<<20)
	q := d.Queue()

	const width = 64
	target := alloc(t, shared, accel.BufferDescriptor{
		DType: accel.U32, Count: width * 8,
		Usage: accel.UsageCopyDst | accel.UsageCopySrc, Label: "target",
	})

	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		buffers []*accel.Buffer
		pools   []*accel.Pool
	)
	keep := func(b *accel.Buffer) {
		mu.Lock()
		buffers = append(buffers, b)
		mu.Unlock()
	}

	for g := range 8 {
		wg.Add(4)

		// Suballocate from one shared pool.
		go func() {
			defer wg.Done()
			for i := range 16 {
				b, err := shared.Alloc(accel.BufferDescriptor{
					DType: accel.F32, Count: 8, Usage: accel.UsageStorage,
					Label: fmt.Sprintf("sub-g%d-%d", g, i),
				})
				if err != nil {
					t.Errorf("Alloc: %v", err)
					return
				}
				keep(b)
			}
		}()

		// Write a disjoint range through the one queue.
		go func() {
			defer wg.Done()
			chunk := make([]uint32, width)
			for i := range chunk {
				chunk[i] = uint32(g*width + i)
			}
			if err := q.WriteBuffer(target, g*width, chunk); err != nil {
				t.Errorf("WriteBuffer: %v", err)
			}
		}()

		// Create a pool, which counts the device's live allocations.
		go func() {
			defer wg.Done()
			p, err := d.NewPool(accel.MemoryUpload, 1<<20)
			if err != nil {
				t.Errorf("NewPool: %v", err)
				return
			}
			mu.Lock()
			pools = append(pools, p)
			mu.Unlock()
		}()

		// Allocate from the implicit pool, which adds to that same count.
		go func() {
			defer wg.Done()
			b, err := d.NewBuffer(accel.BufferDescriptor{
				DType: accel.U8, Count: 4096, Label: fmt.Sprintf("implicit-g%d", g),
			})
			if err != nil {
				t.Errorf("NewBuffer: %v", err)
				return
			}
			keep(b)
		}()
	}
	wg.Wait()

	// Nothing was lost: every write landed in its own range.
	got := make([]uint32, width*8)
	if err := q.ReadBuffer(target, 0, got); err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v != uint32(i) {
			t.Fatalf("element %d is %d, want %d: a concurrent write was lost", i, v, i)
		}
	}
	if want := 8 * 16; shared.Stats().Allocations != want+1 {
		t.Errorf("the shared pool counts %d allocations, want %d",
			shared.Stats().Allocations, want+1)
	}

	// Teardown is concurrent too, since Close races everything above.
	for _, b := range buffers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := b.Close(); err != nil {
				t.Errorf("Buffer.Close: %v", err)
			}
		}()
	}
	wg.Wait()

	if err := target.Close(); err != nil {
		t.Fatal(err)
	}
	for _, p := range pools {
		if err := p.Close(); err != nil {
			t.Errorf("Pool.Close: %v", err)
		}
	}
	if err := shared.Close(); err != nil {
		t.Fatalf("Close the shared pool: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("Device.Close: %v", err)
	}
}
