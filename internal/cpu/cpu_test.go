// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package cpu

import (
	"bytes"
	"strings"
	"sync"
	"testing"

	"golang.design/x/accel/internal/driver"
)

func open(t *testing.T, o Options) driver.Device {
	t.Helper()
	d, err := Adapter{}.Open(&o)
	if err != nil {
		t.Fatalf("Open(%+v): %v", o, err)
	}
	return d
}

// TestOpenAcceptsEveryOptionsForm covers the three ways the seam may be handed
// options, and rejects the fourth.
func TestOpenAcceptsEveryOptionsForm(t *testing.T) {
	want := Adapter{}.Info()
	for _, opts := range []any{nil, (*Options)(nil), &Options{}, Options{}} {
		d, err := Adapter{}.Open(opts)
		if err != nil {
			t.Fatalf("Open(%T): %v", opts, err)
		}
		if info := d.Info(); info != want {
			t.Errorf("Open(%T) reported a different profile than the adapter promised", opts)
		}
		d.Close()
	}
	if _, err := (Adapter{}).Open("not options"); err == nil {
		t.Error("Open accepted an unrelated options type")
	}
}

// TestDeviceMappability is the rule that makes this backend an oracle: memory it
// physically could map is reported unmappable when no GPU could map it. See spec
// 006 section 1.
func TestDeviceMappability(t *testing.T) {
	d := open(t, Options{})
	defer d.Close()

	kinds := []struct {
		kind   driver.MemoryKind
		mapped bool
		name   string
	}{
		{driver.MemoryDevice, false, "Device"},
		{driver.MemoryUpload, true, "Upload"},
		{driver.MemoryReadback, true, "Readback"},
		{driver.MemoryShared, true, "Shared"},
	}
	for _, tc := range kinds {
		if !d.Supports(tc.kind) {
			t.Errorf("the CPU backend reports %s absent; spec 006's memory-kind table is yes in all four rows", tc.name)
			continue
		}
		b, err := d.Alloc(tc.kind, 64, "probe")
		if err != nil {
			t.Fatalf("Alloc(%s): %v", tc.name, err)
		}
		if got := b.Bytes() != nil; got != tc.mapped {
			t.Errorf("%s: host mapping present = %v, want %v", tc.name, got, tc.mapped)
		}
		if got := b.Size(); got != 64 {
			t.Errorf("%s: Size = %d, want 64", tc.name, got)
		}
		b.Free()
	}
	if d.Supports(driver.MemoryKind(99)) {
		t.Error("an unknown memory kind is reported supported")
	}
}

// TestBlockWriteRead checks the device-side transfer path, which is the only way
// to reach memory Bytes does not map.
func TestBlockWriteRead(t *testing.T) {
	d := open(t, Options{})
	defer d.Close()

	b, err := d.Alloc(driver.MemoryDevice, 16, "weights")
	if err != nil {
		t.Fatal(err)
	}
	defer b.Free()

	src := []byte{1, 2, 3, 4}
	if err := b.Write(4, src); err != nil {
		t.Fatalf("Write: %v", err)
	}
	got := make([]byte, 16)
	if err := b.Read(0, got); err != nil {
		t.Fatalf("Read: %v", err)
	}
	want := []byte{0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0}
	if !bytes.Equal(got, want) {
		t.Errorf("after a partial write the block reads %v, want %v", got, want)
	}

	// A write must copy out of the caller's slice, so mutating it afterwards
	// cannot reach the device.
	src[0] = 99
	if err := b.Read(4, got[:4]); err != nil {
		t.Fatal(err)
	}
	if got[0] != 1 {
		t.Error("the block aliased the caller's slice instead of copying out of it")
	}
}

// Size and Bytes read the block under the same lock Free writes it under.
//
// Free sets mem to nil, and a Size or Bytes racing it read the field without
// the mutex. The race detector is the assertion here: the test drives the two
// from separate goroutines and passes only when every access is ordered.
func TestBlockSizeAndBytesAreOrderedAgainstFree(t *testing.T) {
	d := open(t, Options{})
	defer d.Close()

	for range 64 {
		b, err := d.Alloc(driver.MemoryShared, 64, "racy")
		if err != nil {
			t.Fatal(err)
		}
		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			b.Free()
		}()
		go func() {
			defer wg.Done()
			// Either answer is acceptable; what is not is reading mem while
			// Free writes it.
			_ = b.Size()
			_ = b.Bytes()
		}()
		wg.Wait()
	}
}

// TestBlockRangeChecks proves an out-of-range transfer is reported rather than
// corrupting neighbouring memory.
func TestBlockRangeChecks(t *testing.T) {
	d := open(t, Options{})
	defer d.Close()

	b, err := d.Alloc(driver.MemoryUpload, 8, "staging")
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name string
		err  error
	}{
		{"write past the end", b.Write(6, make([]byte, 4))},
		{"write at a negative offset", b.Write(-1, make([]byte, 1))},
		{"read past the end", b.Read(4, make([]byte, 8))},
	} {
		if tc.err == nil {
			t.Errorf("%s was accepted", tc.name)
		} else if !strings.Contains(tc.err.Error(), "outside") {
			t.Errorf("%s: error %q does not name the allocation's extent", tc.name, tc.err)
		}
	}

	b.Free()
	b.Free() // idempotent
	if err := b.Write(0, []byte{1}); err == nil {
		t.Error("a freed block accepted a write")
	}
	if err := b.Read(0, make([]byte, 1)); err == nil {
		t.Error("a freed block accepted a read")
	}
	if b.Bytes() != nil {
		t.Error("a freed block still hands out a host mapping")
	}
}

// TestAllocRejections covers what a backend must refuse before it touches
// memory.
func TestAllocRejections(t *testing.T) {
	d := open(t, Options{})

	if _, err := d.Alloc(driver.MemoryDevice, 0, "empty"); err == nil {
		t.Error("a zero-byte allocation was accepted")
	}
	if _, err := d.Alloc(driver.MemoryDevice, -1, "negative"); err == nil {
		t.Error("a negative allocation was accepted")
	}
	if _, err := d.Alloc(driver.MemoryKind(99), 16, "bogus"); err == nil {
		t.Error("an unknown memory kind was accepted")
	}

	if err := d.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := d.Close(); err != nil {
		t.Fatalf("Close is idempotent: %v", err)
	}
	if _, err := d.Alloc(driver.MemoryDevice, 16, "after close"); err == nil {
		t.Error("a closed device allocated")
	}
}

// TestCloseRefusesToStrandAllocations is the backend half of the rule that
// closing is ordered rather than recursive: accel guarantees blocks are freed
// first, and the backend refuses rather than trusting it.
func TestCloseRefusesToStrandAllocations(t *testing.T) {
	d := open(t, Options{})

	b, err := d.Alloc(driver.MemoryShared, 32, "live")
	if err != nil {
		t.Fatal(err)
	}
	err = d.Close()
	if err == nil {
		t.Fatal("the device closed with a live allocation")
	}
	if !strings.Contains(err.Error(), "still live") {
		t.Errorf("error %q does not say what is still live", err)
	}

	b.Free()
	if err := d.Close(); err != nil {
		t.Fatalf("Close after freeing: %v", err)
	}
}

// TestSubgroupAndSeedAreCarried checks that the emulation knobs reach the
// device, since nothing else reads them until the kernel executor lands at M2.
func TestSubgroupAndSeedAreCarried(t *testing.T) {
	d := open(t, Options{SubgroupSize: 32, ShuffleSeed: 7})
	defer d.Close()

	type knobs interface {
		SubgroupSize() int
		ShuffleSeed() uint64
	}
	k, ok := d.(knobs)
	if !ok {
		t.Fatal("the CPU device does not expose its emulation knobs")
	}
	if got := k.SubgroupSize(); got != 32 {
		t.Errorf("SubgroupSize = %d, want 32", got)
	}
	if got := k.ShuffleSeed(); got != 7 {
		t.Errorf("ShuffleSeed = %d, want 7", got)
	}
}

// TestBaselinesAreConservative checks the one rule that turns spec 006's
// capability matrix into a baseline profile: only `yes` and `emul` survive.
func TestBaselinesAreConservative(t *testing.T) {
	for backend, caps := range baselines {
		name := backendNames[backend]
		if caps.Subgroups {
			t.Errorf("%s: no target's subgroup support is `yes` in spec 006's matrix", name)
		}
		if caps.AtomicFloatAddStorage || caps.AtomicFloatAddShared {
			t.Errorf("%s: float atomics are cap or no on every target", name)
		}
		if caps.Presentation || caps.Multisampling || caps.NativeGraphReplay {
			t.Errorf("%s: presentation, multisampling and native replay are cap on every target", name)
		}
		if caps.SharedMemoryKind {
			t.Errorf("%s: MemoryShared is cap or no on every target and is never assumed", name)
		}
		if !caps.IndirectDispatch {
			t.Errorf("%s: indirect dispatch is `yes` on every target and must survive", name)
		}
	}
}

// TestPortableFloorIsUsable guards the numbers spec 002 section 1.5 pins, since
// a floor that drifts silently invalidates every strict-mode conclusion.
func TestPortableFloorIsUsable(t *testing.T) {
	f := portableFloor
	for _, tc := range []struct {
		name string
		got  int
		want int
	}{
		{"MaxWorkgroupInvocations", f.MaxWorkgroupInvocations, 128},
		{"MaxWorkgroupSize[0]", f.MaxWorkgroupSize[0], 128},
		{"MaxWorkgroupSize[1]", f.MaxWorkgroupSize[1], 128},
		{"MaxWorkgroupSize[2]", f.MaxWorkgroupSize[2], 64},
		{"MaxWorkgroupCount[0]", f.MaxWorkgroupCount[0], 65535},
		{"MaxSharedMemoryBytes", f.MaxSharedMemoryBytes, 16384},
		{"MinStorageBufferOffsetAlignment", f.MinStorageBufferOffsetAlignment, 256},
		{"MinUniformBufferOffsetAlignment", f.MinUniformBufferOffsetAlignment, 256},
		{"MinBufferCopyOffsetAlignment", f.MinBufferCopyOffsetAlignment, 16},
	} {
		if tc.got != tc.want {
			t.Errorf("portable floor %s = %d, want %d (spec 002 section 1.5, spec 001 section 3.1)",
				tc.name, tc.got, tc.want)
		}
	}
}
