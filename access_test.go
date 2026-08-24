// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	"golang.design/x/accel"
)

// A caller writes straight into device memory, so the converted copy never
// exists.
//
// # The allocation this removes
//
// A model loader reads a shard, converts it, and uploads it — and without this
// it holds all three at once. On a multi-gigabyte checkpoint the converted host
// tensor is the largest transient allocation in the process, and it exists only
// because the bytes have nowhere to go but a slice of the caller's own.
//
// The consumer who reported this asked for the opposite: a buffer *over* memory
// they already own. That needs a promise about a lifetime accel cannot see —
// the caller guaranteeing their memory outlives the buffer — and a broken
// promise there is a use-after-free whose symptom is a plausible tensor. This
// direction needs no promise: accel owns the memory and lends it for the
// duration of a call.
func TestAccessLetsACallerConvertIntoDeviceMemory(t *testing.T) {
	const n = 512
	d := openDevice(t)

	pool, err := d.NewPool(accel.PoolDescriptor{
		Kind: accel.MemoryShared, Bytes: n * 4 * 2,
		Label: "weights",
	})
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	defer pool.Close()

	buf, err := pool.AllocBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: n,
		Usage: accel.BufferStorage | accel.BufferCopySrc | accel.BufferCopyDst,
		Label: "w",
	})
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	defer buf.Close()

	// The loader: it never materializes the converted tensor. Each value is
	// computed and written where the device will read it.
	err = buf.Access(func(host []byte) error {
		if len(host) != n*4 {
			return errors.New("the mapping is not the buffer's own bytes")
		}
		for i := range n {
			v := math.Float32bits(float32(i) * 0.5)
			host[i*4+0] = byte(v)
			host[i*4+1] = byte(v >> 8)
			host[i*4+2] = byte(v >> 16)
			host[i*4+3] = byte(v >> 24)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("access: %v", err)
	}

	// And the device sees it, with no upload in between.
	got := make([]float32, n)
	if err := d.Queue().ReadBuffer(buf, 0, got); err != nil {
		t.Fatalf("readback: %v", err)
	}
	for i := range got {
		if want := float32(i) * 0.5; got[i] != want {
			t.Fatalf("element %d is %v, want %v; a write through the mapping must be "+
				"visible to the device without a further call", i, got[i], want)
		}
	}
}

// An error from the caller's function reaches the caller.
func TestAccessReportsWhatTheCallerReturned(t *testing.T) {
	d := openDevice(t)
	pool, err := d.NewPool(accel.PoolDescriptor{
		Kind: accel.MemoryShared, Bytes: 4096,
		Label: "p",
	})
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	defer pool.Close()
	buf, err := pool.AllocBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: 16,
		Usage: accel.BufferStorage | accel.BufferCopyDst, Label: "b",
	})
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	defer buf.Close()

	sentinel := errors.New("the shard was truncated")
	if err := buf.Access(func([]byte) error { return sentinel }); !errors.Is(err, sentinel) {
		t.Errorf("Access returned %v, want the caller's own error", err)
	}
	if err := buf.Access(nil); err == nil {
		t.Error("Access accepted no function; the mapping is valid only inside the call, " +
			"so there is nothing to return instead")
	}
}

// Memory with no host mapping refuses, and the refusal says what to do.
//
// Reported rather than discovered, and refused on the CPU backend too — where
// the memory physically could be mapped — because a rule only one backend
// enforces is a rule that fails in production.
func TestAccessOnDeviceMemoryRefusesWithAdvice(t *testing.T) {
	d := openDevice(t)
	pool, err := d.NewPool(accel.PoolDescriptor{
		Kind: accel.MemoryDevice, Bytes: 4096,
		Label: "private",
	})
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	defer pool.Close()
	buf, err := pool.AllocBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: 16,
		Usage: accel.BufferStorage | accel.BufferCopyDst, Label: "b",
	})
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	defer buf.Close()

	err = buf.Access(func([]byte) error { return nil })
	if err == nil {
		t.Fatal("device-private memory handed out a host mapping")
	}
	if !errors.Is(err, accel.ErrUsage) {
		t.Errorf("the refusal should be an ErrUsage, got %v", err)
	}
	for _, want := range []string{"WriteBuffer", "Shared pool"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal should name %q as the way forward, got %v", want, err)
		}
	}
}

// A closed buffer hands out nothing.
func TestAccessOnAClosedBuffer(t *testing.T) {
	d := openDevice(t)
	pool, err := d.NewPool(accel.PoolDescriptor{
		Kind: accel.MemoryShared, Bytes: 4096,
		Label: "p",
	})
	if err != nil {
		t.Fatalf("pool: %v", err)
	}
	defer pool.Close()
	buf, err := pool.AllocBuffer(accel.BufferDescriptor{
		DType: accel.F32, Count: 16,
		Usage: accel.BufferStorage | accel.BufferCopyDst, Label: "b",
	})
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	if err := buf.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := buf.Access(func([]byte) error { return nil }); err == nil {
		t.Error("a closed buffer handed out its mapping")
	}
}
