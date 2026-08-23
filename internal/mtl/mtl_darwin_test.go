// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl_test

import (
	"strings"
	"testing"
	"unsafe"

	"golang.design/x/accel/internal/mtl"
)

func open(t *testing.T) *mtl.Device {
	t.Helper()
	devs, err := mtl.Devices()
	if err != nil {
		t.Fatalf("enumerate: %v", err)
	}
	if len(devs) == 0 {
		t.Fatal("no Metal device on a darwin build: this suite promises one")
	}
	for _, d := range devs[1:] {
		d.Close()
	}
	t.Cleanup(devs[0].Close)
	return devs[0]
}

func f32s(b []byte) []float32 {
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

func u32s(b []byte) []uint32 {
	return unsafe.Slice((*uint32)(unsafe.Pointer(&b[0])), len(b)/4)
}

// Runtime compilation is real: bad source is rejected, and the device
// compiler's own message reaches the caller.
//
// The negative half is the whole test. A test that only compiles valid source
// cannot distinguish a working compile path from one that ignores its NSError
// out-parameter and returns a nil library as success, and that is the same
// reinstatement discipline M3 and M4 used, applied to a toolchain rather than
// to a bug.
func TestCompileReportsWhatTheDeviceCompilerSaid(t *testing.T) {
	d := open(t)

	p, err := d.Compile(`
#include <metal_stdlib>
using namespace metal;
kernel void ok(device float *out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  out[gid] = 1.0f;
}`, "ok")
	if err != nil {
		t.Fatalf("valid MSL should compile: %v", err)
	}
	p.Close()

	_, err = d.Compile(`kernel void bad(device float *out) { this is not MSL }`, "bad")
	if err == nil {
		t.Fatal("malformed MSL compiled, so the error out-parameter is not being read")
	}
	// Metal's diagnostics name the offending construct. If this ever stops
	// holding, the assertion to keep is that *something* specific survives:
	// an error that says only "compile failed" would pass a caller nothing.
	if !strings.Contains(err.Error(), "program_source") && !strings.Contains(err.Error(), "error") {
		t.Errorf("the compiler's own diagnostic should reach the caller, got %v", err)
	}

	_, err = d.Compile(`
#include <metal_stdlib>
using namespace metal;
kernel void present(device float *out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  out[gid] = 1.0f;
}`, "absent")
	if err == nil {
		t.Fatal("a missing entry point should be an error, not a nil function dispatched later")
	}
	if !strings.Contains(err.Error(), "absent") {
		t.Errorf("the error should name the entry point that is missing, got %v", err)
	}
}

// The grid the device runs is the grid that was asked for, in all three
// dimensions.
//
// This is the test for MTLSize being passed by value. objc_msgSend is variadic
// in C and is not variadic in the ABI, so passing a pointer where the real
// signature takes three 64-bit integers compiles and runs and dispatches
// something else. A two-dimensional grid with unequal extents is the shape that
// catches it: a swapped width and height still covers the right number of
// threads, so a test at 4x4 would pass while the mapping was transposed.
func TestDispatchLaunchesTheGridItWasGiven(t *testing.T) {
	d := open(t)
	p, err := d.Compile(`
#include <metal_stdlib>
using namespace metal;
kernel void ids(device uint *out [[buffer(0)]],
                uint2 gid [[thread_position_in_grid]],
                uint2 size [[threads_per_grid]]) {
  out[gid.y * size.x + gid.x] = gid.y * 100 + gid.x;
}`, "ids")
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer p.Close()

	const gx, gy, tx, ty = 3, 2, 4, 2 // 12 wide, 4 tall
	const w, h = gx * tx, gy * ty
	buf, err := d.NewBuffer(w*h*4, mtl.StorageShared)
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer buf.Close()

	q := d.NewQueue()
	defer q.Close()
	cb := q.Begin()
	defer cb.Close()
	e := cb.Compute()
	e.SetPipeline(p)
	e.SetBuffer(buf, 0, 0)
	e.Dispatch(mtl.Size{Width: gx, Height: gy, Depth: 1}, mtl.Size{Width: tx, Height: ty, Depth: 1})
	e.End()
	cb.Commit()
	cb.Wait()
	if err := cb.Err(); err != nil {
		t.Fatalf("submission: %v", err)
	}

	got := u32s(buf.Bytes())
	for y := range h {
		for x := range w {
			if want := uint32(y*100 + x); got[y*w+x] != want {
				t.Fatalf("thread (%d,%d) reported %d, want %d: the launched grid is not "+
					"the one that was encoded", x, y, got[y*w+x], want)
			}
		}
	}
}

// Storage modes decide host visibility, and private memory round-trips through
// a blit at an offset that is not zero.
//
// The offset is the point. A copy that ignores its offsets passes every test
// that starts at zero, and every real transfer into a suballocated pool starts
// somewhere else.
func TestStorageModesAndTheBlitPath(t *testing.T) {
	d := open(t)

	shared, err := d.NewBuffer(64, mtl.StorageShared)
	if err != nil {
		t.Fatalf("shared: %v", err)
	}
	defer shared.Close()
	if shared.Bytes() == nil {
		t.Fatal("shared storage must be host-visible")
	}

	private, err := d.NewBuffer(64, mtl.StoragePrivate)
	if err != nil {
		t.Fatalf("private: %v", err)
	}
	defer private.Close()
	// Not a restatement of the mode: on Apple silicon -contents returns a
	// valid pointer for private storage, contradicting what Metal documents.
	// So this fails if the implementation ever goes back to asking the object,
	// which is the mistake that was actually made and reverted here.
	if private.Bytes() != nil {
		t.Fatal("private storage must report no host mapping even where the device " +
			"would happily give one: specs/006-backends.md section 1 makes Bytes the " +
			"authority on mappability, and unified memory is exactly the excuse that " +
			"would erode it")
	}

	src := f32s(shared.Bytes())
	for i := range src {
		src[i] = float32(i) + 0.5
	}

	q := d.NewQueue()
	defer q.Close()

	// shared[16:48] -> private[8:40] -> shared[0:32]
	cb := q.Begin()
	b := cb.Blit()
	b.Copy(private, 8, shared, 16, 32)
	b.End()
	cb.Commit()
	cb.Wait()
	cb.Close()

	cb = q.Begin()
	b = cb.Blit()
	b.Copy(shared, 0, private, 8, 32)
	b.End()
	cb.Commit()
	cb.Wait()
	if err := cb.Err(); err != nil {
		t.Fatalf("blit: %v", err)
	}
	cb.Close()

	got := f32s(shared.Bytes())
	for i := range 8 {
		if want := float32(4+i) + 0.5; got[i] != want {
			t.Fatalf("element %d round-tripped as %v, want %v: an offset was dropped",
				i, got[i], want)
		}
	}
}
