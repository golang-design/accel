// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl_test

import (
	"os"
	"strings"
	"testing"
	"unsafe"

	"golang.design/x/accel/internal/kernelc/emit"
	"golang.design/x/accel/internal/mtl"
)

func open(t *testing.T) *mtl.Device {
	t.Helper()
	devs := requireDevice(t)
	for _, d := range devs[1:] {
		d.Close()
	}
	t.Cleanup(devs[0].Close)
	return devs[0]
}

// requireDevice returns the Metal devices, or ends the test.
//
// Whether "no device" is a failure or a skip depends on what the *job* promised,
// which is specs/006-backends.md section 7 and the rule .github/workflows/ci.yml
// states in its own header: "a job that promises a backend and finds no device
// is a failure, not a skip, so it cannot share a matrix with the jobs that
// promise only the CPU."
//
// Tier 1 runs plain `go test ./...` on three platforms and promises only the CPU
// backend, so a hosted macOS runner without a usable GPU must not turn it red.
// Tier 2 promises Metal, and sets ACCEL_REQUIRE_METAL so that the same tests
// fail instead. A developer on a Mac gets the failure too, because a device is
// there and a skip would hide a backend that stopped enumerating.
func requireDevice(t *testing.T) []*mtl.Device {
	t.Helper()
	devs, err := mtl.Devices()
	if err == nil && len(devs) > 0 {
		return devs
	}
	if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
		t.Fatalf("this job promises Metal and found no device (err=%v)", err)
	}
	t.Skipf("no Metal device on this machine (err=%v); set ACCEL_REQUIRE_METAL to "+
		"make that a failure", err)
	return nil
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

// Contraction is off because the emitter's pragma turns it off, and it is on
// without it.
//
// Both halves are asserted, and the second is the one worth having. It records
// what this device does by default -- a*b+c fuses -- so a later reader can see
// that the pragma is load-bearing rather than decorative, and so that a Metal
// release which changed the default would show up here rather than as a
// one-bit disagreement inside some kernel's output.
//
// This test also disproved the obvious implementation. MTLCompileOptions with
// MTLMathMode.safe looks like the control for this and is not: safe math
// disables reassociation and denormal flushing and leaves the multiply-add free
// to fuse. Measured, not assumed.
//
// The inputs are the ones specs/008-numerics.md uses for its contraction probe:
// with x = 1 + 2^-12, the product x*x is 1 + 2^-11 + 2^-24, which f32 cannot
// hold and rounds to 1 + 2^-11. Subtracting one leaves exactly 2^-11 when the
// product was rounded, and 2^-11 + 2^-24 when it was not. The two answers
// differ, which is the whole reason this input was chosen: a test at ordinary
// values would agree either way.
//
// specs/022-msl-target.md owns the full probe set and the recorded profile.
// This is the one probe that belongs with the option it checks.
func TestTheEmittedPragmaDisablesContraction(t *testing.T) {
	d := open(t)
	const body = `
kernel void contract(const device float *in [[buffer(0)]],
                     device float *out [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  float x = in[0];
  float c = in[1];
  out[0] = x * x + c;
}`
	const head = "#include <metal_stdlib>\nusing namespace metal;\n"

	in, err := d.NewBuffer(8, mtl.StorageShared)
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer in.Close()
	out, err := d.NewBuffer(4, mtl.StorageShared)
	if err != nil {
		t.Fatalf("buffer: %v", err)
	}
	defer out.Close()

	const x = float32(1) + 1.0/4096 // 1 + 2^-12
	src := f32s(in.Bytes())
	src[0], src[1] = x, -1

	q := d.NewQueue()
	defer q.Close()

	run := func(t *testing.T, source string) float32 {
		t.Helper()
		p, err := d.Compile(source, "contract")
		if err != nil {
			t.Fatalf("compile: %v", err)
		}
		defer p.Close()
		f32s(out.Bytes())[0] = 0
		cb := q.Begin()
		defer cb.Close()
		e := cb.Compute()
		e.SetPipeline(p)
		e.SetBuffer(in, 0, 0)
		e.SetBuffer(out, 0, 1)
		e.Dispatch(mtl.Size{Width: 1, Height: 1, Depth: 1}, mtl.Size{Width: 1, Height: 1, Depth: 1})
		e.End()
		cb.Commit()
		cb.Wait()
		if err := cb.Err(); err != nil {
			t.Fatalf("submission: %v", err)
		}
		return f32s(out.Bytes())[0]
	}

	rounded := x*x - 1 // what a separately rounded product gives
	fused := float32(float64(x)*float64(x) - 1)
	if rounded == fused {
		t.Fatal("the chosen inputs no longer distinguish a fused multiply-add from a " +
			"rounded one, so this test would pass either way")
	}

	if got := run(t, head+emit.MSLContractOff+"\n"+body); got != rounded {
		t.Errorf("with the pragma, x*x+c produced %v, want the separately rounded %v: "+
			"the emitter's contraction control does not reach the device", got, rounded)
	}
	if got := run(t, head+body); got != fused {
		t.Logf("without the pragma this device produced %v rather than the fused %v, so "+
			"Metal's default no longer contracts; the pragma is now belt and braces "+
			"rather than load-bearing, which is worth knowing but is not a failure",
			got, fused)
	}
}

// An allocation the device cannot make is an error, not a nil buffer.
//
// Both directions are checked because they fail differently. A zero or negative
// size is refused here without asking Metal, since -newBufferWithLength: with
// zero raises rather than returning nil; a size above the device's own limit is
// refused by the device, and the point is that the nil it returns becomes an
// error rather than a buffer nobody notices is unusable.
func TestBadAllocationsAreErrors(t *testing.T) {
	d := open(t)
	for _, n := range []int{0, -1} {
		if _, err := d.NewBuffer(n, mtl.StorageShared); err == nil {
			t.Errorf("a buffer of %d bytes was allocated", n)
		}
	}
	if _, err := d.NewBuffer(d.MaxBufferBytes*2, mtl.StorageShared); err == nil {
		t.Error("an allocation above the device's own maximum buffer length succeeded")
	}
}

// The SIMD width is reported by compiling something, and every pipeline this
// device makes agrees with it.
//
// MTLDevice has no query for the width, so the only source is a compiled
// pipeline's threadExecutionWidth. This checks the cached device-level answer
// against a freshly compiled pipeline, which is what would diverge if the cache
// were filled from something other than the device.
func TestSubgroupWidthMatchesAPipeline(t *testing.T) {
	d := open(t)
	width := d.SubgroupSize()
	if width <= 0 {
		t.Fatalf("SubgroupSize is %d, which is not a width", width)
	}
	p, err := d.Compile(`
#include <metal_stdlib>
using namespace metal;
kernel void w(device float *out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  out[gid] = 1.0f;
}`, "w")
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer p.Close()
	if p.Name() != "w" {
		t.Errorf("pipeline name is %q, want %q", p.Name(), "w")
	}
	if p.ThreadExecutionWidth != width {
		t.Errorf("this pipeline executes %d wide and the device reports %d",
			p.ThreadExecutionWidth, width)
	}
	// Twice, to exercise the cached path rather than only the first call.
	if again := d.SubgroupSize(); again != width {
		t.Errorf("SubgroupSize returned %d then %d", width, again)
	}
}

// Binding no bytes is a no-op rather than a crash.
//
// SetBytes with an empty slice would otherwise take the address of element
// zero of an empty slice, which is what a kernel with no bindings would
// produce.
func TestSetBytesTolerlatesEmptyData(t *testing.T) {
	d := open(t)
	p, err := d.Compile(`
#include <metal_stdlib>
using namespace metal;
kernel void nop(device float *out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  out[gid] = 7.0f;
}`, "nop")
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer p.Close()
	buf, err := d.NewBuffer(4, mtl.StorageShared)
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
	e.SetBytes(nil, 1)
	e.Dispatch(mtl.Size{Width: 1, Height: 1, Depth: 1}, mtl.Size{Width: 1, Height: 1, Depth: 1})
	e.End()
	cb.Commit()
	cb.Wait()
	if err := cb.Err(); err != nil {
		t.Fatalf("submission: %v", err)
	}
	if got := f32s(buf.Bytes())[0]; got != 7 {
		t.Errorf("the dispatch produced %v, want 7", got)
	}
	if !cb.Done() {
		t.Error("a command buffer that has been waited on must report done")
	}
}
