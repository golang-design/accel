// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl

import (
	"fmt"
	"sync/atomic"
	"unsafe"

	"github.com/ebitengine/purego/objc"
)

// Pipeline is a compiled compute pipeline state, retained, together with the
// library and function it came from.
//
// The three are held together because releasing a library while a pipeline
// built from it is still in use is exactly the class of lifetime bug this
// backend is most likely to have, and holding them means the question never
// arises.
type Pipeline struct {
	lib  objc.ID
	fn   objc.ID
	id   objc.ID
	name string

	// MaxTotalThreadsPerThreadgroup is the pipeline's own ceiling, which can be
	// lower than the device's when the kernel uses many registers. It is the
	// number a dispatch must respect, so it is read here rather than assumed
	// from the device.
	MaxTotalThreadsPerThreadgroup int

	// ThreadExecutionWidth is the SIMD width this pipeline actually executes at,
	// which is the device's subgroup size as the kernel sees it.
	ThreadExecutionWidth int
}

var (
	selNewLibraryWithSource = objc.RegisterName("newLibraryWithSource:options:error:")
	selAlloc                = objc.RegisterName("alloc")
	selInit                 = objc.RegisterName("init")
	selRespondsToSelector   = objc.RegisterName("respondsToSelector:")
	selSetMathMode          = objc.RegisterName("setMathMode:")
	selSetFastMathEnabled   = objc.RegisterName("setFastMathEnabled:")
	selNewFunctionWithName  = objc.RegisterName("newFunctionWithName:")
	selNewComputePipeline   = objc.RegisterName("newComputePipelineStateWithFunction:error:")
	selMaxTotalThreads      = objc.RegisterName("maxTotalThreadsPerThreadgroup")
	selThreadExecutionWidth = objc.RegisterName("threadExecutionWidth")
)

// mathModeSafe is MTLMathModeSafe.
//
// It disables the reassociation and denormal-flushing half of fast math and
// does *not* disable contraction: a multiply-add still fuses under it, measured
// on an M2 rather than assumed. Contraction is controlled by a pragma the
// emitter puts in every kernel, emit.MSLContractOff, which is where the reason
// is written down.
const mathModeSafe = 0

// compileOptions builds MTLCompileOptions asking for safe math, +1 from alloc.
//
// Safe rather than the default, and this is a correctness decision rather than
// a conservative one. Metal's default permits contraction, so a*b+c may become
// fma(a,b,c) and differ from the CPU backend in the last bit -- and
// specs/006-backends.md makes the CPU backend the oracle, so a difference is a
// failure by definition rather than a tolerance to widen. specs/008-numerics.md
// section 6 requires contraction to be controlled, not observed.
//
// The selector moved: newer SDKs deprecate -setFastMathEnabled: in favour of
// -setMathMode:. Asking the object which it answers to is not defensive
// programming here; it is the only way to be right on both, since a selector
// the receiver does not implement raises rather than being ignored.
func compileOptions() objc.ID {
	opts := objc.ID(objc.GetClass("MTLCompileOptions")).Send(selAlloc).Send(selInit)
	if opts == 0 {
		return 0
	}
	if opts.Send(selRespondsToSelector, selSetMathMode) != 0 {
		opts.Send(selSetMathMode, uintptr(mathModeSafe))
		return opts
	}
	opts.Send(selSetFastMathEnabled, uintptr(0))
	return opts
}

// Compile builds a compute pipeline from MSL source.
//
// This is the Metal compiler, running on the device at runtime. The offline
// toolchain (xcrun metal) is not required and is frequently not installed;
// specs/021-metal-bringup.md section 1 argues that this is the stronger check
// anyway, because the compiler that accepts the text is the one that will run
// it.
func (d *Device) Compile(source, entryPoint string) (*Pipeline, error) {
	compiles.Add(1)
	p := &Pipeline{name: entryPoint}
	var err error
	withPool(func() {
		// nsErr must be initialized to nil and must outlive the call: Metal
		// writes it only on failure, so a leftover value would turn a success
		// into a reported error.
		var nsErr objc.ID
		opts := compileOptions()
		defer release(opts)
		p.lib = d.id.Send(selNewLibraryWithSource, nsstring(source), opts, unsafe.Pointer(&nsErr))
		if p.lib == 0 {
			err = describe("compiling MSL", nsErr)
			return
		}
		p.fn = p.lib.Send(selNewFunctionWithName, nsstring(entryPoint))
		if p.fn == 0 {
			err = fmt.Errorf("accel/mtl: the compiled library has no kernel named %q", entryPoint)
			return
		}
		nsErr = 0
		p.id = d.id.Send(selNewComputePipeline, p.fn, unsafe.Pointer(&nsErr))
		if p.id == 0 {
			err = describe("creating a compute pipeline for "+entryPoint, nsErr)
			return
		}
		p.MaxTotalThreadsPerThreadgroup = int(p.id.Send(selMaxTotalThreads))
		p.ThreadExecutionWidth = int(p.id.Send(selThreadExecutionWidth))
	})
	if err != nil {
		p.Close()
		return nil, err
	}
	return p, nil
}

// compiles counts every call into the device compiler, for a test that checks
// a cached result is not compiled again.
var compiles atomic.Int64

// CompileCount reports how many compute pipelines have been compiled.
func CompileCount() int64 { return compiles.Load() }

// subgroupProbe is the smallest kernel that makes a pipeline exist.
//
// The SIMD width is a property of a compiled pipeline, not of a device:
// MTLDevice has no query for it, and -threadExecutionWidth lives on
// MTLComputePipelineState. So the only honest way to report a width is to
// compile something and ask. This kernel is that something, and it is
// deliberately trivial so that nothing about its body can influence the answer.
const subgroupProbe = `#include <metal_stdlib>
using namespace metal;
kernel void _accel_width(device uint *out [[buffer(0)]],
                         uint gid [[thread_position_in_grid]]) {
  out[gid] = gid;
}`

// SubgroupSize reports this device's SIMD width, compiled once.
//
// Zero if the probe fails, which a caller must treat as unknown rather than as
// a width: specs/001-device-resources.md section 1.1 forbids an opened device
// reporting a zero limit, so a zero here has to become a refusal to open rather
// than a number nobody can use.
func (d *Device) SubgroupSize() int {
	d.widthOnce.Do(func() {
		p, err := d.Compile(subgroupProbe, "_accel_width")
		if err != nil {
			return
		}
		defer p.Close()
		d.width = p.ThreadExecutionWidth
	})
	return d.width
}

// Name is the entry point this pipeline runs.
func (p *Pipeline) Name() string { return p.name }

// Close releases the pipeline, its function, and its library.
func (p *Pipeline) Close() {
	release(p.id)
	release(p.fn)
	release(p.lib)
	p.id, p.fn, p.lib = 0, 0, 0
}

// CompileFunction compiles MSL and resolves one named function, without
// building a pipeline state around it.
//
// A graphics stage cannot go through Compile: a vertex or fragment function is
// not a compute kernel, and -newComputePipelineStateWithFunction: rejects one.
// A render pipeline state would need attachment formats and both stages
// together, which is more than "does this text compile" needs to know.
//
// What this proves is what specs/021-metal-bringup.md section 1 argues is worth
// proving: -newLibraryWithSource: *is* the Metal compiler, so text it accepts
// here is text it accepts in production. Resolving the function on top of that
// catches the case a bare compile misses -- source that parses but declares the
// entry point under another name, or not at all.
func (d *Device) CompileFunction(source, name string) (*Function, error) {
	f := &Function{name: name}
	var err error
	withPool(func() {
		var nsErr objc.ID
		opts := compileOptions()
		defer release(opts)
		f.lib = d.id.Send(selNewLibraryWithSource, nsstring(source), opts, unsafe.Pointer(&nsErr))
		if f.lib == 0 {
			err = describe("compiling MSL", nsErr)
			return
		}
		f.fn = f.lib.Send(selNewFunctionWithName, nsstring(name))
		if f.fn == 0 {
			err = fmt.Errorf("accel/mtl: the compiled library has no function named %q", name)
		}
	})
	if err != nil {
		f.Close()
		return nil, err
	}
	return f, nil
}

// Function is a compiled MSL function that is not a compute pipeline.
type Function struct {
	name string
	lib  objc.ID
	fn   objc.ID
}

// Name is the entry point this function was resolved by.
func (f *Function) Name() string { return f.name }

// Close releases the function and its library.
func (f *Function) Close() {
	withPool(func() {
		release(f.fn)
		release(f.lib)
	})
	f.fn, f.lib = 0, 0
}
