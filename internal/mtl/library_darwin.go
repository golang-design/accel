// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl

import (
	"fmt"
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
	selNewFunctionWithName  = objc.RegisterName("newFunctionWithName:")
	selNewComputePipeline   = objc.RegisterName("newComputePipelineStateWithFunction:error:")
	selMaxTotalThreads      = objc.RegisterName("maxTotalThreadsPerThreadgroup")
	selThreadExecutionWidth = objc.RegisterName("threadExecutionWidth")
)

// Compile builds a compute pipeline from MSL source.
//
// This is the Metal compiler, running on the device at runtime. The offline
// toolchain (xcrun metal) is not required and is frequently not installed;
// specs/021-metal-bringup.md section 1 argues that this is the stronger check
// anyway, because the compiler that accepts the text is the one that will run
// it.
func (d *Device) Compile(source, entryPoint string) (*Pipeline, error) {
	p := &Pipeline{name: entryPoint}
	var err error
	withPool(func() {
		// nsErr must be initialized to nil and must outlive the call: Metal
		// writes it only on failure, so a leftover value would turn a success
		// into a reported error.
		var nsErr objc.ID
		p.lib = d.id.Send(selNewLibraryWithSource, nsstring(source), objc.ID(0), unsafe.Pointer(&nsErr))
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

// Name is the entry point this pipeline runs.
func (p *Pipeline) Name() string { return p.name }

// Close releases the pipeline, its function, and its library.
func (p *Pipeline) Close() {
	release(p.id)
	release(p.fn)
	release(p.lib)
	p.id, p.fn, p.lib = 0, 0, 0
}
