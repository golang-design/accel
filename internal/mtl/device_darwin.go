// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl

import (
	"sync"

	"github.com/ebitengine/purego/objc"
)

// Size is MTLSize: three unsigned lengths, passed and returned by value.
//
// By value is the detail that matters. objc_msgSend is variadic in C and is not
// variadic in the ABI, so an MTLSize argument is passed as three registers
// rather than as a pointer. Passing a pointer compiles and runs, and dispatches
// a grid nobody asked for, which is why specs/021-metal-bringup.md section 2
// puts a test on the grid rather than a comment on the call.
type Size struct{ Width, Height, Depth uint64 }

// Device is an open MTLDevice, retained.
type Device struct {
	id objc.ID

	name       string
	registryID uint64

	// The device-level ceilings. Read once at open: they do not change, and a
	// caller asking per dispatch would be paying a message send for a constant.
	MaxThreadsPerThreadgroup      Size
	MaxTotalThreadsPerThreadgroup int
	MaxThreadgroupMemoryBytes     int
	MaxBufferBytes                int
	UnifiedMemory                 bool
	LowPower                      bool

	// The SIMD width, which only a compiled pipeline can report. See
	// [Device.SubgroupSize].
	widthOnce sync.Once
	width     int
}

var (
	selMaxThreadsPerThreadgroup   = objc.RegisterName("maxThreadsPerThreadgroup")
	selMaxThreadgroupMemoryLength = objc.RegisterName("maxThreadgroupMemoryLength")
	selMaxBufferLength            = objc.RegisterName("maxBufferLength")
	selHasUnifiedMemory           = objc.RegisterName("hasUnifiedMemory")
	selIsLowPower                 = objc.RegisterName("isLowPower")
)

// Devices returns every Metal device, each retained.
//
// The caller closes each one. An empty result is not an error: it is a machine
// with no Metal device, which specs/006-backends.md section 6.4 wants reported
// as a diagnostic rather than as a failure to open something nobody asked for.
func Devices() ([]*Device, error) {
	if err := load(); err != nil {
		return nil, err
	}
	var out []*Device
	var err error
	withPool(func() {
		// MTLCopyAllDevices returns the array +1 and the devices inside it
		// unowned, so each device is retained separately and the array is
		// released. This is the ownership rule doing real work: the naming
		// convention says Copy* is +1, and says nothing about the contents.
		if copyAllDevices != nil {
			if arr := objc.ID(copyAllDevices()); arr != 0 {
				defer release(arr)
				n := int(arr.Send(selCount))
				for i := range n {
					d := arr.Send(selObjectAtIndex, uintptr(i))
					out = append(out, newDevice(retain(d)))
				}
			}
		}
		if len(out) == 0 {
			if d := objc.ID(createSystemDefaultDevice()); d != 0 {
				// Already +1: a name beginning with Create is the C spelling of
				// the same convention, so it is not retained again.
				out = append(out, newDevice(d))
			}
		}
	})
	return out, err
}

// newDevice reads the ceilings that never change. The device is already
// retained by the caller.
func newDevice(id objc.ID) *Device {
	d := &Device{id: id}
	d.name = utf8(id.Send(selName))
	d.registryID = uint64(id.Send(selRegistryID))
	d.MaxThreadsPerThreadgroup = objc.Send[Size](id, selMaxThreadsPerThreadgroup)
	d.MaxTotalThreadsPerThreadgroup = int(d.MaxThreadsPerThreadgroup.Width)
	d.MaxThreadgroupMemoryBytes = int(id.Send(selMaxThreadgroupMemoryLength))
	d.MaxBufferBytes = int(id.Send(selMaxBufferLength))
	d.UnifiedMemory = id.Send(selHasUnifiedMemory) != 0
	d.LowPower = id.Send(selIsLowPower) != 0
	return d
}

// Name is the device's product name, as Metal reports it.
func (d *Device) Name() string { return d.name }

// RegistryID is the device's identity in the IO registry. It is stable for as
// long as the device is attached, which is what makes it usable as the seed of
// an adapter token.
func (d *Device) RegistryID() uint64 { return d.registryID }

// Close releases the device.
func (d *Device) Close() {
	release(d.id)
	d.id = 0
}
