// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"
)

// # Ownership
//
// specs/021-metal-bringup.md section 2 states the rule this file implements:
//
//	The backend retains every object it stores in a Go struct, and releases it
//	exactly once, in Close. An object obtained from a new* selector is already
//	+1 and is not retained again.
//
// Objective-C's naming convention is load-bearing here rather than stylistic: a
// selector beginning with new, alloc, copy or mutableCopy returns +1, and every
// other selector returns an object the caller does not own. Getting that
// backwards produces either a leak, which nothing notices, or a release of an
// object still in use, which crashes inside objc_msgSend with a stack that
// points nowhere useful.
//
// So every wrapper in this package documents which of the two it is, and the
// [Object.Release] calls all sit in one place per type.

const (
	metalPath      = "/System/Library/Frameworks/Metal.framework/Metal"
	libobjcPath    = "/usr/lib/libobjc.A.dylib"
	foundationPath = "/System/Library/Frameworks/Foundation.framework/Foundation"
)

var (
	loadOnce sync.Once
	loadErr  error

	// createSystemDefaultDevice returns the default device, +1 retained.
	createSystemDefaultDevice func() uintptr
	// copyAllDevices returns an NSArray of every device, +1 retained. It is
	// absent on some platforms, and nil here means "ask for the default one".
	copyAllDevices func() uintptr

	autoreleasePoolPush func() uintptr
	autoreleasePoolPop  func(uintptr)
)

// load resolves every symbol this package needs, once.
//
// It returns an error rather than panicking because a machine without Metal is
// an ordinary outcome that specs/006-backends.md section 6.4 requires be
// reported as a probe diagnostic, not a crash: a caller must be able to tell
// "no Metal device" from "Metal was not built".
func load() error {
	loadOnce.Do(func() {
		metal, err := purego.Dlopen(metalPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			loadErr = fmt.Errorf("accel/mtl: Metal.framework: %w", err)
			return
		}
		libobjc, err := purego.Dlopen(libobjcPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			loadErr = fmt.Errorf("accel/mtl: libobjc: %w", err)
			return
		}
		if _, err := purego.Dlopen(foundationPath, purego.RTLD_NOW|purego.RTLD_GLOBAL); err != nil {
			loadErr = fmt.Errorf("accel/mtl: Foundation.framework: %w", err)
			return
		}

		sym, err := purego.Dlsym(metal, "MTLCreateSystemDefaultDevice")
		if err != nil {
			loadErr = fmt.Errorf("accel/mtl: MTLCreateSystemDefaultDevice: %w", err)
			return
		}
		purego.RegisterFunc(&createSystemDefaultDevice, sym)

		// Absent on iOS-derived platforms. Its absence is not an error: the
		// default device is then the only device.
		if sym, err := purego.Dlsym(metal, "MTLCopyAllDevices"); err == nil {
			purego.RegisterFunc(&copyAllDevices, sym)
		}

		push, err := purego.Dlsym(libobjc, "objc_autoreleasePoolPush")
		if err != nil {
			loadErr = fmt.Errorf("accel/mtl: objc_autoreleasePoolPush: %w", err)
			return
		}
		pop, err := purego.Dlsym(libobjc, "objc_autoreleasePoolPop")
		if err != nil {
			loadErr = fmt.Errorf("accel/mtl: objc_autoreleasePoolPop: %w", err)
			return
		}
		purego.RegisterFunc(&autoreleasePoolPush, push)
		purego.RegisterFunc(&autoreleasePoolPop, pop)
	})
	return loadErr
}

// Selectors, registered once. Registering a selector is a hash lookup in the
// runtime and would be correct to do at every call; they are cached because
// they are also spelled wrong exactly once, and a var block is a place to read
// the whole spine of names this backend depends on.
var (
	selName                 = objc.RegisterName("name")
	selUTF8String           = objc.RegisterName("UTF8String")
	selRetain               = objc.RegisterName("retain")
	selRelease              = objc.RegisterName("release")
	selRegistryID           = objc.RegisterName("registryID")
	selLocalizedDescription = objc.RegisterName("localizedDescription")
	selStringWithUTF8String = objc.RegisterName("stringWithUTF8String:")
	selCount                = objc.RegisterName("count")
	selObjectAtIndex        = objc.RegisterName("objectAtIndex:")

	classNSString = objc.GetClass("NSString")
)

// withPool runs f on one OS thread with its own autorelease pool.
//
// Both halves matter and neither is optional. Metal returns autoreleased
// objects from selectors like -commandBuffer, so a pool must exist or they
// leak; and a pool must be popped on the thread that pushed it, so the
// goroutine may not migrate in between. See docs/conventions.md, "Objective-C
// object lifetime across completion handlers".
func withPool(f func()) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	p := autoreleasePoolPush()
	defer autoreleasePoolPop(p)
	f()
}

// gostring copies a NUL-terminated C string.
//
// It copies rather than aliasing because the bytes belong to an Objective-C
// object that may be autoreleased out from under the caller: -UTF8String
// returns storage owned by the receiver, not by us.
//
// It takes an unsafe.Pointer rather than the uintptr the message send would
// naturally produce, because go vet is right about the difference: a uintptr
// converted back to a pointer is only valid if nothing moved in between, and
// writing it that way here would train the reader to accept it where a Go
// pointer is involved and it is genuinely unsound.
func gostring(p unsafe.Pointer) string {
	if p == nil {
		return ""
	}
	b := (*byte)(p)
	var n int
	for *(*byte)(unsafe.Add(p, n)) != 0 {
		n++
	}
	return string(unsafe.Slice(b, n))
}

// utf8 sends -UTF8String and copies the result. Every NSString this package
// reads goes through it, so the unsafe conversion lives in one place.
func utf8(s objc.ID) string {
	if s == 0 {
		return ""
	}
	return gostring(objc.Send[unsafe.Pointer](s, selUTF8String))
}

// nsstring makes an autoreleased NSString. Valid only until the enclosing pool
// drains, which is why every caller is inside [withPool].
func nsstring(s string) objc.ID {
	return objc.ID(classNSString).Send(selStringWithUTF8String, s)
}

// describe turns an NSError into a Go error, or nil.
//
// Metal writes its out-parameter only on failure, so a caller must decide
// failure from the returned object rather than from this pointer: an NSError
// left over from an earlier call would otherwise turn a success into an error.
func describe(what string, err objc.ID) error {
	if err == 0 {
		return fmt.Errorf("accel/mtl: %s failed, and Metal reported no reason", what)
	}
	desc := err.Send(selLocalizedDescription)
	if desc == 0 {
		return fmt.Errorf("accel/mtl: %s failed, and the error had no description", what)
	}
	return errors.New("accel/mtl: " + what + " failed: " + utf8(desc))
}

// retain and release are the two halves of the ownership rule, named so that
// grepping for them finds every place ownership changes.
func retain(id objc.ID) objc.ID {
	if id != 0 {
		id.Send(selRetain)
	}
	return id
}

func release(id objc.ID) {
	if id != 0 {
		id.Send(selRelease)
	}
}
