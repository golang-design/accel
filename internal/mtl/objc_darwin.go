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

	// The same three entry points again, as raw addresses called without
	// reflection. See [send] for why both spellings exist.
	fnPoolPush uintptr
	fnPoolPop  uintptr
	fnMsgSend  uintptr
)

// send calls objc_msgSend directly, with the arguments already in the shape the
// ABI wants.
//
// # Why this exists next to objc.ID.Send
//
// purego's Send resolves the call through reflect.MakeFunc, which builds an
// argument frame per call from a signature it re-reads every time. That is the
// right tool for a call whose shape is decided at run time, and every call in
// this package's hot path has a shape decided at compile time. Measured on an
// M2, the reflected form is 620ns a send; this one is the C call and nothing
// else. At five sends per dispatch node and a graph of several hundred nodes,
// that difference is milliseconds a submission -- see
// BenchmarkSubmitAttribution in internal/metal.
//
// # Why the arity is fixed at five
//
// Every selector on this path takes at most five arguments after the receiver
// and the selector, and passing a register a callee does not read is free: the
// AAPCS64 and SysV argument registers are caller-populated and the callee
// simply ignores what it was not declared to take. A variadic Go signature here
// would allocate the slice this avoids.
//
// # What it may not be used for
//
// Arguments that are not integers or pointers. A float or a struct passed by
// value goes in a different register class, and objc_msgSend cannot be told
// that through a uintptr. Those calls keep the reflected form, which reads the
// Go signature and does place them correctly. [ComputeEncoder.Dispatch] is the
// one on this path: MTLSize is a three-word struct.
func send(id objc.ID, sel objc.SEL, a ...uintptr) objc.ID {
	var f [7]uintptr
	f[0], f[1] = uintptr(id), uintptr(sel)
	copy(f[2:], a)
	r, _, _ := purego.SyscallN(fnMsgSend, f[:]...)
	return objc.ID(r)
}

// load resolves every symbol this package needs, once.
//
// It returns an error rather than panicking because a machine without Metal is
// an ordinary outcome that specs/006-backends.md section 6.4 requires be
// reported as a probe diagnostic, not a crash: a caller must be able to tell
// "no Metal device" from "Metal was not built".
func load() error {
	loadOnce.Do(func() {
		var s symbols
		if s, loadErr = resolve(purego.Dlopen, purego.Dlsym); loadErr != nil {
			return
		}
		bind(s)
	})
	return loadErr
}

// symbols is what resolve found, before anything is bound to it.
//
// Separating the lookup from the binding is what makes the lookup testable. An
// earlier version wrote the package's function pointers as it went, so a test
// passing fake lookups overwrote the real MTLCreateSystemDefaultDevice with the
// fake's return value and the next real call jumped to address 1. Finding
// symbols and installing them are different jobs and now say so.
type symbols struct {
	createDevice   uintptr
	copyAllDevices uintptr // zero when absent, which is not an error
	poolPush       uintptr
	poolPop        uintptr
	msgSend        uintptr
}

// resolve looks up every symbol this package needs, through injected lookups.
//
// Injected because every branch in here is an error branch, and every one is on
// the path taken when this package runs somewhere it does not belong: a machine
// with no Metal, a stripped framework, an SDK that renamed something. Those are
// the messages a user actually reads, and a message nobody has seen printed is
// a message nobody has checked.
func resolve(dlopen func(string, int) (uintptr, error), dlsym func(uintptr, string) (uintptr, error)) (symbols, error) {
	var out symbols
	metal, err := dlopen(metalPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return out, fmt.Errorf("accel/mtl: Metal.framework: %w", err)
	}
	libobjc, err := dlopen(libobjcPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return out, fmt.Errorf("accel/mtl: libobjc: %w", err)
	}
	if _, err := dlopen(foundationPath, purego.RTLD_NOW|purego.RTLD_GLOBAL); err != nil {
		return out, fmt.Errorf("accel/mtl: Foundation.framework: %w", err)
	}

	if out.createDevice, err = dlsym(metal, "MTLCreateSystemDefaultDevice"); err != nil {
		return out, fmt.Errorf("accel/mtl: MTLCreateSystemDefaultDevice: %w", err)
	}
	// Absent on iOS-derived platforms. Its absence is not an error: the default
	// device is then the only device.
	out.copyAllDevices, _ = dlsym(metal, "MTLCopyAllDevices")

	if out.poolPush, err = dlsym(libobjc, "objc_autoreleasePoolPush"); err != nil {
		return out, fmt.Errorf("accel/mtl: objc_autoreleasePoolPush: %w", err)
	}
	if out.poolPop, err = dlsym(libobjc, "objc_autoreleasePoolPop"); err != nil {
		return out, fmt.Errorf("accel/mtl: objc_autoreleasePoolPop: %w", err)
	}
	if out.msgSend, err = dlsym(libobjc, "objc_msgSend"); err != nil {
		return out, fmt.Errorf("accel/mtl: objc_msgSend: %w", err)
	}
	return out, nil
}

// bind installs resolved symbols and the classes that need a loaded image.
//
// NSString is looked up here rather than in a var initializer, and the
// distinction is not stylistic. A selector can be registered before anything is
// loaded, because registering one creates it; a class cannot, because
// objc_getClass returns nil for a class whose image is not mapped. Resolved too
// early it is nil, +stringWithUTF8String: to nil returns nil, and Metal aborts
// the process on an assertion inside -newLibraryWithSource: rather than
// returning the error this package is careful to read.
func bind(s symbols) {
	purego.RegisterFunc(&createSystemDefaultDevice, s.createDevice)
	if s.copyAllDevices != 0 {
		purego.RegisterFunc(&copyAllDevices, s.copyAllDevices)
	}
	purego.RegisterFunc(&autoreleasePoolPush, s.poolPush)
	purego.RegisterFunc(&autoreleasePoolPop, s.poolPop)
	fnPoolPush, fnPoolPop, fnMsgSend = s.poolPush, s.poolPop, s.msgSend
	classNSString = objc.GetClass("NSString")
	if classNSString == 0 {
		loadErr = errors.New("accel/mtl: Foundation is loaded but NSString is not registered")
	}
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
// The pool is pushed and popped without reflection for the reason [send] gives:
// a pool is taken around every Metal call this package makes, so its cost is
// multiplied by the size of a graph. The discipline is unchanged -- same thread,
// same nesting, same drain -- and only the way the two C functions are reached
// is different.
func withPool(f func()) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	p, _, _ := purego.SyscallN(fnPoolPush)
	defer purego.SyscallN(fnPoolPop, p)
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
		send(id, selRetain)
	}
	return id
}

func release(id objc.ID) {
	if id != 0 {
		send(id, selRelease)
	}
}
