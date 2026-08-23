// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package mtl

import (
	"errors"
	"os"
	"strings"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego/objc"
)

// The helpers that translate between Objective-C and Go answer for the null
// cases, which is where a shim crashes rather than fails.
//
// These are reachable only from inside the package, and they are worth reaching:
// every one of them is on the path taken when Metal reports a failure, which is
// exactly the path that is not exercised when everything works. A shim whose
// error handling has never run is a shim that turns a Metal error into a
// segmentation fault.
func TestNullHandling(t *testing.T) {
	if got := gostring(nil); got != "" {
		t.Errorf("gostring(nil) = %q, want empty", got)
	}
	if got := utf8(0); got != "" {
		t.Errorf("utf8 of a nil NSString = %q, want empty", got)
	}

	// describe is called with whatever Metal wrote into the out-parameter, and
	// Metal writes nothing on some failures. Both degenerate cases must produce
	// an error that still says what failed.
	err := describe("compiling", 0)
	if err == nil {
		t.Fatal("describe with no NSError must still be an error: the caller already " +
			"decided the operation failed")
	}
	if !strings.Contains(err.Error(), "compiling") {
		t.Errorf("the error should name the operation: %v", err)
	}
	if !strings.Contains(err.Error(), "no reason") {
		t.Errorf("the error should say Metal gave no reason: %v", err)
	}
}

// A C string is copied rather than aliased, so it survives its owner.
//
// -UTF8String returns storage owned by the receiver, which is typically
// autoreleased. A Go string aliasing it would read freed memory at an
// unpredictable later moment, which is the least debuggable failure this
// package could have.
func TestGostringCopies(t *testing.T) {
	buf := []byte("hello\x00")
	got := gostring(unsafe.Pointer(&buf[0]))
	if got != "hello" {
		t.Fatalf("gostring = %q, want %q", got, "hello")
	}
	buf[0] = 'j'
	if got != "hello" {
		t.Errorf("the string changed with its source, so it aliases rather than copies")
	}
}

// The two allocation errors say the size, because a caller who cannot see the
// size cannot tell a bad request from a device limit.
func TestAllocationErrors(t *testing.T) {
	if !strings.Contains(errSize(0).Error(), "0") {
		t.Errorf("errSize should name the size: %v", errSize(0))
	}
	if !strings.Contains(errAlloc(1<<40).Error(), "1099511627776") {
		t.Errorf("errAlloc should name the size: %v", errAlloc(1<<40))
	}
}

// The compile options object is created and answers to one of the two math-mode
// selectors.
//
// Which one it answers to is the point: -setFastMathEnabled: is deprecated in
// favour of -setMathMode: and both exist in the wild, so the code asks the
// object rather than assuming. This asserts that at least one of them is
// present, because a Metal that answered to neither would leave the compile
// silently on its default -- which permits contraction, and which the CPU
// oracle would then disagree with.
func TestCompileOptionsRespondToAMathSelector(t *testing.T) {
	if err := load(); err != nil {
		t.Fatalf("load: %v", err)
	}
	opts := compileOptions()
	if opts == 0 {
		t.Fatal("MTLCompileOptions could not be created")
	}
	defer release(opts)

	modern := opts.Send(selRespondsToSelector, selSetMathMode) != 0
	legacy := opts.Send(selRespondsToSelector, selSetFastMathEnabled) != 0
	if !modern && !legacy {
		t.Fatal("MTLCompileOptions answers to neither -setMathMode: nor " +
			"-setFastMathEnabled:, so nothing is controlling fast math")
	}
	t.Logf("setMathMode: %v, setFastMathEnabled: %v", modern, legacy)
}

// retain and release tolerate nil, because the ownership rule is applied
// uniformly and half the objects it is applied to may not exist.
func TestRetainReleaseTolerateNil(t *testing.T) {
	if got := retain(0); got != 0 {
		t.Errorf("retain(nil) = %v, want 0", got)
	}
	release(0) // must not crash
}

// Every symbol this package needs has a failure message that names it.
//
// These messages are the ones a user sees on a machine where this package does
// not belong, and until now none of them had ever been produced. A message
// nobody has seen printed is a message nobody has checked -- and the failure it
// describes is the one where the reader has least other information to go on.
func TestResolveNamesWhatIsMissing(t *testing.T) {
	fail := errors.New("dlopen said no")
	okOpen := func(string, int) (uintptr, error) { return 1, nil }
	okSym := func(uintptr, string) (uintptr, error) { return 1, nil }

	failNth := func(n int) func(string, int) (uintptr, error) {
		calls := 0
		return func(path string, mode int) (uintptr, error) {
			calls++
			if calls == n {
				return 0, fail
			}
			return 1, nil
		}
	}
	failSym := func(name string) func(uintptr, string) (uintptr, error) {
		return func(_ uintptr, s string) (uintptr, error) {
			if s == name {
				return 0, fail
			}
			return 1, nil
		}
	}

	for _, tc := range []struct {
		name   string
		dlopen func(string, int) (uintptr, error)
		dlsym  func(uintptr, string) (uintptr, error)
		want   string
	}{
		{"no Metal.framework", failNth(1), okSym, "Metal.framework"},
		{"no libobjc", failNth(2), okSym, "libobjc"},
		{"no Foundation", failNth(3), okSym, "Foundation.framework"},
		{"no device constructor", okOpen, failSym("MTLCreateSystemDefaultDevice"),
			"MTLCreateSystemDefaultDevice"},
		{"no pool push", okOpen, failSym("objc_autoreleasePoolPush"), "objc_autoreleasePoolPush"},
		{"no pool pop", okOpen, failSym("objc_autoreleasePoolPop"), "objc_autoreleasePoolPop"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := resolve(tc.dlopen, tc.dlsym)
			if err == nil {
				t.Fatalf("%s produced no error", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("the error should name %q, got %v", tc.want, err)
			}
		})
	}

	// MTLCopyAllDevices missing is not an error: the default device is then the
	// only device. This is the one absent symbol that must be tolerated, and it
	// is easy to turn into a failure by accident.
	s, err := resolve(okOpen, failSym("MTLCopyAllDevices"))
	if err != nil {
		t.Errorf("MTLCopyAllDevices is optional and its absence must not fail: %v", err)
	}
	if s.copyAllDevices != 0 {
		t.Error("a missing MTLCopyAllDevices should resolve to zero, not to a stale address")
	}
}

// Enumeration falls back to the default device, and reports nothing rather than
// crashing when there is no device at all.
//
// MTLCopyAllDevices is absent on iOS-derived platforms and returns an empty
// array on a Mac whose GPU is not ready. Neither can be produced by asking this
// machine, and both lead to code that has to be right.
func TestDeviceEnumerationFallsBack(t *testing.T) {
	if err := load(); err != nil {
		t.Fatalf("load: %v", err)
	}
	zero := func() uintptr { return 0 }

	// The two fallback cases below need a real default device to fall back to.
	// Whether its absence is a failure or a skip is what the *job* promised:
	// see specs/006-backends.md section 7 and the header of
	// .github/workflows/ci.yml. Tier 2 sets ACCEL_REQUIRE_METAL; Tier 1 runs the
	// same tests on a macOS runner and promises only the CPU backend, so it must
	// not go red for want of a GPU.
	//
	// The last two cases need no device and run either way, which is the point
	// of splitting them out rather than skipping the whole test.
	if d := objc.ID(createSystemDefaultDevice()); d == 0 {
		if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
			t.Fatal("this job promises Metal and there is no default device")
		}
		t.Log("no Metal device: checking only the cases that need none")
	} else {
		release(d)
		if got := devicesFrom(nil, createSystemDefaultDevice); len(got) != 1 {
			t.Errorf("with no MTLCopyAllDevices the default device should be the only "+
				"one, got %d", len(got))
		} else {
			got[0].Close()
		}
		if got := devicesFrom(zero, createSystemDefaultDevice); len(got) != 1 {
			t.Errorf("an empty device array should fall back to the default device, got %d",
				len(got))
		} else {
			got[0].Close()
		}
	}

	if got := devicesFrom(zero, zero); len(got) != 0 {
		t.Errorf("a machine with no device should enumerate none, got %d", len(got))
	}
	if got := devicesFrom(nil, nil); len(got) != 0 {
		t.Errorf("with no constructor at all, enumeration should report none, got %d", len(got))
	}
}
