// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal

import (
	"errors"
	"os"
	"testing"

	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/mtl"
)

// Enumerating twice and opening twice compiles the subgroup probe once.
//
// The layer above enumerates on every Enumerate, OpenDevice and OpenBest. Each
// call used to retain a fresh set of MTLDevice objects that nothing closed and,
// because the probe is cached per object, compile it again on each of them.
func TestAdaptersAreEnumeratedOncePerProcess(t *testing.T) {
	first, err := Adapters()
	if err != nil {
		if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
			t.Fatalf("this job promises Metal and found no adapter: %v", err)
		}
		t.Skipf("no Metal adapter (err=%v)", err)
	}
	compiled := mtl.CompileCount()
	devices := mtl.LiveDevices()

	second, err := Adapters()
	if err != nil {
		t.Fatalf("the second enumeration failed: %v", err)
	}
	if len(second) != len(first) {
		t.Fatalf("the second enumeration found %d adapters and the first %d", len(second), len(first))
	}
	for i := range first {
		if first[i].Token() != second[i].Token() {
			t.Errorf("adapter %d changed token between enumerations", i)
		}
	}
	for _, a := range [][]driver.Adapter{first, second} {
		d, err := a[0].Open(nil)
		if err != nil {
			t.Fatalf("open: %v", err)
		}
		if err := d.Close(); err != nil {
			t.Fatalf("close: %v", err)
		}
	}
	if n := mtl.CompileCount() - compiled; n != 0 {
		t.Errorf("a second enumeration and two opens compiled %d pipelines; the probe "+
			"is compiled once per device object, so a fresh object was made", n)
	}
	if n := mtl.LiveDevices() - devices; n != 0 {
		t.Errorf("a second enumeration retained %d more devices", n)
	}
}

// A device that cannot answer for itself fails the enumeration, and every
// device the enumeration retained is released -- not only the one that failed.
func TestAFailedEnumerationReleasesEveryDevice(t *testing.T) {
	before := mtl.LiveDevices()
	devs, err := mtl.Devices()
	if err != nil || len(devs) == 0 {
		if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
			t.Fatalf("this job promises Metal and found no device (err=%v)", err)
		}
		t.Skipf("no Metal device (err=%v)", err)
	}
	if mtl.LiveDevices()-before != int64(len(devs)) {
		t.Fatalf("enumerating %d devices changed the live count by %d", len(devs),
			mtl.LiveDevices()-before)
	}

	// The real devices answer; a device with nothing behind it is the one that
	// fails, and it fails *after* the real ones were wrapped.
	broken := errors.New("no SIMD width")
	bad := &mtl.Device{}
	info := func(d *mtl.Device) (driver.Info, error) {
		if d == bad {
			return driver.Info{}, broken
		}
		return infoFor(d)
	}
	_, err = adaptersFrom(append(devs, bad), info)
	if !errors.Is(err, broken) {
		t.Fatalf("the enumeration reported %v, want the device's own failure", err)
	}
	if n := mtl.LiveDevices() - before; n != 0 {
		t.Fatalf("%d devices are still retained after the enumeration failed", n)
	}
}
