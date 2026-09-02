// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package metal

import (
	"os"
	"testing"

	"golang.design/x/accel/internal/mtl"
)

// MaxWorkgroupInvocations is the total a compiled pipeline reports, not the
// limit along one axis.
//
// The two coincide on Apple silicon, both 1024, which is why reporting the
// width went unnoticed; the assertion here is that the report follows the
// pipeline, so on a device where they differ it follows the right one.
func TestMaxWorkgroupInvocationsIsThePipelinesTotal(t *testing.T) {
	devs, err := mtl.Devices()
	if err != nil || len(devs) == 0 {
		if os.Getenv("ACCEL_REQUIRE_METAL") != "" {
			t.Fatalf("this job promises Metal and found no device (err=%v)", err)
		}
		t.Skipf("no Metal device (err=%v)", err)
	}
	d := devs[0]
	defer func() {
		for _, x := range devs {
			x.Close()
		}
	}()
	info, err := infoFor(d)
	if err != nil {
		t.Fatalf("infoFor: %v", err)
	}
	p, err := d.Compile(clampSource, "_accel_clamp")
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer p.Close()
	if got := info.Limits.MaxWorkgroupInvocations; got != p.MaxTotalThreadsPerThreadgroup {
		t.Errorf("MaxWorkgroupInvocations is %d and a compiled pipeline reports %d",
			got, p.MaxTotalThreadsPerThreadgroup)
	}
	if got := info.Limits.MaxWorkgroupInvocations; got <= 0 {
		t.Errorf("MaxWorkgroupInvocations is %d", got)
	}
}
