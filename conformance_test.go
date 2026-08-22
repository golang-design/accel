// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"math"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/device"
	"golang.design/x/accel/internal/conformance/numeq"
)

// TestConformanceRoundTrip runs M1's byte-exactness case through the harness
// rather than opening a device inline, which is what spec 011 section 2 asks
// for: every test receives a profile explicitly, and a failure names the
// complete device identity and mode.
//
// It runs the same case under every profile the tier reports, including the
// strict portable ones, because a result that holds in developer mode and not
// under the portable intersection is the failure mode strict mode exists to
// catch. Transfers move bytes and do not depend on a capability, so all of them
// must agree here; the value is that adding Metal adds a row rather than a test.
func TestConformanceRoundTrip(t *testing.T) {
	device.Each(t, device.All(), func(t *testing.T, p device.Profile) {
		d := p.Open(t)
		q := d.Queue()

		pool, err := d.NewPool(accel.MemoryDevice, 1<<20)
		if err != nil {
			t.Fatalf("%s: NewPool: %v", p, err)
		}
		defer pool.Close()

		want := []uint32{0, 1, math.MaxUint32, 0x0f0f0f0f, 7}
		b, err := pool.Alloc(accel.BufferDescriptor{
			DType: accel.U32, Count: len(want),
			Usage: accel.UsageCopyDst | accel.UsageCopySrc, Label: "conformance",
		})
		if err != nil {
			t.Fatalf("%s: Alloc: %v", p, err)
		}
		defer b.Close()

		if err := q.WriteBuffer(b, 0, want); err != nil {
			t.Fatalf("%s: WriteBuffer: %v", p, err)
		}
		got := make([]uint32, len(want))
		if err := q.ReadBuffer(b, 0, got); err != nil {
			t.Fatalf("%s: ReadBuffer: %v", p, err)
		}

		// Exact, not approximate. A transfer moves bytes and never converts, so a
		// byte that comes back different is a defect rather than a rounding
		// difference, and comparing under any tolerance would hide the only class
		// of bug this path can have.
		if r := numeq.Exact(got, want); !r.Equal {
			t.Errorf("%s: round trip: %v", p, r)
		}
	})
}

// TestConformanceProfilesReportTheirIdentity checks that the runner reports
// what a failure would have to name, since a result without that context is
// not actionable.
func TestConformanceProfilesReportTheirIdentity(t *testing.T) {
	for _, p := range device.All() {
		if p.DeviceName == "" {
			t.Errorf("%v profile has no device name", p.Mode)
		}
		if got := p.String(); got == "" {
			t.Error("a profile has no identity string")
		}
		for _, v := range accel.LimitValues(p.Limits) {
			if v.Value <= 0 {
				t.Errorf("%s: %s is %d; a profile carries the opened device's limits", p, v.Name, v.Value)
			}
		}
	}

	// Strict profiles name their target set, because "strict" alone does not say
	// what a kernel that builds under it is portable to.
	strict := device.Strict(accel.BackendMetal, accel.BackendVulkan)
	if got := strict.String(); got == "" {
		t.Fatal("the strict profile has no identity")
	}
	for _, want := range []string{"Metal", "Vulkan", "strict"} {
		if !contains(strict.String(), want) {
			t.Errorf("the strict profile identity %q does not name %q", strict.String(), want)
		}
	}
}

// TestConformanceMimicLowersOnly is spec 011 section 3: a forced profile may
// remove or lower what a device reports, never claim support the CPU does not
// emulate.
func TestConformanceMimicLowersOnly(t *testing.T) {
	p := device.Mimicking("captured", accel.Capabilities{}, accel.Limits{
		MaxPools: 4, MaxPoolBytes: 1 << 20, MaxBufferBytes: 1 << 20,
		MinStorageBufferOffsetAlignment: 256, MinUniformBufferOffsetAlignment: 256,
		MinBufferCopyOffsetAlignment: 16, MaxSharedMemoryBytes: 1024,
	})
	d := p.Open(t)

	if d.Info().Capabilities.Subgroups {
		t.Error("a mimicked profile with no capabilities reports subgroups")
	}
	if got := d.Limits().MaxSharedMemoryBytes; got != 1024 {
		t.Errorf("MaxSharedMemoryBytes = %d, want the captured 1024", got)
	}
	// The subgroup sentinel is filled in, since an opened device never reports a
	// zero-valued limit.
	if got := d.Limits().MinSubgroupSize; got != 1 {
		t.Errorf("MinSubgroupSize = %d, want the 1 sentinel", got)
	}

	// A skip names what was missing and on which device rather than passing
	// quietly, which in a summary is indistinguishable from a pass.
	t.Run("capability-absent case skips loudly", func(t *testing.T) {
		p.RequireCapability(t, "subgroups", d.Info().Capabilities.Subgroups)
		t.Error("the case ran on a profile without subgroups")
	})
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
