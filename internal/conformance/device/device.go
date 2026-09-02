// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package device is the conformance harness's device runner: discovery,
// profiles, modes, and skips.
//
// Every test receives a profile explicitly rather than opening a device for
// itself. A failure has to carry the complete device identity, the mode, and
// the capabilities relevant to the case, because a result without that context
// is not actionable: "the reduction was wrong" is not a bug report, and
// "the reduction was wrong on the CPU backend in strict Metal+Vulkan mode with
// subgroups absent" is.
//
// This is test infrastructure and must not become a second implementation of
// device semantics. It opens devices, describes them, and skips; it does not
// decide what a device means.
package device

import (
	"fmt"
	"strings"
	"testing"

	"golang.design/x/accel"
)

// Mode is how a profile constrains what the device reports.
type Mode int

const (
	// StrictPortable enforces the portable intersection of an explicit target
	// set, including forced capability absence. A kernel that runs here runs on
	// that stated set, not on an implied future backend.
	StrictPortable Mode = iota

	// Permissive exposes the host's natural behaviour.
	Permissive

	// Mimic reproduces a captured device's contract so a remote failure can be
	// investigated locally.
	Mimic
)

func (m Mode) String() string {
	switch m {
	case StrictPortable:
		return "strict"
	case Permissive:
		return "permissive"
	case Mimic:
		return "mimic"
	}
	return fmt.Sprintf("Mode(%d)", int(m))
}

// Profile is one device under test, with everything a failure has to name.
type Profile struct {
	Backend      accel.Backend
	DeviceName   string
	Mode         Mode
	Targets      []accel.Backend // the strict target set, when Mode is StrictPortable
	Capabilities accel.Capabilities
	Limits       accel.Limits

	open func() (*accel.Device, error)
}

// String is the identity that goes into every log line and failure.
func (p Profile) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%v/%s %q", p.Backend, p.Mode, p.DeviceName)
	if len(p.Targets) > 0 {
		names := make([]string, len(p.Targets))
		for i, t := range p.Targets {
			names[i] = t.String()
		}
		fmt.Fprintf(&b, " targeting %s", strings.Join(names, "+"))
	}
	return b.String()
}

// Open opens the profile's device and closes it when the test ends.
//
// The cleanup is a hard failure rather than a best effort: a device that will
// not close is holding a resource a test leaked, and letting that pass makes
// the lifetime rules advisory.
func (p Profile) Open(t *testing.T) *accel.Device {
	t.Helper()
	d, err := p.open()
	if err != nil {
		t.Fatalf("%s: open: %v", p, err)
	}
	t.Cleanup(func() {
		if err := d.Close(); err != nil {
			t.Errorf("%s: close: %v", p, err)
		}
	})
	return d
}

// CPU returns the CPU backend's permissive profile.
//
// The CPU backend is mandatory on every platform, so this never skips: a run
// that cannot produce it is a broken environment rather than an absent device.
func CPU() Profile {
	opts := accel.CPUOptions{}
	return profileFor(Permissive, nil, opts)
}

// Strict returns the CPU backend enforcing the portable intersection of the
// given targets.
func Strict(targets ...accel.Backend) Profile {
	opts := accel.CPUOptions{Mode: accel.CPUStrict, StrictTargets: targets}
	return profileFor(StrictPortable, targets, opts)
}

// Mimicking returns the CPU backend reproducing a captured contract.
//
// A forced profile may only remove or lower what a device reports. It cannot
// claim hardware support the CPU implementation does not emulate, which is why
// this takes a captured contract rather than a set of flags to raise.
func Mimicking(name string, caps accel.Capabilities, lim accel.Limits) Profile {
	lim = withPositiveSentinels(lim)
	opts := accel.CPUOptions{
		Mode: accel.CPUMimic,
		Mimic: &accel.DeviceProfile{Info: accel.DeviceInfo{
			Name: name, Capabilities: caps, Limits: lim,
		}},
	}
	return profileFor(Mimic, nil, opts)
}

// withPositiveSentinels fills the subgroup bounds a mimicked profile leaves at
// zero, since an opened device never reports a zero-valued limit.
func withPositiveSentinels(lim accel.Limits) accel.Limits {
	if lim.MinSubgroupSize == 0 {
		lim.MinSubgroupSize = 1
	}
	if lim.MaxSubgroupSize == 0 {
		lim.MaxSubgroupSize = 1
	}
	return lim
}

func profileFor(mode Mode, targets []accel.Backend, opts accel.CPUOptions) Profile {
	p := Profile{
		Backend: accel.BackendCPU,
		Mode:    mode,
		Targets: targets,
		open:    func() (*accel.Device, error) { return accel.OpenCPU(opts) },
	}
	// Reporting the profile requires opening once, which is cheap for the CPU
	// backend and is the only way to report what the device actually says rather
	// than what the options asked for.
	if d, err := accel.OpenCPU(opts); err == nil {
		p.DeviceName = d.Info().Name
		p.Capabilities = d.Info().Capabilities
		p.Limits = d.Limits()
		d.Close()
	} else {
		p.DeviceName = fmt.Sprintf("unopenable: %v", err)
	}
	return p
}

// All returns every profile the current tier runs.
//
// The three CPU rows are mandatory on every platform. The Metal row is a
// darwin fact: present when an adapter enumerates, absent when none does, and
// present-but-failing when the job promised one. See metalProfiles.
func All() []Profile {
	all := []Profile{
		CPU(),
		Strict(accel.BackendMetal),
		Strict(accel.BackendMetal, accel.BackendVulkan),
	}
	return append(all, metalProfiles()...)
}

// Each runs fn once per profile as a subtest named by that profile.
func Each(t *testing.T, profiles []Profile, fn func(t *testing.T, p Profile)) {
	t.Helper()
	for _, p := range profiles {
		t.Run(p.String(), func(t *testing.T) { fn(t, p) })
	}
}

// RequireCapability skips when a profile lacks something a case needs, naming
// what was missing and on which device.
//
// A skip has to say what it skipped for. A silent one reads as a pass in a
// summary, which is how a capability ends up untested on every device at once.
func (p Profile) RequireCapability(t *testing.T, name string, have bool) {
	t.Helper()
	if !have {
		t.Skipf("%s: %s is absent on this profile", p, name)
	}
}
