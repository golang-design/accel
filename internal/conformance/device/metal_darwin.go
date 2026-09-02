// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package device

import (
	"fmt"
	"os"

	"golang.design/x/accel"
)

// metalProfiles is the Metal row of [All].
//
// One row when an adapter enumerates, none when it does not -- unless the job
// promised Metal. specs/006-backends.md section 7: a Tier 1 job promises only
// the CPU, so a Mac without a usable GPU must not turn it red, and a row that
// skipped would be reported alongside the CPU rows as if it had run. Under
// ACCEL_REQUIRE_METAL a missing adapter is a row whose Open fails, so the
// promise is broken loudly in every case that would have used the device.
func metalProfiles() []Profile {
	return metalProfilesFrom(accel.Enumerate(), os.Getenv("ACCEL_REQUIRE_METAL") != "")
}

// metalProfilesFrom is metalProfiles over a given enumeration and promise, so
// the two no-adapter rows can be tested on a machine that has one.
func metalProfilesFrom(e accel.Enumeration, required bool) []Profile {
	for _, info := range e.Devices {
		if info.Backend != accel.BackendMetal {
			continue
		}
		id := info.ID
		return []Profile{{
			Backend:      accel.BackendMetal,
			DeviceName:   info.Name,
			Mode:         Permissive,
			Capabilities: info.Capabilities,
			Limits:       info.Limits,
			open:         func() (*accel.Device, error) { return accel.OpenDevice(id) },
		}}
	}
	if !required {
		return nil
	}
	err := fmt.Errorf("this job promises Metal and enumerated no adapter; diagnostics: %v",
		e.Diagnostics)
	return []Profile{{
		Backend:    accel.BackendMetal,
		DeviceName: "no Metal adapter",
		Mode:       Permissive,
		open:       func() (*accel.Device, error) { return nil, err },
	}}
}
