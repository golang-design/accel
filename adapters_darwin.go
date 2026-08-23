// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build darwin

package accel

import (
	"golang.design/x/accel/internal/driver"
	"golang.design/x/accel/internal/metal"
)

// platformAdapters adds the Metal backend on darwin.
//
// A failure to load or probe becomes a diagnostic rather than an error or an
// absence, which is the distinction [Enumerate] exists to preserve: a caller
// must be able to tell "this machine has no Metal device" from "this binary was
// not built with Metal". Spec 006 section 6.4 makes the same point the other
// way round, that a probe failure is not an open error.
func platformAdapters() ([]driver.Adapter, []ProbeDiagnostic) {
	all, err := metal.Adapters()
	if err != nil {
		return nil, []ProbeDiagnostic{{
			Backend: BackendMetal,
			Stage:   ProbeLoadLibrary,
			Err:     err,
		}}
	}
	return all, nil
}
