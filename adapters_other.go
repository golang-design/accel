// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

//go:build !darwin

package accel

import "golang.design/x/accel/internal/driver"

// platformAdapters adds no backend beyond the CPU on platforms whose native
// backend is not built yet.
//
// It returns no diagnostic, because there is nothing to report: a backend that
// was not compiled in is a fact about this binary, and spec 006 section 6.1
// makes that discoverable from the absence of its adapter rather than from a
// diagnostic about a library nobody tried to load.
func platformAdapters() ([]driver.Adapter, []ProbeDiagnostic) { return nil, nil }
