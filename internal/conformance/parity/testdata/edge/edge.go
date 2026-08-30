// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package edge is the extractor's refusal fixture: three alias forms it must
// refuse rather than silently report as an empty surface. An empty surface is
// the dangerous answer, because a gate over one always passes.
package edge

import (
	"time"

	"golang.design/x/accel/internal/conformance/parity/testdata/aliased"
)

// Outside aliases an enumeration from outside the module, which this extractor
// does not follow: it maps an import path to a directory by module prefix.
type Outside = time.Duration

const ASecond = time.Second

// Missing aliases a package these files do not import.
type Missing = nowhere.Thing

// Bare aliases an enumeration and re-exports none of its constants, so the
// surface would come back empty.
type Bare = aliased.Op
