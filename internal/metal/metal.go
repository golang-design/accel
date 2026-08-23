// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package metal is the Metal backend.
//
// It implements the driver seam over [golang.design/x/accel/internal/mtl],
// which is the Objective-C shim. The split is the same one every backend has:
// this package knows about plans, blocks and capabilities, and knows nothing
// about selectors.
//
// The build for any other platform sees this file and no implementation, which
// is what lets accel link the backend in on darwin without a build tag at the
// call site.
package metal
