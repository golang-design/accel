// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package mtl is the Objective-C shim the Metal backend is built on.
//
// It is cgo-free: every call reaches Metal.framework through purego's
// objc_msgSend, which is what specs/000-decisions.md decision 2 requires and
// what specs/006-backends.md section 2.2 says the predecessor proved.
//
// The package has no non-darwin implementation because there is nothing to
// stub. A caller reaches it only from a //go:build darwin file, and the file
// this comment is in exists so that a build for another platform sees a package
// rather than a directory whose every file is excluded.
package mtl
