// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package aliased is the far side of the re-export form: an enumeration
// declared here and aliased by the fixture package that imports it, which is
// how the render enumerations reach the public package.
package aliased

// Op is the aliased enumeration.
type Op uint8

const (
	OpKeep Op = iota
	OpZero
	OpReplace
)

// Unrelated is an enumeration of another type in the same block position, so a
// gate that ignored the type would pick it up.
type Unrelated uint8

const (
	SomethingElse Unrelated = iota
)
