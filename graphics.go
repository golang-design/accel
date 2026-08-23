// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import "golang.design/x/accel/internal/kernel"

// Vec2, Vec3 and Vec4 are the vector spellings a kernel signature uses.
//
// Aliases for Go arrays rather than new named types, because the kernel
// language already spells a vector that way: std140 maps [3]float32 to a
// three-component vector consuming twelve bytes aligned to sixteen, and the
// uniform encoder is generated from exactly those array types. A parallel set
// of named vector types would give the compiler two spellings for one thing,
// and the second is the one nobody teaches the layout code about.
type (
	Vec2 = kernel.Vec2
	Vec3 = kernel.Vec3
	Vec4 = kernel.Vec4
)

// The graphics stage receivers are withdrawn from the public API until a stage
// can run.
//
// specs/032-stage-abi.md designs Vertex, Fragment, Clip and NoVaryings, and
// their doc comments showed //accel:vertex and //accel:fragment examples. The
// front end knows //accel:kernel and //accel:helper only, and drops any other
// directive with no diagnostic — so a reader who copied the doc comment got a
// clean exit code and no stage.
//
// They come back in the commit that lands 032's front end, which is the point
// at which naming them is a promise the library can keep.
