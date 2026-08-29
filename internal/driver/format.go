// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package driver

import "fmt"

// Format is an attachment's pixel format, in the vocabulary a plan speaks.
//
// # Why this is a second spelling rather than the only one
//
// [LoadOp] is defined here and aliased by accel, and its doc argues that two
// definitions with nothing pinning them together is how LoadKeep and
// LoadDontCare would swap silently. That argument is about a pair a backend
// cannot tell apart by its effect: no test can see the swap.
//
// A format swap is not that. Two formats that trade places produce different
// pixels, and the CPU oracle compares pixels, so the mistranslation is loud.
// What the accel-side type carries and this one must not is the *capability*
// table -- renderable, sampleable, blendable, per device. A backend is told
// what to do rather than asked what is possible, so moving that table down to
// the seam would put device-capability policy where a backend could read it.
//
// The mapping is written out as a switch rather than a numeric cast, and
// TestEveryFormatHasOnePlanSpelling walks every value. That is the same
// arrangement [BlendFactor] has with internal/raster's, and for the same
// reason: two lists that happen to agree are one insertion away from shifting
// every value after it.
type Format uint8

const (
	// FormatInvalid is the zero value and names no format. An attachment that
	// carried it would be one a backend decoded by guessing.
	FormatInvalid Format = iota

	RGBA8Unorm
	RGBA8UnormSRGB
	BGRA8Unorm
	R16Float
	RG16Float
	RGBA16Float
	R32Float
	RG32Float
	RGBA32Float
	Depth32Float
	Depth24PlusStencil8
	Depth32FloatStencil8
)

// formatNames are the names accel spells these with. Equality of the two
// spellings is what TestEveryFormatHasOnePlanSpelling asserts, so a name is
// part of the contract rather than a diagnostic convenience.
var formatNames = map[Format]string{
	RGBA8Unorm:           "RGBA8Unorm",
	RGBA8UnormSRGB:       "RGBA8UnormSRGB",
	BGRA8Unorm:           "BGRA8Unorm",
	R16Float:             "R16Float",
	RG16Float:            "RG16Float",
	RGBA16Float:          "RGBA16Float",
	R32Float:             "R32Float",
	RG32Float:            "RG32Float",
	RGBA32Float:          "RGBA32Float",
	Depth32Float:         "Depth32Float",
	Depth24PlusStencil8:  "Depth24PlusStencil8",
	Depth32FloatStencil8: "Depth32FloatStencil8",
}

func (f Format) String() string {
	if n, ok := formatNames[f]; ok {
		return n
	}
	if f == FormatInvalid {
		return "FormatInvalid"
	}
	return fmt.Sprintf("Format(%d)", uint8(f))
}

// Formats is every format a plan may name, in declaration order. It exists so
// a test can walk the enumeration rather than restate it, which is what makes
// a value added here and forgotten elsewhere a failure rather than a silence.
var Formats = []Format{
	RGBA8Unorm, RGBA8UnormSRGB, BGRA8Unorm,
	R16Float, RG16Float, RGBA16Float,
	R32Float, RG32Float, RGBA32Float,
	Depth32Float, Depth24PlusStencil8, Depth32FloatStencil8,
}
