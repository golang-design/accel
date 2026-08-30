// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package surface is the universe extractor's fixture. It holds all three
// declaration forms the module actually uses, so a change to the extractor is
// checked against the shapes rather than against one of them.
package surface

import "golang.design/x/accel/internal/conformance/parity/testdata/aliased"

// Colour is the iota form: the type on the first spec, inherited by the rest.
type Colour uint8

const (
	Red Colour = iota
	Green
	Blue

	// unexported is not part of a public surface and must not appear.
	unexported
)

// Mask is the bit form, with a derived constant after it that is not a member.
type Mask uint8

const (
	MaskLow Mask = 1 << iota
	MaskHigh
)

// MaskAll is derived and is not a member of the enumeration.
const MaskAll = MaskLow | MaskHigh

// Op is the alias form. The constants are selectors into the aliased package.
type Op = aliased.Op

const (
	OpKeep    = aliased.OpKeep
	OpZero    = aliased.OpZero
	OpReplace = aliased.OpReplace
)

// Builder is the receiver an operator takes first.
type Builder struct{}

// Value is what an operator returns.
type Value struct{}

// Add is an operator.
func Add(b *Builder, x, y *Value) *Value { return x }

// Mul is an operator.
func Mul(b *Builder, x, y *Value) *Value { return x }

// Configure is not an operator: it takes the builder as a receiver.
func (b *Builder) Configure() {}

// Helper is not an operator: its first parameter is not a *Builder.
func Helper(v *Value) *Value { return v }

// unexportedOp is not part of the surface.
func unexportedOp(b *Builder) *Value { return nil }

var _ = unexported
var _ = unexportedOp
