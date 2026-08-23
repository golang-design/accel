// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor

import "fmt"

// ScalarKind is the type of a named per-step value.
type ScalarKind uint8

const (
	// ScalarU32 is an unsigned integer, such as a sequence length.
	ScalarU32 ScalarKind = iota
	// ScalarF32 is a float, such as a softmax scale or a RoPE base.
	ScalarF32
)

func (k ScalarKind) String() string {
	switch k {
	case ScalarF32:
		return "f32"
	case ScalarU32:
		return "u32"
	}
	return fmt.Sprintf("ScalarKind(%d)", uint8(k))
}

// ScalarDesc declares a named runtime value.
type ScalarDesc struct {
	Name string
	Kind ScalarKind
}

// ScalarValue is one named value, supplied at submission.
type ScalarValue struct {
	Kind ScalarKind
	U32  uint32
	F32  float32
}

// F32 makes a float scalar value.
func F32(v float32) ScalarValue { return ScalarValue{Kind: ScalarF32, F32: v} }

// U32 makes an unsigned integer scalar value.
func U32(v uint32) ScalarValue { return ScalarValue{Kind: ScalarU32, U32: v} }

func (v ScalarValue) String() string {
	if v.Kind == ScalarF32 {
		return fmt.Sprintf("%v", v.F32)
	}
	return fmt.Sprintf("%d", v.U32)
}

// Scalar declares a named per-step value an operator may read.
//
// Declared rather than inferred from use, so a misspelled operator argument is
// an error instead of a new value nobody binds. specs/007-tensor-layer.md gives
// that reason and it is worth keeping: what it prevents is a plan that compiles,
// runs, and reads zero.
//
// The value may change on every submission. What may not change is anything
// structural: an attribute that alters a shape, a layout, or which kernel is
// selected needs another plan, because the barriers and the transient layout
// were computed from it.
func Scalar(b *Builder, d ScalarDesc) {
	if b.scalarPos == nil {
		b.scalarPos = map[string]int{}
	}
	switch {
	case d.Name == "":
		b.fail(1, "Scalar", "a scalar needs a name")
		return
	case d.Kind != ScalarF32 && d.Kind != ScalarU32:
		b.fail(1, "Scalar", "%q is declared %v, and a scalar is f32 or u32", d.Name, d.Kind)
		return
	}
	if _, dup := b.scalarPos[d.Name]; dup {
		b.fail(1, "Scalar", "%q is declared twice", d.Name)
		return
	}
	b.scalarPos[d.Name] = len(b.scalars)
	b.scalars = append(b.scalars, d)
}

// scalarKind reports a declared scalar's kind.
func (b *Builder) scalarKind(name string) (ScalarKind, bool) {
	i, ok := b.scalarPos[name]
	if !ok {
		return 0, false
	}
	return b.scalars[i].Kind, true
}
