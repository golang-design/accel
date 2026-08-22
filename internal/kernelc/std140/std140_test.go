// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package std140_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/std140"
)

// structNamed type-checks a source snippet and returns one named struct type.
func structNamed(t *testing.T, src, name string) types.Type {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "p.go", "package p\n\n"+src, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	conf := types.Config{Importer: nil, Error: func(error) {}}
	pkg, _ := conf.Check("example.com/p", fset, []*ast.File{f}, nil)
	if pkg == nil {
		t.Fatal("the snippet did not type-check")
	}
	obj := pkg.Scope().Lookup(name)
	if obj == nil {
		t.Fatalf("no type %q", name)
	}
	return obj.Type()
}

// TestSpecExample is spec 001 section 3.3's worked example, offset for offset.
//
// The example exists in the spec precisely because the offsets are not what a
// reader guesses: Steps lands at 28, inside the sixteen-byte slot Origin's three
// components only half fill. That single case is the one an unsafe cast gets
// wrong while working for every struct of four floats, so it is the case worth
// pinning first.
func TestSpecExample(t *testing.T) {
	typ := structNamed(t, `type Params struct {
	Scale   float32
	Origin  [3]float32
	Steps   uint32
	Inverse [4][4]float32
}`, "Params")

	l, err := std140.Of("Params", typ)
	if err != nil {
		t.Fatalf("Of: %v", err)
	}

	for i, want := range []struct {
		name   string
		offset int
		size   int
	}{
		{"Scale", 0, 4},
		{"Origin", 16, 12},
		{"Steps", 28, 4},
		{"Inverse", 32, 64},
	} {
		got := l.Fields[i]
		if got.Name != want.name || got.Offset != want.offset || got.Size != want.size {
			t.Errorf("field %d is %s at %d for %d bytes, want %s at %d for %d\n%s",
				i, got.Name, got.Offset, got.Size, want.name, want.offset, want.size, l)
		}
	}
	if l.Size != 96 {
		t.Errorf("block size is %d, want 96\n%s", l.Size, l)
	}
}

// TestScalarsAndVectors covers each row of spec 001 section 3.3's table on its
// own, so a failure names the rule rather than a struct.
func TestScalarsAndVectors(t *testing.T) {
	for _, tc := range []struct {
		name  string
		decl  string
		align int
		size  int
		block int
	}{
		{"f32", "type T struct{ A float32 }", 4, 4, 16},
		{"i32", "type T struct{ A int32 }", 4, 4, 16},
		{"u32", "type T struct{ A uint32 }", 4, 4, 16},
		{"2-vector", "type T struct{ A [2]float32 }", 8, 8, 16},
		{"3-vector", "type T struct{ A [3]float32 }", 16, 12, 16},
		{"4-vector", "type T struct{ A [4]float32 }", 16, 16, 16},
		{"1-element array", "type T struct{ A [1]float32 }", 16, 16, 16},
		{"array of 5", "type T struct{ A [5]float32 }", 16, 80, 80},
		{"4x4 matrix", "type T struct{ A [4][4]float32 }", 16, 64, 64},
		{"3x3 matrix", "type T struct{ A [3][3]float32 }", 16, 48, 48},
	} {
		t.Run(tc.name, func(t *testing.T) {
			l, err := std140.Of("T", structNamed(t, tc.decl, "T"))
			if err != nil {
				t.Fatalf("Of: %v", err)
			}
			f := l.Fields[0]
			if f.Align != tc.align || f.Size != tc.size {
				t.Errorf("align %d size %d, want align %d size %d\n%s", f.Align, f.Size, tc.align, tc.size, l)
			}
			if l.Size != tc.block {
				t.Errorf("block size %d, want %d\n%s", l.Size, tc.block, l)
			}
		})
	}
}

// TestArrayStrideIsSixteen is the number spec 001 says makes arrays belong in
// storage buffers, asserted directly because a reader who has not seen it does
// not believe it.
func TestArrayStrideIsSixteen(t *testing.T) {
	l, err := std140.Of("T", structNamed(t, "type T struct{ A [64]float32 }", "T"))
	if err != nil {
		t.Fatal(err)
	}
	if got := l.Fields[0].Size; got != 1024 {
		t.Errorf("an array of 64 floats occupies %d bytes, want 1024: the stride rounds up to "+
			"sixteen, which is why arrays belong in storage buffers", got)
	}
	if l.Size != 1024 {
		t.Errorf("block size %d, want 1024", l.Size)
	}
}

// TestNestedStruct covers the rule that a nested struct aligns to sixteen and
// consumes its own size rounded up, which changes where everything after it
// lands.
func TestNestedStruct(t *testing.T) {
	l, err := std140.Of("T", structNamed(t, `type Inner struct {
	A float32
	B float32
}

type T struct {
	Lead  float32
	Nest  Inner
	Trail float32
}`, "T"))
	if err != nil {
		t.Fatal(err)
	}

	for i, want := range []struct {
		name   string
		offset int
		size   int
	}{
		{"Lead", 0, 4},
		{"Nest", 16, 16},
		{"Trail", 32, 4},
	} {
		got := l.Fields[i]
		if got.Name != want.name || got.Offset != want.offset || got.Size != want.size {
			t.Errorf("field %d is %s at %d for %d, want %s at %d for %d\n%s",
				i, got.Name, got.Offset, got.Size, want.name, want.offset, want.size, l)
		}
	}
	// The nested struct's own two floats are eight bytes, and it still consumes
	// sixteen: that rounding is what pushes Trail to 32 rather than 24.
	if l.Fields[1].Elem == nil || l.Fields[1].Elem.Size != 16 {
		t.Errorf("the nested struct's size is not rounded to sixteen\n%s", l)
	}
}

// TestPackingAfterAThreeVector is the case an unsafe cast gets wrong.
//
// A scalar after a three-component vector occupies the tail of the same
// sixteen-byte slot, so the block is smaller than a reader counting members
// would expect. Getting this wrong produces a block whose every later member is
// shifted by four bytes, with no error anywhere.
func TestPackingAfterAThreeVector(t *testing.T) {
	l, err := std140.Of("T", structNamed(t, `type T struct {
	V [3]float32
	S uint32
	W [3]float32
	U uint32
}`, "T"))
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []int{0, 12, 16, 28} {
		if got := l.Fields[i].Offset; got != want {
			t.Errorf("field %d (%s) is at %d, want %d\n%s", i, l.Fields[i].Name, got, want, l)
		}
	}
	if l.Size != 32 {
		t.Errorf("block size is %d, want 32: each scalar fits the tail of the vector's slot\n%s", l.Size, l)
	}
}

// TestForbiddenTypes is spec 001 section 3.3's exclusion table, and each
// message says why rather than only that.
//
// Why, because the fixes differ: bool needs a uint32, int needs int32, an array
// of structs needs restructuring, and a slice needs a storage buffer. A reader
// told only "unsupported type" has to guess which.
func TestForbiddenTypes(t *testing.T) {
	for _, tc := range []struct {
		name string
		decl string
		want string
	}{
		{"bool", "type T struct{ A bool }", "one byte in Go and four on every device"},
		{"int", "type T struct{ A int }", "platform-width"},
		{"uint", "type T struct{ A uint }", "platform-width"},
		{"float64", "type T struct{ A float64 }", "no f64 dtype"},
		{"string", "type T struct{ A string }", "no memory model on a GPU"},
		{"int8", "type T struct{ A int8 }", "only 4-byte scalars"},
		{"uint64", "type T struct{ A uint64 }", "only 4-byte scalars"},
		{"complex", "type T struct{ A complex64 }", "no complex dtype"},
		{"slice", "type T struct{ A []float32 }", "no device representation"},
		{"pointer", "type T struct{ A *float32 }", "no device representation"},
		{"map", "type T struct{ A map[uint32]float32 }", "no device representation"},
		{"unexported field", "type T struct{ a float32 }", "cannot address it"},
		{"embedded field", "type Inner struct{ A float32 }\n\ntype T struct{ Inner }", "embeds"},
		{"array of structs", "type Inner struct{ A float32 }\n\ntype T struct{ A [2]Inner }", "padding trap"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := std140.Of("T", structNamed(t, tc.decl, "T"))
			if err == nil {
				t.Fatal("was accepted")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not explain %q", err, tc.want)
			}
		})
	}
}

// TestNotAStruct covers the entry point's own guard.
func TestNotAStruct(t *testing.T) {
	if _, err := std140.Of("T", types.Typ[types.Float32]); err == nil {
		t.Error("a scalar was accepted as a uniform block")
	}
}

// TestNestingIsBounded checks the depth guard, which exists because a
// generator that crashes is worse than one that refuses.
func TestNestingIsBounded(t *testing.T) {
	var b strings.Builder
	depth := std140.MaxNesting + 3
	for i := range depth {
		if i == 0 {
			b.WriteString("type S0 struct{ A float32 }\n\n")
			continue
		}
		b.WriteString("type S" + itoa(i) + " struct{ A S" + itoa(i-1) + " }\n\n")
	}
	b.WriteString("type T struct{ A S" + itoa(depth-1) + " }")

	if _, err := std140.Of("T", structNamed(t, b.String(), "T")); err == nil {
		t.Error("a struct nested past the bound was accepted")
	} else if !strings.Contains(err.Error(), "nests more than") {
		t.Errorf("error %q does not say what the limit is", err)
	}
}

func itoa(i int) string {
	if i < 10 {
		return string(rune('0' + i))
	}
	return string(rune('0'+i/10)) + string(rune('0'+i%10))
}

// TestLayoutStringExplainsOffsets covers the diagnostic, since a mismatch
// between a caller's expectation and the device's layout is answered by seeing
// where each member landed.
func TestLayoutStringExplainsOffsets(t *testing.T) {
	l, err := std140.Of("Params", structNamed(t, `type Params struct {
	Scale  float32
	Origin [3]float32
	Steps  uint32
	M      [4][4]float32
	N      [8]float32
	Inner  struct{ A float32 }
}`, "Params"))
	if err != nil {
		t.Fatal(err)
	}
	s := l.String()
	for _, want := range []string{
		"Params (", "Scale", "float32", "3-vector", "columns of", "stride 16", "nested struct",
	} {
		if !strings.Contains(s, want) {
			t.Errorf("the description omits %q:\n%s", want, s)
		}
	}
}
