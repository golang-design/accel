// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package parity_test

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"golang.design/x/accel/internal/conformance/parity"
)

// fixture opens the extractor's fixture package.
//
// It lives under testdata so the go tool does not build it, which is what lets
// it hold a deliberately unexported constant and an operator-shaped function
// that is not an operator without either becoming part of a real package's
// surface.
func fixture(t *testing.T, name string) parity.Package {
	t.Helper()
	p, err := parity.Open(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("open %s: %v", name, err)
	}
	return p
}

// All three declaration forms this module uses, checked against one fixture.
//
// The extractor is the part of the gate nobody watches: a gate that silently
// reported three of a surface's twelve members would pass every day and mean
// nothing, and the only way to notice is to hold a surface whose membership is
// known and assert the whole list.
func TestEnumReadsEveryDeclarationForm(t *testing.T) {
	p := fixture(t, "surface")

	for _, c := range []struct {
		typeName string
		want     []string
		why      string
	}{
		{"Colour", []string{"Red", "Green", "Blue"},
			"the iota form, with the unexported member left out of a public surface"},
		{"Mask", []string{"MaskLow", "MaskHigh"},
			"the bit form: MaskAll is derived from the members and is not one"},
		{"Op", []string{"OpKeep", "OpZero", "OpReplace"},
			"the alias form, whose members are selectors into the aliased package"},
	} {
		t.Run(c.typeName, func(t *testing.T) {
			got, err := parity.Enum(p, c.typeName)
			if err != nil {
				t.Fatalf("Enum(%s): %v", c.typeName, err)
			}
			if !slices.Equal(got, c.want) {
				t.Fatalf("Enum(%s) = %v, want %v\n  %s", c.typeName, got, c.want, c.why)
			}
		})
	}
}

// Declaration order, not sorted order: a failure names members the way the
// source declares them, so a reader looking for the gap knows where to look.
func TestEnumKeepsDeclarationOrder(t *testing.T) {
	got, err := parity.Enum(fixture(t, "surface"), "Colour")
	if err != nil {
		t.Fatalf("Enum: %v", err)
	}
	if got[0] != "Red" {
		t.Fatalf("first member is %q, want Red: sorting would have put Blue first",
			got[0])
	}
}

func TestEnumRefusesATypeWithNoMembers(t *testing.T) {
	_, err := parity.Enum(fixture(t, "surface"), "Builder")
	if err == nil {
		t.Fatal("Enum over a type with no constants returned no error; " +
			"a surface with no members is a gate that can never fail")
	}
	if !strings.Contains(err.Error(), "Builder") {
		t.Errorf("the error does not name the type: %v", err)
	}
}

func TestFuncsFindsOperatorsAndNothingElse(t *testing.T) {
	got, err := parity.Funcs(fixture(t, "surface"), "*Builder")
	if err != nil {
		t.Fatalf("Funcs: %v", err)
	}
	want := []string{"Add", "Mul"}
	if !slices.Equal(got, want) {
		t.Fatalf("Funcs = %v, want %v\n  Configure is a method, Helper takes no "+
			"builder, and unexportedOp is not part of the surface", got, want)
	}
}

func TestFuncsRefusesAShapeNothingHas(t *testing.T) {
	if _, err := parity.Funcs(fixture(t, "surface"), "*Nothing"); err == nil {
		t.Fatal("Funcs over a shape nothing has returned no error")
	}
}

func TestOpenRefusesADirectoryOutsideAModule(t *testing.T) {
	if _, err := parity.Open(string(filepath.Separator)); err == nil {
		t.Fatal("Open above every go.mod returned no error")
	}
}

// The three alias forms the extractor refuses, and the reason it must.
//
// Each of these would otherwise come back as an empty surface, and an empty
// surface is worse than a wrong one: a gate over no members passes every time,
// so a whole enumeration would report full coverage while nothing compared it.
func TestEnumRefusesWhatItCannotFollow(t *testing.T) {
	p := fixture(t, "edge")
	for _, c := range []struct {
		typeName string
		want     string
	}{
		{"Outside", "outside module"},
		{"Missing", "no import"},
		{"Bare", "re-exports none"},
	} {
		t.Run(c.typeName, func(t *testing.T) {
			_, err := parity.Enum(p, c.typeName)
			if err == nil {
				t.Fatalf("Enum(%s) returned no error; an empty surface is a gate "+
					"that always passes", c.typeName)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("the error does not say %q: %v", c.want, err)
			}
		})
	}
}

func TestADirectoryWithNoGoFilesIsRefused(t *testing.T) {
	p := fixture(t, "empty")
	if _, err := parity.Enum(p, "Anything"); err == nil {
		t.Error("Enum over a directory with no Go files returned no error")
	}
	if _, err := parity.Funcs(p, "*Builder"); err == nil {
		t.Error("Funcs over a directory with no Go files returned no error")
	}
}
