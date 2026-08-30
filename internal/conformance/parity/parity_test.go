// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package parity_test

import (
	"strings"
	"testing"

	"golang.design/x/accel/internal/conformance/parity"
)

// A gate nobody has seen fail is a gate nobody knows works.
//
// Each case here is a way the pairing of a universe and a registry can be
// wrong, and asserts that the failure names the thing a reader has to fix.
func TestCheckNamesEveryKindOfGap(t *testing.T) {
	universe := []string{"Red", "Green", "Blue"}

	for _, c := range []struct {
		name string
		set  parity.Set
		want []string
	}{
		{
			name: "a member with no case",
			set: parity.Set{Surface: "Colour",
				Covered: parity.Covers{"Red", "Green"}},
			want: []string{"Colour has 3 members", "Blue"},
		},
		{
			name: "an exclusion with no reason",
			set: parity.Set{Surface: "Colour",
				Covered:  parity.Covers{"Red", "Green"},
				Excludes: []parity.Excluded{{Name: "Blue"}}},
			want: []string{"Blue is excluded with no reason"},
		},
		{
			name: "a misspelled claim",
			set: parity.Set{Surface: "Colour",
				Covered: parity.Covers{"Red", "Green", "Bleu"}},
			want: []string{"Bleu is claimed", "check the spelling", "Blue"},
		},
		{
			name: "both covered and excluded",
			set: parity.Set{Surface: "Colour",
				Covered: parity.Covers{"Red", "Green", "Blue"},
				Excludes: []parity.Excluded{
					{Name: "Blue", Why: "it is refused on both backends"}}},
			want: []string{"Blue is both excluded and covered"},
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			errs := parity.Check(universe, c.set)
			if len(errs) == 0 {
				t.Fatalf("Check found nothing wrong with %+v", c.set)
			}
			all := joined(errs)
			for _, want := range c.want {
				if !strings.Contains(all, want) {
					t.Errorf("the failure does not say %q:\n%s", want, all)
				}
			}
		})
	}
}

func TestCheckPassesACompleteSet(t *testing.T) {
	errs := parity.Check([]string{"Red", "Green", "Blue"}, parity.Set{
		Surface:  "Colour",
		Covered:  parity.Covers{"Red", "Green"},
		Excludes: []parity.Excluded{{Name: "Blue", Why: "refused on both backends"}},
	})
	if len(errs) != 0 {
		t.Fatalf("a complete set failed: %s", joined(errs))
	}
}

// The gap list is in the universe's order, because that is the order the
// source declares the members in and the order a reader scans.
func TestCheckReportsGapsInDeclarationOrder(t *testing.T) {
	errs := parity.Check([]string{"Red", "Green", "Blue"},
		parity.Set{Surface: "Colour", Covered: parity.Covers{"Green"}})
	if len(errs) != 1 {
		t.Fatalf("want one failure, got %s", joined(errs))
	}
	if !strings.Contains(errs[0].Error(), "Red, Blue") {
		t.Errorf("gaps are not in declaration order: %v", errs[0])
	}
}

func TestACeilingWithoutASourceIsRefused(t *testing.T) {
	if err := (parity.Ceiling{ULP: 4}).Validate(); err == nil {
		t.Fatal("a ULP ceiling naming no primitive was accepted; " +
			"a tolerance without a source is a number somebody tuned")
	}
	if err := (parity.Ceiling{}).Validate(); err != nil {
		t.Fatalf("the exact ceiling needs no source: %v", err)
	}
	if err := (parity.Ceiling{ULP: 4, Abs: 1, Why: "both"}).Validate(); err == nil {
		t.Fatal("a ceiling stating both a ULP and an absolute bound was accepted")
	}
	c := parity.Ceiling{Abs: 1e-6, Why: "the softmax exponential"}
	if err := c.Validate(); err != nil {
		t.Fatalf("a sourced absolute ceiling was refused: %v", err)
	}
	if !strings.Contains(c.String(), "softmax") {
		t.Errorf("the ceiling does not carry its source into a failure: %s", c)
	}
	if !(parity.Ceiling{}).Exact() {
		t.Error("the zero ceiling is not exact")
	}
}

func joined(errs []error) string {
	var b strings.Builder
	for _, err := range errs {
		b.WriteString(err.Error())
		b.WriteString("\n")
	}
	return b.String()
}

// The matrix gate routes a qualified claim to its surface, and says so when it
// cannot.
func TestCheckMatrixRoutesByQualifiedName(t *testing.T) {
	surfaces := []parity.Surface{
		{Name: "Colour", Members: []string{"Red", "Green"}},
		{Name: "Op", Members: []string{"OpKeep"}},
	}

	errs := parity.CheckMatrix(surfaces,
		parity.Covers{"Colour.Red", "Colour.Green", "Op.OpKeep"}, nil)
	if len(errs) != 0 {
		t.Fatalf("a complete matrix failed: %s", joined(errs))
	}

	// A member of one surface claimed under another is a gap in both, and the
	// unqualified form would have hidden it as coverage.
	errs = parity.CheckMatrix(surfaces,
		parity.Covers{"Colour.Red", "Colour.OpKeep", "Op.OpKeep"}, nil)
	all := joined(errs)
	for _, want := range []string{"Colour: OpKeep is claimed", "Green"} {
		if !strings.Contains(all, want) {
			t.Errorf("the failure does not say %q:\n%s", want, all)
		}
	}

	for _, c := range []struct{ claim, want string }{
		{"Red", "is not a qualified name"},
		{"Nothing.Red", `does not gate`},
	} {
		if got := joined(parity.CheckMatrix(surfaces, parity.Covers{c.claim}, nil)); !strings.Contains(got, c.want) {
			t.Errorf("claiming %q does not say %q:\n%s", c.claim, c.want, got)
		}
	}

	if got := joined(parity.CheckMatrix([]parity.Surface{
		{Name: "Colour", Members: []string{"Red"}},
		{Name: "Colour", Members: []string{"Red"}},
	}, nil, nil)); !strings.Contains(got, "declared twice") {
		t.Errorf("a surface declared twice is accepted: %s", got)
	}
}
