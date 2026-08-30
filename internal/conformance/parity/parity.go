// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package parity is the CPU/Metal agreement matrix of
// specs/062-backend-parity.md: a registry of cases, and a gate that fails when
// a member of a public enumeration or operator set has no case in it.
//
// The split this package exists to enforce is between the gate and the run.
// The gate compares two lists of names and needs no device, so it compiles and
// runs on every platform; only the half that opens a GPU is build-tagged. The
// corpus differential got this backwards -- its completeness check sits behind
// //go:build darwin, so an unlisted kernel added on Linux is invisible until
// somebody runs the suite on a Mac -- and repeating that shape here would put
// the same hole in every surface at once.
//
// This is test infrastructure. It decides nothing about what a device means:
// it enumerates, it compares, and it explains a failure.
package parity

import (
	"fmt"
	"sort"
	"strings"
)

// Ceiling is how far two backends may differ on one case.
//
// specs/008-numerics.md gives the vocabulary and this reuses it rather than
// inventing a second one. The zero value is Exact, which is the right default:
// two lowerings of one IR over integer or copy-only paths must agree bit for
// bit, and a case that needs more says so and says why.
type Ceiling struct {
	// ULP is the maximum unit-in-the-last-place distance, when positive.
	ULP uint64

	// Abs is the maximum absolute difference, when positive. Separate from ULP
	// because the two answer different questions: ULP near a zero crossing is
	// meaningless, and an absolute bound across many binades is.
	Abs float64

	// Why names the bounded primitive the ceiling comes from. Required whenever
	// the ceiling is not exact, because a tolerance without a source is a
	// number somebody tuned until the test passed.
	Why string
}

// Exact reports whether the ceiling demands bit-for-bit agreement.
func (c Ceiling) Exact() bool { return c.ULP == 0 && c.Abs == 0 }

// String is the ceiling as it appears in a failure.
func (c Ceiling) String() string {
	switch {
	case c.Exact():
		return "exact: no bounded primitive in this case, so the two backends must agree bit for bit"
	case c.Abs > 0:
		return fmt.Sprintf("within %g absolute, from %s", c.Abs, c.Why)
	default:
		return fmt.Sprintf("within %d ULP, from %s", c.ULP, c.Why)
	}
}

// Validate refuses a ceiling that states a tolerance and no source.
func (c Ceiling) Validate() error {
	if !c.Exact() && strings.TrimSpace(c.Why) == "" {
		return fmt.Errorf("a ceiling of %v states no source; "+
			"specs/062-backend-parity.md section 4 requires the bounded primitive be named",
			c)
	}
	if c.ULP > 0 && c.Abs > 0 {
		return fmt.Errorf("a ceiling states both %d ULP and %g absolute; "+
			"the two answer different questions and a case picks one", c.ULP, c.Abs)
	}
	return nil
}

// Covers is what one case says it covers.
//
// A case names universe members rather than being named by them, because one
// recorded graph often covers several: a blend pass covers a factor and an op
// together, and splitting it into two cases would run the same pass twice.
type Covers []string

// Excluded is a universe member that cannot have a case, and the reason.
//
// The reason is required. A member that is untested for a good reason and one
// that is untested by oversight look identical in a coverage table, and the
// only difference a reader can act on is whether somebody wrote the reason
// down where the next reader looks.
type Excluded struct {
	Name string
	Why  string
}

// Set is one surface's registry: what the cases cover, and what is excluded.
type Set struct {
	// Surface names the enumeration or operator set, for failure text.
	Surface string

	Covered  Covers
	Excludes []Excluded
}

// Add records that a case covers these members.
func (s *Set) Add(names ...string) { s.Covered = append(s.Covered, names...) }

// Exclude records a member that cannot have a case, and why.
func (s *Set) Exclude(name, why string) {
	s.Excludes = append(s.Excludes, Excluded{Name: name, Why: why})
}

// Gap is one universe member with no case and no exclusion.
type Gap struct{ Name string }

// Check compares a universe against a set and returns everything wrong with
// the pairing: members with no case, exclusions with no reason, and names in
// the set that are in no universe.
//
// The third kind matters as much as the first. A case covering "CompareLesser"
// is covering nothing -- the constant is spelled CompareLess -- and without
// this check it would read in the table as coverage while the real member sits
// in the gap list under a name nobody looked at twice.
func Check(universe []string, s Set) []error {
	var errs []error

	inUniverse := make(map[string]bool, len(universe))
	for _, n := range universe {
		inUniverse[n] = true
	}

	claimed := make(map[string]string, len(s.Covered))
	for _, n := range s.Covered {
		claimed[n] = "a case"
	}
	for _, e := range s.Excludes {
		if strings.TrimSpace(e.Why) == "" {
			errs = append(errs, fmt.Errorf(
				"%s: %s is excluded with no reason; an exclusion states why, "+
					"or it is a gap wearing a different word", s.Surface, e.Name))
			continue
		}
		if prev, dup := claimed[e.Name]; dup {
			errs = append(errs, fmt.Errorf(
				"%s: %s is both excluded and covered by %s; it is one or the other",
				s.Surface, e.Name, prev))
			continue
		}
		claimed[e.Name] = "an exclusion"
	}

	var unknown []string
	for n := range claimed {
		if !inUniverse[n] {
			unknown = append(unknown, n)
		}
	}
	sort.Strings(unknown)
	for _, n := range unknown {
		errs = append(errs, fmt.Errorf(
			"%s: %s is claimed by %s and is not a member of the surface; "+
				"check the spelling, because a misspelled claim reads as coverage",
			s.Surface, n, claimed[n]))
	}

	var gaps []string
	for _, n := range universe {
		if _, ok := claimed[n]; !ok {
			gaps = append(gaps, n)
		}
	}
	if len(gaps) > 0 {
		errs = append(errs, fmt.Errorf(
			"%s has %d members and %d of them have no parity case: %s\n  "+
				"a member with no case has never been compared between the CPU backend "+
				"and Metal. Add a case, or exclude it with a reason: "+
				"specs/062-backend-parity.md section 3.1",
			s.Surface, len(universe), len(gaps), strings.Join(gaps, ", ")))
	}
	return errs
}
