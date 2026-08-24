// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel

import (
	"testing"

	"golang.design/x/accel/internal/driver"
)

// The two format enumerations name the same set, one for one.
//
// A plan carries [driver.Format] and the public API carries [Format], and the
// mapping between them is written out as a switch. Two lists that happen to
// agree are one insertion away from shifting every value after the insertion,
// and a shifted format is a plausible image rather than an error -- so the
// agreement is asserted over the whole enumeration rather than trusted.
//
// The names are compared as well as the values, because a mapping that is
// bijective and wrong -- RGBA8Unorm to driver.BGRA8Unorm, and back -- passes a
// bijection check on its own.
func TestEveryFormatHasOnePlanSpelling(t *testing.T) {
	seen := map[driver.Format]Format{}
	for f := range formatTable {
		p := f.plan()
		if p == driver.FormatInvalid {
			t.Errorf("%v has no plan spelling, so an attachment of it would reach a "+
				"backend with no format at all", f)
			continue
		}
		if p.String() != f.String() {
			t.Errorf("%v maps to %v; the two spellings name different formats, which is "+
				"a mistranslation a bijection check cannot see", f, p)
		}
		if prev, ok := seen[p]; ok {
			t.Errorf("%v and %v both map to %v, so one of them decodes as the other", prev, f, p)
		}
		seen[p] = f
	}
	for _, p := range driver.Formats {
		if _, ok := seen[p]; !ok {
			t.Errorf("no public format maps to %v, so a plan could never name it", p)
		}
	}
	if len(seen) != len(driver.Formats) {
		t.Errorf("%d public formats map onto %d plan formats", len(seen), len(driver.Formats))
	}
}

// A format outside the table has no plan spelling, and says so as a value
// rather than by guessing at one.
func TestAnUnknownFormatHasNoPlanSpelling(t *testing.T) {
	for _, f := range []Format{FormatInvalid, Format(len(formatTable) + 100)} {
		if got := f.plan(); got != driver.FormatInvalid {
			t.Errorf("%v maps to %v, want FormatInvalid: a format a backend cannot name "+
				"must not decode as one it can", f, got)
		}
	}
}
