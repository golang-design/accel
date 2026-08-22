// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package cover_test

import (
	"strings"
	"testing"

	"golang.design/x/accel/internal/conformance/cover"
)

// FuzzParseProfile is a parser over text a tool wrote, which is exactly the
// input class where a crash is worst: this runs in CI after the tests, so a
// panic here turns a passing suite into a failed job with a stack trace that
// names the wrong package.
//
// A cancelled or interrupted `go test -coverprofile` leaves a truncated file,
// so malformed input is not hypothetical.
func FuzzParseProfile(f *testing.F) {
	for _, s := range []string{
		"mode: set\ngolang.design/x/accel/a.go:10.20,12.3 2 1\n",
		"mode: atomic\na.go:1.1,2.2 1 0\n",
		"mode: set\n",
		"",
		"a.go:1.1,2.2 1 1\n",
		"mode: set\nnonsense\n",
		"mode: set\na.go:1.1,2.2 1\n",
		"mode: set\na.go:-1.-1,-2.-2 -1 -1\n",
		"mode: set\n:::: 1 1\n",
	} {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, profile string) {
		blocks, err := cover.ParseProfile(strings.NewReader(profile))
		if err != nil {
			if blocks != nil {
				t.Fatalf("a rejected profile still produced %d blocks", len(blocks))
			}
			return
		}

		// It parsed, so every block must be usable: Merge and the summary
		// arithmetic assume a block describes a range.
		for i, b := range blocks {
			if b.File == "" {
				t.Fatalf("block %d has no file", i)
			}
			if b.Statements < 0 {
				t.Fatalf("block %d claims %d statements", i, b.Statements)
			}
		}

		// Merging is idempotent and never invents or loses a block.
		once := cover.Merge(blocks)
		twice := cover.Merge(once)
		if len(once) != len(twice) {
			t.Fatalf("merging twice changed the count: %d then %d", len(once), len(twice))
		}
		if len(once) > len(blocks) {
			t.Fatalf("merging invented blocks: %d from %d", len(once), len(blocks))
		}
	})
}

// FuzzStubSpans is the other parser: it reads Go source and decides which
// declarations spec 011 section 10.1 excludes from the coverage gate. A crash
// here fails the gate rather than reporting it, and a rule that over-matches
// silently excuses real code.
func FuzzStubSpans(f *testing.F) {
	for _, s := range []string{
		"package p\n\nfunc A() int { panic(ErrNotImplemented) }\n",
		"package p\n\nfunc A() int {\n\tpanic(ErrNotImplemented)\n}\n",
		"package p\n\nfunc A() int { panic(\"x\") }\n",
		"package p\n\nfunc A(n int) int {\n\tif n < 0 {\n\t\treturn 0\n\t}\n\tpanic(ErrNotImplemented)\n}\n",
		"package p\n\ntype T struct{}\n\nfunc (T) M() { panic(ErrNotImplemented) }\n",
		"package",
		"",
	} {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, src string) {
		spans, err := cover.StubSpans("p.go", []byte(src))
		if err != nil {
			if spans != nil {
				t.Fatal("a rejected file still produced spans")
			}
			return
		}
		for i, s := range spans {
			if s.Start <= 0 || s.End < s.Start {
				t.Fatalf("span %d is %+v, which is not a line range", i, s)
			}
		}
		// The rule is narrow on purpose. A file with no panic at all can have no
		// stub, and a rule that drifted wider would silently excuse real code
		// from the gate.
		if !strings.Contains(src, "panic") && len(spans) > 0 {
			t.Fatalf("%d stubs found in a file with no panic:\n%s", len(spans), src)
		}
	})
}
