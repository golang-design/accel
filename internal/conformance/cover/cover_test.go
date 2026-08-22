// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package cover_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.design/x/accel/internal/conformance/cover"
)

func TestParseProfile(t *testing.T) {
	const profile = `mode: set
golang.design/x/accel/a.go:10.20,12.3 2 1
golang.design/x/accel/a.go:14.30,16.4 1 0
`
	blocks, err := cover.ParseProfile(strings.NewReader(profile))
	if err != nil {
		t.Fatalf("ParseProfile: %v", err)
	}
	if len(blocks) != 2 {
		t.Fatalf("parsed %d blocks, want 2", len(blocks))
	}
	want := cover.Block{
		File: "golang.design/x/accel/a.go", StartLine: 10, StartCol: 20,
		EndLine: 12, EndCol: 3, Statements: 2, Count: 1,
	}
	if blocks[0] != want {
		t.Errorf("block 0 = %+v, want %+v", blocks[0], want)
	}
}

func TestParseProfileRejectsGarbage(t *testing.T) {
	for _, tc := range []struct{ name, in string }{
		{"no mode line", "golang.design/x/accel/a.go:1.1,2.2 1 1\n"},
		{"no colon", "mode: set\nnonsense\n"},
		{"wrong field count", "mode: set\na.go:1.1,2.2 1\n"},
		{"no span comma", "mode: set\na.go:1.1 1 1\n"},
		{"bad position", "mode: set\na.go:x.y,2.2 1 1\n"},
		{"bad line number", "mode: set\na.go:1.1,2.2 x 1\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := cover.ParseProfile(strings.NewReader(tc.in)); err == nil {
				t.Error("was accepted")
			}
		})
	}
}

// TestMergeUnionsTheBinaries is the bug this package had on its first run.
//
// Running the suite with -coverpkg makes every test binary report every
// package, so one statement arrives once per binary. Summing them counted each
// statement several times and reported a statement covered by one binary and
// not another as both covered and uncovered, which put a 99% package at 60%.
func TestMergeUnionsTheBinaries(t *testing.T) {
	b := func(count int) cover.Block {
		return cover.Block{File: "a.go", StartLine: 1, StartCol: 1, EndLine: 2, EndCol: 2, Statements: 3, Count: count}
	}
	merged := cover.Merge([]cover.Block{b(0), b(2), b(0)})
	if len(merged) != 1 {
		t.Fatalf("merged to %d blocks, want 1", len(merged))
	}
	if merged[0].Count != 2 {
		t.Errorf("count is %d, want 2: a statement is covered when any binary ran it", merged[0].Count)
	}
	if merged[0].Statements != 3 {
		t.Errorf("statements is %d, want 3: merging must not multiply the count", merged[0].Statements)
	}

	// Distinct blocks stay distinct.
	other := cover.Block{File: "a.go", StartLine: 9, EndLine: 10, Statements: 1}
	if got := cover.Merge([]cover.Block{b(1), other}); len(got) != 2 {
		t.Errorf("distinct blocks merged into %d", len(got))
	}
}

// TestStubSpans covers spec 011 section 10.1's rule, which is deliberately
// narrow: a body of exactly panic(ErrNotImplemented) and nothing else.
func TestStubSpans(t *testing.T) {
	const src = `package p

func Stub() int { panic(ErrNotImplemented) }

func MultilineStub() int {
	panic(ErrNotImplemented)
}

func ValidatesFirst(n int) int {
	if n < 0 {
		return 0
	}
	panic(ErrNotImplemented)
}

func OtherPanic() int { panic("nope") }

func WrongSentinel() int { panic(ErrUnsupported) }

func PanicsWithACall() int { panic(fmt.Errorf("x")) }

func Real() int { return 1 }

type T struct{}

func (T) Method() int { panic(ErrNotImplemented) }
`
	spans, err := cover.StubSpans("p.go", []byte(src))
	if err != nil {
		t.Fatalf("StubSpans: %v", err)
	}
	// Stub, MultilineStub, and Method. Nothing else: a function that validates
	// its arguments before panicking has behaviour somebody should test, and a
	// different panic is not a design-stage stub at all.
	if len(spans) != 3 {
		t.Fatalf("found %d stubs, want 3: %+v", len(spans), spans)
	}
	if spans[0].Start != 3 || spans[0].End != 3 {
		t.Errorf("the one-line stub spans %+v, want line 3", spans[0])
	}
	if spans[1].Start != 5 || spans[1].End != 7 {
		t.Errorf("the multiline stub spans %+v, want lines 5 to 7", spans[1])
	}
}

func TestStubSpansRejectsUnparseableSource(t *testing.T) {
	if _, err := cover.StubSpans("p.go", []byte("package")); err == nil {
		t.Error("unparseable source was accepted")
	}
}

// TestSummarizeExcludesStubs is the whole point: a package of stubs plus one
// real function reports the real function's coverage, and says how many
// declarations it left out.
func TestSummarizeExcludesStubs(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "pkg")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	const src = `package p

var ErrNotImplemented = errors.New("x")

func Stub() int { panic(ErrNotImplemented) }

func Real(n int) int {
	if n > 0 {
		return n
	}
	return -n
}
`
	if err := os.WriteFile(filepath.Join(dir, "p.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}

	// Line 5 is the stub; lines 7 to 12 are Real, in two blocks, one covered.
	profile := strings.Join([]string{
		"mode: set",
		"example.com/m/pkg/p.go:5.17,5.43 1 0",
		"example.com/m/pkg/p.go:7.22,9.11 1 1",
		"example.com/m/pkg/p.go:11.2,11.11 1 0",
		"",
	}, "\n")

	blocks, err := cover.ParseProfile(strings.NewReader(profile))
	if err != nil {
		t.Fatal(err)
	}
	// The profile names files by import path; the module prefix is stripped and
	// the rest read from root.
	report, err := cover.Summarize(blocks, "example.com/m", root)
	if err != nil {
		t.Fatalf("Summarize: %v", err)
	}
	if len(report.Packages) != 1 {
		t.Fatalf("reported %d packages, want 1", len(report.Packages))
	}
	got := report.Packages[0]
	if got.Total != 2 {
		t.Errorf("Total = %d, want 2: the stub's statement is excluded", got.Total)
	}
	if got.Covered != 1 {
		t.Errorf("Covered = %d, want 1", got.Covered)
	}
	if got.Stubs != 1 || got.Excluded != 1 {
		t.Errorf("Stubs = %d and Excluded = %d, want 1 and 1", got.Stubs, got.Excluded)
	}
	if got.Percent != 50 {
		t.Errorf("Percent = %v, want 50", got.Percent)
	}

	// The exclusion is reported, never silent.
	var out strings.Builder
	report.Print(&out)
	if !strings.Contains(out.String(), "1 design-stage stubs excluded") {
		t.Errorf("the report does not say what it excluded:\n%s", out.String())
	}

	if fails := report.Failures(90); len(fails) != 1 {
		t.Errorf("a 50%% package produced %d failures, want 1", len(fails))
	}
	if fails := report.Failures(10); len(fails) != 0 {
		t.Errorf("a 50%% package failed a 10%% gate: %v", fails)
	}
}

// TestSummarizeReportsAllStubPackages checks that a package which is nothing
// but stubs is named rather than silently scoring 100%.
func TestSummarizeReportsAllStubPackages(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "pkg")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	const src = `package p

func A() int { panic(ErrNotImplemented) }
`
	if err := os.WriteFile(filepath.Join(dir, "p.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	blocks, err := cover.ParseProfile(strings.NewReader(
		"mode: set\nexample.com/m/pkg/p.go:3.15,3.41 1 0\n"))
	if err != nil {
		t.Fatal(err)
	}
	report, err := cover.Summarize(blocks, "example.com/m", root)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Packages) != 0 {
		t.Errorf("a package of nothing but stubs reported a percentage: %+v", report.Packages)
	}
	if len(report.Missing) != 1 {
		t.Fatalf("Missing = %v, want the one all-stub package", report.Missing)
	}
	var out strings.Builder
	report.Print(&out)
	if !strings.Contains(out.String(), "every counted statement is a design-stage stub") {
		t.Errorf("the report hides an all-stub package:\n%s", out.String())
	}
}

func TestSummarizeReportsAMissingSource(t *testing.T) {
	blocks, err := cover.ParseProfile(strings.NewReader("mode: set\nexample.com/m/gone.go:1.1,2.2 1 1\n"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cover.Summarize(blocks, "example.com/m", t.TempDir()); err == nil {
		t.Error("a profile naming a file that is not there was accepted")
	}
}
