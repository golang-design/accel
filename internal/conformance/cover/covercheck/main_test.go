// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fixture writes a package and a profile over it, and returns the module root
// and the profile's path.
func fixture(t *testing.T, src, profile string) (root, profilePath string) {
	t.Helper()
	root = t.TempDir()
	dir := filepath.Join(root, "pkg")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "p.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	profilePath = filepath.Join(root, "cover.out")
	if err := os.WriteFile(profilePath, []byte(profile), 0o644); err != nil {
		t.Fatal(err)
	}
	return root, profilePath
}

const goodSource = `package p

func Real(n int) int {
	if n > 0 {
		return n
	}
	return -n
}
`

// TestPassingGateExitsZero covers the ordinary run.
func TestPassingGateExitsZero(t *testing.T) {
	root, prof := fixture(t, goodSource, "mode: set\n"+
		"example.com/m/pkg/p.go:3.22,5.11 1 1\n"+
		"example.com/m/pkg/p.go:7.2,7.11 1 1\n")

	var out, errOut bytes.Buffer
	code := run([]string{"-profile", prof, "-module", "example.com/m", "-root", root}, &out, &errOut)
	if code != 0 {
		t.Fatalf("exit %d\n%s\n%s", code, out.String(), errOut.String())
	}
	if !strings.Contains(out.String(), "every gated package is above 90%") {
		t.Errorf("stdout does not report the pass: %s", out.String())
	}
}

// TestFailingGateExitsOneAndNamesThePackage is what CI depends on. A gate that
// failed without saying which package would send a reader to read every
// package's number by hand.
func TestFailingGateExitsOneAndNamesThePackage(t *testing.T) {
	root, prof := fixture(t, goodSource, "mode: set\n"+
		"example.com/m/pkg/p.go:3.22,5.11 1 1\n"+
		"example.com/m/pkg/p.go:7.2,7.11 1 0\n")

	var out, errOut bytes.Buffer
	if code := run([]string{"-profile", prof, "-module", "example.com/m", "-root", root}, &out, &errOut); code != 1 {
		t.Fatalf("exit %d, want 1", code)
	}
	for _, want := range []string{"::error::", "example.com/m/pkg", "50.0%", "below the 90% gate"} {
		if !strings.Contains(errOut.String(), want) {
			t.Errorf("stderr does not carry %q: %s", want, errOut.String())
		}
	}
}

// TestSkipReportsWithoutGating covers the escape hatch, which is now empty by
// default: a package can be reported and not gated only when somebody names it
// on the command line, so an exemption appears in the workflow where a reviewer
// sees it rather than in a default nobody reads.
func TestSkipReportsWithoutGating(t *testing.T) {
	root, prof := fixture(t, goodSource, "mode: set\n"+
		"example.com/m/pkg/p.go:3.22,5.11 1 1\n"+
		"example.com/m/pkg/p.go:7.2,7.11 1 0\n")

	var out, errOut bytes.Buffer
	code := run([]string{"-profile", prof, "-module", "example.com/m", "-root", root, "-skip", "pkg"}, &out, &errOut)
	if code != 0 {
		t.Fatalf("exit %d with the package skipped: %s", code, errOut.String())
	}
	// Skipped means not gated, never not reported.
	if !strings.Contains(out.String(), "example.com/m/pkg") {
		t.Errorf("a skipped package vanished from the report: %s", out.String())
	}
}

// TestStubsAreExcludedAndCounted is spec 011 section 10.1's rule, end to end.
func TestStubsAreExcludedAndCounted(t *testing.T) {
	src := `package p

var ErrNotImplemented = errors.New("x")

func Stub() int { panic(ErrNotImplemented) }

func Real() int { return 1 }
`
	root, prof := fixture(t, src, "mode: set\n"+
		"example.com/m/pkg/p.go:5.17,5.43 1 0\n"+
		"example.com/m/pkg/p.go:7.17,7.28 1 1\n")

	var out, errOut bytes.Buffer
	if code := run([]string{"-profile", prof, "-module", "example.com/m", "-root", root}, &out, &errOut); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "1 design-stage stubs excluded") {
		t.Errorf("the exclusion is not reported, so a package that scores well because most of "+
			"it does not exist yet would look like a pass: %s", out.String())
	}
}

// TestMissingProfileExitsOne covers the case where the measure step did not run.
func TestMissingProfileExitsOne(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run([]string{"-profile", filepath.Join(t.TempDir(), "nope.out")}, &out, &errOut); code != 1 {
		t.Errorf("exit %d, want 1", code)
	}
	if !strings.Contains(errOut.String(), "covercheck:") {
		t.Errorf("stderr does not name the tool: %s", errOut.String())
	}
}

// TestMalformedProfileExitsOne covers a truncated or corrupted profile, which a
// cancelled test run produces.
func TestMalformedProfileExitsOne(t *testing.T) {
	root, prof := fixture(t, goodSource, "not a profile\n")
	var out, errOut bytes.Buffer
	if code := run([]string{"-profile", prof, "-module", "example.com/m", "-root", root}, &out, &errOut); code != 1 {
		t.Errorf("exit %d, want 1", code)
	}
}

// TestUnreadableSourceExitsOne covers a profile naming a file that is not
// there, which happens when a profile outlives the tree it was taken over.
func TestUnreadableSourceExitsOne(t *testing.T) {
	root, prof := fixture(t, goodSource, "mode: set\nexample.com/m/gone/p.go:1.1,2.2 1 1\n")
	var out, errOut bytes.Buffer
	if code := run([]string{"-profile", prof, "-module", "example.com/m", "-root", root}, &out, &errOut); code != 1 {
		t.Errorf("exit %d, want 1", code)
	}
}

// TestBadFlagsExitTwo separates a usage error from a failing gate.
func TestBadFlagsExitTwo(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run([]string{"-nosuchflag"}, &out, &errOut); code != 2 {
		t.Errorf("exit %d, want 2", code)
	}
}

// TestSkippedPrefixMatching covers the prefix rule, since a skip that matched
// too much would silently exempt packages nobody meant to.
func TestSkippedPrefixMatching(t *testing.T) {
	root, prof := fixture(t, goodSource, "mode: set\n"+
		"example.com/m/pkg/p.go:3.22,5.11 1 1\n"+
		"example.com/m/pkg/p.go:7.2,7.11 1 0\n")

	var out, errOut bytes.Buffer
	// A prefix that does not match must not exempt anything.
	if code := run([]string{"-profile", prof, "-module", "example.com/m", "-root", root,
		"-skip", "other"}, &out, &errOut); code != 1 {
		t.Errorf("exit %d with a non-matching skip, want 1", code)
	}
	// An empty entry in the list must not match everything.
	out.Reset()
	errOut.Reset()
	if code := run([]string{"-profile", prof, "-module", "example.com/m", "-root", root,
		"-skip", ", ,"}, &out, &errOut); code != 1 {
		t.Errorf("exit %d with an empty skip list, want 1", code)
	}
}
