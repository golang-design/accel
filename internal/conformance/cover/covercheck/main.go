// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Command covercheck reports per-package coverage and fails below the gate.
//
// Usage:
//
//	go test -coverprofile=cover.out -coverpkg=./... ./...
//	go run ./internal/conformance/cover/covercheck -profile=cover.out
//
// The gate is spec 011 section 10's: greater than 90% statement coverage for
// each production package, reported independently rather than as one repository
// average, with section 10.1's checked exclusions applied.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"golang.design/x/accel/internal/conformance/cover"
)

func main() {
	profile := flag.String("profile", "cover.out", "coverage profile to read")
	module := flag.String("module", "golang.design/x/accel", "module path to strip from profile file names")
	root := flag.String("root", ".", "directory the module's files live under")
	minPercent := flag.Float64("min", 90, "per-package statement coverage gate, exclusive")
	skip := flag.String("skip", "internal/conformance", "comma-separated import-path prefixes to report but not gate")
	flag.Parse()

	f, err := os.Open(*profile)
	if err != nil {
		fatal(err)
	}
	defer f.Close()

	blocks, err := cover.ParseProfile(f)
	if err != nil {
		fatal(err)
	}
	report, err := cover.Summarize(blocks, *module, *root)
	if err != nil {
		fatal(err)
	}

	report.Print(os.Stdout)

	// The harness is test infrastructure, so it is reported and not gated: its
	// own coverage is a statement about how much of the harness the tests
	// happen to use, which is not what the gate is for.
	var gated cover.Report
	gated.Missing = report.Missing
	for _, p := range report.Packages {
		if !skipped(p.Package, *module, *skip) {
			gated.Packages = append(gated.Packages, p)
		}
	}

	failures := gated.Failures(*minPercent)
	if len(failures) == 0 {
		fmt.Printf("\nevery gated package is above %.0f%%\n", *minPercent)
		return
	}
	fmt.Fprintln(os.Stderr)
	for _, f := range failures {
		fmt.Fprintf(os.Stderr, "::error::%s\n", f)
	}
	os.Exit(1)
}

func skipped(pkg, module, prefixes string) bool {
	rel := strings.TrimPrefix(pkg, module+"/")
	for p := range strings.SplitSeq(prefixes, ",") {
		if p = strings.TrimSpace(p); p != "" && strings.HasPrefix(rel, p) {
			return true
		}
	}
	return false
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "covercheck: %v\n", err)
	os.Exit(1)
}
