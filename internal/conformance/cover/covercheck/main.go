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
	"io"
	"os"
	"strings"

	"golang.design/x/accel/internal/conformance/cover"
)

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

// run is main with its process wiring passed in, so this command is testable in
// process like any other package.
//
// It was not, at first. The harness was exempted from its own gate on the
// grounds that test infrastructure is not production code, which is a weak
// reason for a package that had already shipped a bug: this tool reported a 99%
// package at 60% because it summed duplicate coverage blocks. A gate nobody
// checks is a gate, and the argument for exempting the thing that runs it is
// the worst place to make an exception.
func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("covercheck", flag.ContinueOnError)
	fs.SetOutput(stderr)
	profile := fs.String("profile", "cover.out", "coverage profile to read")
	module := fs.String("module", "golang.design/x/accel", "module path to strip from profile file names")
	root := fs.String("root", ".", "directory the module's files live under")
	minPercent := fs.Float64("min", 90, "per-package statement coverage gate, exclusive")
	skip := fs.String("skip", "", "comma-separated import-path prefixes to report but not gate")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	f, err := os.Open(*profile)
	if err != nil {
		fmt.Fprintf(stderr, "covercheck: %v\n", err)
		return 1
	}
	defer f.Close()

	blocks, err := cover.ParseProfile(f)
	if err != nil {
		fmt.Fprintf(stderr, "covercheck: %v\n", err)
		return 1
	}
	report, err := cover.Summarize(blocks, *module, *root)
	if err != nil {
		fmt.Fprintf(stderr, "covercheck: %v\n", err)
		return 1
	}

	report.Print(stdout)

	var gated cover.Report
	gated.Missing = report.Missing
	for _, p := range report.Packages {
		if !skipped(p.Package, *module, *skip) {
			gated.Packages = append(gated.Packages, p)
		}
	}

	failures := gated.Failures(*minPercent)
	if len(failures) == 0 {
		fmt.Fprintf(stdout, "\nevery gated package is above %.0f%%\n", *minPercent)
		return 0
	}
	fmt.Fprintln(stderr)
	for _, f := range failures {
		fmt.Fprintf(stderr, "::error::%s\n", f)
	}
	return 1
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
