// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Command accel-kernel compiles kernels written in the Go subset.
//
// It is run under go generate, and it is the only thing that produces the
// generated lowering a backend executes:
//
//	//go:generate go run golang.design/x/accel/cmd/accel-kernel ./...
//
// Compilation is ahead of time because type checking needs the package loaded
// with its import graph and the go tool present, which a deployed binary does
// not have. See specs/004-kernel-authoring.md.
//
// Usage:
//
//	accel-kernel [flags] [packages]
//
//	-check   verify the generated files rather than writing them
//	-C dir   resolve patterns in dir
//	-o name  the generated file's name inside each package
//
// With -check it writes nothing and exits non-zero if any generated file is not
// what the current source produces, naming the kernel and the input that moved.
// That is what CI runs, because a kernel edited without regenerating leaves a
// source and a generated file that disagree while both still compile.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"golang.design/x/accel/internal/kernelc"
)

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

// run is main with its process wiring passed in, so the command's own behaviour
// is testable in process.
//
// Without it the only way to check that -check exits non-zero on a stale file
// is to build and exec the binary, which tests the go tool as much as the
// command and reports nothing to coverage.
func run(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("accel-kernel", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var (
		check = fs.Bool("check", false, "verify the generated files rather than writing them")
		dir   = fs.String("C", ".", "resolve patterns in this directory")
		out   = fs.String("o", kernelc.GeneratedFile, "the generated file's name inside each package")
	)
	fs.Usage = func() {
		fmt.Fprintf(stderr, "usage: accel-kernel [flags] [packages]\n\n")
		fmt.Fprintf(stderr, "Compiles kernels marked //accel:kernel into a generated Go lowering.\n\n")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return 2
	}

	results, err := kernelc.Run(kernelc.Options{
		Dir:      *dir,
		Patterns: fs.Args(),
		Check:    *check,
		Out:      *out,
	})
	if err != nil {
		fmt.Fprintf(stderr, "accel-kernel: %v\n", err)
		return 1
	}

	stale := 0
	for _, r := range results {
		switch {
		case r.Stale != "":
			stale++
			fmt.Fprintf(stderr, "accel-kernel: %s\n", r.Stale)
		case r.Written:
			fmt.Fprintf(stdout, "accel-kernel: wrote %s (%v)\n", r.Path, r.Kernels)
		case *check:
			fmt.Fprintf(stdout, "accel-kernel: %s is fresh (%v)\n", r.Package, r.Kernels)
		}
	}
	if stale > 0 {
		return 1
	}
	return 0
}
