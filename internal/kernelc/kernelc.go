// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package kernelc drives the kernel compiler: load, check, digest, emit, write.
//
// It is the library `cmd/accel-kernel` is a flag parser around, so that the
// generator can be run from a test without exec'ing a binary and without a
// build step standing between a change and the thing it broke.
//
// Compilation is ahead of time and this is the tool that does it. Type checking
// needs the package loaded with its import graph and the go tool present, which
// a deployed binary does not have; that is the whole reason kernels are compiled
// by a generator rather than at startup. See specs/004-kernel-authoring.md.
package kernelc

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"golang.design/x/accel/internal/kernelc/emit"
	"golang.design/x/accel/internal/kernelc/front"
	"golang.design/x/accel/internal/kernelc/ir"
	"golang.org/x/tools/go/packages"
)

// GeneratedFile is the name a generated file takes inside its package.
//
// One file per package rather than one per kernel: the package is the unit
// go/packages loads and the unit a digest's intrinsic table applies to, and a
// file per kernel would make a table change touch every file in a diff nobody
// can read.
const GeneratedFile = "accel_kernels.go"

// Options configures a run.
type Options struct {
	// Dir is where patterns resolve, usually the module root.
	Dir string

	// Patterns name the packages to compile. Empty means the current directory.
	Patterns []string

	// Check verifies the generated files on disk instead of writing them. This
	// is what CI runs: a kernel edited without regenerating is a source and a
	// generated file that disagree, and nothing else notices, because both halves
	// compile.
	Check bool

	// Out overrides the generated file's name, for tests.
	Out string
}

// Result is one package's outcome.
type Result struct {
	Package string
	Path    string
	Kernels []string

	// Written is true when the file changed on disk. False in check mode, and
	// false when the file was already what it should be.
	Written bool

	// Stale is set in check mode when the file on disk is not what this run
	// would produce, with an explanation of which input moved.
	Stale string
}

// Run compiles every package the options name.
func Run(opts Options) ([]Result, error) {
	patterns := opts.Patterns
	if len(patterns) == 0 {
		patterns = []string{"."}
	}
	name := opts.Out
	if name == "" {
		name = GeneratedFile
	}

	pkgs, err := front.Load(opts.Dir, patterns...)
	if err != nil {
		return nil, err
	}

	var results []Result
	var problems []string
	for _, pkg := range pkgs {
		r, diags, err := compile(pkg, name, opts.Check)
		if err != nil {
			return nil, err
		}
		if len(diags) > 0 {
			problems = append(problems, diags.Error())
			continue
		}
		if r != nil {
			results = append(results, *r)
		}
	}
	if len(problems) > 0 {
		return results, fmt.Errorf("%s", strings.Join(problems, "\n"))
	}
	return results, nil
}

// compile handles one package.
func compile(pkg *packages.Package, name string, check bool) (*Result, front.Diagnostics, error) {
	kernels, diags := front.Check(pkg)
	if len(diags) > 0 {
		return nil, diags, nil
	}
	if len(kernels) == 0 {
		// A package with no kernels is not an error. A corpus is loaded by
		// pattern and most packages in one have none.
		return nil, nil, nil
	}

	dir, err := packageDir(pkg, name)
	if err != nil {
		return nil, nil, err
	}

	for _, k := range kernels {
		k.Digest = emit.Digest(k)
	}
	out, err := emit.Generate(emit.Package{Name: pkg.Types.Name(), Kernels: kernels})
	if err != nil {
		return nil, nil, err
	}

	path := filepath.Join(dir, name)
	r := &Result{Package: pkg.PkgPath, Path: path, Kernels: names(kernels)}

	existing, readErr := os.ReadFile(path)

	// Line endings are a property of the checkout, not of the content. A
	// .gitattributes pins these files to LF, and comparing normalized as well
	// means a checkout configured otherwise reports "fresh" rather than
	// reporting every kernel stale with a diff that looks identical.
	same := readErr == nil && bytes.Equal(normalizeEOL(existing), normalizeEOL(out))

	if check {
		if !same {
			r.Stale = explain(path, existing, out, readErr != nil, kernels)
		}
		return r, nil, nil
	}
	if same {
		return r, nil, nil
	}
	if err := os.WriteFile(path, out, 0o644); err != nil {
		return nil, nil, fmt.Errorf("accel: writing %s: %w", path, err)
	}
	r.Written = true
	return r, nil, nil
}

// explain says which input moved, rather than only that something did.
//
// "The generated file is stale" sends a reader to diff two files by eye. The
// digest's preimage is a list of exactly the inputs a generated file depends
// on, so the first line that differs is the answer, and printing it is the
// difference between a two-minute fix and a hunt.
func explain(path string, existing, want []byte, missing bool, kernels []*ir.Func) string {
	var b strings.Builder
	if missing {
		fmt.Fprintf(&b, "%s does not exist", path)
	} else {
		fmt.Fprintf(&b, "%s is not what the current source generates", path)
	}
	fmt.Fprintf(&b, "\n  re-run: go generate ./...")

	for _, k := range kernels {
		if bytes.Contains(existing, []byte(k.Digest)) {
			continue
		}
		fmt.Fprintf(&b, "\n  kernel %s has digest %s, which the file does not carry.", k.Name, k.Digest)
		fmt.Fprintf(&b, " Its inputs are:\n%s", indentLines(emit.Preimage(k), "    "))
	}
	return b.String()
}

// normalizeEOL removes carriage returns so a comparison is about text.
func normalizeEOL(b []byte) []byte { return bytes.ReplaceAll(b, []byte("\r\n"), []byte("\n")) }

func indentLines(s, prefix string) string {
	lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
	for i, l := range lines {
		lines[i] = prefix + l
	}
	return strings.Join(lines, "\n")
}

// packageDir finds where a package's files live.
//
// It prefers a file that is not the generated one, because in check mode the
// generated file may be the only thing present and a package whose only file is
// generated has nothing to generate from.
func packageDir(pkg *packages.Package, generated string) (string, error) {
	for _, f := range pkg.GoFiles {
		if filepath.Base(f) != generated {
			return filepath.Dir(f), nil
		}
	}
	if len(pkg.GoFiles) > 0 {
		return filepath.Dir(pkg.GoFiles[0]), nil
	}
	return "", fmt.Errorf("accel: package %s has no files on disk to generate beside", pkg.PkgPath)
}

func names(kernels []*ir.Func) []string {
	out := make([]string, len(kernels))
	for i, k := range kernels {
		out[i] = k.Name
	}
	sort.Strings(out)
	return out
}
