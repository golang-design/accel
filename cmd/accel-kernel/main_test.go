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

func repoRoot(t *testing.T) string {
	t.Helper()
	abs, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return abs
}

// TestCheckPassesOnTheCommittedTree is the invocation CI runs.
func TestCheckPassesOnTheCommittedTree(t *testing.T) {
	var out, errOut bytes.Buffer
	code := run([]string{"-C", repoRoot(t), "-check", "./internal/kernels"}, &out, &errOut)
	if code != 0 {
		t.Fatalf("exit %d\nstdout: %s\nstderr: %s", code, out.String(), errOut.String())
	}
	if !strings.Contains(out.String(), "is fresh") {
		t.Errorf("stdout does not report freshness: %s", out.String())
	}
}

// TestCheckFailsOnAnEditedKernel is the reason the flag exists: a kernel edited
// without regenerating leaves a source and a generated file that disagree, and
// nothing else notices because both still compile.
func TestCheckFailsOnAnEditedKernel(t *testing.T) {
	dir := throwawayModule(t)

	var out, errOut bytes.Buffer
	if code := run([]string{"-C", dir, "./kernels"}, &out, &errOut); code != 0 {
		t.Fatalf("generating: exit %d, %s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "wrote") {
		t.Errorf("stdout does not report the write: %s", out.String())
	}

	src := filepath.Join(dir, "kernels", "scale.go")
	b, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(src, []byte(strings.Replace(string(b), "* 2", "* 3", 1)), 0o644); err != nil {
		t.Fatal(err)
	}

	out.Reset()
	errOut.Reset()
	if code := run([]string{"-C", dir, "-check", "./kernels"}, &out, &errOut); code == 0 {
		t.Fatal("an edited kernel passed the freshness check")
	}
	if !strings.Contains(errOut.String(), "go generate") {
		t.Errorf("stderr does not say how to fix it: %s", errOut.String())
	}
}

// TestRejectedKernelExitsNonZero checks that a subset error reaches the shell,
// since go generate reports nothing itself.
func TestRejectedKernelExitsNonZero(t *testing.T) {
	dir := throwawayModule(t)
	src := filepath.Join(dir, "kernels", "scale.go")
	b, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	broken := strings.Replace(string(b), "i := t.GlobalID().X",
		"i := t.GlobalID().X\n\tswitch {\n\t}", 1)
	if err := os.WriteFile(src, []byte(broken), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errOut bytes.Buffer
	if code := run([]string{"-C", dir, "./kernels"}, &out, &errOut); code != 1 {
		t.Fatalf("exit %d, want 1", code)
	}
	if !strings.Contains(errOut.String(), "outside the closed IR node set") {
		t.Errorf("stderr does not carry the rejection: %s", errOut.String())
	}
}

// TestBadFlagsExitTwo separates a usage error from a compilation failure, so a
// script can tell "you called me wrong" from "your kernel is wrong".
func TestBadFlagsExitTwo(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run([]string{"-nosuchflag"}, &out, &errOut); code != 2 {
		t.Errorf("exit %d, want 2", code)
	}
	errOut.Reset()
	if code := run([]string{"-h"}, &out, &errOut); code != 2 {
		t.Errorf("exit %d for -h, want 2", code)
	}
	if !strings.Contains(errOut.String(), "usage: accel-kernel") {
		t.Errorf("no usage on stderr: %s", errOut.String())
	}
}

// TestUnloadablePatternExitsOne covers a typo in a go:generate line.
func TestUnloadablePatternExitsOne(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := run([]string{"-C", repoRoot(t), "./nosuchpackage"}, &out, &errOut); code != 1 {
		t.Errorf("exit %d, want 1", code)
	}
}

// TestCustomOutputName covers the -o flag, which exists so a package that
// already has a file by the default name can still be generated into.
func TestCustomOutputName(t *testing.T) {
	dir := throwawayModule(t)
	var out, errOut bytes.Buffer
	if code := run([]string{"-C", dir, "-o", "kernels_gen.go", "./kernels"}, &out, &errOut); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	if _, err := os.Stat(filepath.Join(dir, "kernels", "kernels_gen.go")); err != nil {
		t.Errorf("the named file was not written: %v", err)
	}
}

// throwawayModule holds a copy of the corpus kernel, so a test may edit it
// without leaving the repository in a state that depends on cleanup.
func throwawayModule(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	repo := repoRoot(t)

	write := func(path, content string) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	write(filepath.Join(dir, "go.mod"), moduleFile(t, repo, "example.com/cmdtest"))
	if b, err := os.ReadFile(filepath.Join(repo, "go.sum")); err == nil {
		write(filepath.Join(dir, "go.sum"), string(b))
	}

	src, err := os.ReadFile(filepath.Join(repo, "internal", "kernels", "scale.go"))
	if err != nil {
		t.Fatal(err)
	}
	body := strings.Replace(string(src), "package kernels", "package kernels", 1)
	body = strings.Replace(body,
		"//go:generate go run golang.design/x/accel/cmd/accel-kernel -C ../.. ./internal/kernels\n", "", 1)
	write(filepath.Join(dir, "kernels", "scale.go"), body)
	return dir
}

// moduleFile builds the throwaway module's go.mod from the repository's own.
//
// Derived rather than written, because a hand-written require line goes stale
// the moment accel gains a dependency, and the failure is opaque: the generator
// reports "updates to go.mod needed" from inside a temp directory the reader
// cannot see. Taking the repository's requirements verbatim means the throwaway
// module has exactly what the replace target needs, always.
func moduleFile(t *testing.T, repo, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(repo, "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	body := strings.Replace(string(b), "module golang.design/x/accel", "module "+name, 1)
	return body + "\nrequire golang.design/x/accel v0.0.0\n\n" +
		"replace golang.design/x/accel => " + repo + "\n"
}
