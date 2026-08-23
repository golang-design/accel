// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"os/exec"
	"strings"
	"testing"
)

// The root package must not depend on the compiler's toolchain.
//
// specs/012-kernel-pipeline.md asserts twice that a test proves this, and until
// now no such test existed — the fact was true and nothing held it there. It
// matters because a kernel is compiled ahead of time by cmd/accel-kernel, which
// needs golang.org/x/tools/go/packages and the go tool; a deployed binary
// linking accel must need neither. One import from the root package into that
// tree would pull a type checker and a subprocess launcher into every program
// that uses this library, and nothing about the build would look wrong.
//
// The check is on the whole transitive graph rather than on this package's own
// import block, because the way this regresses is indirect: a helper added to
// internal/kernel that reaches for go/packages is the plausible mistake, not an
// import written at the top of a file here.
func TestRootPackageDoesNotDependOnTheToolchain(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "golang.design/x/accel").Output()
	if err != nil {
		t.Skipf("go list unavailable: %v", err)
	}
	var bad []string
	for _, dep := range strings.Fields(string(out)) {
		switch {
		case strings.HasPrefix(dep, "golang.org/x/tools"),
			dep == "go/packages", dep == "go/types", dep == "go/parser", dep == "go/ast":
			bad = append(bad, dep)
		}
	}
	if len(bad) > 0 {
		t.Errorf("the root package transitively depends on the kernel compiler's "+
			"toolchain: %s\n\nA program that merely runs a compiled kernel would link a "+
			"type checker. Kernels are compiled ahead of time by cmd/accel-kernel; only "+
			"that command may reach these.", strings.Join(bad, ", "))
	}
}
