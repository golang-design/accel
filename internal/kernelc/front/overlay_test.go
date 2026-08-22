// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package front_test

import (
	"path/filepath"
	"testing"

	"golang.design/x/accel/internal/kernelc/front"
	"golang.org/x/tools/go/packages"
)

// loadOverlay type-checks packages that do not exist on disk.
//
// packages.Config.Overlay creates them inside this module, so each case resolves
// imports of the real accel and type-checks normally, and one Load takes the
// whole set. That is what keeps the rejection corpus affordable: spec 013 makes
// it the executable form of the subset, and a corpus costing a module
// resolution per case is one nobody runs.
//
// files maps a directory name under internal/kernelc/front to its contents.
func loadOverlay(t *testing.T, files map[string]string, patterns []string) map[string]*packages.Package {
	t.Helper()

	root := repoRoot(t)
	overlay := make(map[string][]byte, len(files))
	for dir, src := range files {
		overlay[filepath.Join(root, "internal", "kernelc", "front", dir, "k.go")] = []byte(src)
	}

	cfg := &packages.Config{Mode: front.LoadMode, Dir: root, Overlay: overlay}
	pkgs, err := packages.Load(cfg, patterns...)
	if err != nil {
		t.Fatalf("loading overlay cases: %v", err)
	}

	out := make(map[string]*packages.Package, len(pkgs))
	for i, p := range pkgs {
		if i < len(patterns) {
			out[patterns[i]] = p
		}
	}
	return out
}
