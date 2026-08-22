// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package intrin_test

import (
	"sync"
	"testing"

	"go/types"

	"golang.org/x/tools/go/packages"
)

var (
	kernelPkgOnce sync.Once
	kernelPkg     *types.Package
	kernelPkgErr  error
)

// loadKernelPackage type-checks the real internal/kernel package.
//
// The table is checked against the objects it will actually see rather than
// against a restatement of them, because a test that declares its own Thread is
// testing the test. It loads once per binary: go/packages shells out to the go
// tool, and paying that per case is how a corpus becomes one nobody runs.
func loadKernelPackage(t *testing.T) *types.Package {
	t.Helper()
	kernelPkgOnce.Do(func() {
		cfg := &packages.Config{Mode: packages.NeedName | packages.NeedTypes | packages.NeedDeps | packages.NeedImports}
		pkgs, err := packages.Load(cfg, "golang.design/x/accel/internal/kernel")
		if err != nil {
			kernelPkgErr = err
			return
		}
		for _, p := range pkgs {
			if p.Types != nil && p.PkgPath == "golang.design/x/accel/internal/kernel" {
				kernelPkg = p.Types
			}
		}
	})
	if kernelPkgErr != nil {
		t.Fatalf("loading internal/kernel: %v", kernelPkgErr)
	}
	if kernelPkg == nil {
		t.Fatal("internal/kernel did not load")
	}
	return kernelPkg
}
