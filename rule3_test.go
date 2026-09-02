// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"go/types"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages"
)

// Rule 3: no device-backend type appears in a public signature.
//
// specs/000-decisions.md, amended 2026-09-02: the CPU backend is the reference
// implementation rule 4 requires, so OpenCPU, CPUOptions and CPUMode are the
// oracle's surface and public by design. What the rule forbids is a type of a
// device backend -- internal/metal, internal/mtl, internal/mslabi -- reaching
// a public declaration, which is how mslabi.StageVertexBufferLimit came to
// refuse callers on every device. Checked over the type-checked package: every
// exported object's type string must name none of those packages.
func TestNoDeviceBackendTypeIsPublic(t *testing.T) {
	cfg := &packages.Config{Mode: packages.NeedTypes | packages.NeedName | packages.NeedImports | packages.NeedDeps}
	pkgs, err := packages.Load(cfg, "golang.design/x/accel")
	if err != nil {
		t.Fatalf("packages.Load: %v", err)
	}
	if len(pkgs) != 1 || pkgs[0].Types == nil {
		t.Fatalf("loaded %d packages, want the root with types", len(pkgs))
	}
	scope := pkgs[0].Types.Scope()
	checked := 0
	for _, name := range scope.Names() {
		obj := scope.Lookup(name)
		if !obj.Exported() {
			continue
		}
		checked++
		sig := types.TypeString(obj.Type(), nil)
		if named, ok := obj.Type().(*types.Named); ok {
			// Methods too: a method returning a backend type is a public
			// signature as much as a function is.
			for i := 0; i < named.NumMethods(); i++ {
				if m := named.Method(i); m.Exported() {
					sig += " " + types.TypeString(m.Type(), nil)
				}
			}
			if st, ok := named.Underlying().(*types.Struct); ok {
				for i := 0; i < st.NumFields(); i++ {
					if f := st.Field(i); f.Exported() {
						sig += " " + types.TypeString(f.Type(), nil)
					}
				}
			}
		}
		for _, backend := range []string{"internal/metal", "internal/mtl", "internal/mslabi"} {
			if strings.Contains(sig, backend) {
				t.Errorf("%s names a type from %s in its public signature: %s", name, backend, sig)
			}
		}
	}
	if checked < 50 {
		t.Fatalf("checked %d exported declarations; the root package has more, so the walk is wrong", checked)
	}
}
