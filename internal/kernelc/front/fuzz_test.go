// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package front_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"sync"
	"testing"

	"golang.design/x/accel/internal/kernelc/front"
	"golang.org/x/tools/go/packages"
)

// accelOnce loads the real accel package once per binary.
//
// Once, because go/packages shells out to the go tool and a fuzz target that
// pays that per input runs a few hundred cases a minute instead of tens of
// thousands. With the package cached, each input costs a parse and a type check.
var (
	accelOnce sync.Once
	accelPkg  *types.Package
	accelErr  error
)

func loadAccel(t testing.TB) *types.Package {
	t.Helper()
	accelOnce.Do(func() {
		cfg := &packages.Config{Mode: packages.NeedName | packages.NeedTypes | packages.NeedDeps | packages.NeedImports}
		pkgs, err := packages.Load(cfg, "golang.design/x/accel")
		if err != nil {
			accelErr = err
			return
		}
		for _, p := range pkgs {
			if p.PkgPath == "golang.design/x/accel" {
				accelPkg = p.Types
			}
		}
	})
	if accelErr != nil {
		t.Fatalf("loading accel: %v", accelErr)
	}
	if accelPkg == nil {
		t.Fatal("accel did not load")
	}
	return accelPkg
}

// cachedImporter serves the one import a kernel package is allowed.
type cachedImporter struct{ accel *types.Package }

func (i cachedImporter) Import(path string) (*types.Package, error) {
	if path == "golang.design/x/accel" {
		return i.accel, nil
	}
	return nil, errNoSuchImport
}

type noSuchImport struct{}

func (noSuchImport) Error() string { return "accel: kernels import only accel" }

var errNoSuchImport = noSuchImport{}

// checkSource parses and type-checks one source file, returning a package
// shaped the way front.Check consumes.
func checkSource(t testing.TB, src string) *packages.Package {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "k.go", src, parser.ParseComments)
	if err != nil {
		return nil
	}
	info := &types.Info{
		Types:      map[ast.Expr]types.TypeAndValue{},
		Defs:       map[*ast.Ident]types.Object{},
		Uses:       map[*ast.Ident]types.Object{},
		Selections: map[*ast.SelectorExpr]*types.Selection{},
	}
	conf := types.Config{Importer: cachedImporter{loadAccel(t)}, Error: func(error) {}}
	pkg, _ := conf.Check("example.com/fuzz", fset, []*ast.File{f}, info)
	if pkg == nil {
		return nil
	}
	return &packages.Package{
		PkgPath: "example.com/fuzz", Fset: fset,
		Syntax: []*ast.File{f}, Types: pkg, TypesInfo: info,
	}
}

// FuzzCheck feeds arbitrary kernel bodies to the front end.
//
// The property is not that any particular program is accepted. It is that the
// front end **always terminates with an answer**: a set of kernels, a set of
// positioned diagnostics, or both, and never a panic and never a kernel whose
// IR contains a nil the emitter will dereference. A compiler that crashes on
// malformed input is one whose diagnostics nobody ever sees.
//
// It also asserts the invariant the whole design rests on: a kernel that came
// back is a kernel with a body, bindings, and a workgroup extent. Returning a
// half-built kernel beside its own rejection is the failure that would let the
// emitter run on something the checker refused.
func FuzzCheck(f *testing.F) {
	seeds := []string{
		"i := t.GlobalID().X\n\tif i < uint32(len(out)) {\n\t\tout[i] = in[i] * 2\n\t}",
		"out[0] = 1",
		"for i := 0; i < 4; i++ {\n\t\tout[i] = 1\n\t}",
		"t.Barrier()",
		"var n uint32 = 1\n\tout[n] = in[n]",
		"switch {\n\t}",
		"out[t.GlobalIndex()] = float32(len(in))",
		"n := t.LocalID().Z\n\tn += 1\n\tout[n] = float32(n)",
		"",
	}
	for _, s := range seeds {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, body string) {
		src := "package fuzz\n\nimport \"golang.design/x/accel\"\n\n" +
			"//accel:kernel workgroup=64\n" +
			"func K(t accel.Thread, in []float32, out []float32) {\n\t" + body + "\n}\n"

		pkg := checkSource(t, src)
		if pkg == nil {
			return // not Go, which is the parser's business and not the subset's
		}

		kernels, diags := front.Check(pkg)

		for _, d := range diags {
			if d.Msg == "" {
				t.Fatal("a diagnostic with no message")
			}
			if d.Pos.Line <= 0 {
				t.Fatalf("a diagnostic with no line: %q", d.Msg)
			}
		}
		for _, k := range kernels {
			if k.Body == nil {
				t.Fatalf("kernel %q came back with no body", k.Name)
			}
			if k.Workgroup == [3]uint32{} {
				t.Fatalf("kernel %q came back with no workgroup extent", k.Name)
			}
			if len(k.Bindings) == 0 {
				t.Fatalf("kernel %q came back with no bindings", k.Name)
			}
			if k.Thread < 0 {
				t.Fatalf("kernel %q came back with no Thread parameter", k.Name)
			}
			assertNoNilNodes(t, k.Body)
		}
	})
}

// FuzzParseWorkgroup covers the directive parser, which reads text a human
// typed in a comment.
func FuzzParseWorkgroup(f *testing.F) {
	for _, s := range []string{
		"//accel:kernel workgroup=64", "//accel:kernel workgroup=16,8",
		"//accel:kernel workgroup=4,4,4", "//accel:kernel", "//accel:kernel workgroup=",
		"//accel:kernel workgroup=0", "//accel:kernel workgroup=99999999999999999999",
		"//accel:helper", "// ordinary comment",
	} {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, directive string) {
		src := "package fuzz\n\nimport \"golang.design/x/accel\"\n\n" +
			directive + "\nfunc K(t accel.Thread, out []float32) { out[0] = 1 }\n"
		pkg := checkSource(t, src)
		if pkg == nil {
			return
		}
		kernels, _ := front.Check(pkg)
		for _, k := range kernels {
			// An extent that reached a kernel is usable: zero on any axis would
			// mean a dispatch of nothing, and the emitter would bake it in.
			for axis, n := range k.Workgroup {
				if n == 0 {
					t.Fatalf("kernel %q has extent 0 on axis %d, from %q", k.Name, axis, directive)
				}
			}
		}
	})
}
