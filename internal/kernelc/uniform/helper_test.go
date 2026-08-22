// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package uniform_test

// Type-checking helpers, the same shape the front end's own tests use: a kernel
// has to be type-checked before it can be analysed, and the analysis is about
// what go/types resolved rather than about an AST.

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"sync"
	"testing"

	"golang.org/x/tools/go/packages"
)

// accelOnce loads the real accel package once per binary.
//
// Once, because go/packages shells out to the go tool and a fuzz target that
// pays that per input runs a few hundred cases a minute instead of tens of
// thousands. With the package cached, each input costs a parse and a type check.
var (
	accelOnce sync.Once
	accelPkgs map[string]*types.Package
	accelErr  error
)

// loadAccel loads the packages a kernel may import: accel itself and kmath,
// whose bounded scalar math is an intrinsic family the analysis has to level.
func loadAccel(t testing.TB) map[string]*types.Package {
	t.Helper()
	accelOnce.Do(func() {
		cfg := &packages.Config{Mode: packages.NeedName | packages.NeedTypes | packages.NeedDeps | packages.NeedImports}
		pkgs, err := packages.Load(cfg, "golang.design/x/accel", "golang.design/x/accel/kmath")
		if err != nil {
			accelErr = err
			return
		}
		accelPkgs = map[string]*types.Package{}
		for _, p := range pkgs {
			accelPkgs[p.PkgPath] = p.Types
		}
	})
	if accelErr != nil {
		t.Fatalf("loading accel: %v", accelErr)
	}
	if accelPkgs["golang.design/x/accel"] == nil {
		t.Fatal("accel did not load")
	}
	return accelPkgs
}

// cachedImporter serves the imports a kernel package is allowed.
type cachedImporter struct{ pkgs map[string]*types.Package }

func (i cachedImporter) Import(path string) (*types.Package, error) {
	if p := i.pkgs[path]; p != nil {
		return p, nil
	}
	return nil, errNoSuchImport
}

type noSuchImport struct{}

func (noSuchImport) Error() string { return "accel: kernels import only accel and accel/kmath" }

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
