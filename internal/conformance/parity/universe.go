// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package parity

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

// The universe is read out of the source rather than written down.
//
// A hand-maintained list of "every format" is the same artefact as a
// hand-maintained list of "every format with a parity case", and it goes stale
// the same way: somebody adds a member, nobody adds the row, and the gate
// reports full coverage of a universe that is one short. Parsing the
// declaration means the only way to add a member is to add it where the gate
// is looking.
//
// Syntax and no type checking. go/types would resolve the alias forms more
// directly, but the shapes here are syntactic -- a const block with a type, or
// a const block of selectors into an aliased package -- and pulling a type
// checker into the test tree for that trade is not worth the dependency
// importgraph_test.go exists to keep out of this module.

// Package is one parsed Go package directory, plus what it needs to follow an
// alias into another package of the same module.
type Package struct {
	Dir string // directory holding the .go files

	modRoot string
	modPath string
}

// Open parses the package in dir and locates the module it belongs to, so an
// enumeration aliased from another package of the same module can be followed.
func Open(dir string) (Package, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return Package{}, err
	}
	root, path, err := findModule(abs)
	if err != nil {
		return Package{}, err
	}
	return Package{Dir: abs, modRoot: root, modPath: path}, nil
}

func findModule(dir string) (root, path string, err error) {
	for d := dir; ; {
		b, err := os.ReadFile(filepath.Join(d, "go.mod"))
		if err == nil {
			for _, line := range strings.Split(string(b), "\n") {
				if rest, ok := strings.CutPrefix(strings.TrimSpace(line), "module "); ok {
					return d, strings.TrimSpace(rest), nil
				}
			}
			return "", "", fmt.Errorf("%s/go.mod names no module", d)
		}
		parent := filepath.Dir(d)
		if parent == d {
			return "", "", fmt.Errorf("no go.mod above %s", dir)
		}
		d = parent
	}
}

// files parses the package's non-test Go files.
//
// Test files are excluded because the universe is the public surface, and a
// constant declared only in a test is not part of it.
func (p Package) files() ([]*ast.File, error) {
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, p.Dir, func(fi os.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, 0)
	if err != nil {
		return nil, fmt.Errorf("parse %s: %w", p.Dir, err)
	}
	var out []*ast.File
	for _, pkg := range pkgs {
		for _, f := range pkg.Files {
			out = append(out, f)
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s holds no non-test Go files", p.Dir)
	}
	return out, nil
}

// Enum returns the exported constants of the named type, in the order the
// source declares them.
//
// Three declaration forms occur in this module and all three are handled: an
// iota block with the type on the first spec, a block with the type repeated,
// and the re-export form the render enumerations use, where the local type is
// an alias (`type BlendFactor = driver.BlendFactor`) and each constant is a
// selector into the aliased package. The third is why this takes a Package
// rather than a directory: it has to follow the alias to find out which
// selectors are members.
func Enum(p Package, typeName string) ([]string, error) {
	files, err := p.files()
	if err != nil {
		return nil, err
	}

	names := directEnum(files, typeName)
	if len(names) > 0 {
		return names, nil
	}

	target, aliased := aliasTarget(files, typeName)
	if !aliased {
		return nil, fmt.Errorf("%s declares no constants of type %s and no alias for it; "+
			"a surface with no members is a gate that can never fail", p.Dir, typeName)
	}
	names, err = reExported(p, files, target)
	if err != nil {
		return nil, err
	}
	if len(names) == 0 {
		return nil, fmt.Errorf("%s aliases %s.%s and re-exports none of its constants",
			p.Dir, target.pkg, target.name)
	}
	return names, nil
}

// directEnum collects constants whose spec states the type, plus the untyped
// specs that follow one in the same declaration and inherit it from iota.
//
// A spec with no type but with a value ends the run. That is the `WriteAll =
// WriteRed | WriteGreen | ...` form, which is a derived constant rather than a
// member: counting it would put a name in the universe that no case can
// meaningfully cover on its own.
func directEnum(files []*ast.File, typeName string) []string {
	var out []string
	forEachConstDecl(files, func(d *ast.GenDecl) {
		typed := false
		for _, spec := range d.Specs {
			vs, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}
			switch {
			case vs.Type != nil:
				typed = isIdent(vs.Type, typeName)
			case len(vs.Values) > 0:
				typed = false
			}
			if !typed {
				continue
			}
			for _, n := range vs.Names {
				if n.IsExported() {
					out = append(out, n.Name)
				}
			}
		}
	})
	return out
}

type qualified struct{ pkg, name string }

// aliasTarget finds `type T = pkg.U` and returns pkg.U.
func aliasTarget(files []*ast.File, typeName string) (qualified, bool) {
	var out qualified
	found := false
	for _, f := range files {
		for _, d := range f.Decls {
			gd, ok := d.(*ast.GenDecl)
			if !ok || gd.Tok != token.TYPE {
				continue
			}
			for _, spec := range gd.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok || !ts.Assign.IsValid() || ts.Name.Name != typeName {
					continue
				}
				if sel, ok := ts.Type.(*ast.SelectorExpr); ok {
					if id, ok := sel.X.(*ast.Ident); ok {
						out, found = qualified{id.Name, sel.Sel.Name}, true
					}
				}
			}
		}
	}
	return out, found
}

// reExported returns the local names of constants assigned from the aliased
// package's members of the aliased type, in the aliased package's declaration
// order.
//
// The order comes from the other package rather than from this one because it
// is the enumeration's own order, and a failure that lists members out of
// order sends the reader to the wrong place in the source.
func reExported(p Package, files []*ast.File, target qualified) ([]string, error) {
	dir, err := p.importDir(files, target.pkg)
	if err != nil {
		return nil, err
	}
	other := Package{Dir: dir, modRoot: p.modRoot, modPath: p.modPath}
	otherFiles, err := other.files()
	if err != nil {
		return nil, err
	}
	members := directEnum(otherFiles, target.name)

	// local[<member of the other package>] = the name re-exported here.
	local := map[string]string{}
	forEachConstDecl(files, func(d *ast.GenDecl) {
		for _, spec := range d.Specs {
			vs, ok := spec.(*ast.ValueSpec)
			if !ok || len(vs.Names) != 1 || len(vs.Values) != 1 {
				continue
			}
			sel, ok := vs.Values[0].(*ast.SelectorExpr)
			if !ok {
				continue
			}
			id, ok := sel.X.(*ast.Ident)
			if !ok || id.Name != target.pkg || !vs.Names[0].IsExported() {
				continue
			}
			local[sel.Sel.Name] = vs.Names[0].Name
		}
	})

	var out []string
	for _, m := range members {
		if name, ok := local[m]; ok {
			out = append(out, name)
		}
	}
	return out, nil
}

// importDir maps a package identifier used in these files to a directory of
// the same module.
func (p Package) importDir(files []*ast.File, ident string) (string, error) {
	for _, f := range files {
		for _, imp := range f.Imports {
			path := strings.Trim(imp.Path.Value, `"`)
			name := filepath.Base(path)
			if imp.Name != nil {
				name = imp.Name.Name
			}
			if name != ident {
				continue
			}
			rest, ok := strings.CutPrefix(path, p.modPath)
			if !ok {
				return "", fmt.Errorf("package %q is outside module %s; "+
					"an enumeration aliased from outside the module is not this gate's to follow",
					path, p.modPath)
			}
			return filepath.Join(p.modRoot, filepath.FromSlash(rest)), nil
		}
	}
	return "", fmt.Errorf("no import in %s is named %q", p.Dir, ident)
}

func forEachConstDecl(files []*ast.File, fn func(*ast.GenDecl)) {
	for _, f := range files {
		for _, d := range f.Decls {
			if gd, ok := d.(*ast.GenDecl); ok && gd.Tok == token.CONST {
				fn(gd)
			}
		}
	}
}

func isIdent(e ast.Expr, name string) bool {
	id, ok := e.(*ast.Ident)
	return ok && id.Name == name
}

// Funcs returns the exported package-level functions whose first parameter has
// the given type, in source order within each file and files sorted by name.
//
// This is how the tensor operator set is enumerated: an operator is a function
// taking a *Builder first, and nothing else in that package has that shape.
// Methods are excluded, which is deliberate -- a method on Builder configures
// the builder, and an operator records a node.
func Funcs(p Package, firstParam string) ([]string, error) {
	files, err := p.files()
	if err != nil {
		return nil, err
	}
	var out []string
	for _, f := range files {
		for _, d := range f.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Recv != nil || !fd.Name.IsExported() {
				continue
			}
			if fd.Type.Params == nil || len(fd.Type.Params.List) == 0 {
				continue
			}
			if typeString(fd.Type.Params.List[0].Type) == firstParam {
				out = append(out, fd.Name.Name)
			}
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%s declares no exported function taking %s first",
			p.Dir, firstParam)
	}
	return out, nil
}

// typeString renders the small set of type expressions a first parameter uses.
func typeString(e ast.Expr) string {
	switch t := e.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + typeString(t.X)
	case *ast.SelectorExpr:
		return typeString(t.X) + "." + t.Sel.Name
	}
	return ""
}
