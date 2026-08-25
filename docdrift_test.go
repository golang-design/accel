// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"go/ast"
	"go/doc"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// Every API name the documentation uses still exists.
//
// # Why this test exists
//
// specs/036-documentation.md §3.1 requires tutorial code to live in `Example`
// functions precisely so `go test` compiles it and *"a tutorial cannot drift
// from the API"*. That guard was specified and never built, and the drift it
// predicted happened: `docs/tutorial/04-graphs.md` called `g.Rebind`, which the
// freeze fixes replaced with `Bind`, and nothing noticed.
//
// Examples are still the better form and would catch more than this. This
// catches the class that actually bit — a name in prose that no longer exists —
// over *all* the documentation rather than only the parts anyone rewrote as an
// example, and it costs no restructuring of the tutorials.
//
// # What it checks and what it cannot
//
// A qualified name (`accel.Foo`) must be exported by this package. An unqualified
// method call (`.Foo(`) must be a method or function *somewhere* in it. The
// second is deliberately loose: the alternative is resolving the receiver's
// type from prose, which needs a type checker and a complete program, and a
// loose check that catches a deleted method beats an exact one nobody writes.
func TestTheDocumentationNamesNothingThatIsGone(t *testing.T) {
	known := exportedNames(t)

	qualified := regexp.MustCompile(`\baccel\.([A-Z][A-Za-z0-9_]*)`)
	// The receiver is captured so a call qualified by a package this repo does
	// not own can be skipped. Without it `atomic.AddInt32` and
	// `unsafe.Pointer` read as missing accel methods, and a guard with false
	// alarms is a guard nobody keeps.
	called := regexp.MustCompile(`(\w*)\.([A-Z][A-Za-z0-9_]*)\(`)

	var files []string
	for _, dir := range []string{"docs", "docs/tutorial", "."} {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".md") {
				files = append(files, filepath.Join(dir, e.Name()))
			}
		}
	}
	if len(files) == 0 {
		t.Fatal("no documentation was scanned, so this test proves nothing")
	}

	var missing []string
	for _, f := range files {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("read %s: %v", f, err)
		}
		text := string(src)
		for _, m := range qualified.FindAllStringSubmatch(text, -1) {
			if !known[m[1]] {
				missing = append(missing, f+": accel."+m[1])
			}
		}
		for _, m := range called.FindAllStringSubmatch(text, -1) {
			if foreign[m[1]] || !ours(m[2]) {
				continue
			}
			if !known[m[2]] {
				missing = append(missing, f+": ."+m[2]+"()")
			}
		}
	}

	sort.Strings(missing)
	missing = dedupe(missing)
	for _, m := range missing {
		t.Errorf("the documentation names %s, which this package no longer exports; "+
			"a document that advises a call is a promise the call is there", m)
	}
}

// foreign is every package qualifier the documentation uses that this repo does
// not own. A call through one of them is not ours to check.
var foreign = map[string]bool{
	"atomic": true, "unsafe": true, "math": true, "fmt": true, "log": true,
	"time": true, "errors": true, "os": true, "strings": true, "sort": true,
	"sync": true, "testing": true, "rand": true, "binary": true, "bytes": true,
	"json": true, "http": true, "io": true, "filepath": true, "exec": true,
}

// ours reports whether a bare method name is one this package plausibly owns.
//
// A deny list rather than an allow list, because the allow list is exactly the
// export set this test is checking against — and using it both ways would make
// a deleted method disappear from the question as well as from the answer.
func ours(name string) bool {
	switch name {
	// The standard library, on types a tutorial legitimately uses.
	case "Fatal", "Fatalf", "Printf", "Println", "Print", "Error", "Errorf",
		"Sprintf", "Sqrt", "Abs", "Sin", "Cos", "Exp", "Log", "Now", "Since",
		"String", "Len", "Cap", "Run", "Skip", "Helper", "Cleanup", "Logf",
		// math/rand's, reached through a *rand.Rand a tutorial holds in a
		// local. The qualifier list above cannot catch these: it matches
		// `rand.Float32()` and a tutorial writes `rng.Float32()`, where `rng`
		// is a variable this test does not resolve types for.
		"Float32", "Float64", "IntN", "Perm", "NormFloat64", "Uint32":
		return false
	}
	return true
}

// exportedNames is every exported identifier this package declares, including
// methods and struct fields.
func exportedNames(t *testing.T) map[string]bool {
	t.Helper()
	out := map[string]bool{}

	// This package and every package whose types it aliases. accel.Thread is
	// internal/kernel's Thread, so t.GlobalID() is a name the documentation may
	// legitimately use and this package does not declare -- and a scan that
	// missed those would report a wall of false alarms nobody reads, which is
	// the same as having no guard.
	for _, dir := range []string{
		".", "internal/kernel", "internal/driver", "kernelabi", "tensor", "quant", "kmath",
	} {
		collectExported(t, dir, out)
	}
	if len(out) < 100 {
		t.Fatalf("only %d exported names were found, so the scan is broken rather than "+
			"the documentation being clean", len(out))
	}
	return out
}

func collectExported(t *testing.T, dir string, out map[string]bool) {
	t.Helper()
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, dir, func(fi os.FileInfo) bool {
		return !strings.HasSuffix(fi.Name(), "_test.go")
	}, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse %s: %v", dir, err)
	}
	for name, pkg := range pkgs {
		if strings.HasSuffix(name, "_test") {
			continue
		}
		collectFromDoc(doc.New(pkg, dir, doc.AllDecls), out)
	}
}

func collectFromDoc(d *doc.Package, out map[string]bool) {
	add := func(n string) {
		if n != "" && ast.IsExported(n) {
			out[n] = true
		}
	}
	for _, f := range d.Funcs {
		add(f.Name)
	}
	for _, ty := range d.Types {
		add(ty.Name)
		for _, f := range ty.Funcs {
			add(f.Name)
		}
		for _, m := range ty.Methods {
			add(m.Name)
		}
		for _, c := range ty.Consts {
			for _, n := range c.Names {
				add(n)
			}
		}
		for _, v := range ty.Vars {
			for _, n := range v.Names {
				add(n)
			}
		}
		// Struct fields, because a tutorial writes them in a literal and a
		// renamed one is the same drift as a renamed method.
		if ts, ok := ty.Decl.Specs[0].(*ast.TypeSpec); ok {
			if st, ok := ts.Type.(*ast.StructType); ok && st.Fields != nil {
				for _, fld := range st.Fields.List {
					for _, n := range fld.Names {
						add(n.Name)
					}
				}
			}
		}
	}
	for _, c := range d.Consts {
		for _, n := range c.Names {
			add(n)
		}
	}
	for _, v := range d.Vars {
		for _, n := range v.Names {
			add(n)
		}
	}
}

func dedupe(s []string) []string {
	out := s[:0]
	var last string
	for _, v := range s {
		if v != last {
			out = append(out, v)
		}
		last = v
	}
	return out
}
