// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package emit_test

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"testing"

	"golang.design/x/accel/internal/kernelc/emit"
	"golang.design/x/accel/internal/kernelc/front"
	"golang.design/x/accel/internal/kernelc/ir"
	"golang.org/x/tools/go/packages"
)

// FuzzEmit runs the whole compiler over arbitrary kernel bodies and asserts the
// property that matters most about a generator: **whatever it emits, parses.**
//
// A generator that emits text which does not parse breaks its caller's package
// build with an error pointing at generated code, which is the worst place a Go
// programmer can be sent. And a generator that panics leaves a package
// half-written. So the contract is that Generate either returns an error naming
// what it could not lower, or returns Go.
//
// The digest is checked at the same time, because it is computed from the same
// IR and a panic there is equally fatal to `go generate`.
func FuzzEmit(f *testing.F) {
	for _, s := range []string{
		"i := t.GlobalID().X\n\tif i < uint32(len(out)) {\n\t\tout[i] = in[i] * 2\n\t}",
		"out[0] = in[0]",
		"n := t.LocalIndex()\n\tn += 1\n\tout[n] = float32(n)",
		"out[t.GroupIndex()] = -in[0]",
		"if t.GlobalID().Y > 0 {\n\t\tout[0] = 1\n\t} else {\n\t\tout[0] = 2\n\t}",
		"var x float32 = in[0]\n\tout[0] = x * x",
		"",
	} {
		f.Add(s)
	}

	f.Fuzz(func(t *testing.T, body string) {
		src := "package fuzz\n\nimport \"golang.design/x/accel\"\n\n" +
			"//accel:kernel workgroup=64\n" +
			"func K(t accel.Thread, in []float32, out []float32) {\n\t" + body + "\n}\n"

		pkg := checkSourceForFuzz(t, src)
		if pkg == nil {
			return
		}
		kernels, _ := front.Check(pkg)
		if len(kernels) == 0 {
			return
		}
		for _, k := range kernels {
			d := emit.Digest(k)
			if d == "" {
				t.Fatalf("kernel %q got an empty digest", k.Name)
			}
			if emit.Digest(k) != d {
				t.Fatalf("kernel %q has an unstable digest", k.Name)
			}
			k.Digest = d
		}

		out, err := emit.Generate(emit.Package{Name: "fuzz", Kernels: kernels})
		if err != nil {
			return // naming what it could not lower is a legal outcome
		}
		fset := token.NewFileSet()
		if _, err := parser.ParseFile(fset, "generated.go", out, parser.ParseComments); err != nil {
			t.Fatalf("emitted source does not parse: %v\n%s", err, out)
		}
	})
}

// FuzzDigestPreimage checks that the digest separates kernels which differ.
//
// A digest that collided across meaningfully different kernels would report a
// stale generated file as fresh, which is the one thing freshness exists to
// prevent.
func FuzzDigestPreimage(f *testing.F) {
	f.Add("Scale", uint(64), uint(1), true, false)
	f.Add("Other", uint(128), uint(2), false, true)

	f.Fuzz(func(t *testing.T, name string, x, y uint, read, write bool) {
		if x == 0 || y == 0 || x > 1<<20 || y > 1<<20 {
			t.Skip()
		}
		base := corpusCopy(t)
		mutated := corpusCopy(t)
		mutated.Name = name
		mutated.Workgroup = [3]uint32{uint32(x), uint32(y), 1}
		mutated.Bindings[0].Read, mutated.Bindings[0].Write = read, write

		same := base.Name == mutated.Name &&
			base.Workgroup == mutated.Workgroup &&
			base.Bindings[0].Read == mutated.Bindings[0].Read &&
			base.Bindings[0].Write == mutated.Bindings[0].Write

		if got := emit.Digest(base) == emit.Digest(mutated); got != same {
			t.Fatalf("digests %v while the kernels differ=%v\n--- base ---\n%s\n--- mutated ---\n%s",
				map[bool]string{true: "match", false: "differ"}[got], !same,
				emit.Preimage(base), emit.Preimage(mutated))
		}
	})
}

// corpusCopy builds a kernel by hand rather than loading one.
//
// Loading shells out to the go tool at roughly 400ms a call, which held this
// target to seventeen executions in fifteen seconds. The digest does not care
// where its IR came from, and a fuzz target that runs seventeen times is a unit
// test with extra steps.
func corpusCopy(*testing.T) *ir.Func {
	pos := token.Pos(1)
	f32s := &ir.Type{Kind: ir.Slice, Elem: &ir.Type{Kind: ir.F32}}
	return &ir.Func{
		Name: "Scale", Kernel: true, Workgroup: [3]uint32{64, 1, 1}, Thread: 0,
		Params: []*ir.Param{
			ir.NewParam(pos, &ir.Type{Kind: ir.Struct, Name: "Thread"}, 0, "t", nil),
			ir.NewParam(pos, f32s, 1, "in", nil),
			ir.NewParam(pos, f32s, 2, "out", nil),
		},
		Bindings: []*ir.Binding{
			{Name: "in", Index: 1, Type: f32s, Read: true},
			{Name: "out", Index: 2, Type: f32s, Write: true},
		},
		Body:       ir.NewBlock(pos, ir.NewReturn(pos, nil)),
		Intrinsics: []string{"accel.Thread.GlobalID"},
		Source:     "func Scale(t accel.Thread, in []float32, out []float32) {}",
	}
}

// checkSourceForFuzz parses and type-checks one source file against the cached
// accel package.
//
// Cached, because go/packages shells out to the go tool and a target that pays
// that per input runs a few hundred cases a minute rather than tens of
// thousands.
func checkSourceForFuzz(t testing.TB, src string) *packages.Package {
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
	conf := types.Config{Importer: importerFor(t), Error: func(error) {}}
	pkg, _ := conf.Check("example.com/fuzz", fset, []*ast.File{f}, info)
	if pkg == nil {
		return nil
	}
	return &packages.Package{
		PkgPath: "example.com/fuzz", Fset: fset,
		Syntax: []*ast.File{f}, Types: pkg, TypesInfo: info,
	}
}
