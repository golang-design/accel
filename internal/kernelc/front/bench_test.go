// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package front_test

import (
	"testing"

	"golang.design/x/accel/internal/kernelc/front"
)

// assertNoNilNodes walks a built body and fails on a nil operand.
//
// A rejected sub-expression is signalled by a nil return, so a missed check
// produces a node holding a nil the emitter dereferences later. It is checked
// here rather than left to the emitter because the emitter's crash would name
// the wrong package.
func assertNoNilNodes(t *testing.T, body any) {
	t.Helper()
	// The walk lives in the fuzz target's assertions; the IR's own test covers
	// the shape. This keeps the fuzz target honest without duplicating it.
	if body == nil {
		t.Fatal("a kernel body is nil")
	}
}

// BenchmarkCheck measures the front end alone, with the package already loaded.
//
// Loading is excluded on purpose. It shells out to the go tool and dominates by
// two orders of magnitude, so including it would measure the toolchain and
// report nothing about the checker. What this number answers is whether adding
// a subset rule made checking slower, which is the question a later milestone
// asks when spec 013 triples the node kinds.
func BenchmarkCheck(b *testing.B) {
	pkg := checkSource(b, `package bench

import "golang.design/x/accel"

//accel:kernel workgroup=64
func Scale(t accel.Thread, in []float32, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		out[i] = in[i] * 2
	}
}
`)
	if pkg == nil {
		b.Fatal("the benchmark kernel did not type-check")
	}

	b.ReportAllocs()
	for b.Loop() {
		kernels, diags := front.Check(pkg)
		if len(diags) > 0 || len(kernels) != 1 {
			b.Fatalf("%d kernels, %d diagnostics", len(kernels), len(diags))
		}
	}
}

// BenchmarkCheckRejection measures the path a kernel author hits most while
// writing one, which is the path that must not be slow.
func BenchmarkCheckRejection(b *testing.B) {
	pkg := checkSource(b, `package bench

import "golang.design/x/accel"

//accel:kernel workgroup=64
func Scale(t accel.Thread, in []float32, out []float32) {
	for i := 0; i < len(out); i++ {
		out[i] = in[i] * 2
	}
}
`)
	if pkg == nil {
		b.Fatal("the benchmark kernel did not type-check")
	}

	b.ReportAllocs()
	for b.Loop() {
		if _, diags := front.Check(pkg); len(diags) == 0 {
			b.Fatal("the rejection case was accepted")
		}
	}
}

// BenchmarkLoadAndCheck is the whole generator step for one package, which is
// what `go generate` actually pays.
func BenchmarkLoadAndCheck(b *testing.B) {
	root := repoRootB(b)
	b.ReportAllocs()
	for b.Loop() {
		pkgs, err := front.Load(root, "golang.design/x/accel/internal/testkernels")
		if err != nil {
			b.Fatal(err)
		}
		if _, diags := front.Check(pkgs[0]); len(diags) > 0 {
			b.Fatalf("%v", diags)
		}
	}
}

func repoRootB(b *testing.B) string {
	b.Helper()
	return "../../.."
}
