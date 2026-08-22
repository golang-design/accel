// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package emit_test

import (
	"testing"

	"golang.design/x/accel/internal/kernelc/emit"
	"golang.design/x/accel/internal/kernelc/ir"
)

// BenchmarkGenerate measures emission alone, with the IR already built.
//
// It answers whether a change to the emitter made generation slower, which
// matters because `go generate` runs it over a growing corpus and spec 010's
// inventory is not small.
func BenchmarkGenerate(b *testing.B) {
	kernels := loadCorpus(b)
	for _, k := range kernels {
		k.Digest = emit.Digest(k)
	}
	pkg := emit.Package{Name: "testkernels", Kernels: kernels}

	b.ReportAllocs()
	for b.Loop() {
		if _, err := emit.Generate(pkg); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDigest measures the freshness check's cost, which is paid once per
// kernel per generate and once per kernel per build.
func BenchmarkDigest(b *testing.B) {
	k := corpusKernel(b, "Scale")
	b.ReportAllocs()
	for b.Loop() {
		if emit.Digest(k) == "" {
			b.Fatal("empty digest")
		}
	}
}

// BenchmarkGenerateManyKernels measures how emission scales with a corpus,
// since spec 010's inventory is what this will really be run over.
func BenchmarkGenerateManyKernels(b *testing.B) {
	one := corpusKernel(b, "Scale")
	one.Digest = emit.Digest(one)

	kernels := make([]*ir.Func, 64)
	for i := range kernels {
		kernels[i] = one
	}
	pkg := emit.Package{Name: "testkernels", Kernels: kernels}

	b.ReportAllocs()
	for b.Loop() {
		if _, err := emit.Generate(pkg); err != nil {
			b.Fatal(err)
		}
	}
}
