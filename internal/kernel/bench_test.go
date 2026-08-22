// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernel_test

import (
	"testing"

	"golang.design/x/accel/internal/kernel"
)

// BenchmarkBind measures the check a dispatch pays once, which is the reason it
// is once rather than per invocation.
func BenchmarkBind(b *testing.B) {
	k := &kernel.Kernel{
		Name: "Scale", Generator: kernel.ABIVersion,
		Bindings: []kernel.Binding{
			{Name: "in", DType: kernel.F32, Access: kernel.Read},
			{Name: "out", DType: kernel.F32, Access: kernel.Write},
		},
	}
	args := kernel.Args{Slices: []any{make([]float32, 1024), make([]float32, 1024)}}

	b.ReportAllocs()
	for b.Loop() {
		if err := k.Bind(args); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkSlice measures what a generated entry point pays per binding per
// dispatch, which is why it is a type assertion and not a lookup.
func BenchmarkSlice(b *testing.B) {
	args := kernel.Args{Slices: []any{make([]float32, 1024)}}
	b.ReportAllocs()
	for b.Loop() {
		if s := kernel.Slice[float32](args, 0); len(s) != 1024 {
			b.Fatal("wrong length")
		}
	}
}

// BenchmarkThreadIndices measures the linearization a kernel does per
// invocation to index a buffer, which at a million invocations is not free.
func BenchmarkThreadIndices(b *testing.B) {
	size := kernel.ID3{X: 8, Y: 8, Z: 1}
	count := kernel.ID3{X: 16, Y: 16, Z: 1}
	t := kernel.NewThread(
		kernel.ID3{X: 3, Y: 5}, kernel.ID3{X: 3, Y: 5}, kernel.ID3{X: 1, Y: 2}, size, count)

	b.ReportAllocs()
	for b.Loop() {
		if t.GlobalIndex() == 1<<31 {
			b.Fatal("impossible")
		}
	}
}
