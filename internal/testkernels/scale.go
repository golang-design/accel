// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package testkernels is the kernel corpus: every compute kernel and render
// stage the repository ships, authored in the Go subset and compiled by
// cmd/accel-kernel.
//
// The name is older than the role. It began as the corpus the compiler was
// developed against, and it is still that, but it is also the production
// kernel set: package tensor imports it from thirteen non-test files and every
// operator there lowers to a kernel registered here. Nothing else authors a
// kernel. A change to a body in this package changes what a tensor plan
// computes on every backend, which is why each kernel carries a differential
// test against the CPU oracle and why the generated file is checked for
// freshness.
//
// It is the authored source, not the generated output: the generated file sits
// beside it and is committed, so freshness has something to compare and a
// reader can see both halves of what the generator claims.
package testkernels

import "golang.design/x/accel"

//go:generate go run golang.design/x/accel/cmd/accel-kernel -C ../.. ./internal/testkernels

// Scale multiplies every element of in by two into out.
//
// It is spec 012's kernel, and it is deliberately the smallest body that is
// still a real one: an id, a bounds check, a load, an arithmetic operation, and
// a store. Everything the milestone must compile appears in it once.
//
//accel:kernel workgroup=64
func Scale(t accel.Thread, in []float32, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		out[i] = in[i] * 2
	}
}
