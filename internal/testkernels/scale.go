// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package testkernels is the kernel corpus the compiler is developed against.
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
