// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernels

import "golang.design/x/accel"

// Add sums two inputs elementwise into a third.
//
// It is spec 009's M3 kernel, and the reason it is two inputs rather than one
// is the graph rather than the compiler: a kernel with a single input cannot
// distinguish a dispatch that read the rebound resource from one that read the
// resource it replaced, because there is only one slot to get wrong.
//
//accel:kernel workgroup=64
func Add(t accel.Thread, a []float32, b []float32, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		out[i] = a[i] + b[i]
	}
}
