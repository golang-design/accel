// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package front_test

import (
	"testing"

	"golang.design/x/accel/internal/kernelc/front"
)

// Func.Helpers is the transitive closure of what a body reaches, with a callee
// before every caller.
//
// Transitive, because the generator emits exactly this list: a direct list
// left a helper reached only through another helper out of the generated file,
// and the Go lowering did not compile -- `undefined: innerFlat`. Callee first,
// because the MSL target requires a declaration before its use and emits the
// list in order.
func TestHelpersAreTransitiveAndCalleeFirst(t *testing.T) {
	const body = `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	out[t.GlobalID().X] = outer(1)
}

//accel:helper
func outer(x float32) float32 { return inner(x) + shared(x) }

//accel:helper
func inner(x float32) float32 { return shared(x) * 2 }

//accel:helper
func shared(x float32) float32 { return x + 1 }`

	pkgs := loadOverlay(t, map[string]string{"helperchain": header("helperchain") + body + "\n"},
		[]string{"./internal/kernelc/front/helperchain"})
	kernels, diags := front.Check(pkgs["./internal/kernelc/front/helperchain"])
	if len(diags) > 0 {
		t.Fatalf("rejected:\n%v", diags)
	}
	if len(kernels) != 1 {
		t.Fatalf("found %d kernels, want 1", len(kernels))
	}

	var got []string
	for _, h := range kernels[0].Helpers {
		got = append(got, h.Name)
	}
	// shared is reached twice and listed once, before both of its callers;
	// inner is reached only through outer and is listed before it.
	want := []string{"shared", "inner", "outer"}
	if len(got) != len(want) {
		t.Fatalf("Helpers = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Helpers = %v, want %v", got, want)
		}
	}
}
