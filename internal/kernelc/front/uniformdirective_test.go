// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package front_test

import (
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/front"
)

// The //accel:uniform declaration is checked against the kernel's bindings.
//
// specs/063-uniform-loads.md: a name that is not a binding is a typo, a
// binding the body writes cannot be uniform because the writes are what the
// promise excludes, an empty list declares nothing, and a helper has no
// dispatch to promise anything about.
func TestUniformDirectiveIsCheckedAgainstTheBindings(t *testing.T) {
	cases := []struct {
		name string
		src  string
		says string
	}{{
		name: "a name that is not a binding",
		src: `//accel:kernel workgroup=64
//accel:uniform table
func K(t accel.Thread, in []uint32, out []float32) { out[t.GlobalID().X] = float32(in[0]) }`,
		says: `names "table", which is not one of its bindings`,
	}, {
		name: "a binding the body writes",
		src: `//accel:kernel workgroup=64
//accel:uniform out
func K(t accel.Thread, in []uint32, out []float32) { out[t.GlobalID().X] = float32(in[0]) }`,
		says: `names "out", which the body writes`,
	}, {
		name: "a binding written through a helper",
		src: `//accel:kernel workgroup=64
//accel:uniform out
func K(t accel.Thread, in []uint32, out []float32) { store(out, t.GlobalID().X, float32(in[0])) }

//accel:helper
func store(out []float32, i uint32, v float32) { out[i] = v }`,
		says: `names "out", which the body writes`,
	}, {
		name: "no names",
		src: `//accel:kernel workgroup=64
//accel:uniform
func K(t accel.Thread, in []uint32, out []float32) { out[t.GlobalID().X] = float32(in[0]) }`,
		says: "names no binding",
	}, {
		name: "on a helper",
		src: `//accel:kernel workgroup=64
func K(t accel.Thread, in []uint32, out []float32) { out[t.GlobalID().X] = first(in) }

//accel:helper
//accel:uniform in
func first(in []uint32) float32 { return float32(in[0]) }`,
		says: "a helper has no dispatch",
	}}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			pkg := checkSource(t, "package k\n\nimport \"golang.design/x/accel\"\n\n"+c.src+"\n")
			if pkg == nil {
				t.Fatal("the source did not type-check")
			}
			_, diags := front.Check(pkg)
			if len(diags) == 0 {
				t.Fatalf("accepted; want a diagnostic saying %q", c.says)
			}
			if !strings.Contains(diags.Error(), c.says) {
				t.Fatalf("the diagnostic does not say %q:\n%v", c.says, diags)
			}
		})
	}
}

// A declaration that names read-only bindings marks them, and nothing else.
func TestUniformDirectiveMarksTheNamedBindings(t *testing.T) {
	pkg := checkSource(t, `package k

import "golang.design/x/accel"

//accel:kernel workgroup=64
//accel:uniform offsets, lengths
func K(t accel.Thread, offsets []uint32, lengths []uint32, in []float32, out []float32) {
	i := t.GlobalID().X
	out[i] = in[i] + float32(offsets[0]) + float32(lengths[0])
}
`)
	if pkg == nil {
		t.Fatal("the source did not type-check")
	}
	fns, diags := front.Check(pkg)
	if len(diags) > 0 {
		t.Fatalf("rejected: %v", diags)
	}
	want := map[string]bool{"offsets": true, "lengths": true, "in": false, "out": false}
	for _, b := range fns[0].Bindings {
		if b.Uniform != want[b.Name] {
			t.Errorf("binding %q: Uniform = %v, want %v", b.Name, b.Uniform, want[b.Name])
		}
	}
}
