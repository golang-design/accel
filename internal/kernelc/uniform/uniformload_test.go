// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package uniform_test

import (
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/front"
	"golang.design/x/accel/internal/kernelc/uniform"
)

// A load from a binding declared uniform is as uniform as its index.
//
// specs/063-uniform-loads.md. Without the declaration every load is
// non-uniform (002 section 3.3's seed); with it, the index decides, so a load
// at a per-invocation index stays non-uniform.
func TestALoadFromADeclaredUniformBindingIsAsUniformAsItsIndex(t *testing.T) {
	src := `package k

import "golang.design/x/accel"

//accel:kernel workgroup=64
//accel:uniform table
func K(t accel.Thread, table []uint32, in []float32, out []float32) {
	seed := in[t.GlobalID().X]
	out[t.GlobalID().X] = seed
	byGroup := table[t.GroupIndex()]
	byLane := table[t.LocalIndex()]
	plain := in[t.GroupIndex()]
	out[0] = float32(byGroup+byLane) + plain
}
`
	levels := levelsOf(t, src)
	for name, want := range map[string]uniform.Level{
		"byGroup": uniform.Workgroup,
		"byLane":  uniform.Non,
		"plain":   uniform.Non,
	} {
		if got := levels[name]; got != want {
			t.Errorf("%s is %v, want %v", name, got, want)
		}
	}
}

// A barrier in a loop bounded by a declared-uniform load is accepted, and the
// same loop without the declaration is refused. This is the case the
// declaration exists for: a routing table read at a group index bounding a
// barrier-bearing loop, which 002 section 3.3 names as a rejected family.
func TestABarrierBoundedByADeclaredUniformLoadIsAccepted(t *testing.T) {
	kernel := func(directive string) string {
		return "package k\n\nimport \"golang.design/x/accel\"\n\n" +
			"//accel:kernel workgroup=64\n" + directive +
			"func K(t accel.Thread, table []uint32, in []float32, out []float32) {\n" +
			"\tseed := in[t.GlobalID().X]\n\tout[t.GlobalID().X] = seed\n" +
			"\tfor i := uint32(0); i < table[t.GroupIndex()]; i++ {\n\t\tt.Barrier()\n\t}\n}\n"
	}
	check := func(t *testing.T, src string) []uniform.Rejection {
		pkg := checkSource(t, src)
		if pkg == nil {
			t.Fatalf("the source did not type-check:\n%s", src)
		}
		fns, diags := front.Check(pkg)
		if len(diags) > 0 {
			t.Fatalf("the front end rejected it: %v", diags)
		}
		return uniform.AcceptBarriers(fns[0])
	}
	if rej := check(t, kernel("//accel:uniform table\n")); len(rej) != 0 {
		t.Fatalf("declared uniform and still rejected: %v", rej)
	}
	rej := check(t, kernel(""))
	if len(rej) == 0 {
		t.Fatal("without the declaration the loop's bound is a load, and the barrier was accepted")
	}
	if !strings.Contains(rej[0].Msg, "workgroup-uniform") {
		t.Errorf("the rejection does not name the rule: %q", rej[0].Msg)
	}
}
