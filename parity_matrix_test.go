// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"math"
	"strconv"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/internal/conformance/numeq"
	"golang.design/x/accel/internal/conformance/parity"
)

// The public surface's CPU/Metal parity matrix, per specs/062-backend-parity.md.
//
// This file is the half with no build tag on purpose. It holds the cases and
// the completeness gate, and the gate needs no device: it compares the members
// of a public enumeration, read out of the source, against what the cases say
// they cover. So it runs on Linux and Windows too, where the answer to "does
// every format have a parity case" is just as knowable as it is on a Mac and
// used to be unaskable -- the corpus differential's equivalent check sits
// behind //go:build darwin, and an unlisted kernel added on Linux is invisible
// until somebody runs the suite on a Mac.
//
// parity_darwin_test.go is the other half: it opens both devices and compares.

// parityCase is one recorded piece of work, run on both backends.
//
// A case returns bytes rather than a typed result because the surfaces here
// disagree about what a result is -- a dtype round trip is a byte pattern, a
// colour attachment is a packed texel, a depth attachment is float32 -- and the
// comparison a case wants is stated by its ceiling rather than by its type. An
// exact ceiling compares the bytes; a bounded one decodes float32 and uses
// numeq, which is why every bounded case returns a multiple of four bytes.
type parityCase struct {
	name    string
	covers  parity.Covers
	ceiling parity.Ceiling
	run     func(t *testing.T, d *accel.Device) []byte
}

// parityCases is the registry. Adding a case is adding a row here.
func parityCases() []parityCase {
	var cases []parityCase
	cases = append(cases, dtypeParityCases()...)
	cases = append(cases, formatParityCases()...)
	cases = append(cases, renderStateParityCases()...)
	return cases
}

// parityExclusions are universe members that cannot have a case, and why.
//
// Every entry here is a backend disagreement or an unimplementable comparison,
// stated where the next reader looks. When a backend gains what an entry names,
// the entry is deleted and the gate demands a case -- which is the mechanism
// that makes this list shrink rather than accumulate.
func parityExclusions() []parity.Excluded {
	var out []parity.Excluded
	out = append(out, formatParityExclusions()...)
	out = append(out, renderStateParityExclusions()...)
	return out
}

// paritySurfaces is what the gate enumerates: the public enumerations whose
// members must each have been compared between the two backends.
func paritySurfaces(t *testing.T) []parity.Surface {
	t.Helper()
	pkg, err := parity.Open(".")
	if err != nil {
		t.Fatalf("open the package: %v", err)
	}
	names := []string{
		"DType", "Format",
		"Topology", "FrontFace", "CullMode", "CompareFunc", "StencilOp",
		"IndexFormat", "AttrFormat", "ColorWriteMask",
		"BlendFactor", "BlendOp", "LoadOp", "StoreOp",
	}
	out := make([]parity.Surface, 0, len(names))
	for _, n := range names {
		members, err := parity.Enum(pkg, n)
		if err != nil {
			t.Fatalf("enumerate %s: %v", n, err)
		}
		out = append(out, parity.Surface{Name: n, Members: members})
	}
	return out
}

// Every member of every enumerated surface has a parity case, or a stated
// reason why it cannot.
//
// This is the test that makes the matrix a matrix. Without it the surface is a
// list somebody maintained by hand, and the failure mode is the one this
// project keeps finding: a twelfth format lands, no row is added, and the
// coverage table reports full coverage of an eleven-member universe.
func TestTheParityMatrixCoversEverySurface(t *testing.T) {
	var covered parity.Covers
	for _, c := range parityCases() {
		if len(c.covers) == 0 {
			t.Errorf("parity case %q covers nothing; a case that names no member "+
				"of any surface cannot be reported as coverage", c.name)
		}
		if err := c.ceiling.Validate(); err != nil {
			t.Errorf("parity case %q: %v", c.name, err)
		}
		covered = append(covered, c.covers...)
	}
	for _, err := range parity.CheckMatrix(paritySurfaces(t), covered, parityExclusions()) {
		t.Error(err)
	}
}

// Every case runs on the CPU backend and produces something worth comparing.
//
// It runs on every platform, and it is not the parity comparison: it is what
// keeps a case honest between Metal runs. A case that panics, that reads back
// nothing, or that returns a buffer of zeros would agree with Metal perfectly,
// and the agreement would mean nothing -- two blank images are equal.
func TestEveryParityCaseProducesAResultOnTheCPU(t *testing.T) {
	d := openDevice(t)
	for _, c := range parityCases() {
		t.Run(c.name, func(t *testing.T) {
			got := c.run(t, d)
			assertNotDegenerate(t, "the CPU backend", got)
		})
	}
}

// assertNotDegenerate refuses an empty or all-zero result.
func assertNotDegenerate(t *testing.T, who string, got []byte) {
	t.Helper()
	if len(got) == 0 {
		t.Fatalf("%s produced no bytes, so comparing it proves nothing", who)
	}
	for _, b := range got {
		if b != 0 {
			return
		}
	}
	t.Fatalf("%s produced %d bytes and every one is zero, so an agreement here "+
		"would be two blank results agreeing", who, len(got))
}

// compareParity applies a case's ceiling to two backends' results.
func compareParity(t *testing.T, c parityCase, onCPU, onMetal []byte) {
	t.Helper()
	if len(onCPU) != len(onMetal) {
		t.Fatalf("%s: %d bytes on the CPU backend and %d on Metal",
			c.name, len(onCPU), len(onMetal))
	}
	if c.ceiling.Exact() {
		if r := numeq.Exact(onMetal, onCPU); !r.Equal {
			t.Fatalf("%s: %v\n  %s", c.name, r, c.ceiling)
		}
		return
	}
	cpu, metal := asFloat32(t, onCPU), asFloat32(t, onMetal)
	var r numeq.Report
	if c.ceiling.Abs > 0 {
		r = withinAbsolute(metal, cpu, c.ceiling.Abs)
	} else {
		r = numeq.WithinULP(metal, cpu, c.ceiling.ULP)
	}
	if !r.Equal {
		t.Fatalf("%s: %v\n  %s", c.name, r, c.ceiling)
	}
}

func asFloat32(t *testing.T, raw []byte) []float32 {
	t.Helper()
	if len(raw)%4 != 0 {
		t.Fatalf("a bounded case returned %d bytes, which is not a whole number of "+
			"float32: a ceiling other than exact decodes the result", len(raw))
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 |
			uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24)
	}
	return out
}

// withinAbsolute compares against an absolute ceiling, which is what a value
// crossing zero needs: a ULP distance across a zero crossing is enormous and
// says nothing about whether the two agree.
func withinAbsolute(got, want []float32, ceiling float64) numeq.Report {
	r := numeq.Report{Equal: true, FirstDiff: -1, Len: len(got), WantLen: len(want)}
	if len(got) != len(want) {
		r.Equal = false
		return r
	}
	for i := range got {
		if d := math.Abs(float64(got[i]) - float64(want[i])); d > ceiling {
			r.Diffs++
			if r.Equal {
				r.Equal, r.FirstDiff = false, i
				r.Got, r.Want = fmtFloat(got[i]), fmtFloat(want[i])
			}
		}
	}
	return r
}

func fmtFloat(v float32) string { return strconv.FormatFloat(float64(v), 'g', -1, 32) }
