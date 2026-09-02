// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"flag"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
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
	cases = append(cases, formatCopyParityCases()...)
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
	out = append(out, formatCopyParityExclusions()...)
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
	out := make([]parity.Surface, 0, len(names)+1)
	for _, n := range names {
		members, err := parity.Enum(pkg, n)
		if err != nil {
			t.Fatalf("enumerate %s: %v", n, err)
		}
		out = append(out, parity.Surface{Name: n, Members: members})
	}

	// The format enumeration a second time, against the copy entry points.
	//
	// One enumeration can be a surface twice when there are two independent
	// ways to get a format wrong, and there are: section 6.2 compares what a
	// pass *encodes* into a texel and section 6.8 compares how a copy
	// *addresses* the rows around it. Sharing one surface would let a render
	// case's claim answer for a copy that no test has ever made.
	formats, err := parity.Enum(pkg, "Format")
	if err != nil {
		t.Fatalf("enumerate Format: %v", err)
	}
	return append(out, parity.Surface{Name: formatCopySurface, Members: formats})
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

// updateParityGoldens rewrites testdata/parity from the CPU backend's results.
var updateParityGoldens = flag.Bool("update-parity-goldens", false,
	"rewrite testdata/parity/*.bin from the CPU backend's results")

// Every case runs on the CPU backend and produces the result it produced when
// its golden was committed.
//
// It runs on every platform, and it is not the parity comparison: it is what
// keeps a case honest between Metal runs. Before the goldens existed the only
// check here was "some byte is non-zero", and every render case clears to a
// non-zero colour, so a case whose draw was skipped passed on Linux and
// Windows. The golden is the CPU backend's own output, compared under the
// case's ceiling, and it makes the case's result a fact on every platform
// rather than only where a Metal device is present to disagree.
//
// A golden is generated from the CPU backend and validated by
// TestTheParityMatrixAgreesOnCPUAndMetal on darwin, which is where it earns
// its standing: a golden that Metal disagreed with would fail there. When a
// case or the rasterizer changes on purpose, regenerate with
// -update-parity-goldens and run the darwin comparison before committing.
func TestEveryParityCaseProducesAResultOnTheCPU(t *testing.T) {
	d := openDevice(t)
	for _, c := range parityCases() {
		t.Run(c.name, func(t *testing.T) {
			got := c.run(t, d)
			assertNotDegenerate(t, "the CPU backend", got)

			path := filepath.Join("testdata", "parity", parityGoldenName(c.name))
			if *updateParityGoldens {
				if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, got, 0o644); err != nil {
					t.Fatal(err)
				}
				return
			}
			want, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("no golden for this case: %v\n  generate one with\n    go test "+
					"-run TestEveryParityCaseProducesAResultOnTheCPU -update-parity-goldens ./\n"+
					"  and run TestTheParityMatrixAgreesOnCPUAndMetal on a Mac before committing", err)
			}
			compareParity(t, c, want, got, "the committed golden", "the CPU backend")
		})
	}
}

// parityGoldenName is the file a case's golden lives in: the case name with
// everything that is not a letter or a digit folded to one dash.
func parityGoldenName(name string) string {
	var b strings.Builder
	dash := false
	for _, r := range strings.ToLower(name) {
		if r >= 'a' && r <= 'z' || r >= '0' && r <= '9' {
			b.WriteRune(r)
			dash = false
		} else if !dash {
			b.WriteByte('-')
			dash = true
		}
	}
	return strings.Trim(b.String(), "-") + ".bin"
}

// Two cases cannot share a golden.
func TestParityGoldenNamesAreDistinct(t *testing.T) {
	seen := map[string]string{}
	for _, c := range parityCases() {
		g := parityGoldenName(c.name)
		if prev, ok := seen[g]; ok {
			t.Errorf("%q and %q both keep their golden in %s", prev, c.name, g)
		}
		seen[g] = c.name
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

// compareParity applies a case's ceiling to two results: got against want,
// each named for the failure.
func compareParity(t *testing.T, c parityCase, want, got []byte, wantName, gotName string) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: %d bytes from %s and %d from %s",
			c.name, len(want), wantName, len(got), gotName)
	}
	if c.ceiling.Exact() {
		if r := numeq.Exact(got, want); !r.Equal {
			t.Fatalf("%s: %s against %s: %v\n  %s", c.name, gotName, wantName, r, c.ceiling)
		}
		return
	}
	w, g := asFloat32(t, want), asFloat32(t, got)
	var r numeq.Report
	if c.ceiling.Abs > 0 {
		r = withinAbsolute(g, w, c.ceiling.Abs)
	} else {
		r = numeq.WithinULP(g, w, c.ceiling.ULP)
	}
	if !r.Equal {
		t.Fatalf("%s: %s against %s: %v\n  %s", c.name, gotName, wantName, r, c.ceiling)
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
