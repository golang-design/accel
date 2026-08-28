// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package accel_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The root package must not depend on the compiler's toolchain.
//
// specs/012-kernel-pipeline.md asserts twice that a test proves this, and until
// now no such test existed — the fact was true and nothing held it there. It
// matters because a kernel is compiled ahead of time by cmd/accel-kernel, which
// needs golang.org/x/tools/go/packages and the go tool; a deployed binary
// linking accel must need neither. One import from the root package into that
// tree would pull a type checker and a subprocess launcher into every program
// that uses this library, and nothing about the build would look wrong.
//
// The check is on the whole transitive graph rather than on this package's own
// import block, because the way this regresses is indirect: a helper added to
// internal/kernel that reaches for go/packages is the plausible mistake, not an
// import written at the top of a file here.
func TestRootPackageDoesNotDependOnTheToolchain(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "golang.design/x/accel").Output()
	if err != nil {
		t.Skipf("go list unavailable: %v", err)
	}
	var bad []string
	for _, dep := range strings.Fields(string(out)) {
		switch {
		case strings.HasPrefix(dep, "golang.org/x/tools"),
			dep == "go/packages", dep == "go/types", dep == "go/parser", dep == "go/ast":
			bad = append(bad, dep)
		}
	}
	if len(bad) > 0 {
		t.Errorf("the root package transitively depends on the kernel compiler's "+
			"toolchain: %s\n\nA program that merely runs a compiled kernel would link a "+
			"type checker. Kernels are compiled ahead of time by cmd/accel-kernel; only "+
			"that command may reach these.", strings.Join(bad, ", "))
	}
}

// 000's layering rules, which nothing checked.
//
// specs/000-decisions.md is normative — "a spec that contradicts it is wrong" —
// and states four layering rules. The 2026-08-27 audit found that **no test
// checked any of them**: three were true of the tree and held there by nothing,
// and the fourth was already violated. That is the shape STATUS.md's fourth tier
// is about, since a normative rule with no check is a rule that decays quietly.
//
// Rule 3, "no backend-specific type appears in a public signature", is not here.
// It is violated by OpenCPU, CPUOptions and CPUMode, and closing it means either
// changing the public surface or amending the rule — a decision rather than a
// test. Asserting a rule the tree breaks would mean skipping or weakening it,
// and a weakened rule is worse than an absent one because it reads as enforced.
func TestTheLayeringRulesHold(t *testing.T) {
	// Rule 1: layer 1 never imports layer 2. The tensor layer is built on the
	// device layer, and a dependency the other way would make the split
	// decorative — and would pull a whole inference stack into a program that
	// only wanted to dispatch a kernel.
	t.Run("layer 1 does not import layer 2", func(t *testing.T) {
		out, err := exec.Command("go", "list", "-deps", "golang.design/x/accel").Output()
		if err != nil {
			t.Skipf("go list unavailable: %v", err)
		}
		for _, dep := range strings.Fields(string(out)) {
			if strings.HasPrefix(dep, "golang.design/x/accel/tensor") {
				t.Errorf("the root package transitively depends on %s, so layer 1 "+
					"imports layer 2 and 000's rule 1 is broken", dep)
			}
		}
	})

	// Rule 2: backends implement an unexported interface, so adding one touches
	// no public API. Checked as the property that makes it true — the interface
	// lives in an internal package, which the language then enforces.
	t.Run("the backend interface is unexported", func(t *testing.T) {
		out, err := exec.Command("go", "list", "-deps", "golang.design/x/accel").Output()
		if err != nil {
			t.Skipf("go list unavailable: %v", err)
		}
		found := false
		for _, dep := range strings.Fields(string(out)) {
			if dep == "golang.design/x/accel/internal/driver" {
				found = true
			}
		}
		if !found {
			t.Error("the root package does not reach internal/driver, so this test is " +
				"no longer looking at the backend interface")
		}
		// An internal package cannot be named by a caller outside the module, so
		// the rule holds by construction as long as the interface lives there.
		// What would break it is moving driver out of internal/, which is what
		// this asserts against.
	})

	// Rule 4: the CPU backend is always buildable on every platform and is never
	// build-tagged away. It is the oracle every differential compares against,
	// so a platform without it is a platform where nothing can be proven.
	t.Run("the CPU backend is never build-tagged away", func(t *testing.T) {
		files, err := filepath.Glob("internal/cpu/*.go")
		if err != nil || len(files) == 0 {
			t.Fatalf("no CPU backend sources found: %v", err)
		}
		for _, f := range files {
			if strings.HasSuffix(f, "_test.go") {
				continue
			}
			src, err := os.ReadFile(f)
			if err != nil {
				t.Fatalf("read %s: %v", f, err)
			}
			for _, line := range strings.Split(string(src), "\n") {
				line = strings.TrimSpace(line)
				if !strings.HasPrefix(line, "//go:build") {
					continue
				}
				// A constraint that is always satisfied is not a removal. Any
				// other one means some platform builds without the oracle.
				t.Errorf("%s carries %q: the CPU backend must build everywhere, "+
					"because it is the oracle every other backend is checked "+
					"against (000 rule 4)", f, line)
			}
		}
	})
}

// Every corpus kernel either reaches a tensor operator or is a named
// device-layer proof.
//
// specs/010-kernel-corpus.md quotes the rule — "a kernel is not a capability,
// and a corpus entry with no operator reaching it is recorded as unreachable
// rather than as done" — and the 2026-08-27 audit found it had never been
// applied to 010 itself. Fifty of seventy-two kernels reach a tensor operator,
// and four of the rest carried rows that read as done.
//
// The unreached set is pinned rather than reported, which is the difference
// between a guard and a note. Every name below is a kernel some spec *below* the
// tensor layer needs — a numeric proof, an atomic differential, a subgroup
// sweep, a graphics stage — so giving them operators would be inventing callers.
// What must not happen silently is a *new* kernel joining them, because that is
// how a corpus entry starts reading as a capability nobody can use. Adding one
// fails here until someone says which it is.
func TestUnreachableKernelsAreTheOnesWeNamed(t *testing.T) {
	// Kernels no tensor operator reaches, and the spec that needs each.
	proofs := map[string]string{
		"ReduceSum":      "008's tree-reduction proof",
		"ReduceLoop":     "018's lowering shapes",
		"ReduceUnrolled": "018's lowering shapes",
		"ElemBias":       "a broadcast gap recorded in 010",
		"AtomicOps":      "020's unsigned atomics",
		"AtomicOpsI32":   "020's signed atomics",
		"AtomicAddF32":   "020's float-atomic capability",
		"Exchange":       "020's exchange family",
		"Histogram":      "020's contention case",
		// Not the dispatch shape, which is what this row used to say. It
		// counts workgroups *atomically* so 003's indirect clamp is
		// observable, and the two are only adjacent in subject: nothing in it
		// asks a Thread for a shape. The accessors' proof is DispatchShape
		// below, added when 052 built them.
		"CountWorkgroups":            "003's indirect-clamp proof",
		"DispatchShape":              "052's three accessors",
		"ShapeBoundedSum":            "052's compile-time-uniform loop bound",
		"PublishStorage":             "050's storage-scope barrier",
		"PublishShared":              "050's shared-scope barrier",
		"SubgroupPublish":            "050's subgroup-scope barrier",
		"SubgroupStagger":            "050's per-subgroup arrival check",
		"Ballot":                     "058's ballot and its five mask methods",
		"IntReduce":                  "059's integer minima and maxima",
		"BitReduce":                  "059's bitwise reductions",
		"MulReduce":                  "059's product reductions",
		"CountAbove":                 "011's harness fixture",
		"SegmentSum":                 "046's prefix-sum proof",
		"Normalize":                  "028's sampling primitive",
		"Scale":                      "012's straight-line pipeline",
		"Transform":                  "014's uniform round trip",
		"Add":                        "012's first kernel end to end",
		"SaturatingConvert":          "051's boundary differential",
		"SubgroupReduce":             "020's subgroup sweep",
		"SubgroupReduceFallback":     "020's subgroup sweep",
		"SubgroupScan":               "020's subgroup sweep",
		"SubgroupScanFallback":       "020's subgroup sweep",
		"SubgroupShuffleMix":         "020's subgroup sweep",
		"SubgroupShuffleMixFallback": "020's subgroup sweep",
	}

	src, err := os.ReadFile("internal/testkernels/accel_kernels.go")
	if err != nil {
		t.Fatalf("read corpus: %v", err)
	}
	var names []string
	for _, line := range strings.Split(string(src), "\n") {
		const p = "var "
		if !strings.HasPrefix(line, p) || !strings.Contains(line, "Kernel = kernelabi.Kernel{") {
			continue
		}
		names = append(names, strings.TrimSuffix(
			strings.Fields(strings.TrimPrefix(line, p))[0], "Kernel"))
	}
	if len(names) < 50 {
		t.Fatalf("found %d kernels in the corpus, too few to be reading it correctly",
			len(names))
	}

	operators, err := filepath.Glob("tensor/*.go")
	if err != nil {
		t.Fatal(err)
	}
	var reached strings.Builder
	for _, f := range operators {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		reached.Write(b)
	}
	body := reached.String()

	seen := map[string]bool{}
	for _, n := range names {
		if strings.Contains(body, "testkernels."+n+"Kernel") {
			continue
		}
		// Graphics stages are 032's and are identified by their ABI suffix
		// rather than listed, since that set grows with the stage corpus.
		if strings.HasSuffix(n, "VS") || strings.HasSuffix(n, "FS") {
			continue
		}
		seen[n] = true
		if _, ok := proofs[n]; !ok {
			t.Errorf("%s reaches no tensor operator and is not a named device-layer "+
				"proof.\n  010's rule: a kernel with no operator reaching it is recorded "+
				"as unreachable rather than as done. Either give it an operator, or add "+
				"it above with the spec that needs it.", n)
		}
	}
	for n := range proofs {
		if !seen[n] {
			t.Errorf("%s is listed as unreachable and an operator now reaches it, or it "+
				"no longer exists; remove it from the list", n)
		}
	}
}

// No new ad hoc float comparison enters the tree.
//
// specs/011-conformance-harness.md §4: "the static conformance check scans
// comparison call sites and rejects known ad hoc helpers or numeric tolerance
// parameters." The 2026-08-27 audit found it had no code, and
// specs/008-numerics.md §9's rule — that a tolerance is derived and never tuned
// — had nothing holding it up.
//
// # Why this is a ratchet and not a refusal
//
// The tree does not satisfy the rule yet: there are dozens of
// `math.Abs(got-want) > 1e-5` sites, written before anything checked. Turning
// the rule on as a hard refusal would mean rewriting all of them in one change,
// and every rewrite risks altering what a test asserts — which is the one thing
// a conformance change must not do quietly. So the count is pinned. Existing
// sites stay and are visible; a new one fails.
//
// Lowering the number as sites move to numeq is the intended direction, and the
// constant is meant to be edited downward.
//
// # What this check cannot see
//
// A derived bound and a tuned tolerance are syntactically identical. `> 1e-5`
// might be somebody's guess or might be 008 §7's reduction bound written out.
// This counts the shape, not the reasoning, which is why §4's rule cannot be
// fully mechanical and why the number below is a budget rather than a verdict on
// any one site.
func TestNoNewAdHocFloatComparisons(t *testing.T) {
	// The count when the ratchet was installed. Lower it, never raise it.
	const budget = 74

	var files []string
	for _, pat := range []string{"*_test.go", "*/*_test.go", "*/*/*_test.go"} {
		m, err := filepath.Glob(pat)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, m...)
	}
	if len(files) < 20 {
		t.Fatalf("found %d test files, too few to be scanning the tree", len(files))
	}

	// math.Abs(...) compared against a bare numeric literal: a threshold chosen
	// rather than computed. A bound held in a variable is excluded, since that
	// is the shape a derived one takes.
	adHoc := regexp.MustCompile(`math\.Abs\(.*\)\s*[<>]=?\s*[0-9]`)

	found := 0
	perFile := map[string]int{}
	for _, f := range files {
		if strings.Contains(f, "worktrees") {
			continue
		}
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		for _, line := range strings.Split(string(src), "\n") {
			if adHoc.MatchString(line) {
				found++
				perFile[f]++
			}
		}
	}

	if found > budget {
		worst := ""
		for f, n := range perFile {
			if n > perFile[worst] {
				worst = f
			}
		}
		t.Errorf("%d ad hoc float comparisons, and the budget is %d.\n"+
			"  A threshold written as a literal is a tolerance chosen rather than "+
			"derived, which specs/008-numerics.md §9 rejects and "+
			"specs/011-conformance-harness.md §4 exists to catch.\n"+
			"  Use internal/conformance/numeq, or compute the bound from 008's table "+
			"and hold it in a named variable so a reader can see where it came from.\n"+
			"  Most sites today: %s (%d).", found, budget, worst, perFile[worst])
	}
	if found < budget {
		t.Logf("%d ad hoc comparisons, under the budget of %d — lower the constant "+
			"in this test to hold the ground", found, budget)
	}
}
