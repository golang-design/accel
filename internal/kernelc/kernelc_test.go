// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package kernelc_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc"
)

const corpus = "./internal/testkernels"

func root(t testing.TB) string {
	t.Helper()
	abs, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return abs
}

// TestCommittedFileIsFresh is the check CI runs.
//
// A kernel edited without regenerating leaves a source and a generated file
// that disagree, and nothing else notices, because both halves still compile.
// The generated file is committed precisely so this comparison exists: a
// generator whose output is not in the tree can only be checked by running it,
// which proves it agrees with itself.
func TestCommittedFileIsFresh(t *testing.T) {
	results, err := kernelc.Run(kernelc.Options{Dir: root(t), Patterns: []string{corpus}, Check: true})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("checked %d packages, want 1", len(results))
	}
	if results[0].Stale != "" {
		t.Errorf("the committed generated file is stale:\n%s", results[0].Stale)
	}
	if results[0].Written {
		t.Error("check mode wrote a file")
	}
	// Sorted, so adding a kernel to the corpus does not reorder this. The three
	// stage names are here too: specs/032-stage-abi.md compiles them through the
	// same front end, and a list that omitted them would let a stage vanish.
	want := []string{"Add", "AtomicAddF32", "AtomicOps", "AtomicOpsI32", "AttentionDecode", "AttentionDecodeBatched", "AttentionDecodeF16", "AttentionDecodePaged", "AttentionDecodePagedF16", "AttentionPrefill", "AttentionPrefillF16", "AttentionPrefillPaged", "AttentionRagged", "AttributeVS", "BlitFS", "CastBF16ToF32", "CastF16ToF32", "CastF32ToF16", "CountAbove", "CountWorkgroups", "DisplacedVS", "ElemAdd", "ElemBias", "ElemMul", "ElemScale", "Exchange", "FullScreenVS", "GatherRows", "GatherRowsF16", "GeometryVS", "HalfTriangleVS", "Histogram", "LinearAttention", "LinearTiled", "MatMulTiled", "MatMulTiledF32", "MatMulTiledF32F16", "MatVec", "Normalize", "Pack", "PenaltyApply", "PenaltyClear", "PenaltyCount", "QuantMatMul", "QuantMatMulF32", "QuantMatVec", "QuantMatVecF32", "QuantMatVecInt4", "QuantRows", "RMSNorm", "ReduceLoop", "ReduceSum", "ReduceUnrolled", "RoPE", "SampleArgmax", "SampleCategorical", "SampledFS", "Scale", "ScaledVS", "ScatterRows", "ScatterRowsF16", "SegmentOffsets", "SegmentSum", "ShadeFS", "SiLU", "Softmax", "SolidFS", "SubgroupReduce", "SubgroupReduceFallback", "SubgroupScan", "SubgroupScanFallback", "SubgroupShuffleMix", "SubgroupShuffleMixFallback", "SwiGLU", "TintFS", "TintedFS", "TopKMask", "TopPMask", "Transform"}
	if len(results[0].Kernels) != len(want) {
		t.Fatalf("kernels = %v, want %v", results[0].Kernels, want)
	}
	for i := range want {
		if results[0].Kernels[i] != want[i] {
			t.Errorf("kernels = %v, want %v", results[0].Kernels, want)
			break
		}
	}
}

// TestGeneratingIsIdempotent checks that a second run over unchanged source
// writes nothing. A generator that rewrites an identical file makes every build
// dirty a working tree, and a dirty tree is how a real change goes unreviewed.
func TestGeneratingIsIdempotent(t *testing.T) {
	dir := copyCorpus(t)

	first, err := kernelc.Run(kernelc.Options{Dir: dir, Patterns: []string{"./kernels"}})
	if err != nil {
		t.Fatalf("first run: %v", err)
	}
	if len(first) != 1 || !first[0].Written {
		t.Fatalf("the first run wrote nothing: %+v", first)
	}

	second, err := kernelc.Run(kernelc.Options{Dir: dir, Patterns: []string{"./kernels"}})
	if err != nil {
		t.Fatalf("second run: %v", err)
	}
	if len(second) != 1 || second[0].Written {
		t.Errorf("a second run over unchanged source rewrote the file")
	}
}

// TestStaleFileIsReportedWithItsCause is what makes a freshness failure
// actionable. "The generated file is stale" sends a reader to diff two files by
// eye; the digest's preimage is the list of inputs, so the one that moved is
// the answer.
func TestStaleFileIsReportedWithItsCause(t *testing.T) {
	dir := copyCorpus(t)
	if _, err := kernelc.Run(kernelc.Options{Dir: dir, Patterns: []string{"./kernels"}}); err != nil {
		t.Fatal(err)
	}

	// Edit the kernel without regenerating, which is the mistake.
	src := filepath.Join(dir, "kernels", "scale.go")
	b, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(src, []byte(strings.Replace(string(b), "* 2", "* 3", 1)), 0o644); err != nil {
		t.Fatal(err)
	}

	results, err := kernelc.Run(kernelc.Options{Dir: dir, Patterns: []string{"./kernels"}, Check: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Stale == "" {
		t.Fatalf("an edited kernel was reported fresh: %+v", results)
	}
	for _, want := range []string{
		"go generate", "kernel Scale", "source\t",
		"binding\t", "intrinsic\taccel.Thread.GlobalID", "generator/",
	} {
		if !strings.Contains(results[0].Stale, want) {
			t.Errorf("the explanation does not carry %q:\n%s", want, results[0].Stale)
		}
	}
}

// TestMissingFileIsStale covers the first run in check mode, which is what a
// fresh checkout of a repository that forgot to commit its generated files
// looks like.
func TestMissingFileIsStale(t *testing.T) {
	dir := copyCorpus(t)
	results, err := kernelc.Run(kernelc.Options{Dir: dir, Patterns: []string{"./kernels"}, Check: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Stale == "" {
		t.Fatalf("a missing generated file was reported fresh: %+v", results)
	}
	if !strings.Contains(results[0].Stale, "does not exist") {
		t.Errorf("the explanation does not say the file is absent:\n%s", results[0].Stale)
	}
}

// TestRejectedKernelIsAnError checks that a package the front end refuses stops
// the generator rather than producing a file for the kernels that happened to
// pass. A partially generated package compiles and is missing a kernel, which
// is a link error at best and a silently absent code path at worst.
func TestRejectedKernelIsAnError(t *testing.T) {
	dir := copyCorpus(t)
	src := filepath.Join(dir, "kernels", "scale.go")
	b, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	broken := strings.Replace(string(b),
		"i := t.GlobalID().X",
		"i := t.GlobalID().X\n\tswitch {\n\t}", 1)
	if err := os.WriteFile(src, []byte(broken), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err = kernelc.Run(kernelc.Options{Dir: dir, Patterns: []string{"./kernels"}})
	if err == nil {
		t.Fatal("a rejected kernel produced no error")
	}
	if !strings.Contains(err.Error(), "outside the closed IR node set") {
		t.Errorf("the error does not carry the rejection: %v", err)
	}
	if _, statErr := os.Stat(filepath.Join(dir, "kernels", kernelc.GeneratedFile)); statErr == nil {
		t.Error("a file was written for a package whose kernel was rejected")
	}
}

// TestPackageWithoutKernelsIsSkipped checks that an ordinary package is not an
// error. A corpus is loaded by pattern and most packages in one have no kernels.
func TestPackageWithoutKernelsIsSkipped(t *testing.T) {
	results, err := kernelc.Run(kernelc.Options{
		Dir: root(t), Patterns: []string{"golang.design/x/accel/internal/alloc"},
	})
	if err != nil {
		t.Fatalf("a package with no kernels was an error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("results = %+v, want none", results)
	}
}

// TestUnloadablePatternIsAnError checks the failure a typo in a go:generate
// line produces.
func TestUnloadablePatternIsAnError(t *testing.T) {
	if _, err := kernelc.Run(kernelc.Options{Dir: root(t), Patterns: []string{"./nosuchpackage"}}); err == nil {
		t.Error("a pattern matching nothing was accepted")
	}
}

// TestDefaultsAreUsable checks the zero-ish options a go:generate line relies
// on, since a directive that has to spell out every flag is one people get
// wrong.
func TestDefaultsAreUsable(t *testing.T) {
	dir := copyCorpus(t)
	results, err := kernelc.Run(kernelc.Options{Dir: filepath.Join(dir, "kernels")})
	if err != nil {
		t.Fatalf("Run with default patterns: %v", err)
	}
	if len(results) != 1 || !results[0].Written {
		t.Fatalf("results = %+v", results)
	}
	if filepath.Base(results[0].Path) != kernelc.GeneratedFile {
		t.Errorf("wrote %s, want %s", results[0].Path, kernelc.GeneratedFile)
	}
}

// copyCorpus builds a throwaway module holding a copy of the corpus kernel.
//
// A throwaway module, because these tests edit a kernel and check what the
// generator says, and doing that in the repository would leave a working tree
// whose state depends on whether the last test cleaned up.
func copyCorpus(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	repo := root(t)
	write(t, filepath.Join(dir, "go.mod"), moduleFile(t, repo, "example.com/kerneltest"))

	// The replace target has its own requirements, so the sum file is inherited
	// rather than resolved again.
	if b, err := os.ReadFile(filepath.Join(repo, "go.sum")); err == nil {
		write(t, filepath.Join(dir, "go.sum"), string(b))
	}

	src, err := os.ReadFile(filepath.Join(repo, "internal", "testkernels", "scale.go"))
	if err != nil {
		t.Fatal(err)
	}
	body := strings.Replace(string(src), "package testkernels", "package kernels", 1)
	body = strings.Replace(body, "//go:generate go run golang.design/x/accel/cmd/accel-kernel -C ../.. ./internal/testkernels\n", "", 1)
	write(t, filepath.Join(dir, "kernels", "scale.go"), body)
	return dir
}

func write(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// moduleFile builds the throwaway module's go.mod from the repository's own.
//
// Derived rather than written, because a hand-written require line goes stale
// the moment accel gains a dependency, and the failure is opaque: the generator
// reports "updates to go.mod needed" from inside a temp directory the reader
// cannot see. Taking the repository's requirements verbatim means the throwaway
// module has exactly what the replace target needs, always.
func moduleFile(t *testing.T, repo, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(repo, "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	body := strings.Replace(string(b), "module golang.design/x/accel", "module "+name, 1)
	return body + "\nrequire golang.design/x/accel v0.0.0\n\n" +
		"replace golang.design/x/accel => " + repo + "\n"
}

// The check CI actually runs, over the whole module rather than one package.
//
// The generator hides its own previous output from the type checker so a
// generated file that no longer compiles cannot block the command that would
// rewrite it. Hiding it unconditionally broke every *other* package that
// imports the generated declarations, which a ./... pattern loads too -- so the
// per-package check passed and the CI gate failed. This is that gate.
func TestTheWholeModuleChecks(t *testing.T) {
	results, err := kernelc.Run(kernelc.Options{Dir: root(t), Patterns: []string{"./..."}, Check: true})
	if err != nil {
		t.Fatalf("checking ./...: %v", err)
	}
	var withKernels int
	for _, r := range results {
		if r.Stale != "" {
			t.Errorf("%s is stale:\n%s", r.Package, r.Stale)
		}
		if len(r.Kernels) > 0 {
			withKernels++
		}
	}
	if withKernels == 0 {
		t.Error("no package in the module reported a kernel, so this checked nothing")
	}
}
