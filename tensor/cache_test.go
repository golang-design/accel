// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package tensor_test

import (
	"math"
	"reflect"
	"strings"
	"testing"

	"golang.design/x/accel"
	"golang.design/x/accel/tensor"
)

// model records a small graph, parameterized so a test can vary one thing at a
// time and watch the identity move.
type model struct {
	n     int
	dtype accel.DType
	name  string
	extra bool
	scale bool
}

func (m model) record(b *tensor.Builder) {
	n := m.n
	if n == 0 {
		n = 64
	}
	dt := m.dtype
	if dt == 0 && m.dtype != accel.F32 {
		dt = accel.F32
	}
	name := m.name
	if name == "" {
		name = "x"
	}
	if m.scale {
		tensor.Scalar(b, tensor.ScalarDesc{Name: "f", Kind: tensor.ScalarF32})
	}
	x := tensor.Input(b, tensor.ValueDesc{Name: name, DType: dt, Shape: tensor.Shape{n}})
	h := tensor.SiLU(b, x)
	if m.extra {
		h = tensor.SiLU(b, h)
	}
	if m.scale {
		h = tensor.Scale(b, h, "f")
	}
	tensor.Output(b, "y", h)
}

func identityOf(t *testing.T, rt *tensor.Runtime, m model) tensor.Identity {
	t.Helper()
	b := rt.NewBuilder("id")
	m.record(b)
	if err := b.Err(); err != nil {
		t.Fatalf("record: %v", err)
	}
	return b.Identity()
}

// Two builders recording the same graph produce the same identity, and every
// structural change produces a different one.
//
// The negatives are the substance. An identity that only covered shapes would
// pass the first case and fail every one below it -- and would return one
// model's plan for another, which specs/007-tensor-layer.md names as the
// failure a shape-only key produces.
func TestIdentityDistinguishesWhatMatters(t *testing.T) {
	rt := newRuntime(t)
	base := model{}
	want := identityOf(t, rt, base)

	if got := identityOf(t, rt, base); got != want {
		t.Fatalf("the same graph recorded twice gave %v and %v", want, got)
	}

	for _, tc := range []struct {
		name string
		m    model
	}{
		{"a different shape", model{n: 128}},
		{"a different dtype", model{dtype: accel.F16}},
		{"a different port name", model{name: "z"}},
		{"an extra operator", model{extra: true}},
		{"a declared scalar", model{scale: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := identityOf(t, rt, tc.m); got == want {
				t.Errorf("%s did not change the identity, so a cache would return the "+
					"wrong plan for it", tc.name)
			}
		})
	}

	if want.String() == "" {
		t.Error("an identity should format for a log")
	}
}

// The identity covers the kernel a node selected, not only the operator.
//
// This is the component a naive cache omits and the one whose absence survives
// longest: a plan compiled before `go generate` ran and reused after would
// execute a lowering whose source no longer exists. Checked by varying the
// selection rather than by editing a digest, since the selection is the thing
// that carries it.
func TestIdentityCoversTheSelectedKernel(t *testing.T) {
	rt := newRuntime(t)

	matmul := func(m int) tensor.Identity {
		b := rt.NewBuilder("mm")
		x := tensor.Input(b, tensor.ValueDesc{
			Name: "x", DType: accel.F16, Shape: tensor.Shape{m, 32},
		})
		w := tensor.Weight(b, tensor.ValueDesc{
			Name: "w", DType: accel.F16, Shape: tensor.Shape{32, 8},
		})
		tensor.Output(b, "y", tensor.MatMul(b, x, w))
		if err := b.Err(); err != nil {
			t.Fatalf("record: %v", err)
		}
		return b.Identity()
	}
	// M=1 selects MatVec and M=4 selects the tiled GEMM. The shapes differ too,
	// so this alone does not prove the kernel is in the digest -- what it
	// proves is that two plans a cache must not confuse are not confused.
	if matmul(1) == matmul(4) {
		t.Error("a matrix-vector plan and a tiled GEMM plan share an identity")
	}
}

// A cache compiles once and returns the same plan afterwards.
func TestPlanCacheReturnsTheSamePlan(t *testing.T) {
	rt := newRuntime(t)
	cache := tensor.NewPlanCache(rt)
	defer cache.Close()

	first, err := cache.Compile(model{}.record, tensor.CompileOptions{Label: "m"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	second, err := cache.Compile(model{}.record, tensor.CompileOptions{Label: "m"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	if first != second {
		t.Error("the same graph compiled twice returned two plans")
	}
	if cache.Len() != 1 {
		t.Errorf("the cache holds %d plans for one graph", cache.Len())
	}

	third, err := cache.Compile(model{n: 128}.record, tensor.CompileOptions{Label: "m"})
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	if third == first {
		t.Error("a different shape returned the first plan")
	}
	if cache.Len() != 2 {
		t.Errorf("the cache holds %d plans for two graphs", cache.Len())
	}

	// The cached plan still works, which is the point: a hit returns something
	// submittable rather than a handle to a closed graph.
	d := rt.Device()
	f := first.Submit(d.Queue(), tensor.Bindings{Buffers: map[string]accel.BufferView{
		"x": f32Buffer(t, d, "x", make([]float32, 64)),
		"y": f32Buffer(t, d, "y", make([]float32, 64)),
	}})
	if err := f.Wait(); err != nil {
		t.Fatalf("submitting a cached plan: %v", err)
	}
}

// A cache reports a recording error rather than caching a broken graph.
func TestPlanCacheRefusesABrokenGraph(t *testing.T) {
	rt := newRuntime(t)
	cache := tensor.NewPlanCache(rt)
	defer cache.Close()

	_, err := cache.Compile(func(b *tensor.Builder) {
		x := tensor.Input(b, tensor.ValueDesc{
			Name: "x", DType: accel.F32, Shape: tensor.Shape{8},
		})
		y := tensor.Input(b, tensor.ValueDesc{
			Name: "y", DType: accel.F32, Shape: tensor.Shape{3},
		})
		tensor.Output(b, "z", tensor.Add(b, x, y))
	}, tensor.CompileOptions{})
	if err == nil {
		t.Fatal("a graph with mismatched shapes was cached")
	}
	if cache.Len() != 0 {
		t.Errorf("the cache holds %d plans after a failure", cache.Len())
	}
}

// Closing the cache closes its plans, and a runtime notices if it does not.
func TestPlanCacheClosesWhatItHolds(t *testing.T) {
	d, err := accel.OpenCPU(accel.CPUOptions{})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer d.Close()
	rt, err := tensor.NewRuntime(d)
	if err != nil {
		t.Fatalf("runtime: %v", err)
	}

	cache := tensor.NewPlanCache(rt)
	for _, n := range []int{16, 32, 64} {
		if _, err := cache.Compile(model{n: n}.record, tensor.CompileOptions{}); err != nil {
			t.Fatalf("compile: %v", err)
		}
	}
	if rt.Close() == nil {
		t.Error("a runtime closed with three cached plans still open")
	}
	if err := cache.Close(); err != nil {
		t.Fatalf("cache close: %v", err)
	}
	if err := cache.Close(); err != nil {
		t.Errorf("closing a cache twice must be harmless: %v", err)
	}
	if _, err := cache.Compile(model{}.record, tensor.CompileOptions{}); err == nil {
		t.Error("a closed cache compiled")
	}
	if err := rt.Close(); err != nil {
		t.Errorf("runtime close after its cache: %v", err)
	}
}

// A bucket set picks the smallest bucket that fits, and refuses what it cannot
// hold.
func TestBuckets(t *testing.T) {
	b, err := tensor.NewBuckets(128, 32, 512, 64)
	if err != nil {
		t.Fatalf("buckets: %v", err)
	}

	for _, tc := range []struct {
		n    int
		want int
	}{
		{1, 32}, {32, 32}, {33, 64}, {64, 64}, {65, 128}, {512, 512},
	} {
		got, err := b.For(tc.n)
		if err != nil {
			t.Fatalf("a prompt of %d: %v", tc.n, err)
		}
		if got != tc.want {
			t.Errorf("a prompt of %d chose bucket %d, want %d", tc.n, got, tc.want)
		}
	}

	// Longer than the largest is an error, not a truncation: truncating changes
	// what the model was asked and answers a different question plausibly.
	if _, err := b.For(513); err == nil {
		t.Error("a prompt longer than the largest bucket was accepted")
	} else if !strings.Contains(err.Error(), "answer a different question") {
		t.Errorf("the refusal should say why truncation is not the answer: %v", err)
	}
	if _, err := b.For(0); err == nil {
		t.Error("a prompt of no tokens was accepted")
	}

	if _, err := tensor.NewBuckets(); err == nil {
		t.Error("an empty bucket set was accepted")
	}
	if _, err := tensor.NewBuckets(32, 32); err == nil {
		t.Error("a duplicate bucket size was accepted")
	}
	if _, err := tensor.NewBuckets(32, -1); err == nil {
		t.Error("a negative bucket size was accepted")
	}
}

// A padded prefill produces the same rows for the real tokens as an
// exact-length plan does.
//
// This is what makes bucketing correct rather than merely cheap, and it is not
// obvious: the padded plan attends over a longer cache and computes rows nobody
// reads. It works because the mask is causal -- query position s sees at most
// cached position s -- so a real token's window never reaches the padding that
// follows it.
//
// If that reasoning were wrong the failure would be quiet: a model would give
// slightly different answers depending on which bucket a prompt landed in,
// which is indistinguishable from ordinary nondeterminism until somebody
// diffs two runs of the same prompt.
func TestPaddingDoesNotChangeTheRealRows(t *testing.T) {
	const (
		qHeads   = 2
		kvHeads  = 1
		headDim  = 8
		capacity = 16
		real     = 5  // the prompt
		bucket   = 12 // the bucket it runs in
	)
	rt := newRuntime(t)
	d := rt.Device()
	scale := float32(1) / float32(math.Sqrt(headDim))

	// One cache, filled once. Both plans read it; the padded one reads further
	// into it, which is exactly the situation a bucket creates.
	kv := make([]float32, capacity*kvHeads*headDim)
	for i := range kv {
		kv[i] = float32(math.Sin(float64(i) * 0.27))
	}
	q := make([]float32, bucket*qHeads*headDim)
	for i := range q {
		q[i] = float32(math.Cos(float64(i) * 0.19))
	}

	run := func(seq int) []float32 {
		t.Helper()
		b := rt.NewBuilder("prefill")
		tensor.Scalar(b, tensor.ScalarDesc{Name: "base", Kind: tensor.ScalarU32})
		tensor.Scalar(b, tensor.ScalarDesc{Name: "scale", Kind: tensor.ScalarF32})
		qt := tensor.Input(b, tensor.ValueDesc{
			Name: "q", DType: accel.F32, Shape: tensor.Shape{seq, qHeads, headDim},
		})
		kc := tensor.NewState(b, tensor.StateDesc{
			Name: "k", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
		})
		vc := tensor.NewState(b, tensor.StateDesc{
			Name: "v", DType: accel.F32, Shape: tensor.Shape{capacity, kvHeads, headDim},
		})
		lengths := tensor.Input(b, tensor.ValueDesc{
			Name: "len", DType: accel.U32, Shape: tensor.Shape{1},
		})
		tensor.Output(b, "out", tensor.Attention(b, qt, kc, vc, tensor.AttentionOptions{
			Lengths: lengths, ScaleName: "scale", BaseName: "base",
		}))
		plan, err := b.Compile(rt, tensor.CompileOptions{})
		if err != nil {
			t.Fatalf("compile at %d: %v", seq, err)
		}
		defer plan.Close()

		out := f32Buffer(t, d, "out", make([]float32, seq*qHeads*headDim))
		f := plan.Submit(d.Queue(), tensor.Bindings{
			Buffers: map[string]accel.BufferView{
				"q":   f32Buffer(t, d, "q", q[:seq*qHeads*headDim]),
				"k":   f32Buffer(t, d, "k", kv),
				"v":   f32Buffer(t, d, "v", kv),
				"out": out,
				// The length is the *real* prompt in both cases: a bucket pads
				// the query, not the cache, so the padded rows attend over the
				// same span and simply produce values nobody reads.
				"len": u32Buffer(t, d, "len", []uint32{real}),
			},
			Scalars: map[string]tensor.ScalarValue{
				"base": tensor.U32(0), "scale": tensor.F32(scale),
			},
		})
		if err := f.Wait(); err != nil {
			t.Fatalf("submit at %d: %v", seq, err)
		}
		got := make([]float32, seq*qHeads*headDim)
		if err := d.Queue().ReadBuffer(out.Buffer, 0, got); err != nil {
			t.Fatalf("readback: %v", err)
		}
		return got
	}

	exact := run(real)
	padded := run(bucket)

	for i := range exact {
		if exact[i] != padded[i] {
			t.Fatalf("row %d element %d differs between an exact plan (%v) and a padded one "+
				"(%v); a prompt would then answer differently depending on which bucket it "+
				"landed in", i/(qHeads*headDim), i%(qHeads*headDim), exact[i], padded[i])
		}
	}

	// And the padded plan really did compute more than it was compared on,
	// which is what makes the comparison meaningful rather than trivially true.
	if len(padded) <= len(exact) {
		t.Fatalf("the padded plan produced %d values and the exact one %d",
			len(padded), len(exact))
	}
}

// The block pool hands out blocks, takes them back, and refuses when empty.
//
// Refuses rather than evicts: choosing a victim is a policy question about
// which sequence matters, and a wrong answer silently truncates somebody's
// context.

// Sizes reports the bucket set smallest-first, and hands out a copy.
//
// The copy is the property worth guarding. [NewBuckets] is the only way to
// build a set, and it sorts precisely so that [Buckets.For]
// can pick the smallest bucket that fits by walking in order. Returning the
// backing slice would let any caller reorder or truncate that invariant from
// outside, and the symptom is not a panic: For would return a bucket that is
// not the smallest fit, so a prompt gets a larger plan than it needs and
// nothing reports anything.
//
// Nothing called Sizes at all before this — found by sweeping the surface for
// functions no test reaches.
func TestBucketSizesAreSortedAndCopied(t *testing.T) {
	b, err := tensor.NewBuckets(512, 64, 128)
	if err != nil {
		t.Fatalf("NewBuckets: %v", err)
	}

	got := b.Sizes()
	want := []int{64, 128, 512}
	if len(got) != len(want) {
		t.Fatalf("Sizes returned %v, want %v: the set is sorted", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Sizes returned %v, want %v", got, want)
		}
	}

	// Mutating what Sizes handed back must not reach the set. Reversing it is
	// the mutation that matters, because For depends on the order.
	for i, j := 0, len(got)-1; i < j; i, j = i+1, j-1 {
		got[i], got[j] = got[j], got[i]
	}
	if again := b.Sizes(); again[0] != 64 {
		t.Fatalf("after reversing the slice Sizes returned, the set reads %v: Sizes "+
			"handed out its backing array, so a caller can reorder the invariant For "+
			"walks and get a bucket that is not the smallest fit", again)
	}

	// The order is what For consumes, so assert it through For as well rather
	// than only through the accessor that reports it.
	if n, err := b.For(65); err != nil || n != 128 {
		t.Fatalf("For(65) = %d, %v; want 128 and no error", n, err)
	}

	// A duplicate is refused rather than collapsed, which is what the
	// constructor's documentation said it did until writing this test read the
	// code. Collapsing would compile one plan where a caller asked for two.
	if _, err := tensor.NewBuckets(128, 64, 128); err == nil {
		t.Fatal("a repeated bucket size was accepted; a caller's list saying " +
			"something they did not mean should be refused rather than tidied")
	}
}

// A new CompileOptions field fails until someone decides whether the plan-cache
// key covers it.
//
// specs/029-plan-cache.md §2 says the key is "every compile option that affects
// lowering". The digest hashes the constant string "opts v1" and no option at
// all (tensor/cache.go). That is *correct today* — CompileOptions carries only
// Label, which the spec excludes on purpose, since two plans differing in name
// are the same plan. It stops being correct the moment the struct grows.
//
// The failure it would cause is the worst kind this cache can have: a second
// compile with a different option returns the first plan, silently, because the
// key could not tell them apart. Nothing about the graph would look wrong.
//
// So the field set is pinned. This is the same reflection guard
// TestEveryAttentionOptionReachesTheKernelOrIsRefused uses on AttentionOptions,
// for the same reason — a struct that grows a field nobody wired up is a defect
// this package has shipped before.
func TestCompileOptionsFieldsAreAccountedForInTheKey(t *testing.T) {
	// Every field, and whether the key must hash it.
	inKey := map[string]bool{
		// Excluded by 029 §2: two plans differing only in what they are called
		// are the same plan, and including it would double the cache for nothing.
		"Label": false,
	}

	ty := reflect.TypeOf(tensor.CompileOptions{})
	for i := range ty.NumField() {
		f := ty.Field(i)
		if _, known := inKey[f.Name]; !known {
			t.Errorf("CompileOptions.%s is new and nothing here says whether the "+
				"plan-cache key covers it.\n"+
				"  tensor/cache.go hashes a constant, so an option that changes "+
				"lowering would let a second compile return the first plan — "+
				"silently, since the key cannot tell them apart.\n"+
				"  Decide, hash it in key() if it affects lowering, and add it above "+
				"with the reason (specs/029-plan-cache.md section 2).", f.Name)
		}
	}
	for name := range inKey {
		if _, ok := ty.FieldByName(name); !ok {
			t.Errorf("CompileOptions.%s is listed here and no longer exists", name)
		}
	}
}
