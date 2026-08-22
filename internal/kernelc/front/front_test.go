// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package front_test

import (
	"fmt"
	"go/token"
	"path/filepath"
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/front"
	"golang.design/x/accel/internal/kernelc/ir"
)

// TestChecksTheCorpusKernel builds spec 012's kernel from its real source.
func TestChecksTheCorpusKernel(t *testing.T) {
	pkgs, err := front.Load(repoRoot(t), "golang.design/x/accel/internal/testkernels")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(pkgs) != 1 {
		t.Fatalf("loaded %d packages, want 1", len(pkgs))
	}

	kernels, diags := front.Check(pkgs[0])
	if len(diags) > 0 {
		t.Fatalf("the corpus was rejected:\n%v", diags)
	}
	var k *ir.Func
	for _, c := range kernels {
		if c.Name == "Scale" {
			k = c
		}
	}
	if k == nil {
		t.Fatalf("the corpus has no Scale among %d kernels", len(kernels))
	}

	if k.Name != "Scale" {
		t.Errorf("name is %q", k.Name)
	}
	if k.Workgroup != [3]uint32{64, 1, 1} {
		t.Errorf("workgroup is %v, want 64,1,1: an omitted axis defaults to 1", k.Workgroup)
	}
	if k.Thread != 0 {
		t.Errorf("Thread parameter is at index %d, want 0", k.Thread)
	}

	// The signature is the binding layout, so the bindings come from it and the
	// accesses come from the body.
	if len(k.Bindings) != 2 {
		t.Fatalf("found %d bindings, want 2", len(k.Bindings))
	}
	in, out := k.Bindings[0], k.Bindings[1]
	if in.Name != "in" || !in.Read || in.Write {
		t.Errorf("in is %+v, want read and not written", in)
	}
	if out.Name != "out" || !out.Write {
		t.Errorf("out is %+v, want written", out)
	}
	// out is read by len(out) only, which is not an element read.
	if out.Read {
		t.Errorf("out is marked read; len is not an element access")
	}

	if want := []string{"accel.Thread.GlobalID"}; len(k.Intrinsics) != 1 || k.Intrinsics[0] != want[0] {
		t.Errorf("intrinsics are %v, want %v recorded by authored spelling", k.Intrinsics, want)
	}

	// The body is the shape the IR test built by hand at the previous slice.
	if len(k.Body.List) != 2 {
		t.Fatalf("body has %d statements, want a declaration and an if", len(k.Body.List))
	}
	decl, ok := k.Body.List[0].(*ir.Declare)
	if !ok {
		t.Fatalf("first statement is %T", k.Body.List[0])
	}
	if decl.Local.Type().Kind != ir.U32 {
		t.Errorf("i has kind %v, want u32: the id components are uint32, which is what lets "+
			"i < uint32(len(out)) typecheck without a conversion", decl.Local.Type().Kind)
	}
	sel := decl.Init.(*ir.FieldSel)
	if sel.Name != "X" || sel.Index != 0 {
		t.Errorf("selected %q at index %d, want X at 0", sel.Name, sel.Index)
	}
	call := sel.X.(*ir.IntrinsicCall)
	if call.Op != ir.OpGlobalID || call.Recv == nil {
		t.Errorf("intrinsic is %v with receiver %v", call.Op, call.Recv)
	}

	cons := k.Body.List[1].(*ir.If)
	store := cons.Then.List[0].(*ir.Assign)
	if idx := store.LHS.(*ir.IndexExpr); idx.Binding != 2 {
		t.Errorf("the store names binding %d, want 2", idx.Binding)
	}
	mul := store.RHS.(*ir.Binary)
	if mul.Op != token.MUL {
		t.Errorf("stored value uses %v, want *", mul.Op)
	}
	// The literal 2 arrives with its resolved type rather than as an untyped
	// int, which is where the GLSL integer-literal divergence is settled.
	if lit, ok := mul.Y.(*ir.Const); !ok {
		t.Errorf("the literal is %T, want a constant", mul.Y)
	} else if lit.Type().Kind != ir.F32 {
		t.Errorf("the literal 2 has kind %v, want f32: go/types resolved it against the "+
			"multiplication, and the emitter spells it accordingly", lit.Type().Kind)
	}
}

// TestRejections is the corpus. Every case asserts the message *and* the line,
// because a diagnostic naming the right problem at the wrong line sends a
// reader to the wrong place.
//
// Cases are loaded as overlay packages that do not exist on disk, in one
// packages.Load for the whole set: a corpus costing a module resolution per
// case is one nobody runs, and spec 013 makes this corpus the executable form
// of the subset.
func TestRejections(t *testing.T) {
	cases := []struct {
		name string
		body string
		line int    // 1-based line within body, not within the file
		want string // substring of the message
	}{
		{
			name: "generic kernel",
			body: `//accel:kernel workgroup=64
func K[T any](t accel.Thread, out []float32) { _ = t }`,
			line: 2, want: "generic kernels are out of scope for v0",
		},
		{
			name: "range",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	for range out {
	}
}`,
			line: 3, want: "write a three-clause loop",
		},
		{
			name: "switch",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	switch t.GlobalID().X {
	}
}`,
			line: 3, want: "outside the closed IR node set",
		},
		{
			name: "defer",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	defer func() {}()
	out[0] = 1
}`,
			line: 3, want: "no unwinding on a GPU",
		},
		{
			name: "goroutine",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	go func() {}()
	out[0] = 1
}`,
			line: 3, want: "launched by the dispatch",
		},
		{
			name: "shared parameter",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, tile *[64]float32, out []float32) { out[0] = 1 }`,
			line: 2, want: "cooperative kernels arrive at M4",
		},
		{
			name: "no thread",
			body: `//accel:kernel workgroup=64
func K(out []float32) { out[0] = 1 }`,
			line: 2, want: "first parameter must be accel.Thread",
		},
		{
			name: "thread not first",
			body: `//accel:kernel workgroup=64
func K(out []float32, t accel.Thread) { out[0] = 1 }`,
			line: 2, want: "must be the first parameter",
		},
		{
			name: "returns a value",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) uint32 { return 0 }`,
			line: 2, want: "returns a value",
		},
		{
			name: "no workgroup extent",
			body: `//accel:kernel
func K(t accel.Thread, out []float32) { out[0] = 1 }`,
			line: 2, want: "needs a workgroup extent",
		},
		{
			name: "zero workgroup extent",
			body: `//accel:kernel workgroup=0
func K(t accel.Thread, out []float32) { out[0] = 1 }`,
			line: 2, want: "needs a workgroup extent",
		},
		{
			name: "unsupported element type",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float64) { out[0] = 1 }`,
			line: 2, want: "not one of float32, int32, uint32",
		},
		{
			name: "barrier",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	t.Barrier()
	out[0] = 1
}`,
			line: 3, want: "cooperative kernels arrive at M4",
		},
		{
			name: "unbound binding",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, in []float32, out []float32) { out[0] = 1 }`,
			line: 2, want: `binding "in" is never read or written`,
		},
		{
			name: "map",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	m := map[uint32]float32{}
	out[0] = m[0]
}`,
			line: 3, want: "outside the closed IR node set",
		},
		{
			name: "builtin append",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	_ = append(out, 1)
	out[0] = 1
}`,
			line: 3, want: "no allocator and no runtime",
		},
		{
			name: "unknown call",
			body: `func other(x uint32) uint32 { return x }

//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	out[other(0)] = 1
}`,
			line: 5, want: "is not marked //accel:helper",
		},
		{
			name: "reslice",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	s := out[0:1]
	s[0] = 1
}`,
			line: 3, want: "a binding's extent is fixed by its descriptor",
		},
		{
			name: "no bindings",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread) { _ = t }`,
			line: 2, want: "nowhere to put a result",
		},
	}

	cases = append(cases, moreRejections...)
	cases = append(cases, nestedRejections()...)

	// One Load for every case.
	files := make(map[string]string, len(cases))
	patterns := make([]string, 0, len(cases))
	for i, tc := range cases {
		dir := fmt.Sprintf("rejectcase%02d", i)
		files[dir] = header(dir) + tc.body + "\n"
		patterns = append(patterns, "./internal/kernelc/front/"+dir)
	}
	pkgs := loadOverlay(t, files, patterns)

	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pkg := pkgs[patterns[i]]
			if pkg == nil {
				t.Fatalf("case package %s did not load", patterns[i])
			}
			_, diags := front.Check(pkg)
			if len(diags) == 0 {
				t.Fatalf("was accepted; every case here is outside the subset")
			}
			var found bool
			for _, d := range diags {
				if strings.Contains(d.Msg, tc.want) {
					found = true
					if want := tc.line + headerLines; d.Pos.Line != want {
						t.Errorf("reported at line %d, want %d (body line %d): a diagnostic at "+
							"the wrong line sends a reader to the wrong place\n  %v",
							d.Pos.Line, want, tc.line, d)
					}
				}
			}
			if !found {
				t.Errorf("no diagnostic carries %q; got:\n%v", tc.want, diags)
			}
		})
	}
}

// moreRejections continues the corpus. They are in a second table only because
// the first is the set a reader most wants to see; every entry carries the same
// obligation to assert a message and a line.
var moreRejections = []struct {
	name string
	body string
	line int
	want string
}{
	{
		name: "const declaration",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	const n = 4
	out[n] = 1
}`,
		line: 3, want: "only var declares anything",
	},
	{
		name: "var without an initializer",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	var n uint32
	out[n] = 1
}`,
		line: 3, want: "implicit zero value",
	},
	{
		name: "blank local",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	_ = t.GlobalID().X
	out[0] = 1
}`,
		line: 3, want: "discarded value",
	},
	{
		name: "multiple assignment",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	a, b := uint32(1), uint32(2)
	out[a] = float32(b)
}`,
		line: 3, want: "one value at a time",
	},
	{
		name: "if with an init statement",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	if i := t.GlobalID().X; i < 4 {
		out[i] = 1
	}
}`,
		line: 3, want: "declare the value on its own line",
	},
	{
		name: "closure",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	f := func() float32 { return 1 }
	out[0] = f()
}`,
		line: 3, want: "closures have no spelling",
	},
	{
		name: "composite literal",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	xs := []float32{1, 2}
	out[0] = xs[0]
}`,
		line: 3, want: "composite literals",
	},
	{
		name: "id component that is not X, Y, or Z",
		body: `type P struct{ W uint32 }

//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	p := t.GlobalID()
	out[p.X] = 1
	_ = P{}
}`,
		line: 7, want: "composite literals",
	},
	{
		name: "method value",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	f := t.GlobalIndex
	out[f()] = 1
}`,
		line: 3, want: "a method value has no lowering",
	},
	{
		name: "int variable",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	n := int(t.GlobalID().X)
	out[n] = 1
}`,
		line: 3, want: "platform-width",
	},
	{
		name: "len of a scalar",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		out[i] = float32(len(out[0:1]))
	}
}`,
		line: 5, want: "extent is fixed by its descriptor",
	},
	{
		name: "indexing something that is not a binding",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	id := t.GlobalID()
	out[id.X] = float32(id.X)
	_ = out[0:1]
}`,
		line: 5, want: "extent is fixed by its descriptor",
	},
	{
		name: "assignment to a parameter",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	out = out
}`,
		line: 3, want: "not something a kernel can assign to",
	},
	{
		name: "unnamed parameter",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, _ []float32) {}`,
		line: 2, want: "cannot be named _",
	},
	{
		name: "a method as a kernel",
		body: `type R struct{}

//accel:kernel workgroup=64
func (R) K(t accel.Thread, out []float32) { out[0] = 1 }`,
		line: 4, want: "a kernel is a package-level function",
	},
	{
		name: "workgroup with four extents",
		body: `//accel:kernel workgroup=1,1,1,1
func K(t accel.Thread, out []float32) { out[0] = 1 }`,
		line: 2, want: "needs a workgroup extent",
	},
	{
		name: "break outside a loop",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	break
	out[0] = 1
}`,
		line: 3, want: "break is outside a loop",
	},
	{
		name: "continue outside a loop",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	continue
}`,
		line: 3, want: "continue is outside a loop",
	},
	{
		name: "labelled break",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
outer:
	for {
		break outer
	}
}`,
		line: 3, want: "labels have no structured control-flow lowering",
	},
	{
		name: "goto",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	goto done
done:
	out[0] = 1
}`,
		line: 3, want: "a labelled goto has no structured control-flow lowering",
	},
	{
		name: "non-bool loop condition",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	for i := uint32(0); i; {
		out[i] = 1
	}
}`,
		line: 3, want: "loop condition is a bool",
	},
	{
		name: "expression as a loop post",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	for i := uint32(0); i < 4; len(out) {
		out[i] = 1
	}
}`,
		line: 3, want: "post takes an assignment or an increment",
	},
	{
		name: "direct recursion",
		body: `//accel:helper
func f(x float32) float32 { return f(x) }

//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) { out[0] = f(1) }`,
		line: 2, want: "f is recursive (f -> f)",
	},
	{
		name: "mutual recursion",
		body: `//accel:helper
func a(x float32) float32 { return b(x) }

//accel:helper
func b(x float32) float32 { return a(x) }

//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) { out[0] = a(1) }`,
		line: 2, want: "recursive (a -> b -> a)",
	},
	{
		name: "generic helper",
		body: `//accel:helper
func f[T any](x T) T { return x }

//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) { out[0] = 1 }`,
		line: 2, want: "generic helpers are out of scope for v0",
	},
	{
		name: "helper with two results",
		body: `//accel:helper
func f(x float32) (float32, float32) { return x, x }

//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) { out[0] = 1 }`,
		line: 2, want: "multiple helper results are out of scope for v0",
	},
	{
		name: "helper as a method",
		body: `type R struct{}

//accel:helper
func (R) f(x float32) float32 { return x }

//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) { out[0] = 1 }`,
		line: 4, want: "a helper is a package-level function",
	},
	{
		name: "call to another package",
		body: `import "math"

//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	out[0] = float32(math.Abs(1))
}`,
		line: 5, want: "there is no body to lower",
	},
	{
		name: "unread uniform",
		body: `type P struct{ K uint32 }

//accel:kernel workgroup=64
func K(t accel.Thread, p P, out []float32) { out[0] = 1 }`,
		line: 4, want: `uniform "p" is never read`,
	},
	{
		name: "uniform with a forbidden field",
		body: `type P struct {
	K uint32
	B bool
}

//accel:kernel workgroup=64
func K(t accel.Thread, p P, out []float32) { out[0] = float32(p.K) }`,
		line: 7, want: "one byte in Go and four on every device",
	},
	{
		name: "uniform with an int field",
		body: `type P struct{ N int }

//accel:kernel workgroup=64
func K(t accel.Thread, p P, out []float32) { out[0] = float32(p.N) }`,
		line: 4, want: "platform-width",
	},
	{
		name: "unnamed uniform struct",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, p struct{ K uint32 }, out []float32) { out[0] = float32(p.K) }`,
		line: 2, want: "generated for a named type",
	},
	{
		name: "boolean binding element",
		body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32, flags []bool) {
	if flags[0] {
		out[0] = 1
	}
}`,
		line: 2, want: "not one of float32, int32, uint32",
	},
}

// nestedRejections puts one out-of-subset expression in every position a value
// can appear in.
//
// It exists because the builders propagate a rejection by returning nil, and a
// missed nil check does not error: it produces a node with a nil operand that
// the emitter later dereferences, or worse, silently drops. Each case here is
// the same rejection reached through a different path, so a swallowed one shows
// up as an accepted kernel rather than as a crash three slices later.
func nestedRejections() []struct {
	name string
	body string
	line int
	want string
} {
	type c = struct {
		name string
		body string
		line int
		want string
	}
	// Each %s is a closure literal, which is permanently outside the subset.
	positions := []struct{ name, stmt string }{
		{"binary left", "out[0] = float32(%s()) * 2"},
		{"binary right", "out[0] = 2 * float32(%s())"},
		{"unary operand", "out[0] = -float32(%s())"},
		{"index", "out[uint32(%s())] = 1"},
		{"conversion operand", "out[0] = float32(%s())"},
		{"assignment target index", "out[uint32(%s())] += 1"},
		{"assignment value", "out[0] += float32(%s())"},
		{"if condition", "if %s() > 0 {\n\t\tout[0] = 1\n\t}"},
		{"len operand", "out[0] = float32(len(out)) + float32(%s())"},
		{"declaration initializer", "n := float32(%s())\n\tout[0] = n"},
		{"increment target", "out[uint32(%s())]++"},
		{"assignment to a discarded name", "_ = float32(%s())"},
	}

	out := make([]c, 0, len(positions))
	for _, p := range positions {
		lit := "func() float32 { return 1 }"
		body := "//accel:kernel workgroup=64\nfunc K(t accel.Thread, out []float32) {\n\t" +
			fmt.Sprintf(p.stmt, lit) + "\n}"
		// A closure that is called is a call whose target has no lowering, and one
		// that is not is a closure. Both are rejected and both propagate, which
		// is what these cases are about.
		want := "outside the subset"
		out = append(out, c{name: "nested in " + p.name, body: body, line: 3, want: want})
	}

	return out
}

// headerLines is how many lines [header] occupies, so a case states the line
// within its own body and does not have to be renumbered when the preamble
// changes.
const headerLines = 8

// header is the preamble every rejection case shares.
func header(pkgName string) string {
	return "// Copyright 2026 The golang.design Initiative Authors.\n" +
		"// All rights reserved. Use of this source code is governed by\n" +
		"// a BSD-style license that can be found in the LICENSE file.\n\n" +
		"package " + pkgName + "\n\n" +
		"import \"golang.design/x/accel\"\n\n"
}

// TestAccepts is the positive half of the corpus.
//
// A rejection corpus alone proves only that the checker says no. These say what
// it says yes to, and each asserts the inferred accesses, because access
// inference is the one thing here a caller cannot correct: it is derived from
// the body and never declared.
func TestAccepts(t *testing.T) {
	cases := []struct {
		name  string
		body  string
		read  []string
		write []string
	}{
		{
			name: "var declaration and else",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, in []float32, out []float32) {
	var i uint32 = t.GlobalID().X
	if i < uint32(len(out)) {
		out[i] = in[i]
	} else {
		out[0] = 0
	}
}`,
			read: []string{"in"}, write: []string{"out"},
		},
		{
			name: "compound assignment and increment",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	i := t.GlobalID().X
	n := uint32(0)
	n += 3
	n *= 2
	n--
	n++
	if i < uint32(len(out)) {
		out[i] = float32(n)
	}
}`,
			write: []string{"out"},
		},
		{
			name: "every compound assignment operator",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []int32) {
	n := int32(t.GlobalID().X)
	n += 3
	n -= 1
	n *= 2
	n /= 2
	n %= 7
	n &= 15
	n |= 1
	n ^= 2
	n <<= 1
	n >>= 1
	if uint32(n) < uint32(len(out)) {
		out[n] = n
	}
}`,
			write: []string{"out"},
		},
		{
			name: "three-clause loop with break and continue",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, in []float32, out []float32) {
	total := float32(0)
	for i := uint32(0); i < uint32(len(in)); i++ {
		if in[i] < 0 {
			continue
		}
		if in[i] > 100 {
			break
		}
		total += in[i]
	}
	out[t.GlobalID().X] = total
}`,
			read: []string{"in"}, write: []string{"out"},
		},
		{
			name: "condition-only loop",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	i := uint32(0)
	for i < uint32(len(out)) {
		out[i] = 1
		i++
	}
}`,
			write: []string{"out"},
		},
		{
			name: "infinite loop with a break",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	i := uint32(0)
	for {
		if i >= uint32(len(out)) {
			break
		}
		out[i] = 1
		i++
	}
}`,
			write: []string{"out"},
		},
		{
			name: "nested loops",
			body: `//accel:kernel workgroup=8,8
func K(t accel.Thread, out []float32) {
	for y := uint32(0); y < 4; y++ {
		for x := uint32(0); x < 4; x++ {
			i := y*4 + x
			if i < uint32(len(out)) {
				out[i] = float32(i)
			}
		}
	}
}`,
			write: []string{"out"},
		},
		{
			name: "loop with an assignment init",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	i := uint32(0)
	for i = 0; i < uint32(len(out)); i++ {
		out[i] = 2
	}
}`,
			write: []string{"out"},
		},
		{
			name: "helper called by a kernel",
			body: `//accel:helper
func square(x float32) float32 { return x * x }

//accel:kernel workgroup=64
func K(t accel.Thread, in []float32, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		out[i] = square(in[i])
	}
}`,
			read: []string{"in"}, write: []string{"out"},
		},
		{
			name: "helper declared after its caller",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, in []float32, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		out[i] = twice(in[i])
	}
}

//accel:helper
func twice(x float32) float32 { return x + x }`,
			read: []string{"in"}, write: []string{"out"},
		},
		{
			name: "helper calling a helper",
			body: `//accel:helper
func a(x float32) float32 { return b(x) + 1 }

//accel:helper
func b(x float32) float32 { return x * 2 }

//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	out[t.GlobalID().X] = a(1)
}`,
			write: []string{"out"},
		},
		{
			name: "helper taking a binding infers the caller's access",
			body: `//accel:helper
func store(buf []float32, i uint32, v float32) {
	buf[i] = v
}

//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		store(out, i, 1)
	}
}`,
			write: []string{"out"},
		},
		{
			name: "helper taking the thread",
			body: `//accel:helper
func index(t accel.Thread) uint32 { return t.GlobalID().X }

//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	i := index(t)
	if i < uint32(len(out)) {
		out[i] = 1
	}
}`,
			write: []string{"out"},
		},
		{
			name: "helper with no result",
			body: `//accel:helper
func clear(buf []float32, i uint32) {
	buf[i] = 0
}

//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	i := t.GlobalID().X
	if i < uint32(len(out)) {
		clear(out, i)
	}
}`,
			write: []string{"out"},
		},
		{
			name: "uniform read by a kernel",
			body: `type Params struct {
	Scale  float32
	Origin [3]float32
	Steps  uint32
}

//accel:kernel workgroup=64
func K(t accel.Thread, p Params, in []float32, out []float32) {
	i := t.GlobalID().X
	if i < p.Steps && i < uint32(len(out)) {
		out[i] = in[i]*p.Scale + p.Origin[0]
	}
}`,
			read: []string{"in"}, write: []string{"out"},
		},
		{
			name: "read-write binding",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, buf []float32) {
	i := t.GlobalID().X
	if i < uint32(len(buf)) {
		buf[i] = buf[i] * 2
	}
}`,
			read: []string{"buf"}, write: []string{"buf"},
		},
		{
			name: "every scalar dtype",
			body: `//accel:kernel workgroup=8,4
func K(t accel.Thread, a []int32, b []uint32, c []int8, d []uint8, e []float32) {
	i := t.GlobalIndex()
	if i < uint32(len(a)) {
		a[i] = int32(b[i]) + int32(c[i]) + int32(d[i]) + int32(e[i])
	}
}`,
			read: []string{"b", "c", "d", "e"}, write: []string{"a"},
		},
		{
			name: "unary and every id form",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []int32) {
	i := t.LocalIndex() + t.GroupIndex() + t.GlobalID().Y + t.LocalID().Z + t.GroupID().X
	if i < uint32(len(out)) {
		out[i] = -int32(i)
	}
}`,
			write: []string{"out"},
		},
		{
			name: "three-axis workgroup",
			body: `//accel:kernel workgroup=4,4,4
func K(t accel.Thread, out []uint32) {
	if t.GlobalIndex() < uint32(len(out)) {
		out[t.GlobalIndex()] = t.LocalIndex()
	}
}`,
			write: []string{"out"},
		},
		{
			name: "bare return and empty statement",
			body: `//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) {
	i := t.GlobalID().X
	if i >= uint32(len(out)) {
		return
	}
	;
	out[i] = 1
}`,
			write: []string{"out"},
		},
	}

	files := make(map[string]string, len(cases))
	patterns := make([]string, 0, len(cases))
	for i, tc := range cases {
		dir := "acceptcase" + string(rune('a'+i))
		files[dir] = header(dir) + tc.body + "\n"
		patterns = append(patterns, "./internal/kernelc/front/"+dir)
	}
	pkgs := loadOverlay(t, files, patterns)

	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			kernels, diags := front.Check(pkgs[patterns[i]])
			if len(diags) > 0 {
				t.Fatalf("rejected:\n%v", diags)
			}
			if len(kernels) != 1 {
				t.Fatalf("found %d kernels, want 1", len(kernels))
			}
			k := kernels[0]

			got := map[string]*struct{ read, write bool }{}
			for _, b := range k.Bindings {
				got[b.Name] = &struct{ read, write bool }{b.Read, b.Write}
			}
			for _, name := range tc.read {
				if b := got[name]; b == nil || !b.read {
					t.Errorf("%q is not marked read", name)
				}
			}
			for _, name := range tc.write {
				if b := got[name]; b == nil || !b.write {
					t.Errorf("%q is not marked written", name)
				}
			}
			for name, b := range got {
				if b.write && !contains(tc.write, name) {
					t.Errorf("%q is marked written and should not be: a caller would have to "+
						"declare a usage the kernel does not need", name)
				}
				if b.read && !contains(tc.read, name) {
					t.Errorf("%q is marked read and should not be", name)
				}
			}
		})
	}
}

func contains(xs []string, s string) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

// TestLoadRejectsWhatDoesNotTypeCheck checks that a package which does not
// compile is reported as such. A kernel that silently vanished is
// indistinguishable from one nobody wrote, which is the worse failure.
func TestLoadRejectsWhatDoesNotTypeCheck(t *testing.T) {
	if _, err := front.Load(repoRoot(t), "./internal/kernelc/front/nosuchpackage"); err == nil {
		t.Error("a pattern matching nothing was accepted")
	}
}

// TestNonKernelsAreIgnored checks that ordinary Go in a kernel package is not
// walked. A package holding both kernels and host helpers is the normal case.
func TestNonKernelsAreIgnored(t *testing.T) {
	files := map[string]string{
		"mixedcase": header("mixedcase") + `func Host(xs []float64) float64 {
	total := 0.0
	for _, x := range xs {
		total += x
	}
	return total
}

//accel:kernel workgroup=64
func K(t accel.Thread, out []float32) { out[t.GlobalID().X] = 1 }
`,
	}
	pkgs := loadOverlay(t, files, []string{"./internal/kernelc/front/mixedcase"})
	kernels, diags := front.Check(pkgs["./internal/kernelc/front/mixedcase"])
	if len(diags) > 0 {
		t.Fatalf("ordinary Go beside a kernel was walked:\n%v", diags)
	}
	if len(kernels) != 1 {
		t.Fatalf("found %d kernels, want 1", len(kernels))
	}
}

// repoRoot is the module root, which overlay paths are relative to.
func repoRoot(t *testing.T) string {
	t.Helper()
	abs, err := filepath.Abs("../../..")
	if err != nil {
		t.Fatal(err)
	}
	return abs
}

// TestDiagnosticsAreDeterministic is the property CI found the hard way.
//
// The recursion check ranged over a map, and Go randomizes map iteration, so the
// same source reported "a -> b -> a" on one machine and "b -> a -> b" on
// another. A compiler whose diagnostics depend on the run cannot have a golden
// test, and a bug report against it cannot be reproduced from the message.
//
// The property is stronger than the instance: every diagnostic a package
// produces, in order, must be the same on every run.
func TestDiagnosticsAreDeterministic(t *testing.T) {
	files := map[string]string{
		"determcase": header("determcase") + `//accel:helper
func a(x float32) float32 { return b(x) }

//accel:helper
func b(x float32) float32 { return a(x) }

//accel:helper
func c(x float32) float32 { return c(x) }

//accel:kernel workgroup=64
func K(t accel.Thread, unusedA []float32, unusedB []float32, out []float32) {
	switch {
	}
	for range out {
	}
	out[0] = a(1) + c(1)
}
`,
	}
	pattern := "./internal/kernelc/front/determcase"
	pkgs := loadOverlay(t, files, []string{pattern})

	var first string
	for run := range 20 {
		_, diags := front.Check(pkgs[pattern])
		if len(diags) == 0 {
			t.Fatal("the case produced no diagnostics")
		}
		var b strings.Builder
		for _, d := range diags {
			fmt.Fprintf(&b, "%d:%d: %s\n", d.Pos.Line, d.Pos.Column, d.Msg)
		}
		if run == 0 {
			first = b.String()
			continue
		}
		if b.String() != first {
			t.Fatalf("run %d produced different diagnostics:\n--- first ---\n%s\n--- run %d ---\n%s",
				run, first, run, b.String())
		}
	}

	// And the cycles are reported from their first-declared member, which is
	// what makes the message stable rather than merely repeatable.
	if !strings.Contains(first, "a is recursive (a -> b -> a)") {
		t.Errorf("the mutual cycle is not reported from its first member:\n%s", first)
	}
	if !strings.Contains(first, "c is recursive (c -> c)") {
		t.Errorf("the direct cycle is not reported:\n%s", first)
	}
}
