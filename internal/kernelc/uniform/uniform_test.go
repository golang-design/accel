// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package uniform_test

import (
	"fmt"
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/front"
	"golang.design/x/accel/internal/kernelc/ir"
	"golang.design/x/accel/internal/kernelc/uniform"
)

// analyze compiles one kernel body and returns the level of each local by name.
//
// By name, because the assertion a reader wants to make is about the variable
// they wrote, not about a node identity they cannot see.
func analyze(t *testing.T, body string) map[string]uniform.Level {
	t.Helper()
	return analyzeWith(t, body, false)
}

func analyzeWith(t *testing.T, body string, kmath bool) map[string]uniform.Level {
	t.Helper()
	imports := "import \"golang.design/x/accel\""
	if kmath {
		imports = "import (\n\t\"golang.design/x/accel\"\n\t\"golang.design/x/accel/kmath\"\n)"
	}
	_ = imports
	// Every binding must be read or written, so the preamble touches `in` once.
	// It is a load at a non-uniform index, which is the ordinary shape and
	// keeps the cases below about the thing each is testing.
	src := "package k\n\n" + imports + "\n\n" +
		"//accel:kernel workgroup=64\n" +
		"func K(t accel.Thread, in []float32, out []float32) {\n" +
		"\tseed := in[t.GlobalID().X]\n\tout[t.GlobalID().X] = seed\n" +
		body + "\n}\n"

	return levelsOf(t, src)
}

// levelsOf analyses a whole source file and reports each local's level by name.
func levelsOf(t *testing.T, src string) map[string]uniform.Level {
	t.Helper()
	pkg := checkSource(t, src)
	if pkg == nil {
		t.Fatalf("the source did not type-check:\n%s", src)
	}
	fns, diags := front.Check(pkg)
	if len(diags) > 0 {
		t.Fatalf("the front end rejected it: %v\n%s", diags, src)
	}
	if len(fns) != 1 {
		t.Fatalf("got %d kernels, want 1", len(fns))
	}

	in := uniform.Of(fns[0])
	out := map[string]uniform.Level{}
	var walk func(s ir.Stmt)
	walkBlock := func(b *ir.Block) {
		if b == nil {
			return
		}
		for _, s := range b.List {
			walk(s)
		}
	}
	walk = func(s ir.Stmt) {
		switch n := s.(type) {
		case *ir.Block:
			walkBlock(n)
		case *ir.Declare:
			out[n.Local.Name] = in.Level(n.Local)
		case *ir.If:
			walkBlock(n.Then)
			if n.Else != nil {
				walk(n.Else)
			}
		case *ir.For:
			if n.Init != nil {
				walk(n.Init)
			}
			walkBlock(n.Body)
			if n.Post != nil {
				walk(n.Post)
			}
		}
	}
	walkBlock(fns[0].Body)
	return out
}

// Spec 002 section 3.3's seed table, row by row. It is the input to everything
// else here, so a wrong seed is a wrong answer no amount of propagation fixes.
func TestSeeds(t *testing.T) {
	cases := []struct {
		name string
		expr string
		want uniform.Level
	}{
		{"a literal", "uint32(7)", uniform.Workgroup},
		{"the group id", "t.GroupID().X", uniform.Workgroup},
		{"the group index", "t.GroupIndex()", uniform.Workgroup},
		{"the local id", "t.LocalID().X", uniform.Non},
		{"the local index", "t.LocalIndex()", uniform.Non},
		{"the global id", "t.GlobalID().X", uniform.Non},
		{"the global index", "t.GlobalIndex()", uniform.Non},
		{"a binding's length", "uint32(len(in))", uniform.Workgroup},
		{"the workgroup size", "t.WorkgroupSize().X", uniform.Workgroup},
		{"the group count", "t.NumGroups().X", uniform.Workgroup},
		{"the global size", "t.GlobalSize().X", uniform.Workgroup},
		{"the subgroup size", "t.SubgroupSize()", uniform.Workgroup},
		{"the subgroup index", "t.SubgroupIndex()", uniform.Subgroup},
		{"the subgroup lane", "t.SubgroupLane()", uniform.Non},
		// A subgroup reduction returns the same value to every active lane,
		// and is per-invocation anyway: the active set is not portable.
		{"a subgroup reduction", "t.SubgroupAddF32(1)", uniform.Non},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := analyze(t, "v := "+c.expr+"\nout[0] = float32(v)")
			if got["v"] != c.want {
				t.Errorf("%s is %v, want %v", c.expr, got["v"], c.want)
			}
		})
	}
}

// A load is non-uniform even when its index is uniform, because another
// invocation may have written the location. Conservative, and known to be:
// this is one of the two families spec 002 section 3.3 says it rejects.
func TestLoadsAreNonUniformEvenAtAUniformIndex(t *testing.T) {
	got := analyze(t, "v := in[t.GroupIndex()]\nout[0] = v")
	if got["v"] != uniform.Non {
		t.Errorf("a load at a uniform index is %v, want non-uniform: another "+
			"invocation may have written it", got["v"])
	}
}

// Propagation: the least-uniform operand decides.
func TestPropagationTakesTheLeastUniformOperand(t *testing.T) {
	cases := []struct {
		name string
		body string
		want map[string]uniform.Level
	}{{
		name: "uniform plus uniform",
		body: "a := t.GroupIndex()\nb := a + 1\nout[0] = float32(b)",
		want: map[string]uniform.Level{"a": uniform.Workgroup, "b": uniform.Workgroup},
	}, {
		name: "uniform plus non-uniform",
		body: "a := t.GroupIndex()\nb := a + t.LocalIndex()\nout[0] = float32(b)",
		want: map[string]uniform.Level{"a": uniform.Workgroup, "b": uniform.Non},
	}, {
		name: "a conversion carries its operand's level",
		body: "a := float32(t.LocalIndex())\nout[0] = a",
		want: map[string]uniform.Level{"a": uniform.Non},
	}, {
		name: "a unary operator carries its operand's level",
		body: "a := -float32(t.GroupIndex())\nout[0] = a",
		want: map[string]uniform.Level{"a": uniform.Workgroup},
	}}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := analyze(t, c.body)
			for name, want := range c.want {
				if got[name] != want {
					t.Errorf("%s is %v, want %v", name, got[name], want)
				}
			}
		})
	}
}

// The clause that matters most: a definition inherits the control flow it sits
// under. Without it, both branches assigning literals makes the variable look
// workgroup-uniform, and that is precisely the case spec 002 section 3.3 calls
// out.
func TestADefinitionInheritsItsControlFlow(t *testing.T) {
	got := analyze(t, `
	var x uint32 = 0
	if t.LocalIndex() < 4 {
		y := uint32(1)
		out[0] = float32(y)
	} else {
		z := uint32(2)
		out[1] = float32(z)
	}
	out[2] = float32(x)`)

	if got["y"] != uniform.Non || got["z"] != uniform.Non {
		t.Errorf("y is %v and z is %v: both are literals, but they are assigned "+
			"under a non-uniform predicate, so neither is uniform", got["y"], got["z"])
	}
	// The variable outside the conditional keeps its own level, so the rule is
	// about control dependence and not about poisoning a whole function.
	if got["x"] != uniform.Workgroup {
		t.Errorf("x is %v, want workgroup-uniform: it is declared outside the "+
			"conditional", got["x"])
	}
}

// A uniform predicate does not make its body non-uniform, or the analysis would
// reject every kernel with an `if`.
func TestAUniformPredicateKeepsItsBodyUniform(t *testing.T) {
	got := analyze(t, `
	if t.GroupIndex() < 4 {
		y := uint32(1)
		out[0] = float32(y)
	}`)
	if got["y"] != uniform.Workgroup {
		t.Errorf("y is %v, want workgroup-uniform: its predicate is uniform", got["y"])
	}
}

// A loop's body is control-dependent on its condition, which is what the
// barrier rule's "every enclosing loop has a uniform trip count" reduces to.
func TestALoopBodyInheritsItsCondition(t *testing.T) {
	nonUniform := analyze(t, `
	for i := uint32(0); i < t.LocalIndex(); i++ {
		y := uint32(1)
		out[0] = float32(y)
	}`)
	if nonUniform["y"] != uniform.Non {
		t.Errorf("y is %v, want non-uniform: its loop's trip count is non-uniform",
			nonUniform["y"])
	}

	bounded := analyze(t, `
	for i := uint32(0); i < 8; i++ {
		y := uint32(1)
		out[0] = float32(y)
	}`)
	if bounded["y"] != uniform.Workgroup {
		t.Errorf("y is %v, want workgroup-uniform: the trip count is a literal",
			bounded["y"])
	}
}

// A value the walk never reached is non-uniform, not uniform. Defaulting the
// other way would make an unhandled node silently admit a barrier.
func TestTheDefaultIsNonUniform(t *testing.T) {
	in := uniform.Of(&ir.Func{Body: ir.NewBlock(0)})
	var absent ir.Value = ir.NewConst(0, &ir.Type{Kind: ir.U32}, nil)
	if got := in.Level(absent); got != uniform.Non {
		t.Errorf("an unreached value is %v, want non-uniform", got)
	}
	if got := in.Control(ir.NewBreak(0)); got != uniform.Non {
		t.Errorf("an unreached statement's control is %v, want non-uniform", got)
	}
}

func TestLevelsName(t *testing.T) {
	for _, c := range []struct {
		l    uniform.Level
		want string
	}{
		{uniform.Workgroup, "workgroup-uniform"},
		{uniform.Subgroup, "subgroup-uniform"},
		{uniform.Non, "non-uniform"},
		{uniform.Level(9), "non-uniform"},
	} {
		if got := c.l.String(); got != c.want {
			t.Errorf("got %q, want %q", got, c.want)
		}
	}
}

// The lattice order is what makes max() the join, so it is asserted rather than
// assumed: a reordering of the constants would silently invert the analysis.
func TestTheLatticeIsOrdered(t *testing.T) {
	if !(uniform.Workgroup < uniform.Subgroup && uniform.Subgroup < uniform.Non) {
		t.Fatal("the levels must be ordered workgroup < subgroup < non, because " +
			"the analysis joins with max")
	}
}

func TestAnalysisIsDeterministic(t *testing.T) {
	body := `
	a := t.GroupIndex()
	b := a + t.LocalIndex()
	for i := uint32(0); i < a; i++ {
		c := b + i
		out[0] = float32(c)
	}`
	first := fmt.Sprint(sorted(analyze(t, body)))
	for range 20 {
		if got := fmt.Sprint(sorted(analyze(t, body))); got != first {
			t.Fatalf("the analysis is not deterministic:\n%s\nwant\n%s", got, first)
		}
	}
}

func sorted(m map[string]uniform.Level) []string {
	out := make([]string, 0, len(m))
	for k, v := range m {
		out = append(out, fmt.Sprintf("%s=%v", k, v))
	}
	slicesSort(out)
	return out
}

func slicesSort(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

func TestLevelStringsAreDistinct(t *testing.T) {
	seen := map[string]bool{}
	for _, l := range []uniform.Level{uniform.Workgroup, uniform.Subgroup, uniform.Non} {
		if seen[l.String()] {
			t.Fatalf("two levels share the name %q", l.String())
		}
		seen[l.String()] = true
	}
	if strings.Contains(uniform.Workgroup.String(), "non") {
		t.Error("the workgroup level's name should not contain \"non\"")
	}
}

// The rest of the IR's shapes, so no node reaches the analysis unhandled and
// falls to the Non default without anyone noticing. A node defaulting to Non is
// correct but silent, and silence here means a rejected valid kernel with a
// diagnostic nobody can act on.
func TestEveryShapeIsReached(t *testing.T) {
	cases := []struct {
		name string
		body string
		want map[string]uniform.Level
	}{{
		name: "a field of a uniform struct",
		body: "", // covered by the uniform-parameter case below
	}, {
		name: "a helper call takes its least uniform argument",
		body: "",
	}, {
		name: "an assignment to a local",
		body: `
	v := t.GroupIndex()
	v = t.LocalIndex()
	out[0] = float32(v)`,
		want: map[string]uniform.Level{"v": uniform.Non},
	}, {
		name: "a store into a binding levels its index and value",
		body: `
	v := t.GroupIndex()
	out[v] = float32(v)`,
		want: map[string]uniform.Level{"v": uniform.Workgroup},
	}, {
		name: "a nested block",
		body: `
	if t.GroupIndex() < 2 {
		if t.GroupIndex() < 1 {
			y := uint32(3)
			out[0] = float32(y)
		}
	}`,
		want: map[string]uniform.Level{"y": uniform.Workgroup},
	}, {
		name: "an else branch",
		body: `
	if t.LocalIndex() < 2 {
		out[0] = 1
	} else {
		z := uint32(4)
		out[1] = float32(z)
	}`,
		want: map[string]uniform.Level{"z": uniform.Non},
	}, {
		name: "a break under a loop",
		body: `
	for i := uint32(0); i < 4; i++ {
		if i > 2 {
			break
		}
		y := i
		out[0] = float32(y)
	}`,
		want: map[string]uniform.Level{"y": uniform.Workgroup},
	}, {
		name: "an intrinsic on a uniform argument stays uniform",
		body: `
	v := kmathSqrt(float32(t.GroupIndex()))
	out[0] = v`,
		want: map[string]uniform.Level{"v": uniform.Workgroup},
	}, {
		name: "an intrinsic on a non-uniform argument does not",
		body: `
	v := kmathSqrt(float32(t.LocalIndex()))
	out[0] = v`,
		want: map[string]uniform.Level{"v": uniform.Non},
	}}

	for _, c := range cases {
		if c.body == "" {
			continue
		}
		t.Run(c.name, func(t *testing.T) {
			body := strings.ReplaceAll(c.body, "kmathSqrt", "kmath.Sqrt")
			got := analyzeWith(t, body, strings.Contains(body, "kmath."))
			for name, want := range c.want {
				if got[name] != want {
					t.Errorf("%s is %v, want %v", name, got[name], want)
				}
			}
		})
	}
}

// A uniform struct parameter's fields are workgroup-uniform: the struct is one
// value for the whole dispatch, which is what "uniform" means in Go here.
func TestUniformStructFieldsAreUniform(t *testing.T) {
	src := `package k

import "golang.design/x/accel"

type Params struct{ Scale float32 }

//accel:kernel workgroup=64
func K(t accel.Thread, p Params, in []float32, out []float32) {
	v := p.Scale
	out[t.GlobalID().X] = in[t.GlobalID().X] * v
}
`
	got := levelsOf(t, src)
	if got["v"] != uniform.Workgroup {
		t.Errorf("a uniform struct's field is %v, want workgroup-uniform", got["v"])
	}
}

// An atomic's previous value is spec 002 section 3.3's own row, and it was
// not what the analysis said before the seed came from the intrinsic table:
// the analysis joined an atomic's arguments, which are a binding, an index and
// a literal, and called the result workgroup-uniform.
func TestAnAtomicsReturnValueIsNonUniform(t *testing.T) {
	got := levelsOf(t, `package k

import "golang.design/x/accel"

//accel:kernel workgroup=64
func K(t accel.Thread, counter []uint32, out []uint32) {
	prev := accel.AddU32(counter, 0, 1)
	out[t.GlobalID().X] = prev
}
`)
	if got["prev"] != uniform.Non {
		t.Errorf("an atomic's return value is %v, want non-uniform: every invocation "+
			"receives a different previous value", got["prev"])
	}
}

// A helper's result is at least as non-uniform as its least uniform argument.
// What its body reads on its own is joined in as well; join_test.go has that
// half.
func TestAHelperTakesItsLeastUniformArgument(t *testing.T) {
	src := `package k

import "golang.design/x/accel"

//accel:helper
func twice(x uint32) uint32 { return x * 2 }

//accel:kernel workgroup=64
func K(t accel.Thread, in []float32, out []float32) {
	a := twice(t.GroupIndex())
	b := twice(t.LocalIndex())
	out[a%uint32(len(out))] = in[b%uint32(len(in))]
}
`
	got := levelsOf(t, src)
	if got["a"] != uniform.Workgroup {
		t.Errorf("a helper of a uniform argument is %v, want workgroup-uniform", got["a"])
	}
	if got["b"] != uniform.Non {
		t.Errorf("a helper of a non-uniform argument is %v, want non-uniform", got["b"])
	}
}
