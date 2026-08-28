// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package intrin_test

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"sort"
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/intrin"
	"golang.design/x/accel/internal/kernelc/ir"
)

// check type-checks one source file with no imports and returns its package.
func check(t *testing.T, src string) *types.Package {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "p.go", src, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	conf := types.Config{Importer: nil}
	pkg, err := conf.Check("example.com/p", fset, []*ast.File{f}, nil)
	if err != nil {
		t.Fatalf("check: %v", err)
	}
	return pkg
}

// method finds a method on a named type in a checked package.
func method(t *testing.T, pkg *types.Package, typeName, methodName string) *types.Func {
	t.Helper()
	obj := pkg.Scope().Lookup(typeName)
	if obj == nil {
		t.Fatalf("no type %q", typeName)
	}
	named, ok := obj.Type().(*types.Named)
	if !ok {
		t.Fatalf("%q is not a named type", typeName)
	}
	for m := range named.Methods() {
		if m.Name() == methodName {
			return m
		}
	}
	t.Fatalf("no method %q on %q", methodName, typeName)
	return nil
}

// TestLookupResolvesTheRealIntrinsics uses the actual kernel package's objects,
// so the table is checked against the types it will really see rather than
// against a restatement of itself.
func TestLookupResolvesTheRealIntrinsics(t *testing.T) {
	pkg := loadKernelPackage(t)

	for _, tc := range []struct {
		method   string
		op       ir.Opcode
		result   ir.Kind
		uniform  intrin.Uniformity
		stage    intrin.Stage
		authored string
	}{
		{"GlobalID", ir.OpGlobalID, ir.ID3Kind, intrin.PerInvocation, intrin.Flat, "accel.Thread.GlobalID"},
		{"LocalID", ir.OpLocalID, ir.ID3Kind, intrin.PerInvocation, intrin.Flat, "accel.Thread.LocalID"},
		{"GroupID", ir.OpGroupID, ir.ID3Kind, intrin.PerWorkgroup, intrin.Flat, "accel.Thread.GroupID"},
		{"GlobalIndex", ir.OpGlobalIndex, ir.U32, intrin.PerInvocation, intrin.Flat, "accel.Thread.GlobalIndex"},
		{"LocalIndex", ir.OpLocalIndex, ir.U32, intrin.PerInvocation, intrin.Flat, "accel.Thread.LocalIndex"},
		{"GroupIndex", ir.OpGroupIndex, ir.U32, intrin.PerWorkgroup, intrin.Flat, "accel.Thread.GroupIndex"},
		{"Barrier", ir.OpBarrier, ir.Invalid, intrin.PerWorkgroup, intrin.Cooperative, "accel.Thread.Barrier"},
	} {
		t.Run(tc.method, func(t *testing.T) {
			fn := method(t, pkg, "Thread", tc.method)
			got, ok := intrin.Lookup(fn)
			if !ok {
				t.Fatalf("Thread.%s did not resolve to an intrinsic", tc.method)
			}
			if got.Op != tc.op {
				t.Errorf("Op = %v, want %v", got.Op, tc.op)
			}
			if got.Result != tc.result {
				t.Errorf("Result = %v, want %v", got.Result, tc.result)
			}
			if got.Uniformity != tc.uniform {
				t.Errorf("Uniformity = %v, want %v", got.Uniformity, tc.uniform)
			}
			if got.Stage != tc.stage {
				t.Errorf("Stage = %v, want %v", got.Stage, tc.stage)
			}
			// The authored spelling is what the digest records, and it is
			// deliberately not the resolved path.
			if got.Authored != tc.authored {
				t.Errorf("Authored = %q, want %q", got.Authored, tc.authored)
			}
			if strings.Contains(got.Authored, "internal/kernel") {
				t.Errorf("Authored is %q, which is the resolved path rather than the "+
					"authored spelling: relocating the type would invalidate every digest",
					got.Authored)
			}
		})
	}
}

// TestUniformityIsRecorded is the distinction the M4 barrier analysis is built
// on, so getting it wrong here is a bug that surfaces two milestones later.
func TestUniformityIsRecorded(t *testing.T) {
	pkg := loadKernelPackage(t)
	group, _ := intrin.Lookup(method(t, pkg, "Thread", "GroupID"))
	local, _ := intrin.Lookup(method(t, pkg, "Thread", "LocalID"))
	if group.Uniformity != intrin.PerWorkgroup {
		t.Error("GroupID is uniform across a workgroup")
	}
	if local.Uniformity != intrin.PerInvocation {
		t.Error("LocalID varies between invocations of one workgroup")
	}
}

// TestSameNamedMethodOnAnotherTypeIsNotAnIntrinsic is the predecessor's bug,
// checked directly. A user type with a GlobalID method must not lower to the
// builtin, and the failure it prevents is a kernel that compiles, runs, and
// computes something else.
func TestSameNamedMethodOnAnotherTypeIsNotAnIntrinsic(t *testing.T) {
	pkg := check(t, `package p

type Thread struct{}

func (Thread) GlobalID() uint32 { return 0 }
func (Thread) Barrier()         {}

type Other struct{}

func (Other) GlobalID() uint32 { return 1 }
`)
	for _, tn := range []string{"Thread", "Other"} {
		for _, mn := range []string{"GlobalID", "Barrier"} {
			obj := pkg.Scope().Lookup(tn)
			named := obj.Type().(*types.Named)
			for m := range named.Methods() {
				if m.Name() != mn {
					continue
				}
				if _, ok := intrin.Lookup(m); ok {
					t.Errorf("%s.%s in package %q resolved to an intrinsic; the table is "+
						"keyed on identity, not on name", tn, mn, pkg.Path())
				}
			}
		}
	}
}

// TestSameNamedFunctionIsNotAnIntrinsic covers the free-function half.
func TestSameNamedFunctionIsNotAnIntrinsic(t *testing.T) {
	pkg := check(t, `package p

func GlobalID() uint32 { return 0 }
func Barrier()         {}
`)
	for _, name := range []string{"GlobalID", "Barrier"} {
		fn := pkg.Scope().Lookup(name).(*types.Func)
		if _, ok := intrin.Lookup(fn); ok {
			t.Errorf("a package-level %s resolved to an intrinsic", name)
		}
	}
}

// TestLookupRejectsWhatItCannotIdentify covers the shapes that must not resolve
// rather than panic.
func TestLookupRejectsWhatItCannotIdentify(t *testing.T) {
	if _, ok := intrin.Lookup(nil); ok {
		t.Error("a nil func resolved")
	}

	// A method on an unnamed receiver type has no receiver type name to key on.
	pkg := check(t, `package p

type T struct{}

func (T) GlobalID() uint32 { return 0 }
`)
	fn := method(t, pkg, "T", "GlobalID")
	if _, ok := intrin.Lookup(fn); ok {
		t.Error("a method on an unrelated type resolved")
	}
}

// TestPointerReceiversResolve checks that the receiver's name is taken through
// a pointer, since a cooperative intrinsic on *Thread would otherwise silently
// stop resolving.
func TestPointerReceiversResolve(t *testing.T) {
	pkg := check(t, `package p

type Thread struct{}

func (*Thread) GlobalID() uint32 { return 0 }
`)
	fn := method(t, pkg, "Thread", "GlobalID")
	// Not this package's Thread, so it must not resolve, but it must also not
	// have failed for want of a receiver name.
	if _, ok := intrin.Lookup(fn); ok {
		t.Error("another package's *Thread resolved")
	}
}

// TestNamesAndDigestAreStable guards the two summaries the rest of the compiler
// depends on.
func TestNamesAndDigestAreStable(t *testing.T) {
	names := intrin.Names()
	want := []string{
		"accel.AddF32",
		"accel.AddI32",
		"accel.AddU32",
		"accel.AndU32",
		"accel.BFloat16.F32",
		"accel.CompareExchangeI32",
		"accel.CompareExchangeU32",
		"accel.ExchangeI32",
		"accel.ExchangeU32",
		"accel.Fetch",
		"accel.KernelMask.Any",
		"accel.KernelMask.Bit",
		"accel.KernelMask.Count",
		"accel.KernelMask.CountLower",
		"accel.KernelMask.LowestSet",
		"accel.Float16.F32",
		"accel.Fragment.Coord",
		"accel.Fragment.FrontFacing",
		"accel.MaxI32",
		"accel.MaxU32",
		"accel.MinI32",
		"accel.MinU32",
		"accel.OrU32",
		"accel.SubI32",
		"accel.SubU32",
		"accel.Thread.Barrier",
		"accel.Thread.BarrierShared",
		"accel.Thread.BarrierStorage",
		"accel.Thread.GlobalID",
		"accel.Thread.GlobalIndex",
		"accel.Thread.GlobalSize",
		"accel.Thread.GroupID",
		"accel.Thread.GroupIndex",
		"accel.Thread.LocalID",
		"accel.Thread.LocalIndex",
		"accel.Thread.NumGroups",
		"accel.Thread.SubgroupAddF32",
		"accel.Thread.SubgroupAll",
		"accel.Thread.SubgroupBallot",
		"accel.Thread.SubgroupBarrier",
		"accel.Thread.SubgroupAny",
		"accel.Thread.SubgroupBroadcastF32",
		"accel.Thread.SubgroupBroadcastFirstF32",
		"accel.Thread.SubgroupElect",
		"accel.Thread.SubgroupExclusiveAddF32",
		"accel.Thread.SubgroupInclusiveAddF32",
		"accel.Thread.SubgroupIndex",
		"accel.Thread.SubgroupLane",
		"accel.Thread.SubgroupMaxF32",
		"accel.Thread.SubgroupMaxI32",
		"accel.Thread.SubgroupMaxU32",
		"accel.Thread.SubgroupMinF32",
		"accel.Thread.SubgroupMinI32",
		"accel.Thread.SubgroupMinU32",
		"accel.Thread.SubgroupShuffleDownF32",
		"accel.Thread.SubgroupShuffleF32",
		"accel.Thread.SubgroupShuffleUpF32",
		"accel.Thread.SubgroupShuffleXorF32",
		"accel.Thread.SubgroupSize",
		"accel.Thread.WorkgroupSize",
		"accel.ToBFloat16",
		"accel.ToFloat16",
		"accel.Vertex.InstanceIndex",
		"accel.Vertex.VertexIndex",
		"accel.XorU32",
		"accel/kmath.Abs",
		"accel/kmath.Cos",
		"accel/kmath.Exp",
		"accel/kmath.Log",
		"accel/kmath.Max",
		"accel/kmath.Min",
		"accel/kmath.RSqrt",
		"accel/kmath.Sin",
		"accel/kmath.Sqrt",
		"accel/kmath.Tanh",
		"accel/kmath.ToI32",
		"accel/kmath.ToU32",
	}
	sort.Strings(want)
	if len(names) != len(want) {
		t.Fatalf("Names() = %v, want %v", names, want)
	}
	for i := range want {
		if names[i] != want[i] {
			t.Errorf("Names()[%d] = %q, want %q (sorted, so a table change shows here)", i, names[i], want[i])
		}
	}

	// The version is read from the constant rather than written out, so a bump
	// does not need this test edited. What the test is for is the *contents*
	// changing silently, which the name list above catches; the version line's
	// job is to make a table whose shape changed produce a different digest.
	d := intrin.Digest()
	if !strings.HasPrefix(d, fmt.Sprintf("intrin/%d\n", intrin.ABIVersion)) {
		t.Errorf("digest does not start with its ABI version: %q", strings.SplitN(d, "\n", 2)[0])
	}
	if intrin.Digest() != d {
		t.Error("the digest is not stable across calls")
	}
	for _, name := range want {
		if !strings.Contains(d, name) {
			t.Errorf("the digest omits %s, so adding or retyping it would not make a "+
				"generated file stale", name)
		}
	}
	// The digest carries the resolved key too, since that is what resolution
	// actually uses and a change to it is equally ABI-visible.
	if !strings.Contains(d, "internal/kernel.Thread.GlobalID") {
		t.Error("the digest omits the resolved identity")
	}
}

func TestStageString(t *testing.T) {
	if got := intrin.Flat.String(); got != "flat" {
		t.Errorf("Flat = %q", got)
	}
	if got := intrin.Cooperative.String(); got != "cooperative" {
		t.Errorf("Cooperative = %q", got)
	}
}

// TestKMathResolvesAndCarriesItsClass checks the free-function half of the
// table against the real package, and the class each entry records.
//
// The class is not decoration. A test that asserts bits where only a bound
// holds is flaky on the first backend that rounds differently, and one that
// asserts a tolerance where bits are guaranteed hides a real difference. Which
// of the two applies is a property of the operation, so it lives here.
func TestKMathResolvesAndCarriesItsClass(t *testing.T) {
	pkg := loadKMathPackage(t)

	for _, tc := range []struct {
		name   string
		op     ir.Opcode
		params int
		class  intrin.Class
		// result is per-case since specs/051-float-to-int.md: kmath was all f32
		// in and f32 out until the saturating conversions, which take a float
		// and return an integer.
		result ir.Kind
	}{
		{"Sqrt", ir.OpSqrt, 1, intrin.ClassBounded, ir.F32},
		{"RSqrt", ir.OpRSqrt, 1, intrin.ClassBounded, ir.F32},
		{"ToI32", ir.OpToI32, 1, intrin.ClassExact, ir.I32},
		{"ToU32", ir.OpToU32, 1, intrin.ClassExact, ir.U32},
		{"Exp", ir.OpExp, 1, intrin.ClassBounded, ir.F32},
		{"Log", ir.OpLog, 1, intrin.ClassBounded, ir.F32},
		{"Sin", ir.OpSin, 1, intrin.ClassBounded, ir.F32},
		{"Cos", ir.OpCos, 1, intrin.ClassBounded, ir.F32},
		{"Tanh", ir.OpTanh, 1, intrin.ClassBounded, ir.F32},
		{"Abs", ir.OpAbs, 1, intrin.ClassExact, ir.F32},
		{"Min", ir.OpMin, 2, intrin.ClassExact, ir.F32},
		{"Max", ir.OpMax, 2, intrin.ClassExact, ir.F32},
	} {
		t.Run(tc.name, func(t *testing.T) {
			obj := pkg.Scope().Lookup(tc.name)
			if obj == nil {
				t.Fatalf("accel/kmath has no %s, so the table names something that is not there", tc.name)
			}
			fn, ok := obj.(*types.Func)
			if !ok {
				t.Fatalf("%s is %T, not a function", tc.name, obj)
			}
			got, ok := intrin.Lookup(fn)
			if !ok {
				t.Fatalf("kmath.%s did not resolve to an intrinsic", tc.name)
			}
			if got.Op != tc.op {
				t.Errorf("Op = %v, want %v", got.Op, tc.op)
			}
			if got.Params != tc.params {
				t.Errorf("Params = %d, want %d", got.Params, tc.params)
			}
			if got.Class != tc.class {
				t.Errorf("Class = %v, want %v", got.Class, tc.class)
			}
			if got.Stage != intrin.Flat {
				t.Errorf("Stage = %v: scalar math needs no rendezvous", got.Stage)
			}
			if got.Result != tc.result {
				t.Errorf("Result = %v, want %v", got.Result, tc.result)
			}
		})
	}
}

// TestEveryKMathExportIsAnIntrinsic is the check that keeps the package and the
// table from drifting apart.
//
// A function exported from accel/kmath that the table does not know is one a
// kernel can call and the compiler cannot lower, and the failure lands on
// whoever writes the kernel rather than on whoever added the function.
func TestEveryKMathExportIsAnIntrinsic(t *testing.T) {
	pkg := loadKMathPackage(t)
	for _, name := range pkg.Scope().Names() {
		obj := pkg.Scope().Lookup(name)
		fn, ok := obj.(*types.Func)
		if !ok || !obj.Exported() {
			continue
		}
		if _, ok := intrin.Lookup(fn); !ok {
			t.Errorf("accel/kmath exports %s and the intrinsic table does not know it: a kernel "+
				"could call it and the compiler could not lower it", name)
		}
	}
}

func TestClassString(t *testing.T) {
	if got := intrin.ClassExact.String(); got != "exact" {
		t.Errorf("ClassExact = %q", got)
	}
	if got := intrin.ClassBounded.String(); got != "bounded" {
		t.Errorf("ClassBounded = %q", got)
	}
}
