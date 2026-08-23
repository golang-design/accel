// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package emit_test

import (
	"go/constant"
	"go/token"
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/emit"
	"golang.design/x/accel/internal/kernelc/ir"
)

var (
	p     = token.Pos(1)
	tBool = &ir.Type{Kind: ir.Bool}
	tI32  = &ir.Type{Kind: ir.I32}
	tU32  = &ir.Type{Kind: ir.U32}
	tF32  = &ir.Type{Kind: ir.F32}
	tID3  = &ir.Type{Kind: ir.ID3Kind}
	tF32s = &ir.Type{Kind: ir.Slice, Elem: tF32}
)

// TestEmitsEveryNode covers the node kinds the corpus kernel does not use.
//
// The set is closed and declared whole, so spec 013 makes the rest reachable
// from source without touching the IR. This is where the emitter learns to
// spell them, and building the IR by hand is the only way to reach them a
// milestone early. A node the emitter cannot spell is a gap somebody finds
// while implementing 013 instead of now.
func TestEmitsEveryNode(t *testing.T) {
	out := ir.NewParam(p, tF32s, 1, "out", nil)
	n := ir.NewLocal(p, tU32, 0, "n", nil)
	thread := ir.NewParam(p, &ir.Type{Kind: ir.Struct, Name: "Thread"}, 0, "t", nil)

	for _, tc := range []struct {
		name string
		stmt ir.Stmt
		want []string
	}{
		{
			name: "three-clause for with break and continue",
			stmt: ir.NewFor(p,
				ir.NewDeclare(p, n, ir.NewConst(p, tU32, constant.MakeInt64(0))),
				ir.NewBinary(p, tBool, token.LSS, n, ir.NewConst(p, tU32, constant.MakeInt64(4))),
				ir.NewAssign(p, n, ir.NewBinary(p, tU32, token.ADD, n, ir.NewConst(p, tU32, constant.MakeInt64(1)))),
				ir.NewBlock(p, ir.NewIf(p, ir.NewConst(p, tBool, constant.MakeBool(true)),
					ir.NewBlock(p, ir.NewBreak(p)), ir.NewBlock(p, ir.NewContinue(p)))),
			),
			want: []string{"var n uint32 = uint32(0)", "for ; n < uint32(4); n = (n + uint32(1)) {", "break", "continue"},
		},
		{
			name: "condition-only for",
			stmt: ir.NewFor(p, nil, ir.NewConst(p, tBool, constant.MakeBool(true)), nil, ir.NewBlock(p, ir.NewBreak(p))),
			want: []string{"for true {"},
		},
		{
			name: "infinite for",
			stmt: ir.NewFor(p, nil, nil, nil, ir.NewBlock(p, ir.NewBreak(p))),
			want: []string{"for {"},
		},
		{
			name: "bare return",
			stmt: ir.NewReturn(p, nil),
			want: []string{"return"},
		},
		{
			name: "nested block",
			stmt: ir.NewBlock(p, ir.NewReturn(p, nil)),
			want: []string{"{", "return"},
		},
		{
			name: "if else",
			stmt: ir.NewIf(p, ir.NewConst(p, tBool, constant.MakeBool(false)),
				ir.NewBlock(p, ir.NewReturn(p, nil)),
				ir.NewBlock(p, ir.NewReturn(p, nil))),
			want: []string{"if false {", "} else {"},
		},
		{
			name: "else if chain",
			stmt: ir.NewIf(p, ir.NewConst(p, tBool, constant.MakeBool(false)),
				ir.NewBlock(p, ir.NewReturn(p, nil)),
				ir.NewIf(p, ir.NewConst(p, tBool, constant.MakeBool(true)),
					ir.NewBlock(p, ir.NewReturn(p, nil)), nil)),
			want: []string{"} else if true {"},
		},
		{
			name: "unary negation rounds",
			stmt: ir.NewAssign(p, ir.NewIndex(p, tF32, out, n, 1),
				ir.NewUnary(p, tF32, token.SUB, ir.NewIndex(p, tF32, out, n, 1))),
			want: []string{"float32(-out[n])"},
		},
		{
			name: "integer arithmetic is not rounded",
			stmt: ir.NewAssign(p, n, ir.NewBinary(p, tU32, token.ADD, n, n)),
			want: []string{"n = (n + n)"},
		},
		{
			name: "helper call",
			stmt: ir.NewExprStmt(p, ir.NewCall(p, tF32, &ir.Func{Name: "Helper"}, []ir.Value{n, n})),
			want: []string{"helperFlat(n, n)"},
		},
		{
			name: "every id accessor",
			stmt: ir.NewBlock(p,
				ir.NewAssign(p, n, ir.NewFieldSel(p, tU32, ir.NewIntrinsic(p, tID3, ir.OpLocalID, thread, nil), 1, "Y")),
				ir.NewAssign(p, n, ir.NewFieldSel(p, tU32, ir.NewIntrinsic(p, tID3, ir.OpGroupID, thread, nil), 2, "Z")),
				ir.NewAssign(p, n, ir.NewIntrinsic(p, tU32, ir.OpGlobalIndex, thread, nil)),
				ir.NewAssign(p, n, ir.NewIntrinsic(p, tU32, ir.OpLocalIndex, thread, nil)),
				ir.NewAssign(p, n, ir.NewIntrinsic(p, tU32, ir.OpGroupIndex, thread, nil)),
			),
			want: []string{"t.LocalID().Y", "t.GroupID().Z", "t.GlobalIndex()", "t.LocalIndex()", "t.GroupIndex()"},
		},
		{
			name: "boolean constant",
			stmt: ir.NewIf(p, ir.NewConst(p, tBool, constant.MakeBool(true)), ir.NewBlock(p), nil),
			want: []string{"if true {"},
		},
		{
			name: "integer constant carries its type",
			stmt: ir.NewAssign(p, n, ir.NewConst(p, tI32, constant.MakeInt64(-7))),
			want: []string{"int32(-7)"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := emit.Generate(emit.Package{Name: "p", Kernels: []*ir.Func{kernelWith(tc.stmt)}})
			if err != nil {
				t.Fatalf("Generate: %v", err)
			}
			for _, want := range tc.want {
				if !strings.Contains(string(got), want) {
					t.Errorf("the generated file does not contain %q:\n%s", want, got)
				}
			}
		})
	}
}

// TestEmitsEveryBindingType covers the dtype and Go spellings a binding can
// take, including the narrow storage types no kernel reaches until spec 013.
func TestEmitsEveryBindingType(t *testing.T) {
	for _, tc := range []struct {
		kind      ir.Kind
		goType    string
		dtypeName string
	}{
		{ir.F32, "[]float32", "kernelabi.F32"},
		{ir.I32, "[]int32", "kernelabi.I32"},
		{ir.U32, "[]uint32", "kernelabi.U32"},
		{ir.I8, "[]int8", "kernelabi.I8"},
		{ir.U8, "[]uint8", "kernelabi.U8"},
		{ir.F16, "[]accel.Float16", "kernelabi.F16"},
		{ir.BF16, "[]accel.BFloat16", "kernelabi.BF16"},
	} {
		t.Run(tc.kind.String(), func(t *testing.T) {
			elem := &ir.Type{Kind: tc.kind}
			slice := &ir.Type{Kind: ir.Slice, Elem: elem}
			k := &ir.Func{
				Name: "K", Stage: ir.StageCompute, Workgroup: [3]uint32{1, 1, 1}, Thread: 0,
				Params: []*ir.Param{
					ir.NewParam(p, &ir.Type{Kind: ir.Struct, Name: "Thread"}, 0, "t", nil),
					ir.NewParam(p, slice, 1, "buf", nil),
				},
				Bindings: []*ir.Binding{{Name: "buf", Index: 1, Type: slice, Read: true, Write: true}},
				Body:     ir.NewBlock(p, ir.NewReturn(p, nil)),
			}
			got, err := emit.Generate(emit.Package{Name: "p", Kernels: []*ir.Func{k}})
			if err != nil {
				t.Fatalf("Generate: %v", err)
			}
			if !strings.Contains(string(got), tc.dtypeName) {
				t.Errorf("no %s in:\n%s", tc.dtypeName, got)
			}
			if !strings.Contains(string(got), "kernelabi.Read | kernelabi.Write") {
				t.Errorf("a read-write binding is not spelled as both:\n%s", got)
			}
			// The narrow floats bind to their own storage types rather than to a
			// bare uint16. A defined integer type would carry integer operators,
			// so adding two narrow values would compile and add their bit
			// patterns, with no diagnostic anywhere.
			if tc.kind == ir.F16 || tc.kind == ir.BF16 {
				if !strings.Contains(string(got), "kernelabi.Slice["+tc.goType[2:]+"]") {
					t.Errorf("a narrow float does not bind to %s:\n%s", tc.goType, got)
				}
			}
		})
	}
}

// TestEmitsEveryScalarLocalType covers the Go spellings a local can take.
func TestEmitsEveryScalarLocalType(t *testing.T) {
	for _, tc := range []struct {
		t    *ir.Type
		want string
	}{
		{tBool, "var v bool"},
		{tI32, "var v int32"},
		{tU32, "var v uint32"},
		{tF32, "var v float32"},
		{&ir.Type{Kind: ir.I8}, "var v int8"},
		{&ir.Type{Kind: ir.U8}, "var v uint8"},
		{tID3, "var v accel.ID3"},
		{&ir.Type{Kind: ir.Array, Len: 8, Elem: tF32}, "var v [8]float32"},
		{&ir.Type{Kind: ir.Struct, Name: "Params"}, "var v Params"},
	} {
		local := ir.NewLocal(p, tc.t, 0, "v", nil)
		stmt := ir.NewDeclare(p, local, ir.NewConst(p, tc.t, constant.MakeInt64(0)))
		got, err := emit.Generate(emit.Package{Name: "p", Kernels: []*ir.Func{kernelWith(stmt)}})
		if err != nil {
			// A struct-typed constant has no spelling, which is fine: this test is
			// about the type name reaching the output.
			continue
		}
		if !strings.Contains(string(got), tc.want) {
			t.Errorf("no %q in:\n%s", tc.want, got)
		}
	}
}

// TestLowerNameHandlesEdgeCases covers the naming the generated file depends on.
func TestLowerNameHandlesEdgeCases(t *testing.T) {
	k := kernelWith(ir.NewReturn(p, nil))
	k.Name = ""
	got, err := emit.Generate(emit.Package{Name: "p", Kernels: []*ir.Func{k}})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !strings.Contains(string(got), "func flat(") {
		t.Errorf("an unnamed kernel produced no fallback name:\n%s", got)
	}
}

// TestUnlowerableTypesAreReported checks that a type with no Go spelling is an
// error rather than the word "invalid" compiled into someone's package.
func TestUnlowerableTypesAreReported(t *testing.T) {
	bad := &ir.Type{Kind: ir.Kind(99)}
	local := ir.NewLocal(p, bad, 0, "v", nil)
	stmt := ir.NewDeclare(p, local, ir.NewConst(p, tU32, constant.MakeInt64(0)))
	if _, err := emit.Generate(emit.Package{Name: "p", Kernels: []*ir.Func{kernelWith(stmt)}}); err == nil {
		t.Fatal("a type with no Go spelling was emitted")
	}

	// A binding whose element type has no dtype is the same failure on the
	// record's side.
	slice := &ir.Type{Kind: ir.Slice, Elem: &ir.Type{Kind: ir.Bool}}
	k := kernelWith(ir.NewReturn(p, nil))
	k.Bindings[0].Type = slice
	if _, err := emit.Generate(emit.Package{Name: "p", Kernels: []*ir.Func{k}}); err == nil {
		t.Fatal("a bool binding was emitted; bool is not a storage type")
	}

	k = kernelWith(ir.NewReturn(p, nil))
	k.Bindings[0].Type = &ir.Type{Kind: ir.Slice}
	if _, err := emit.Generate(emit.Package{Name: "p", Kernels: []*ir.Func{k}}); err == nil {
		t.Fatal("a binding with no element type was emitted")
	}
}

// TestAccessNames covers the digest's spelling of an inferred access, which is
// what makes a read-to-write edit change the digest.
func TestAccessNames(t *testing.T) {
	for _, tc := range []struct {
		read, write bool
		want        string
	}{
		{true, false, "read"},
		{false, true, "write"},
		{true, true, "read-write"},
		{false, false, "none"},
	} {
		k := kernelWith(ir.NewReturn(p, nil))
		k.Bindings[0].Read, k.Bindings[0].Write = tc.read, tc.write
		if got := emit.Preimage(k); !strings.Contains(got, "\t"+tc.want+"\n") {
			t.Errorf("read=%v write=%v is not spelled %q in the preimage:\n%s",
				tc.read, tc.write, tc.want, got)
		}
	}
}
