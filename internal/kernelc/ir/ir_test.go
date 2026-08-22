// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package ir_test

import (
	"go/constant"
	"go/token"
	"reflect"
	"testing"

	"golang.design/x/accel/internal/kernelc/ir"
)

var (
	u32   = &ir.Type{Kind: ir.U32}
	f32   = &ir.Type{Kind: ir.F32}
	boolT = &ir.Type{Kind: ir.Bool}
	id3   = &ir.Type{Kind: ir.ID3Kind}
	f32s  = &ir.Type{Kind: ir.Slice, Elem: f32}
)

// scaleBody builds the IR for spec 012's kernel, by hand.
//
// Building it by hand is the point at this slice: the front end does not exist
// yet, so this is the only thing that says the node set can express the kernel
// the milestone is defined by. If it could not, that is a finding about the set
// rather than about the front end, and it is much cheaper to learn now.
func scaleBody(t *testing.T) (*ir.Func, ir.Value) {
	t.Helper()
	p := token.Pos(1)

	in := ir.NewParam(p, f32s, 1, "in", nil)
	out := ir.NewParam(p, f32s, 2, "out", nil)

	// i := t.GlobalID().X
	gid := ir.NewIntrinsic(p, id3, ir.OpGlobalID, ir.NewParam(p, &ir.Type{Kind: ir.Struct, Name: "Thread"}, 0, "t", nil), nil)
	x := ir.NewFieldSel(p, u32, gid, 0, "X")
	i := ir.NewLocal(p, u32, 0, "i", nil)
	decl := ir.NewDeclare(p, i, x)

	// if i < uint32(len(out)) { out[i] = in[i] * 2 }
	cond := ir.NewBinary(p, boolT, token.LSS, i, ir.NewConvert(p, u32, ir.NewLen(p, &ir.Type{Kind: ir.I32}, out)))
	load := ir.NewIndex(p, f32, in, i, 1)
	scaled := ir.NewBinary(p, f32, token.MUL, load, ir.NewConst(p, f32, constant.MakeInt64(2)))
	store := ir.NewAssign(p, ir.NewIndex(p, f32, out, i, 2), scaled)
	cons := ir.NewIf(p, cond, ir.NewBlock(p, store), nil)

	fn := &ir.Func{
		Name: "Scale", Kernel: true, Workgroup: [3]uint32{64, 1, 1}, Thread: 0,
		Params:     []*ir.Param{ir.NewParam(p, nil, 0, "t", nil), in, out},
		Bindings:   []*ir.Binding{{Name: "in", Index: 1, Type: f32s, Read: true}, {Name: "out", Index: 2, Type: f32s, Write: true}},
		Body:       ir.NewBlock(p, decl, cons),
		Intrinsics: []string{"accel.Thread.GlobalID"},
	}
	return fn, cond
}

// TestNodeSetExpressesTheMilestoneKernel checks that spec 012's kernel is
// representable, node for node.
func TestNodeSetExpressesTheMilestoneKernel(t *testing.T) {
	fn, _ := scaleBody(t)

	if len(fn.Body.List) != 2 {
		t.Fatalf("body has %d statements, want a declaration and an if", len(fn.Body.List))
	}
	decl, ok := fn.Body.List[0].(*ir.Declare)
	if !ok {
		t.Fatalf("first statement is %T, want *ir.Declare", fn.Body.List[0])
	}
	if decl.Local.Name != "i" || decl.Local.Type().Kind != ir.U32 {
		t.Errorf("declared %q of kind %v, want i of kind u32", decl.Local.Name, decl.Local.Type().Kind)
	}
	sel, ok := decl.Init.(*ir.FieldSel)
	if !ok {
		t.Fatalf("initializer is %T, want a field selection", decl.Init)
	}
	call, ok := sel.X.(*ir.IntrinsicCall)
	if !ok {
		t.Fatalf("selection is on %T, want an intrinsic call", sel.X)
	}
	if call.Op != ir.OpGlobalID {
		t.Errorf("intrinsic is %v, want GlobalID", call.Op)
	}
	if call.Recv == nil {
		t.Error("the intrinsic lost its Thread receiver, which is what distinguishes it from a free function")
	}

	cons := fn.Body.List[1].(*ir.If)
	store := cons.Then.List[0].(*ir.Assign)
	lhs, ok := store.LHS.(*ir.IndexExpr)
	if !ok {
		t.Fatalf("assignment target is %T, want an index", store.LHS)
	}
	// The binding index is on the node, which is what lets access inference run
	// on the IR instead of as a second pass over the AST per target.
	if lhs.Binding != 2 {
		t.Errorf("the written index names binding %d, want 2", lhs.Binding)
	}
	if rhs, ok := store.RHS.(*ir.Binary); !ok || rhs.Op != token.MUL {
		t.Errorf("stored value is %T, want a multiplication", store.RHS)
	}
}

// TestEveryNodeCarriesAPosition is the property a diagnostic depends on. A node
// with a zero position produces an error pointing at the top of the file, which
// sends a reader to the wrong place, and spec 013 asserts positions and not
// only messages.
func TestEveryNodeCarriesAPosition(t *testing.T) {
	fn, _ := scaleBody(t)
	var walk func(n ir.Node, path string)
	seen := 0
	walk = func(n ir.Node, path string) {
		if n == nil || reflect.ValueOf(n).IsNil() {
			return
		}
		seen++
		if n.Pos() == token.NoPos {
			t.Errorf("%s (%T) has no position", path, n)
		}
		switch v := n.(type) {
		case *ir.Block:
			for i, s := range v.List {
				walk(s, path+".List["+itoa(i)+"]")
			}
		case *ir.Declare:
			walk(v.Local, path+".Local")
			walk(v.Init, path+".Init")
		case *ir.Assign:
			walk(v.LHS, path+".LHS")
			walk(v.RHS, path+".RHS")
		case *ir.If:
			walk(v.Cond, path+".Cond")
			walk(v.Then, path+".Then")
			walk(v.Else, path+".Else")
		case *ir.FieldSel:
			walk(v.X, path+".X")
		case *ir.IndexExpr:
			walk(v.X, path+".X")
			walk(v.Index, path+".Index")
		case *ir.Binary:
			walk(v.X, path+".X")
			walk(v.Y, path+".Y")
		case *ir.Unary:
			walk(v.X, path+".X")
		case *ir.Convert:
			walk(v.X, path+".X")
		case *ir.Len:
			walk(v.X, path+".X")
		case *ir.IntrinsicCall:
			walk(v.Recv, path+".Recv")
			for i, a := range v.Args {
				walk(a, path+".Args["+itoa(i)+"]")
			}
		}
	}
	walk(fn.Body, "body")
	if seen < 10 {
		t.Fatalf("walked only %d nodes; the traversal is not reaching the tree", seen)
	}
}

func itoa(i int) string { return string(rune('0' + i)) }

// TestTypeStrings covers what a diagnostic prints.
func TestTypeStrings(t *testing.T) {
	for _, tc := range []struct {
		t    *ir.Type
		want string
	}{
		{f32, "f32"},
		{f32s, "[]f32"},
		{&ir.Type{Kind: ir.Array, Len: 256, Elem: f32}, "[256]f32"},
		{&ir.Type{Kind: ir.Struct, Name: "Params"}, "Params"},
		{id3, "ID3"},
		{&ir.Type{Kind: ir.Invalid}, "invalid"},
		{&ir.Type{Kind: ir.Kind(99)}, "Kind(99)"},
	} {
		if got := tc.t.String(); got != tc.want {
			t.Errorf("Type.String() = %q, want %q", got, tc.want)
		}
	}
}

// TestNarrowKindsCarryNoArithmetic is spec 002's rule that narrow dtypes are
// storage formats which convert on load and store. If F16 were numeric here the
// emitter could produce arithmetic no backend without native narrow support can
// run, which is a portability failure rather than a compile error.
func TestNarrowKindsCarryNoArithmetic(t *testing.T) {
	for _, k := range []ir.Kind{ir.I32, ir.U32, ir.F32} {
		if !k.Numeric() {
			t.Errorf("%v is not numeric", k)
		}
	}
	for _, k := range []ir.Kind{ir.F16, ir.BF16, ir.I8, ir.U8, ir.Bool, ir.ID3Kind, ir.Slice, ir.Array, ir.Struct, ir.Invalid} {
		if k.Numeric() {
			t.Errorf("%v reports arithmetic; narrow dtypes are storage and the rest are not scalars", k)
		}
	}
}

func TestOpcodeStrings(t *testing.T) {
	for _, tc := range []struct {
		op   ir.Opcode
		want string
	}{
		{ir.OpGlobalID, "GlobalID"},
		{ir.OpBarrier, "Barrier"},
		{ir.OpInvalid, "invalid"},
		{ir.Opcode(99), "Opcode(99)"},
	} {
		if got := tc.op.String(); got != tc.want {
			t.Errorf("Opcode.String() = %q, want %q", got, tc.want)
		}
	}
}

// TestAllNodeConstructorsWork exercises the builders the front end will use,
// including the ones this milestone does not construct. They are declared
// because the set is closed by design, and a declared-but-unbuildable node is a
// gap somebody discovers while implementing 013 instead of now.
func TestAllNodeConstructorsWork(t *testing.T) {
	p := token.Pos(7)
	local := ir.NewLocal(p, u32, 0, "n", nil)

	values := []ir.Value{
		ir.NewConst(p, u32, constant.MakeInt64(1)),
		ir.NewParam(p, f32s, 0, "buf", nil),
		local,
		ir.NewFieldSel(p, u32, ir.NewParam(p, id3, 0, "id", nil), 1, "Y"),
		ir.NewIndex(p, f32, ir.NewParam(p, f32s, 0, "buf", nil), local, 0),
		ir.NewUnary(p, u32, token.SUB, local),
		ir.NewBinary(p, u32, token.ADD, local, local),
		ir.NewConvert(p, f32, local),
		ir.NewCall(p, u32, &ir.Func{Name: "helper"}, []ir.Value{local}),
		ir.NewIntrinsic(p, u32, ir.OpLocalIndex, nil, nil),
		ir.NewLen(p, u32, ir.NewParam(p, f32s, 0, "buf", nil)),
	}
	for _, v := range values {
		if v.Pos() != p {
			t.Errorf("%T lost its position", v)
		}
		if v.Type() == nil {
			t.Errorf("%T has no type", v)
		}
	}

	stmts := []ir.Stmt{
		ir.NewBlock(p),
		ir.NewDeclare(p, local, nil),
		ir.NewAssign(p, local, local),
		ir.NewExprStmt(p, values[0]),
		ir.NewIf(p, values[0], ir.NewBlock(p), nil),
		ir.NewFor(p, nil, values[0], nil, ir.NewBlock(p)),
		ir.NewBreak(p),
		ir.NewContinue(p),
		ir.NewReturn(p, nil),
	}
	for _, s := range stmts {
		if s.Pos() != p {
			t.Errorf("%T lost its position", s)
		}
	}
	if len(stmts) != 9 {
		t.Errorf("the statement set has %d members; spec 004 closes it at 9", len(stmts))
	}
}
