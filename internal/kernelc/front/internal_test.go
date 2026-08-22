// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package front

import (
	"go/ast"
	"go/token"
	"go/types"
	"strings"
	"testing"

	"golang.design/x/accel/internal/kernelc/intrin"
	"golang.design/x/accel/internal/kernelc/ir"
)

// newChecker builds a checker over an empty package, for exercising the
// builders directly.
//
// Directly, because the branches below are defensive: they are reached when
// go/types hands the front end something a Go source file cannot spell. That
// does not make them dead. The front end consumes whatever the toolchain
// produced, and Go 1.27 already moved that boundary once by admitting generic
// methods the parser used to discard.
func newChecker(t *testing.T) *checker {
	t.Helper()
	fset := token.NewFileSet()
	fset.AddFile("synth.go", -1, 4096)
	return &checker{
		fset:    fset,
		info:    &types.Info{Types: map[ast.Expr]types.TypeAndValue{}, Uses: map[*ast.Ident]types.Object{}, Defs: map[*ast.Ident]types.Object{}, Selections: map[*ast.SelectorExpr]*types.Selection{}},
		locals:  map[types.Object]*ir.Local{},
		current: &ir.Func{Name: "K"},
	}
}

func TestDiagnosticFormatting(t *testing.T) {
	d := Diagnostic{Pos: token.Position{Filename: "k.go", Line: 12, Column: 3}, Msg: "no"}
	if got, want := d.Error(), "k.go:12:3: no"; got != want {
		t.Errorf("Diagnostic.Error() = %q, want %q", got, want)
	}
	ds := Diagnostics{d, Diagnostic{Pos: token.Position{Filename: "k.go", Line: 13}, Msg: "also no"}}
	if got := ds.Error(); !strings.Contains(got, "no") || !strings.Contains(got, "also no") {
		t.Errorf("Diagnostics.Error() = %q, want both messages", got)
	}
}

func TestStmtName(t *testing.T) {
	for _, tc := range []struct {
		s    ast.Stmt
		want string
	}{
		{&ast.SwitchStmt{}, "switch"},
		{&ast.TypeSwitchStmt{}, "switch"},
		{&ast.SelectStmt{}, "select"},
		{&ast.RangeStmt{}, "range"},
		{&ast.BadStmt{}, "this statement"},
	} {
		if got := stmtName(tc.s); got != tc.want {
			t.Errorf("stmtName(%T) = %q, want %q", tc.s, got, tc.want)
		}
	}
}

// TestUnhandledStatementIsRejected covers the fall-through, which is what makes
// the node set closed rather than merely documented: a statement kind nobody
// listed is rejected, not passed through.
func TestUnhandledStatementIsRejected(t *testing.T) {
	c := newChecker(t)
	if got := c.stmt(&ast.BadStmt{}); got != nil {
		t.Error("an unhandled statement was accepted")
	}
	if len(c.diags) != 1 || !strings.Contains(c.diags[0].Msg, "outside the closed IR node set") {
		t.Errorf("diagnostics = %v", c.diags)
	}
}

func TestUnhandledExpressionIsRejected(t *testing.T) {
	for _, tc := range []struct {
		e    ast.Expr
		want string
	}{
		{&ast.StarExpr{X: &ast.BadExpr{}}, "pointer indirection"},
		{&ast.TypeAssertExpr{X: &ast.BadExpr{}}, "interfaces have no memory model"},
		{&ast.BadExpr{}, "outside the closed IR node set"},
		{&ast.ChanType{}, "outside the closed IR node set"},
	} {
		c := newChecker(t)
		if got := c.value(tc.e); got != nil {
			t.Errorf("%T was accepted", tc.e)
		}
		if len(c.diags) != 1 || !strings.Contains(c.diags[0].Msg, tc.want) {
			t.Errorf("%T: diagnostics = %v, want %q", tc.e, c.diags, tc.want)
		}
	}
}

func TestUnaryRejectsOperatorsWithNoLowering(t *testing.T) {
	c := newChecker(t)
	// A receive is a unary operator with no lowering, and it is spelled the same
	// way as the ones that do have lowerings.
	if got := c.unary(&ast.UnaryExpr{OpPos: 1, Op: token.ARROW, X: &ast.BadExpr{}}); got != nil {
		t.Error("a receive was accepted")
	}
	if len(c.diags) != 1 || !strings.Contains(c.diags[0].Msg, "unary <- is outside the subset") {
		t.Errorf("diagnostics = %v", c.diags)
	}
}

func TestCompoundOpCoversEveryForm(t *testing.T) {
	for tok, want := range map[token.Token]token.Token{
		token.ADD_ASSIGN: token.ADD, token.SUB_ASSIGN: token.SUB,
		token.MUL_ASSIGN: token.MUL, token.QUO_ASSIGN: token.QUO,
		token.REM_ASSIGN: token.REM, token.AND_ASSIGN: token.AND,
		token.OR_ASSIGN: token.OR, token.XOR_ASSIGN: token.XOR,
		token.SHL_ASSIGN: token.SHL, token.SHR_ASSIGN: token.SHR,
	} {
		got, ok := compoundOp(tok)
		if !ok || got != want {
			t.Errorf("compoundOp(%v) = %v, %v; want %v", tok, got, ok, want)
		}
	}
	// Go's &^= has no counterpart in every target's operator set, so it is not
	// silently lowered to something close.
	if _, ok := compoundOp(token.AND_NOT_ASSIGN); ok {
		t.Error("&^= was accepted")
	}
	if _, ok := compoundOp(token.ASSIGN); ok {
		t.Error("plain assignment is not a compound operator")
	}
}

func TestAssignable(t *testing.T) {
	p := token.Pos(1)
	u32 := &ir.Type{Kind: ir.U32}
	if !assignable(ir.NewLocal(p, u32, 0, "n", nil)) {
		t.Error("a local is assignable")
	}
	if !assignable(ir.NewIndex(p, u32, nil, nil, 0)) {
		t.Error("an index is assignable")
	}
	if assignable(ir.NewConst(p, u32, nil)) {
		t.Error("a constant is not assignable")
	}
	if assignable(ir.NewParam(p, u32, 0, "in", nil)) {
		t.Error("a whole binding is not assignable: a kernel writes elements")
	}
}

func TestIRTypeRejections(t *testing.T) {
	c := newChecker(t)
	for _, tc := range []struct {
		t    types.Type
		want string
	}{
		{nil, "has no type"},
		{types.Typ[types.Int], "platform-width"},
		{types.Typ[types.Uintptr], "platform-width"},
		{types.Typ[types.Complex64], "not one of float32"},
		{types.NewMap(types.Typ[types.Int32], types.Typ[types.Int32]), "outside the subset"},
		{types.NewSlice(types.Typ[types.Bool]), "not one of float32"},
		{types.NewArray(types.Typ[types.String], 4), "not one of float32"},
		{types.NewChan(types.SendRecv, types.Typ[types.Int32]), "outside the subset"},
	} {
		if _, err := c.irType(tc.t); err == nil {
			t.Errorf("%v was accepted", tc.t)
		} else if !strings.Contains(err.Error(), tc.want) {
			t.Errorf("%v: error %q does not carry %q", tc.t, err, tc.want)
		}
	}
}

func TestIRTypeAccepts(t *testing.T) {
	c := newChecker(t)
	for _, tc := range []struct {
		t    types.Type
		want ir.Kind
	}{
		{types.Typ[types.Float32], ir.F32},
		{types.Typ[types.Int32], ir.I32},
		{types.Typ[types.Uint32], ir.U32},
		{types.Typ[types.Int8], ir.I8},
		{types.Typ[types.Uint8], ir.U8},
		{types.Typ[types.Bool], ir.Bool},
		{types.NewSlice(types.Typ[types.Float32]), ir.Slice},
		{types.NewArray(types.Typ[types.Float32], 64), ir.Array},
	} {
		got, err := c.irType(tc.t)
		if err != nil {
			t.Errorf("%v: %v", tc.t, err)
			continue
		}
		if got.Kind != tc.want {
			t.Errorf("%v mapped to %v, want %v", tc.t, got.Kind, tc.want)
		}
	}

	// A fixed-size array's extent comes off the type, which is what lets shared
	// memory have a static extent without inventing const generics.
	arr, err := c.irType(types.NewArray(types.Typ[types.Float32], 256))
	if err != nil {
		t.Fatal(err)
	}
	if arr.Len != 256 {
		t.Errorf("array extent is %d, want 256", arr.Len)
	}
}

// TestConstTypeWidensRatherThanRejecting is the rule that a constant's type is
// not a value's type: an untyped 0 indexing a buffer is fine, and a variable of
// type int is not.
func TestConstTypeWidensRatherThanRejecting(t *testing.T) {
	c := newChecker(t)
	for _, tc := range []struct {
		t    types.Type
		want ir.Kind
	}{
		{types.Typ[types.Int], ir.I32},
		{types.Typ[types.Uint], ir.I32},
		{types.Typ[types.Int64], ir.I32},
		{types.Typ[types.Uint16], ir.I32},
		{types.Typ[types.Float64], ir.F32},
		{types.Typ[types.Float32], ir.F32},
		{types.Typ[types.UntypedInt], ir.I32},
	} {
		got, err := c.constType(tc.t)
		if err != nil {
			t.Errorf("a constant of type %v was rejected: %v", tc.t, err)
			continue
		}
		if got.Kind != tc.want {
			t.Errorf("a constant of type %v mapped to %v, want %v", tc.t, got.Kind, tc.want)
		}
	}
	if _, err := c.constType(types.Typ[types.String]); err == nil {
		t.Error("a string constant was accepted")
	}
}

func TestParseWorkgroup(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want [3]uint32
		bad  bool
	}{
		{" workgroup=64", [3]uint32{64, 1, 1}, false},
		{" workgroup=16,8", [3]uint32{16, 8, 1}, false},
		{" workgroup=4,4,4", [3]uint32{4, 4, 4}, false},
		{" workgroup=16, 8", [3]uint32{16, 8, 1}, false},
		{"", [3]uint32{}, true},
		{" size=64", [3]uint32{}, true},
		{" workgroup=", [3]uint32{}, true},
		{" workgroup=0", [3]uint32{}, true},
		{" workgroup=-1", [3]uint32{}, true},
		{" workgroup=x", [3]uint32{}, true},
		{" workgroup=1,1,1,1", [3]uint32{}, true},
	} {
		got, err := parseWorkgroup(tc.in)
		if tc.bad {
			if err == nil {
				t.Errorf("parseWorkgroup(%q) was accepted", tc.in)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseWorkgroup(%q): %v", tc.in, err)
			continue
		}
		if got != tc.want {
			t.Errorf("parseWorkgroup(%q) = %v, want %v: an omitted axis is 1", tc.in, got, tc.want)
		}
	}
}

// TestRejectionsPropagate feeds every builder an operand that cannot be built.
//
// A builder signals a rejection by returning nil, and a missed nil check does
// not error: it produces a node with a nil operand that the emitter dereferences
// later, or drops silently. So each slot that can hold a value is given one that
// fails, and the builder must decline rather than construct.
func TestRejectionsPropagate(t *testing.T) {
	bad := func() ast.Expr { return &ast.BadExpr{From: 1, To: 2} }
	id := func(name string) *ast.Ident { return &ast.Ident{NamePos: 1, Name: name} }

	values := []struct {
		name string
		e    ast.Expr
	}{
		{"binary left", &ast.BinaryExpr{X: bad(), OpPos: 1, Op: token.ADD, Y: bad()}},
		{"unary operand", &ast.UnaryExpr{OpPos: 1, Op: token.SUB, X: bad()}},
		{"index target", &ast.IndexExpr{X: bad(), Index: bad()}},
		{"paren", &ast.ParenExpr{Lparen: 1, X: bad()}},
		{"selector target", &ast.SelectorExpr{X: bad(), Sel: id("X")}},
		{"call target", &ast.CallExpr{Fun: bad(), Lparen: 1}},
	}
	for _, tc := range values {
		c := newChecker(t)
		if got := c.value(tc.e); got != nil {
			t.Errorf("%s: built %T from an operand that could not be built", tc.name, got)
		}
		if len(c.diags) == 0 {
			t.Errorf("%s: declined silently", tc.name)
		}
	}

	stmts := []struct {
		name string
		s    ast.Stmt
	}{
		{"assign value", &ast.AssignStmt{Lhs: []ast.Expr{id("n")}, TokPos: 1, Tok: token.ASSIGN, Rhs: []ast.Expr{bad()}}},
		{"assign target", &ast.AssignStmt{Lhs: []ast.Expr{bad()}, TokPos: 1, Tok: token.ASSIGN, Rhs: []ast.Expr{id("n")}}},
		{"define value", &ast.AssignStmt{Lhs: []ast.Expr{id("n")}, TokPos: 1, Tok: token.DEFINE, Rhs: []ast.Expr{bad()}}},
		{"define non-name", &ast.AssignStmt{Lhs: []ast.Expr{bad()}, TokPos: 1, Tok: token.DEFINE, Rhs: []ast.Expr{id("n")}}},
		{"multiple assignment", &ast.AssignStmt{Lhs: []ast.Expr{id("a"), id("b")}, TokPos: 1, Tok: token.ASSIGN, Rhs: []ast.Expr{id("n")}}},
		{"compound with no lowering", &ast.AssignStmt{Lhs: []ast.Expr{id("n")}, TokPos: 1, Tok: token.AND_NOT_ASSIGN, Rhs: []ast.Expr{id("m")}}},
		{"increment target", &ast.IncDecStmt{X: bad(), TokPos: 1, Tok: token.INC}},
		{"if condition", &ast.IfStmt{If: 1, Cond: bad(), Body: &ast.BlockStmt{Lbrace: 1, Rbrace: 2}}},
		{"expression statement", &ast.ExprStmt{X: bad()}},
		{"return with a value", &ast.ReturnStmt{Return: 1, Results: []ast.Expr{id("n")}}},
		{"block member", &ast.BlockStmt{Lbrace: 1, List: []ast.Stmt{&ast.ExprStmt{X: bad()}}, Rbrace: 2}},
		{"non-var declaration", &ast.DeclStmt{Decl: &ast.GenDecl{TokPos: 1, Tok: token.CONST}}},
	}
	for _, tc := range stmts {
		c := newChecker(t)
		if got := c.stmt(tc.s); got != nil {
			t.Errorf("%s: built %T from a statement that could not be built", tc.name, got)
		}
		if len(c.diags) == 0 {
			t.Errorf("%s: declined silently", tc.name)
		}
	}
}

// TestBuiltinAndConversionRejections covers the two call shapes go/types tells
// apart, which is the whole reason the front end type-checks rather than
// walking an AST: float32(x) and Sqrt(x) have the same shape.
func TestBuiltinAndConversionRejections(t *testing.T) {
	c := newChecker(t)
	capBuiltin := types.Universe.Lookup("cap").(*types.Builtin)
	lenBuiltin := types.Universe.Lookup("len").(*types.Builtin)
	call := &ast.CallExpr{Fun: &ast.Ident{NamePos: 1, Name: "cap"}, Lparen: 1}
	if got := c.builtin(call, capBuiltin); got != nil {
		t.Error("cap was accepted")
	}
	if len(c.diags) != 1 || !strings.Contains(c.diags[0].Msg, "no allocator and no runtime") {
		t.Errorf("diagnostics = %v", c.diags)
	}

	// len with the wrong argument count cannot be spelled in Go, and the guard
	// is still here because the front end consumes whatever go/types produced.
	c = newChecker(t)
	if got := c.builtin(&ast.CallExpr{Fun: &ast.BadExpr{From: 1}, Lparen: 1}, lenBuiltin); got != nil {
		t.Error("len with no argument was accepted")
	}

	c = newChecker(t)
	if got := c.conversion(&ast.CallExpr{Fun: &ast.BadExpr{From: 1}, Lparen: 1}, types.Typ[types.Float32]); got != nil {
		t.Error("a conversion of nothing was accepted")
	}

	c = newChecker(t)
	if got := c.conversion(&ast.CallExpr{Fun: &ast.BadExpr{From: 1}, Lparen: 1, Args: []ast.Expr{&ast.BadExpr{From: 1}}}, types.Typ[types.String]); got != nil {
		t.Error("a conversion to string was accepted")
	}
}

// TestCalleeFuncShapes covers the call targets go/types can present.
func TestCalleeFuncShapes(t *testing.T) {
	c := newChecker(t)
	id := &ast.Ident{NamePos: 1, Name: "f"}

	if got := c.calleeFunc(id); got != nil {
		t.Error("an unresolved identifier produced a callee")
	}
	if got := c.calleeFunc(&ast.SelectorExpr{X: id, Sel: id}); got != nil {
		t.Error("an unresolved selector produced a callee")
	}
	// An instantiated generic function or method. Go 1.27 admits generic methods,
	// so this shape is reachable and is rejected rather than unwrapped: the
	// subset has no generics, and unwrapping one would lower a body as though its
	// type parameters were not there.
	if got := c.calleeFunc(&ast.IndexExpr{X: id, Lbrack: 1, Index: id, Rbrack: 2}); got != nil {
		t.Error("an instantiated generic produced a callee")
	}
	if got := c.calleeFunc(&ast.IndexListExpr{X: id, Lbrack: 1, Indices: []ast.Expr{id}, Rbrack: 2}); got != nil {
		t.Error("a multi-parameter instantiation produced a callee")
	}
	if got := c.calleeFunc(&ast.BadExpr{From: 1}); got != nil {
		t.Error("an unbuildable expression produced a callee")
	}
}

// TestSelectorRejections covers the field selections that are not an id
// component.
func TestSelectorRejections(t *testing.T) {
	p := token.Pos(1)
	for _, tc := range []struct {
		name string
		x    ir.Value
		sel  string
		want string
	}{
		{"not an id", ir.NewLocal(p, &ir.Type{Kind: ir.U32}, 0, "n", nil), "X", "has no field"},
		{"not a component", ir.NewLocal(p, &ir.Type{Kind: ir.ID3Kind}, 0, "id", nil), "W", "components X, Y, and Z"},
	} {
		c := newChecker(t)
		x := &ast.Ident{NamePos: 1, Name: "x"}
		obj := types.NewVar(1, nil, "x", types.Typ[types.Uint32])
		c.info.Uses[x] = obj
		c.locals[obj] = tc.x.(*ir.Local)

		got := c.selector(&ast.SelectorExpr{X: x, Sel: &ast.Ident{NamePos: 2, Name: tc.sel}})
		if got != nil {
			t.Errorf("%s: was accepted", tc.name)
		}
		if len(c.diags) != 1 || !strings.Contains(c.diags[0].Msg, tc.want) {
			t.Errorf("%s: diagnostics = %v, want %q", tc.name, c.diags, tc.want)
		}
	}
}

// TestIntrinsicCallArity covers the guard on a call's argument count. It cannot
// be spelled in Go for the zero-argument intrinsics this milestone has, and the
// guard is what keeps that true when spec 013 adds ones that take arguments.
func TestIntrinsicCallArity(t *testing.T) {
	c := newChecker(t)
	in := &intrin.Intrinsic{Authored: "accel.Thread.GlobalIndex", Op: ir.OpGlobalIndex, Result: ir.U32, Params: 0}
	call := &ast.CallExpr{
		Fun:    &ast.Ident{NamePos: 1, Name: "GlobalIndex"},
		Lparen: 1,
		Args:   []ast.Expr{&ast.BadExpr{From: 1}},
	}
	if got := c.intrinsicCall(call, nil, in); got != nil {
		t.Error("an intrinsic called with too many arguments was accepted")
	}
	if len(c.diags) != 1 || !strings.Contains(c.diags[0].Msg, "takes 0 arguments and got 1") {
		t.Errorf("diagnostics = %v", c.diags)
	}

	// An argument that cannot be built declines rather than producing a call
	// with a nil operand.
	c = newChecker(t)
	in = &intrin.Intrinsic{Authored: "x", Op: ir.OpGlobalIndex, Result: ir.U32, Params: 1}
	if got := c.intrinsicCall(call, nil, in); got != nil {
		t.Error("an intrinsic with an unbuildable argument was accepted")
	}

	// A receiver that cannot be built declines too. It is passed explicitly
	// rather than dug out of the selector, because whether a call has one is
	// something only the caller can tell: a method call and a package-qualified
	// call are the same AST shape.
	c = newChecker(t)
	in = &intrin.Intrinsic{Authored: "x", Op: ir.OpGlobalIndex, Result: ir.U32}
	if got := c.intrinsicCall(&ast.CallExpr{Lparen: 1}, &ast.BadExpr{From: 1}, in); got != nil {
		t.Error("an intrinsic with an unbuildable receiver was accepted")
	}
}

// TestIntrinsicsAreRecordedOnce checks the digest input: an intrinsic used
// twice appears once, in first-use order, so the digest does not depend on how
// often a body happens to call something.
func TestIntrinsicsAreRecordedOnce(t *testing.T) {
	c := newChecker(t)
	in := &intrin.Intrinsic{Authored: "accel.Thread.GlobalIndex", Op: ir.OpGlobalIndex, Result: ir.U32}
	call := &ast.CallExpr{Fun: &ast.Ident{NamePos: 1, Name: "GlobalIndex"}, Lparen: 1}
	for range 3 {
		if got := c.intrinsicCall(call, nil, in); got == nil {
			t.Fatalf("a well-formed intrinsic call was rejected: %v", c.diags)
		}
	}
	if want := []string{"accel.Thread.GlobalIndex"}; len(c.current.Intrinsics) != 1 || c.current.Intrinsics[0] != want[0] {
		t.Errorf("intrinsics = %v, want %v recorded once", c.current.Intrinsics, want)
	}
}

// TestInferAccessReachesEveryNode walks a body containing the statements this
// milestone does not build, since spec 013 will and an analysis that skips them
// would report a binding untouched.
func TestInferAccessReachesEveryNode(t *testing.T) {
	p := token.Pos(1)
	f32 := &ir.Type{Kind: ir.F32}
	f32s := &ir.Type{Kind: ir.Slice, Elem: f32}
	in := ir.NewParam(p, f32s, 0, "in", nil)
	out := ir.NewParam(p, f32s, 1, "out", nil)
	i := ir.NewLocal(p, &ir.Type{Kind: ir.U32}, 0, "i", nil)

	read := func() ir.Value { return ir.NewIndex(p, f32, in, i, 0) }
	k := &ir.Func{
		Name:     "K",
		Bindings: []*ir.Binding{{Name: "in", Index: 0}, {Name: "out", Index: 1}},
		Body: ir.NewBlock(p,
			ir.NewFor(p,
				ir.NewDeclare(p, i, ir.NewLen(p, f32, out)),
				ir.NewBinary(p, &ir.Type{Kind: ir.Bool}, token.LSS, i, read()),
				ir.NewAssign(p, i, ir.NewUnary(p, f32, token.SUB, read())),
				ir.NewBlock(p,
					ir.NewAssign(p, ir.NewIndex(p, f32, out, i, 1), ir.NewConvert(p, f32, read())),
					ir.NewExprStmt(p, ir.NewCall(p, f32, &ir.Func{Name: "h"}, []ir.Value{read()})),
					ir.NewExprStmt(p, ir.NewIntrinsic(p, f32, ir.OpGlobalIndex, read(), []ir.Value{read()})),
					ir.NewReturn(p, read()),
					ir.NewIf(p, ir.NewBinary(p, &ir.Type{Kind: ir.Bool}, token.LSS, read(), read()), ir.NewBlock(p), nil),
				),
			),
		),
	}
	inferAccess(k)

	if !k.Bindings[0].Read {
		t.Error("in is not marked read, so the walk misses a statement kind spec 013 will build")
	}
	if k.Bindings[0].Write {
		t.Error("in is marked written")
	}
	if !k.Bindings[1].Write {
		t.Error("out is not marked written")
	}
	if k.Bindings[1].Read {
		t.Error("out is marked read: len is not an element access")
	}
}
