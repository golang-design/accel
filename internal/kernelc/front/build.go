// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package front

import (
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"slices"

	"golang.design/x/accel/internal/kernelc/intrin"
	"golang.design/x/accel/internal/kernelc/ir"
)

// block builds a statement list. A nil result means something in it was
// rejected; the rejections are already recorded, so callers stop rather than
// report again.
func (c *checker) block(b *ast.BlockStmt) *ir.Block {
	out := ir.NewBlock(b.Pos())
	ok := true
	for _, s := range b.List {
		st := c.stmt(s)
		if st == nil {
			ok = false
			continue
		}
		out.List = append(out.List, st)
	}
	if !ok {
		return nil
	}
	return out
}

// stmt builds one statement, rejecting anything outside the closed set.
func (c *checker) stmt(s ast.Stmt) ir.Stmt {
	switch s := s.(type) {
	case *ast.BlockStmt:
		// Not `return c.block(s)`. That returns a non-nil ir.Stmt wrapping a nil
		// *ir.Block, so every caller's nil check passes and a nil block lands in
		// the tree for the emitter to dereference. Go's typed nil is the trap, and
		// a rejection that turns into a crash three slices later is worse than
		// either outcome on its own.
		if b := c.block(s); b != nil {
			return b
		}
		return nil

	case *ast.DeclStmt:
		return c.declStmt(s)

	case *ast.AssignStmt:
		return c.assign(s)

	case *ast.IfStmt:
		return c.ifStmt(s)

	case *ast.ExprStmt:
		v := c.value(s.X)
		if v == nil {
			return nil
		}
		return ir.NewExprStmt(s.Pos(), v)

	case *ast.ReturnStmt:
		return c.returnStmt(s)

	case *ast.IncDecStmt:
		return c.incDec(s)

	case *ast.ForStmt:
		return c.forStmt(s)

	case *ast.BranchStmt:
		return c.branch(s)

	// range is outside the closed node set rather than merely unscheduled.
	// `for range n` would lower to a three-clause loop mechanically and is
	// spec 013's open question, but admitting it is an amendment to spec 004's
	// node set and not a decision to take in passing.
	case *ast.RangeStmt:
		c.errorf(s.Pos(), "range is outside the closed IR node set: write a three-clause loop "+
			"(specs/004-kernel-authoring.md)")
		return nil

	// Permanently outside the subset. Each names the reason, because a reader
	// who cannot tell a wall from a schedule will argue with the wall.
	case *ast.SwitchStmt, *ast.TypeSwitchStmt, *ast.SelectStmt:
		c.errorf(s.Pos(), "%s is outside the closed IR node set: it has no structured "+
			"control-flow lowering (specs/004-kernel-authoring.md)", stmtName(s))
		return nil
	case *ast.DeferStmt:
		c.errorf(s.Pos(), "defer has no lowering on any target: there is no unwinding on a GPU")
		return nil
	case *ast.GoStmt:
		c.errorf(s.Pos(), "go has no lowering on any target: a kernel's concurrency is its "+
			"invocations, and they are launched by the dispatch")
		return nil
	case *ast.LabeledStmt:
		c.errorf(s.Pos(), "labels have no structured control-flow lowering")
		return nil
	case *ast.SendStmt:
		c.errorf(s.Pos(), "channels have no memory model on a GPU")
		return nil
	case *ast.EmptyStmt:
		return ir.NewBlock(s.Pos())
	}

	c.errorf(s.Pos(), "%s is outside the closed IR node set (specs/004-kernel-authoring.md)", stmtName(s))
	return nil
}

func stmtName(s ast.Stmt) string {
	switch s.(type) {
	case *ast.SwitchStmt, *ast.TypeSwitchStmt:
		return "switch"
	case *ast.SelectStmt:
		return "select"
	case *ast.RangeStmt:
		return "range"
	}
	return "this statement"
}

// declStmt builds `var x T = e`.
func (c *checker) declStmt(s *ast.DeclStmt) ir.Stmt {
	gen, ok := s.Decl.(*ast.GenDecl)
	if !ok || gen.Tok != token.VAR {
		c.errorf(s.Pos(), "only var declares anything inside a kernel")
		return nil
	}
	out := ir.NewBlock(s.Pos())
	for _, spec := range gen.Specs {
		vs, ok := spec.(*ast.ValueSpec)
		if !ok {
			c.errorf(spec.Pos(), "only var declares anything inside a kernel")
			return nil
		}
		if len(vs.Values) != len(vs.Names) {
			c.errorf(spec.Pos(), "a local needs exactly one initializer: an implicit zero value "+
				"hides which target's zero it is")
			return nil
		}
		for i, name := range vs.Names {
			d := c.declare(name, vs.Values[i])
			if d == nil {
				return nil
			}
			out.List = append(out.List, d)
		}
	}
	if len(out.List) == 1 {
		return out.List[0]
	}
	return out
}

// declare introduces one local with its initializer.
func (c *checker) declare(name *ast.Ident, init ast.Expr) ir.Stmt {
	v := c.value(init)
	if v == nil {
		return nil
	}
	obj := c.info.Defs[name]
	if obj == nil {
		c.errorf(name.Pos(), "%s is not a declaration", name.Name)
		return nil
	}
	if name.Name == "_" {
		c.errorf(name.Pos(), "a discarded local has no lowering: remove the statement instead")
		return nil
	}
	l := ir.NewLocal(name.Pos(), v.Type(), c.nextID, name.Name, obj)
	c.nextID++
	c.locals[obj] = l
	return ir.NewDeclare(name.Pos(), l, v)
}

// assign builds `x = e`, `x := e`, and the compound forms.
func (c *checker) assign(s *ast.AssignStmt) ir.Stmt {
	if len(s.Lhs) != 1 || len(s.Rhs) != 1 {
		c.errorf(s.Pos(), "a kernel assigns one value at a time: multiple assignment has no "+
			"lowering that preserves evaluation order on every target")
		return nil
	}

	if s.Tok == token.DEFINE {
		id, ok := s.Lhs[0].(*ast.Ident)
		if !ok {
			c.errorf(s.Pos(), ":= declares a name")
			return nil
		}
		return c.declare(id, s.Rhs[0])
	}

	rhs := c.value(s.Rhs[0])
	if rhs == nil {
		return nil
	}
	lhs := c.value(s.Lhs[0])
	if lhs == nil {
		return nil
	}
	if !assignable(lhs) {
		c.errorf(s.Pos(), "this is not something a kernel can assign to")
		return nil
	}

	if s.Tok != token.ASSIGN {
		// Compound assignment is the binary operation spelled out, which keeps
		// the IR's operator set the same size as Go's rather than doubling it.
		op, ok := compoundOp(s.Tok)
		if !ok {
			c.errorf(s.Pos(), "%s has no lowering", s.Tok)
			return nil
		}
		rhs = ir.NewBinary(s.Pos(), lhs.Type(), op, lhs, rhs)
	}
	return ir.NewAssign(s.Pos(), lhs, rhs)
}

// incDec builds `x++` and `x--` as the assignment they are.
func (c *checker) incDec(s *ast.IncDecStmt) ir.Stmt {
	lhs := c.value(s.X)
	if lhs == nil {
		return nil
	}
	if !assignable(lhs) {
		c.errorf(s.Pos(), "this is not something a kernel can assign to")
		return nil
	}
	op := token.ADD
	if s.Tok == token.DEC {
		op = token.SUB
	}
	one := ir.NewConst(s.Pos(), lhs.Type(), constant.MakeInt64(1))
	return ir.NewAssign(s.Pos(), lhs, ir.NewBinary(s.Pos(), lhs.Type(), op, lhs, one))
}

func compoundOp(t token.Token) (token.Token, bool) {
	switch t {
	case token.ADD_ASSIGN:
		return token.ADD, true
	case token.SUB_ASSIGN:
		return token.SUB, true
	case token.MUL_ASSIGN:
		return token.MUL, true
	case token.QUO_ASSIGN:
		return token.QUO, true
	case token.REM_ASSIGN:
		return token.REM, true
	case token.AND_ASSIGN:
		return token.AND, true
	case token.OR_ASSIGN:
		return token.OR, true
	case token.XOR_ASSIGN:
		return token.XOR, true
	case token.SHL_ASSIGN:
		return token.SHL, true
	case token.SHR_ASSIGN:
		return token.SHR, true
	}
	return token.ILLEGAL, false
}

func assignable(v ir.Value) bool {
	switch v.(type) {
	case *ir.Local, *ir.IndexExpr:
		return true
	}
	return false
}

// inferAccess walks a finished body and records how each binding is touched.
//
// One pass over the IR, after it is built, rather than marks dropped at
// construction. That is the same argument the IR itself rests on: an analysis
// belongs on one representation and runs once, and marking during construction
// gets the one case that matters wrong. The left side of an assignment is an
// index expression like any other, so a walk that marks every index as a read
// reports a write-only output buffer as read-write, and a caller then has to
// supply a usage the kernel does not need.
//
// A binding's length is not an element access. len(out) tells a kernel how far
// it may go and reads nothing.
// atomicBinding reports the binding an atomic operates on.
//
// The first argument, by the shape of every atomic in the table: a buffer and
// an index, because GLSL cannot form a pointer into a buffer. A shared array
// arrives here as a slice of the shared parameter and has no binding index,
// which is right -- shared memory is not a binding and carries no access mode.
func atomicBinding(v *ir.IntrinsicCall) (int, bool) {
	if !v.Op.IsAtomic() || len(v.Args) == 0 {
		return 0, false
	}
	p, ok := v.Args[0].(*ir.Param)
	if !ok || p.Type() == nil || p.Type().Kind != ir.Slice {
		return 0, false
	}
	return p.Index, true
}

func inferAccess(k *ir.Func) {
	mark := func(binding int, read, write bool) {
		for _, b := range k.Bindings {
			if b.Index == binding {
				b.Read = b.Read || read
				b.Write = b.Write || write
			}
		}
	}

	var walkValue func(v ir.Value)
	walkValue = func(v ir.Value) {
		switch v := v.(type) {
		case nil:
		case *ir.IndexExpr:
			if v.Binding >= 0 {
				mark(v.Binding, true, false)
			}
			walkValue(v.X)
			walkValue(v.Index)
		case *ir.FieldSel:
			walkValue(v.X)
		case *ir.Unary:
			walkValue(v.X)
		case *ir.Binary:
			walkValue(v.X)
			walkValue(v.Y)
		case *ir.Convert:
			walkValue(v.X)
		case *ir.Len:
			// Deliberately not a read of the elements.
		case *ir.Call:
			for _, a := range v.Args {
				walkValue(a)
			}
		case *ir.IntrinsicCall:
			// An atomic reads *and* writes the buffer it names, and its first
			// argument is that buffer rather than an index expression -- so a
			// walk that only understands indexing sees no access at all and the
			// binding looks untouched.
			//
			// That is not a cosmetic gap: access is what the graph builder
			// infers dependency edges from, so an unrecorded write is a missing
			// barrier, which is a race. See specs/003-command-graph.md.
			if b, ok := atomicBinding(v); ok {
				mark(b, true, true)
			}
			walkValue(v.Recv)
			for _, a := range v.Args {
				walkValue(a)
			}
		}
	}

	var walkStmt func(s ir.Stmt)
	walkStmt = func(s ir.Stmt) {
		switch s := s.(type) {
		case nil:
		case *ir.Block:
			for _, st := range s.List {
				walkStmt(st)
			}
		case *ir.Declare:
			walkValue(s.Init)
		case *ir.Assign:
			// The target is written. Its index expression is still evaluated, so
			// whatever computes the index is read.
			if idx, ok := s.LHS.(*ir.IndexExpr); ok {
				if idx.Binding >= 0 {
					mark(idx.Binding, false, true)
				}
				walkValue(idx.Index)
			}
			walkValue(s.RHS)
		case *ir.ExprStmt:
			walkValue(s.X)
		case *ir.If:
			walkValue(s.Cond)
			walkStmt(s.Then)
			walkStmt(s.Else)
		case *ir.For:
			walkStmt(s.Init)
			walkValue(s.Cond)
			walkStmt(s.Post)
			walkStmt(s.Body)
		case *ir.Return:
			walkValue(s.Value)
		}
	}
	walkStmt(k.Body)
}

// returnStmt builds a return, which a compute kernel and a helper mean
// differently.
//
// A compute kernel returns nothing: it writes through its bindings, and a value
// would have nowhere to go. A helper returns one value or none, and the two have
// to agree with the signature go/types already checked, so the only thing left
// to reject here is a compute kernel that tries.
//
// The graphics stages of specs/032-stage-abi.md are the exception and are
// excluded here rather than in a second place: a vertex stage returns a position
// and its varyings, and a fragment stage returns its attachment struct, because
// those values go somewhere fixed-function rather than into a binding.
func (c *checker) returnStmt(s *ast.ReturnStmt) ir.Stmt {
	if c.current != nil && c.current.Stage == ir.StageCompute && len(s.Results) > 0 {
		c.errorf(s.Pos(), "a kernel returns nothing: it writes through its bindings")
		return nil
	}
	if len(s.Results) > 1 {
		c.errorf(s.Pos(), "a helper returns one value or none")
		return nil
	}
	if len(s.Results) == 0 {
		return ir.NewReturn(s.Pos(), nil)
	}
	v := c.value(s.Results[0])
	if v == nil {
		return nil
	}
	return ir.NewReturn(s.Pos(), v)
}

// forStmt builds a loop in any of Go's three forms.
//
// All three, because they are one node with optional parts rather than three
// constructs: an omitted condition is an infinite loop and an omitted init and
// post is the while form. Every target spells that the same way, and SPIR-V
// wants the structure declared rather than recovered.
func (c *checker) forStmt(s *ast.ForStmt) ir.Stmt {
	var init, post ir.Stmt
	if s.Init != nil {
		init = c.simple(s.Init, "a loop's init")
		if init == nil {
			return nil
		}
	}
	if s.Post != nil {
		post = c.simple(s.Post, "a loop's post")
		if post == nil {
			return nil
		}
	}

	var cond ir.Value
	if s.Cond != nil {
		cond = c.value(s.Cond)
		if cond == nil {
			return nil
		}
		if cond.Type().Kind != ir.Bool {
			c.errorf(s.Cond.Pos(), "a loop condition is a bool, and this is %v", cond.Type())
			return nil
		}
	}

	c.loops++
	body := c.block(s.Body)
	c.loops--
	if body == nil {
		return nil
	}
	return ir.NewFor(s.Pos(), init, cond, post, body)
}

// simple builds the statement forms a loop clause admits.
//
// A loop clause takes a simple statement, and the subset narrows that further:
// no short variable declaration of several names, no send, no bare expression.
// Naming what is allowed rather than filtering what is not keeps this in step
// with the closed node set.
func (c *checker) simple(s ast.Stmt, where string) ir.Stmt {
	switch s := s.(type) {
	case *ast.AssignStmt:
		return c.assign(s)
	case *ast.IncDecStmt:
		return c.incDec(s)
	}
	c.errorf(s.Pos(), "%s takes an assignment or an increment, and this is not one", where)
	return nil
}

// branch builds break and continue.
//
// Neither carries a label, because the subset admits none: a labelled branch
// has no structured control-flow lowering, which is the same reason goto is
// permanently out. Outside a loop they have nothing to bind to, and Go's own
// checker permits `break` inside a switch, which the subset does not have.
func (c *checker) branch(s *ast.BranchStmt) ir.Stmt {
	if s.Label != nil {
		c.errorf(s.Pos(), "a labelled %s has no structured control-flow lowering", s.Tok)
		return nil
	}
	switch s.Tok {
	case token.BREAK:
		if c.loops == 0 {
			c.errorf(s.Pos(), "break is outside a loop")
			return nil
		}
		return ir.NewBreak(s.Pos())
	case token.CONTINUE:
		if c.loops == 0 {
			c.errorf(s.Pos(), "continue is outside a loop")
			return nil
		}
		return ir.NewContinue(s.Pos())
	}
	c.errorf(s.Pos(), "%s has no lowering: the subset has no labels and no goto", s.Tok)
	return nil
}

// ifStmt builds a conditional.
func (c *checker) ifStmt(s *ast.IfStmt) ir.Stmt {
	if s.Init != nil {
		c.errorf(s.Init.Pos(), "an if with an init statement is out of scope for now: declare "+
			"the value on its own line")
		return nil
	}
	cond := c.value(s.Cond)
	if cond == nil {
		return nil
	}
	if cond.Type().Kind != ir.Bool {
		c.errorf(s.Cond.Pos(), "an if condition is a bool, and this is %v", cond.Type())
		return nil
	}
	then := c.block(s.Body)
	if then == nil {
		return nil
	}
	var els ir.Stmt
	if s.Else != nil {
		els = c.stmt(s.Else)
		if els == nil {
			return nil
		}
	}
	return ir.NewIf(s.Pos(), cond, then, els)
}

// value builds an expression.
func (c *checker) value(e ast.Expr) ir.Value {
	// A constant go/types already folded arrives with its resolved type, which
	// is where the GLSL integer-literal divergence is settled: the emitter knows
	// whether the 2 in gid*2 is u32 or f32 and spells it accordingly.
	if tv, ok := c.info.Types[e]; ok && tv.Value != nil {
		t, err := c.constType(tv.Type)
		if err != nil {
			c.errorf(e.Pos(), "%s", err)
			return nil
		}
		return ir.NewConst(e.Pos(), t, tv.Value)
	}

	switch e := e.(type) {
	case *ast.Ident:
		return c.ident(e)
	case *ast.ParenExpr:
		return c.value(e.X)
	case *ast.SelectorExpr:
		return c.selector(e)
	case *ast.IndexExpr:
		return c.index(e)
	case *ast.BinaryExpr:
		return c.binary(e)
	case *ast.UnaryExpr:
		return c.unary(e)
	case *ast.CallExpr:
		return c.call(e)
	case *ast.FuncLit:
		c.errorf(e.Pos(), "closures have no spelling on any target")
		return nil
	case *ast.CompositeLit:
		c.errorf(e.Pos(), "composite literals are outside the closed IR node set")
		return nil
	case *ast.SliceExpr:
		c.errorf(e.Pos(), "reslicing is outside the closed IR node set: a binding's extent is "+
			"fixed by its descriptor")
		return nil
	case *ast.StarExpr:
		c.errorf(e.Pos(), "pointer indirection is outside the subset")
		return nil
	case *ast.TypeAssertExpr:
		c.errorf(e.Pos(), "interfaces have no memory model on a GPU")
		return nil
	}
	c.errorf(e.Pos(), "this expression is outside the closed IR node set "+
		"(specs/004-kernel-authoring.md)")
	return nil
}

// ident resolves a name to a parameter or a local.
func (c *checker) ident(e *ast.Ident) ir.Value {
	obj := c.info.Uses[e]
	if obj == nil {
		obj = c.info.Defs[e]
	}
	if e.Name == "_" {
		c.errorf(e.Pos(), "a discarded value has no lowering: remove the statement instead")
		return nil
	}
	if obj == nil {
		c.errorf(e.Pos(), "%s does not resolve", e.Name)
		return nil
	}
	if l, ok := c.locals[obj]; ok {
		return l
	}
	if c.current != nil {
		for _, p := range c.current.Params {
			if p.Obj == obj {
				return p
			}
		}
	}
	c.errorf(e.Pos(), "%s is neither a parameter nor a local declared in this kernel: a kernel "+
		"reads only what its signature binds", e.Name)
	return nil
}

// selector builds a field selection, which at this milestone is an ID3
// component or an intrinsic's receiver.
func (c *checker) selector(e *ast.SelectorExpr) ir.Value {
	// A method value used as anything but a call has no lowering. A call reaches
	// this only through the call path, which handles the receiver itself.
	if sel, ok := c.info.Selections[e]; ok && sel.Kind() == types.MethodVal {
		c.errorf(e.Pos(), "a method value has no lowering: call %s instead", e.Sel.Name)
		return nil
	}

	// A package-qualified name that is not a call has no lowering either, and
	// saying so beats walking a package identifier as though it were a value.
	if id, ok := e.X.(*ast.Ident); ok {
		if _, isPkg := c.info.Uses[id].(*types.PkgName); isPkg {
			c.errorf(e.Pos(), "%s.%s is not something a kernel reads: a kernel calls intrinsics "+
				"and helpers", id.Name, e.Sel.Name)
			return nil
		}
	}

	x := c.value(e.X)
	if x == nil {
		return nil
	}

	switch x.Type().Kind {
	case ir.ID3Kind:
		idx := slices.Index([]string{"X", "Y", "Z"}, e.Sel.Name)
		if idx < 0 {
			c.errorf(e.Pos(), "an id has components X, Y, and Z, not %s", e.Sel.Name)
			return nil
		}
		return ir.NewFieldSel(e.Pos(), &ir.Type{Kind: ir.U32}, x, idx, e.Sel.Name)

	case ir.Struct:
		// A uniform's member. The offset is the layout's, not Go's, and it is
		// resolved at emission; here the field only needs a type and an index.
		ft, err := c.irType(c.info.TypeOf(e))
		if err != nil {
			c.errorf(e.Pos(), "%s.%s: %s", x.Type(), e.Sel.Name, err)
			return nil
		}
		idx := c.uniformFieldIndex(x.Type().Name, e.Sel.Name)
		if idx < 0 {
			c.errorf(e.Pos(), "%s has no field %s that a uniform block can hold",
				x.Type(), e.Sel.Name)
			return nil
		}
		if p, ok := x.(*ir.Param); ok && c.current != nil {
			for _, u := range c.current.Uniforms {
				if u.Index == p.Index {
					u.Reads = true
				}
			}
		}
		return ir.NewFieldSel(e.Pos(), ft, x, idx, e.Sel.Name)
	}

	c.errorf(e.Pos(), "%v has no field %s in the subset", x.Type(), e.Sel.Name)
	return nil
}

// uniformFieldIndex is a member's position in its block's layout.
func (c *checker) uniformFieldIndex(typeName, field string) int {
	l, ok := c.layouts[typeName]
	if !ok {
		return -1
	}
	for i, f := range l.Fields {
		if f.Name == field {
			return i
		}
	}
	return -1
}

// index builds a load from a binding, recording the read.
func (c *checker) index(e *ast.IndexExpr) ir.Value {
	x := c.value(e.X)
	if x == nil {
		return nil
	}
	if x.Type().Kind != ir.Slice && x.Type().Kind != ir.Array {
		c.errorf(e.Pos(), "%v is not something a kernel indexes", x.Type())
		return nil
	}
	i := c.value(e.Index)
	if i == nil {
		return nil
	}

	binding := -1
	if p, ok := x.(*ir.Param); ok && x.Type().Kind == ir.Slice {
		binding = p.Index
	}
	return ir.NewIndex(e.Pos(), x.Type().Elem, x, i, binding)
}

// binary builds an arithmetic, comparison, or logical operation.
func (c *checker) binary(e *ast.BinaryExpr) ir.Value {
	x := c.value(e.X)
	if x == nil {
		return nil
	}
	y := c.value(e.Y)
	if y == nil {
		return nil
	}
	t, err := c.irType(c.info.TypeOf(e))
	if err != nil {
		c.errorf(e.Pos(), "%s", err)
		return nil
	}
	if t.Kind != ir.Bool && !t.Kind.Numeric() {
		c.errorf(e.Pos(), "arithmetic on %v is outside the subset: narrow dtypes are storage "+
			"and convert to f32 on load", t)
		return nil
	}
	return ir.NewBinary(e.Pos(), t, e.Op, x, y)
}

// unary builds negation and complement.
func (c *checker) unary(e *ast.UnaryExpr) ir.Value {
	switch e.Op {
	case token.SUB, token.ADD, token.XOR, token.NOT:
	default:
		c.errorf(e.Pos(), "unary %s is outside the subset", e.Op)
		return nil
	}
	x := c.value(e.X)
	if x == nil {
		return nil
	}
	t, err := c.irType(c.info.TypeOf(e))
	if err != nil {
		c.errorf(e.Pos(), "%s", err)
		return nil
	}
	return ir.NewUnary(e.Pos(), t, e.Op, x)
}

// call builds a conversion, len, or an intrinsic call. Telling a conversion
// from a call is the second thing go/types buys: float32(x) and Sqrt(x) are the
// same AST shape, and the predecessor resolved that by putting float32 in its
// builtins map next to sqrt.
func (c *checker) call(e *ast.CallExpr) ir.Value {
	if tv, ok := c.info.Types[e.Fun]; ok && tv.IsType() {
		return c.conversion(e, tv.Type)
	}

	if id, ok := e.Fun.(*ast.Ident); ok {
		if b, ok := c.info.Uses[id].(*types.Builtin); ok {
			return c.builtin(e, b)
		}
	}

	fn := c.calleeFunc(e.Fun)
	if fn == nil {
		c.errorf(e.Pos(), "this call is outside the subset: a kernel calls intrinsics and "+
			"helpers, and helpers arrive with spec 013")
		return nil
	}

	in, ok := intrin.Lookup(fn)
	if !ok {
		return c.helperCall(e, fn)
	}
	// A capability the body implies. Accumulated here rather than declared,
	// because this is where the compiler learns what the kernel actually uses.
	if c.current != nil {
		c.current.Caps |= uint32(in.Cap)
	}
	// A barrier or a subgroup rendezvous makes the whole kernel cooperative,
	// which selects the resumable lowering. Both need something the flat path
	// cannot give: a barrier needs every invocation to arrive, and a subgroup
	// operation needs every lane's value at the point of the call.
	//
	// Derived from the body rather than declared, because a declaration can be
	// forgotten and the failure would be a kernel silently lowered the wrong
	// way -- which for a subgroup reduction means every lane receiving its own
	// value instead of the total, a plausible number rather than an error.
	if c.current != nil && (in.Stage == intrin.Cooperative || in.Op.IsSubgroupRendezvous()) {
		c.current.Cooperative = true
	}
	// The caller decides whether there is a receiver, because it is the only
	// place that has both the selector and the type information to tell a method
	// call from a package-qualified one. They are the same AST shape.
	var recv ast.Expr
	if sel, ok := e.Fun.(*ast.SelectorExpr); ok && isMethodCall(c.info, sel) {
		recv = sel.X
	}
	return c.intrinsicCall(e, recv, in)
}

// helperCall builds a call to a //accel:helper in the same package.
//
// Same package, because a helper is compiled from source and the compiler loads
// one package. A call to anything else has no body to lower, and the message
// says which of the two it is: an undirected function in this package is a
// missing directive, and one from elsewhere is not compilable at all.
func (c *checker) helperCall(e *ast.CallExpr, fn *types.Func) ir.Value {
	obj := types.Object(fn)
	h, ok := c.helpers[obj]
	if !ok {
		if fn.Pkg() != nil && c.pkg.Types != nil && fn.Pkg().Path() == c.pkg.Types.Path() {
			c.errorf(e.Pos(), "%s is in this package but is not marked %s, so it has no "+
				"lowering: a kernel calls intrinsics and helpers", fn.Name(), HelperDirective)
			return nil
		}
		c.errorf(e.Pos(), "%s is not an intrinsic and not a helper in this package: a kernel "+
			"is compiled from one package's source, so there is no body to lower", fn.Name())
		return nil
	}

	if len(e.Args) != len(h.Params) {
		c.errorf(e.Pos(), "%s takes %d arguments and got %d", h.Name, len(h.Params), len(e.Args))
		return nil
	}
	args := make([]ir.Value, 0, len(e.Args))
	for _, a := range e.Args {
		v := c.value(a)
		if v == nil {
			return nil
		}
		args = append(args, v)
	}

	if c.current != nil {
		c.calls[c.current] = append(c.calls[c.current], h)
		if !slices.Contains(c.current.Helpers, h) {
			c.current.Helpers = append(c.current.Helpers, h)
		}
		// A helper's own accesses become the caller's, mapped through the
		// argument list. The access is a property of the call site rather than
		// of the helper, which is why it is merged here and not recorded once.
		c.propagateAccess(h, args)
	}

	result := h.Result
	if result == nil {
		result = &ir.Type{Kind: ir.Invalid}
	}
	return ir.NewCall(e.Pos(), result, h, args)
}

// propagateAccess maps a helper's parameter accesses onto the caller's bindings.
func (c *checker) propagateAccess(h *ir.Func, args []ir.Value) {
	for _, hb := range h.Bindings {
		if hb.Index >= len(args) {
			continue
		}
		p, ok := args[hb.Index].(*ir.Param)
		if !ok {
			continue
		}
		for _, cb := range c.current.Bindings {
			if cb.Index == p.Index {
				cb.Read = cb.Read || hb.Read
				cb.Write = cb.Write || hb.Write
			}
		}
	}
}

// isMethodCall reports whether a selector names a method on a value rather than
// a function in a package.
func isMethodCall(info *types.Info, sel *ast.SelectorExpr) bool {
	s, ok := info.Selections[sel]
	return ok && s.Kind() == types.MethodVal
}

// calleeFunc resolves a call target to the function object go/types found.
func (c *checker) calleeFunc(fun ast.Expr) *types.Func {
	switch f := fun.(type) {
	case *ast.Ident:
		fn, _ := c.info.Uses[f].(*types.Func)
		return fn
	case *ast.SelectorExpr:
		if sel, ok := c.info.Selections[f]; ok {
			fn, _ := sel.Obj().(*types.Func)
			return fn
		}
		fn, _ := c.info.Uses[f.Sel].(*types.Func)
		return fn
	case *ast.IndexExpr, *ast.IndexListExpr:
		// An instantiated generic function or method. Go 1.27 admits generic
		// methods, so this is reachable and is ours to reject.
		return nil
	}
	return nil
}

// intrinsicCall builds a resolved intrinsic.
func (c *checker) intrinsicCall(e *ast.CallExpr, recvExpr ast.Expr, in *intrin.Intrinsic) ir.Value {
	var recv ir.Value
	if recvExpr != nil {
		recv = c.value(recvExpr)
		if recv == nil {
			return nil
		}
	}
	if len(e.Args) != in.Params {
		c.errorf(e.Pos(), "%s takes %d arguments and got %d", in.Authored, in.Params, len(e.Args))
		return nil
	}
	args := make([]ir.Value, 0, len(e.Args))
	for _, a := range e.Args {
		v := c.value(a)
		if v == nil {
			return nil
		}
		args = append(args, v)
	}

	// The digest records the authored spelling, in first-use order, so that
	// editing the table makes a generated file stale.
	if c.current != nil && !slices.Contains(c.current.Intrinsics, in.Authored) {
		c.current.Intrinsics = append(c.current.Intrinsics, in.Authored)
	}
	return ir.NewIntrinsic(e.Pos(), &ir.Type{Kind: in.Result}, in.Op, recv, args)
}

// conversion builds an explicit conversion.
func (c *checker) conversion(e *ast.CallExpr, to types.Type) ir.Value {
	if len(e.Args) != 1 {
		c.errorf(e.Pos(), "a conversion takes one value")
		return nil
	}
	t, err := c.irType(to)
	if err != nil {
		c.errorf(e.Pos(), "%s", err)
		return nil
	}
	x := c.value(e.Args[0])
	if x == nil {
		return nil
	}
	return ir.NewConvert(e.Pos(), t, x)
}

// builtin admits len and rejects the rest by name.
func (c *checker) builtin(e *ast.CallExpr, b *types.Builtin) ir.Value {
	if b.Name() != "len" {
		c.errorf(e.Pos(), "the builtin %s has no lowering: a kernel has no allocator and no "+
			"runtime", b.Name())
		return nil
	}
	if len(e.Args) != 1 {
		c.errorf(e.Pos(), "len takes one value")
		return nil
	}
	x := c.value(e.Args[0])
	if x == nil {
		return nil
	}
	if x.Type().Kind != ir.Slice && x.Type().Kind != ir.Array {
		c.errorf(e.Pos(), "len of %v is outside the subset", x.Type())
		return nil
	}
	return ir.NewLen(e.Pos(), &ir.Type{Kind: ir.I32}, x)
}

// irType maps a go/types type onto an IR type.
func (c *checker) irType(t types.Type) (*ir.Type, error) {
	if t == nil {
		return nil, errType{"this expression has no type"}
	}
	if isID3(t) {
		return &ir.Type{Kind: ir.ID3Kind}, nil
	}
	if isThread(t) {
		return &ir.Type{Kind: ir.Struct, Name: "Thread"}, nil
	}
	if isKernelType(t, "Float16") {
		return &ir.Type{Kind: ir.F16}, nil
	}
	if isKernelType(t, "BFloat16") {
		return &ir.Type{Kind: ir.BF16}, nil
	}
	// A comparison has type untyped bool until something gives it one, so the
	// untyped form is defaulted rather than rejected. Defaulting is also what
	// keeps an untyped integer honest: it defaults to int, which is
	// platform-width and therefore has no single device layout, and the
	// rejection below says so.
	switch u := types.Default(types.Unalias(t)).Underlying().(type) {
	case *types.Basic:
		if u.Info()&types.IsUntyped != 0 {
			return nil, errType{"this value has no resolved type"}
		}
		if u.Kind() == types.Int || u.Kind() == types.Uint || u.Kind() == types.Uintptr {
			return nil, errType{"int, uint, and uintptr are platform-width, so a value of that " +
				"type has no single device layout: use int32 or uint32"}
		}
		k, err := scalarKind(u)
		if err != nil {
			return nil, errType{err.Error()}
		}
		return &ir.Type{Kind: k}, nil
	case *types.Slice:
		k, err := elementKind(u.Elem())
		if err != nil {
			return nil, errType{err.Error()}
		}
		return &ir.Type{Kind: ir.Slice, Elem: &ir.Type{Kind: k}}, nil
	case *types.Array:
		// Recursive, because a uniform's matrix member is an array of arrays and
		// a kernel indexes it twice. A flat rule would have admitted the outer
		// array and rejected the element it is made of, which reads as the
		// element type being wrong rather than the nesting.
		elem, err := c.irType(u.Elem())
		if err != nil {
			return nil, err
		}
		return &ir.Type{Kind: ir.Array, Len: int(u.Len()), Elem: elem}, nil
	}
	return nil, errType{"the type " + t.String() + " is outside the subset"}
}

// constType maps a constant's type, which is not quite the rule for a value.
//
// A constant of type int is fine and a variable of type int is not. The
// platform-width objection is that a value's device layout would depend on
// GOARCH, and a constant has no layout: its value is known at generate time and
// the emitter spells it in whatever type the target needs. Rejecting it would
// mean writing int32(0) to index a buffer, which is noise that teaches nobody
// anything.
func (c *checker) constType(t types.Type) (*ir.Type, error) {
	if b, ok := types.Default(types.Unalias(t)).Underlying().(*types.Basic); ok {
		switch b.Kind() {
		case types.Int, types.Uint, types.Uintptr, types.Int64, types.Uint64, types.Int16, types.Uint16:
			return &ir.Type{Kind: ir.I32}, nil
		case types.Float64:
			return &ir.Type{Kind: ir.F32}, nil
		}
	}
	return c.irType(t)
}

type errType struct{ msg string }

func (e errType) Error() string { return e.msg }
