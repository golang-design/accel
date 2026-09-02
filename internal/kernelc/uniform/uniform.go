// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package uniform decides which values are equal across a workgroup.
//
// # Why this exists before the transform that needs it
//
// specs/002-compute-model.md section 3.1 requires every barrier to sit in
// control flow uniform at its scope. That rule is what makes the resumable
// lowering a transform over a *sequence* of suspension points rather than over
// an arbitrary graph: a sequence needs a program counter, and a graph needs a
// relooper. So this analysis is not a diagnostic beside the transform, it is
// the transform's precondition, and specs/018-cooperative-lowering.md places it
// here for that reason.
//
// # When it cannot decide, it rejects
//
// This is a choice and the alternative is worse. An admitted invalid barrier is
// undefined behaviour that passes every test this project can run without
// buying hardware, then hangs on a user's phone. A rejected valid barrier is a
// compile error with a position, and the fix -- hoist the barrier out of the
// region the compiler could not prove uniform -- is always available and always
// correct.
//
// The cost is real and is stated rather than hidden: some correct kernels are
// rejected. See specs/002-compute-model.md section 3.3 for the two known
// families.
package uniform

import (
	"golang.design/x/accel/internal/kernelc/intrin"
	"golang.design/x/accel/internal/kernelc/ir"
)

// Level is how widely a value is known to be equal.
//
// The three form a lattice ordered Workgroup < Subgroup < Non: a
// workgroup-uniform value is also subgroup-uniform, and the least-uniform
// operand decides a result. The order of the constants is that lattice, so
// max() is the join.
type Level int

const (
	// Workgroup means equal across every invocation of the workgroup.
	Workgroup Level = iota

	// Subgroup means equal within each subgroup and possibly differing between
	// subgroups.
	Subgroup

	// Non means nothing is known.
	Non
)

func (l Level) String() string {
	switch l {
	case Workgroup:
		return "workgroup-uniform"
	case Subgroup:
		return "subgroup-uniform"
	}
	return "non-uniform"
}

// Info is the analysis result for one function.
type Info struct {
	// fn is the kernel under analysis, for the binding declarations its
	// parameters carry.
	fn *ir.Func

	// values is every value's level. Keyed by the node itself, which is stable
	// because the IR is a tree of pointers built once.
	values map[ir.Value]Level

	// control is the level of the control flow reaching each statement: the
	// least-uniform predicate any enclosing conditional or loop depends on.
	control map[ir.Stmt]Level

	// escapes is, per loop, the least uniform control any break in its body
	// sits under. A break some invocations take and others do not gives the
	// loop a different trip count for each of them, so every definition in the
	// body is control-dependent on that break's predicate -- the ones before it
	// in the body included, since they run once more for the invocations that
	// stayed. It is found on one pass and applied on the next, which is what
	// the fixed point is for.
	escapes map[*ir.For]Level

	// summaries is, per helper, how uniform its result is when every argument
	// is workgroup-uniform. A call's level is the join of this and its
	// arguments: the summary carries what the body reads on its own -- an id, a
	// load -- and the arguments carry what the call site supplies.
	summaries map[*ir.Func]Level
}

// Of computes uniformity for a function.
//
// One forward pass suffices for the value lattice because the IR is structured
// and a variable's definitions all precede its uses in the tree. Loops are the
// exception, and they are handled by seeding a loop-carried local at its
// declared level and re-running the body until nothing changes -- at most twice
// in practice, because the lattice has three levels and a value only ever moves
// toward Non.
func Of(fn *ir.Func) *Info {
	in := &Info{
		fn:        fn,
		values:    map[ir.Value]Level{},
		control:   map[ir.Stmt]Level{},
		escapes:   map[*ir.For]Level{},
		summaries: map[*ir.Func]Level{},
	}
	// Iterate to a fixed point. Every recorded level only ever rises and the
	// lattice is finite, so a pass that changes nothing is the last one and
	// the loop needs no bound: a bound that was too small would be a wrong
	// answer rather than a slow one.
	for {
		if changed, _ := in.walkBlock(fn.Body, Workgroup, nil); !changed {
			break
		}
	}
	return in
}

// Level reports a value's uniformity, defaulting to Non for anything the walk
// did not reach. Defaulting to Non rather than Workgroup is the conservative
// direction: an unreached value is one this analysis does not understand.
func (in *Info) Level(v ir.Value) Level {
	if l, ok := in.values[v]; ok {
		return l
	}
	return Non
}

// Control reports the uniformity of the control flow reaching a statement.
func (in *Info) Control(s ir.Stmt) Level {
	if l, ok := in.control[s]; ok {
		return l
	}
	return Non
}

func (in *Info) set(v ir.Value, l Level) bool {
	if old, ok := in.values[v]; ok && old >= l {
		return false
	}
	in.values[v] = l
	return true
}

// walkBlock walks a block under a control level inside a loop, or none.
//
// It returns whether anything became less uniform, and the least uniform
// control any break or continue in the block sits under. A statement after
// such an escape runs only for the invocations that did not take it, so it is
// control-dependent on the escape's predicate exactly as if it sat inside the
// conditional -- which is what stops
//
//	for i := 0; i < 8; i++ { if lane > i { break }; last = i }
//
// from calling last uniform because i is.
func (in *Info) walkBlock(b *ir.Block, ctrl Level, loop *ir.For) (changed bool, esc Level) {
	if b == nil {
		return false, Workgroup
	}
	esc = Workgroup
	for _, s := range b.List {
		c, e := in.walkStmt(s, max(ctrl, esc), loop)
		changed = c || changed
		esc = max(esc, e)
	}
	return changed, esc
}

func (in *Info) walkStmt(s ir.Stmt, ctrl Level, loop *ir.For) (changed bool, esc Level) {
	if old, ok := in.control[s]; !ok || ctrl > old {
		in.control[s] = ctrl
		changed = true
	}
	esc = Workgroup

	switch n := s.(type) {
	case *ir.Block:
		c, e := in.walkBlock(n, ctrl, loop)
		changed = c || changed
		esc = e

	case *ir.Declare:
		l := in.value(n.Init)
		// A definition inherits the control flow it sits under. This is the
		// clause that stops `if l < 4 { x = 1 } else { x = 2 }` from being
		// called uniform because both 1 and 2 are literals.
		changed = in.set(n.Local, max(l, ctrl)) || changed

	case *ir.Assign:
		l := max(in.value(n.RHS), ctrl)
		if lhs, ok := n.LHS.(*ir.Local); ok {
			changed = in.set(lhs, l) || changed
		} else {
			// A store into a binding or shared memory. The location's contents
			// become non-uniform, which the load rule already assumes, so
			// nothing is recorded; the index and value still need levels.
			in.value(n.LHS)
		}

	case *ir.ExprStmt:
		in.value(n.X)

	case *ir.If:
		cond := max(in.value(n.Cond), ctrl)
		c, e := in.walkBlock(n.Then, cond, loop)
		changed = c || changed
		esc = e
		if n.Else != nil {
			c, e := in.walkStmt(n.Else, cond, loop)
			changed = c || changed
			esc = max(esc, e)
		}

	case *ir.For:
		// A loop's body is control-dependent on its condition, on anything its
		// init or post computes, and on any break some invocations take: those
		// invocations run fewer iterations than the rest, so every definition
		// in the body -- before the break as much as after it -- is one they
		// may or may not have executed a last time.
		inner := ctrl
		if n.Init != nil {
			c, _ := in.walkStmt(n.Init, ctrl, loop)
			changed = c || changed
		}
		if n.Cond != nil {
			inner = max(inner, in.value(n.Cond))
		}
		inner = max(inner, in.escapes[n])
		c, _ := in.walkBlock(n.Body, inner, n)
		changed = c || changed
		if n.Post != nil {
			c, _ := in.walkStmt(n.Post, inner, n)
			changed = c || changed
		}
		// The loop absorbs its own escapes: after it, every invocation is out
		// of it, whichever way it left.

	case *ir.Return:
		if n.Value != nil {
			in.value(n.Value)
		}
		// A return is an escape from the function rather than from a loop,
		// and it is [AcceptBarriers] that reads its control level: the
		// invocations that took it never arrive at a barrier after it. The
		// values of the ones that stayed are unaffected, so it raises nothing
		// here.

	case *ir.Break:
		esc = ctrl
		if loop != nil && ctrl > in.escapes[loop] {
			in.escapes[loop] = ctrl
			changed = true
		}

	case *ir.Continue:
		// What follows in the iteration is skipped by the invocations that took
		// it; the trip count is unchanged, so only what follows is raised.
		esc = ctrl
	}
	return changed, esc
}

// LoopEscape reports the least uniform control any break in a loop's body
// sits under, or Workgroup when every invocation leaves the loop together.
func (in *Info) LoopEscape(loop *ir.For) Level {
	return in.escapes[loop]
}

// value computes and records a value's level.
func (in *Info) value(v ir.Value) Level {
	if v == nil {
		return Workgroup
	}
	l := in.compute(v)
	in.set(v, l)
	return in.Level(v)
}

func (in *Info) compute(v ir.Value) Level {
	switch n := v.(type) {
	case *ir.Const:
		return Workgroup

	case *ir.Param:
		// A parameter is workgroup-uniform, bindings included: every invocation
		// receives the same slice, and a by-value uniform struct is one value
		// for the whole dispatch.
		//
		// The binding rather than its contents. What is non-uniform is what a
		// load finds there, which is the IndexExpr rule below — stated where
		// spec 002 section 3.3 states it, so that the reason a load is
		// non-uniform is "another invocation may have written it" rather than
		// an accident of how the handle is levelled.
		return Workgroup

	case *ir.Local:
		// Whatever the definitions gave it, or Non if none has been seen yet.
		// Non is the safe default under iteration to a fixed point: a local
		// read before its declaration is walked settles on the next pass.
		return in.Level(n)

	case *ir.FieldSel:
		// A field of the uniform-struct parameter is uniform; a field of
		// anything else inherits it.
		return in.value(n.X)

	case *ir.IndexExpr:
		// A load is non-uniform even when the index is uniform, because another
		// invocation may have written the location. Conservative and known to
		// be so: specs/002-compute-model.md section 3.3 names the two kernel
		// families this rejects.
		//
		// The exception is a binding the kernel declared with //accel:uniform:
		// the author's promise that no invocation of the dispatch writes it,
		// which the graph enforces at record and bind time. A load from one is
		// as uniform as its index. specs/063-uniform-loads.md.
		in.value(n.X)
		idx := in.value(n.Index)
		if p, ok := n.X.(*ir.Param); ok && in.uniformBinding(p) {
			return idx
		}
		return Non

	case *ir.Len:
		// A binding's length is fixed for the dispatch.
		in.value(n.X)
		return Workgroup

	case *ir.Unary:
		return in.value(n.X)

	case *ir.Binary:
		return max(in.value(n.X), in.value(n.Y))

	case *ir.Convert:
		return in.value(n.X)

	case *ir.Call:
		// A helper's result is the join of its least uniform argument and what
		// its body reads on its own. The arguments alone are not enough, and
		// this was wrong before it was written down: a helper returning
		// t.LocalIndex() or in[0] takes no non-uniform argument, so the join
		// over its arguments called its result workgroup-uniform, and a
		// barrier under `if lane(t) > 2` was admitted while the same predicate
		// spelled inline was refused.
		l := in.summary(n.Callee)
		for _, a := range n.Args {
			l = max(l, in.value(a))
		}
		return l

	case *ir.IntrinsicCall:
		return in.intrinsic(n)
	}
	return Non
}

// intrinsic reads the seed table of specs/002-compute-model.md section 3.3,
// which the intrinsic table states per entry.
//
// Read from the table rather than kept here, because this function used to
// keep its own switch and the two disagreed: SubgroupIndex was per-invocation
// in the table and subgroup-uniform here, and nothing read the table's column
// at all. One statement of the seed, and a test over the table that every
// entry makes one.
//
// The level of a result that is OfOperands is the join over the arguments and
// the receiver. The receiver matters and was **wrong** once: a mask method
// such as `m.Count()` takes no arguments, so a join over the arguments alone
// is Workgroup, and a barrier under `if m.Count() > 1` was accepted while
// different subgroups can take different branches of it. specs/058-ballot.md.
func (in *Info) intrinsic(n *ir.IntrinsicCall) Level {
	operands := Workgroup
	if n.Recv != nil {
		operands = max(operands, in.value(n.Recv))
	}
	for _, a := range n.Args {
		operands = max(operands, in.value(a))
	}

	entry, ok := intrin.ByOp(n.Op)
	if !ok {
		// An opcode no intrinsic lowers to is one this analysis does not
		// understand, and Non is the direction that refuses rather than admits.
		return Non
	}
	switch entry.Uniformity {
	case intrin.PerWorkgroup:
		return Workgroup
	case intrin.PerSubgroup:
		return Subgroup
	case intrin.OfOperands:
		return operands
	}
	return Non
}

// summary is how uniform a helper's result is when every argument is
// workgroup-uniform: the level its body reaches on its own.
//
// The body is analysed with its parameters seeded at Workgroup, and the result
// is the join over every return of the returned value and the control it sits
// under -- a literal returned under `if t.LocalIndex() > 2` is as non-uniform
// as the id, because which literal an invocation gets depends on it. What the
// arguments contribute is joined in at the call site, so the summary is
// computed once per callee rather than once per call.
//
// A helper reached through another helper is summarised by the callee's own
// analysis, which is what makes the level two helpers deep the same as the
// level one deep. The front end refuses recursion, and a cycle would still
// terminate here: a callee already being summarised answers Non.
func (in *Info) summary(fn *ir.Func) Level {
	if fn == nil || fn.Result == nil {
		return Workgroup
	}
	if l, ok := in.summaries[fn]; ok {
		return l
	}
	in.summaries[fn] = Non
	body := Of(fn)
	l := Workgroup
	var walk func(s ir.Stmt)
	walk = func(s ir.Stmt) {
		switch n := s.(type) {
		case *ir.Block:
			for _, st := range n.List {
				walk(st)
			}
		case *ir.If:
			walk(n.Then)
			if n.Else != nil {
				walk(n.Else)
			}
		case *ir.For:
			walk(n.Body)
		case *ir.Return:
			l = max(l, body.Control(n))
			if n.Value != nil {
				l = max(l, body.Level(n.Value))
			}
		}
	}
	walk(fn.Body)
	in.summaries[fn] = l
	return l
}

// uniformBinding reports whether p is one of the kernel's bindings declared
// with //accel:uniform. A helper's parameter is never one: the declaration is
// the kernel's, about its dispatch, and a helper reached through a call is
// analysed on its own parameters.
func (in *Info) uniformBinding(p *ir.Param) bool {
	if in.fn == nil {
		return false
	}
	for _, b := range in.fn.Bindings {
		if b.Uniform && b.Index == p.Index && b.Name == p.Name {
			// The kernel's own parameter object, not a helper's of the same
			// name and index.
			if p.Index < len(in.fn.Params) && in.fn.Params[p.Index] == p {
				return true
			}
		}
	}
	return false
}
