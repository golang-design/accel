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

import "golang.design/x/accel/internal/kernelc/ir"

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
	// values is every value's level. Keyed by the node itself, which is stable
	// because the IR is a tree of pointers built once.
	values map[ir.Value]Level

	// control is the level of the control flow reaching each statement: the
	// least-uniform predicate any enclosing conditional or loop depends on.
	control map[ir.Stmt]Level
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
		values:  map[ir.Value]Level{},
		control: map[ir.Stmt]Level{},
	}
	// Iterate to a fixed point. A value only ever becomes less uniform, so this
	// terminates in at most three passes over the tree; the loop bound is
	// generous rather than tight because a wrong bound here is a wrong answer.
	for range 8 {
		before := len(in.values)
		changed := in.walkBlock(fn.Body, Workgroup)
		if !changed && len(in.values) == before {
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

// walkBlock walks a block under a control level, returning whether anything
// became less uniform.
func (in *Info) walkBlock(b *ir.Block, ctrl Level) bool {
	if b == nil {
		return false
	}
	changed := false
	for _, s := range b.List {
		changed = in.walkStmt(s, ctrl) || changed
	}
	return changed
}

func (in *Info) walkStmt(s ir.Stmt, ctrl Level) bool {
	if old, ok := in.control[s]; !ok || ctrl > old {
		in.control[s] = ctrl
	}
	changed := false

	switch n := s.(type) {
	case *ir.Block:
		changed = in.walkBlock(n, ctrl)

	case *ir.Declare:
		l := in.value(n.Init)
		// A definition inherits the control flow it sits under. This is the
		// clause that stops `if l < 4 { x = 1 } else { x = 2 }` from being
		// called uniform because both 1 and 2 are literals.
		changed = in.set(n.Local, max(l, ctrl))

	case *ir.Assign:
		l := max(in.value(n.RHS), ctrl)
		if lhs, ok := n.LHS.(*ir.Local); ok {
			changed = in.set(lhs, l)
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
		changed = in.walkBlock(n.Then, cond) || changed
		if n.Else != nil {
			changed = in.walkStmt(n.Else, cond) || changed
		}

	case *ir.For:
		// A loop's body is control-dependent on its condition, and on anything
		// its init or post computes.
		inner := ctrl
		if n.Init != nil {
			changed = in.walkStmt(n.Init, ctrl) || changed
		}
		if n.Cond != nil {
			inner = max(inner, in.value(n.Cond))
		}
		changed = in.walkBlock(n.Body, inner) || changed
		if n.Post != nil {
			changed = in.walkStmt(n.Post, inner) || changed
		}

	case *ir.Return:
		if n.Value != nil {
			in.value(n.Value)
		}

	case *ir.Break, *ir.Continue:
		// No values, and their control level is what [AcceptBarriers] reads to
		// enforce the rule's third clause: a uniform loop that some invocations
		// leave early does not have uniform arrival inside it.
	}
	return changed
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
		in.value(n.X)
		in.value(n.Index)
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
		// A helper's result is as uniform as its least uniform argument. That is
		// sound and imprecise: a helper ignoring its arguments returns something
		// uniform, and this calls it non-uniform. Precision here needs an
		// interprocedural summary, which is more machinery than the subset's
		// helper depth justifies.
		l := Workgroup
		for _, a := range n.Args {
			l = max(l, in.value(a))
		}
		return l

	case *ir.IntrinsicCall:
		return in.intrinsic(n)
	}
	return Non
}

// intrinsic is the seed table of specs/002-compute-model.md section 3.3.
func (in *Info) intrinsic(n *ir.IntrinsicCall) Level {
	if n.Recv != nil {
		in.value(n.Recv)
	}
	args := Workgroup
	for _, a := range n.Args {
		args = max(args, in.value(a))
	}

	switch n.Op {
	// Workgroup-uniform: the same for every invocation of one workgroup.
	//
	// The dispatch shape is here because it is uniform across more than the
	// workgroup and the lattice has no level above it: specs/052.
	case ir.OpGroupID, ir.OpGroupIndex,
		ir.OpWorkgroupSize, ir.OpNumGroups, ir.OpGlobalSize,
		ir.OpSubgroupSize:
		return Workgroup

	// Subgroup-uniform: equal within a subgroup and differing between them.
	//
	// The only seed at this level, and until it existed nothing produced one at
	// all -- the lattice had three levels and the analysis could reach two, so
	// every subgroup-scope rule was vacuous. specs/002-compute-model.md §3.3's
	// seed table is where these rows come from.
	case ir.OpSubgroupID:
		return Subgroup

	// Non-uniform: these are what distinguish one invocation from another, and
	// a kernel with no non-uniform seed computes the same thing everywhere.
	//
	// SubgroupInvocationID is here rather than at Subgroup because it is the
	// lane's own index: it differs *within* a subgroup, which is exactly what
	// the level below means.
	case ir.OpGlobalID, ir.OpLocalID, ir.OpGlobalIndex, ir.OpLocalIndex,
		ir.OpSubgroupInvocationID:
		return Non
	}

	// A subgroup operation's result is non-uniform, and §3.3 says so with the
	// reason: conservative even when the operation broadcasts. A reduction does
	// return the same value to every active lane, and the *active set* is not
	// portable (§5.1), so a value that is uniform on one device is not on
	// another. This is checked after the switch because the range predicate
	// covers a family rather than a list.
	if n.Op.IsSubgroupRendezvous() {
		return Non
	}

	// Everything else is arithmetic on its arguments: sqrt of a uniform value
	// is uniform. Barriers have no result and their level is never read.
	return args
}
