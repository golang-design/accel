// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package uniform

import (
	"fmt"
	"go/token"

	"golang.design/x/accel/internal/kernelc/ir"
)

// Rejection is one barrier the analysis will not admit.
type Rejection struct {
	// Pos is the barrier's position, which is where a caller has to look.
	Pos token.Pos

	// Because is the position of the predicate, loop, or branch that made the
	// control flow non-uniform. It is separate from Pos because they are
	// usually different lines and the fix is at this one.
	Because token.Pos

	// Msg says which of the rule's three clauses failed.
	Msg string
}

func (r Rejection) Error() string { return r.Msg }

// AcceptBarriers reports every barrier whose control flow this analysis cannot
// prove uniform at the barrier's scope.
//
// # The rule
//
// From specs/002-compute-model.md section 3.3, a barrier B is accepted iff
//
//  1. every predicate in B's control dependence set is uniform at B's scope,
//  2. every loop enclosing B has a trip count uniform at that scope, and
//  3. no break or continue controlled by a less uniform predicate can skip it.
//
// Clause 3 is the one that is easy to miss, because a loop can have a
// perfectly uniform trip count and still be left early by some invocations:
//
//	for i := uint32(0); i < 8; i++ {   // uniform trip count
//	    if t.LocalIndex() > 2 { break } // but not everyone gets here
//	    t.Barrier()                     // so arrival is not uniform
//	}
//
// An analysis checking only clauses 1 and 2 admits that barrier, and an
// admitted invalid barrier is undefined behaviour that passes every test this
// project can run without buying hardware and then hangs on a user's device.
// specs/002-compute-model.md says over-rejecting is the cheaper mistake, so
// clause 3 is checked and this kernel is refused.
func AcceptBarriers(fn *ir.Func) []Rejection {
	in := Of(fn)
	c := &acceptor{in: in}
	c.walk(fn.Body, Workgroup, nil)
	return c.out
}

type acceptor struct {
	in  *Info
	out []Rejection
}

// escape is a break or continue whose predicate is less uniform than the loop
// it leaves, recorded while walking a loop body so that a barrier appearing
// after it can be refused.
type escape struct {
	pos   token.Pos
	level Level
	what  string
}

// walk visits statements under a control level, carrying the non-uniform
// escapes seen so far in the enclosing loop.
func (c *acceptor) walk(b *ir.Block, ctrl Level, escapes []escape) []escape {
	if b == nil {
		return escapes
	}
	for _, s := range b.List {
		escapes = c.stmt(s, ctrl, escapes)
	}
	return escapes
}

func (c *acceptor) stmt(s ir.Stmt, ctrl Level, escapes []escape) []escape {
	switch n := s.(type) {
	case *ir.Block:
		return c.walk(n, ctrl, escapes)

	case *ir.If:
		cond := max(ctrl, c.in.Level(n.Cond))
		escapes = c.walk(n.Then, cond, escapes)
		if n.Else != nil {
			escapes = c.stmt(n.Else, cond, escapes)
		}

	case *ir.For:
		// A new loop, so escapes from an enclosing one cannot skip anything
		// inside this one: they would have left this loop's enclosing scope
		// before entering it.
		inner := ctrl
		if n.Cond != nil {
			inner = max(inner, c.in.Level(n.Cond))
		}
		c.walk(n.Body, inner, nil)
		if n.Post != nil {
			c.stmt(n.Post, inner, nil)
		}

	case *ir.Break:
		if ctrl > Workgroup {
			escapes = append(escapes, escape{pos: n.Pos(), level: ctrl, what: "break"})
		}

	case *ir.Continue:
		if ctrl > Workgroup {
			escapes = append(escapes, escape{pos: n.Pos(), level: ctrl, what: "continue"})
		}

	case *ir.ExprStmt:
		if call, ok := n.X.(*ir.IntrinsicCall); ok && call.Op == ir.OpBarrier {
			c.barrier(call, ctrl, escapes)
		}
	}
	return escapes
}

// barrier checks one barrier against the three clauses.
func (c *acceptor) barrier(call *ir.IntrinsicCall, ctrl Level, escapes []escape) {
	// Clauses 1 and 2 together: the control level a statement sits under
	// already folds in every enclosing predicate and loop condition, because a
	// loop's body is control-dependent on its condition.
	if ctrl > Workgroup {
		c.out = append(c.out, Rejection{
			Pos: call.Pos(), Because: call.Pos(),
			Msg: fmt.Sprintf("a barrier must sit in workgroup-uniform control flow and this "+
				"one is reached under %v control: every invocation of the workgroup has to "+
				"reach the same barrier, and here some may not "+
				"(specs/002-compute-model.md section 3.1)", ctrl),
		})
		return
	}
	// Clause 3: a uniform loop that some invocations leave early.
	for _, e := range escapes {
		c.out = append(c.out, Rejection{
			Pos: call.Pos(), Because: e.pos,
			Msg: fmt.Sprintf("a barrier must sit in workgroup-uniform control flow, and a %s "+
				"under %v control can skip this one even though the enclosing loop's trip "+
				"count is uniform: the invocations that took it never arrive "+
				"(specs/002-compute-model.md section 3.1)", e.what, e.level),
		})
		return
	}
}
