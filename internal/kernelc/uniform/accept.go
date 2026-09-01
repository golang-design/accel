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
//  3. no break, continue or return controlled by a less uniform predicate can
//     skip it.
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
//
// The three escapes differ in reach. A break leaves the loop, so it can skip
// every barrier in that loop -- the ones before it in the body as well, since
// the invocations that broke do not arrive at the next iteration's. A continue
// leaves the iteration, so it skips only what follows it. A return leaves the
// function, so it skips every barrier after it, at any depth.
func AcceptBarriers(fn *ir.Func) []Rejection {
	in := Of(fn)
	c := &acceptor{in: in}
	c.walk(fn.Body, Workgroup, nil)
	return c.out
}

type acceptor struct {
	in  *Info
	out []Rejection

	// returns is every return under less than workgroup-uniform control seen
	// so far. It is on the acceptor rather than threaded through the walk
	// because a return is not scoped to a loop: an invocation that returned
	// never arrives anywhere after it.
	returns []escape
}

// escape is a break, continue or return whose predicate is less uniform than
// the scope it leaves, recorded so that a barrier it can skip can be refused.
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
		// before entering it. Its own breaks are collected before the body is
		// walked, because a break skips the barriers before it as well.
		inner := ctrl
		if n.Cond != nil {
			inner = max(inner, c.in.Level(n.Cond))
		}
		c.walk(n.Body, inner, c.breaks(n.Body, nil))
		if n.Post != nil {
			c.stmt(n.Post, inner, nil)
		}

	case *ir.Break:
		// Collected by breaks when the loop was entered.

	case *ir.Continue:
		if ctrl > Workgroup {
			escapes = append(escapes, escape{pos: n.Pos(), level: ctrl, what: "continue"})
		}

	case *ir.Return:
		if ctrl > Workgroup {
			c.returns = append(c.returns, escape{pos: n.Pos(), level: ctrl, what: "return"})
		}

	case *ir.ExprStmt:
		call, ok := n.X.(*ir.IntrinsicCall)
		if !ok {
			break
		}
		switch {
		case call.Op.IsWorkgroupBarrier():
			c.barrier(call, ctrl, escapes)
		case call.Op == ir.OpSubgroupBarrier:
			c.subgroupBarrier(call, ctrl, escapes)
		}
	}
	return escapes
}

// breaks collects every break in a loop body that sits under less than
// workgroup-uniform control, not descending into nested loops, whose breaks
// leave only themselves.
//
// The level is the analysis's own control level for the statement, which
// already folds in the loop's escape: once one break is non-uniform the whole
// body is, so a uniform-looking break after it is reported at the raised level.
// That attributes the arrival failure to a second line as well as the first,
// and both are lines the fix has to consider.
func (c *acceptor) breaks(b *ir.Block, out []escape) []escape {
	if b == nil {
		return out
	}
	for _, s := range b.List {
		switch n := s.(type) {
		case *ir.Block:
			out = c.breaks(n, out)
		case *ir.If:
			out = c.breaks(n.Then, out)
			if els, ok := n.Else.(*ir.Block); ok {
				out = c.breaks(els, out)
			} else if els, ok := n.Else.(*ir.If); ok {
				out = c.breaks(ir.NewBlock(els.Pos(), els), out)
			}
		case *ir.Break:
			if l := c.in.Control(n); l > Workgroup {
				out = append(out, escape{pos: n.Pos(), level: l, what: "break"})
			}
		}
	}
	return out
}

// subgroupBarrier checks a subgroup barrier, which obeys the same three clauses
// at **subgroup** scope.
//
// specs/002-compute-model.md §5.3: it is a barrier, so §3.1 applies, and the
// scope it applies at is one subgroup. That is the whole difference and it is
// the reason the call exists. Control predicated on SubgroupIndex is
// subgroup-uniform, so every lane of each subgroup still reaches this, and 002
// §12's testing section names exactly this pair: "a SubgroupBarrier controlled
// by SubgroupID is accepted while the same control around a workgroup barrier
// is rejected. A predicate involving SubgroupInvocationID rejects both."
//
// The lattice is what makes this one comparison rather than a second analysis:
// Workgroup < Subgroup < Non already means "uniform across a subgroup", so the
// clause is the same clause with a different threshold.
func (c *acceptor) subgroupBarrier(call *ir.IntrinsicCall, ctrl Level, escapes []escape) {
	if e, ok := firstAbove(escapes, c.returns, Subgroup); ok {
		c.out = append(c.out, Rejection{
			Pos: call.Pos(), Because: e.pos,
			Msg: "a subgroup barrier must sit in subgroup-uniform control flow, and " +
				e.describe("lanes") + " (specs/002-compute-model.md section 5.3)",
		})
		return
	}
	if ctrl > Subgroup {
		c.out = append(c.out, Rejection{
			Pos: call.Pos(), Because: call.Pos(),
			Msg: fmt.Sprintf("a subgroup barrier must sit in subgroup-uniform control flow "+
				"and this one is reached under %v control: every lane of the subgroup has "+
				"to reach the same barrier, and here some may not "+
				"(specs/002-compute-model.md section 5.3)", ctrl),
		})
	}
}

// barrier checks one barrier against the three clauses.
//
// Clause 3 is checked first. A break some invocations take raises the level
// of everything in its loop, the loop variable's increment included, so the
// loop's condition reads as non-uniform too and clauses 1 and 2 would report
// the barrier under non-uniform control without saying why. The escape's own
// line is the one the fix is at, so when there is one it is the one named.
func (c *acceptor) barrier(call *ir.IntrinsicCall, ctrl Level, escapes []escape) {
	// Clause 3: a uniform loop that some invocations leave early, or a
	// function some invocations have already left.
	if e, ok := firstAbove(escapes, c.returns, Workgroup); ok {
		c.out = append(c.out, Rejection{
			Pos: call.Pos(), Because: e.pos,
			Msg: "a barrier must sit in workgroup-uniform control flow, and " +
				e.describe("invocations") + " (specs/002-compute-model.md section 3.1)",
		})
		return
	}
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
	}
}

// firstAbove is the first escape in either list whose level is below the
// scope a barrier needs, in source order: loop escapes first, then returns.
func firstAbove(loop, returns []escape, scope Level) (escape, bool) {
	for _, list := range [][]escape{loop, returns} {
		for _, e := range list {
			if e.level > scope {
				return e, true
			}
		}
	}
	return escape{}, false
}

// describe says how an escape skips a barrier, naming who never arrives.
func (e escape) describe(who string) string {
	if e.what == "return" {
		return fmt.Sprintf("a return under %v control leaves the function before this one: "+
			"the %s that returned never arrive", e.level, who)
	}
	return fmt.Sprintf("a %s under %v control can skip this one even though the enclosing "+
		"loop's trip count is uniform: the %s that took it never arrive", e.what, e.level, who)
}
