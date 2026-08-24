// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package emit

import (
	"fmt"
	"go/token"
	"path/filepath"
	"strings"

	"golang.design/x/accel/internal/kernelc/ir"
)

// The resumable cooperative lowering.
//
// # The shape, and why it is this one
//
// A cooperative kernel's invocations rendezvous, so they cannot run one after
// another to completion. The generated lowering is therefore a function that
// runs from a program counter to the next suspension point and returns, and a
// scheduler that calls it for every invocation before releasing the epoch:
//
//	func kCoop(t accel.Thread, in, out []float32, f *kFrame) bool {
//		switch f.pc {
//		case 0:
//			... code before the first barrier ...
//			f.pc = 1
//			return true          // suspended
//		case 1:
//			... code after it ...
//		}
//		return false             // finished
//	}
//
// The switch is flat rather than nested because
// specs/002-compute-model.md section 3.1 requires every barrier to sit in
// uniform control flow, which is what makes suspension points a *sequence*.
// Without that rule this would need a relooper. The uniformity analysis that
// enforces it runs before this, and a kernel reaching here has passed it.
//
// # Why every local goes in the frame
//
// A local declared before a suspension point and read after it must survive the
// return, and computing exactly which ones do is a liveness analysis. Putting
// all of them in the frame is a superset of that answer and is therefore
// correct; the cost is a larger frame, which for the loop counters and
// accumulators a kernel declares is a few words per invocation. A liveness
// analysis can shrink it later without changing anything a caller sees, which
// is the argument for taking the correct-and-larger version first.

// renumber puts the states in source order, with the entry first.
//
// They are *built* backwards, because a state has to know its successor's index
// and the successor is created first. That leaves the entry state last, and a
// machine whose program counter starts at zero would begin in the middle. The
// build order is an implementation detail; the numbering is not, because it is
// what a reader of the generated file sees.
func renumber(segs []segment) []segment {
	n := len(segs)
	at := func(i int) int {
		if i < 0 {
			return -1
		}
		return n - 1 - i
	}
	out := make([]segment, n)
	for i, s := range segs {
		s.next = at(s.next)
		s.bodyNext = at(s.bodyNext)
		s.exitNext = at(s.exitNext)
		out[at(i)] = s
	}
	return out
}

// countSuspensions is how many states end at a barrier, which bounds the
// scheduler's epoch loop.
//
// The state count is not the answer: a loop's check and post states are states
// that never suspend, so counting them would let a kernel that stopped
// advancing run for extra epochs before being reported.
func countSuspensions(segs []segment) int {
	n := 0
	for _, s := range segs {
		if s.suspend {
			n++
		}
	}
	return n
}

// position resolves an IR position to a file name, line, and column.
//
// The base name rather than the full path, because this string is written into
// a committed generated file: an absolute path would differ between machines
// and make the freshness check fail on every checkout but the one that ran the
// generator. The base name is stable and still actionable, since a kernel and
// its generated file sit in one package.
func (e *emitter) position(p token.Pos) string {
	if e.fset == nil || !p.IsValid() {
		return ""
	}
	pos := e.fset.Position(p)
	return fmt.Sprintf("%s:%d:%d", filepath.Base(pos.Filename), pos.Line, pos.Column)
}

// paramType is a parameter's Go spelling in a generated lowering.
//
// Shared memory is a pointer to its array. That is not a style choice: a
// workgroup shares one copy, and passing the array by value gives each
// invocation its own, which compiles and computes something else. Go's own
// rules are what make this visible, so the generated signature says it.
func (e *emitter) paramType(t *ir.Type) string {
	if t != nil && t.Kind == ir.Array {
		return "*" + e.goType(t)
	}
	return e.goType(t)
}

// segment is one state of the generated machine: a run of statements ending in
// a suspension, a jump, or the end of the kernel.
type segment struct {
	stmts []ir.Stmt

	// sub is the subgroup operation this state takes part in, or SubNone.
	// subContribute writes this lane's value into the frame and suspends;
	// subResult reads the combined value back into the local.
	sub           ir.Opcode
	subCall       *ir.IntrinsicCall
	subLocal      *ir.Local
	subContribute bool
	subResult     bool

	// next is the state to enter when this one falls off its end, or -1 for the
	// end of the kernel. It is explicit because a loop's states do not run in
	// numeric order: the state after a mid-loop barrier jumps back to the
	// loop's condition rather than forward.
	next int

	// suspend reports that this state ends at a barrier, and pos is where.
	suspend bool
	pos     string

	// loop, when non-nil, makes this state a loop check: evaluate the condition
	// and enter body or exit accordingly.
	loop     *ir.For
	bodyNext int
	exitNext int
}

type splitter struct {
	fn   *ir.Func
	segs []segment
	pos  func(ir.Stmt) string
	err  error
}

// nestedRendezvousIn reports a subgroup rendezvous buried in a statement's
// expressions, which the split cannot express.
func nestedRendezvousIn(s ir.Stmt) (*ir.IntrinsicCall, bool) {
	var found *ir.IntrinsicCall
	var walk func(ir.Value)
	walk = func(v ir.Value) {
		if found != nil || v == nil {
			return
		}
		switch n := v.(type) {
		case *ir.IntrinsicCall:
			if n.Op.IsSubgroupRendezvous() {
				found = n
				return
			}
			walk(n.Recv)
			for _, a := range n.Args {
				walk(a)
			}
		case *ir.Binary:
			walk(n.X)
			walk(n.Y)
		case *ir.Unary:
			walk(n.X)
		case *ir.Convert:
			walk(n.X)
		case *ir.IndexExpr:
			walk(n.X)
			walk(n.Index)
		case *ir.Call:
			for _, a := range n.Args {
				walk(a)
			}
		case *ir.FieldSel:
			walk(n.X)
		case *ir.Len:
			walk(n.X)
		}
	}
	switch n := s.(type) {
	case *ir.Declare:
		walk(n.Init)
	case *ir.Assign:
		walk(n.LHS)
		walk(n.RHS)
	case *ir.ExprStmt:
		walk(n.X)
	case *ir.If:
		walk(n.Cond)
	case *ir.Return:
		walk(n.Value)
	}
	return found, found != nil
}

// emit lays out a statement list as states, and returns the index of the first.
//
// after is the state to enter once the list is exhausted. Building backwards
// from it is what lets a loop's body know where to jump: the successor is
// always already numbered when a state that needs it is created.
func (c *splitter) emit(list []ir.Stmt, after int) int {
	// Walk backwards, so each state's successor exists before it does.
	next := after
	run := []ir.Stmt{}
	flush := func() {
		if len(run) == 0 {
			return
		}
		rev := make([]ir.Stmt, len(run))
		for i := range run {
			rev[i] = run[len(run)-1-i]
		}
		next = c.add(segment{stmts: rev, next: next})
		run = run[:0]
	}

	for i := len(list) - 1; i >= 0; i-- {
		s := list[i]
		switch {
		case isBarrier(s):
			flush()
			// The suspension is its own state boundary: everything before it
			// runs, then the machine returns and resumes at what follows.
			next = c.add(segment{next: next, suspend: true, pos: c.position(s)})

		case isSubgroupStmt(s):
			flush()
			local, call, _ := subgroupRendezvous(s)
			// Two states, and the suspension sits *between* them: one
			// contributes this lane's value and returns, the scheduler
			// combines, and the other reads the result back. Suspending on the
			// second instead would read a value nothing had written yet.
			next = c.add(segment{
				next: next, sub: call.Op, subLocal: local, subResult: true,
			})
			next = c.add(segment{
				next: next, suspend: true, pos: c.position(s),
				sub: call.Op, subCall: call, subContribute: true,
			})

		case hasLoopBarrier(s):
			flush()
			loop := s.(*ir.For)
			// Three states: a check, the body, and whatever follows the loop.
			// The check's index is reserved first so the body can jump back to
			// it, which is the whole reason this is not a simple split.
			check := c.add(segment{loop: loop, exitNext: next})
			body := c.emit(loop.Body.List, c.postState(loop, check))
			c.segs[check].bodyNext = body
			next = check
			if loop.Init != nil {
				next = c.add(segment{stmts: []ir.Stmt{loop.Init}, next: check})
			}

		default:
			// A rendezvous inside a larger expression cannot be split: the
			// machine would have to suspend mid-evaluation and resume at a
			// point Go has no way to name. Refused with the position rather
			// than lowered as an ordinary call, which would combine nothing and
			// return the lane's own value.
			if v, bad := nestedRendezvousIn(s); bad {
				c.err = fmt.Errorf("a subgroup operation may only be assigned directly to a "+
					"local, as `v := t.SubgroupAddF32(x)`: this one is inside a larger "+
					"expression, and the state machine would have to suspend part-way "+
					"through evaluating it (%v at %s)", v.Op, c.position(s))
				return next
			}
			run = append(run, s)
		}
	}
	flush()
	return next
}

// postState is the state that runs a loop's post statement and returns to its
// check. A loop with no post goes straight back.
func (c *splitter) postState(loop *ir.For, check int) int {
	if loop.Post == nil {
		return check
	}
	return c.add(segment{stmts: []ir.Stmt{loop.Post}, next: check})
}

func (c *splitter) add(s segment) int {
	c.segs = append(c.segs, s)
	return len(c.segs) - 1
}

func (c *splitter) position(s ir.Stmt) string {
	if c.pos == nil {
		return ""
	}
	return c.pos(s)
}

// isSubgroupStmt reports a statement that is a subgroup rendezvous.
func isSubgroupStmt(s ir.Stmt) bool {
	_, _, ok := subgroupRendezvous(s)
	return ok
}

// hasLoopBarrier reports a loop whose body reaches a barrier.
func hasLoopBarrier(s ir.Stmt) bool {
	loop, ok := s.(*ir.For)
	return ok && blockHasBarrier(loop.Body)
}

// checkBarrierPlacement refuses a barrier this transform cannot express.
//
// A barrier inside a conditional is legal by specs/002-compute-model.md and is
// still refused: the state machine would have to resume inside the branch, and
// unlike a loop the branch has no back edge to hang a state on. It is refused
// by position rather than mis-lowered, because a barrier lowered as a no-op is
// a different program that compiles.
func checkBarrierPlacement(b *ir.Block, top bool) error {
	if b == nil {
		return nil
	}
	for _, s := range b.List {
		if isBarrier(s) {
			if !top {
				return fmt.Errorf("a barrier inside a conditional needs the state machine to " +
					"resume inside that branch, which this split does not express; a barrier " +
					"must sit in workgroup-uniform control flow anyway, so hoist it out of " +
					"the conditional (specs/018-cooperative-lowering.md)")
			}
			continue
		}
		if err := checkNested(s); err != nil {
			return err
		}
	}
	return nil
}

func checkNested(s ir.Stmt) error {
	switch n := s.(type) {
	case *ir.Block:
		return checkBarrierPlacement(n, false)
	case *ir.If:
		if err := checkBarrierPlacement(n.Then, false); err != nil {
			return err
		}
		if n.Else != nil {
			return checkNested(n.Else)
		}
	case *ir.For:
		// A loop body is a new top level: its barriers become states of their
		// own, which is what the three-state split is for.
		return checkBarrierPlacement(n.Body, true)
	}
	return nil
}

// blockHasBarrier reports whether a block reaches a barrier at any depth.
func blockHasBarrier(b *ir.Block) bool {
	if b == nil {
		return false
	}
	for _, s := range b.List {
		if isBarrier(s) {
			return true
		}
		switch n := s.(type) {
		case *ir.Block:
			if blockHasBarrier(n) {
				return true
			}
		case *ir.If:
			if blockHasBarrier(n.Then) {
				return true
			}
			if inner, ok := n.Else.(*ir.Block); ok && blockHasBarrier(inner) {
				return true
			}
		case *ir.For:
			if blockHasBarrier(n.Body) {
				return true
			}
		}
	}
	return false
}

func isBarrier(s ir.Stmt) bool {
	e, ok := s.(*ir.ExprStmt)
	if !ok {
		return false
	}
	c, ok := e.X.(*ir.IntrinsicCall)
	return ok && c.Op == ir.OpBarrier
}

// subgroupRendezvous reports a statement that is a subgroup operation whose
// result is assigned to a local, which is the only shape the split handles.
//
// The restriction is what keeps the state machine flat. A rendezvous nested in
// a larger expression would have to suspend in the middle of evaluating it, and
// resuming would mean re-entering that expression at a point Go has no way to
// name. `v := t.SubgroupAddF32(x)` is what a kernel writes anyway, and anything
// else is refused with the position rather than lowered wrongly.
func subgroupRendezvous(s ir.Stmt) (*ir.Local, *ir.IntrinsicCall, bool) {
	switch n := s.(type) {
	case *ir.Declare:
		if c, ok := n.Init.(*ir.IntrinsicCall); ok && c.Op.IsSubgroupRendezvous() {
			return n.Local, c, true
		}
	case *ir.Assign:
		if c, ok := n.RHS.(*ir.IntrinsicCall); ok && c.Op.IsSubgroupRendezvous() {
			if l, ok := n.LHS.(*ir.Local); ok {
				return l, c, true
			}
		}
	}
	return nil, nil, false
}

// coopLowering emits the frame type and the resumable function.
func (e *emitter) coopLowering(k *ir.Func, segs []segment) {
	frame := frameName(k.Name)
	lower := coopName(k.Name)

	e.sharedIndex = map[string]int{}
	for i, sh := range k.Shared {
		e.sharedIndex[sh.Name] = i
	}
	defer func() { e.sharedIndex = nil }()

	e.printf("// %s is one invocation's saved state between suspension points.\n", frame)
	e.printf("//\n")
	e.printf("// Every local lives here rather than only those live across a barrier: that\n")
	e.printf("// is a superset of the right answer and therefore correct, and a liveness\n")
	e.printf("// analysis can shrink it later without changing anything a caller sees.\n")
	e.printf("type %s struct {\n", frame)
	e.printf("\tpc int\n")
	locals := collectLocals(k.Body)
	for _, l := range locals {
		e.printf("\t%s %s\n", localField(l), e.goType(l.Type()))
	}
	e.printf("}\n\n")

	e.printf("// %s runs one invocation of %s to its next suspension point.\n", lower, k.Name)
	e.printf("//\n")
	e.printf("// It reports whether the invocation suspended. False means it finished, and\n")
	e.printf("// the scheduler stops calling it. Each case is one state; the assignment to\n")
	e.printf("// pc before continuing is the jump, which is explicit because a loop's states\n")
	e.printf("// do not run in numeric order.\n")
	e.printf("func %s(", lower)
	for i, p := range k.Params {
		if i > 0 {
			e.printf(", ")
		}
		e.printf("%s %s", p.Name, e.paramType(p.Type()))
	}
	e.printf(", f *%s, frame *kernelabi.Frame, tr *kernelabi.SharedTracker) bool {\n", frame)

	// A loop over a switch, so a state that jumps backwards is a `continue`
	// rather than a recursive call: the machine's own control flow stays flat
	// however deeply the kernel nested its loops.
	e.printf("\tfor {\n")
	e.printf("\t\tswitch f.pc {\n")
	for i, seg := range segs {
		e.printf("\t\tcase %d:\n", i)
		e.coopSegment(seg, k, locals, i, 3)
	}
	e.printf("\t\t}\n")
	e.printf("\t\treturn false\n")
	e.printf("\t}\n")
	e.printf("}\n\n")
}

// coopSegment emits one state.
func (e *emitter) coopSegment(seg segment, k *ir.Func, locals []*ir.Local, index, depth int) {
	prev := e.frameLocals
	e.frameLocals = map[*ir.Local]bool{}
	for _, l := range locals {
		e.frameLocals[l] = true
	}
	defer func() { e.frameLocals = prev }()

	pad := indent(depth)

	if seg.loop != nil {
		// A loop check: evaluate the condition and jump to the body or past it.
		e.printf("%sif ", pad)
		if seg.loop.Cond != nil {
			e.value(seg.loop.Cond)
		} else {
			e.printf("true")
		}
		e.printf(" {\n")
		e.jump(seg.bodyNext, depth+1)
		e.printf("%s}\n", pad)
		e.jump(seg.exitNext, depth)
		return
	}

	for _, s := range seg.stmts {
		e.stmt(s, depth)
	}

	// A contribution state writes this lane's value into the frame and falls
	// through to the suspending state; a result state reads the combined value
	// back. They are separate states so the suspension sits between them, which
	// is what gives the scheduler its chance to combine.
	if seg.subContribute {
		name, ok := subOpName(seg.sub)
		if !ok {
			e.fail("subgroup operation %v has no runtime name, so the lowering would "+
				"suspend without saying what to combine", seg.sub)
			return
		}
		e.printf("%sframe.Sub = %s\n", pad, name)
		switch subCarrier(seg.sub) {
		case carrierF32:
			e.printf("%sframe.SubF32 = ", pad)
			e.rounded(seg.subCall.Args[0])
			e.printf("\n")
		case carrierF32Lane:
			if len(seg.subCall.Args) != 2 {
				e.fail("%v takes a value and a lane, and got %d arguments",
					seg.sub, len(seg.subCall.Args))
				return
			}
			e.printf("%sframe.SubF32 = ", pad)
			e.rounded(seg.subCall.Args[0])
			e.printf("\n")
			e.printf("%sframe.SubLane = ", pad)
			e.value(seg.subCall.Args[1])
			e.printf("\n")
		case carrierBool:
			if len(seg.subCall.Args) == 1 {
				e.printf("%sframe.SubBool = ", pad)
				e.value(seg.subCall.Args[0])
				e.printf("\n")
			}
		default:
			e.fail("subgroup operation %v carries no value between a lane and the "+
				"scheduler, so its result would be whatever the frame last held", seg.sub)
			return
		}
	}
	if seg.subResult {
		switch subCarrier(seg.sub) {
		case carrierF32, carrierF32Lane:
			e.printf("%s%s = frame.SubF32\n", pad, e.local(seg.subLocal))
		case carrierBool:
			e.printf("%s%s = frame.SubBool\n", pad, e.local(seg.subLocal))
		}
	}

	if seg.suspend {
		e.printf("%sf.pc = %d\n", pad, seg.next)
		e.printf("%sframe.Barrier = kernelabi.BarrierID{Index: %d, Pos: %q}\n",
			pad, index, seg.pos)
		e.printf("%sreturn true\n", pad)
		return
	}
	e.jump(seg.next, depth)
}

// The value a subgroup operation carries between a lane and the scheduler.
type carrier int

const (
	// carrierNone is an opcode this table does not know, and it is reported
	// rather than lowered: an unmapped rendezvous suspends, combines nothing,
	// and resumes reading whatever the frame last held, which is a wrong answer
	// that compiles.
	carrierNone carrier = iota
	carrierF32
	carrierBool

	// carrierF32Lane is a lane-addressed read: a value, and the lane operand
	// naming which lane to take it from.
	carrierF32Lane
)

func subCarrier(op ir.Opcode) carrier {
	switch op {
	case ir.OpSubgroupAddF32, ir.OpSubgroupMinF32, ir.OpSubgroupMaxF32, ir.OpBroadcastFirstF32,
		ir.OpSubgroupInclusiveAddF32, ir.OpSubgroupExclusiveAddF32:
		return carrierF32
	case ir.OpElect, ir.OpSubgroupAny, ir.OpSubgroupAll:
		return carrierBool
	}
	if op.IsSubgroupLaneRead() {
		return carrierF32Lane
	}
	return carrierNone
}

// subOpName is the runtime constant naming one subgroup rendezvous, and whether
// there is one.
//
// Whether, because the zero answer used to be kernelabi.SubNone, which is an
// ordinary barrier: an opcode added to the IR and forgotten here lowered to a
// suspension that combined nothing and read back the lane's own contribution.
func subOpName(op ir.Opcode) (string, bool) {
	switch op {
	case ir.OpSubgroupAddF32:
		return "kernelabi.SubAddF32", true
	case ir.OpSubgroupMinF32:
		return "kernelabi.SubMinF32", true
	case ir.OpSubgroupMaxF32:
		return "kernelabi.SubMaxF32", true
	case ir.OpBroadcastFirstF32:
		return "kernelabi.SubBroadcastFirstF32", true
	case ir.OpElect:
		return "kernelabi.SubElect", true
	case ir.OpSubgroupAny:
		return "kernelabi.SubAny", true
	case ir.OpSubgroupAll:
		return "kernelabi.SubAll", true
	case ir.OpBroadcastF32:
		return "kernelabi.SubBroadcastF32", true
	case ir.OpShuffleF32:
		return "kernelabi.SubShuffleF32", true
	case ir.OpShuffleXorF32:
		return "kernelabi.SubShuffleXorF32", true
	case ir.OpShuffleUpF32:
		return "kernelabi.SubShuffleUpF32", true
	case ir.OpShuffleDownF32:
		return "kernelabi.SubShuffleDownF32", true
	case ir.OpSubgroupInclusiveAddF32:
		return "kernelabi.SubInclusiveAddF32", true
	case ir.OpSubgroupExclusiveAddF32:
		return "kernelabi.SubExclusiveAddF32", true
	}
	return "", false
}

// jump enters another state, or finishes.
func (e *emitter) jump(to, depth int) {
	pad := indent(depth)
	if to < 0 {
		e.printf("%sreturn false\n", pad)
		return
	}
	e.printf("%sf.pc = %d\n", pad, to)
	e.printf("%scontinue\n", pad)
}

// collectLocals is every local the body declares, in declaration order.
//
// Declaration order rather than a map walk, because the frame's field order is
// part of the generated file and a golden that reorders between runs is a
// golden nobody can use.
func collectLocals(b *ir.Block) []*ir.Local {
	var out []*ir.Local
	var walkBlock func(*ir.Block)
	var walk func(ir.Stmt)
	walkBlock = func(b *ir.Block) {
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
			out = append(out, n.Local)
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
	walkBlock(b)
	return out
}

func frameName(k string) string { return lowerFirst(k) + "Frame" }
func coopName(k string) string  { return lowerFirst(k) + "Coop" }

// lowerFirst is the unexported spelling of a kernel or local name.
func lowerFirst(name string) string {
	if name == "" {
		return "x"
	}
	return strings.ToLower(name[:1]) + name[1:]
}

// localField is a local's field name in the frame. The IR's local id makes it
// unique, since two blocks may each declare an `i`.
func localField(l *ir.Local) string { return fmt.Sprintf("%s%d", lowerFirst(l.Name), l.ID) }

// cooperativeKernel emits a cooperative kernel's frame, resumable lowering,
// record, and entry point.
//
// The record carries `Cooperative` rather than `Flat`, which is what selects
// the scheduler at run time. A cooperative kernel has no flat entry point at
// all: running its invocations one after another is a different program, not a
// slower one, so there is nothing to fall back to.
func (e *emitter) cooperativeKernel(k *ir.Func) {
	sp := &splitter{fn: k, pos: func(s ir.Stmt) string { return e.position(s.Pos()) }}
	if err := checkBarrierPlacement(k.Body, true); err != nil {
		e.fail("kernel %s: %v", k.Name, err)
		return
	}
	sp.emit(k.Body.List, -1)
	if sp.err != nil {
		e.fail("kernel %s: %v", k.Name, sp.err)
		return
	}
	segs := renumber(sp.segs)
	if len(segs) == 0 {
		segs = []segment{{next: -1}}
	}
	e.coopLowering(k, segs)

	e.printf("// %s is the compiled form of %s.\n", k.Name+"Kernel", k.Name)
	e.printf("var %s = kernelabi.Kernel{\n", k.Name+"Kernel")
	e.printf("\tName: %q,\n", k.Name)
	e.printf("\tWorkgroupSize: accel.ID3{X: %d, Y: %d, Z: %d},\n",
		k.Workgroup[0], k.Workgroup[1], k.Workgroup[2])
	e.printf("\tBindings: []kernelabi.Binding{\n")
	for _, b := range k.Bindings {
		e.printf("\t\t{Name: %q, DType: %s, Access: %s},\n", b.Name, e.dtype(b.Type), access(b))
	}
	e.printf("\t},\n")
	e.printf("\tDigest: %q,\n", k.Digest)
	e.printf("\tGenerator: kernelabi.Version,\n")
	// The Metal artifact is the *authored* structure, not this resumable
	// lowering: specs/022-msl-target.md section 2 says why. A target with real
	// barriers has no use for a program counter.
	e.mslArtifact(k)
	if k.Caps != 0 {
		e.printf("\tCaps: %d,\n", k.Caps)
	}
	e.printf("\tSuspensions: %d,\n", countSuspensions(segs))
	if len(k.Shared) > 0 {
		e.printf("\tSharedSizes: []int{")
		for i, sh := range k.Shared {
			if i > 0 {
				e.printf(", ")
			}
			e.printf("%d", sh.Type.Len)
		}
		e.printf("},\n")
		total := 0
		for _, sh := range k.Shared {
			b := sh.Type.Elem.Kind.Bytes()
			if b == 0 {
				e.fail("shared %s has element type %v, which has no size, so the "+
					"shared-memory budget cannot be checked against it", sh.Name, sh.Type.Elem)
				return
			}
			total += sh.Type.Len * b
		}
		e.printf("\tSharedBytes: %d,\n", total)
		e.printf("\tNewShared: func() []any {\n")
		for i, sh := range k.Shared {
			e.printf("\t\tvar s%d %s\n", i, e.goType(sh.Type))
			e.printf("\t\tkernelabi.Poison(s%d[:])\n", i)
		}
		e.printf("\t\treturn []any{")
		for i := range k.Shared {
			if i > 0 {
				e.printf(", ")
			}
			e.printf("&s%d", i)
		}
		e.printf("}\n")
		e.printf("\t},\n")
	}
	e.uniformRecords(k)

	// The entry point allocates one frame per invocation on first call and
	// resumes it afterwards. The scheduler owns the frames, so it passes an
	// opaque slot rather than the kernel keeping state of its own: two
	// workgroups run concurrently and a package-level frame would alias them.
	e.printf("\tCooperative: func(t accel.Thread, a kernelabi.Args, slot *kernelabi.Frame) bool {\n")
	e.printf("\t\tf, _ := slot.State.(*%s)\n", frameName(k.Name))
	e.printf("\t\tif f == nil {\n")
	e.printf("\t\t\tf = &%s{}\n", frameName(k.Name))
	e.printf("\t\t\tslot.State = f\n")
	e.printf("\t\t}\n")
	e.printf("\t\treturn %s(t", coopName(k.Name))
	slot := 0
	uniformSlot := 0
	sharedSlot := 0
	for _, p := range k.Params {
		if p.Index == k.Thread {
			continue
		}
		switch {
		case p.Type() != nil && p.Type().Kind == ir.Slice:
			e.printf(", kernelabi.Slice[%s](a, %d)", e.goType(p.Type().Elem), slot)
			slot++
		case p.Type() != nil && p.Type().Kind == ir.Array:
			// A pointer, so every invocation of the workgroup addresses one
			// copy. By value each would get its own, which compiles.
			e.printf(", kernelabi.Shared[%s](a, %d)", e.goType(p.Type()), sharedSlot)
			sharedSlot++
		default:
			e.printf(", kernelabi.UniformValue[%s](a, %d)", e.goType(p.Type()), uniformSlot)
			uniformSlot++
		}
	}
	e.printf(", f, slot, slot.Shared)\n")
	e.printf("\t},\n")
	e.printf("}\n\n")
}
