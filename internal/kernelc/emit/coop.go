// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

package emit

import (
	"fmt"
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

// coopSegments splits a body at top-level barriers.
//
// Returns the segments and the number of suspension points, or an error naming
// a barrier this split cannot express.
func coopSegments(k *ir.Func) ([][]ir.Stmt, error) {
	if err := checkBarrierPlacement(k.Body, true); err != nil {
		return nil, err
	}
	var segs [][]ir.Stmt
	cur := []ir.Stmt{}
	for _, s := range k.Body.List {
		if isBarrier(s) {
			segs = append(segs, cur)
			cur = []ir.Stmt{}
			continue
		}
		cur = append(cur, s)
	}
	segs = append(segs, cur)
	return segs, nil
}

// checkBarrierPlacement refuses a barrier this transform cannot yet split at.
//
// A barrier inside a loop is legal by specs/002-compute-model.md and is what a
// tree reduction is made of, so this is a scheduling gap rather than a rule: it
// needs the state machine to carry a loop's induction variable across an epoch,
// which is a numbering problem this split does not solve. It is refused by
// position rather than mis-lowered, because a barrier lowered as a no-op is a
// different program that compiles.
func checkBarrierPlacement(b *ir.Block, top bool) error {
	if b == nil {
		return nil
	}
	for _, s := range b.List {
		if isBarrier(s) {
			if !top {
				return fmt.Errorf("a barrier inside a loop or conditional needs the state " +
					"machine to resume in the middle of that construct, which the current " +
					"split does not express; hoist it to the top level of the kernel body " +
					"for now (specs/018-cooperative-lowering.md)")
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
		return checkBarrierPlacement(n.Body, false)
	}
	return nil
}

func isBarrier(s ir.Stmt) bool {
	e, ok := s.(*ir.ExprStmt)
	if !ok {
		return false
	}
	c, ok := e.X.(*ir.IntrinsicCall)
	return ok && c.Op == ir.OpBarrier
}

// coopLowering emits the frame type and the resumable function.
func (e *emitter) coopLowering(k *ir.Func, segs [][]ir.Stmt) {
	frame := frameName(k.Name)
	lower := coopName(k.Name)

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
	e.printf("// the scheduler stops calling it. The switch is flat because every barrier\n")
	e.printf("// sits in uniform control flow, so the suspension points are a sequence.\n")
	e.printf("func %s(", lower)
	for i, p := range k.Params {
		if i > 0 {
			e.printf(", ")
		}
		e.printf("%s %s", p.Name, e.paramType(p.Type()))
	}
	e.printf(", f *%s) bool {\n", frame)
	e.printf("\tswitch f.pc {\n")
	for i, seg := range segs {
		e.printf("\tcase %d:\n", i)
		e.coopSegment(seg, locals, 2)
		if i < len(segs)-1 {
			e.printf("\t\tf.pc = %d\n", i+1)
			e.printf("\t\treturn true\n")
		}
	}
	e.printf("\t}\n")
	e.printf("\treturn false\n")
	e.printf("}\n\n")
}

// coopSegment emits one segment's statements with locals redirected to the
// frame.
func (e *emitter) coopSegment(stmts []ir.Stmt, locals []*ir.Local, depth int) {
	prev := e.frameLocals
	e.frameLocals = map[*ir.Local]bool{}
	for _, l := range locals {
		e.frameLocals[l] = true
	}
	defer func() { e.frameLocals = prev }()

	if len(stmts) == 0 {
		// A segment can be empty: two adjacent barriers, or a barrier first.
		// Go needs no statement here, and emitting one would be noise.
		return
	}
	for _, s := range stmts {
		e.stmt(s, depth)
	}
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
	segs, err := coopSegments(k)
	if err != nil {
		e.fail("kernel %s: %v", k.Name, err)
		return
	}
	e.coopLowering(k, segs)

	e.printf("// %s is the compiled form of %s.\n", k.Name+"Kernel", k.Name)
	e.printf("var %s = accel.Kernel{\n", k.Name+"Kernel")
	e.printf("\tName: %q,\n", k.Name)
	e.printf("\tWorkgroupSize: accel.ID3{X: %d, Y: %d, Z: %d},\n",
		k.Workgroup[0], k.Workgroup[1], k.Workgroup[2])
	e.printf("\tBindings: []accel.KernelBinding{\n")
	for _, b := range k.Bindings {
		e.printf("\t\t{Name: %q, DType: %s, Access: %s},\n", b.Name, e.dtype(b.Type), access(b))
	}
	e.printf("\t},\n")
	e.printf("\tDigest: %q,\n", k.Digest)
	e.printf("\tGenerator: accel.KernelABIVersion,\n")
	e.printf("\tSuspensions: %d,\n", len(segs)-1)
	if len(k.Shared) > 0 {
		e.printf("\tNewShared: func() []any {\n")
		for i, sh := range k.Shared {
			e.printf("\t\tvar s%d %s\n", i, e.goType(sh.Type))
			e.printf("\t\taccel.KernelPoison(s%d[:])\n", i)
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
	if len(k.Uniforms) > 0 {
		e.printf("\tUniforms: []accel.KernelUniform{\n")
		for _, u := range k.Uniforms {
			e.printf("\t\t{Name: %q, Type: %q, Size: %d},\n", u.Name, u.TypeName, u.Size)
		}
		e.printf("\t},\n")
	}

	// The entry point allocates one frame per invocation on first call and
	// resumes it afterwards. The scheduler owns the frames, so it passes an
	// opaque slot rather than the kernel keeping state of its own: two
	// workgroups run concurrently and a package-level frame would alias them.
	e.printf("\tCooperative: func(t accel.Thread, a accel.KernelArgs, slot *accel.KernelFrame) bool {\n")
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
			e.printf(", accel.KernelSlice[%s](a, %d)", e.goType(p.Type().Elem), slot)
			slot++
		case p.Type() != nil && p.Type().Kind == ir.Array:
			// A pointer, so every invocation of the workgroup addresses one
			// copy. By value each would get its own, which compiles.
			e.printf(", accel.KernelShared[%s](a, %d)", e.goType(p.Type()), sharedSlot)
			sharedSlot++
		default:
			e.printf(", accel.KernelUniformValue[%s](a, %d)", e.goType(p.Type()), uniformSlot)
			uniformSlot++
		}
	}
	e.printf(", f)\n")
	e.printf("\t},\n")
	e.printf("}\n\n")
}
