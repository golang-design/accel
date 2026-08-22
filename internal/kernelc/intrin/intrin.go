// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package intrin resolves an intrinsic by object identity.
//
// # Why identity and not name
//
// The predecessor keyed its builtin table by bare name, so any user function
// called Dot or Mix lowered to the GPU builtin. Nothing errors. The kernel
// compiles, runs, and computes something else, which is the worst failure a
// compiler can have because it is indistinguishable from working. The table
// here is keyed on what go/types resolved, so a same-named function in another
// package and a same-named method on another type are both simply not this
// intrinsic.
//
// For a method the key includes the **receiver type name**. Keying on (package
// path, method name) alone would be the same bare-name bug wearing a package
// prefix: a user type declared in a package that also holds an intrinsic would
// capture it.
//
// # Two identities, on purpose
//
// The key is the *resolved* identity, which for accel.Thread is
// internal/kernel.Thread because the root package aliases it. The digest records
// the *authored* spelling, accel.Thread.GlobalID, which is what a kernel author
// wrote. Keeping them independent is what stops relocating a type from
// invalidating every committed digest, and M4 grows Thread's rendezvous state,
// which is exactly when a relocation happens. See
// specs/012-kernel-pipeline.md section 3.
package intrin

import (
	"fmt"
	"go/types"
	"sort"
	"strings"

	"golang.design/x/accel/internal/kernelc/ir"
)

// ABIVersion versions the table's contents. It participates in the kernel
// digest, so adding, removing, or retyping an intrinsic makes every generated
// file stale rather than letting one generated against a different table run.
const ABIVersion = 1

// Stage is when an intrinsic becomes usable.
type Stage int

const (
	// Flat is available to a kernel with no shared memory, barriers, or subgroup
	// operations, which is everything M2 compiles.
	Flat Stage = iota

	// Cooperative needs the resumable lowering and the workgroup scheduler, which
	// arrive at M4. An intrinsic marked this way is in the table so that a kernel
	// using it is rejected by name, with a position, saying when it arrives,
	// rather than failing as an unknown call.
	Cooperative
)

func (s Stage) String() string {
	if s == Cooperative {
		return "cooperative"
	}
	return "flat"
}

// Uniformity is what an intrinsic's result is uniform across.
//
// It exists here rather than being inferred later because it is a property of
// the intrinsic and not of its use: GroupID is uniform across a workgroup and
// LocalID is not, and the barrier analysis at M4 is built on that distinction.
type Uniformity int

const (
	// PerInvocation varies between invocations of one workgroup.
	PerInvocation Uniformity = iota
	// PerWorkgroup is the same for every invocation of one workgroup.
	PerWorkgroup
)

// Intrinsic is one table entry.
type Intrinsic struct {
	// Authored is what a kernel author wrote, and what the digest records.
	Authored string

	Op         ir.Opcode
	Stage      Stage
	Uniformity Uniformity

	// Result is the kind the call produces.
	Result ir.Kind

	// Params is how many arguments the call takes, not counting the receiver.
	Params int
}

// key is the resolved identity. Recv is empty for a free function.
type key struct{ pkg, recv, name string }

func (k key) String() string {
	if k.recv == "" {
		return k.pkg + "." + k.name
	}
	return k.pkg + "." + k.recv + "." + k.name
}

// kernelPkg is where the aliased types actually live. The authored spelling
// differs, which is the whole point of recording both.
const kernelPkg = "golang.design/x/accel/internal/kernel"

var table = map[key]*Intrinsic{
	{kernelPkg, "Thread", "GlobalID"}: {
		Authored: "accel.Thread.GlobalID", Op: ir.OpGlobalID,
		Uniformity: PerInvocation, Result: ir.ID3Kind,
	},
	{kernelPkg, "Thread", "LocalID"}: {
		Authored: "accel.Thread.LocalID", Op: ir.OpLocalID,
		Uniformity: PerInvocation, Result: ir.ID3Kind,
	},
	{kernelPkg, "Thread", "GroupID"}: {
		Authored: "accel.Thread.GroupID", Op: ir.OpGroupID,
		Uniformity: PerWorkgroup, Result: ir.ID3Kind,
	},
	{kernelPkg, "Thread", "GlobalIndex"}: {
		Authored: "accel.Thread.GlobalIndex", Op: ir.OpGlobalIndex,
		Uniformity: PerInvocation, Result: ir.U32,
	},
	{kernelPkg, "Thread", "LocalIndex"}: {
		Authored: "accel.Thread.LocalIndex", Op: ir.OpLocalIndex,
		Uniformity: PerInvocation, Result: ir.U32,
	},
	{kernelPkg, "Thread", "GroupIndex"}: {
		Authored: "accel.Thread.GroupIndex", Op: ir.OpGroupIndex,
		Uniformity: PerWorkgroup, Result: ir.U32,
	},

	// Recognized and not available. Being in the table is what makes the
	// rejection say "barriers arrive at M4" at the right line, rather than
	// leaving a kernel author with an unknown-call error about a method that
	// plainly exists.
	{kernelPkg, "Thread", "Barrier"}: {
		Authored: "accel.Thread.Barrier", Op: ir.OpBarrier, Stage: Cooperative,
		Uniformity: PerWorkgroup, Result: ir.Invalid,
	},
}

// Lookup resolves a func object to an intrinsic.
//
// It takes the object go/types resolved, never a name, which is what makes a
// user function that shares a name simply not found rather than silently
// captured.
func Lookup(fn *types.Func) (*Intrinsic, bool) {
	if fn == nil || fn.Pkg() == nil {
		return nil, false
	}
	k := key{pkg: fn.Pkg().Path(), name: fn.Name()}
	if sig, ok := fn.Type().(*types.Signature); ok && sig.Recv() != nil {
		k.recv = receiverName(sig.Recv().Type())
		if k.recv == "" {
			return nil, false
		}
	}
	in, ok := table[k]
	return in, ok
}

// receiverName is the receiver's named type, with any pointer stripped.
func receiverName(t types.Type) string {
	if p, ok := t.(*types.Pointer); ok {
		t = p.Elem()
	}
	n, ok := t.(*types.Named)
	if !ok || n.Obj() == nil {
		return ""
	}
	return n.Obj().Name()
}

// Names lists every intrinsic's authored spelling, sorted. It is what a
// diagnostic offers when something looks like an intrinsic and is not, and what
// a test uses to assert the table did not change silently.
func Names() []string {
	out := make([]string, 0, len(table))
	for _, in := range table {
		out = append(out, in.Authored)
	}
	sort.Strings(out)
	return out
}

// Digest is a stable summary of the table's contents.
//
// It goes into the kernel digest so that adding or retyping an intrinsic makes
// every generated file stale. A version number alone would not: it depends on
// somebody remembering to bump it, and the case that matters is the change
// nobody thought was ABI-visible.
func Digest() string {
	keys := make([]key, 0, len(table))
	for k := range table {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i].String() < keys[j].String() })

	var b strings.Builder
	fmt.Fprintf(&b, "intrin/%d\n", ABIVersion)
	for _, k := range keys {
		in := table[k]
		fmt.Fprintf(&b, "%s\t%s\t%v\t%v\t%v\t%d\n",
			in.Authored, k, in.Op, in.Stage, in.Result, in.Params)
	}
	return b.String()
}
