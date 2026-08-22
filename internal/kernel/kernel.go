// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package kernel is the vocabulary a compiled kernel executes in.
//
// It holds what a kernel author names on the root accel package ([ID3] and
// [Thread], aliased there) and what a generated kernel carries with it ([Kernel]
// and [Binding]).
//
// It lives below accel because of who has to construct a [Thread]. From M3 the
// CPU backend executes generated kernels, and that backend cannot import accel:
// accel links it in. So the type is declared here and accel aliases it, which
// also makes the two spellings one type. Generated code in an author's package
// says accel.Thread and the entry point it fills in says kernel.Thread, and
// those compose only because the alias makes them identical. See
// specs/012-kernel-pipeline.md section 3.
package kernel

import (
	"fmt"
	"strings"
)

// ABIVersion is the contract between a generated kernel and the runtime that
// executes it.
//
// It participates in the kernel digest, so a change here makes every generated
// file stale rather than making a mismatched one run. That is the point: a
// generated adapter compiled against one shape of this package and loaded by
// another is a wrong-answer bug, and there is no version of it that is a
// compile error.
const ABIVersion = 1

// ID3 is a three-dimensional invocation identifier.
//
// Ids are three-dimensional rather than scalar because a two-dimensional shared
// tile cannot be addressed from a scalar id without index arithmetic the
// compiler then cannot prove uniform. The components are uint32 so that `l < s`
// and `tile[l+s]` type-check without a conversion, which is also where the GLSL
// integer-literal divergence is settled: go/types reports each literal's
// resolved type and the emitter spells it accordingly. See
// specs/002-compute-model.md section 1 and specs/004-kernel-authoring.md.
type ID3 struct{ X, Y, Z uint32 }

// Thread carries an invocation's identity, and from M4 the CPU backend's
// rendezvous state.
//
// It is a kernel's first parameter. Its fields are unexported because an id a
// caller can set is an id the backend does not own; a kernel body cannot
// construct one either, since composite literals of it are outside the subset.
type Thread struct {
	global, local, group ID3
	groupSize            ID3 // workgroup extents, for the linear forms
	groupCount           ID3 // grid extents, for the linear forms
}

// NewThread builds one invocation's [Thread]. It is for the backend and the
// harness that drive generated kernels, never for a kernel body.
func NewThread(global, local, group, groupSize, groupCount ID3) Thread {
	return Thread{global: global, local: local, group: group, groupSize: groupSize, groupCount: groupCount}
}

// GlobalID is this invocation's position in the grid.
func (t Thread) GlobalID() ID3 { return t.global }

// LocalID is this invocation's position within its workgroup.
func (t Thread) LocalID() ID3 { return t.local }

// GroupID is this workgroup's position in the grid.
func (t Thread) GroupID() ID3 { return t.group }

// GlobalIndex is the workgroup-contiguous linearization of [Thread.GlobalID].
//
// Workgroup-contiguous, not grid-linearized: it is GroupIndex*invocations +
// LocalIndex, so one workgroup's invocations occupy a contiguous range. The two
// definitions agree only when the grid is one workgroup wide, and the
// contiguous one is what a reduction over a workgroup's slice of a buffer
// needs. See specs/002-compute-model.md section 1.4.
func (t Thread) GlobalIndex() uint32 {
	return t.GroupIndex()*linear(t.groupSize) + t.LocalIndex()
}

// LocalIndex is the x-fastest linearization of [Thread.LocalID].
func (t Thread) LocalIndex() uint32 {
	return t.local.X + t.groupSize.X*(t.local.Y+t.groupSize.Y*t.local.Z)
}

// GroupIndex is the x-fastest linearization of [Thread.GroupID].
func (t Thread) GroupIndex() uint32 {
	return t.group.X + t.groupCount.X*(t.group.Y+t.groupCount.Y*t.group.Z)
}

// Barrier synchronises a workgroup.
//
// It exists at M2 so that a kernel using it is rejected by name and position,
// saying when barriers arrive, rather than failing as a call to a method that
// plainly exists. Its body is never what runs: a barrier's meaning is the
// workgroup scheduler's, and a flat kernel has no scheduler to rendezvous with.
func (t Thread) Barrier() {
	panic("accel: a barrier only has meaning inside a kernel executed by a backend; " +
		"cooperative kernels arrive at M4 (specs/009-sequencing.md)")
}

// linear is an extent's invocation count.
func linear(e ID3) uint32 { return max(e.X, 1) * max(e.Y, 1) * max(e.Z, 1) }

// DType mirrors accel.DType. It crosses into generated code, which names the
// dtype of every binding it declares.
type DType int

const (
	F32 DType = iota
	F16
	BF16
	I32
	U32
	I8
	U8
)

var dtypeNames = [...]string{F32: "f32", F16: "f16", BF16: "bf16", I32: "i32", U32: "u32", I8: "i8", U8: "u8"}

func (d DType) String() string {
	if d < 0 || int(d) >= len(dtypeNames) {
		return fmt.Sprintf("DType(%d)", int(d))
	}
	return dtypeNames[d]
}

// Access is how a kernel touches a binding, inferred from the source rather
// than declared by its author.
type Access uint8

const (
	Read Access = 1 << iota
	Write
)

func (a Access) String() string {
	switch a {
	case Read:
		return "read"
	case Write:
		return "write"
	case Read | Write:
		return "read-write"
	}
	return "no access"
}

// Binding is one resource a kernel's signature declares.
//
// The signature is the binding layout, so every field here is inferred from the
// source and none of it is written by hand. A caller who could disagree with it
// would be a second source of truth for something the compiler already knows.
type Binding struct {
	Name   string
	DType  DType
	Access Access
}

// Args carries the host slices a generated kernel runs over.
//
// Host slices, not device buffers: the direct executor this feeds is compiler
// bring-up and not a submission API, and there is no graph to submit to until
// M3.
type Args struct {
	Slices []any
}

// Kernel is everything generation inferred about one kernel, plus the entry
// point that runs it.
type Kernel struct {
	Name          string
	WorkgroupSize ID3
	Bindings      []Binding

	// Digest identifies the source this was generated from, and Generator the
	// ABI it was generated against. Freshness compares the first; the second
	// makes a stale adapter refuse to run rather than run differently.
	Digest    string
	Generator int

	// Flat runs one invocation. It is nil for a cooperative kernel, which has no
	// direct-call form by construction.
	Flat func(t Thread, a Args)
}

// Bind checks a whole argument set against the declared bindings, once.
//
// Once, before the invocation loop, and not inside it. The signature is the
// binding layout, so a dtype mismatch is a thing generation already proved;
// rechecking it per invocation reopens at runtime what the compiler settled and
// reports it once per invocation rather than once, naming the binding.
func (k *Kernel) Bind(a Args) error {
	if k.Generator != ABIVersion {
		return fmt.Errorf("accel: kernel %q was generated against ABI %d and this runtime is ABI %d; "+
			"re-run go generate", k.Name, k.Generator, ABIVersion)
	}
	if len(a.Slices) != len(k.Bindings) {
		return fmt.Errorf("accel: kernel %q takes %d bindings and got %d",
			k.Name, len(k.Bindings), len(a.Slices))
	}
	for i, b := range k.Bindings {
		want := goTypeFor(b.DType)
		got := fmt.Sprintf("%T", a.Slices[i])
		if got != want {
			return fmt.Errorf("accel: kernel %q binding %d (%q, %v) takes %s and got %s",
				k.Name, i, b.Name, b.DType, want, got)
		}
	}
	return nil
}

// goTypeFor is the host slice type a dtype binds to.
func goTypeFor(d DType) string {
	switch d {
	case F32:
		return "[]float32"
	case F16, BF16:
		return "[]uint16"
	case I32:
		return "[]int32"
	case U32:
		return "[]uint32"
	case I8:
		return "[]int8"
	case U8:
		return "[]uint8"
	}
	return "an unknown slice type"
}

// Slice recovers one bound argument. [Kernel.Bind] has already proved the type,
// so this cannot fail for a bound argument set and panics rather than returning
// an error a generated caller would have to handle at every binding.
func Slice[T any](a Args, i int) []T {
	s, ok := a.Slices[i].([]T)
	if !ok {
		panic(fmt.Sprintf("accel: binding %d is %T, not %T: Kernel.Bind was not called", i, a.Slices[i], s))
	}
	return s
}

// String describes a kernel for a diagnostic.
func (k *Kernel) String() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s workgroup=%d,%d,%d", k.Name, k.WorkgroupSize.X, k.WorkgroupSize.Y, k.WorkgroupSize.Z)
	for _, bind := range k.Bindings {
		fmt.Fprintf(&b, " %s:%v/%v", bind.Name, bind.DType, bind.Access)
	}
	return b.String()
}
