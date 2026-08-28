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
	"math"
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
const ABIVersion = 3

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

	// subgroupSize is the emulated lane count, which the CPU backend reports
	// and a caller may set: specs/006-backends.md section 5 makes it an option
	// so that a kernel can be swept across the sizes real hardware has.
	subgroupSize uint32

	// rendezvous is what [Thread.Barrier] calls, or nil.
	//
	// Nil in the generated lowering, where a barrier is a suspension the
	// scheduler handles and this method is never reached. Non-nil only when the
	// *authored* function is run directly as a reference, which needs a real
	// rendezvous: see [RunAuthored].
	rendezvous func()

	// subRendezvous is what [Thread.SubgroupBarrier] calls, and it is a
	// separate one because it releases a different set of lanes.
	//
	// A subgroup barrier that reused the workgroup's rendezvous would release
	// the wrong set of lanes. See [RunAuthored] for what that does and does
	// not change: with retirement on exit it does not deadlock, and no test
	// distinguishes the two — which is recorded there rather than assumed away.
	subRendezvous func()
}

// NewThread builds one invocation's [Thread]. It is for the backend and the
// harness that drive generated kernels, never for a kernel body.
func NewThread(global, local, group, groupSize, groupCount ID3) Thread {
	return Thread{global: global, local: local, group: group, groupSize: groupSize, groupCount: groupCount}
}

// NewThreadWithSubgroup is [NewThread] with the emulated lane count, for the
// backend and the sweeps that vary it.
func NewThreadWithSubgroup(global, local, group, groupSize, groupCount ID3, subgroup uint32) Thread {
	t := NewThread(global, local, group, groupSize, groupCount)
	t.subgroupSize = subgroup
	return t
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

// WorkgroupSize is the declared workgroup extent, per axis.
//
// It is the `accel:kernel workgroup=` directive's value, so it is a
// compile-time constant rather than a read: a backend lowers it to a literal,
// and a loop bounded by it stays compile-time uniform. That is what makes it
// worth having over the uniform field a kernel would otherwise carry, because a
// uniform's value is not uniform to the *compiler* and a barrier inside a loop
// bounded by one cannot be proved to be reached by every invocation.
//
// specs/052-dispatch-shape.md.
func (t Thread) WorkgroupSize() ID3 { return t.groupSize }

// NumGroups is how many workgroups this dispatch has, per axis.
//
// The grid the recorder set, which is a dispatch parameter rather than a
// compile-time one: two dispatches of one pipeline differ here.
func (t Thread) NumGroups() ID3 { return t.groupCount }

// GlobalSize is [Thread.WorkgroupSize] times [Thread.NumGroups], per axis.
//
// Derived rather than passed. Two numbers that must agree eventually disagree,
// and the multiplication is free.
func (t Thread) GlobalSize() ID3 {
	return ID3{
		X: t.groupSize.X * t.groupCount.X,
		Y: t.groupSize.Y * t.groupCount.Y,
		Z: t.groupSize.Z * t.groupCount.Z,
	}
}

// Barrier synchronises a workgroup.
//
// # Why its body does nothing
//
// A barrier's meaning is the workgroup scheduler's, not this function's. What
// actually runs is the generated resumable lowering, where the barrier is a
// return to the scheduler and a resume afterwards; the authored function is
// type-checking input, and this method exists so that input names something.
//
// It does nothing rather than panicking so that the authored function can be
// run as a reference, which is what spec 004's fifth testing level compares the
// generated lowering against. A panic here would make a cooperative kernel the
// one kind whose authored form nothing can execute, and an unexecutable
// reference is not a reference.
//
// Calling it on a thread with no rendezvous does nothing, which is the case in
// generated code. [RunAuthored] supplies one, which is what makes the authored
// function runnable as a reference — spec 004's fifth testing level compares the
// generated lowering against it, and an unexecutable reference is not a
// reference.
func (t Thread) Barrier() {
	if t.rendezvous != nil {
		t.rendezvous()
	}
}

// BarrierShared synchronises a workgroup and makes **shared** writes visible.
//
// specs/002-compute-model.md §2.5. It is the cheaper half of [Thread.Barrier]:
// a storage memory barrier costs materially more than a shared one on tiled
// mobile hardware, and a tiled GEMM runs two barriers per k-tile in its
// innermost structure.
//
// # It is not a faster Barrier
//
// §2.4's first consequence: a barrier orders the class it names and nothing
// more. This between a storage write and a peer's storage read is a **data
// race**, and it is a race that works on every desktop GPU because their caches
// are coherent enough to hide it. Reach for [Thread.Barrier] unless the shared
// array is the only thing being published.
//
// Its execution half is identical, so the rendezvous is the same one: what
// differs is which memory the backend is told to make visible, which on the CPU
// is nothing (one address space, and the scheduler's epochs order everything)
// and on Metal is a narrower mem_flags mask.
func (t Thread) BarrierShared() {
	if t.rendezvous != nil {
		t.rendezvous()
	}
}

// BarrierStorage synchronises a workgroup and makes **storage** writes visible.
//
// specs/002-compute-model.md §2.5. The counterpart to [Thread.BarrierShared]:
// a lane publishes through a buffer and its peers read what it wrote. Shared
// writes are not ordered by it.
func (t Thread) BarrierStorage() {
	if t.rendezvous != nil {
		t.rendezvous()
	}
}

// SubgroupBarrier synchronises one **subgroup**, and makes shared and storage
// writes visible across it.
//
// specs/002-compute-model.md §2.5 and §5.3, capability-gated on
// `subgroup_basic`. It is a barrier, so §3.1 applies at subgroup scope: every
// lane of the subgroup must reach it. That is a weaker requirement than
// [Thread.Barrier]'s, and it is the whole reason the call exists — control flow
// predicated on `SubgroupIndex` is legal around this and illegal around a
// workgroup barrier.
//
// # It is not a cheap Barrier
//
// The lanes of *one* subgroup rendezvous, not the workgroup. A kernel that
// wrote a shared array from every lane and read it back across this one is
// reading what its own subgroup wrote and whatever the other subgroups happened
// to have done, which on hardware is a race that looks correct at a subgroup
// size equal to the workgroup — the CPU backend's degenerate default, which is
// exactly why [Options.SubgroupSize] exists to sweep it.
func (t Thread) SubgroupBarrier() {
	if t.subRendezvous != nil {
		t.subRendezvous()
	}
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

// Uniform is one by-value parameter a kernel declares.
//
// It is separate from [Binding] because a caller supplies it differently: a
// binding is a slice and a uniform is a value, and the record says which is
// which so a mismatched argument set is refused rather than reinterpreted.
type Uniform struct {
	Name string
	Type string

	// Size is the encoded std140 block size in bytes.
	Size int

	// Encode writes v into dst in std140 layout, and reports an error rather
	// than panicking when v is the wrong type.
	//
	// # Why the record carries this at all
	//
	// A GPU backend holds a by-value parameter as an any and needs the bytes the
	// device reads. The generated codec knows how, but it is a named type a
	// caller writes (ScaleParamsCodec), and a backend has no route to it: the
	// alternatives were reflecting over the Go struct, which would put a second
	// std140 implementation beside the generated one, and asking callers to
	// encode, which moves a compiler-owned fact into code written by hand. So
	// generation, which already knows the layout, closes over the codec here.
	//
	// Nil for a kernel generated before this field existed, and a backend that
	// needs it says so by name rather than encoding zeros.
	Encode func(dst []byte, v any) error
}

// Args carries the host slices a generated kernel runs over.
//
// Host slices, not device buffers: the direct executor this feeds is compiler
// bring-up and not a submission API, and there is no graph to submit to until
// M3.
type Args struct {
	Slices []any

	// Uniforms are the by-value parameters, in signature order.
	Uniforms []any

	// Shared is the workgroup-shared storage, in signature order. One entry per
	// declared array, and the *same* backing for every invocation of one
	// workgroup: that is what "shared" means, and handing each invocation its
	// own copy compiles and computes something else.
	//
	// The scheduler allocates it per workgroup rather than per dispatch,
	// because two workgroups sharing storage would be a hazard no barrier
	// covers -- specs/002-compute-model.md section 2.7 gives no ordering
	// between workgroups at all.
	Shared []any
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

	// Uniforms are the by-value parameters, in signature order.
	Uniforms []Uniform

	// Flat runs one invocation. It is nil for a cooperative kernel, which has no
	// direct-call form by construction.
	Flat func(t Thread, a Args)

	// Cooperative runs one invocation to its next suspension point and reports
	// whether it suspended. It is nil for a flat kernel.
	//
	// The frame is the scheduler's, not the kernel's: two workgroups run
	// concurrently, and a kernel keeping state of its own would alias them.
	Cooperative func(t Thread, a Args, f *Frame) bool

	// NewShared allocates this kernel's workgroup-shared storage, poisoned.
	//
	// Generated, because only the generated code knows each array's element
	// type and extent; the runtime would need reflection to do it. Called once
	// per workgroup, since two workgroups sharing storage would be a hazard no
	// barrier covers.
	NewShared func() []any

	// Caps is every capability this kernel's body implies, as a bit set matching
	// accel.Capability. Inferred by the compiler from what the body reaches,
	// never declared by its author.
	Caps uint32

	// OrderIndependent reports that this kernel's observable result cannot
	// depend on the order its workgroups run in, which is what lets the CPU
	// backend run them at once.
	//
	// # Why it is not simply true
	//
	// specs/002-compute-model.md section 2.7 gives no ordering between
	// workgroups, so a workgroup that read what another wrote is already
	// undefined on every target. What is *defined* between workgroups is an
	// atomic, and every atomic accel offers returns the value the location held
	// before the operation. That return is the order dependence, and it is
	// there even for the operations whose accumulator is associative: after
	//
	//	out[t.GlobalID().X] = accel.AddU32(counter, 0, 1)
	//
	// the counter holds the same total whatever order the workgroups ran in,
	// and out holds a different permutation for each one. So the property is
	// per kernel and not per operation, and the compiler infers it as the
	// absence of any atomic rather than as a judgement about which atomics are
	// commutative.
	//
	// # Why false is the safe value
	//
	// A kernel generated before this field existed carries false, runs on one
	// worker, and produces exactly the bytes it produced before. The cost of a
	// stale generated file is therefore the speed, never the answer, which is
	// why this is inferred metadata rather than something an author writes:
	// an author who forgot it would only be slow, and an author who wrote it
	// wrongly would be silently wrong.
	OrderIndependent bool

	// SharedSizes is each declared shared array's element count, in signature
	// order. The tracker needs it to size its shadow bits, and only generation
	// knows it.
	SharedSizes []int

	// SharedBytes is what every declared shared array occupies together, which
	// is what a device's shared-memory budget is expressed in.
	//
	// Separate from SharedSizes because that holds element *counts* and a count
	// cannot be converted without the element size, which the record did not
	// carry. specs/016-graph-execution.md's validation rule V11 was stated and
	// could not fire for exactly that reason: the number it compares against
	// the device budget did not exist.
	SharedBytes int

	// Suspensions is how many barriers the body reaches, which is what the
	// scheduler uses to size an epoch bound. Zero for a flat kernel.
	Suspensions int

	// MSL is the generated Metal Shading Language source, and is empty when
	// this kernel was not generated for that target.
	//
	// A field rather than a member of a TargetArtifacts struct, which is what
	// specs/004-kernel-authoring.md draws. M2 flattened that struct into fields
	// here and specs/021-metal-bringup.md section 6 records the divergence
	// rather than reintroducing a wrapper for one string; the shape is settled
	// or ratified when a third target arrives.
	//
	// Empty is not a fallback. A Metal dispatch of a kernel with no MSL is an
	// error naming the kernel, because running the Go lowering on a device the
	// caller selected specifically would be correct, fast enough to miss, and
	// would mean the GPU was never exercised.
	MSL string
}

// Poison fills workgroup-shared storage with a pattern no sensible kernel
// computes.
//
// Not zero. Zero is a value a kernel legitimately expects, so a read before a
// write would return something plausible and the mistake would survive every
// test. This makes the same mistake produce a number nobody mistakes for an
// answer, which is what specs/002-compute-model.md requires of this backend.
//
// specs/019-cooperative-diagnostics.md makes the read itself *detected*, which
// is stronger and is the reason this is not the whole answer: a sentinel is a
// value a kernel could compute, so a check comparing against one either misses
// the read or fires on a correct kernel.
func Poison[T any](s []T) {
	var zero T
	switch any(zero).(type) {
	case float32:
		p := any(math.Float32frombits(poisonBits)).(T)
		for i := range s {
			s[i] = p
		}
	case uint32:
		p := any(uint32(poisonBits)).(T)
		for i := range s {
			s[i] = p
		}
	case int32:
		p := any(int32(poisonBits)).(T)
		for i := range s {
			s[i] = p
		}
	case Float16:
		// A quiet NaN in the narrow format, so a read before a write propagates
		// through the widening rather than arriving as a plausible number.
		p := any(Float16FromBits(quietNaNF16)).(T)
		for i := range s {
			s[i] = p
		}
	case BFloat16:
		p := any(BFloat16FromBits(quietNaNBF16)).(T)
		for i := range s {
			s[i] = p
		}
	}
}

// poisonBits is a quiet NaN when read as an f32, so arithmetic on it propagates
// rather than producing a plausible number, and a value with bits set in every
// byte when read as an integer.
const poisonBits = 0x7FC0DEAD

// SharedSlice recovers one workgroup-shared array from an argument set.
//
// It returns a pointer, because a workgroup shares one copy and a value would
// give each invocation its own.
func SharedSlice[T any](a Args, i int) *T {
	if i < 0 || i >= len(a.Shared) {
		return nil
	}
	p, _ := a.Shared[i].(*T)
	return p
}

// Frame is one invocation's saved state between suspension points.
//
// Its contents belong to the generated lowering, which is the only thing that
// knows what a particular kernel has to carry across a barrier. The scheduler
// owns the slot and hands the same one back to the same invocation.
type Frame struct {
	// State is the generated frame, allocated by the kernel on its first call.
	State any

	// Done reports that this invocation has run to completion.
	Done bool

	// Shared is the workgroup's shared-memory tracker, or nil in strict mode
	// where the instrumentation costs nothing because every call is a no-op the
	// compiler removes.
	Shared *SharedTracker

	// Pass is a hand-written kernel's program counter, for the tests that drive
	// the scheduler without going through the generator. A generated lowering
	// keeps its own counter inside State.
	Pass int

	// Sub is the subgroup rendezvous this invocation is suspended at, or
	// SubNone for an ordinary barrier.
	//
	// A subgroup operation needs every lane's contribution at the point of the
	// call, and the scheduler advances one invocation at a time -- so it is a
	// suspension like a barrier, with the contribution travelling in the fields
	// below and the result arriving in them on resume.
	Sub SubgroupOp

	// SubF32 carries a float contribution in and the result out. SubBool and
	// SubMask do the same for the predicate and ballot operations.
	SubF32  float32
	SubBool bool
	SubMask Mask

	// SubLane is the second operand of a lane-addressed read: which lane to
	// read for a broadcast or a shuffle, the mask for a butterfly shuffle, the
	// delta for a shuffle up or down.
	//
	// It travels beside the value rather than being recomputed by the
	// scheduler, because it is an expression the kernel wrote and only the
	// generated lowering can evaluate it.
	SubLane uint32

	// Barrier is which suspension point this invocation stopped at, set by the
	// generated lowering before it returns true. Every active invocation must
	// report the same one at each rendezvous.
	//
	// A stable id rather than a count of arrivals: counting live invocations
	// against the number blocked catches one way an arrival becomes impossible
	// and not the only one, so it misses the case where an invocation is still
	// running and will never arrive. Keying on identity also makes reaching A
	// while a peer waits at B a reported mismatch rather than a silent pairing.
	// See specs/002-compute-model.md section 3.4.
	Barrier BarrierID
}

// BarrierID identifies one suspension point in a kernel.
//
// It carries the source position as well as an index, because a report saying
// two invocations reached different barriers is only useful if a reader can see
// which two lines they are.
type BarrierID struct {
	Index int
	Pos   string

	// Subgroup marks a barrier whose scope is one subgroup rather than the
	// workgroup, specs/002-compute-model.md §5.3.
	//
	// It changes who has to arrive. The arrival check compares every active
	// invocation against one expected barrier, which is right for a workgroup
	// barrier and wrong here: lanes of *different* subgroups are under no
	// obligation to be at the same subgroup barrier, and control predicated on
	// SubgroupIndex -- which the acceptor allows around this and refuses
	// around a workgroup barrier -- puts them at different ones on purpose.
	Subgroup bool
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
		if !matches(a.Slices[i], b.DType) {
			return fmt.Errorf("accel: kernel %q binding %d (%q, %v) takes %s and got %T",
				k.Name, i, b.Name, b.DType, goTypeFor(b.DType), a.Slices[i])
		}
	}

	// A uniform's type cannot be checked structurally here, because the record
	// names it as a string and the value arrives as an any. What is checked is
	// the count, which is what catches a caller who supplied them in the wrong
	// order or omitted one; the type itself is checked where the generated
	// entry point recovers it, and that call names the type.
	if len(a.Uniforms) != len(k.Uniforms) {
		return fmt.Errorf("accel: kernel %q takes %d uniforms and got %d",
			k.Name, len(k.Uniforms), len(a.Uniforms))
	}
	return nil
}

// matches reports whether a bound argument is the slice its dtype names.
//
// A type switch rather than comparing formatted type names. This runs once per
// binding per dispatch, and formatting a type to compare it allocates on the
// path that succeeds, which is every path that matters. The name is built only
// to explain a failure.
func matches(v any, d DType) bool {
	switch d {
	case F32:
		_, ok := v.([]float32)
		return ok
	case F16:
		_, ok := v.([]Float16)
		return ok
	case BF16:
		_, ok := v.([]BFloat16)
		return ok
	case I32:
		_, ok := v.([]int32)
		return ok
	case U32:
		_, ok := v.([]uint32)
		return ok
	case I8:
		_, ok := v.([]int8)
		return ok
	case U8:
		_, ok := v.([]uint8)
		return ok
	}
	return false
}

// goTypeFor is the host slice type a dtype binds to.
func goTypeFor(d DType) string {
	switch d {
	case F32:
		return "[]float32"
	case F16:
		return "[]accel.Float16"
	case BF16:
		return "[]accel.BFloat16"
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

// UniformValue recovers one by-value parameter for a generated entry point.
//
// It names the type in its failure, which is the check [Kernel.Bind] cannot
// make: the record carries a uniform's type as a string, and only the generated
// code knows the Go type it was generated for.
func UniformValue[T any](a Args, i int) T {
	if i < 0 || i >= len(a.Uniforms) {
		panic(fmt.Sprintf("accel: uniform %d of %d requested: Kernel.Bind was not called", i, len(a.Uniforms)))
	}
	v, ok := a.Uniforms[i].(T)
	if !ok {
		panic(fmt.Sprintf("accel: uniform %d is %T, not %T", i, a.Uniforms[i], v))
	}
	return v
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
