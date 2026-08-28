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
const ABIVersion = 9

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

// Class is what an intrinsic's result may be compared as.
//
// It is recorded here rather than derived at the test, because it is a property
// of the operation and not of the case: asserting a tolerance where bits are
// guaranteed hides a real difference, and asserting bits where only a bound
// holds produces a flaky test on the first backend that rounds differently.
type Class int

const (
	// ClassExact moves bits or does arithmetic every target agrees on exactly.
	ClassExact Class = iota
	// ClassBounded has a normative per-operation ceiling in spec 008 section 6.
	ClassBounded
)

func (c Class) String() string {
	if c == ClassBounded {
		return "bounded"
	}
	return "exact"
}

// Intrinsic is one table entry.
type Intrinsic struct {
	// Authored is what a kernel author wrote, and what the digest records.
	Authored string

	Op         ir.Opcode
	Stage      Stage
	Uniformity Uniformity

	// Result is the kind the call produces.
	Result ir.Kind

	// Class is how a result may be compared. See [Class].
	Class Class

	// Params is how many arguments the call takes, not counting the receiver.
	Params int

	// Cap is the capability this intrinsic requires, or zero.
	//
	// Inferred from the body through this table rather than declared by the
	// kernel's author: a declaration can be forgotten, and the failure is
	// silent -- a kernel using a feature the device lacks produces wrong
	// results rather than an error, because nothing checked. See
	// specs/020-cooperative-atomics.md section 3.
	Cap Capability
}

// Capability mirrors accel.Capability. It is declared here rather than imported
// because this package sits below the root, and the two are kept in step by a
// parity test.
type Capability uint32

const (
	CapSubgroupBasic Capability = 1 << iota
	CapSubgroupVote
	CapSubgroupBallot
	CapSubgroupShuffle
	CapSubgroupArithmetic
	CapF16Arithmetic
	CapBF16Arithmetic
	CapAtomicFloatAddStorage
	CapAtomicFloatAddShared
	CapI8DotProduct
)

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

// kmathPkg is where the bounded scalar math lives. It is a real import path a
// kernel author writes, not an alias target, so the authored and resolved
// spellings agree here where they do not for Thread.
const kmathPkg = "golang.design/x/accel/kmath"

// accelPkg is the root package, where the narrowing conversions live as
// ordinary functions rather than as aliases.
const accelPkg = "golang.design/x/accel"

var table = map[key]*Intrinsic{
	// The graphics stage built-ins. specs/032-stage-abi.md section 2.1 and 4.1.
	{kernelPkg, "Vertex", "VertexIndex"}: {
		Authored: "accel.Vertex.VertexIndex", Op: ir.OpVertexIndex,
		Uniformity: PerInvocation, Result: ir.U32,
	},
	{kernelPkg, "Vertex", "InstanceIndex"}: {
		Authored: "accel.Vertex.InstanceIndex", Op: ir.OpInstanceIndex,
		Uniformity: PerInvocation, Result: ir.U32,
	},
	{kernelPkg, "Fragment", "Coord"}: {
		Authored: "accel.Fragment.Coord", Op: ir.OpFragCoord,
		Uniformity: PerInvocation, Result: ir.Array,
	},
	{kernelPkg, "Fragment", "FrontFacing"}: {
		Authored: "accel.Fragment.FrontFacing", Op: ir.OpFrontFacing,
		Uniformity: PerInvocation, Result: ir.Bool,
	},
	// Integer texel fetch. A free function on the root package taking the
	// texture and two signed coordinates, for the reason an atomic takes a
	// buffer and an index: the shading languages name the resource and the
	// coordinate separately and there is no intermediate value to pass.
	//
	// Exact, and that is the whole argument for admitting it. A fetch is an
	// indexed load, so given one texture and one coordinate every backend
	// returns the same four floats and there is nothing to reconcile. A
	// filtered sample would be neither exact nor bounded -- half-texel
	// addressing, LOD rounding and per-tap integer lerps are per-vendor, so its
	// class would be "whatever this driver does" -- which is why
	// specs/032-stage-abi.md section 5 admits this one and refuses that one.
	//
	// No capability. Reading a texel from a float 2D texture is baseline on
	// every target this project emits for, so gating it would refuse kernels on
	// devices that can run them.
	{accelPkg, "", "Fetch"}: {
		Authored: "accel.Fetch", Op: ir.OpTexelFetch, Params: 3,
		Uniformity: PerInvocation, Result: ir.Array, Class: ClassExact,
	},

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

	// The dispatch shape, specs/052-dispatch-shape.md. All three are
	// PerWorkgroup -- in fact they are uniform across the whole dispatch, and
	// the lattice has no level above the workgroup because nothing needs one:
	// what a barrier's control flow requires is workgroup uniformity, and a
	// value uniform across more than that is uniform across it too.
	{kernelPkg, "Thread", "WorkgroupSize"}: {
		Authored: "accel.Thread.WorkgroupSize", Op: ir.OpWorkgroupSize,
		Uniformity: PerWorkgroup, Result: ir.ID3Kind,
	},
	{kernelPkg, "Thread", "NumGroups"}: {
		Authored: "accel.Thread.NumGroups", Op: ir.OpNumGroups,
		Uniformity: PerWorkgroup, Result: ir.ID3Kind,
	},
	{kernelPkg, "Thread", "GlobalSize"}: {
		Authored: "accel.Thread.GlobalSize", Op: ir.OpGlobalSize,
		Uniformity: PerWorkgroup, Result: ir.ID3Kind,
	},

	// accel/kmath, the bounded scalar math. Free functions rather than methods,
	// because they do not depend on the invocation, and in their own package so
	// that the key rejects a same-named function from anywhere else.
	{kmathPkg, "", "Sqrt"}:  {Authored: "accel/kmath.Sqrt", Op: ir.OpSqrt, Result: ir.F32, Params: 1, Class: ClassBounded},
	{kmathPkg, "", "RSqrt"}: {Authored: "accel/kmath.RSqrt", Op: ir.OpRSqrt, Result: ir.F32, Params: 1, Class: ClassBounded},
	// Saturating float-to-integer, specs/051-float-to-int.md. Exact rather than
	// bounded: the result is an integer and there is nothing to round, so the
	// two backends must agree bit for bit.
	{kmathPkg, "", "ToI32"}: {Authored: "accel/kmath.ToI32", Op: ir.OpToI32, Result: ir.I32, Params: 1, Class: ClassExact},
	{kmathPkg, "", "ToU32"}: {Authored: "accel/kmath.ToU32", Op: ir.OpToU32, Result: ir.U32, Params: 1, Class: ClassExact},
	{kmathPkg, "", "Exp"}:   {Authored: "accel/kmath.Exp", Op: ir.OpExp, Result: ir.F32, Params: 1, Class: ClassBounded},
	{kmathPkg, "", "Log"}:   {Authored: "accel/kmath.Log", Op: ir.OpLog, Result: ir.F32, Params: 1, Class: ClassBounded},
	{kmathPkg, "", "Sin"}:   {Authored: "accel/kmath.Sin", Op: ir.OpSin, Result: ir.F32, Params: 1, Class: ClassBounded},
	{kmathPkg, "", "Cos"}:   {Authored: "accel/kmath.Cos", Op: ir.OpCos, Result: ir.F32, Params: 1, Class: ClassBounded},
	{kmathPkg, "", "Tanh"}:  {Authored: "accel/kmath.Tanh", Op: ir.OpTanh, Result: ir.F32, Params: 1, Class: ClassBounded},

	// Exact rather than bounded: these move bits and do no arithmetic, so they
	// have no error to bound. Recording the class is what keeps a test from
	// asserting a tolerance where bits are guaranteed.
	{kmathPkg, "", "Abs"}: {Authored: "accel/kmath.Abs", Op: ir.OpAbs, Result: ir.F32, Params: 1, Class: ClassExact},
	{kmathPkg, "", "Min"}: {Authored: "accel/kmath.Min", Op: ir.OpMin, Result: ir.F32, Params: 2, Class: ClassExact},
	{kmathPkg, "", "Max"}: {Authored: "accel/kmath.Max", Op: ir.OpMax, Result: ir.F32, Params: 2, Class: ClassExact},

	// Conversions between narrow storage and f32. Exact by spec 008 section 4:
	// widening cannot round, and narrowing has a stated rule rather than a
	// tolerance.
	{kernelPkg, "Float16", "F32"}: {
		Authored: "accel.Float16.F32", Op: ir.OpF16ToF32, Result: ir.F32, Class: ClassExact,
	},
	{kernelPkg, "BFloat16", "F32"}: {
		Authored: "accel.BFloat16.F32", Op: ir.OpBF16ToF32, Result: ir.F32, Class: ClassExact,
	},
	{accelPkg, "", "ToFloat16"}: {
		Authored: "accel.ToFloat16", Op: ir.OpF32ToF16, Result: ir.F16, Params: 1, Class: ClassExact,
	},
	{accelPkg, "", "ToBFloat16"}: {
		Authored: "accel.ToBFloat16", Op: ir.OpF32ToBF16, Result: ir.BF16, Params: 1, Class: ClassExact,
	},

	// Atomics. Free functions on the root package taking a buffer and an index,
	// because GLSL cannot form a pointer into a buffer. Each returns the
	// previous value, and each is exact: they move integers, or in AddF32's
	// case do one f32 addition whose rounding every target agrees on. What is
	// not deterministic about a float atomic is the *order* several of them run
	// in, which is a property of a reduction rather than of the operation.
	{accelPkg, "", "AddU32"}: {Authored: "accel.AddU32", Op: ir.OpAtomicAddU32, Result: ir.U32, Params: 3, Class: ClassExact},
	{accelPkg, "", "AddI32"}: {Authored: "accel.AddI32", Op: ir.OpAtomicAddI32, Result: ir.I32, Params: 3, Class: ClassExact},
	{accelPkg, "", "SubU32"}: {Authored: "accel.SubU32", Op: ir.OpAtomicSubU32, Result: ir.U32, Params: 3, Class: ClassExact},
	{accelPkg, "", "SubI32"}: {Authored: "accel.SubI32", Op: ir.OpAtomicSubI32, Result: ir.I32, Params: 3, Class: ClassExact},
	{accelPkg, "", "MinU32"}: {Authored: "accel.MinU32", Op: ir.OpAtomicMinU32, Result: ir.U32, Params: 3, Class: ClassExact},
	{accelPkg, "", "MinI32"}: {Authored: "accel.MinI32", Op: ir.OpAtomicMinI32, Result: ir.I32, Params: 3, Class: ClassExact},
	{accelPkg, "", "MaxU32"}: {Authored: "accel.MaxU32", Op: ir.OpAtomicMaxU32, Result: ir.U32, Params: 3, Class: ClassExact},
	{accelPkg, "", "MaxI32"}: {Authored: "accel.MaxI32", Op: ir.OpAtomicMaxI32, Result: ir.I32, Params: 3, Class: ClassExact},
	{accelPkg, "", "AndU32"}: {Authored: "accel.AndU32", Op: ir.OpAtomicAndU32, Result: ir.U32, Params: 3, Class: ClassExact},
	{accelPkg, "", "OrU32"}:  {Authored: "accel.OrU32", Op: ir.OpAtomicOrU32, Result: ir.U32, Params: 3, Class: ClassExact},
	{accelPkg, "", "XorU32"}: {Authored: "accel.XorU32", Op: ir.OpAtomicXorU32, Result: ir.U32, Params: 3, Class: ClassExact},

	{accelPkg, "", "ExchangeU32"}: {Authored: "accel.ExchangeU32", Op: ir.OpAtomicExchangeU32, Result: ir.U32, Params: 3, Class: ClassExact},
	{accelPkg, "", "ExchangeI32"}: {Authored: "accel.ExchangeI32", Op: ir.OpAtomicExchangeI32, Result: ir.I32, Params: 3, Class: ClassExact},

	{accelPkg, "", "CompareExchangeU32"}: {Authored: "accel.CompareExchangeU32", Op: ir.OpAtomicCompareExchangeU32, Result: ir.U32, Params: 4, Class: ClassExact},
	{accelPkg, "", "CompareExchangeI32"}: {Authored: "accel.CompareExchangeI32", Op: ir.OpAtomicCompareExchangeI32, Result: ir.I32, Params: 4, Class: ClassExact},

	// Ballot, specs/058-ballot.md. Its result is a mask rather than a scalar,
	// which is the one new kind the kernel subset gains for it, and the five
	// methods below are what a kernel asks the mask. They are keyed on the
	// Mask receiver, so a same-named method on a user type is simply not this
	// intrinsic.
	//
	// No capability on the methods: a mask cannot exist without the Ballot
	// that produced it, and that carries CapSubgroupBallot.
	{kernelPkg, "Thread", "SubgroupBallot"}: {
		Authored: "accel.Thread.SubgroupBallot", Op: ir.OpBallot, Params: 1,
		Result: ir.MaskKind, Class: ClassExact, Cap: CapSubgroupBallot,
	},
	{kernelPkg, "Mask", "Count"}: {
		Authored: "accel.KernelMask.Count", Op: ir.OpMaskCount,
		Result: ir.I32, Class: ClassExact,
	},
	{kernelPkg, "Mask", "Bit"}: {
		Authored: "accel.KernelMask.Bit", Op: ir.OpMaskBit, Params: 1,
		Result: ir.Bool, Class: ClassExact,
	},
	{kernelPkg, "Mask", "LowestSet"}: {
		Authored: "accel.KernelMask.LowestSet", Op: ir.OpMaskLowestSet,
		Result: ir.U32, Class: ClassExact,
	},
	{kernelPkg, "Mask", "CountLower"}: {
		Authored: "accel.KernelMask.CountLower", Op: ir.OpMaskCountLower, Params: 1,
		Result: ir.I32, Class: ClassExact,
	},
	{kernelPkg, "Mask", "Any"}: {
		Authored: "accel.KernelMask.Any", Op: ir.OpMaskAny,
		Result: ir.Bool, Class: ClassExact,
	},

	// A capability rather than a baseline: several targets lack it, so a kernel
	// using it is refused on a device that does rather than lowered to
	// something else.
	{accelPkg, "", "AddF32"}: {
		Authored: "accel.AddF32", Op: ir.OpAtomicAddF32, Result: ir.F32, Params: 3,
		Class: ClassExact, Cap: CapAtomicFloatAddStorage,
	},

	// Subgroup operations. The id accessors combine nothing and are
	// subgroup-uniform or not by their own definition; the rest are rendezvous.
	{kernelPkg, "Thread", "SubgroupSize"}: {
		Authored: "accel.Thread.SubgroupSize", Op: ir.OpSubgroupSize,
		Uniformity: PerWorkgroup, Result: ir.U32, Cap: CapSubgroupBasic,
	},
	{kernelPkg, "Thread", "SubgroupIndex"}: {
		Authored: "accel.Thread.SubgroupIndex", Op: ir.OpSubgroupID,
		Uniformity: PerInvocation, Result: ir.U32, Cap: CapSubgroupBasic,
	},
	{kernelPkg, "Thread", "SubgroupLane"}: {
		Authored: "accel.Thread.SubgroupLane", Op: ir.OpSubgroupInvocationID,
		Uniformity: PerInvocation, Result: ir.U32, Cap: CapSubgroupBasic,
	},

	{kernelPkg, "Thread", "SubgroupAddF32"}: {
		Authored: "accel.Thread.SubgroupAddF32", Op: ir.OpSubgroupAddF32, Params: 1,
		Result: ir.F32, Class: ClassBounded, Cap: CapSubgroupArithmetic,
	},
	{kernelPkg, "Thread", "SubgroupMinF32"}: {
		Authored: "accel.Thread.SubgroupMinF32", Op: ir.OpSubgroupMinF32, Params: 1,
		Result: ir.F32, Class: ClassExact, Cap: CapSubgroupArithmetic,
	},
	{kernelPkg, "Thread", "SubgroupMaxF32"}: {
		Authored: "accel.Thread.SubgroupMaxF32", Op: ir.OpSubgroupMaxF32, Params: 1,
		Result: ir.F32, Class: ClassExact, Cap: CapSubgroupArithmetic,
	},
	{kernelPkg, "Thread", "SubgroupBroadcastFirstF32"}: {
		Authored: "accel.Thread.SubgroupBroadcastFirstF32", Op: ir.OpBroadcastFirstF32, Params: 1,
		Result: ir.F32, Class: ClassExact, Cap: CapSubgroupBallot,
	},
	{kernelPkg, "Thread", "SubgroupElect"}: {
		Authored: "accel.Thread.SubgroupElect", Op: ir.OpElect,
		Result: ir.Bool, Class: ClassExact, Cap: CapSubgroupBasic,
	},
	{kernelPkg, "Thread", "SubgroupAny"}: {
		Authored: "accel.Thread.SubgroupAny", Op: ir.OpSubgroupAny, Params: 1,
		Result: ir.Bool, Class: ClassExact, Cap: CapSubgroupVote,
	},
	{kernelPkg, "Thread", "SubgroupAll"}: {
		Authored: "accel.Thread.SubgroupAll", Op: ir.OpSubgroupAll, Params: 1,
		Result: ir.Bool, Class: ClassExact, Cap: CapSubgroupVote,
	},

	// The lane-addressed reads. Each moves a value between lanes without
	// arithmetic, so each is exact: what comes back is the bit pattern that
	// went in, and a test comparing anything but equality would be testing
	// nothing.
	//
	// All five require CapSubgroupShuffle, including the broadcast from a
	// chosen lane. Vulkan groups that one under its ballot feature; accel's own
	// capability names what the operation does, which is read the value held by
	// a lane an operand names, and that is the shuffle bit. The deviation is
	// recorded in specs/002-compute-model.md section 5.2.
	{kernelPkg, "Thread", "SubgroupBroadcastF32"}: {
		Authored: "accel.Thread.SubgroupBroadcastF32", Op: ir.OpBroadcastF32, Params: 2,
		Result: ir.F32, Class: ClassExact, Cap: CapSubgroupShuffle,
	},
	{kernelPkg, "Thread", "SubgroupShuffleF32"}: {
		Authored: "accel.Thread.SubgroupShuffleF32", Op: ir.OpShuffleF32, Params: 2,
		Result: ir.F32, Class: ClassExact, Cap: CapSubgroupShuffle,
	},
	{kernelPkg, "Thread", "SubgroupShuffleXorF32"}: {
		Authored: "accel.Thread.SubgroupShuffleXorF32", Op: ir.OpShuffleXorF32, Params: 2,
		Result: ir.F32, Class: ClassExact, Cap: CapSubgroupShuffle,
	},
	{kernelPkg, "Thread", "SubgroupShuffleUpF32"}: {
		Authored: "accel.Thread.SubgroupShuffleUpF32", Op: ir.OpShuffleUpF32, Params: 2,
		Result: ir.F32, Class: ClassExact, Cap: CapSubgroupShuffle,
	},
	{kernelPkg, "Thread", "SubgroupShuffleDownF32"}: {
		Authored: "accel.Thread.SubgroupShuffleDownF32", Op: ir.OpShuffleDownF32, Params: 2,
		Result: ir.F32, Class: ClassExact, Cap: CapSubgroupShuffle,
	},

	// The scans. Bounded rather than exact, for the reason the reduction is:
	// the result is a sum of f32 values and f32 addition is not associative, so
	// two backends that accumulate in different orders differ in the last bit.
	// specs/002-compute-model.md section 5.2 fixes the *order* -- ascending
	// active lane -- which is what makes a comparison against a fallback
	// meaningful at all.
	{kernelPkg, "Thread", "SubgroupInclusiveAddF32"}: {
		Authored: "accel.Thread.SubgroupInclusiveAddF32", Op: ir.OpSubgroupInclusiveAddF32,
		Params: 1, Result: ir.F32, Class: ClassBounded, Cap: CapSubgroupArithmetic,
	},
	{kernelPkg, "Thread", "SubgroupExclusiveAddF32"}: {
		Authored: "accel.Thread.SubgroupExclusiveAddF32", Op: ir.OpSubgroupExclusiveAddF32,
		Params: 1, Result: ir.F32, Class: ClassBounded, Cap: CapSubgroupArithmetic,
	},

	// Recognized and not available. Being in the table is what makes the
	// rejection say "barriers arrive at M4" at the right line, rather than
	// leaving a kernel author with an unknown-call error about a method that
	// plainly exists.
	{kernelPkg, "Thread", "Barrier"}: {
		Authored: "accel.Thread.Barrier", Op: ir.OpBarrier, Stage: Cooperative,
		Uniformity: PerWorkgroup, Result: ir.Invalid,
	},

	// The masked barriers, specs/050-barrier-scopes.md §2. Cooperative for the
	// same reason Barrier is -- they rendezvous -- and no capability, because
	// a narrower memory scope is available wherever the wider one is.
	{kernelPkg, "Thread", "BarrierShared"}: {
		Authored: "accel.Thread.BarrierShared", Op: ir.OpBarrierShared, Stage: Cooperative,
		Uniformity: PerWorkgroup, Result: ir.Invalid,
	},
	{kernelPkg, "Thread", "BarrierStorage"}: {
		Authored: "accel.Thread.BarrierStorage", Op: ir.OpBarrierStorage, Stage: Cooperative,
		Uniformity: PerWorkgroup, Result: ir.Invalid,
	},

	// A barrier at subgroup scope, specs/050-barrier-scopes.md §2 and
	// 002 §5.3. Capability-gated, unlike the workgroup barriers: GLES 3.1 has
	// no subgroup concept at all, so a kernel using this is refused on a device
	// that lacks it rather than lowered to a workgroup barrier that happens to
	// work.
	{kernelPkg, "Thread", "SubgroupBarrier"}: {
		Authored: "accel.Thread.SubgroupBarrier", Op: ir.OpSubgroupBarrier, Stage: Cooperative,
		Uniformity: PerWorkgroup, Result: ir.Invalid, Cap: CapSubgroupBasic,
	},
}

// capNames is each capability's spelling in an //accel:requires directive.
//
// The public spelling rather than the constant's Go name, so a kernel author
// writes what they would read in the capability table.
var capNames = map[string]Capability{
	"subgroup_basic":           CapSubgroupBasic,
	"subgroup_vote":            CapSubgroupVote,
	"subgroup_ballot":          CapSubgroupBallot,
	"subgroup_shuffle":         CapSubgroupShuffle,
	"subgroup_arithmetic":      CapSubgroupArithmetic,
	"f16_arithmetic":           CapF16Arithmetic,
	"bf16_arithmetic":          CapBF16Arithmetic,
	"atomic_float_add_storage": CapAtomicFloatAddStorage,
	"atomic_float_add_shared":  CapAtomicFloatAddShared,
	"i8_dot_product":           CapI8DotProduct,
}

// CapByName resolves a capability's directive spelling.
func CapByName(name string) (Capability, bool) {
	c, ok := capNames[name]
	return c, ok
}

// CapNames lists every capability's directive spelling, sorted, which is what a
// diagnostic offers when a directive names something that is not one.
func CapNames() []string {
	out := make([]string, 0, len(capNames))
	for n := range capNames {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// DescribeCaps names the capabilities in a set, sorted, for a diagnostic.
func DescribeCaps(c Capability) string {
	if c == 0 {
		return "none"
	}
	var out []string
	for _, n := range CapNames() {
		if capNames[n]&c != 0 {
			out = append(out, n)
		}
	}
	return strings.Join(out, ", ")
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

// HasOp reports whether some intrinsic lowers to this opcode.
//
// It exists for the gate that walks the subgroup opcodes and checks each is
// reachable from a kernel: an opcode in the IR that no table entry names is one
// the compiler can emit and an author cannot call, which is the shape of a
// half-landed operation.
func HasOp(op ir.Opcode) bool {
	for _, in := range table {
		if in.Op == op {
			return true
		}
	}
	return false
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
		fmt.Fprintf(&b, "%s\t%s\t%v\t%v\t%v\t%d\t%v\t%d\n",
			in.Authored, k, in.Op, in.Stage, in.Result, in.Params, in.Class, uint32(in.Cap))
	}
	return b.String()
}
