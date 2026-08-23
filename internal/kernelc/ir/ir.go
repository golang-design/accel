// Copyright 2026 The golang.design Initiative Authors.
// All rights reserved. Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

// Package ir is the typed structured IR every target is emitted from.
//
// # Why an IR and not an AST walk per target
//
// The predecessor emitted GLSL and MSL by walking the Go AST with a `glsl bool`
// threaded through every method, and ran a separate inspection pass per target
// to find written buffers. That is the two-target shape of a problem that is
// quadratic in targets. Every analysis here (recursion, access inference,
// capability requirements, and at M4 barrier divergence) runs once, on one
// representation.
//
// SPIR-V is the argument that makes it mandatory rather than tidy. It is a
// binary SSA format with explicit result ids and structured control flow
// declared through OpSelectionMerge and OpLoopMerge, which you do not print by
// walking an AST, and there is no cgo-free path from GLSL text to SPIR-V
// because glslang and shaderc are C++. Vulkan consumes SPIR-V only.
//
// # Why structured control flow, and why not go/ssa
//
// The IR is a tree of typed statements with `if` and `for` as nodes, not a
// general CFG. Three of the four GPU targets are structured source languages
// and SPIR-V demands structured control flow anyway, so golang.org/x/tools/go/ssa
// is rejected: it discards exactly the structure every target needs back, and
// recovering it means writing a relooper to undo work nobody needed done.
// Because the Go subset excludes goto and arbitrary labeled jumps, the structure
// survives for free.
//
// # The set is closed
//
// There is deliberately no generic AST escape node. A construct outside this
// set is a source-positioned subset error, never a passthrough. See
// specs/004-kernel-authoring.md.
package ir

import (
	"fmt"
	"go/constant"
	"go/token"
	"go/types"
)

// Kind is a type's shape in the IR.
type Kind int

const (
	Invalid Kind = iota
	Bool
	I32
	U32
	F32
	// I8 and U8 are storage and conversion types, for quantized weights. Like
	// the narrow floats they are storage rather than arithmetic kinds.
	I8
	U8
	// F16 and BF16 are storage kinds. They carry no arithmetic: a value converts
	// to F32 on load and back on store, which is what makes narrow dtypes work
	// on every backend rather than only where native narrow arithmetic exists.
	F16
	BF16
	// ID3Kind is the three-component id struct, which is a distinct kind rather
	// than an ordinary struct because every target has a native spelling for it.
	ID3Kind
	Struct
	// Array is a fixed-extent workgroup-shared array, whose extent go/types
	// reads off the type so the IR never invents const generics.
	Array
	// Slice is a storage-buffer binding.
	Slice
)

var kindNames = [...]string{
	Invalid: "invalid", Bool: "bool", I32: "i32", U32: "u32", F32: "f32",
	I8: "i8", U8: "u8", F16: "f16", BF16: "bf16", ID3Kind: "ID3", Struct: "struct", Array: "array", Slice: "slice",
}

// IsAtomic reports whether an opcode is an atomic read-modify-write.
//
// It exists so that access inference does not have to enumerate the set: an
// atomic added to the table and forgotten here would be a binding that looks
// untouched, which the graph builder turns into a missing barrier.
// IsSubgroupRendezvous reports whether an opcode needs every lane's value at
// the point of the call, and therefore suspends.
//
// The id accessors are excluded: they read this invocation's own position and
// combine nothing, so making them suspend would cost an epoch for an answer
// already in hand.
func (o Opcode) IsSubgroupRendezvous() bool {
	return o >= OpSubgroupAddF32 && o <= OpBallot
}

// IsSubgroup reports whether an opcode is any subgroup operation, rendezvous or
// accessor. It is what capability inference and the uniformity requirement key
// on.
func (o Opcode) IsSubgroup() bool {
	return o >= OpSubgroupSize && o <= OpBallot
}

func (o Opcode) IsAtomic() bool {
	return o >= OpAtomicAddU32 && o <= OpAtomicAddF32
}

func (k Kind) String() string {
	if k < 0 || int(k) >= len(kindNames) {
		return fmt.Sprintf("Kind(%d)", int(k))
	}
	return kindNames[k]
}

// Numeric reports whether arithmetic is expressible on this kind directly.
// F16 and BF16 are excluded on purpose: they are storage formats and Go itself
// forces the conversion, so f32 accumulation is not a convention but the only
// thing that compiles.
func (k Kind) Numeric() bool { return k == I32 || k == U32 || k == F32 }

// Type is a resolved IR type.
type Type struct {
	Kind   Kind
	Elem   *Type   // Array and Slice
	Len    int     // Array
	Name   string  // Struct
	Fields []Field // Struct
}

// Field is one member of a struct type.
type Field struct {
	Name string
	Type *Type
}

func (t *Type) String() string {
	switch t.Kind {
	case Array:
		return fmt.Sprintf("[%d]%s", t.Len, t.Elem)
	case Slice:
		return "[]" + t.Elem.String()
	case Struct:
		return t.Name
	}
	return t.Kind.String()
}

// Node is anything in the IR, which is everything carrying a source position.
// A diagnostic that cannot name a line is one a reader cannot act on.
type Node interface {
	Pos() token.Pos
}

// pos is embedded by every node.
type pos struct{ P token.Pos }

func (p pos) Pos() token.Pos { return p.P }

// Value produces a value. The set is closed: the unexported method is what
// makes adding a case outside this file impossible rather than merely
// discouraged.
type Value interface {
	Node
	Type() *Type
	isValue()
}

type value struct {
	pos
	T *Type
}

func (v value) Type() *Type { return v.T }
func (value) isValue()      {}

// Const is a compile-time constant with its resolved type and value.
//
// Resolved, which is where the GLSL integer-literal divergence is settled: the
// emitter knows whether the 2 in gid*2 is u32, i32, or f32 and spells it
// accordingly, instead of coercing the id to int to keep literals legal.
type Const struct {
	value
	Val constant.Value
}

// Param is a kernel or helper parameter, addressed by index because the
// signature is the binding layout.
type Param struct {
	value
	Index int
	Name  string
	Obj   types.Object
}

// Local is a declared local variable.
type Local struct {
	value
	Name string
	ID   int
	Obj  types.Object
}

// FieldSel is a struct or ID3 field selection.
type FieldSel struct {
	value
	X     Value
	Index int
	Name  string
}

// IndexExpr is an index into a slice or a shared array.
type IndexExpr struct {
	value
	X     Value
	Index Value

	// Binding is the parameter index of the resource this reaches, or -1 when
	// the indexed value is not a binding. Access inference reads it, which is
	// why an access is a property of the IR rather than of a second AST pass.
	Binding int
}

// Unary is a unary operation.
type Unary struct {
	value
	Op token.Token
	X  Value
}

// Binary is a binary operation.
type Binary struct {
	value
	Op   token.Token
	X, Y Value
}

// Convert is an explicit conversion. There are no implicit ones: Go's own rules
// already forbid them between the numeric types here, which is what keeps a
// narrow dtype from silently participating in arithmetic.
type Convert struct {
	value
	X Value
}

// Call is a call to a helper in the same compilation.
type Call struct {
	value
	Callee *Func
	Args   []Value
}

// IntrinsicCall is a call to a known intrinsic, identified by opcode rather
// than by name. Resolution happens in the front end against object identity;
// by the time it reaches the IR the name is gone and cannot be confused with a
// user function that shares it.
type IntrinsicCall struct {
	value
	Op   Opcode
	Recv Value // the Thread receiver, or nil for a free function
	Args []Value
}

// Len is the length of a slice binding. It is a node rather than an intrinsic
// because every target spells it differently and none of them spells it as a
// call.
type Len struct {
	value
	X Value
}

func (Const) isValue()         {}
func (Param) isValue()         {}
func (Local) isValue()         {}
func (FieldSel) isValue()      {}
func (IndexExpr) isValue()     {}
func (Unary) isValue()         {}
func (Binary) isValue()        {}
func (Convert) isValue()       {}
func (Call) isValue()          {}
func (IntrinsicCall) isValue() {}
func (Len) isValue()           {}

// Stmt is a statement. Closed, for the same reason [Value] is.
type Stmt interface {
	Node
	isStmt()
}

type stmt struct{ pos }

func (stmt) isStmt() {}

// Block is a statement sequence.
type Block struct {
	stmt
	List []Stmt
}

// Declare introduces a local, with its initializer.
type Declare struct {
	stmt
	Local *Local
	Init  Value
}

// Assign stores to a local, an index, or a field.
type Assign struct {
	stmt
	LHS Value
	RHS Value
}

// ExprStmt is a call evaluated for its effect.
type ExprStmt struct {
	stmt
	X Value
}

// If is a conditional. Else is nil, a *Block, or another *If.
type If struct {
	stmt
	Cond Value
	Then *Block
	Else Stmt
}

// For covers all three Go loop forms: Init and Post are nil for the
// condition-only form, and Cond is nil for the infinite form.
type For struct {
	stmt
	Init Stmt
	Cond Value
	Post Stmt
	Body *Block
}

// Break and Continue carry no label, because the subset admits none.
type Break struct{ stmt }

// Continue leaves the current iteration.
type Continue struct{ stmt }

// Return is single-value or empty.
type Return struct {
	stmt
	Value Value
}

func (Block) isStmt()    {}
func (Declare) isStmt()  {}
func (Assign) isStmt()   {}
func (ExprStmt) isStmt() {}
func (If) isStmt()       {}
func (For) isStmt()      {}
func (Break) isStmt()    {}
func (Continue) isStmt() {}
func (Return) isStmt()   {}

// Opcode identifies an intrinsic. It is versioned as part of the intrinsic
// table's ABI, which participates in the kernel digest.
type Opcode int

const (
	OpInvalid Opcode = iota

	// Thread ids. Available to a flat kernel.
	OpGlobalID
	OpLocalID
	OpGroupID
	OpGlobalIndex
	OpLocalIndex
	OpGroupIndex

	// Bounded scalar math from accel/kmath. Each has a normative per-operation
	// domain and error ceiling in spec 008 section 6; an operation with no bound
	// is not admitted rather than admitted with a tuned tolerance.
	OpSqrt
	OpRSqrt
	OpExp
	OpLog
	OpSin
	OpCos
	OpTanh
	OpAbs
	OpMin
	OpMax

	// Conversions between narrow storage and f32. They are intrinsics rather
	// than IR conversions because every target spells them differently: a native
	// instruction where the format exists, and a bit-packing sequence where it
	// does not.
	OpF16ToF32
	OpBF16ToF32
	OpF32ToF16
	OpF32ToBF16

	// Atomics. Free functions taking a buffer and an index, because GLSL cannot
	// form a pointer into a buffer (specs/002-compute-model.md section 4.1).
	// Each returns the previous value.
	OpAtomicAddU32
	OpAtomicAddI32
	OpAtomicSubU32
	OpAtomicSubI32
	OpAtomicMinU32
	OpAtomicMinI32
	OpAtomicMaxU32
	OpAtomicMaxI32
	OpAtomicAndU32
	OpAtomicOrU32
	OpAtomicXorU32
	OpAtomicExchangeU32
	OpAtomicExchangeI32
	OpAtomicCompareExchangeU32
	OpAtomicCompareExchangeI32

	// OpAtomicAddF32 is a capability rather than a baseline, and it makes a
	// reduction non-deterministic because the hardware picks the accumulation
	// order.
	OpAtomicAddF32

	// Subgroup operations. Each is a rendezvous in the generated lowering,
	// because it needs every lane's contribution at the point of the call and
	// the scheduler advances one invocation at a time.
	OpSubgroupSize
	OpSubgroupID
	OpSubgroupInvocationID
	OpSubgroupAddF32
	OpSubgroupMinF32
	OpSubgroupMaxF32
	OpBroadcastFirstF32
	OpElect
	OpSubgroupAny
	OpSubgroupAll
	OpBallot

	// Cooperative. Recognized so that a kernel using one is rejected by name
	// with a position, rather than failing as an unknown call. See
	// specs/012-kernel-pipeline.md.
	OpBarrier
)

var opcodeNames = [...]string{
	OpInvalid:                  "invalid",
	OpGlobalID:                 "GlobalID",
	OpLocalID:                  "LocalID",
	OpGroupID:                  "GroupID",
	OpGlobalIndex:              "GlobalIndex",
	OpLocalIndex:               "LocalIndex",
	OpGroupIndex:               "GroupIndex",
	OpSqrt:                     "Sqrt",
	OpRSqrt:                    "RSqrt",
	OpExp:                      "Exp",
	OpLog:                      "Log",
	OpSin:                      "Sin",
	OpCos:                      "Cos",
	OpTanh:                     "Tanh",
	OpAbs:                      "Abs",
	OpMin:                      "Min",
	OpMax:                      "Max",
	OpF16ToF32:                 "Float16.F32",
	OpBF16ToF32:                "BFloat16.F32",
	OpF32ToF16:                 "ToFloat16",
	OpF32ToBF16:                "ToBFloat16",
	OpBarrier:                  "Barrier",
	OpAtomicAddU32:             "AtomicAddU32",
	OpAtomicAddI32:             "AtomicAddI32",
	OpAtomicSubU32:             "AtomicSubU32",
	OpAtomicSubI32:             "AtomicSubI32",
	OpAtomicMinU32:             "AtomicMinU32",
	OpAtomicMinI32:             "AtomicMinI32",
	OpAtomicMaxU32:             "AtomicMaxU32",
	OpAtomicMaxI32:             "AtomicMaxI32",
	OpAtomicAndU32:             "AtomicAndU32",
	OpAtomicOrU32:              "AtomicOrU32",
	OpAtomicXorU32:             "AtomicXorU32",
	OpAtomicExchangeU32:        "AtomicExchangeU32",
	OpAtomicExchangeI32:        "AtomicExchangeI32",
	OpAtomicCompareExchangeU32: "AtomicCompareExchangeU32",
	OpAtomicCompareExchangeI32: "AtomicCompareExchangeI32",
	OpAtomicAddF32:             "AtomicAddF32",
	OpSubgroupSize:             "SubgroupSize",
	OpSubgroupID:               "SubgroupID",
	OpSubgroupInvocationID:     "SubgroupInvocationID",
	OpSubgroupAddF32:           "SubgroupAddF32",
	OpSubgroupMinF32:           "SubgroupMinF32",
	OpSubgroupMaxF32:           "SubgroupMaxF32",
	OpBroadcastFirstF32:        "BroadcastFirstF32",
	OpElect:                    "Elect",
	OpSubgroupAny:              "SubgroupAny",
	OpSubgroupAll:              "SubgroupAll",
	OpBallot:                   "Ballot",
}

func (o Opcode) String() string {
	if o < 0 || int(o) >= len(opcodeNames) {
		return fmt.Sprintf("Opcode(%d)", int(o))
	}
	return opcodeNames[o]
}

// UniformField is one member of a uniform block, placed.
//
// It carries the offset rather than a way to compute one, because the offset is
// the whole point: std140's padding is not Go's, and a generated encoder that
// recomputed it would be a second implementation of the layout.
type UniformField struct {
	Name   string
	Offset int

	// Kind is "scalar", "vector", "array", "matrix", or "struct".
	Kind string

	// Scalar is the Go spelling of the element type.
	Scalar string

	// Len is an array's length, a vector's component count, or a matrix's
	// column count.
	Len int

	// Stride is the byte distance between elements of an array or columns of a
	// matrix, which std140 rounds up to sixteen.
	Stride int
}

// Uniform is one by-value struct parameter.
//
// It is a separate list from [Binding] because it is a different resource kind
// with a different layout rule: a binding is a tightly packed array of one
// dtype and a uniform is a std140 block whose padding is not the caller's to
// compute. See specs/001-device-resources.md section 3.3.
type Uniform struct {
	Name  string
	Index int

	// TypeName is the Go type's name, which the generated codec is named after
	// and which a caller writes when constructing a value.
	TypeName string

	// Size is the encoded block size in bytes, rounded up to sixteen.
	Size int

	// Fields is the block's placement, which the generated codec is emitted
	// from.
	Fields []UniformField

	// Reads is whether the body reads any of it. A uniform nothing reads is a
	// value the caller has to supply for no reason.
	Reads bool
}

// Binding is one resource parameter, as the IR sees it.
type Binding struct {
	Name  string
	Index int
	Type  *Type

	// Read and Write are inferred from the body rather than declared. A caller
	// who could declare them would be a second source of truth for something the
	// compiler already knows, and one that can be wrong.
	Read, Write bool
}

// SharedMem is one workgroup-shared array a kernel's signature declares.
//
// Its extent is fixed at pipeline creation on every backend -- it appears in
// the GLSL layout qualifier and in Metal's threadgroup attribute -- which is
// why the authored form is a pointer to a fixed-size array rather than a slice.
type SharedMem struct {
	Name  string
	Index int
	Type  *Type
}

// Func is a kernel or a helper.
type Func struct {
	pos
	Name   string
	Kernel bool

	// Workgroup is the extent from the //accel:kernel directive. Zero for a
	// helper.
	Workgroup [3]uint32

	// Thread is the index of the accel.Thread parameter, or -1 for a helper that
	// does not take one.
	Thread   int
	Params   []*Param
	Bindings []*Binding
	Body     *Block

	// Shared is the workgroup-shared storage the signature declares, in
	// signature order. Its element type and extent come from the Go array type,
	// so the IR never invents const generics.
	Shared []*SharedMem

	// Cooperative reports that the body reaches a barrier, shared memory, or a
	// subgroup operation, so it needs the resumable lowering rather than the
	// flat one. It is derived from the body, never declared: a declaration can
	// be forgotten and the failure would be a kernel lowered the wrong way.
	Cooperative bool

	// Caps is every capability the body implies, inferred from the intrinsics it
	// reaches. Never declared: a declaration can be forgotten, and the failure
	// is silent -- a kernel using a feature the device lacks produces wrong
	// results rather than an error, because nothing checked.
	Caps uint32

	// Intrinsics is every intrinsic the body reaches, in first-use order, by its
	// authored spelling. The digest records these rather than resolved package
	// paths, so relocating a type does not invalidate a committed digest.
	Intrinsics []string

	// Source is the normalized text of the authored declaration, printed from
	// the AST rather than read from the file.
	//
	// Printed, because it has to work for a package that is not on disk, and
	// because normalizing means a gofmt run does not force every kernel to be
	// regenerated while a semantic edit still does. Comments are dropped for the
	// same reason: a comment is not something the generated form depends on.
	Source string

	// Digest identifies everything this kernel's generated form depends on, not
	// only its source. Filled in by the generator.
	Digest string

	// Result is a helper's return type, or nil for a kernel and for a helper
	// that returns nothing. A kernel never returns: it writes through its
	// bindings.
	Result *Type

	// SignatureBuilt reports whether this function's parameters are known. A
	// helper's signature is built before any body, so that a helper calling
	// another can be checked whatever order the file declares them in.
	SignatureBuilt bool

	// Uniforms are the by-value struct parameters, in signature order. Each
	// carries the std140 layout its codec is generated from.
	Uniforms []*Uniform

	// Helpers are the helpers this function's body reaches, transitively, in a
	// stable order. The digest records them so that editing a helper without
	// regenerating its callers is caught.
	Helpers []*Func
}

// Builders. Constructing nodes through functions rather than literals keeps the
// position and the type mandatory: a node with a zero position produces a
// diagnostic pointing at the top of the file, which is worse than none.

func NewConst(p token.Pos, t *Type, v constant.Value) *Const {
	return &Const{value: value{pos{p}, t}, Val: v}
}

func NewParam(p token.Pos, t *Type, i int, name string, obj types.Object) *Param {
	return &Param{value: value{pos{p}, t}, Index: i, Name: name, Obj: obj}
}

func NewLocal(p token.Pos, t *Type, id int, name string, obj types.Object) *Local {
	return &Local{value: value{pos{p}, t}, ID: id, Name: name, Obj: obj}
}

func NewFieldSel(p token.Pos, t *Type, x Value, i int, name string) *FieldSel {
	return &FieldSel{value: value{pos{p}, t}, X: x, Index: i, Name: name}
}

func NewIndex(p token.Pos, t *Type, x, idx Value, binding int) *IndexExpr {
	return &IndexExpr{value: value{pos{p}, t}, X: x, Index: idx, Binding: binding}
}

func NewUnary(p token.Pos, t *Type, op token.Token, x Value) *Unary {
	return &Unary{value: value{pos{p}, t}, Op: op, X: x}
}

func NewBinary(p token.Pos, t *Type, op token.Token, x, y Value) *Binary {
	return &Binary{value: value{pos{p}, t}, Op: op, X: x, Y: y}
}

func NewConvert(p token.Pos, t *Type, x Value) *Convert {
	return &Convert{value: value{pos{p}, t}, X: x}
}

func NewCall(p token.Pos, t *Type, callee *Func, args []Value) *Call {
	return &Call{value: value{pos{p}, t}, Callee: callee, Args: args}
}

func NewIntrinsic(p token.Pos, t *Type, op Opcode, recv Value, args []Value) *IntrinsicCall {
	return &IntrinsicCall{value: value{pos{p}, t}, Op: op, Recv: recv, Args: args}
}

func NewLen(p token.Pos, t *Type, x Value) *Len {
	return &Len{value: value{pos{p}, t}, X: x}
}

func NewBlock(p token.Pos, list ...Stmt) *Block { return &Block{stmt: stmt{pos{p}}, List: list} }

func NewDeclare(p token.Pos, l *Local, init Value) *Declare {
	return &Declare{stmt: stmt{pos{p}}, Local: l, Init: init}
}

func NewAssign(p token.Pos, lhs, rhs Value) *Assign {
	return &Assign{stmt: stmt{pos{p}}, LHS: lhs, RHS: rhs}
}

func NewExprStmt(p token.Pos, x Value) *ExprStmt { return &ExprStmt{stmt: stmt{pos{p}}, X: x} }

func NewIf(p token.Pos, cond Value, then *Block, els Stmt) *If {
	return &If{stmt: stmt{pos{p}}, Cond: cond, Then: then, Else: els}
}

func NewFor(p token.Pos, init Stmt, cond Value, post Stmt, body *Block) *For {
	return &For{stmt: stmt{pos{p}}, Init: init, Cond: cond, Post: post, Body: body}
}

func NewBreak(p token.Pos) *Break       { return &Break{stmt{pos{p}}} }
func NewContinue(p token.Pos) *Continue { return &Continue{stmt{pos{p}}} }

func NewReturn(p token.Pos, v Value) *Return { return &Return{stmt: stmt{pos{p}}, Value: v} }
