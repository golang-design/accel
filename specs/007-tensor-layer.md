---
title: "Tensor layer: values, state, operators, and plans"
status: drafted
layer: tensor
depends_on:
  - 001-device-resources.md
  - 002-compute-model.md
  - 003-command-graph.md
  - 008-numerics.md
---

# Tensor layer

This is layer 2 of [`000-decisions.md`](000-decisions.md)'s two-layer split. It
defines the smallest tensor system that can run an unquantized transformer
prefill and decode step on the CPU and Metal backends. It lowers entirely through
the device layer; a backend never learns what a tensor is.

v0 is deliberately narrower than the eventual inference layer. It supports f32
and f16 storage with f32 accumulation, one caller-owned plan per concrete shape,
a caller-owned contiguous KV cache, and the operator set named below.
Quantization, sampling, automatic fusion, plan caching, and multi-sequence
scheduling are post-v0 work.

## Ownership and core types

Four host-side types and one persistent-state type carry the layer:

```go
// Runtime owns one device and the device-specific compiled-kernel cache.
// It does not cache or own Plans in v0.
type Runtime struct{ ... }

func NewRuntime(dev *accel.Device) (*Runtime, error)
func (r *Runtime) NewBuilder(label string) *Builder
func (r *Runtime) Close() error

// Builder records one tensor DAG. It belongs to one goroutine.
type Builder struct{ ... }

// Tensor is an immutable logical value: dtype, shape, layout, and producer.
// External tensors additionally name a stable input or weight port.
type Tensor struct{ ... }

// State is caller-owned mutable storage represented as SSA-style versions in
// the Builder. A write returns the next version; the underlying buffer is the
// same binding.
type State struct{ ... }

// Plan owns exactly one device Graph and its transient pool. It is immutable,
// caller-owned, and has the Graph's one-submission-in-flight restriction.
type Plan struct{ ... }
```

`Runtime.Close` requires all plans created from it to be closed. `Plan.Close`
releases its graph and transient pool. Pipelines are reference-counted entries in
the runtime kernel cache and are released by the runtime, not by an individual
plan. Tensor and state-version handles are invalid after their builder is
compiled or closed.

There is **no automatic plan cache in v0**. The caller keeps and reuses a plan,
and explicitly compiles another for another prefill bucket or concurrent request.
This makes memory ownership, eviction, and concurrency visible. A future cache
may be layered above this API, but its key must include a stable tensor-DAG
identity, every input shape and dtype, the selected kernel-set version, the
device identity and relevant capabilities, and every compile option that affects
lowering. Shape alone is never a sufficient key.

## Creating graph values

External values are declared before operators. Names are unique within a
builder, appear in errors, and become binding names on the plan.

```go
type Shape []int
type DType = accel.DType

type ValueDesc struct {
	Name  string
	DType DType
	Shape Shape
}

type ScalarKind uint8
const (
	ScalarU32 ScalarKind = iota
	ScalarF32
)

type ScalarDesc struct {
	Name string
	Kind ScalarKind
}

type PortKind uint8
const (
	PortInput PortKind = iota
	PortWeight
	PortState
	PortOutput
)

type PortDesc struct {
	Name     string
	DType    DType
	Shape    Shape
	Kind     PortKind
	Access   accel.Access
}

func Input(b *Builder, d ValueDesc) *Tensor
func Weight(b *Builder, d ValueDesc) *Tensor
func Scalar(b *Builder, d ScalarDesc)
func Output(b *Builder, name string, x *Tensor)

type StateDesc struct {
	Name     string
	DType    DType
	Shape    Shape
	Capacity int // sequence capacity for a KV state, zero otherwise
}

func Persistent(b *Builder, d StateDesc) *State
func ReadState(b *Builder, s *State) *Tensor
func LayerState(b *Builder, s *State, layer int) *State
```

`Input` varies on every submission. `Weight` may be rebound between submissions
but is read-only. `Persistent` declares caller-owned read-write storage and is
never transient or aliased by the planner. `Output` declares a value the caller
wants copied or written into a caller-supplied output binding; undeclared
intermediates are inaccessible after compilation. `LayerState` creates a
compile-time view of one layer while preserving the parent's version chain and
binding identity.

`Scalar` explicitly declares every named per-step value used by an operator.
Operators reject an undeclared name or the wrong scalar kind; two declarations
of one name are an error. This keeps `Bindings.Scalars` closed and type-checkable
instead of treating a misspelled operator argument as a new input.

All dimensions are positive concrete integers at compile time. Dimensions are
listed outermost to innermost, the last dimension varies fastest, and contiguous
strides are suffix products in elements. Negative axes count from the end.

## Compile, bind, submit

```go
type CompileOptions struct {
	Label string
}

func (b *Builder) Compile(r *Runtime, opts CompileOptions) (*Plan, error)

type ScalarValue struct {
	U32 uint32
	F32 float32
	Kind ScalarKind
}

type Bindings struct {
	Buffers map[string]accel.BufferView // inputs, weights, state, outputs
	Scalars map[string]ScalarValue      // per-step values declared by operators
}

func (p *Plan) Ports() []PortDesc
func (p *Plan) Memory() accel.GraphMemory
func (p *Plan) Selections() []KernelSelection
func (p *Plan) Submit(q *accel.Queue, bindings Bindings) *accel.Fence
func (p *Plan) Close() error
```

`Compile` performs shape and dtype inference, validates state versions, resolves
layouts, chooses kernels, creates or reuses pipelines, lowers to a recorder, and
builds one device graph. Errors are collected on the builder and returned
together, with operator, operand/port, and original Go call site. An invalid
operator returns a poisoned tensor so model code does not need an error branch
per line.

`Submit` validates the complete binding set atomically: name, dtype, minimum
size, usage, device ownership, liveness, and forbidden overlap. Missing or extra
bindings fail through an already-signalled fence, matching 003. Scalars are
packed into the plan's generated uniform block before submission. A scalar
declared as runtime data may vary; an operator attribute that changes shape,
layout, or kernel selection requires another plan.

The caller owns input, weight, state, and output buffers and must keep them alive
until the fence signals. The plan owns transients and its small generated scalar
upload block. Output visibility follows 001 and 003: readback memory is
host-visible after `Fence.Wait` returns nil.

## Persistent state and the KV cache

Mutation is explicit in the host DAG:

```go
// ScatterRows writes rows into state at runtime indexName and returns the next
// logical version of the same persistent binding.
func ScatterRows(b *Builder, state *State, rows *Tensor, indexName string) *State
```

For a layer-selected KV state `[maxSeq, kvHeads, headDim]`, rows are
`[writeCount, kvHeads, headDim]`; the runtime index names the first sequence
position written and the range must fit capacity.

Every state handle carries a monotonically increasing logical version. A state
write consumes one version and returns the next. Reading an older version after a
write has been recorded is a build error naming both call sites. Lowering maps
all versions to the same rebindable device slot but declares accurate read/write
access, so 003 infers ordering and barriers. Persistent state is never silently
copied and never aliases plan transients.

The v0 KV cache is for one sequence (`batch=1`) and is two caller-owned
contiguous buffers, keys and values, each
shaped `[layers, maxSeq, kvHeads, headDim]`. Its required bytes are:

```
2 * layers * maxSeq * kvHeads * headDim * sizeof(dtype)
```

The library exposes a checked sizing helper returning bytes and per-buffer
element counts. Capacity never grows. A per-step `u32` write index and current
length arrive through scalar bindings. Passing the index as data avoids rebinding
two views per layer per token. An index or length beyond capacity fails before
submission on the host; strict CPU execution also checks it in the kernel.

Paged caches, cache quantization, multiple sequences in one cache, and cache
eviction are post-v0.

```go
type KVCacheDesc struct {
	Layers, MaxSeq, KVHeads, HeadDim int
	DType DType
}

type KVCacheSizeInfo struct {
	ElementsPerBuffer int
	BytesPerBuffer    int
	TotalBytes        int
}

func KVCacheSize(d KVCacheDesc) (KVCacheSizeInfo, error)
```

The helper checks every dimension, dtype, integer multiplication, and layer-1
buffer limit before returning. The caller allocates two buffers of
`BytesPerBuffer`, one for keys and one for values.

## Layouts, views, and broadcasting

A view is a base element offset plus one element stride per axis. Legal views are
reachable from a contiguous base through unit-step slicing, permutation,
inserting/removing size-one axes, and broadcasting a size-one axis with stride
zero. Negative strides, stepped slices, and overlapping writable layouts are
rejected.

| Operation | Result |
| --- | --- |
| `Permute`, `Transpose`, `Slice`, `Squeeze`, `Unsqueeze`, `Broadcast` | view |
| `Reshape` | view if the affected axes are contiguous; otherwise build error |
| `Contiguous` | copy |
| `Cast` | copy and conversion |

NumPy right-aligned broadcasting applies to elementwise binary operators.
Ranks are left-padded with one, a dimension of one expands, and any other
mismatch is a build error. Broadcasting is read-only and never materialized.

A contiguous subrange lowers to a layer-1 `BufferView`. Any other legal layout
uses a generated layout descriptor and a strided kernel variant. Selection is at
compile time, not a runtime branch.

## v0 dtypes

v0 tensor storage is `f32`, `f16`, `i32`, and `u32`. `i32` and `u32` are for
indices and scalar/control data, not general model arithmetic. All reductions,
dot products, normalization statistics, and matmul accumulators are f32.

f16 storage does **not** imply `CapF16Arithmetic`. The portable v0 kernels load
f16, convert exactly to f32, compute in f32, and convert on store. A plan is
rejected only when it explicitly selects native f16 arithmetic and the device
lacks that capability; v0's required corpus never needs that selection.

There is no silent dtype promotion. An unsupported requested arithmetic mode is
a compile error naming the operator, capability, and device.

## v0 operator contracts

Operators are package functions taking the builder first. Attributes that affect
shape or kernel choice are ordinary Go arguments. Per-step values use a named
scalar or tensor input.

```go
func Add(b *Builder, x, y *Tensor) *Tensor
func Mul(b *Builder, x, y *Tensor) *Tensor
func Scale(b *Builder, x *Tensor, scalarName string) *Tensor
func SiLU(b *Builder, x *Tensor) *Tensor
func SwiGLU(b *Builder, gate, value *Tensor) *Tensor

func Reshape(b *Builder, x *Tensor, shape Shape) *Tensor
func Permute(b *Builder, x *Tensor, axes ...int) *Tensor
func Transpose(b *Builder, x *Tensor, axisA, axisB int) *Tensor
func Slice(b *Builder, x *Tensor, axis, start, end int) *Tensor
func Broadcast(b *Builder, x *Tensor, shape Shape) *Tensor
func Contiguous(b *Builder, x *Tensor) *Tensor
func Rows(b *Builder, table, ids *Tensor) *Tensor

func RMSNorm(b *Builder, x, gain *Tensor, eps float32) *Tensor
func MatMul(b *Builder, x, w *Tensor) *Tensor
func Linear(b *Builder, x, w, bias *Tensor) *Tensor
func RoPE(b *Builder, x, positions *Tensor, rotaryDim int, baseName string) *Tensor

type SoftmaxOptions struct {
	Axis      int
	ScaleName string
	Mask      *Tensor
	Causal    bool
}
func Softmax(b *Builder, x *Tensor, opts SoftmaxOptions) *Tensor

type AttentionOptions struct {
	CurrentLengthName string
	Causal            bool
}
func Attention(b *Builder, q *Tensor, k, v *State, opts AttentionOptions) *Tensor
```

### Elementwise and activation

| Operator | Contract |
| --- | --- |
| `Add`, `Mul` | Same dtype; f16/f32; NumPy broadcasting; broadcasted output shape. |
| `Scale` | f16/f32 tensor and named runtime f32 scalar; same shape and storage dtype. |
| `SiLU` | f16/f32; same shape and dtype; f32 evaluation. |
| `SwiGLU` | equal f16/f32 inputs; same shape; computes `SiLU(gate)*value` in one authored kernel. |

### Shape and indexing

| Operator | Contract |
| --- | --- |
| `Reshape` | Element count unchanged; legal view only. |
| `Permute` | Permutation contains each input axis exactly once. |
| `Transpose` | Swaps two normalized axes. |
| `Slice` | Unit step; `0 <= start <= end <= dim`. |
| `Broadcast` | Only size-one axes expand. |
| `Contiguous` | f16/f32/i32/u32; preserves logical values and shape. |
| `Rows` | Table `[vocab, width]` f16/f32, ids `i32`/`u32` of shape `S`; output `S+[width]`. Out-of-range ids fail in strict mode and are a caller error otherwise. |

### Normalization

`RMSNorm(x, gain, eps)` accepts `x[..., width]` and `gain[width]`, equal f16 or
f32 storage, and a positive finite compile-time `eps`. It returns the shape and
storage dtype of `x`, computes the mean square and reciprocal square root in f32,
and has no in-place form.

### Matrix multiplication

`MatMul(x, w)` accepts ranks at least two. Its matrix axes are the last two:
`x[..., M, K]`, `w[..., K, N]`, with NumPy broadcasting over leading batch axes.
The result is `broadcast(...)+[M,N]`. Storage dtypes must match and be f16 or
f32; accumulation is f32. v0 requires unit stride on `x`'s K axis and `w`'s N
axis, which admits ordinary contiguous row-major operands without silently
materializing either one.

`MatVec(x, w)` is the selected M=1 implementation, not a distinct public
semantic operation. `Linear(x, w, bias)` has the MatMul shape and a bias `[N]`
or broadcast-compatible leading shape. It is an authored epilogue kernel.

### RoPE and attention

`RoPE(x, positions, rotaryDim, baseName)` accepts
`x[..., seq, heads, headDim]`, positions `i32/u32` shaped `[..., seq]`, even
`rotaryDim <= headDim`, and a named runtime f32 base. It returns `x`'s shape and
dtype and rotates the first `rotaryDim` values in pairs.

`Softmax(x, opts)` accepts f16/f32, normalizes `opts.Axis`, and returns the same
shape/dtype. `ScaleName`, when non-empty, names a declared runtime f32 scalar.
The optional additive mask must broadcast to `x`; causal masking is a
compile-time attribute. Maximum subtraction, exponentiation, summation, and
division occur in f32.

`Attention(q, kState, vState, opts)` accepts q shaped
`[1, qSeq, qHeads, headDim]` and layer-selected K/V state shaped
`[maxSeq, kvHeads, headDim]`. `qHeads` must be a multiple of `kvHeads`; current
length must be at least qSeq and no greater than maxSeq. The result has q's shape.
The composed definition is score MatMul, Softmax, and value MatMul and is the
correctness reference.

Fused attention is **runtime kernel selection**, not a device capability.
`Compile` selects it only when a registered kernel variant supports the concrete
dtype, shape, limits, and required primitive capabilities. Otherwise it selects
the composed graph. `Plan.Selections` reports the decision and reason.

## Prefill and decode plans

A plan has one concrete shape. v0 requires:

- a decode plan with `qSeq=1`, reusable for every token up to KV capacity; and
- a minimal prefill plan for the exact sequence length used by the parity E2E
  test.

Production bucketing is not part of v0. A caller may explicitly build and retain
plans for chosen bucket sizes. Padding and masks are ordinary inputs. A plan is
recompiled for any shape/dtype or structural change, including a different
operator DAG or a kernel-selection-affecting attribute.

After warm-up, one decode step updates input/scalar buffers, submits the retained
plan, waits, and reads the declared logits output. It performs no tensor graph
construction, inference, pipeline creation, planning, or tensor-layer allocation.

`Plan.Memory` reports the device graph's unaliased, peak, and allocated transient
bytes. Weights, KV state, inputs, and outputs are caller buffers and are never
included or aliased.

## Fusion

v0 fuses by authorship only: RMSNorm, SwiGLU, Linear's bias epilogue, Softmax's
scale/mask, and eligible Attention variants. Every fused kernel keeps a composed
or naive independent reference. There is no graph pattern matcher and no runtime
shader generation.

## Post-v0 scope

Post-v0 work includes quantized logical dtypes and quantized GEMM/Rows, model
formats, tokenizers, sampling operators and policy, automatic
fusion, production prefill bucketing, plan caches, paged or quantized KV caches,
multi-sequence scheduling, vision/SSM operators, sparse tensors, multi-device
execution, autotuning, autodiff, and training.

Quantization requires its own representation and numeric contract before code:
plane sizes and packing order, scale/zero-point semantics, import/repacking
ownership, legal views, quantizer rounding/clamping, kernel variants, and derived
scheme error bounds. No three-scheme commitment is made by v0.

## Open questions

- Whether post-v0 GEMM should accept a strided K axis or require explicit
  materialization.
- Whether every backend can vary the host-supplied dispatch count cheaply; v0
  may mask to capacity on a backend whose replay object bakes it in.
- Whether a later plan cache belongs in this package or in a model runtime. Its
  stable identity requirements are fixed above even though ownership is not.
- Whether a later server plan may own several transient sets to permit multiple
  in-flight submissions without duplicating immutable graph structure.

None blocks the concrete CPU/Metal v0 path.

## Testing

[`011-conformance-harness.md`](011-conformance-harness.md) owns execution and
comparison mechanics; [`008-numerics.md`](008-numerics.md) owns bounds. Required
tensor-layer cases are:

- every primitive against an independently written f64 or exact integer
  reference, using the operator's derived budget;
- every authored fused operator against its composed/naive reference;
- views against explicit materialization, including the documented axis order;
- state-version validation and KV write/read hazards;
- plan binding failures for missing, extra, undersized, overlapping, closed, and
  wrong-device buffers;
- portable f16 storage on a device with native f16 arithmetic forced absent;
- plan memory matching the device graph's actual transient high-water mark;
- a fresh plan and repeated submissions of the same retained plan producing the
  same bounded result for different inputs;
- overlapping submissions of one plan returning `ErrGraphInFlight`;
- same-backend determinism for kernels that do not use class-E atomic float add;
- the two-layer golden-model E2E; and
- incremental decode for N tokens matching the minimal prefill of those N tokens
  within the composed operator budget, on CPU and Metal.

Every v0 operator has unit coverage in the corpus and is exercised through at
least one plan-level E2E. The package must exceed 90% statement coverage under
the CPU backend before M7 is complete.
