---
title: "Tensor layer: values, state, operators, and plans"
status: in progress
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

## What is built, and where it differs from the drawing above — 2026-08-23

The v0 tensor layer exists, in `golang.design/x/accel/tensor`, built across
[024](024-tensor-bringup.md), [025](025-tensor-operators.md) and
[026](026-tensor-decode.md). This section audits the built API against the one
this spec draws, because a spec whose signatures no longer match the code is
wrong in the way that costs a reader most: they trust it.

**Four signatures differ, and each difference is a decision rather than a
drift:**

| Drawn here | Built | Why |
| --- | --- | --- |
| `ScatterRows(b, state, rows, indexName string)` | `ScatterRows(b, state, rows, ids *Tensor)` | the indices are device data that a previous operator may have produced, not a host-side name; a name would force them through the host |
| `RoPE(b, x, positions *Tensor, rotaryDim, baseName)` | `RoPE(b, x, rotaryDim, baseName, positions *Tensor)` | **this spec was right and the first build was not.** The built form took a starting offset and derived each row's position as `row + offset`, which is correct for one sequence and wrong for two: in a batched decode the row index is the *slot*. Corrected 2026-08-24 after a consumer reported it — see [043](043-per-row-values.md) |
| `SoftmaxOptions{Axis, ScaleName, Mask, Causal}` | `SoftmaxOptions{Axis}` | the registered kernel has no mask parameter and no scale; `Axis` must be the last, which is the only axis it reduces over |
| `AttentionOptions{CurrentLengthName, Causal}` | `AttentionOptions{Lengths *Tensor, Pages *Tensor, Block, ScaleName, BaseName}` | the scale is named for the reason every other model constant is; `BaseName` is the prefill's first query position, which decides what the causal mask hides and which a boolean cannot say. `Lengths` is a tensor because cache lengths are independent across a batch, and `Pages` reaches the paged cache [030](030-paged-kv.md) built and nothing could call — both 2026-08-24, see [043](043-per-row-values.md) |

**`Causal` is not a flag anywhere, and that is the substantive change.** This
spec made it a compile-time attribute; the built form makes it the *kernel*.
`AttentionDecode` attends over the whole cache and `AttentionPrefill` masks
causally, and which one runs follows from the query's rank. A boolean would have
been a third thing that could disagree with the two kernels.

**Two additions this spec does not draw:**

- **`Cast(b, x, to DType)`**, which appears in §"Layouts, views, and
  broadcasting" as an operation that copies and converts but is absent from the
  operator contracts below. It is an operator rather than an implicit rule at
  every dtype boundary for the reason this spec gives elsewhere: a conversion
  costs a pass over the data and changes the numbers, so a caller writes it.
  `Add` refusing two dtypes and `Cast` existing are one decision seen from two
  sides.
- **`Builder.Err()`, `Runtime.Device()`, `Plan.Scalars()`**, and the accessors on
  `Shape` and `Tensor`. Small, and each exists because something needed to ask a
  question the drawing left no way to ask: whether a partly built model is
  already wrong, which device a runtime lowers to, and what a caller has to bind
  when they no longer hold the builder.

**Absent, with what each waits on:**

| | |
| --- | --- |
| `Contiguous` | a gather kernel with strides in a uniform block; [010](010-kernel-corpus.md) registers none |
| `Softmax`'s mask and causal | a kernel with a mask binding |
| `Squeeze`, `Unsqueeze` | `Reshape` expresses both; absent rather than aliased |
| `LayerState` binding | **built, 2026-08-24.** A graph slot binds a *window* of a port -- a range in elements -- so one cache serves every layer and a caller binds it once. The entry above was wrong about the cause: `accel.SlotBinding` takes a `BufferView` and always could express a sub-range; `LayerState`'s offset simply never reached the binding |
| composed attention as a fallback | not buildable, not merely absent: the composed form needs a matrix multiply per head and 025 does not broadcast leading axes, so it exists only at `kvHeads == 1`. It stays the correctness reference over the shapes it can express. See the correction above and [044](044-unbounded-context.md) deviation 3 |
| a plan cache | post-v0 by this spec's own §"Ownership and core types" |

**One rule this spec states that the built layer had to enforce rather than
inherit.** A `Plan` "has the Graph's one-submission-in-flight restriction" — but
a plan *binds* when you submit and the graph *runs* when the queue reaches it,
so a second submission rebound the first one's slots before it ran. Two
submissions with different inputs and outputs produced one result and both
fences reported success. A lifetime rule enforced where a resource is owned does
not automatically hold where it is bound on someone else's behalf.

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

func Cast(b *Builder, x *Tensor, to DType) *Tensor

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
	Lengths   *Tensor // one entry per sequence
	Pages     *Tensor // optional page table; nil is a contiguous cache
	Block     int     // positions per block, with Pages
	ScaleName string
	BaseName  string // prefill only
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

**Correction, 2026-08-24: it is the reference and not a fallback.** This spec
said fused attention is "runtime kernel selection, not a device capability", and
that `Compile` "otherwise selects the composed graph". The second half cannot be
built. Grouped-query attention has several query heads sharing one key/value
head, so the composed form needs one matrix multiply per head, and
[025](025-tensor-operators.md) multiplies two matrices with no leading-axes
broadcast. The composition therefore exists only at `kvHeads == 1`, which no
model this serves uses.

What holds: the composed graph is the correctness reference, and the corpus
tests run it over the shapes it can express. What replaces the fallback: nothing
needs to. [044](044-unbounded-context.md) removed the shape the fallback existed
to catch — a cache longer than a workgroup — by making the fused kernels walk
the cache a block at a time, so the fused path takes every capacity. A shape a
registered variant genuinely cannot take is refused by name.

`Plan.Selections` still reports which variant ran and why, which is the part of
"runtime kernel selection" that was real: the contiguous, paged, f16 and prefill
kernels are selected from one operator call.

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

One constraint on that design is already fixed by layer 1 and is recorded here so
it is not re-derived from scratch: **001 types every buffer by dtype, and an
interleaved block struct has no dtype.** A GGML-style buffer of quant-plus-scale
blocks could only be smuggled through as an opaque byte buffer, giving up the
size and type validation 003 performs at build. So a quantized tensor is a small
set of separately typed plane buffers rather than one block-structured buffer,
which also keeps each plane naturally aligned for coalesced loads and lets a
per-layer slab stay an ordinary 001 view of each plane. The costs are equally
fixed: a tensor is no longer one buffer view, and importing a file whose weights
are stored as blocks requires a repacking pass at load.

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
