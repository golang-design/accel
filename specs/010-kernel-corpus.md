---
title: "Kernel corpus: unquantized v0 inventory, variants, and selection"
status: in progress
layer: tensor
depends_on:
  - 002-compute-model.md
  - 004-kernel-authoring.md
  - 007-tensor-layer.md
  - 008-numerics.md
---

# Kernel corpus

This spec owns the kernels required for 007's unquantized f16/f32 v0. It turns
the tensor operator list into an implementable inventory: package layout, stable
identities, supported layouts and shapes, required variants, deterministic
selection, numeric recipes, and per-kernel test obligations.

Quantized kernels, sampling, graphics, automatic fusion, native f16 arithmetic,
and autotuning are not in this corpus.

## 1. Package and registration layout

Kernels live below the future tensor package so the device layer never imports
them:

```
tensor/
  internal/kernels/
    elementwise/
    layout/
    indexing/
    norm/
    matmul/
    attention/
```

Each package contains ordinary Go kernel source and its generated registration
file. It imports the root `accel` device package. The tensor runtime imports the
kernel packages and builds a registry; the root device package never imports the
tensor layer, preserving 000's layering rule. Conformance tests use external test
packages when they need both the root API and registered corpus, avoiding an
import cycle.

The generator emits one immutable record per variant:

```go
type KernelID struct {
	Semantic string // e.g. "matmul"
	Variant  string // e.g. "tiled_16x8x16_contiguous"
	Storage  DType
}

type KernelMeta struct {
	ID           KernelID
	SourceHash   [32]byte
	Layouts      []LayoutClass
	Requirements accel.Requirements
	Workgroup    accel.WorkgroupSize
	Numeric      NumericRecipe
	Priority     int
}
```

IDs and variant names are stable test/reporting surface. Renaming one changes
goldens and plan selection reports deliberately. `NumericRecipe` names 008's
primitive/composed budget rule; no record contains a literal tolerance.

## 2. Layout classes

The corpus recognizes only these compile-time layout classes:

| Class | Definition | Consumer |
| --- | --- | --- |
| `contiguous` | canonical suffix-product strides | every kernel fast path |
| `row_contiguous` | last axis stride 1; arbitrary non-overlapping outer strides | elementwise, Rows, norm, RoPE |
| `broadcast_read` | legal read-only zero strides | elementwise and bias/mask reads |
| `general_read` | any legal 007 non-overlapping read layout | Contiguous materialization only |

No v0 write kernel accepts zero-stride or overlapping output layouts. MatMul
requires unit stride on the left operand's K axis and the right operand's N axis,
matching contiguous `x[...,M,K]` and `w[...,K,N]`. A layout outside a selected
kernel's class must be explicitly materialized by `Contiguous`; selection never
silently adds a copy.

## 3. Required inventory

Every row is required in both f32 storage and f16 storage with f32 evaluation,
except integer data noted explicitly. f16 variants use conversion on load/store
and do not require `CapF16Arithmetic`.

### Flat kernels

| Semantic ID | Required variants | Layouts | Notes |
| --- | --- | --- | --- |
| `add` | contiguous, row-contiguous/broadcast | read supports broadcast; output row-contiguous | Binary Add. |
| `mul` | contiguous, row-contiguous/broadcast | same | Binary Mul and composed reference building. |
| `scale` | contiguous, row-contiguous | runtime f32 scalar | Scalar is decoded from uniform data. |
| `silu` | contiguous, row-contiguous | row-contiguous | Uses class-C exp and division recipes. |
| `swiglu` | contiguous, row-contiguous | equal inputs | Authored fused SiLU plus multiply. |
| `contiguous_copy` | general-read to contiguous | any legal read layout | Separate dtype variants include i32/u32. |
| `rows` | contiguous table, contiguous ids/output | ids i32/u32 | Strict path checks id range. |
| `scatter_rows` | contiguous rows/state | state read-write | Strict path checks runtime index/capacity. |
| `rope` | row-contiguous | runtime positions/base | Pair rotation over declared rotary dimension. |

These kernels are valid for M2's direct flat executor unless they use persistent
state; `scatter_rows` first executes through a public graph in M3.

### Cooperative kernels

| Semantic ID | Required variants | Workgroup/shape rule | Notes |
| --- | --- | --- | --- |
| `reduce_sum` | tree | 128 invocations; arbitrary reduced length through guarded loads | Internal building block and numeric proof. |
| `rmsnorm` | row | one or more workgroups per row chosen deterministically from width | f32 sum of squares and rsqrt. |
| `softmax` | row, causal-mask row | selected from axis length and limits | stable max subtraction; optional broadcast mask and runtime scale. |
| `matmul` | naive reference, tiled `16x8x16` | M/N/K guarded; required inner strides are 1 | Portable f16/f32 storage, f32 accumulation. |
| `linear` | tiled with bias epilogue | MatMul rules; broadcast-compatible bias | Shares tile body with MatMul, distinct stable ID. |
| `matvec` | row/vector | selected exactly when M=1 | Decode-specialized MatMul implementation. |
| `attention_decode` | fused contiguous KV; paged KV; f16 KV; batched paged | qSeq=1, headDim divisible by 8 and <=128, qHeads%kvHeads=0; any cache capacity | Required on CPU and Metal v0. The cache is walked a block at a time with a running softmax ([044](044-unbounded-context.md)), so the workgroup bounds a block and not a cache. The composed path is the reference where it is expressible, which is `kvHeads == 1`. |
| `attention_prefill` | causal contiguous KV | a sequence of query positions, a base position within the cache | Registered. Bounded by the causal limit rather than the cache, so a prefill scores the triangle the mask describes. |
| `attention_prefill_paged` | **not registered** | — | The gap a consumer found by measuring rather than by reading a refusal (accel issue 10). A paged decode is only useful over blocks a paged prefill wrote, so this is the first operation of every request in a paged design and its absence makes cross-request prefix sharing inexpressible. `tensor.Attention` refuses `Pages` on a prefill until it exists; [044](044-unbounded-context.md) §5 applies, since a prefill already loops over the cache and the indirection goes where the loop is. |
| `gather_rows` | f32 table; int8 table with per-block scales | rows selected by a u32 index tensor | **No f16 variant**, which a consumer costed (accel issue 11): the embedding table is the largest single tensor in a small model, so an otherwise-f16 model holds it at f32 or quantizes the tensor most sensitive to it, with nothing in between. A tied LM head pays twice, since the two planes must then differ in dtype as well as layout. |
| `quant_matmul` | tiled, per-element grid at every M | int8 weights with per-block scales | **No matrix-vector selection**, which the unquantized path has at M=1 (accel issue 11). M=1 is every decode step, and the quantized path is the one a large model lands on, so the configuration chosen for size is the one with the fewest specializations. |

The `16x8x16` GEMM tile means a 16-wide output-N tile, eight output rows, and a
16-wide K step, for 128 invocations and a portable shared-memory footprint.
Variants at other tile sizes are post-v0 and may not be selected under the same
ID.

The fused decode attention kernel is not represented in
`accel.Capabilities`. It is eligible only when a registered variant matches the
concrete dtype/layout/shape and its primitive `Requirements` fit the device.
Minimal prefill may use the composed MatMul → Softmax → MatMul path. Tests can
remove the fused record from a runtime registry to exercise selection absence on
any CPU.

### Added for M7 — 2026-08-23

Three kernels the tensor layer needed and the corpus did not have. Each carries
this section's proof obligations rather than being added because something
upstream was blocked on it:

| Kernel | Obligation met |
| --- | --- |
| `CastF32ToF16` | round-to-nearest-even narrowing; agrees bit for bit between backends, over inputs with bits below f16's precision so the rounding actually happens |
| `CastF16ToF32` | exact widening: every f16 value is an f32 value, so anything but equality is a bug rather than a rounding difference |
| `AttentionPrefill` | matched against a straight quadruple loop in f64 at four shapes, and **equal to incremental decode over the same cache** |

**A conversion is a kernel because the alternative is three synchronisation
points.** A value produced on the device and needed in another format would
otherwise be read back, looped over on the host, and uploaded again, inside what
should be one graph.

**Causal masking is built into the prefill rather than an option**, because a
prefill that let a token attend to its own future is not a slower answer, it is
a different model — and it would still produce plausible numbers, sum to one,
and pass every shape check. So the test attacks the mask directly: it changes a
cached value that only a *later* query position can see, asserts that the
earlier positions do not move, and asserts that a later one *does* — because
without that second half the first passes when nothing reads V at all.

**Prefill-versus-decode parity is the reason both exist.** If they disagreed, a
model would produce different text depending on whether it had been prompted or
generated, which is the least debuggable failure this project could ship. They
agree within the softmax's reduction budget rather than bit for bit, because the
two reduce over different numbers of lanes and §7 bounds that rather than
forbidding it.

## 3.1 What is built — 2026-08-23

Every kernel in both tables above exists on the CPU backend, in
`internal/testkernels`, each checked against an independently written
higher-precision reference and each authored form checked against its generated
lowering.

| Family | Built | Not yet |
| --- | --- | --- |
| Flat: `add`, `mul`, `scale`, `silu`, `swiglu`, `rows`, `scatter_rows`, `rope` | all | `contiguous_copy`'s dtype variants, and the layout classes below |
| Cooperative: `reduce_sum`, `rmsnorm`, `softmax`, `matmul` tiled, `linear`, `matvec`, `attention_decode` | all | `softmax`'s causal-mask variant, `matmul`'s naive reference |

**What is not built is the *registry*, not the kernels.** §4's deterministic
selection, the variant records, the layout classes, and the stable IDs are a
tensor-layer concern that belongs with [007](007-tensor-layer.md)'s `Runtime`,
and none of it exists. What exists is the arithmetic each ID names, which is the
half M7 needs proved before selection can be trusted to pick between them.

### The rules that turned out to matter

Each of these produces a *plausible* result when wrong — right shape, right
magnitude, wrong meaning — so each has a test confirmed by reinstating the bug
rather than by passing.

| Kernel | The rule | What breaks without it |
| --- | --- | --- |
| `softmax` | subtract the row maximum | exp overflows f32 above ~88, every term is Inf, the quotient is NaN |
| `rmsnorm` | ε under the square root | a row of zeros divides by zero |
| `silu` | negate the exponent | the algebraically identical form overflows at the same 88 |
| `attention_decode` | inactive lanes contribute −∞ to the maximum | zero wins over negative scores and shifts every exponent |
| `attention_decode` | query heads group contiguously onto KV heads | every head still attends to *something* |
| `rows` | check the id | reads another token's vector |
| `scatter_rows` | drop an out-of-range write | clamping corrupts a real row |
| `rope` | honour the position offset | every token rotates as though it were the first |

### Running an authored cooperative kernel

[004](004-kernel-authoring.md)'s fifth level compares a generated lowering
against its authored function, and a cooperative kernel's authored form cannot
be run one invocation at a time. The obvious workaround — every invocation
through the whole function once per barrier — is **unsound**: it holds only
while every pre-barrier statement is idempotent, and a tree reduction is not,
since it overwrites the array it reduces. Three tests written that way passed by
luck and a fourth produced NaN, which is how it was found.

`kernel.RunAuthored` gives each invocation a goroutine behind a cyclic barrier,
which is what a workgroup is. That is deliberately *not* how the backend runs a
kernel — the scheduler's one-at-a-time advance is deterministic and this is not
— and the two arriving at the same answer by different means is what makes the
comparison worth making.

## 4. Deterministic selection

Selection is a pure function of registry version, semantic operation, concrete
dtype/shape/layout, and device limits/capabilities. It does not benchmark,
inspect input values, or depend on registration order.

Rules are applied in this order:

1. reject dtype, rank, shape, or writable-layout violations;
2. filter variants whose declared layout classes do not match;
3. filter variants with unmet primitive requirements or limits;
4. for Attention, prefer eligible `attention_decode`, otherwise lower the
   composed definition;
5. for MatMul with M=1 select `matvec`; otherwise select portable tiled MatMul;
6. for elementwise operations select contiguous before row/broadcast general;
7. break any remaining tie by higher declared priority, then lexical KernelID.

The generated registry rejects duplicate KernelIDs and duplicate equal-priority
selectors at init/test time. A plan records selected IDs plus rejected candidates
and reasons. Selection failure names the semantic op, shape/layout, and every
unmet requirement.

### The first half of it exists — 2026-08-23

The generator emits `Kernels`, a slice of every kernel record in the package, in
source order:

```go
// Kernels is every kernel this package generated, in source order.
var Kernels = []*accel.Kernel{&AddKernel, /* ... */}
```

**It is the enumeration the registry above needs, and none of the selection.**
There is no KernelID, no selector, no priority, and no duplicate rejection; what
it provides is the property that a pass over the corpus cannot miss a kernel
somebody added. It landed for [021](021-metal-bringup.md)'s device-compile test,
which had to compile every kernel carrying MSL and would otherwise have kept a
hand-written list beside the corpus — and a hand-written list goes stale the
moment the corpus grows, silently, because the new kernel looks exactly like one
that passed.

Generated rather than reflected, because a package's variables cannot be
enumerated at run time in Go, and generation already knows the answer.

The rest of this section is unbuilt: it arrives with M7, which is the first
consumer that has to *choose* a kernel rather than name one.

## 5. Numeric recipes

| Kernel family | 008 recipe |
| --- | --- |
| Add/Mul/Scale/layout/indexing | class A/D/Special per element |
| SiLU/SwiGLU | composed class-C division/transcendentals and multiplication sensitivity |
| Reduce/RMSNorm/Softmax | actual tree depth plus class-C propagation |
| MatMul/MatVec/Linear | per-output dot-product depth and magnitude; bias composition |
| RoPE | class-C absolute ceilings over 008's bounded sin/cos domain |
| Attention | score MatMul + Softmax + value MatMul composed trace |

RoPE compilation proves that its declared maximum position/base keep every angle
inside 008's bounded sin/cos domain. It rejects a larger domain rather than
running with an ad hoc tolerance.

## 6. Per-kernel obligations

Every variant has:

- a generated-source/hash freshness test;
- a target-compiler acceptance test for each emitted target;
- an independent reference that does not call or mechanically translate the
  kernel body;
- dtype, minimum, typical, non-tile-multiple, and zero-work cases where legal;
- every guarded tail independently exercised;
- layout cases for every declared class and rejection for every excluded class;
- exact Special cases for bounds, NaN/category, overflow, and indices where
  relevant;
- a derived 008 comparison with a stable budget trace;
- CPU direct or graph execution as soon as its required execution model exists;
- CPU-versus-Metal execution once Metal accepts the corpus; and
- mutation/race tests for any read-write kernel.

For authored fusion, the fused result is checked both against an independent
higher-precision reference and against the separately executed semantic
decomposition. Passing only the decomposition comparison is insufficient because
both paths can share a mistaken formula.

## 7. Family-specific cases

- Elementwise: mismatched ranks, every broadcast axis, zero stride reads, and
  exact output-shape inference.
- Rows/ScatterRows: first/last valid row, invalid signed and unsigned IDs, first
  and final KV positions, and a write followed by a read in the same plan.
- Reductions: lengths 1, 2, 127, 128, 129, and a multi-workgroup length; positive,
  mixed-sign, cancellation, and magnitude-skew inputs.
- Norm/Softmax: constant rows, dominant element, masked tail, causal boundary,
  very small but normal values, and axes not divisible by the workgroup size.
- MatMul: M/N/K each independently equal to 1, tile-1, tile, tile+1; batch
  broadcasting; cancellation; and rejection of each non-unit required inner
  stride.
- Attention: GQA ratio 1 and greater than 1, current length 1/capacity, fused
  present/absent selection, and fused/composed parity.
- RoPE: rotary dimension 0/2/headDim, multiple positions, and the maximum angle
  admitted by its future normative numeric domain.

## 8. Coverage and E2E

The generated corpus manifest is the coverage source of truth. A test fails if a
registered v0 variant lacks a reference, numeric recipe, CPU case, or required
Metal case. The kernel packages collectively exceed 90% statement coverage on
the CPU path.

Plan-level E2Es in 011 must exercise every semantic ID at least once. M5's GEMM
E2E exercises the tiled family; M7's two-layer model and prefill/decode parity
exercise all tensor-required families and both fused-attention selection paths.

## 9. Post-v0 extensions

Quantized weights require a separate representation/numerics design and distinct
stable IDs. Additional tiles, native f16 arithmetic, cooperative matrices,
sampling primitives, production prefill attention, generic Go kernels,
autotuning, and runtime-generated fusion are separate scoped specs. They do not
change a v0 variant's semantics or identity.
