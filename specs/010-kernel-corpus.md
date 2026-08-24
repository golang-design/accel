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
| `scatter_rows` | contiguous rows/state; f16 rows into an f16 state | state read-write | Strict path checks runtime index/capacity. The f16 variant closes the write half of accel issue 13: an f16 KV cache was readable and unwritable, so a model could attend over a cache no kernel could populate from inside the graph. A scatter does no arithmetic and this one does not convert either -- both sides are f16, so the two lowerings move the same bits and the differential keeps no budget at all. |
| `rope` | row-contiguous | runtime positions/base | Pair rotation over declared rotary dimension. |

These kernels are valid for M2's direct flat executor unless they use persistent
state; `scatter_rows` first executes through a public graph in M3.

### Cooperative kernels

| Semantic ID | Required variants | Workgroup/shape rule | Notes |
| --- | --- | --- | --- |
| `reduce_sum` | tree | 128 invocations; arbitrary reduced length through guarded loads | Internal building block and numeric proof. |
| `rmsnorm` | row | one or more workgroups per row chosen deterministically from width | f32 sum of squares and rsqrt. |
| `softmax` | row, causal-mask row | selected from axis length and limits | stable max subtraction; optional broadcast mask and runtime scale. |
| `matmul` | naive reference, tiled `16x8x16` over f16, over f32, and mixed f32 activations against f16 weights | M/N/K guarded; required inner strides are 1; the *pair* of operand widths selects the variant | f32 accumulation in every variant. The mixed variant closes accel issue 14: a transformer's two operands are never the same width, because its activations are f32 like every other operator in the tensor layer and its weights are f16 because a four billion parameter model is 16 GB in f32. Defensible because it is not new arithmetic -- the f16 variant already widens both operands before multiplying, so on activations f16 holds exactly the two agree bit for bit, which is the test. The reverse pair, an f16 activation against an f32 weight, is refused: that is the memory decision made in the expensive direction, and it is what keeps the same-width rule for two operands that are both activations. No mixed variant of `matvec` exists, so a mixed decode takes the tile and the selection reports the seven idle rows. |
| `linear` | tiled with bias epilogue | MatMul rules; broadcast-compatible bias | Shares tile body with MatMul, distinct stable ID. |
| `matvec` | row/vector | selected exactly when M=1 | Decode-specialized MatMul implementation. |
| `attention_decode` | fused contiguous KV; paged KV; f16 KV; paged f16 KV; batched paged | qSeq=1, headDim divisible by 8 and <=128, qHeads%kvHeads=0; any cache capacity | Required on CPU and Metal v0. The cache is walked a block at a time with a running softmax ([044](044-unbounded-context.md)), so the workgroup bounds a block and not a cache. The composed path is the reference where it is expressible, which is `kvHeads == 1`. |
| `attention_decode_batched` | batched paged KV | B sequences, one token each, one page-table row per sequence | Reached by a rank-4 `q`, `[batch, qSeq, qHeads, headDim]`. Paged only, and that is not an accident of the corpus: sequences stepping together have different lengths, so a contiguous cache would pad each to the longest. |
| `attention_prefill` | causal contiguous KV; f16 KV | a sequence of query positions, a base position within the cache | Registered. Bounded by the causal limit rather than the cache, so a prefill scores the triangle the mask describes. The f16 variant closes the read half of accel issue 13: a prefill is the first operation of every request, so without it a narrow cache served every step except the one that fills it. |
| `attention_prefill_batched` | **not registered** | — | Several sequences prefilling together, which is rank 4 with `qSeq > 1` — a shape the operator already parses and refuses for want of this kernel (accel issue 16). Less of a special case than it looks: [029](029-plan-cache.md) buckets prefills, so a batch of bucketed prompts has a uniform `qSeq` by construction. A ragged batch — a prefill chunk beside decode steps — is a different shape and [040](040-batch-scheduler.md) §8.2 owns it. |
| `attention_prefill_paged` | causal KV through a page table | a prefill's, plus a block size | Registered, closing accel issue 10. A paged decode is only useful over blocks a paged prefill wrote, so this is the first operation of every request in a paged design. One indirection and nothing else: [044](044-unbounded-context.md) §5 predicted the shape, since a prefill already walks the cache in blocks. Bounded by the page table's reach rather than the pool's, and checked against the same positions gathered contiguously — exactly, because an addressing change computes the same sums in the same order. |
| `gather_rows` | f32 table; f16 table; int8 table with per-block scales | rows selected by a u32 index tensor | The f16 variant closes accel issue 11: an embedding table is the largest single tensor in a small model and had no width between f32 and int8. A gather does no arithmetic, so a narrow table is 002's storage rule with nothing to lose -- the value read is the value written, one conversion wider. The result is f32 whatever the table is, because a normalize follows. |
| `linear_attention_chunked` | **not registered** | — | A gated-delta / SSM recurrence, which is three of every four layers in the hybrid models the open-weights frontier has moved to (accel issue 17). **In scope**, and two things are prior to a kernel: the recurrence is sequential in $t$, so a prefill is a chunked *scan* rather than a batch of independent rows; and the state is matrix-valued **per sequence**, not per position, which [007](007-tensor-layer.md)'s `State` cannot express — see [043](043-per-row-values.md) §9. Not scheduled: the consumer's dense path is unblocked and this decides whether a 27B hybrid target is reachable, not whether they can build. |
| `grouped_gemm` | **not registered** | — | $E$ independent $[n_e, K]\times[K, N]$ products with the $n_e$ decided at runtime, which is what a mixture-of-experts layer needs and what makes it worth having: ~30B parameters at ~3B of compute per token (accel issue 18). **In scope.** The naive form — run every expert and mask — is expressible today and does $E/k$ times the work, which inverts the reason MoE exists. The routing itself composes: a small `MatMul` for the gate and [028](028-sampling.md)'s top-$k$ over $E$. Not scheduled, and if built the ragged extent is [043](043-per-row-values.md) §9's, built once rather than a third time. |
| `quant_matmul_superblock` | **not registered** | — | GGUF's K-quants, which is where the ecosystem's pre-quantized checkpoints are (accel issue 15). A `Q4_K` super-block is 256 weights as eight sub-blocks of 32, each carrying a 6-bit scale and a 6-bit minimum, over two fp16 super-scales: $w_i = d\cdot s_j\cdot q_i - d_\text{min}\cdot m_j$ with $j=\lfloor i/32\rfloor$. `quant` registers one representation — int8 with one fp16 scale per 32, no minimum and one level of scale — so nothing reads a super-block. Not scheduled: the consumer reads safetensors and quantizes at load, and says so. The two workarounds are both bad and are why this is recorded rather than left implicit — dequantizing K-quants at load discards the memory saving that is the whole reason to read GGUF, and requantizing into int8 stacks two lossy steps. |
| `quant_matmul` | per-element at every M and matrix-vector at M=1, each over f16 and over f32 activations | int8 weights with per-block scales; the activation's width selects within each shape | The M=1 variant closes accel issue 11's second half, and is the same selection `matvec` is for the unquantized path. Its reduction is a tree over lanes where the general kernel folds sequentially, so its rounding differs and its bound does not: 027 states the error over the number of terms rather than their order. The f32-activation pair closes accel issue 14: int8 is the width a model reaches for *because* it is large, so requiring f16 activations put a Cast in front of every projection of the configuration least able to afford the pass. Both shapes have one rather than only the general kernel, because M=1 is every decode step and closing the refusal there alone would have repeated issue 11. Defensible on the same ground as the mixed GEMM: each is its f16 form with the activation load already wide, so on activations f16 holds exactly the two agree bit for bit, and 027's bound is stated against an evaluation neither changed. The quant and scale planes keep their widths in every variant. |

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

### Added for accel issue 13 — 2026-08-24

An f16 KV cache was read-only and un-pageable. `AttentionDecodeF16` read a narrow
cache, and nothing wrote one and nothing paged one, so the two savings the narrow
read argues for did not compose and a model could not populate the cache it was
meant to attend over. The three variants are the widening the decode kernel had
already taken, applied where the sequence needed it:

| Kernel | Obligation met |
| --- | --- |
| `ScatterRowsF16` | equal to `ScatterRows` over the widened rows, element for element, over a state seeded to a value no write produces so a dropped write is visible; one id past the capacity, so the range check is compared and not only the addressing |
| `AttentionPrefillF16` | equal to `AttentionPrefill` over the same cache widened, **bit for bit**, at three head geometries and with `Base` moved off zero |
| `AttentionDecodePagedF16` | the same, through a page table that is out of order and non-adjacent, so a kernel that ignored the table would not compare equal |

The exactness claim is the one the f16 decode already makes: the widening loses
nothing, so the narrow kernel and the wide one must agree exactly and a tolerance
would pass a kernel that read the wrong element. The f32 side of each comparison
is **derived** from the narrowed values rather than seeded beside them, so the
claim does not rest on the seed being exact in f16.

The sequence is checked end to end as well -- a prompt's KV scattered into an f16
cache, a prefill over it, one more position scattered, a decode step over the
longer cache -- because each kernel passing alone is what the closed issue #4
already had.

### Added for accel issue 14 — 2026-08-24

Four kernels for the widths a transformer actually has. The corpus assumed the
two operands of a product share a storage width; a transformer's never do and
cannot, so the assumption did not make the graph narrow, it made the graph
`Cast` — four per layer, 144 dispatches per forward pass at 36 layers.

| Kernel | Obligation met |
| --- | --- |
| `MatMulTiledF32F16` | equal to `MatMulTiled` **bit for bit** on activations f16 holds exactly, across a table of shapes with every guarded tail, because the f16 form already widens both operands before multiplying |
| `QuantMatMulF32` | equal to `QuantMatMul` bit for bit on the same class of activations; 027's error bound is stated against an evaluation the widening does not change |
| `QuantMatVecF32` | equal to `QuantMatVec` bit for bit, over shapes including K past one lane's fold — below that, `k` and `lid` are the same number and a lane-indexed load looks correct |
| `CastBF16ToF32` | exact widening, and more strongly than f16's: bf16 is f32's top half, so it is a shift and every input is a witness. Checked over magnitudes outside f16's range, which is what the target being f32 is for |

**An authored form agreeing with its own lowering is not enough for a variant.**
It is the check every kernel carries, and it passes on a kernel that reads the
wrong element of a tile: the authored form and the generated one read the same
wrong element. The obligation above is therefore a comparison against the
*narrow* form on values both hold exactly, which fails on a swapped tile index —
verified by reinstating one.

**bf16 needed the MSL target, not only a kernel.** The emitter refused
`OpBF16ToF32` and `OpF32ToBF16` together because `bfloat` is a Metal family
capability. That is true of the type and irrelevant to the widening: the binding
is `ushort` and the conversion is `as_type<float>(uint(x) << 16)`, which every
family has, and `ir.go` already forbids arithmetic on a storage kind. The
narrowing keeps the refusal, because it has to round.

## 3.1 What is built — 2026-08-23

Every kernel in the tables above exists on the CPU backend, in
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
