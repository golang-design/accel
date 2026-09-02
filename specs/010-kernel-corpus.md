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
| `attention_ragged` | `AttentionRagged` | 2026-08-26 | A step whose sequences contribute different numbers of query tokens, which is what a batched prefill is and what lets a prefill chunk share a dispatch with decodes (accel issue 16). One workgroup per (token, head), a segment lookup for the token's sequence, and [046](046-segmented-extents.md) §2.2's per-token causal position. |
| `segment_offsets` | `SegmentOffsets` | 2026-08-26 | The exclusive prefix sum a segmented extent needs. Rows+1 entries so a kernel owning row r has both ends of it and the row's count is off[r+1]-off[r] — nothing binds the counts as well, so there is no second buffer to disagree. |
| `attention_prefill_batched` | **superseded 2026-08-26** | — | **Registered as `attention_ragged` above, in a more general form.** The rectangular shape this row asked for gives every sequence the same token count; [046](046-segmented-extents.md) built the ragged one instead, which expresses it and the mixed prefill-plus-decode step with one kernel. Original text: several sequences prefilling together, which is rank 4 with `qSeq > 1` — a shape the operator already parses and refuses for want of this kernel (accel issue 16). Less of a special case than it looks: [029](029-plan-cache.md) buckets prefills, so a batch of bucketed prompts has a uniform `qSeq` by construction. A ragged batch — a prefill chunk beside decode steps — is a different shape and [040](040-batch-scheduler.md) §8.2 owns it. |
| `attention_prefill_paged` | causal KV through a page table | a prefill's, plus a block size | Registered, closing accel issue 10. A paged decode is only useful over blocks a paged prefill wrote, so this is the first operation of every request in a paged design. One indirection and nothing else: [044](044-unbounded-context.md) §5 predicted the shape, since a prefill already walks the cache in blocks. Bounded by the page table's reach rather than the pool's, and checked against the same positions gathered contiguously — exactly, because an addressing change computes the same sums in the same order. |
| `attention_prefill_paged_f16` | causal KV through a page table, narrow | a prefill's, plus a block size | Registered 2026-08-27, closing [#25](https://github.com/golang-design/accel/issues/25). `AttentionPrefillPaged`'s body with three edits and no others: k and v become f16 in the signature and the two loads that read them widen on load — the same three that make `attention_prefill_f16` and `attention_decode_paged_f16`. The gap was a *combination*: a narrow cache had a contiguous prefill and a paged decode, and not the pair, which is the only pair a paging consumer reaches. |
| `gather_rows` | f32 table; f16 table; int8 table with per-block scales | rows selected by a u32 index tensor | The f16 variant closes accel issue 11: an embedding table is the largest single tensor in a small model and had no width between f32 and int8. A gather does no arithmetic, so a narrow table is 002's storage rule with nothing to lose -- the value read is the value written, one conversion wider. The result is f32 whatever the table is, because a normalize follows. |
| `linear_attention` | `LinearAttention` | 2026-08-26 | The gated delta recurrence, stepped sequentially over [046](046-segmented-extents.md)'s extent. One workgroup per (sequence, head) walking its own tokens, three passes over the state per token. Makes the layer **expressible**; the chunked form below is what would make it fast. |
| `linear_attention_chunked` | **not registered** | — | **The kernel only — the derivation is done.** [047](047-linear-attention.md) §6.1 derives the UT transform that makes a chunk's writes solvable together: the recurrence becomes $(I+A)W=B$ with $A$ strictly lower triangular and built from one Gram matrix of the chunk's keys, so $C$ dependent steps over a $K\times V$ state become one forward substitution over $C$ rows. Checked against the sequential kernel at seven chunk sizes and guarded by a test. What the form buys is global traffic, not arithmetic: the state crosses memory twice per chunk instead of three times per token. What remains is a residency plan, and the subset decides most of it: a kernel cannot declare an array-typed local, so per-lane running values are inexpressible and everything array-shaped is workgroup-shared. That bounds the chunk size by shared memory before any numeric limit — 12.8 KiB at $C=8$ against 26.6 KiB at $C=16$ ([047](047-linear-attention.md) §6.1).  |
| `attention_ragged_f16` | `AttentionRaggedF16` | 2026-08-27 | [`AttentionRagged`](046-segmented-extents.md) over an f16 cache (accel issue 23). Three lines differ — the two cache bindings and the two loads that widen — and the arithmetic is f32 throughout, so its bound is the f32 kernel's and the two are compared against each other on values f16 holds exactly. |
| `grouped_matvec` | `GroupedMatVec` | 2026-08-26 | Each token multiplied by the weight matrix its segment names, over [046](046-segmented-extents.md)'s extent with the row an expert (accel issue 18). The decode shape: a step routes one token to k experts and reads those k matrices rather than all E. |
| `quant_matvec_int4` | `QuantMatVecInt4` | 2026-08-26 | An asymmetric grouped 4-bit matvec, eight codes per word with a scale and a zero per 128 ([048](048-int4.md)). At 27B this is 13.4 GiB against int8's 26.7. |
| `grouped_gemm` | `GroupedMatMul` | 2026-08-27 | The tiled form, which is the shape a *prefill* has: one workgroup per (expert, column tile), walking that expert's segment in blocks of `TileM` so each weight is read once per block rather than once per token ([049](049-grouped-gemm.md) §5). The matvec is registered above as `grouped_matvec` and is the decode shape. This kernel's row index comes from the offsets rather than the grid, so it carries a `Tokens` bound the matvec does not need — §5.3.  |
| `quant_matmul_superblock` | **not registered** | — | GGUF's K-quants, which is where the ecosystem's pre-quantized checkpoints are (accel issue 15). A `Q4_K` super-block is 256 weights as eight sub-blocks of 32, each carrying a 6-bit scale and a 6-bit minimum, over two fp16 super-scales: $w_i = d\cdot s_j\cdot q_i - d_\text{min}\cdot m_j$ with $j=\lfloor i/32\rfloor$. `quant` registers one representation — int8 with one fp16 scale per 32, no minimum and one level of scale — so nothing reads a super-block. Not scheduled: the consumer reads safetensors and quantizes at load, and says so. The two workarounds are both bad and are why this is recorded rather than left implicit — dequantizing K-quants at load discards the memory saving that is the whole reason to read GGUF, and requantizing into int8 stacks two lossy steps. |
| `quant_matmul_int4` | `QuantMatMulInt4` | 2026-08-27 | The tiled form of the 4-bit product, which is the shape a *prefill* has: several rows of activations against one matrix, so each weight is unpacked once per tile and read `TileM` times rather than decoded on every use ([048](048-int4.md) §5.2). The row kernel is registered above as `quant_matvec_int4` and is the decode shape. Same representation and same bound; what differs is where the unpacking happens.  |
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
| `SaturatingConvert` | `kmath.ToI32` and `ToU32` over their boundaries — NaN, both infinities, and the exact limits — on both backends. The two lowerings are written separately and mirror each other by hand, so this is the only thing that would catch a compare reading `<` where the other reads `<=` ([051](051-float-to-int.md) §2.1) |
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

### Added to cover the signed atomics — 2026-08-26

`AtomicOpsI32`, and it is the same finding as the signed uniform one file down,
reached by the same sweep: reading the public surface for zero-coverage
functions. Every **unsigned** atomic is used by a corpus kernel. Not one
**signed** atomic was — `AddI32`, `SubI32`, `MinI32`, `MaxI32`, `ExchangeI32`
and `CompareExchangeI32` were exported for kernel authors, lowered by an emitter
path nobody ran, and spelled in MSL by a mapping nobody compiled.

| Kernel | Obligation met |
| --- | --- |
| `AtomicOpsI32` | the six signed atomics reach both backends, and **min and max are checked over a negative state** — the only two of the six whose result a signed and an unsigned reading disagree about |

**The two backends are guarded by different things, and that is worth writing
down rather than assuming symmetry.** On the CPU backend the test holds it:
making `MinI32` compare `uint32` fails with the pair named. On Metal the *type
system* holds it — the binding lowers to `atomic_int`, and lowering it as
`atomic_uint` does not compile, because the compare-exchange helper is typed
against the signed pointer. **The differential could not have caught a
signedness error on Metal**, because an unsigned signed-atomic is not
expressible there. A guard that cannot fail is still a guard; it is just not the
one it looks like.

**`AddF32` too, and the reason it was deferred was wrong — 2026-08-26.** The
float atomic was left out of the sweep above on the grounds that its result is
numerics class E, which [011](011-conformance-harness.md) §9 excludes from bit
comparison. That conflated two different things.

**Class E is about contention, not about the operation.** What
[008](008-numerics.md) classifies as non-deterministic is a *reduction* — many
invocations adding into one location in an order nobody fixes. One invocation
doing one read-modify-write is an ordinary float addition and is exactly
reproducible. `AtomicAddF32` therefore dispatches as a sequence, the shape
`AtomicOps` already uses, and asserts exact values rather than a bound.

| Kernel | Obligation met |
| --- | --- |
| `AtomicAddF32` | the float atomic reaches a kernel, returns the previous value like the rest of the family, and is **refused on a device that lacks the capability, at pipeline creation, naming it** |

It carries **no MSL artifact**, and that is correct rather than a gap:
`CapAtomicFloatAddStorage` is false on Metal and the emitter declines to spell
an operation the target cannot run. §3's lowering guard demanded the reason,
which is what that guard exists for.

The second obligation is the one that was missing entirely. `accel.AddF32`'s own
documentation promises that a device without the capability *"refuses the kernel
at pipeline creation rather than producing a wrong sum"* — the machinery existed
and nothing exercised it, because no kernel used the operation. Two independent
reasons stop it on Metal: the absent capability, and the missing MSL artifact
Metal's `SupportsKernel` looks for. **The capability check fires first, which is
the one that names something a caller can act on** — a caller told "no MSL
artifact" learns that this build cannot run it, and a caller told
"CapAtomicFloatAddStorage is absent" learns that another device might.

### Added to cover the signed uniform — 2026-08-26

`ElemBias`, one kernel, added for a gap rather than a feature.
[014](014-kernel-uniforms.md) admits `float32`, `int32` and `uint32` as uniform
field types, and every uniform in this corpus used two of them. So the third was
specified, emitted by a code path nobody ran, and written by an
`accel.UniformWriter` method sitting at zero coverage.

| Kernel | Obligation met |
| --- | --- |
| `ElemBias` | a signed uniform arrives signed, checked by an operation that signedness changes — the kernel branches on the offset's sign rather than adding it, because two's-complement addition is sign-agnostic and an unsigned read of the same four bytes adds identically |

**The obligation is worded that way because the first version failed it while
passing.** Adding a negative offset produces the same thirty-two bits as adding
the unsigned reading of those bytes, so a Metal side declaring the field `uint`
compiled and agreed. Comparison, division, modulo and the right shift are the
operations that differ; the kernel uses the first.

### Added for accel issue 6 — 2026-08-25

Three kernels for [039](039-sampling-policy.md) §4's penalties. They were
recorded here as **unreachable** for part of a day, under the rule
[009](009-sequencing.md) states for exactly this case — a kernel is not a
capability, and a corpus entry with no operator reaching it is recorded as
unreachable rather than as done. `tensor.Sample` now records all three, so they
are reachable, and the note is kept rather than deleted because the interval is
the evidence that the rule is applied rather than quoted.

### The rule applied to this file — 2026-08-27

The audit found the rule quoted here and not applied to 010 itself. **Fifty of
the corpus's seventy-two kernels are reachable from a `tensor/` operator and
twenty-two are not**, and four of those carry rows that read as done:

| Kernel | Why it has no operator |
| --- | --- |
| `ReduceSum` | an internal building block and [008](008-numerics.md)'s tree-reduction proof; its row already says so, which is why it is the least wrong of the four |
| `ElemBias` | added for a broadcast gap rather than a feature, and no operator records it |
| `AtomicOpsI32` | the signed-atomic differential, a device-layer proof |
| `AtomicAddF32` | the float-atomic capability proof, including its refusal on a device that lacks it |

**All four are device-layer proofs rather than tensor capabilities**, and that
is a legitimate reason to exist — [002](002-compute-model.md) and
[020](020-cooperative-atomics.md) need them and neither is a tensor operator. So
the fix is not to give them operators. It is that this file's tables did not
distinguish "an operator reaches it" from "a spec below the tensor layer needs
it", and a reader counting capabilities from these tables would have counted
four things a caller cannot use.

The remaining eighteen are graphics stages and cooperative-lowering fixtures,
outside this file's inventory and correctly not listed here.

**Two more device-layer proofs, 2026-08-28.** `DispatchShape` and
`ShapeBoundedSum` joined the corpus with [052](052-dispatch-shape.md) and are
unreachable in the same legitimate way: they prove `WorkgroupSize`, `NumGroups`
and `GlobalSize` return the recorded dispatch and that a barrier may sit inside
a loop bounded by the workgroup width. No tensor operator wants either, and the
guard that pins this list is what made naming them a step rather than an
oversight.

The same guard also had one row **mislabelled**: `CountWorkgroups` was recorded
as "002's dispatch-shape proof" and is [003](003-command-graph.md)'s
indirect-clamp proof — it increments a counter once per workgroup so the number
that ran is observable, and asks a `Thread` for no shape at all. A pinned list
stops a kernel joining silently; it does not check that the reason given is the
right one, and this is the first instance of that being wrong.

| Kernel | Obligation met |
| --- | --- |
| `PenaltyCount` | counts each token id in the history ring, bounded by how much of the ring is filled rather than by its capacity — the unwritten tail is zeros and zero is a real token id |
| `PenaltyApply` | one update per distinct token from a final count, with the divisive penalty multiplying a non-positive logit rather than dividing it, and an unseen token copied through bit-identical |
| `PenaltyClear` | zeroes the counts, because the accumulation is an atomic add and a reused buffer would penalise each step by every earlier one |

**What blocked the operator was not the kernels.** §4's two-pass shape needs a
`[vocab]u32` counts buffer that one node zeroes and another accumulates into,
and every node in a `tensor` graph produces exactly one output tensor. The
resolution is [039](039-sampling-policy.md)'s deviation 1: the counts are a
caller-owned `State` port, and the state version chain orders the clear before
the count. `PenaltyClear` is therefore the corpus's first kernel reached by a
node with **no inputs at all** — it writes a constant into caller-owned storage
and reads nothing, and giving it a nominal input so that it looked like its
neighbours would have been a dependency the planner must order and nothing must
obey.

### What decides a 4-bit representation — 2026-08-25

Recorded before the shape is fixed, because the two obvious goals pull apart and
the choice is hard to revisit once checkpoints exist.

**`Int8Block` is 32 for a reason that does not carry over.** It was chosen so no
K-step straddles a scale boundary: the tiled GEMM steps K in sixteen and the row
kernels are 128 wide. That is a *tiling* argument, and at 8 bits the metadata it
implies is invisible. At 4 bits it is not:

| representation | bytes per weight | 27B resident | scale overhead, as a share of the payload |
| --- | ---: | ---: | ---: |
| bf16 | 2.0 | 50.3 GiB | — |
| int8, fp16 scale per 32 | 1.0625 | 26.7 GiB | 6.2% |
| int4, fp16 scale per 32 | 0.5625 | 14.1 GiB | **12.5%** |
| int4, fp16 scale + fp16 zero per 128 | 0.53125 | 13.4 GiB | 6.2% |

**Halving the payload doubles the metadata's share of it.** Keeping the block at
32 would spend an eighth of a 4-bit format on scales. A group of 128 carries
*twice* as much metadata per group — a scale and a zero point — and still costs
less, because it is amortised over four times as many weights. And 128 is a
multiple of sixteen and is exactly the row kernels' width, so the tiling
argument that fixed 32 permits 128 rather than forbidding it.

**The tension worth stating is symmetric against asymmetric.** accel's int8 is
symmetric — `Int8Max` is 127 rather than 128 precisely so the representable
range has no value without a counterpart, which is the special case
[027](027-quantization.md)'s error bound is stated without. AWQ and GPTQ both use
group 128 **with a zero point**, which is asymmetric and needs a different bound.
So the consumer's question is the right one and the answer is not obviously yes:
*reading the ecosystem's checkpoints* and *having a good 4-bit format* may not be
one representation. A symmetric int4 has the cleaner bound; an asymmetric one
reads what people publish.

This is a different gap from `quant_matmul_superblock` above and neither
subsumes the other. That row is about reading GGUF's K-quants, which are 4-bit
among other things and carry two levels of scale; this one is about accel having
a 4-bit representation at all. A decision to read K-quants would not give a
caller a format to quantize *into*, and a native int4 would not read a published
`Q4_K` file.

**Not scheduled.** The consumer's dense path runs at ~18 tokens/s on Metal and
they say so. What this row buys is that the gap is visible to whoever picks it
up, which is exactly what they asked for.

## 3.1 What is built — 2026-08-23

Every kernel in the tables above exists on the CPU backend, in
`internal/kernels`, each checked against an independently written
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
