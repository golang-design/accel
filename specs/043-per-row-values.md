---
title: "Per-sequence values are device data, not scalars"
status: implemented
layer: tensor
depends_on:
  - 007-tensor-layer.md
  - 025-tensor-operators.md
  - 030-paged-kv.md
  - 040-batch-scheduler.md
---

# Per-sequence values are device data, not scalars

Seven consumer reports arrived from a team building an inference framework on
this library. Four of them are the same decision seen from four angles, and this
spec names it once rather than fixing it four times.

## 1. The decision, as it stands

An operator that varies per sequence takes that variation as a **declared
scalar** — a value in a std140 uniform block, one per dispatch:

| Operator | Per-sequence value | How it travels today |
| --- | --- | --- |
| `RoPE` | the token's position | `offsetName`, and position is `row + Offset` |
| `Attention` | how much cache is real | `CurrentLengthName` |
| `Attention` | where a prefill starts | `BaseName` |
| `SampleCategorical` | the random draw | `SampleDims.Draw` |
| paged KV | the page table | *unreachable* |

For one sequence each is exactly right. For two in one dispatch each is exactly
wrong, and the failures are the kind that do not announce themselves:

- **`RoPE`.** In a batched decode the row index is the *slot*, so slot 0 rotates
  at `Offset`, slot 1 at `Offset+1`. Exactly one member is ever rotated at its
  own cache length. The output stays finite and fluent; what degrades is
  long-range coherence, which reads as "the model is a bit weak".
- **`SampleCategorical`.** One `Draw` across a batch keeps
  [028](028-sampling.md)'s reproducibility and destroys *independence*: two
  sequences with similar distributions emit the same token, converge, and become
  more similar. Every existing test passes, because reproducibility is what they
  check and reproducibility is what is preserved.

**There is no scheduling arrangement that makes a shared scalar correct**, because
the members' lengths are genuinely independent — that is what continuous
batching *is*. Grouping equal-length sequences converts continuous batching back
into static batching, which gives up the throughput it was for.

## 2. The line

> **A scalar is a value every row of a dispatch shares. A value that differs per
> row is a tensor.**

That line already exists in this codebase and is drawn correctly in two places.
`ScatterRows` takes its ids as a tensor binding, not a uniform. And the corpus
kernel `AttentionDecodeBatched` takes `pages []uint32, lengths []uint32` — per
row, on the device — and has since [030](030-paged-kv.md).

So this is not a redesign. **The kernels already work this way and the operators
never caught up**: nothing in `tensor/` references `AttentionDecodeBatched` or
`AttentionDecodePaged`. The mechanism is built and unreachable, which is what
[001](001-device-resources.md)'s reporter observed about `pagetable` and what
[025](025-tensor-operators.md) observed about `Contiguous`.

### 2.1 What stays a scalar, and why the line is not "everything becomes a tensor"

| Value | Kind | Stays or moves |
| --- | --- | --- |
| attention `Scale` (`1/√d`) | model constant | **scalar** |
| RoPE `Base` (θ base, e.g. 10000) | model constant | **scalar** |
| position, cache length, prefill base | per sequence | **tensor** |
| sampling draw | per sequence | **tensor** |
| page table | per sequence | **tensor** |

A model constant shared by every row is what a uniform block is *for*. Moving
those would cost a binding and a buffer write to say something that does not
change, which is the mirror of the mistake this spec fixes.

## 3. Orthogonality: this replaces, it does not add

Each change removes a declared scalar and adds a tensor operand. The surface is
**neutral or smaller**: a scalar costs a name on the plan, a declaration, and a
binding at submit; a tensor costs one operand.

And **B = 1 is the same path**, not a special case. A one-row positions tensor is
what a single sequence binds, so there is one mechanism rather than a batched
one beside a single one. That is the orthogonality test: after this there is no
question of the form *"which of the two ways do I use?"*

## 4. What changes

| Was | Becomes |
| --- | --- |
| `RoPE(b, x, rotaryDim, baseName, offsetName)` | `RoPE(b, x, rotaryDim, baseName, positions *Tensor)` |
| `AttentionOptions.CurrentLengthName` | `AttentionOptions.Lengths *Tensor` |
| `AttentionOptions.BaseName` | **unchanged** — see below |
| `SampleDims.Draw` scalar | a draws tensor, one per row |
| paged cache unreachable | `AttentionOptions.Pages *Tensor` |

`ScaleName` is untouched, and so is `BaseName`. A prefill is one sequence:
[040](040-batch-scheduler.md) owns the batched form, so until it exists there is
no row for a prefill's base to differ across. This table said otherwise when it
was written, before the prefill path was read closely; the correction is here
rather than left as a silent divergence.

**Every attention kernel reads its cache length from a binding**, including
prefill, so there is exactly one way to say *how much of the cache is real*
rather than one way per kernel. A prefill binds a one-element tensor for the
same reason a single-sequence decode does.

**Paging is not a second cache type.** A `State` addressed through a page table
is the same `State`; what differs is the binding. Introducing a `PagedState`
beside `State` would be exactly the non-orthogonal growth this spec exists to
avoid — a consumer would then ask which one to build, and the answer would
depend on a scheduler they have not written yet.

## 5. What is separate

Two reports are dtype relaxations and share nothing with the above. They add
**no** surface: they remove a refusal.

- **`Attention` requires f32 states**, doubling the largest allocation in a
  serving process. The f32 rule in [002](002-compute-model.md) is about
  *accumulators* — narrow accumulation over a long dot product loses badly — and
  K and V are *operands*. `softmax(qKᵀ/√d)·V` accumulates in f32 whatever the
  operands are stored as, which is the trade `MatMul` already makes when it
  reads f16. Storing the cache narrow applies it to the one buffer where it is
  worth most.
- **`MatMul` is f16-only** while every other operator is f32, so a transformer
  pays seven casts per layer — 252 extra dispatches per forward pass on a
  36-layer model, each a full pass over the activations existing only to satisfy
  a dtype check. [010](010-kernel-corpus.md) already owns an f32 variant.

## 6. Open, and deliberately not decided here

- **Importing host memory** ([001](001-device-resources.md)). On unified memory
  the host and device copies are the same physical bytes, so the upload is pure
  overhead. The reporter filed it as a question rather than a proposal, and they
  are right that the lifetime rules are the hard part: a buffer over memory the
  caller owns is a promise about a lifetime accel cannot see. If it lands, the
  shape that keeps it portable is *refusal with fallback* — the backend may
  decline and copy, and the caller can ask which happened, the way
  `Plan.Selections` reports which kernel was chosen.
- **On-device sampling policy** ([039](039-sampling-policy.md)). A 608 KB logits
  readback per token sets the floor on a decode step and forecloses overlapping
  step *t+1* with step *t*'s sampling. Scheduled, not redesigned. What this spec
  does record, because the reporter identified it and it is documentation
  rather than code: the composition order is penalties before temperature (a
  penalty applied after dividing by *T* has a strength that depends on *T*),
  temperature before truncation (top-p is a mass threshold and temperature is
  what changes the mass), top-k before top-p, and *T=0* as a distinct greedy
  branch rather than a division.

## 7. What was built — 2026-08-24

All of §4, plus the two dtype relaxations of §5.

**One thing the reports did not contain, found while building §5.**
`accel.ToFloat16(-1.9996898)` returned `-1`. The rounded mantissa can carry out
of ten bits and the code OR-ed that carry into the exponent instead of adding
it; where the exponent's low bit was already set the OR did nothing, so every
value in a band just below every other power of two came back at **half its
magnitude**. Where the bit was clear the OR happened to act as a carry, which is
why only half the affected band ever looked wrong.

$$
\text{ToFloat16}(x) \ \text{halved for}\ x \in [2^{e}(1-2^{-11}),\ 2^{e}) \ \text{where}\ e\ \text{is even}
$$

It reached every f16 path: weight conversion, the f16 GEMM corpus, and
[027](027-quantization.md)'s scales, where a scale landing in the band would
halve every weight in its block. The test is now an oracle enumerating all
65536 halves, because a case table is what it had and the band is too narrow for
anyone to write a case inside.

**The generalizable part:** asking for a narrow cache surfaced a correctness bug
in the conversion the narrow cache depends on. A dtype that nothing exercised
end to end was a dtype nothing had checked.

### 7.1 What #7 became

The report asked for a buffer *over* memory the caller already owns, and filed
it as a question because the lifetime rules are the hard part. They are: such a
buffer is a promise about a lifetime accel cannot see, and a broken promise
there is a use-after-free whose symptom is a plausible tensor.

`Buffer.Access` points the same problem the other way. accel owns the memory and
lends it for the duration of one call, so a loader converts *into* the
destination and the intermediate does not shrink — it does not exist. No promise
is required of the caller, and the borrow is bounded by something the compiler
and the reader can both see.

On unified memory the mapping *is* device memory, so the upload is free rather
than fast, which is what the report observed about
[006](006-backends.md) §2.2's hardware.

## 9. The same rule, one level up: extents — 2026-08-24

Three later reports asked for three different features and want the same thing.
Recording it here rather than in each, because the shared concept is worth more
than the three kernels and because this spec's rule already predicted it.

| Report | Asked for | The value that differs per row |
| --- | --- | --- |
| [#16](https://github.com/golang-design/accel/issues/16) | a dispatch mixing prefill chunks with decode steps | how many **query tokens** a sequence contributes |
| [#18](https://github.com/golang-design/accel/issues/18) | a grouped GEMM for mixture-of-experts | how many **tokens routed** to an expert |
| [#17](https://github.com/golang-design/accel/issues/17) | a chunked scan for linear attention | (partly — see below) |

§2's line was *a value every row of a dispatch shares is a scalar, and a value
that differs per row is a tensor*. It was applied to **quantities**: lengths,
positions, page tables, draws. These three apply it to an **extent** — how many
elements a row *has* — which is the same statement one level up and the one
place §2 did not look.

An extent that is device data is the definition of a ragged operation, and the
shape every serving stack converges on is the same in all three cases: a flat
buffer, plus one count per row, plus the prefix sum of those counts.

$$x : \Big[\textstyle\sum_r n_r,\ \ldots\Big], \qquad n : [R], \qquad \text{offset}(r) = \sum_{j<r} n_j$$

**What this means for the design, and it is not "write three kernels".** A
segmented extent is one concept. If it is expressed once — a tensor whose
leading axis is ragged, carrying its counts — then a ragged attention, a grouped
GEMM and a segmented scan are three *uses* of it rather than three special
cases, and the operator surface does not grow by three. That is §3's
orthogonality test applied before the fact rather than after: after a segmented
extent exists there should be no question of the form *which of the two matmuls
do I use?*

The alternative, which is what happens by default, is three kernels with three
spellings of the same count array, and a fourth report that needs a fourth.

**What is genuinely different in [#17](https://github.com/golang-design/accel/issues/17),
and it is not the extent.** A linear-attention layer carries a matrix-valued
recurrent state per head — `[keyHeadDim, valueHeadDim]`, not per position. That
is a second finding and a sharper one: **[007](007-tensor-layer.md)'s `State`
conflates two things.**

| | a cache | a recurrent state |
| --- | --- | --- |
| shape | one value per **position** | one value per **sequence** |
| grows with context | yes | no — which is the entire appeal |
| indexable at a position | yes, and `ScatterRows` does | **no**; resuming means restoring it, not indexing it |
| pageable | yes, `Pages` addresses positions | there are no positions to address |
| what sharing a prefix needs | a **page table**: two sequences name the same blocks | a **snapshot**: save at a branch point, restore when a request diverges |

The last row was added after the consumer answered the question this table
raised, and it is the one that settles the shape:

> everything a cache does is **addressing**, and everything a recurrent state
> does is **copying**.

That is a sharper test than the four rows above it, because it predicts the
operations rather than describing the differences. A cache wants `Pages` and
`ScatterRows` — both address. A recurrent state wants **snapshot** and
**restore**, which have no analogue in this library today and are copy-shaped:
save a sequence's state where it can be kept, put it back into a slot when
another request branches there. Neither is a `Slice` of anything.

It is still a `State` rather than a transient, and the reason is the second use
rather than the first: within one submission a recurrent state is carried
between chunks and could be a transient; across submissions it must survive from
token *t* to token *t+1*, caller-owned, exactly as a KV cache does.

`State`, `ScatterRows` and `AttentionOptions.Pages` all assume the left column.
A hybrid model — three linear-attention layers for every full-attention one — has
**both kinds at once**, so its cache is not one thing and `Pages` covers only the
quarter that is full attention. That is a type-level distinction this library
does not currently have a name for, and naming it is prior to any kernel.

**A hybrid model needs both kinds live in one graph, per layer** — not two
models but two kinds of state in one forward pass, sixteen full-attention layers
with a paged cache beside forty-eight recurrent layers with snapshots. That is
the constraint to design the type distinction against, and it rules out the
cheap answer of a second `State`-like type used in a separate plan.

**One kernel this does not need.** The short depthwise causal convolution these
layers carry composes from what exists: left-pad the input by `K-1` rows so
causality is structural rather than masked, then `Slice`, `Contiguous`,
`Broadcast` and a multiply-add per tap. The consumer verified it element by
element. It costs `K` dispatches and `K-1` packing copies per layer where a
kernel would take one, so it is *one less kernel to unblock* rather than one
less to want — and folding the convolution into the chunked scan is probably
right if that is ever built.

Both `Contiguous` calls in that composition are load-bearing, which is the
refusal working: `Slice` yields a strided view and elementwise `Mul` refuses one
by name.

**Neither is scheduled.** Both reports say no urgency and both asked for a scope
answer rather than an implementation; [010](010-kernel-corpus.md) carries the
rows. What is decided here is that if they are built, the extent is built **once**
and the state distinction is made **before** the first recurrent kernel, not
after three of them have assumed the wrong one.

## 8. Done

- `RoPE` rotates each row at its own position, and a two-sequence batch with
  different cache lengths matches two single-sequence runs element for element;
- a batched sample draws independently per row, and two rows with identical
  distributions and different draws emit different tokens;
- `Attention` accepts f16 states and accumulates f32, checked against the f32
  path within a derived bound;
- `MatMul` accepts f32 operands, and a transformer graph built from f32
  activations contains no `Cast` node;
- an `Attention` bound to a page table produces what the contiguous one produces
  over the same logical positions, including when the pages are out of order;
  and
- no operator takes a per-sequence value as a scalar.
