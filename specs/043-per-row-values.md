---
title: "Per-sequence values are device data, not scalars"
status: drafted
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
| `AttentionOptions.BaseName` | `AttentionOptions.Positions *Tensor` |
| `SampleDims.Draw` scalar | a draws tensor, one per row |
| paged cache unreachable | `AttentionOptions.Pages *Tensor` |

`ScaleName` is untouched.

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

## 7. Done

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
