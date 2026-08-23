---
title: "The v0 tensor operator set: views, indexing, normalization, and matrix multiplication"
status: implemented
layer: tensor
depends_on:
  - 007-tensor-layer.md
  - 010-kernel-corpus.md
  - 024-tensor-bringup.md
---

# The v0 tensor operator set

The second of [009](009-sequencing.md)'s M7 children. [024](024-tensor-bringup.md)
built the machinery and four operators; this fills in the rest of
[007](007-tensor-layer.md)'s v0 list, so that a transformer layer can be
written.

## 1. Views move nothing

`Reshape`, `Permute`, `Transpose`, `Slice` and `Broadcast` return a tensor over
the same storage with different shape, strides or offset. A head split is a
view; a transpose is a view. The layout is a stride vector in elements, and
contiguous strides are the suffix products:

$$\text{stride}_i = \prod_{j>i} \text{shape}_j$$

**A broadcast is a zero stride**, which is the whole trick: every index along
the expanded axis reads the same element, so nothing is materialized until
something needs to read it contiguously.

Two rules follow, and each is a refusal rather than a silent repair:

- **`Reshape` requires a contiguous operand.** A strided view's elements are not
  adjacent, so a different extent over them names *different elements*. The
  error says that, because the fix is `Contiguous` and a reader should not have
  to guess.
- **`Slice` is unit-step.** A strided step would still be a view and nothing in
  v0 needs one, so it is absent rather than untested.

## 2. Materialization is reported, never silent

The corpus kernels index their operands contiguously, so a view they cannot read
has to be packed first. This package packs in one place and refuses in another,
and the difference is what [007](007-tensor-layer.md) asks for:

| | |
| --- | --- |
| Elementwise operands | **materialized**, because that spec gives `Add` and `Mul` NumPy broadcasting and a caller who wrote it expects it to work |
| `MatMul` operands | **refused**, because that spec requires unit stride on the contracted axes "without silently materializing either one" |

The distinction is cost. Repeating a gain vector across rows is small; copying a
weight matrix is not, and a caller must ask for it.

**Every materialization appears in `Plan.Selections`**, with how many copies and
of what size. A copy nobody can see is a performance cliff nobody can explain,
which is the same argument that makes kernel choice a report.

**What can be materialized is a contiguous run repeated a whole number of
times** — a gain across rows, a bias across a batch. An interior axis expanding
repeats with a *stride* rather than as a run, and building that needs a gather
kernel with the strides in a uniform block, which the corpus does not have. It
is refused with the shape it can build named in the message.

## 3. The operators, and the kernels under them

| Operator | Kernel | Grid |
| --- | --- | --- |
| `Add`, `Mul` | `ElemAdd`, `ElemMul` | one invocation per element |
| `Scale` | `ElemScale` | per element, factor in a uniform |
| `SiLU`, `SwiGLU` | `SiLU`, `SwiGLU` | per element |
| `Rows` | `GatherRows` | per output element |
| `RMSNorm` | `RMSNorm` | one workgroup per row |
| `Softmax` | `Softmax` | one workgroup per row |
| `RoPE` | `RoPE` | one invocation per rotated pair |
| `MatMul` | `MatMulTiled`, or `MatVec` when M = 1 | tiles of the output |
| `Linear` | `LinearTiled` | tiles of the output |

`MatVec` is a *selection* rather than a distinct operator, which
[007](007-tensor-layer.md) states and `Plan.Selections` reports: a caller writes
`MatMul` and is told which one they got.

## 4. Where v0 is narrower than 007, and why

Recorded rather than glossed, because each is a real gap a caller meets:

1. **`MatMul`, `Linear` and `MatVec` take f16 operands and produce f32.** That
   is what [010](010-kernel-corpus.md) registers. 007 admits f16 *or* f32
   storage; an f32 GEMM is a corpus kernel that does not exist yet, and adding
   one is 010's rather than something to improvise from up here.
2. **`RMSNorm` and `Softmax` take f32 only**, for the same reason.
3. **`Softmax`'s mask and causal option are absent.** The corpus kernel has no
   mask parameter. `Axis` must be the last, which is the only axis that kernel
   normalizes over.
4. **`RoPE` is in-place in the corpus and immutable here**, so it lowers to a
   copy into a transient followed by an in-place dispatch on it. The copy is
   reported.

## 5. Outcome — 2026-08-23

Every operator builds, infers, lowers, and runs on both backends, and a
feed-forward block — normalize, project, activate — compiles and agrees between
them within [008](008-numerics.md) §6's ceiling for the primitives it reaches.

**The grid belongs beside the kernel.** An elementwise kernel wants one
invocation per element, a row reduction wants one workgroup per row (because the
invocations in it share partial sums through workgroup memory, and a row split
across two workgroups could not), a tiled GEMM wants tiles of the output, and
RoPE wants one invocation per rotated *pair*. That mapping is the only thing the
operator file knows which the contract does not, which is why it sits next to
the kernel rather than in a table elsewhere.

**Broadcasting is elementwise-only, and finding that out was the correction.**
The first lowering packed *any* operand whose shape differed from the result's,
which is right for `Add` and catastrophic for `Rows`: a gather's table is
`[vocab, width]` against a result of `[rows, width]`, and materializing it would
have repeated the wrong rows. Operands have shapes of their own; only the
elementwise family's are the result's.

**An in-place kernel goes through scratch, not through the result.** `RoPE`
rewrites its buffer, so the operand is copied in and the answer copied out. The
first version copied straight into the result, which fails when the operand and
the result are both caller buffers — the recorder moves bytes between a slot and
a view, and not between two slots. The copies are reported for the same reason a
materialization is.

## 6. Done

- every operator above builds, infers, lowers, and runs on both backends;
- the refusals are checked by a table, including every view rule; and
- a transformer feed-forward block — normalize, project, activate — compiles,
  runs, and agrees between the backends, which is the smallest thing that shows
  the set composes.
