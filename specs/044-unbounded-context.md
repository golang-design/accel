---
title: "Attention over a cache larger than a workgroup"
status: implemented
layer: tensor
depends_on:
  - 002-compute-model.md
  - 007-tensor-layer.md
  - 010-kernel-corpus.md
  - 030-paged-kv.md
  - 043-per-row-values.md
---

# Attention over a cache larger than a workgroup

`tensor.Attention` refuses a cache longer than the decode kernel's workgroup:

```go
lanes := int(testkernels.AttentionDecodeKernel.WorkgroupSize.X)
if k.shape[0] > lanes {
    return b.fail(1, "Attention", "the cache holds %d positions and the decode kernel "+
        "scores one per lane over %d; a longer cache needs the looping variant, which "+
        "specs/010-kernel-corpus.md does not register", k.shape[0], lanes)
}
```

The workgroup is 128, and the check sits above the prefill branch, so it binds
both paths. The refusal is honest and names the missing kernel. This spec is
about what it means and what closes it.

## 1. Why this one is different from the other reports

The consumer reports collected in [043](043-per-row-values.md) are **costs**.
A shared `Draw` makes a batch correlated; an f32 cache doubles the largest
allocation in a serving process; an f16-only `MatMul` costs a transformer 252
casts per forward pass. Each makes something worse, and each has a workaround
that runs.

This one has no workaround and no degraded mode. **A 128-position context is
shorter than a system prompt.** There is no model that can be usefully served,
no end-to-end run to measure, and no benchmark whose number means anything —
every measurement taken at 128 positions describes something nobody would
deploy.

It is worth stating why it stayed invisible. Nothing inside this library asks
for a cache longer than a workgroup: the conformance tests use small shapes on
purpose, the kernel is correct at every capacity it accepts, and the refusal is
well written. **A gap can be fully documented, correctly refused, and still
unnoticed until something tries to do the real job.**

## 2. Why the workgroup cannot simply grow

The limit is not the launch geometry. It is **shared memory**:

```go
func AttentionDecode(t accel.Thread, d AttnDims, q, k, v []float32,
    out []float32, scores *[128]float32, red *[128]float32)
```

`scores` holds one probability per cached position, because the second phase
reads *all* of them:

```go
if lane < d.HeadDim {
    acc := float32(0)
    for j := uint32(0); j < d.KVLen; j++ {
        acc = acc + scores[j]*v[j*d.KVHeads*d.HeadDim+kvHead*d.HeadDim+lane]
    }
    out[h*d.HeadDim+lane] = acc / total
}
```

So capacity is bounded by shared memory per workgroup, which
[002](002-compute-model.md) caps portably far below any real context. Widening
to 1024 buys 1024 positions, costs 8 KB of shared memory per workgroup, reduces
occupancy, and is still three orders of magnitude short. **There is no width
that makes this work**, which is what makes it a shape change rather than a
constant.

## 3. The shape: tile the cache, carry a running softmax

The standard formulation, and the one flash attention is built on. Process the
cache in tiles of one workgroup, carrying a running maximum $m$, a running
denominator $\ell$, and a running output accumulator $o$:

$$m_i = \max(m_{i-1}, s_i), \qquad
\ell_i = \ell_{i-1}\,e^{m_{i-1}-m_i} + e^{s_i-m_i}, \qquad
o_i = o_{i-1}\,e^{m_{i-1}-m_i} + e^{s_i-m_i}\,v_i$$

The rescaling factor $e^{m_{i-1}-m_i}$ is what makes this exact rather than
approximate: when a later tile contains a larger score, everything accumulated
so far is corrected by the same factor, and the result equals the one-pass
softmax to floating-point rounding. It is never greater than 1, so it cannot
overflow.

In this kernel's terms, per query head:

```
m, l := -inf, 0
o[lane] := 0                                   // lane < HeadDim owns one output element
for base := 0; base < KVLen; base += 128 {
    j := base + lane
    s := (j < KVLen) ? dot(q, k[j]) * Scale : -inf
    scores[lane] = s

    mTile := reduceMax(s)                      // the existing tree reduction
    mNew  := max(m, mTile)
    r     := exp(m - mNew)                     // 1 on the first tile

    e := (j < KVLen) ? exp(s - mNew) : 0
    scores[lane] = e
    l = l*r + reduceSum(e)

    if lane < HeadDim {
        acc := 0
        for jj := 0; jj < min(128, KVLen-base); jj++ {
            acc += scores[jj] * v[(base+jj) row, kvHead, lane]
        }
        o[lane] = o[lane]*r + acc
    }
    m = mNew
}
out[lane] = o[lane] / l
```

**`scores` and `red` stay `[128]`.** They become per-tile workspace rather than
per-position storage, which is the whole change. Capacity leaves the geometry
and the shared-memory budget at once, and $o$, $m$, $\ell$ live in registers.

The masking rule already in the kernel carries over unchanged and for the same
reason: a lane past the cache contributes $-\infty$ to a maximum and $0$ to a
sum, because those are the respective identities. A tile wholly past the end is
skipped by the loop bound rather than masked.

### 3.1 The prefill path

The same, with the causal bound: query position $p$ tiles over
$[0, p]$ rather than $[0, \text{KVLen})$. The upper bound moves from a uniform
to a per-row value, which is [043](043-per-row-values.md)'s rule already
applied — `Positions` supplies it.

## 4. What it changes above the kernel

**Nothing in the operator's signature.** `Attention` keeps its shape; the
refusal in §0 is deleted, and `Selections()` reports the looping kernel and why.
A caller who was under 128 positions sees no difference.

That is the orthogonality test [043 §3](043-per-row-values.md) states: after
this there is no question of the form *"which of the two attentions do I use?"*
The looping form is correct at every capacity, so it is the only one — unless a
measured margin justifies keeping the single-tile kernel as a **selection** at
$\text{KVLen} \le 128$, reported the way `MatVec` is at $M = 1$. That is a
selection question, not an API one, and it should be settled with a benchmark
rather than assumed in either direction.

## 5. Its relationship to the two kernels being written beside it

This matters for ordering, and it is the reason to write this spec now rather
than after.

| change | relationship |
| --- | --- |
| **f16 cache** ([043 §5](043-per-row-values.md)) | the same kernel body with two loads widened. Written against the single-tile body, it inherits the 128 ceiling and is rewritten when this lands |
| **paged attention** ([030](030-paged-kv.md), [043 §4](043-per-row-values.md)) | a page table exists *because* positions are not contiguous, so a paged kernel must walk the cache in pieces. **The tiling loop is the same loop.** |
| **batched attention** ([040](040-batch-scheduler.md)) | orthogonal: batching adds a leading dimension over rows, tiling changes how one row walks its cache |

So three registered kernels pass through one rewrite. Doing the loop first makes
the f16 and paged variants nearly free; doing them first means deriving the same
numerics three times and discarding two of them.

## 6. What was considered instead

**The composed fallback.** [007](007-tensor-layer.md) says fused attention is
"runtime kernel selection, not a device capability", and that the composed
definition — score `MatMul`, `Softmax`, value `MatMul` — is the correctness
reference. `Attention`'s doc comment says v0 "does not yet fall back to the
composed graph, and says so rather than pretending the choice was made". With
`Contiguous` landed, that composition is now expressible.

Falling back to it over 128 positions would unblock a consumer immediately and
needs no new kernel. It is worth doing **if this spec is not going to be built
soon**, because an answer that is slow is unboundedly better than a refusal. It
is not a substitute for this: the composed graph materialises the full
$[1, \text{KVLen}]$ score matrix, which is the allocation the fused kernel
exists to avoid, and it grows with context exactly where that hurts most.

**A larger workgroup.** §2.

**Letting the consumer compose it.** The consumer declined, correctly: their
own decision 1 forbids routing around this library, and a consumer that quietly
builds its own attention stops reporting the gap that matters most. Recorded
because it is the option that would have made this invisible again.

## 7. Done

- `Attention` accepts a cache of any capacity the device can allocate, and the
  §0 refusal is gone;
- a 4096-position decode matches the composed score/softmax/value graph within
  the tolerance [011](011-conformance-harness.md) derives, and matches the
  single-tile kernel exactly at capacities it also accepts;
- a capacity that is not a multiple of the tile is correct at the boundary, and
  a capacity of 1 is correct;
- shared memory per workgroup does not grow with capacity;
- a cache whose scores span a range wide enough to overflow a one-pass softmax
  produces the same result as the composed graph, which is what the running
  maximum is for; and
- `Selections()` names the kernel and the tile count.

## 8. Outcome, 2026-08-24

Built, in five kernels and one operator. `Attention` accepts a cache of any
capacity the device can allocate; a 4096-position decode runs end to end through
the public operator and matches an f64 reference.

The measured properties §7 asked for:

| §7 asked | outcome |
| --- | --- |
| the refusal is gone | `tensor/attention.go` no longer bounds `k.shape[0]` |
| a long decode matches the reference | 4096 positions with the length at 2001, mid-block, against f64 |
| exact at capacities the single-tile kernel also accepts | bit-identical, asserted at zero tolerance against a float32 reference that reproduces the kernel's reduction order and its rounding -- `TestOneBlockIsExactlyTheSinglePassForm` |
| correct at a boundary that is not a tile multiple | lengths 129, 300, 512 and a paged 300 over a 38-entry table |
| shared memory does not grow with capacity | two `[128]` arrays, unchanged; the accumulator is a local |
| a score range wide enough to overflow one-pass softmax | the running maximum, exercised by the rescale test in deviation 4 |
| `Selections()` names the kernel and the tile count | **not done.** The reason string still names the kernel and not the count |

### Deviation 1: the loop is bounded by the binding, not by the length

§3's pseudocode reads `for base := 0; base < KVLen; base += 128`. That cannot be
compiled. [002](002-compute-model.md) §3.3 seeds every storage load as
non-uniform, `KVLen` is a load -- [043](043-per-row-values.md) made it one, and
correctly -- and the loop body holds fourteen barriers, so the uniformity
analysis rejects it. §3.3 names this exact case as one of its two known false
rejections.

What is used instead is a bound that is workgroup-uniform by §3.3's own seed
table, and each kernel takes it from the right place:

| kernel | bound | why that quantity |
| --- | --- | --- |
| `AttentionDecode`, `AttentionDecodeF16` | `len(k) / (KVHeads·HeadDim)` | the cache binding's extent |
| `AttentionDecodePaged` | `len(pages) · Block` | the table's reach. **Not** the pool's extent: a pool holds every sequence's blocks and is sized for total concurrency |
| `AttentionDecodeBatched` | `MaxPages · Block` | the same reach, from the uniform struct because the table is a `Batch × MaxPages` array |
| `AttentionPrefill` | `min(Base + s + 1, capacity)` | the causal limit. A query at `Base+s` may see nothing past it |

The distinction that makes this sound where the tempting alternative is not: a
binding's **size** is fixed when the node is recorded, while its **contents** can
be changed during the dispatch by an aliased write. See deviation 3.

**The cost, measured.** Positions between the length and the bound are masked
per lane rather than skipped, so an empty block still pays its barriers and work
is proportional to the bound rather than to the length.

`BenchmarkAttentionEmptyBlocks` measures it on an Apple M2, at 32 query heads
over 8 key/value heads with a head dimension of 128, holding the length at 256
and growing the capacity:

| capacity | length | blocks (empty) | per decode | against a tight capacity |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 256 | 2 (0) | 199 µs | — |
| 1024 | 256 | 8 (6) | 258 µs | 1.3× |
| 4096 | 256 | 32 (30) | 504 µs | 2.5× |
| 8192 | 256 | 64 (62) | 837 µs | 4.2× |
| 4096 | 4096 | 32 (0) | 2622 µs | — |

Fitting the two ends: an **empty block costs about 10 µs and a full one about
85 µs**, so an empty block is roughly an eighth of a full one — the barriers
without the arithmetic, which is what the design predicted qualitatively.

Two consequences worth stating, because the ratio alone reads worse than it is.
The overhead is bounded: a decode over a capacity-$C$ cache never costs more
than a decode over a *full* capacity-$C$ cache, which is the cost the caller
already accepted when they sized it. And it shrinks to nothing as the context
fills, so it is worst on a fresh conversation and absent on a long one.

A caller who wants the tight bound sizes the cache to the context, and
`Attention` accepting any capacity is what makes that possible.

### Deviation 2: one block is the closed form, not a separate case

§3 writes `r := exp(m - mNew)` with the comment "1 on the first tile". It is
zero on the first tile, because `m` starts at `-3.4e38` -- and that is the
better value: it multiplies away accumulators that are already zero, so the
first block needs no special case, and `ℓ` and `o` come out as the single-pass
kernel's sum and weighted sum term for term and in the same order.

The consequence is worth more than the tidiness: **every test written against
the 128-position kernel keeps its exact numbers.** The block loop is a longer
reach rather than a different answer, and that is a property of the arithmetic
rather than of a tolerance.

### Deviation 3: §6's composed fallback cannot be built

§6 says the composed graph "is now expressible" with `Contiguous` landed, and
recommends it as the cheap route if this spec were delayed. That is wrong, and
[007](007-tensor-layer.md) is corrected with it.

Grouped-query attention has several query heads sharing one key/value head, so
the composed form needs one matrix multiply per head.
[025](025-tensor-operators.md) multiplies two matrices and does not broadcast
leading axes, so the composition exists only at `kvHeads == 1` -- which no model
this serves uses. The composed path remains the correctness reference over the
shapes it can express, which is what the corpus tests run.

**A route not taken, and why it is recorded.** Making a load from a *read-only*
binding workgroup-uniform would have admitted §3's `KVLen` bound directly and
retired one of 002 §3.3's two false rejections. It is unsound here: soundness
needs no other binding of the node to write that memory, and the library permits
exactly that alias -- `TestInPlaceWorkOnAWrittenTransientIsFine` binds one
transient to a read binding and a write binding of one dispatch, deliberately.
003's check V23 would forbid it and is **not implemented**; implementing it as
worded would delete in-place work. 002 §3.3 and 003 record this.

### Deviation 4: the shared arrays are loop-carried, and that is a new hazard

Not in §3 at all. The arrays are per-tile workspace, so a pass's writes race the
previous pass's reads of them. One barrier at the top of the loop body orders
them.

Nothing else would have reported it. The CPU backend's rendezvous check
([002](002-compute-model.md) §3.4) detects an invocation that fails to *arrive*
at a barrier, and this is a race between arrivals. It was confirmed by removing
the barrier and watching the long-cache test fail, which is also how the missing
rescale and a page lookup that dropped its block offset were confirmed.

### Deviation 5: the accumulator is a local, and the prefill needs no Positions

§3 puts `o` in registers and the first implementation put it in a third shared
array; the array cost a barrier and 512 bytes on all five kernels for nothing,
since each lane owns one element no other lane reads. It is a local, which the
resumable lowering carries across a suspension point.

§3.1 says the prefill's causal bound "moves from a uniform to a per-row value,
which is 043's rule already applied -- `Positions` supplies it". It does not
need to. `Base` is a field of the uniform struct and the query position comes
from the group id, so the limit is already workgroup-uniform and bounds the loop
directly. No new operand.

### What this did not close

- `Selections()` still reports the kernel without the block count.
- Issue 9, the `LayerState` view, is untouched: `Attention` still refuses a
  non-zero offset.
- [040](040-batch-scheduler.md)'s second length cap changed shape rather than
  going away. A length past the page-table row is **truncated** instead of
  reading the next sequence's row. A silently short answer is better than
  another conversation's keys and is still wrong, so admission still owes the
  check.

### Deviation 6: the loop bound does not bound the lane, and that was a bug

Recorded as a deviation because the first implementation of this spec shipped
with it, and because the shape of the mistake outlives it.

The loop advances `base` by `AttnBlock`, and each lane scores `base + lane`. So
the bound limits `base` and the lanes of the last block reach `AttnBlock - 1`
positions **past** it. A length larger than the binding's reach was therefore
scored by those lanes rather than stopped by the loop: for the paged kernels
that read the next sequence's page-table row -- the exact failure 040 names --
and for the contiguous ones it read off the end of the cache.

The claim that the loop bound alone truncated was written into this spec and
into 040 before anything checked it. What makes it true is an explicit clamp of
the length to the reach, in all five kernels.

The generalization: **a bound on a loop variable is not a bound on the index
derived from it.** Where a lane offsets the loop variable, the mask carries the
bound and the loop does not.

Found by writing the test for a claim that had been reasoned rather than
measured. It is the second time in this spec that reasoning about barrier-shaped
code was wrong in a way only a test showed -- see deviation 4.
