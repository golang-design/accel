---
title: "Attention over a cache larger than a workgroup"
status: drafted
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
