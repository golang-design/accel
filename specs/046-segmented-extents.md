---
title: "Segmented extents: a count per row, and the operations that read one"
status: drafted
layer: tensor
depends_on:
  - 007-tensor-layer.md
  - 010-kernel-corpus.md
  - 026-tensor-decode.md
  - 029-plan-cache.md
  - 030-paged-kv.md
  - 040-batch-scheduler.md
  - 043-per-row-values.md
---

# Segmented extents

**One thing:** how many elements a row *has* is device data, not a shape.

[043](043-per-row-values.md) §2 settled that *a value every row of a dispatch
shares is a scalar, and a value that differs per row is a tensor*. It applied
that to quantities — lengths, positions, page tables, draws. §9 found the same
statement one level up and did not build it: an **extent** differs per row too,
and an extent that is device data is the definition of a ragged operation.

This spec is that primitive, and the first operation to read one.

## 1. The primitive, before any caller

A segmented value is a flat buffer, a count per row, and the offsets those
counts imply:

$$
x : \Big[\textstyle\sum_{r<R} n_r,\ \ldots\Big],
\qquad n : [R],
\qquad \mathrm{off}(r) = \sum_{j<r} n_j
$$

```
n        = [ 3 ,   1 ,   4 ]            R = 3 rows
off      = [ 0 ,   3 ,   4 ,  8 ]       R+1 entries, exclusive prefix sum
x        = [ a b c | d | e f g h ]      one flat axis of length 8
             row 0   r1    row 2
```

**`off` has `R+1` entries and the last one is the total.** That is not a
convenience: a kernel that owns row `r` needs both ends of it, and deriving the
end as `off(r) + n[r]` reads two buffers where one would do. The extra entry
also makes the total available without a reduction, which is what lets a caller
be refused for a flat buffer whose length disagrees with the counts.

**Three properties are stated here so that three callers cannot each decide
them differently.**

1. **A count of zero is legal.** A row that contributes nothing is an ordinary
   member of a batch — an expert nothing routed to, a sequence admitted this
   step with no tokens yet. It is not an error and it is not a skip: `off(r) ==
   off(r+1)` and the kernel does nothing for that row.
2. **The counts are u32 and the offsets are u32.** A segmented axis is a count
   of elements, and [043](043-per-row-values.md) already made every per-row
   quantity u32 for the reason a length is one.
3. **$\sum n_r$ must equal the flat axis exactly, and a mismatch is refused.**
   Not padding, not truncation. [044](044-unbounded-context.md) deviation 6 is
   the precedent: a bound the kernel clamps is a wrong answer the kernel cannot
   distinguish from a right one, so anything the host *can* check, the host
   refuses.

### 1.1 Offsets are derived, not supplied

The offsets are a function of the counts, so a caller who supplied both could
supply two that disagree. They are computed by the operator that needs them,
from the counts alone.

`R` is a batch size — tens of entries — so the prefix sum is a rounding error
beside the dispatch it precedes. Making it a caller-visible tensor would buy
nothing and cost a second thing to bind wrongly.

**This is why the primitive is "a count per row" and not "counts and offsets".**
A later caller that genuinely needs offsets it cannot derive is the event that
changes this, and there is none today.

## 2. The first caller: a ragged query extent

[#16](https://github.com/golang-design/accel/issues/16). A batched step takes
one token per sequence, so a batched *prefill* is not expressible and a prefill
cannot share a dispatch with decodes. The shape says why:

```
today     q : [batch, qSeq, qHeads, headDim]     qSeq is one number for the batch
wanted    q : [Σ n_r, qHeads, headDim]           n_r is what sequence r contributes
          QueryExtents : [batch]
```

A step that mixes a 512-token prefill chunk with three decodes is
`n = [512, 1, 1, 1]`, which is the shape both references converge on: vLLM's V1
scheduler has no phase distinction at all, and sglang mixes explicitly. The
throughput argument is the consumer's and is in the issue; what matters here is
that **it is the same primitive §1 states**, not a special case of attention.

### 2.1 `QueryExtents` is not `Lengths`, and the difference is one sentence

They are both `[batch]` u32 and they are different numbers.

| | what it counts |
| --- | --- |
| `Lengths` | how many cached positions this sequence attends **over** |
| `QueryExtents` | how many query tokens this sequence contributes **this step** |

For the mixed step above, `QueryExtents = [512, 1, 1, 1]` while `Lengths` might
be `[512, 97, 1024, 33]`. A caller who binds one to the other gets a plausible
answer, which is why the distinction is stated rather than left to the names.

### 2.2 Causality, which is where the extents actually bite

A prefill token is causal against **its own position in the sequence**, and that
position is not its index in the flat buffer. Sequence `r`'s token `i` sits at
flat index $\mathrm{off}(r) + i$, and its position in the sequence is

$$p = L_r - n_r + i$$

where $L_r$ is `Lengths[r]` after this step's tokens are in the cache. So a
token attends over cache positions $0 \ldots p$ inclusive.

```
sequence r:  cache holds L_r = 6 after this step, n_r = 3 new tokens
             positions:  0  1  2 | 3  4  5
                         └ prior ┘ └ this step ┘
             flat token i=0 -> p=3, attends 0..3
                        i=1 -> p=4, attends 0..4
                        i=2 -> p=5, attends 0..5
```

Getting this wrong is not loud. A token that attends one position too far reads
the token *after* it, which is the leak causal masking exists to prevent, and
the output is a fluent continuation that used information it should not have.

## 3. What is built, and what is deliberately not

**Built here:** the counts-to-offsets derivation, `AttentionOptions.QueryExtents`,
and one corpus kernel that reads it.

**A new corpus entry rather than a generalisation of `AttentionDecodeBatched`.**
The block loop and the running softmax are the same; the mapping is not. The
batched decode puts one workgroup on a (sequence, head) pair, and a ragged step
puts one on a (token, head) pair, which needs a segment lookup to find the
token's sequence and §2.2's arithmetic to find its position. Generalising would
rewrite a kernel five differential cases compare, and change its digest, for a
body that shares a loop and nothing else.

The choice also settles the disambiguation, which is worth stating because the
alternative was a trap. A flat ragged `q` is rank 3, and so is a single-sequence
prefill: the two are separated only by whether `QueryExtents` was supplied.
Presence-of-a-field is the shape this project refused twice this milestone
(`Pages` on prefill, `BaseName` on decode). With a **separate kernel** the plan
digest distinguishes the two structurally — [029](029-plan-cache.md)'s hash
covers the kernel's name and its digest — so nothing has to remember to record a
presence bit. §5 checks that rather than assuming it.

**And the kernel is not the only thing that separates them, which was found by
checking.** Pointing the ragged branch at the prefill kernel *still* produces a
different identity, because a ragged step also binds an operand a prefill does
not — the derived offsets — and the digest covers the operand set as well. So
the two readings differ twice over, and either difference alone is sufficient.
Written down because the first version of this section named only the kernel,
which was true and was not the whole reason: a reader who later merged the two
kernels would have found the guarantee still holding for a reason this spec had
not given them.

**Not built here:** [#18](https://github.com/golang-design/accel/issues/18)'s
grouped GEMM. It is the second caller of §1 and the reason §1 is written as a
primitive: the counts are tokens routed per expert instead of query tokens per
sequence, and nothing above §2 changes. Building it needs a GEMM whose row
extent is device data, which is its own kernel and its own spec.

## 4. Refusals

Every one is host-side, because [043](043-per-row-values.md) §2's rule cuts both
ways: a value that reaches the device as data cannot be checked there.

| Refused | Why not clamped or padded |
| --- | --- |
| `QueryExtents` whose element count is not the batch | one count per row is what the primitive is |
| $\sum n_r \ne$ `q.shape[0]` | §1 property 3: the host can check it and the kernel cannot |
| `QueryExtents` on a rank-2 or rank-4 `q` | rank 2 is one token and rank 4 is the rectangular batch; a ragged step is the flat form or it is nothing |
| `QueryExtents` without `Lengths` | §2.2's position needs $L_r$, and a ragged step with no lengths has no causality |
| `QueryExtents` with no `Pages` | the existing batch argument: sequences of different lengths cannot share a contiguous cache without padding every one to the longest |

A count of **zero** is not in this table. §1 property 1 makes it legal.

## 5. Done

Each assertion names the mutation it catches.

- **A ragged batch of one sequence equals the single-sequence prefill**, element
  for element. This is the accepting half of the whole spec: if the ragged path
  disagrees with the path it generalises, nothing below matters.
- **A mixed step equals its members run separately** — a 3-token chunk beside
  two decodes matches one prefill and two decode steps over the same caches.
  Running the members separately is the oracle, which is the same shape
  [043](043-per-row-values.md) §8 used for the batch axis.
- **A count of zero contributes nothing and disturbs no other row**; a row of
  zero tokens between two non-empty ones is the case an off-by-one in the
  segment lookup turns into a shifted batch.
- **A token attends its own position and not the next one.** Constructed so the
  next position holds a value that would change the output: reading it is the
  causal leak §2.2 describes, and it is invisible in a smooth distribution.
- **$\sum n_r \ne$ `q.shape[0]` is refused at record time**, naming both
  numbers. Treating the tail as padding passes every other test here.
- **Two graphs differing only in whether `QueryExtents` was supplied have
  different plan identities**, asserted through [`Builder.Identity`] rather than
  reasoned from the digest's inputs — §3 claims the kernel choice makes this
  structural and this is what checks it.
- **CPU and Metal agree** on a mixed step, bit for bit where the arithmetic is
  exact and within [008](008-numerics.md) §6's softmax bound where it is not.

## 6. Open

- **The segment lookup's cost.** One workgroup per (token, head) has to find its
  token's sequence. At batch sizes in the tens a linear scan of `off` is a few
  comparisons and is what §5 measures; a binary search matters only if a batch
  reaches the hundreds, and [040](040-batch-scheduler.md) has not said it does.
- **Whether `Lengths` should become derivable.** In a ragged step `Lengths[r]`
  and `QueryExtents[r]` are related through the cache's prior occupancy, which
  the caller also knows. Deriving one would remove a binding and a way to get it
  wrong, and it would also couple two things a scheduler currently sets
  independently. Left alone until a caller asks.
