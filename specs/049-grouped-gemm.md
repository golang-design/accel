---
title: "Grouped GEMM: one weight matrix per segment, chosen at runtime"
status: implemented
layer: tensor
depends_on:
  - 010-kernel-corpus.md
  - 027-quantization.md
  - 043-per-row-values.md
  - 046-segmented-extents.md
---

# Grouped GEMM

**One thing:** which weight matrix a row multiplies against is device data.

[#18](https://github.com/golang-design/accel/issues/18). A mixture-of-experts
layer replaces the dense MLP with $E$ experts and a router that picks $k$ per
token — 30B parameters at 3B of compute, which is the trade that matters on one
device. The naive form is expressible today and useless: running every expert
and masking does $E/k$ times the work, which inverts the reason MoE exists.

## 1. It is 046's extent, applied to a matmul

$E$ independent $[n_e, K] \times [K, N]$ products where the $n_e$ are runtime
values summing to $T\cdot k$. That is
[046](046-segmented-extents.md) §1 unchanged — a flat buffer, a count per row,
the offsets those counts imply — with the row being an **expert** instead of a
sequence:

```
offsets = [0, 3, 3, 7]        experts got 3, 0, 4 tokens
x       = [7, K]              tokens gathered by expert
w       = [E, K, N]           one matrix per expert
out     = [7, N]
```

**A count of zero is an expert nothing routed to**, which
[046](046-segmented-extents.md) §1 already makes an ordinary member rather than
a case. **A token past the last expert's segment routed nowhere**, which the
same property makes padding: it reads no weights and its output row is zero. That matters more here than it did for
attention: with top-2-of-8 routing, six experts get nothing on a single token
and a naive implementation divides by the count.

So this spec adds no concept. It adds a kernel and the operator that records it.

## 2. One workgroup per (token, column)

The row kernels' shape: a workgroup reduces over $K$ with its lanes striding,
and a tree reduction finishes it. What is new is one lookup — a workgroup finds
which expert its token routed to by counting the offsets that end at or before
it, exactly as [046](046-segmented-extents.md)'s ragged attention does, and
indexes the weight tensor by that.

$$\text{out}[t][n] = \sum_k x[t][k]\; w[e(t)][k][n], \qquad
e(t) = \big|\{r : \text{off}(r{+}1) \le t\}\big|$$

**A matvec per token rather than a tiled GEMM per expert**, and that is a
scoping choice with a reason. Decode is where MoE's memory advantage is felt: a
step routes one token to $k$ experts and reads those $k$ matrices. A prefill
wants the tiled form, which is the same extent and a different kernel: §5.

## 3. Routing is not here, and composes

The gate is a small `MatMul` over $[T, E]$ and the top-$k$ is
[028](028-sampling.md)'s, which already walks a distribution and keeps the
largest $k$ — over $E$ instead of over a vocabulary. What this spec does not
have and does not need is a kernel for either.

**The gather is the caller's.** Tokens arrive already ordered by expert, which
is what makes the extent contiguous. Producing that order from a routing table
is a sort, and a sort of $T\cdot k$ small integers on the host is cheaper than
the dispatch it precedes — the same argument [046](046-segmented-extents.md)
§1.1 makes for deriving offsets rather than taking them.

## 3.1 Built — 2026-08-26

§1's shape, §2's kernel, and `tensor.GroupedMatVec`. Every assertion in §4 is
built, and the spec needed no new concept: `segmentOffsets` is the one
[046](046-segmented-extents.md) already had, and this is its third caller.

**The lookup mutation is the one worth naming.** Making every token read
expert zero fails three tests — but it would pass every test a single-expert
fixture could write, which is why §4 states it as an assertion rather than
leaving it to a general correctness check.

## 4. Done

Each assertion names the mutation it catches.

- **One expert's tokens match that expert's matrix multiplied alone**, which is
  the whole claim: a grouped GEMM is right when each segment equals the
  ungrouped product of its own rows.
- **An expert with no tokens contributes nothing and shifts nothing**, the case
  top-$k$ routing produces on every step.
- **Two experts with different matrices give different answers for the same
  token**, which fails for a kernel that indexes the weight tensor by anything
  other than the segment it looked up — including one that always reads expert
  zero, which passes every single-expert test.
- **A token past the last expert's segment writes zero and reads no weights**,
  [046](046-segmented-extents.md) §1 property 3. The stray index lands on the
  weight base here rather than on an offset, so the read is a matrix past the
  tensor — on a GPU, whatever allocation follows it, read as weights.
- **CPU and Metal agree** within [008](008-numerics.md) §7's reduction bound,
  which is what a sum of products carries.

## 5. The prefill shape — 2026-08-27

`GroupedMatMul`. A decode step routes one token to $k$ experts; a prefill has
many tokens per expert, and reading an expert's matrix once per token spends the
bandwidth this layer exists to save.

### 5.1 A workgroup owns an expert, not a token

This is the whole design, and the alternative fails for a specific reason.

```
grid.X = ceil(N / TileN)      column tiles
grid.Y = E                    one workgroup per expert
```

A token-blocked grid — the natural one, and what [MatMulTiled] uses — puts
`TileM` consecutive rows in a tile. Those rows are consecutive in $x$, which is
ordered by expert, so a block that straddles a segment boundary holds rows
needing **two different weight matrices**. A shared tile can hold one. So either
the caller pads every segment to a multiple of `TileM`, or the kernel masks and
re-runs, or the workgroup owns a segment. The third costs nothing and constrains
no caller.

Each workgroup walks its own segment in blocks of `TileM`:

$$
\text{out}[t][n] = \sum_k x[t][k]\,w[e][k][n],
\qquad t \in [\mathrm{off}(e),\ \mathrm{off}(e{+}1))
$$

**An expert nothing routed to has `first == last` and the loop does not run.**
[046](046-segmented-extents.md) §1 property 1 again, and here it costs a
workgroup launch rather than a branch — which matters because top-2-of-8 makes
it the common case.

### 5.2 What the tiling buys, exactly

Each weight is read once per **token block**, so a segment of $n$ tokens reads
its matrix $\lceil n/\text{TileM}\rceil$ times instead of $n$ times.

It is *not* once per expert. The $K$ loop is inside the token-block loop, so a
new block reloads both tiles. Once per expert would need the whole matrix
resident, which is a different kernel with a different constraint. Stated
because the weaker claim is the true one and the stronger one is what a reader
assumes.

### 5.3 The loop bound is clamped, and that is memory safety

`last` comes from the offsets, which are device data. Nothing on the host can
check that the counts sum to $x$'s row count — [046](046-segmented-extents.md)
§1 property 3 — and this kernel's row index comes from those offsets rather than
from the grid.

**That makes the over-sum direction dangerous here in a way it is not in
`GroupedMatVec`.** There the grid is derived from `x.shape[0]`, so a token index
cannot leave the buffer however wrong the counts are, and counts summing past
the rows are merely a wrong answer. Here they would read $x$ and **write** `out`
past their ends.

`Tokens` is `x.shape[0]`, which the host *does* know, and clamping `last` to it
turns the stray write into a wrong answer. The under-sum direction is
[046](046-segmented-extents.md) §1 property 3's padding and needs nothing here:
rows past the last segment fall in no expert's loop and are never written.

### 5.4 Done

- **The tiled product equals the row kernel element for element**, within
  [008](008-numerics.md) §7's reduction bound, over counts that are not
  multiples of `TileM` and including an expert with none. Two different
  traversals of one extent agreeing is the claim.
- **Counts summing past $x$'s rows are clamped, not written past the end.**
  Removing the clamp fails this with an out-of-range index, which is the
  accepting half the guard needs.
- **CPU and Metal agree** within the same bound.
