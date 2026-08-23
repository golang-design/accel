---
title: "A paged KV cache, and the sharing it makes possible"
status: implemented
layer: tensor
depends_on:
  - 007-tensor-layer.md
  - 026-tensor-decode.md
  - 029-plan-cache.md
---

# A paged KV cache

[009](009-sequencing.md)'s M8 item "paged KV, multi-sequence scheduling, and
additional transient sets", and [007](007-tensor-layer.md)'s "paged caches,
cache quantization, multiple sequences in one cache, and cache eviction are
post-v0".

## 1. What a contiguous cache costs

[026](026-tensor-decode.md)'s cache is `[capacity, kvHeads, headDim]` per
sequence, and `capacity` is fixed when the plan compiles. A server that admits
sequences of unknown length has two bad options: size every cache for the
longest sequence it will ever see, and waste the difference on every short one;
or size them small and refuse a long one.

The waste is not marginal. A cache sized for 4096 positions serving a
20-token conversation holds 0.5% of what it reserved, and the reservation is
device memory that nothing else can use.

## 2. Pages

Split the cache into fixed-size **blocks** and give each sequence a **page
table**: a list of physical block indices, one per logical block of its
positions.

$$\text{physical}(j) = \text{pages}\!\left[\left\lfloor j/B \right\rfloor\right] \cdot B + (j \bmod B)$$

```
 sequence A pages: [2, 0]        sequence B pages: [3, 1, 4]
                    │  │                            │  │  │
 block pool:  ┌─────┼──┼────────────────────────────┼──┼──┼─────┐
              │  0  │  │   1        2        3      │  │  │  4  │
              │  A₁ ┘  └── B₁       A₀       B₀     ┘  ┘  └─ B₂ │
              └───────────────────────────────────────────────┘
```

A sequence grows by taking another block from the pool, and the blocks it holds
need not be adjacent — which is the whole point. Two sequences of wildly
different lengths share one pool, and a sequence that ends returns its blocks
for the next one.

**The block size is a policy with two costs.** Small blocks waste less on a
sequence's final partial block and make the page table longer; large blocks
waste more and index less. It is a parameter here rather than a constant,
because the right answer depends on a workload this spec cannot see.

## 3. What the kernel changes, and what it does not

**One indirection.** Where the contiguous kernel reads position `j` at
`j·kvHeads·headDim`, the paged one reads it at `physical(j)·kvHeads·headDim`.
Everything else — the scoring, the softmax, the weighted sum, the tie rules — is
unchanged, and that is deliberate: a paged cache is a different *addressing*, not
a different attention.

**The page table is a binding, not a uniform.** It varies per sequence and per
step, and a uniform would make every sequence its own plan — which is exactly
what paging exists to avoid.

**A logical position beyond the sequence's pages is not read.** The kernel
bounds its work by the current length, as the contiguous one does, so a page
table shorter than the cache is normal rather than an error.

## 4. What this does not build

**Multi-sequence batching, added 2026-08-23.** A pool shared between sequences
was the enabler; `AttentionDecodeBatched` is the thing it enabled. Each
workgroup handles one (sequence, head) pair and reads that sequence's *own*
length and page table.

**Nothing is padded to a common length.** A short sequence's lanes past its end
contribute the identity to each reduction, exactly as they do unbatched, so a
batch of one long and three short sequences costs what the long one costs rather
than four times it. The alternative — padding every sequence to the batch's
longest — is the obvious implementation and was reinstated as a fault: using
`lengths[0]` for everyone makes the second sequence disagree immediately.

**A decode step is memory-bound**, which is why batching is worth a kernel
rather than a loop: running four sequences as four submissions reads four caches
in four dispatches, each with a tail where most of the device is idle.

What remains is the *scheduler* — deciding which sequences run together, when to
admit one, and what to do when a batch's members finish at different steps. That
is policy over this mechanism rather than more of it.

**Not eviction.** The pool refuses when it is empty rather than choosing a
victim, because choosing one is a policy question about which sequence matters,
and a wrong answer silently truncates somebody's context.

## 4.1 Outcome — 2026-08-23

Built as `AttentionDecodePaged` and `tensor.BlockPool`. The paged step produces
**exactly** what the contiguous one does over the same logical positions — not
within a budget, because paging is an addressing change and the arithmetic is
identical, so any difference at all is an indexing bug.

**The kernel compiler caught a mutation before a test could.** Reinstating the
fault as "ignore the page table and read position j at j" was refused at
generation: *"binding `pages` is never read or written"*. A paged kernel that
ignores its page table does not compile, which is a stronger guarantee than a
test — so the fault had to be made subtler to reinstate at all. Dropping the
block multiply from `pages[j/B]·B + j mod B` reads the table, compiles, and
fails every one of the five cases.

**The out-of-order cases are the ones that matter**, and the identity mapping is
the trap: a kernel that ignored its page table would pass identity and nothing
else, and identity is what a first test naturally uses. So the differential
gives Metal a page table that maps logical block 0 to physical 5, for the same
reason.

**Two sequences interleave over one pool**, with adjacent physical blocks
belonging to different sequences, so reading one position past a sequence's end
lands in the other's data. Every value each caches carries its own sign, so an
output with the wrong sign anywhere means one conversation read another's — and
that failure is close to undebuggable from a model's output.

## 5. Done

- a pool hands out blocks, takes them back, and refuses when empty rather than
  overwriting;
- a paged decode step produces the **same output as the contiguous one** over
  the same logical positions, which is what makes paging an addressing change
  rather than a different model;
- it produces that same output when the pages are **deliberately
  out of order**, which is the case a contiguous implementation passes by
  accident;
- two sequences interleave over one pool without seeing each other's positions;
  and
- the paged kernel agrees between the CPU backend and Metal;
- a batch of sequences produces exactly what each sequence produces alone, over
  different lengths and interleaved pages; and
- a batch of one is the unbatched kernel, which is what a scheduler hits
  whenever a single request is in flight.
