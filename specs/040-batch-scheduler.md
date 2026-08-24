---
title: "The batching scheduler: who runs together, and when a request is admitted"
status: drafted
layer: tensor
depends_on:
  - 003-command-graph.md
  - 007-tensor-layer.md
  - 029-plan-cache.md
  - 030-paged-kv.md
  - 031-shared-transients.md
  - 028-sampling.md
  - 039-sampling-policy.md
  - 010-kernel-corpus.md
---

# The batching scheduler

[030](030-paged-kv.md) §4 ends by naming what it does not build:

> What remains is the *scheduler* — deciding which sequences run together, when
> to admit one, and what to do when a batch's members finish at different steps.
> That is policy over this mechanism rather than more of it.

This is that policy. [007](007-tensor-layer.md) places it post-v0 and this spec
does not move it. It states the design now so that the parts which are cheap
today are not made expensive by accident.

## 1. What this is policy over, and what it is not

`AttentionDecodeBatched` gives one dispatch stepping several sequences, each
reading its own length and its own page table, with nothing padded. The
**append** side batches too: `ScatterRows(b, s, rows, ids)` takes ids as *device
data*, so B tokens land in B unrelated physical rows in one op.

The rest of a batched step does not exist. Calling this pure policy would hide
three kernel gaps:

| Gap | What exists today | What a batch needs |
| --- | --- | --- |
| ~~paged attention at the tensor layer~~ **closed 2026-08-24** | `tensor.Attention` selected only the decode and prefill kernels; the batched and paged kernels lived under `internal/testkernels` and nothing in `tensor/` referenced them | `AttentionOptions` binds `Pages` and `Lengths`, and the operator selects the paged kernel. *Batched* attention — several sequences in one dispatch — is still this spec's, and is what the scheduler needs |
| ~~position rotation~~ **closed 2026-08-24** | `RoPE` took one scalar `Offset` and computed `pos := r + Offset`, so the *row index* was part of the position | a positions tensor, one entry per row, read as a binding — see [043](043-per-row-values.md) |
| ~~sampling~~ **closed 2026-08-24** | `SampleDims.Draw` was a scalar and `SampleCategorical` was `workgroup=1` writing `out[0]` | one row and one **independent** draw per slot — built, and `SampleArgmax` is per row too, since one batched sampler beside one single-row sampler would leave a caller choosing |

The `RoPE` gap was the dangerous one, and a consumer building on the library
reported it before this spec's scheduler existed to trip over it. In an unbatched prefill `r` really is the
position within the prompt, so the kernel is correct there. In a batched decode
`r` is the *slot index*: slot 0 rotates at `Offset`, slot 1 at `Offset+1`, and
only one member is ever rotated at its own cache length. The output stays
well-shaped, finite and plausible. Sampling fails the same way: widening a
shared `Draw` keeps [028](028-sampling.md)'s reproducibility and destroys
independence, so two sequences with similar distributions emit the same token
and every existing test still passes.

**This spec states the gaps and does not own them.** The three kernels belong to
[010](010-kernel-corpus.md)'s registry. Everything below is written to be true
once they exist.

## 2. Slots: membership is contents, batch size is structure

A **slot** is an index in `[0, maxBatch)`. It owns three addresses in the
batched kernel — `qBase = (seq*QHeads + h)*HeadDim`, the matching `out` row, and
the page-table row at `pageBase = seq*MaxPages` — plus its entry in `lengths`.
The KV blocks are addressed *through* the page table, so they never move. That
is what paging is for, and it is what makes a slot swap cheap.

Against [003](003-command-graph.md)'s four kinds of variation:

| Change | Which variation | Cost |
| --- | --- | --- |
| a sequence leaves, another takes its slot | contents: `lengths`, the page-table row, the q row, the scatter id | free; no rebind, no recompile |
| a sequence grows by a block | contents: one page-table entry | free |
| the batch gains or loses a member *count* | none of the four; batch is a leading dimension on every port | a different plan, and §5's drain |

A **membership** change at fixed size is variation 1 and costs nothing. A
**size** change is structure. The design below never changes size.

> **`BatchedDims.Batch` is declared and never read.** The kernel derives
> everything from `t.GroupID().X`, `d.QHeads` and `d.MaxPages`. A scheduler that
> "shrinks the batch" by writing `Batch` runs the dropped slot anyway, and that
> slot's `out` row still holds what the previous step left there — so the model
> emits a **repeat of the last token** rather than obvious garbage. Changing the
> dispatch count and leaving `Batch` stale does nothing at all, which teaches a
> reader that the uniform is the lever. Both symptoms point away from the cause.
> `Batch` is not a lever. The dispatch width is `maxBatch` on every step.

## 3. One plan at `maxBatch`, and what an idle slot does

[033](033-render-api.md) §4.1 states the same problem for scenes:

> a scene that gains an object needs either a rebuilt graph or a graph recorded
> for a fixed maximum with absent objects issuing zero-instance draws.
> Zero-instance draws are cheap but not free.

**The analogue: one decode plan recorded for `maxBatch`, with departed slots
left idle.** This is [007](007-tensor-layer.md)'s own fallback — *"v0 may mask
to capacity on a backend whose replay object bakes it in"* — and the tensor
layer computes a node's `accel.WorkgroupCount` at compile time, so a
per-submission dispatch count is not reachable from here anyway.

**Compaction is the road not taken.** The alternative is to slide live
sequences to the front and dispatch `activeBatch·QHeads`. It removes the idle
work, and it moves the slot index — which is the address of three different
things (§2). A compaction that moves a sequence and does not also copy its
`MaxPages` u32s of page table silently hands it another conversation's blocks,
and the dispatch count it needs is not reachable from the tensor layer anyway.
Idle slots trade measurable work for a class of bug that has no symptom.

**A zero-instance draw writes nothing. A zero-length slot writes NaN.** With
`lengths[slot] == 0` every lane fails `lane < kvLen`, `red[]` is all `-3.4e38`,
`total` sums to zero, and `out[qBase+lane] = acc / total` is `0/0`. The obvious
reading of 033 §4.1 does not transfer, and naming that is the most useful thing
this section does.

**So an idle slot is parked, not emptied.** The pool reserves one **parking
block** at construction, zeroed once, withheld from the free list and never
freed. An idle slot's page-table row points every entry at it and its length is
1. Then `s = 0`, `best = 0`, `total = 1`, `acc = 1·0`, and the idle row is
**exactly zero** — the value a `kvLen == 0` kernel amendment would write, so
that amendment stays a cost reduction rather than a behaviour change.

Two details are load-bearing. A parking block left in the free list is handed to
a live sequence, which then writes its tokens into every idle slot's read
source. And the idle slot's **input token id is written to 0** each step rather
than left stale: otherwise the embedding gather reads an arbitrary id, the q row
can carry Inf or NaN, and `0 · Inf` is NaN again — the failure parking exists to
prevent. A zeroed `k` alone is not enough; both operands must be finite.

A parked row of zeros survives the rest of the step because `RMSNorm` carries an
epsilon under the square root, and because **no registered op reduces across the
batch axis**: `SoftmaxOptions.Axis` must be the last, `RMSNorm` reduces over
`width`, `MatMul` broadcasts over leading axes. That is an invariant with a test,
not an observation. The first op that reduces across the batch axis lets one
idle slot poison every live sequence's logits in the same step, and the symptom
is every conversation breaking at once.

**Admission rewrites the whole slot**, including a parked one. A sequence that
inherits a departed one's page-table row reads another conversation's blocks,
which [030](030-paged-kv.md) §4.1 calls *"close to undebuggable from a model's
output"*. Writing the row is `MaxPages` u32s; skipping it is a wrong
conversation.

## 4. What an idle slot costs

An idle workgroup still launches and still runs both halving reductions: two
loops over 128 lanes at 7 iterations each, plus three standalone barriers, is
**17 barriers**. Parking adds one pass of each `kvLen`-bounded loop at length 1,
which a zero-length slot would skip — marginally more work in exchange for a
defined row.

$$
\text{step} \;\approx\; \underbrace{\max_{i \in \text{live}} L_i}_{\text{030: a max, not a sum}} \;+\; \underbrace{(\text{maxBatch} - |\text{live}|)\cdot Q_{\text{heads}}}_{\text{idle workgroups, 17 barriers each}}
$$

The left term is why batching pays at all. The right term counts attention
only, and an idle slot pays more than that. Batch is a leading dimension on the
embedding gather, every matmul, every norm and the sampler, and §3's parked
token id exists precisely so the **whole** forward pass runs on the idle row.
The marginal weight bandwidth is zero — that is why batching pays — but the
FLOPs and the activation bytes are not.

The larger cost is memory. `maxBatch` sizes the decode graph's transients, so it
sets the shared pool's high-water mark, and [031](031-shared-transients.md) §2's
pool *"grows and never shrinks"*. A generous `maxBatch` is therefore a permanent
device-memory reservation, not a spare seat: it is a capacity decision and not a
safety margin.

## 5. Continuous batching is not continuous here

A departure and a re-admission at fixed size are free (§2). Everything else
drains, and the term should not be borrowed without the price.

[031](031-shared-transients.md) §2: *"Two graphs sharing a pool cannot be in
flight together"*. The claim covers execution and the refusal arrives through
the fence rather than queueing. [029](029-plan-cache.md)'s bucket plans and the
decode plan share one pool, because a pool each restores 003's *"five prefill
buckets at 200 MiB is a gigabyte of transients"*. So a prefill cannot overlap a
decode step, and admitting a request costs one drained step.

```
 step:      1     2     3     4                 5     6     7
 slot 0    [A]   [A]   [A]   [A]               [A]   [A]   [A]
 slot 1    [B]   [B]   [B]    ·                 ·    [D]   [D]
 slot 2    [C]   [C]   [C]   [C]               [C]   [C]   [C]
 slot 3     ·     ·     ·     ·                 ·     ·     ·
                        │          ╲
                  B emits EOS       ╲ fence ── prefill D ── fence
                  slot 1 parked      the bubble: one drained decode step
```

The parked slots (`·`) still dispatch, at §4's price. The gap between step 4 and
step 5 is a fence round trip plus D's prefill, and nothing decodes inside it.

**`Build` is refused while anything is in flight** (031 §2), so "compile the plan
we just discovered we need" is not available mid-step either. That is the second
reason the batch-size set has exactly one member.

**The plan set is `B + 1`** for `B` buckets. Batch size is a leading dimension
on every port, so `Builder.Identity` already keys it correctly: a batch-size set
would be *safe*, not *bounded*. 029 §3's cache *"evicts nothing — it grows until
closed"*, so the risk is a plan per observed batch size, each holding a graph
and pipelines.

## 6. Admission, preemption, and the single release path

**Admission refuses what can never run, and queues what cannot run yet.** 031 §2
refuses to queue because a pool that queued *"would turn a design mistake into a
latency mystery"*; a scheduler is the component whose job *is* the queue, so it
sits on the other side of that line and says so.

| Condition | Answer |
| --- | --- |
| prompt longer than the largest bucket | refuse, naming the bucket — 029 §1: never truncate |
| requested length past $L_{\max} = \min(128,\ \text{MaxPages}\cdot\text{Block})$ | refuse, naming **which** cap bound |
| worst-case blocks exceed the whole pool | refuse — it can never be admitted, so queueing it is a hang |
| no free slot, or not enough free blocks right now | **wait**, in arrival order |

**One length cap remains, and admission owes it.** The 128-position cap is gone:
[044](044-unbounded-context.md) made the decode kernels walk the cache a block
at a time, so `workgroup=128` bounds a block and not a cache.

The batched kernel's cap is the one that is left, and it changed shape rather
than going away. It reads `pages[pageBase + pos/Block]` with
`pageBase = seq*MaxPages`, so a length above `MaxPages·Block` used to index into
slot `seq+1`'s page-table row — another conversation's physical blocks — and run
off the buffer for the last slot. The kernel now clamps the length to
`MaxPages·Block`, so such a length is **truncated** instead: the answer attends
over a prefix of the sequence. The clamp is explicit and not a consequence of
the loop bound — the loop advances by a block and each lane offsets it, so the
bound stops `base` and not `base+lane`. See [044](044-unbounded-context.md)
deviation 6. That is better than another conversation's keys and is still
wrong, and the kernel cannot tell — the length is device data. Under `MaxPages = 2, Block = 16` the cap is 32,
and an admission that checked only the cache's capacity takes a 100-token
request and silently answers over its first 32.

**Above the reservation bound, eviction never fires.** $L_{\max}$ makes a
sequence's worst case known at admission, so

$$
N_{\text{blocks}} \;\ge\; \text{maxBatch}\cdot\left\lceil \frac{L_{\max}}{\text{Block}} \right\rceil + 1
\qquad (+1 \text{ is the parking block})
$$

is the pool size at which `Grow` never refuses. Below it eviction is reachable,
and this spec must overrule 030's deferral rather than inherit it.

**Eviction recomputes or refuses; it never truncates** — 029 §1's rule applied
to the other end of a sequence. The consequence is an interface one: the
scheduler retains each sequence's **token ids**, not only its page table,
because recompute means re-prefill. A victim returns its blocks, re-enters the
waiting list and resumes as a prefill, so eviction carries §5's bubble. Like 031
§6's two refusals, the path is specified and tested even while a correctly sized
pool cannot reach it, because a rule nothing checks decays.

**One release path, keyed on the sequence's state.** `BlockPool.Free` refuses a
double free, because *"it would hand one block to two sequences, and the symptom
is one sequence reading another's tokens"*. A scheduler frees on completion and
on eviction, so a sequence evicted and then observed to emit EOS hits that
refusal — an error a serving loop is tempted to log and ignore. Release is one
function guarded by the sequence's state: 031 §7's "one fact, one place".

**Fairness is FCFS, and head-of-line blocking is real.** 030 makes a step cost
what the *longest* member costs, so a batch mixing one long sequence with three
short ones charges everybody the long one's price. The alternatives trade
against each other — packing similar lengths favours throughput,
shortest-remaining favours tail latency — and the 128-position cap keeps the
length spread too narrow for the trade to be measurable yet. A running sequence
is never preempted for a waiting one; eviction is the only preemption, and
departures free slots, so a waiting request cannot starve.

**A bigger batch does not buy finer-grained overlap.** Every paged kernel
declares the whole block pool, because addressing computed from buffer contents
cannot declare a tight sub-range. [003](003-command-graph.md) records this as a
known hole in an otherwise exhaustive list — a per-step *address* is not one of
its four kinds of variation — rather than as one of the four, and citing the
numbered item would turn a documented deferral into an apparent rule. Two sequences in one batch take
edges against each other even though they touch disjoint blocks. The win is the
dispatch, not the dependency graph.

## 7. The surface a serving loop drives

The scheduler holds a `PlanCache`, a `BlockPool`, a `Queue` and per-sequence
host state. It lives in `tensor/serve`, which imports `tensor` and is not
imported by it. [029](029-plan-cache.md) kept `PlanCache` inside `tensor`
because a cache of plans is a plan-shaped object; a scheduler is a queue, a
policy and host state, and putting it in `tensor` would make `BlockPool` grow an
owner instead of staying a value the caller passes in.

```go
package serve // tensor/serve

type Config struct {
    MaxBatch int                // slots, and the recorded dispatch width
    Buckets  tensor.Buckets     // 029's prefill bucket set
    Blocks   *tensor.BlockPool  // caller-owned; the scheduler reserves the parking block
    Plans    *tensor.PlanCache
    Queue    *accel.Queue
}

func New(cfg Config) (*Scheduler, error)

// Admit refuses what can never run and queues what cannot run yet.
func (s *Scheduler) Admit(prompt []uint32, draws func(step int) float32) (*Seq, error)

// Step runs one submission: a prefill if the waiting head needs one, otherwise
// a batched decode over the live slots. It returns the tokens produced.
func (s *Scheduler) Step() ([]Token, error)

// Release is the single free path. Completion, eviction and caller
// cancellation all arrive here, and a second call for one sequence is a no-op
// rather than a double free.
func (s *Scheduler) Release(seq *Seq) error

func (s *Scheduler) Close() error
```

`Step` is the step boundary: one `Plan.Submit` and one fence wait. Every
admission, departure and slot swap happens on the host between that fence and
the next `Submit`, because 003's one-submission-in-flight rule leaves nowhere
else to put them. `Admit` is callable from another goroutine; a loop that
overlaps arrival handling with a running step selects over `Fence.C()` and its
own arrival channel, which is what `C()` returning a channel is for. `draws` is
per sequence, because 028 makes reproducibility a per-sequence property.

## 8. What it costs, stated plainly

- **`maxBatch` is paid in device memory permanently**, because it sets the
  shared pool's high-water mark and the pool never shrinks (§4). The per-step
  idle work is the smaller half of that cost.
- **One parking block** withheld from the pool, plus a token-id write per idle
  slot per step.
- **A drained step per admission and per eviction** (§5). Removing it costs
  either 007's gigabyte or 003's deferred *"several transient sets… a submission
  carries a pool index"*, which is unbuilt.
- **`B + 1` plans held until close**, each with a graph and pipelines. 031 caps
  the transient bytes and nothing caps the rest.
- **Three kernels this design depends on and does not own** (§1).

## 8.1 Two things this spec and [039](039-sampling-policy.md) must settle together

Both were drafted at once and neither owns the seam between them. Recorded here
so it is closed deliberately rather than by whichever is implemented first.

**The draw has two spellings.** 039 §2 makes `tensor.Stream{Seed uint64}` with
`Draw(step uint64)` and `Derive(root, seq)` the whole of its design — "there is
no generator object". This spec's §7 declares `Admit(prompt []uint32, draws
func(step int) float32)`, which is the same concept with a different owner, and
leaves 039's `Derive` — the operation that exists precisely so a batch advances
without interleaving — with no consumer. **`Admit` should take a
`tensor.Stream`**, and the scheduler should call `Derive` per admitted sequence.
A closure would let a caller share one generator across sequences by accident,
which is the failure 039 designed `Stream` to make hard.

**The plan set has a policy dimension this spec omits.** §5 counts `B + 1`:
one plan per prefill bucket plus one decode plan. 039 §7 makes sampling *shape*
structural — greedy-or-categorical, penalties on or off, top-k on or off, top-p
on or off, history capacity, vocabulary — so two shapes are two plans. A loop
admitting one greedy request and one top-p request cannot run them in one decode
plan. Either the count is `B + |shapes|`, or the scheduler imposes one policy per
batch and says so. **Unresolved**, and it decides whether per-sequence `k` and
`p` are worth their bindings — which 039 §10 raises and this spec had not.

**The batched sampler is orphaned.** 039 §8 says a batched sampler needs new
kernels and that they are "not in this spec"; §1 here says the same and points
at [010](010-kernel-corpus.md)'s registry. Two specs deferring to a third that
neither amends is how a gap survives both. 010 has to gain the rows, and
whichever of 039 or 040 is implemented first amends it.

## 8.2 The query extent, which is the last per-dispatch value

A consumer asked for this shape before building admission around it, which is
cheaper to give than the feature (accel issue 16). It is recorded here rather
than built.

`Attention` takes a rank-4 query, `[batch, qSeq, qHeads, headDim]`. Every other
per-sequence value became a tensor under [043](043-per-row-values.md)'s rule —
lengths, page tables, RoPE positions, sampling draws — and `qSeq` did not. It is
**a single leading extent shared by the whole batch**, so it is a value that
differs per row wearing the shape of one that does not. That is the same mistake
043 found five times, in the one place 043 did not look.

It has two consequences and they are not the same size.

### Batched prefill is a kernel away, and the shape already fits

Several sequences prefilling together with the *same* token count is rank 4 with
`qSeq > 1`, which the operator already parses and refuses for want of a kernel.
Nothing about the shape is wrong for it.

It is also less of a special case than it looks, because
[029](029-plan-cache.md) buckets prefills: a bucket set rounds every prompt up
to one of a few lengths, so a batch of bucketed prompts *has* a uniform `qSeq`
by construction. Batched prefill is therefore a corpus item —
`attention_prefill_batched`, the paged prefill with a sequence-major grid — and
not a design question.

### Mixed prefill and decode is the design question

A dispatch carrying one sequence's 512-token chunk alongside four decode steps
is **ragged**: `qSeq` is 512 for one row and 1 for the others, and no dense
leading extent expresses that. This is where the throughput is. Chunked prefill
without it bounds latency and recovers nothing — the decodes still wait for the
chunk's whole forward pass, in smaller pieces — which is why vLLM's V1 scheduler
has no phase distinction at all and sglang mixes explicitly.

The shape that expresses it is the packed one every serving stack converges on:

$$q : [\textstyle\sum_s n_s,\; \text{qHeads},\; \text{headDim}], \qquad
\text{QueryLengths} : [B], \quad \sum_s \text{QueryLengths}[s] = \textstyle\sum_s n_s$$

A flat token buffer plus one count per sequence. Decode is $n_s = 1$, a prefill
chunk is $n_s = c$, and a mixed step is both in one dispatch — with no phase
distinction anywhere in the operator, which is the property to aim for rather
than a mode to add.

**Where the rank goes.** The packed form is rank 3, which today means a
single-sequence prefill. It does not collide: a single prefill is the packed
form with `B = 1`, and `QueryLengths` is what says which reading applies —
present means packed, absent means the whole extent is one sequence's. That is
[043](043-per-row-values.md)'s rule paying for itself a second time; the value
that decides is a tensor, and a batch of one is the same path.

**What it costs.** The causal mask stops being a function of the row index and
becomes a function of the row's position *within its sequence*, so the kernel
needs the per-sequence offset the packed layout implies. That is one extra
lookup per query position, in a kernel that already reads a page table per
position. It is the same shape of change as paging, and 044 §5's observation
applies again: a prefill already walks the cache in blocks, so the indirection
goes where the walk is.

**Order.** Batched prefill first, because it is a kernel over a shape that
already parses and it makes bucketed batching work. The packed form after, and
only when a scheduler exists to drive it — the consumer says their scheduler can
be built against a batched decode plus prefills that run alone, and a shape
built ahead of the thing that drives it is how `AttentionDecodeBatched` sat
unreachable for four milestones.

## 9. Open questions

- Whether `AttentionDecodeBatched` gains a `kvLen == 0` zero-write, retiring the
  parking block. It amends [010](010-kernel-corpus.md) and
  [030](030-paged-kv.md), so this spec cannot decide it alone.
- Whether prefill earns a second transient pool, or waits for 003's deferred
  several-transient-sets. Both remove §5's bubble at different prices.
- Whether more than one decode batch size ever pays: a declared batch-size set
  shaped like `Buckets`, against `B × K` plans and 031's one pool.
- Whether swapping a victim's blocks to host memory is worth a third eviction
  outcome. No path is spec'd; [001](001-device-resources.md)'s memory kinds
  would have to be involved.
- Fairness beyond FCFS, measurable only once the looping attention variant lifts
  the 128-position cap and length spreads widen.

## 10. Done

Checkable against the kernels that exist today:

- a slot vacated and refilled leaves every **other** member's output
  bit-identical to its solo run, which is what makes membership a contents-only
  change;
- a sequence admitted into a previously occupied slot never reads the departed
  one's blocks, tested with per-sequence signed values as 030 §4.1 does, so a
  stale page-table row shows as a wrong sign rather than as a plausible token;
- an idle slot's output row is **exactly zero**, not NaN, over a step where
  other slots are live, and that row survives `RMSNorm` and the rest of the step;
- no registered op reduces across the batch axis, asserted directly, because the
  first one that does poisons every live sequence in the same step;
- the parking block is not in the free list: `Available()` after construction is
  one below capacity, and no `Grow` ever returns it;
- `Free` runs exactly once for a sequence evicted and then observed to finish,
  rather than reaching `BlockPool.Free`'s double-free refusal;
- a prompt past the largest bucket, a length past $L_{\max}$, and a request
  larger than the whole pool are each refused with an error naming which limit
  it hit, while a request that merely has no free slot waits;
- a request past `MaxPages·Block` is refused **even when it is under the
  128-position lane cap**, because the kernel would otherwise read the next
  slot's page-table row;
- the recorded dispatch width is `maxBatch` on every step, asserted against the
  plan rather than against the `Batch` uniform, which nothing reads;
- an eviction returns the victim's blocks and re-admits it as a prefill from
  retained token ids, producing the continuation it would have produced without
  the eviction.

Blocked on §1's three kernels, and listed so they are not mistaken for runnable
today:

- every member of a batch rotates at **its own** cache length, which the
  scalar-offset `RoPE` cannot do;
- two slots with identical logits and different draws emit different tokens,
  which a shared scalar `Draw` cannot do;
- a batched decode step reaches the batched paged kernel through
  `tensor.Attention` rather than through `internal/testkernels`.
