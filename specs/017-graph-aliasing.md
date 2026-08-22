---
title: "Transient aliasing and the whole-plan oracle"
status: drafted
layer: device
depends_on:
  - 015-graph-recording.md
  - 016-graph-execution.md
---

# Transient aliasing and the whole-plan oracle

The third of [009](009-sequencing.md)'s three M3 children. It makes the graph's
memory small, and in the same change it makes the claim that the graph is still
correct something a test can refute.

## 1. Why the optimizer and its oracle land together

[009](009-sequencing.md)'s risk table has one row that blocks later milestones:

> Graph aliasing is unsound | M3 naive-plan fuzz and diamond golden | Block
> later milestones until fixed

A child that landed aliasing and left the fuzz for a fourth would be recorded
complete with that risk still live, which is exactly the kind of tidy history
[009](009-sequencing.md)'s maintenance rule exists to prevent. The risk is
introduced here, so it is retired here.

That is affordable because the oracle already exists.
[015](015-graph-recording.md)'s plan — no aliasing, a full barrier between
consecutive nodes in record order — *is* [003](003-command-graph.md)'s naive
plan, and [015](015-graph-recording.md) §3 proves it correct. It is retained
rather than replaced, so this child's differential test compares two executors
that were designed months apart under different constraints, one of which had
never heard of interference.

## 2. Liveness on a DAG is an interference relation, not an interval

This is the whole content of the child, so it is stated as a claim and a
counterexample rather than as an algorithm.

An interval planner assigns each transient the record-order span from its first
to its last user and aliases transients whose spans are disjoint. On a DAG that
is unsound, and the counterexample is a diamond:

```mermaid
flowchart TD
    n1["n1 writes t0"] --> n2["n2 reads t0"]
    n1 --> n3["n3 reads t0"]
    n2 --> n4["n4 writes t3"]
    n3 -.->|"unordered with n4"| n4
```

`t0`'s record-order interval is `[n1, n3]` and `t3`'s is `[n4, …]`. Disjoint, so
an interval planner aliases them. But `n3` reads `t0` and `n4` writes `t3`, and
nothing orders `n3` before `n4` — so a backend that runs the two arms at once,
which is every backend doing what the DAG permits, corrupts `t0`.

The sound test is reachability, not position:

```
compatible(T, U)  ⟺  ∀ t ∈ users(T), ∀ u ∈ users(U) :
                        reaches(t, u) ∨ reaches(u, t)
```

Every user of one must be ordered against every user of the other. On the worked
graph this is exactly the `t0`/`t3` row of [003](003-command-graph.md)'s
compatibility table, and it is why the pool is 16 MiB rather than the 12 MiB an
interval planner reports.

## 3. Packing is dynamic storage allocation

Not graph colouring: colouring assigns transients to equal slots and transients
have different sizes. The formulation is to give each transient an offset in one
pool such that interfering transients do not overlap in bytes — NP-hard, so what
lands is [003](003-command-graph.md)'s greedy heuristic, size descending with
first-writer node id as the tie break.

The tie break is not cosmetic. Without it the layout depends on sort stability,
and the plan golden flaps.

## 4. What the three `GraphMemory` fields become

[015](015-graph-recording.md) pinned `TransientBytes` to `UnaliasedBytes`. Here
they separate, and the gap between `PeakBytes` and `TransientBytes` becomes the
number that has to be explained rather than hidden:

| Field | Worked graph | Meaning of the gap |
| --- | --- | --- |
| `UnaliasedBytes` | 22 MiB | what no planning costs |
| `PeakBytes` | 12 MiB | the record-order lower bound, achievable only by an executor that ran strictly serially |
| `TransientBytes` | 16 MiB | what is allocated |

The 4 MiB between 12 and 16 is **not** fragmentation — the pool is fully packed.
It is the price of DAG-safe aliasing, and it is precisely the `t0`/`t3` pair
§2 refused to merge. Reporting both is what stops 16 from looking like a planner
failure, and 16 is optimal here: `{t0, t1, t2, t3}` are pairwise interfering, so
no assignment places four 4 MiB transients in less.

## 5. Aliasing handovers, and V24's remaining term

Two transients sharing bytes need a barrier between the last use of one and the
first use of the other, because the second's writes must not be reordered before
the first's reads. On the worked graph both handovers ride on a barrier the data
flow required anyway, so the barrier count does not change — which is an
assertion, not a hope.

V24's transient term lands here, as [015](015-graph-recording.md) §4 said it
would: a resource supplied through a slot may now overlap a *transient's*
placement, and that is rejected on the same rule as any other dynamic overlap.
Adding the term without changing the row's message is the point — a V24 that had
been silently missing a term would have been passing for the wrong reason.

Row V20 also lands here: the planned pool against the device's reported budget.

## 6. The whole-plan oracle

The strongest test available, and cheap because [015](015-graph-recording.md)
already built half of it.

```
for a randomly generated graph G:
    a = execute(G, plan = optimized)     # 016's edges, 017's aliasing
    b = execute(G, plan = record-order)  # 015's plan, retained
    assert bytes(a) == bytes(b)
```

Any disagreement is a planner or barrier bug, localized to the builder rather
than to a kernel, because both sides ran the same kernels over the same inputs.
Generation randomizes node count, access patterns, sub-ranges, transient sizes,
and — critically — graph *shape*, since the bug §2 describes needs a diamond and
a generator that only produces chains will never see it.

Determinism makes the comparison meaningful: the executor is deterministic by
[003](003-command-graph.md) §"Determinism", so a disagreement is a plan
difference and never scheduling noise.

## 7. Testing

- The worked graph of [003](003-command-graph.md) asserts 22 MiB unaliased,
  12 MiB peak, and 16 MiB allocated, and asserts the offset table row by row.
  This completes M3's numeric criterion, whose first two numbers
  [015](015-graph-recording.md) already carried.
- The diamond of §2 asserts `t0` and `t3` are **not** aliased, written as a
  direct assertion on the placement rather than as an output comparison, because
  an output comparison on the CPU backend can pass while the placement is wrong.
- A test drives the interval planner over the same diamond and asserts it
  produces the unsound layout, so the counterexample is executable and not only
  described. It is the regression that keeps someone from "simplifying"
  reachability back into intervals.
- §6's differential fuzz, with the diamond-producing seed committed.
- Aliasing the worked graph adds no barrier, asserted against
  [016](016-graph-execution.md)'s count.
- V20 and V24's transient term have focused negative tests; V24's is a rebind of
  a slot onto a range overlapping a transient placement.
- A benchmark reports packing cost against transient count, since
  [003](003-command-graph.md) claims `O(n² log n)` and claims it is acceptable at
  200 transients.

## 8. What it does not build

[003](003-command-graph.md) §"What this does not optimize" is the list, and none
of it changes here: no node reordering, no transient splitting, no aliasing of
caller buffers, no aliasing across graphs, no locality. Each is a stated ceiling
rather than an omission, and the cross-graph one is the open question
[007](007-tensor-layer.md) prices at a gigabyte per session.
