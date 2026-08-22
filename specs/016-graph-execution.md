---
title: "Edge inference, barrier planning, and flat dispatch"
status: drafted
layer: device
depends_on:
  - 002-compute-model.md
  - 012-kernel-pipeline.md
  - 015-graph-recording.md
---

# Edge inference, barriers, and flat dispatch

The second of [009](009-sequencing.md)'s three M3 children, and the one that
carries M3's end-to-end criterion. [015](015-graph-recording.md) built a graph
that runs correctly and serially; this child computes what actually has to be
ordered, and puts a compiled kernel in it.

## 1. What it adds

- **Edge inference**: the algorithm of [003](003-command-graph.md) §"The
  algorithm", with hazard classification and sub-range comparison.
- **Reachability**, as the relation the DAG is queried through, and the
  determinism requirement on both.
- **Barrier planning**: the per-resource state machine, batching, and the
  one-line backend mapping.
- **The flat dispatch node**: a compiled pipeline from
  [012](012-kernel-pipeline.md) recorded into a graph, with its bindings, its
  uniforms, and its workgroup count, plus CPU lowering for it.
- **Validation rows V8–V11, V17, V22**, which are the dispatch rows plus the
  acyclicity assertion.
- **Run-time counters**, [003](003-command-graph.md) §"Run-time counters".

## 2. Why edges are inferred and not declared

A caller who wrote edges by hand would be writing the thing this design exists
to remove, and would get it wrong in the direction that does not fail: a missing
edge is a race, and a race on a CPU backend that happens to serialize is
invisible until a GPU backend runs the two arms at once. Declared accesses are
the input a caller can be trusted with, because getting one wrong makes their
own node wrong rather than a distant one.

The inference is the standard three hazards over declared accesses, with one
non-standard requirement:

| Later access | Earlier access | Hazard | Edge |
| --- | --- | --- | --- |
| read | write | RAW | yes |
| write | read | WAR | yes |
| write | write | WAW | yes |
| read | read | none | no |

### Sub-ranges, and why whole-resource comparison is not acceptable

Two nodes writing disjoint halves of one buffer have no hazard, and a planner
that compares whole resources says they do. That is not a missed optimization,
it is the optimization: a tiled workload is a stream of nodes touching disjoint
slices of one allocation, and serializing them collapses the parallelism the
graph existed to express.

So a conflict is an overlap of byte ranges, not of resource ids:

```
conflict(a, b)  ⟺  a.resource == b.resource
                ∧  [a.offset, a.offset+a.size) ∩ [b.offset, b.offset+b.size) ≠ ∅
                ∧  ¬(a.read ∧ b.read)
```

Whole-resource comparison is the conservative fallback and it is what a slot
gets, since a slot's eventual resource is unknown at build. That asymmetry is
why V24 exists at all.

### Determinism

[003](003-command-graph.md) requires the inferred edge set and the emitted
barrier list to be identical across runs of one build. Not for aesthetics: the
plan golden and the differential fuzz of [017](017-graph-aliasing.md) both
compare plans, and a plan that depends on map iteration order produces a test
that fails one run in ten and gets marked flaky rather than investigated.

Every traversal is therefore over an explicitly ordered structure — node id
ascending, and within a node, declaration order. This is the same rule the
kernel front end arrived at the hard way when a recursion diagnostic named a
different member of the cycle on CI than locally
([013](013-kernel-subset.md) §5), so it is stated up front here rather than
rediscovered.

## 3. Barriers: the state machine, and why batching collapses most of them

A barrier is emitted from a state transition per resource, not per edge. Each
resource carries a current (stage, access) state; a node's declared access asks
for a state; a transition that requires visibility emits a barrier before that
node.

Two properties make the count much lower than the edge count:

- **A barrier is queue-wide.** One emitted before node *j* for a hazard on
  resource *R* also makes visible every write before it on every other resource.
  So a barrier emitted for one hazard covers every other hazard whose source
  precedes it and whose destination follows it.
- **Barriers merge at a node.** Several transitions required before one node
  become one barrier with the union of their source and destination masks.

On [003](003-command-graph.md)'s worked graph that is nine data hazards emitted
as six barriers plus one submission-boundary acquire:

```
n0   n1   n2   n3   n4   n5   n6   n7
 |    |    |         |    |    |    |
 B    B    B         B    B    B    B      7 barriers
                ^^
                no barrier between n2 and n3: the two GEMMs overlap
```

The pair with no barrier between them is the assertion worth making. A planner
that emitted seven barriers in the wrong places would match the count and lose
the point, so the test asserts the *positions*, and asserts that no barrier sits
between `n2` and `n3`.

The aliasing handovers in [003](003-command-graph.md)'s table are
[017](017-graph-aliasing.md)'s; here the graph has none, so this child's
assertion on the same graph is six barriers plus the boundary, and
[017](017-graph-aliasing.md) asserts that adding aliasing does not add a
barrier — both handovers ride on a barrier the data flow required anyway.

## 4. The flat dispatch node

A dispatch payload carries a compiled pipeline, a binding set, uniform values,
and a direct workgroup count. Flat means what [012](012-kernel-pipeline.md)
means by it: no shared memory, no barriers, no subgroup operations. A pipeline
whose kernel needs any of those is rejected at record time naming M4, using the
same stage information the intrinsic table already carries, rather than failing
during execution.

CPU lowering executes the generated entry point over the workgroup grid. It is
the flat executor [012](012-kernel-pipeline.md) already built, reached through a
graph instead of directly, which is what makes the E2E below a test of the graph
rather than a second test of the kernel.

## 5. Testing

- **E2E, and M3's**: a public recorder uploads to a buffer, dispatches a flat
  `Add` kernel over it, and reads back; the graph is retained, an input rebound,
  and replayed, producing the second input's result with no rebuild.
- The worked graph of [003](003-command-graph.md) is recorded and its inferred
  edge set asserted node by node against the spec's table, including the
  diamond.
- Its barrier list is asserted by position, including the absence of one between
  `n2` and `n3`.
- Two nodes writing disjoint halves of one buffer produce no edge; the same two
  overlapping by one byte produce one. This is the sub-range test and it is
  written as a pair, because a planner that compares whole resources passes the
  second alone.
- Inference and barrier planning are asserted deterministic by building one
  graph twenty times and comparing plans, which is the shape that caught the
  front end's map-order bug.
- V8–V11 and V17 have focused negative tests; V22 is asserted by a test that
  reaches the internal assertion through a deliberately corrupted edge set.
- A fuzz target builds random dispatch graphs and asserts the inferred edge set
  is acyclic and that record order is a topological order of it — the property
  [015](015-graph-recording.md) §3's claim depends on.
- A benchmark reports inference and barrier planning cost against node count,
  since [003](003-command-graph.md) §"Cost" makes a claim about it.

## 6. What it does not build

- **No aliasing**: every transient still gets its own bytes.
  [017](017-graph-aliasing.md).
- **No indirect dispatch.** V9 exists for it and is written; the payload is not.
- **No cooperative dispatch**: M4.
