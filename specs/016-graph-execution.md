---
title: "Edge inference, barrier planning, and flat dispatch"
status: implemented
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
- **Validation rows V8–V10, V17, V22** — V11 is stated and unreachable, see
  below — which are the dispatch rows plus the
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
- V8, V9, V10 and V17 have focused negative tests; V22 is asserted by a test that
  reaches the internal assertion through a deliberately corrupted edge set.
- A fuzz target builds random dispatch graphs and asserts the inferred edge set
  is acyclic and that record order is a topological order of it — the property
  [015](015-graph-recording.md) §3's claim depends on.
- A benchmark reports inference and barrier planning cost against node count,
  since [003](003-command-graph.md) §"Cost" makes a claim about it.

## 6. What it added, beyond what §1 listed

**`Graph.Hazards` and `Graph.Edges`** alongside `Graph.Barriers`. The gap
between the first two numbers is what batching bought, and a caller asking why a
graph does not overlap wants both rather than either. `Edges` exposes the
inferred DAG because a plan is the thing worth asserting on: a test comparing
results cannot tell a graph that overlapped correctly from one that serialized
and got the same answer.

**`ComputePipeline` gained a real body**, with `Kernel()` exposing the record it
was created from, and `Device.Missing` is implemented rather than panicking.
`requirementsOf` returns a zero capability set, and that is a fact about the v0
subset rather than a gap: every value in `Capability` is subgroups, atomics,
native narrow arithmetic, or integer dot product, and a flat kernel can imply
none of them. When [009](009-sequencing.md)'s M4 adds the analysis that infers
them, that function reads a field and `Missing` does not change.

**`internal/kernel.Dispatch`** is the workgroup loop, shared by the CPU backend
and by the bring-up path that runs a generated kernel without a device. Two
copies would be two definitions of what a workgroup id is, and the second to be
updated is the one that quietly stops agreeing with
[002](002-compute-model.md).

**`driver.Dispatch`** carries a kernel record, a workgroup count, binding
operands, and uniforms across the seam. Bindings are positional because the
kernel indexes its arguments by layout index, so the position *is* the contract
and a reordering silently swaps two buffers.

**A dispatch's accesses come from the kernel's binding layout, never from the
caller.** The mode was inferred from the kernel body by the compiler, and
letting a caller restate it would let them under-declare, which is exactly how a
missing dependency becomes a race. §1 implied this by saying accesses are the
input; it is worth stating as a rule.

**The `Add` corpus kernel** takes two inputs rather than one, for a reason about
the graph rather than the compiler: with a single input, a dispatch that read
the resource a rebind replaced is indistinguishable from one that read the new
one.

## 7. Outcome — complete 2026-08-23

Everything in §1 is built and §5's cases pass, including M3's end-to-end
criterion and [003](003-command-graph.md)'s worked graph asserted edge for edge
and barrier for barrier: nine hazards, seven barriers, and no edge between the
two GEMMs.

**Two properties were confirmed by reinstating the bug rather than by passing.**
Whole-resource comparison was substituted for the sub-range test, and it fails
exactly on the GEMM pair the spec named. Queue-wide batching was removed, and
the worked graph goes to eight barriers with one appearing before `n3`. A test
that only passes proves less than one seen to fail for the stated reason.

**Bindings reach a kernel as slices aliasing device memory**, not copies: a
kernel writing its output must write where the graph said it would, and copying
back afterwards would be a second definition of what a binding means. The
reinterpretation refuses a byte range that is not a whole number of elements
rather than truncating, because truncation hides a range computed with the wrong
element size and the kernel then runs over one element fewer than the caller
believes it bound.

**Planning is linear in node count**, which is what
[003](003-command-graph.md)'s cost section claims. An earlier reading suggesting
otherwise was a hundred-iteration benchmark measuring mostly noise, and it is
recorded here because the wrong conclusion was nearly acted on. No wall-clock
assertion guards it: a timing threshold is the shape that produced four red
Windows runs on a coarse clock, so the benchmark reports the number and no gate
fails on someone else's machine.

**One thing this milestone did not have to do.** [004](004-kernel-authoring.md)'s
IR node set did not grow, no new payload kind was needed beyond `OpDispatch`,
and the record-order plan of [015](015-graph-recording.md) needed no change to
be replaced — the barrier list went from "every node" to "what the accesses
need" without the surrounding structure moving, which is the evidence that the
cut between the two children was in the right place.

## 8. What it does not build

- **No aliasing**: every transient still gets its own bytes.
  [017](017-graph-aliasing.md).
- **No indirect dispatch.** V9 exists for it and is written; the payload is not.
- **No cooperative dispatch**: M4.

## Correction: V11 is stated and cannot fire — 2026-08-24

Appended rather than edited in, per [009](009-sequencing.md)'s rule.

§1 and §5 counted V11 (a kernel's shared-memory request against the device
budget) among the rows with focused negative tests. It has none, and cannot: the
test that names V11 exercises `MaxWorkgroupSize` and `MaxWorkgroupInvocations`,
which is V10.

`requirementsOf` never sets `Requirements.SharedBytes`, so `Device.Missing`
always compares 0 against the budget. That is not an oversight to patch in a
line: **the kernel record does not carry the number.** `Kernel.SharedSizes` is
each shared array's element *count*, in signature order, with no element size
beside it — the scheduler needs counts to size its shadow bits, and nothing
needed bytes.

So computing the request means adding a field the generator emits, which changes
the shape of `accel.Kernel` — the type
[036](036-documentation.md) §4 records as an open freeze question, with twelve
exported mutable fields and no constructor. Adding a thirteenth before that is
settled is the wrong order.

**V11 stands as stated and unenforced**, and this note is what stops a reader
inferring from §5 that it is covered. It becomes reachable with whichever change
gives the record a byte count; the spec that makes cooperative kernels
recordable in a graph is the natural owner, since no graph kernel declares shared
memory today either.
