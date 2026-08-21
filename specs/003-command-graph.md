---
title: "Command graph: recording, replay, and submission"
status: drafted
layer: device
depends_on:
  - 001-device-resources.md
  - 002-compute-model.md
---

# Command graph

Implements [`000-decisions.md`](000-decisions.md) decision 1. This is the spec the rest
of layer 1 is shaped around, so read it before the others.

## The model

Work is **recorded** into a `Graph`, which is then **submitted** to a queue, any
number of times.

```
Recorder  --build-->  Graph  --submit-->  Queue  -->  Fence
   ^                    |
   |                    +-- rebind inputs between submissions
   +-- records nodes, never executes
```

A `Graph` is immutable once built. Everything that varies between submissions
varies through **bindings**, not through re-recording.

### Why immutable

Immutability is what makes the graph worth having. A built graph can be
validated once, have its memory planned once, have its barriers computed once,
and be lowered to a native replayable object once. If a graph could be edited
after building, every one of those would have to be redone per submission, which
is the cost the model exists to avoid.

## What varies between submissions

Three things, and nothing else:

1. **Buffer contents.** Written through the normal write path.
2. **Bound resources.** A binding slot declared at record time can be pointed at
   a different resource before submission, provided the new resource matches the
   slot's declared type, dtype, and access.
3. **Dispatch and draw counts**, where a node was recorded with a dynamic count.

**A per-step *address* is not on this list, and that is a real gap.** A KV cache
write offset, for instance, is none of the three: it is neither the contents of a
buffer by nature, nor a different resource, nor a count. The tensor layer routes
it through buffer contents, passing the offset as a single `u32` a kernel reads,
rather than rebinding a view, because rebinding costs a binding update per layer
per step. That works, but the list above reads as exhaustive and this case has
already been hit twice, so it is recorded here rather than rediscovered.

Anything else, a different pipeline, a different node order, a different set of
nodes, is a different graph. Building a graph is cheap enough to build several,
and callers with genuinely dynamic structure are expected to cache graphs keyed
by shape.

This restriction is deliberate. It is the line that keeps a graph
plan-once-replay-many rather than a data structure that happens to be replayed.

## Recording

A recorder accumulates **nodes**. A node is one dispatch, one render pass, one
transfer, or one barrier. Nodes form a DAG: each declares the resources it reads
and writes, and the edges are inferred from that, not stated by the caller.

Inferring edges from declared access, rather than trusting a caller-supplied
order, is what lets the builder compute barriers correctly and reorder or
overlap independent work. It also means a missing dependency is a validation
error rather than a race.

### Recording is not thread-safe, and a graph has one submission in flight

One recorder belongs to one goroutine.

A built `Graph` is immutable, but **it may have only one submission in flight at
a time**. This is narrower than immutability suggests, and the reason is memory
planning: a graph's transients are aliased into a single pool, so two overlapping
submissions would write each other's intermediates. Rebindable slots have the
same problem from the other direction, since a rebind between two in-flight
submissions is a race on which one sees it.

Both the graphics and tensor layers hit this independently, so the rule is set
here rather than worked around in each. To run the same work concurrently, build
a graph per concurrent user: they share pipelines and caller-owned buffers, and
only the transient pool is duplicated. A caller that needs ordering rather than
concurrency waits on the fence.

## Validation happens at build time

Build is where every check lives: type and dtype agreement at every binding,
workgroup sizes within device limits, resource sizes sufficient for declared
access, capability requirements met by the target device, no read-write hazard
the barrier pass cannot resolve, no cycle.

This is the tradeoff [`000-decisions.md`](000-decisions.md) decision 1 accepts: errors
arrive at build rather than at the call that caused them. To keep that
diagnosable, a build error must name the node, the binding slot, and the
originating call site. A recorder captures enough source context per node to do
that. An error that says only "type mismatch" is a defect in this design.

## Memory planning

The builder computes each transient's live range across the graph and assigns
transients into a pool, aliasing those whose ranges do not overlap.

This is why the tensor layer can run a model without allocating per operation.
It is also why `Graph` exposes its memory requirement before submission: a
caller sizing a KV cache, or deciding how many layers fit, needs the number
ahead of time. Ollama's `Reserve` and `BackendMemory` exist for this and the
requirement is the same here.

Buffers the caller created are never aliased. Only transients the builder owns
participate.

## Barriers are computed, not written

The caller never writes a barrier for correctness. The builder inserts what the
access declarations require, and the backend lowers that to its native
primitive. Explicit barriers exist in the compute model spec for
*intra*-kernel synchronisation, which is a different thing entirely.

Automatic barriers are worth the machinery because manual ones are the single
most common source of nondeterministic GPU bugs, and because a correct manual
barrier still needs different lowering per backend.

## Submission and completion

Submission is asynchronous and returns a fence. A fence can be waited on, polled,
or used as a dependency for a later submission. Nothing in the API blocks
implicitly.

A single-shot convenience records a one-use graph and submits it. It exists for
readability in simple cases and carries the full cost of building a graph, so it
is documented as inappropriate in a hot loop.

## Open questions

- **Conditional and iterative execution.** A model with early exit, or a loop
  whose trip count depends on device data, cannot be a pure DAG. Options are
  indirect dispatch with a device-written count, sub-graphs invoked by the host
  per iteration, or leaving it out of v0. Not decided.
- **Cross-graph aliasing.** Two graphs alive at once each plan their own pool.
  Whether they can share is unresolved and matters for memory-constrained
  inference.
- ~~Graph lowering per backend, and how much replay actually saves.~~
  **Resolved in [006](006-backends.md).** Replay saves three separable things,
  and only one of them, encode-once, needs a native replayable object, which is
  two backends of six. Plan-once, validating, memory planning, and barrier
  computation, is most of the value and every backend gets it, including those
  that replay a recorded list in software. So the model is justified even where
  lowering is not native. Two corrections to the framing above came out of it:
  Vulkan's replayable object is a reusable *primary* command buffer rather than a
  secondary, and a D3D12 bundle cannot contain resource barriers, so any graph
  with a dependency cannot be one bundle.

## Testing

- A graph submitted twice with identical bindings produces identical results.
- Rebinding an input between submissions changes the result, and rebinding a
  mismatched resource is rejected.
- Every validation failure names its node and binding slot.
- Aliased transients that should not overlap do not: a graph whose transients
  have disjoint live ranges uses less memory than the sum of its transients, and
  a graph whose live ranges overlap does not alias them.
- Barrier correctness is checked by a read-after-write chain whose result is
  wrong if the barrier is missing, run enough times to catch a race, and on
  every backend including CPU.
