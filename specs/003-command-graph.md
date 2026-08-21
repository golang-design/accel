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

Item 3 read "dispatch counts" in the first draft of this spec.
[005](005-graphics.md) proposed widening it to draws, and the amendment is
accepted here: an indirect draw's vertex count, instance count, and first-index
offsets vary exactly as a dispatch's workgroup count does, through a
device-written argument buffer, with the graph structure unchanged.

**A per-step *address* is not on this list, and that is a real gap.** A KV cache
write offset, for instance, is none of the three: it is neither the contents of a
buffer by nature, nor a different resource, nor a count. The tensor layer routes
it through buffer contents, passing the offset as a single `u32` a kernel reads,
rather than rebinding a view, because rebinding costs a binding update per layer
per step. That works, but the list above reads as exhaustive and this case has
already been hit twice, so it is recorded here rather than rediscovered.

It has a second consequence, made precise under
[edge inference](#edge-inference): a node whose addressing is computed inside the
kernel from buffer contents cannot declare a tight sub-range, so it declares the
whole slab it might touch and takes edges a tighter declaration would not need.
The address gap is paid in lost parallelism, not in correctness.

Anything else, a different pipeline, a different node order, a different set of
nodes, is a different graph. Building a graph is cheap enough to build several,
and callers with genuinely dynamic structure are expected to cache graphs keyed
by shape.

This restriction is deliberate. It is the line that keeps a graph
plan-once-replay-many rather than a data structure that happens to be replayed.

## Recording

A recorder accumulates **nodes**. A node is one dispatch, one render pass, or one
transfer. Nodes form a DAG: each declares the resources it reads and writes, and
the edges are inferred from that, not stated by the caller.

Inferring edges from declared access, rather than trusting a caller-supplied
order, is what lets the builder compute barriers correctly and reorder or
overlap independent work. It also means a missing dependency is a validation
error rather than a race.

A barrier is **not** a node the caller can record. Barriers appear in the lowered
plan as builder-synthesized entries between nodes, and the recorder has no entry
point that produces one. See [Barrier insertion](#barrier-insertion).

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

This rule also settles [005](005-graphics.md)'s open question about rebindable
slots versus concurrent submission. The answer is the second of the two options
005 lists: a graph with rebindable slots is single-submitter, and so is every
other graph, because transient aliasing has the same problem for graphs with no
rebindable slots at all. Snapshotting bindings at submit was the alternative, and
it is rejected because it fixes only half the race and costs a per-submission
copy of the binding set.

---

## The graph IR

Everything below operates on this. The types are sketches of the internal
representation, not public API: the public surface is [`graph.go`](../graph.go).
They are written out because the algorithms are unreadable without knowing
exactly what a node carries.

### Node

```go
type NodeKind uint8

const (
    NodeDispatch         NodeKind = iota // compute dispatch, count recorded
    NodeDispatchIndirect                 // compute dispatch, count read from a buffer
    NodeRenderPass                       // one render pass and all its draws (005)
    NodeCopyBuffer                       // device to device, buffer to buffer
    NodeCopyTextureToBuffer              // device to device, texture to buffer
    NodeCopyBufferToTexture              // device to device, buffer to texture
    NodeHostWrite                        // host to device, staged through an Upload pool
    NodeBarrier                          // builder-synthesized only, never recorded
)

type node struct {
    id      NodeID
    kind    NodeKind
    label   string       // caller label, or the pipeline's, for diagnostics
    site    callSite     // file, line, column of the recording call
    access  []accessDecl // canonical: sorted by (resource, offset, mode)
    payload nodePayload  // one of the structs below, by kind
}

// callSite is captured with runtime.Caller at record time. It is the whole
// reason a build error can name the call that caused it, which
// 000-decisions.md decision 1 requires as the price of deferred validation.
type callSite struct {
    file string
    line int32
    col  int32 // 0 where unavailable; runtime.Caller gives no column
}
```

A call site per node costs one `runtime.Caller` per recorded node: for a 3000
node model graph, 3000 shallow stack walks, at build only, never per submission.
That is the right side of the trade, because recording happens once.

### Payloads

```go
type dispatchPayload struct {
    pipeline *ComputePipeline
    bindings []bindingRef   // slot index to resource, or to a rebindable slot
    count    WorkgroupCount // recorded count, for NodeDispatch
}

type dispatchIndirectPayload struct {
    pipeline *ComputePipeline
    bindings []bindingRef
    args     resourceRef    // the buffer holding [x, y, z] as u32
    maxCount WorkgroupCount // build-time upper bound, see below
}

type renderPassPayload struct {
    colour   []attachment // load and store actions per 005
    depth    *attachment
    area     Rect         // must be within every attachment extent
    draws    []draw       // recorded order, never reordered by the builder
    maxDraws int          // for indirect-count draws, per 005
}

type attachment struct {
    tex        resourceRef // one mip, one layer, one aspect
    load       LoadAction  // Clear, Load, DontCare
    store      StoreAction // Store, DontCare
    clearValue ClearValue  // explicit, no zero-value special case (005)
}

type copyBufferPayload struct{ dst, src resourceRef }

type copyTexBufPayload struct {
    buf        resourceRef // buffer side
    tex        resourceRef // texture side, one mip and layer range
    rowStride  int         // bytes, backend-aligned, computed not declared
    rowsPerImg int
    origin     Origin3D
    extent     Extent
}

type hostWritePayload struct {
    dst resourceRef
    src []byte // owned by the graph: copied out of the caller's slice at record
}
```

Notes on the payloads, each of which is a decision:

- **`maxCount` on an indirect dispatch.** [002](002-compute-model.md) leaves
  indirect dispatch open because a device-written count sits awkwardly with an
  immutable graph. The resolution mirrors [005](005-graphics.md)'s for indirect
  draw counts: the node records a build-time maximum, the device supplies the
  actual count, and the builder validates the maximum against device limits.
  Without it there is nothing to validate at build and nothing to size transients
  against, and exceeding the indirect workgroup count limit is undefined
  behaviour on Vulkan rather than a clean error. The honest gap: the clamp is not
  free to enforce on device. Strict mode inserts a one-workgroup clamp dispatch
  that mins the count buffer against the maximum; release mode makes the maximum
  a documented caller obligation.
- **`rowStride` is computed, never declared.** Backends impose row alignment on
  texture-buffer copies (256 bytes on D3D12, backend-specific elsewhere) and
  bytes per pixel comes from the format
  ([`conventions.md`](../docs/conventions.md)). A caller who computes stride gets
  it wrong on one backend, so the caller does not compute it.
- **A recorded host write copies at record, and the graph owns the bytes.** The
  alternative, holding the caller's slice until submit, makes an immutable graph
  mutable through a back door: the bytes a submission writes would be whatever the
  slice held at that moment, and there is no submission the caller could point at
  to say when it stopped being safe to touch. It also contradicts what
  [001](001-device-resources.md) §8.2 teaches, that a slice is reusable the moment
  a write call returns. So the copy happens at `CopyToBuffer`, and the payload's
  `src` is the graph's.

  The cost is that a graph's build-time footprint includes every byte recorded
  this way, and a graph submitted many times rewrites the same bytes every time.
  Both make this the wrong entry point for bulk upload and for anything that
  varies per submission. Bulk upload is 001 §8.4's shape, a staging buffer plus
  `CopyBuffer` nodes; per-submission variance is a mapped `Upload` buffer written
  between submissions, which is [007](007-tensor-layer.md)'s parameter buffer.
  This node exists for small constants baked into a graph, and its doc comment
  says so.
- **A draw is not a node, and present is not a node.**
  [005](005-graphics.md) settles both. The render pass is the finest granularity
  at which synchronisation is expressible on tile-based hardware, so it is the
  finest the graph models; present is a queue operation taking a fence, and what
  the graph carries instead is a final-state annotation on the swapchain slot
  (see [external state](#external-state)).

### Resource access declaration

This is the substrate. Every algorithm below reads only this.

```go
type accessDecl struct {
    res   resourceRef
    mode  AccessMode // bitmask, what the node does to the range
    stage StageMask  // bitmask, which pipeline stages do it
    slot  int        // binding slot index, or slotNone for attachments and copies
}

type resourceRef struct {
    id   resourceID   // stable within one graph: see below
    kind resourceKind // resourceBuffer or resourceTexture

    // Buffer sub-range, in bytes, half-open [off, off+size). Whole buffer is
    // [0, byteSize). Bytes, not elements: a view can reinterpret dtype (001),
    // so elements are not a common unit across two views of one buffer.
    off, size uint64

    // Texture sub-resources. Half-open ranges over mip level and array layer.
    // There is deliberately no region within a mip; see the cost note below.
    baseMip, mipCount     uint16
    baseLayer, layerCount uint16
    aspect                Aspect // AspectColour, AspectDepth, AspectStencil
}

// resourceID names the thing hazards are tracked against. Three cases:
//   - a caller-created Buffer or Texture, identified by its object
//   - a transient the builder owns
//   - a rebindable slot, identified by the slot, NOT by whatever is bound to it
// The third is why an overlap check has to happen again at submit, see V21.
type resourceID uint32
```

```go
type AccessMode uint16

const (
    ReadStorage    AccessMode = 1 << iota // storage buffer or storage texture read
    ReadUniform                           // uniform buffer read
    ReadSampled                           // sampled texture read
    ReadIndirect                          // indirect argument fetch
    ReadVertex                            // vertex or index buffer fetch
    ReadCopySrc                           // transfer source
    ReadAttachment                        // LoadAction Load, or depth test without write
    WriteStorage                          // storage buffer or storage texture write
    WriteAttachment                       // colour or depth attachment write
    WriteCopyDst                          // transfer destination
    AtomicRMW                             // atomic read-modify-write, both at once
)

const anyRead = ReadStorage | ReadUniform | ReadSampled | ReadIndirect |
    ReadVertex | ReadCopySrc | ReadAttachment | AtomicRMW

const anyWrite = WriteStorage | WriteAttachment | WriteCopyDst | AtomicRMW

type StageMask uint16

const (
    StageHost StageMask = 1 << iota
    StageTransfer
    StageIndirectFetch // reading the argument buffer: a distinct stage on Vulkan and D3D12
    StageVertexInput
    StageVertex
    StageFragment
    StageEarlyDepth
    StageLateDepth
    StageColourOutput
    StageCompute
)
```

**`AtomicRMW` is one bit, not read plus write.** An atomic is both, but a run of
atomics on the same range from consecutive nodes needs no barrier between them on
any target backend, and classifying it as a plain write would insert one per
pair. Two nodes that both only `AtomicRMW` the same range get no edge. A node
that reads or writes non-atomically after an `AtomicRMW` node does.

**Texture layout is derived, never declared.** The layout a texture must be in
(`ShaderRead`, `ColourAttachment`, `DepthAttachment`, `DepthReadOnly`, `CopySrc`,
`CopyDst`, `General`, `Present`) is a function of `(mode, stage)`, computed by the
builder. Callers get layouts wrong, the information is fully redundant with the
access declaration, and three of the six backends have no layouts at all. The
mapping is in the [barrier](#barrier-insertion) section.

**Declared access is an upper bound on actual access. Declaring less is
undefined behaviour.** This is the one caller obligation the model cannot check
statically, and it is the counterpart of the per-step address gap: a kernel
computing offsets from a `u32` it read must declare the whole range those offsets
can reach. The debug-mode CPU backend bounds-checks every kernel buffer access
against the declaring node's ranges and fails on a violation, which is what makes
the obligation enforceable at all: on the oracle, before it is a race on a GPU.

### External state

A graph has an edge with the world at both ends.

- **External reads**: a resource read by some node before any node in this graph
  writes it. Its contents come from outside the submission.
- **External writes**: a resource whose last access in the graph is a write. Its
  contents outlive the submission.
- **Final state**: an annotation on external writes whose next consumer is not a
  graph node. The only case at v0 is the swapchain image, which the builder knows
  is presentable and leaves in `Present` layout, folded into the writing pass's
  store action per [005](005-graphics.md).

These two sets are what make the cross-submission rule in
[Submission](#submission-ordering-and-fences) implementable: the resources a
submission must acquire and release are exactly these, computed at build.

---

## Edge inference

### Acyclicity is structural, not checked

Edges are inferred only from a node to a later-recorded node, because inference
walks nodes in record order and links each access against state left by earlier
nodes. So **record order is always a topological order of the inferred DAG, and a
cycle is impossible by construction.**

The first draft listed "no cycle" among the build checks. It is kept as an
internal assertion against a builder defect, not advertised as a caller-facing
validation that can never fire. It becomes a real check the moment the recorder
gains caller-stated edges or sub-graphs, neither of which exists at v0.

### The algorithm

Per resource the builder keeps a small state record. Nodes are visited in record
order. For each node, for each access declaration, the overlapping entries of that
resource's state are found and classified.

```go
// Per resource, updated as nodes are visited in record order.
type resState struct {
    writers []span // {off, size, node, mode, stage}: writes not yet fully overwritten
    readers []span // reads since the last write that covered their range
}

func inferEdges(nodes []node) [][]NodeID {
    st := map[resourceID]*resState{}
    edges := make([][]NodeID, len(nodes))
    for _, n := range nodes {
        for _, a := range n.access {
            s := st[a.res.id]
            if a.mode&anyRead != 0 {
                for _, w := range s.writers { // read after write
                    if overlaps(w, a.res) && !bothAtomic(w, a) {
                        addEdge(edges, w.node, n.id, hazardRAW)
                    }
                }
            }
            if a.mode&anyWrite != 0 {
                for _, w := range s.writers { // write after write
                    if overlaps(w, a.res) && !bothAtomic(w, a) {
                        addEdge(edges, w.node, n.id, hazardWAW)
                    }
                }
                for _, r := range s.readers { // write after read
                    if overlaps(r, a.res) && !bothAtomic(r, a) {
                        addEdge(edges, r.node, n.id, hazardWAR)
                    }
                }
            }
        }
        for _, a := range n.access {
            commit(st, n, a) // update after classifying, not during
        }
    }
    return edges
}
```

`commit` records the access into the resource's state: a read appends to
`readers`; a write appends to `writers`, drops any `writers` entry its range
fully covers, and drops the `readers` entries its range fully covers (those are
ordered before it by the WAR edge just added, so no later node needs them).
Trimming on full cover rather than partial cover is deliberate: partial-cover
trimming needs interval splitting, and the ranges that occur in practice are
either identical or disjoint.

The two-phase structure, classify all of a node's accesses and then commit all of
them, matters for a node that both reads and writes one range: without it the
node would hazard against itself.

### Hazard classification

| Hazard | Pattern | Edge | Why the edge exists |
| --- | --- | --- | --- |
| RAW | earlier writes, later reads | yes | The read must see the write. Needs execution ordering **and** memory visibility. |
| WAW | earlier writes, later writes | yes | Otherwise the earlier write can land last. Needs execution ordering and visibility. |
| WAR | earlier reads, later writes | yes | Otherwise the read observes the new value. Needs execution ordering only, no cache flush. |
| RAR | earlier reads, later reads | no | No hazard. Two readers of one range may overlap freely, and this is the common case in a model where every layer reads the same weights. |
| atomic to atomic | both `AtomicRMW` | no | The device orders them. See the note above. |

The RAR exclusion is worth stating as a number: a read-only resource, one no node
writes, generates **zero** edges no matter how many nodes read it. Weight buffers
are read-only and are most of the resources and most of the accesses in a
transformer graph, so this is the largest single cost saving in inference, free.

### Sub-ranges, and why whole-resource comparison is not acceptable

`overlaps` compares half-open intervals:

```go
func overlaps(a, b resourceRef) bool {
    if a.kind == resourceBuffer {
        return a.off < b.off+b.size && b.off < a.off+a.size
    }
    return a.aspect&b.aspect != 0 &&
        a.baseMip < b.baseMip+b.mipCount && b.baseMip < a.baseMip+a.mipCount &&
        a.baseLayer < b.baseLayer+b.layerCount && b.baseLayer < a.baseLayer+a.layerCount
}
```

**Exact interval overlap on buffers, exact subresource-rectangle overlap on
textures, and nothing finer.**

The alternative, treating every access as covering the whole resource, is simpler
and wrong for this design. Two cases make it concrete:

- The tensor layer packs Q, K, and V into one transient and has three kernels
  write disjoint thirds. Whole-resource comparison gives WAW edges among all
  three, serializing exactly the work that should overlap.
- A KV cache is one buffer per layer. Attention reads rows `[0, L)` and the cache
  append writes row `L`. Whole-resource comparison gives a WAR edge and forces a
  barrier per layer per token, which is the hot path of the workload this library
  exists for.

**What is deliberately not modelled**: a region within one texture mip. A node
declaring the top-left quadrant and one declaring the bottom-right quadrant of
the same mip get an edge they do not need. This is conservative in the safe
direction, it costs a barrier in a case nobody has hit, and modelling it would
put rectangle-set intersection in the hazard loop for a speculative benefit.
Revisit when a real workload produces it.

### Cost

Let `N` be the node count, `k` the mean accesses per node, and for resource `r`,
`A_r` the number of live spans in its state.

| Step | Cost | Notes |
| --- | --- | --- |
| Per access classification | `O(A_r)` | Linear scan of one resource's spans. |
| Whole graph | `O(N * k * Ā)` | `Ā` is the mean live span count. |
| Read-only resources | `O(1)` per access | Never enter the writer loop, and `readers` is not consulted for a read. |

`Ā` stays small because a write trims the spans it covers, so the common shapes,
a transient written once and read twice, and a weight read many times and never
written, keep `A_r` at one or zero. The pathological shape is a resource with
many disjoint writers and no covering write, the packed QKV transient being the
example, where `A_r` grows to the writer count. That is three, not three thousand.
One fast path is worth building in: an exact-match key of `(id, off, size)`,
which the tensor layer hits constantly because a plan reads the same view of the
same weight in the same slot every layer. If `Ā` does grow, the escape is an
interval tree per resource for `O(log A_r)` lookup; it is not built at v0 because
the linear scan wins below roughly thirty spans and measured shapes are far
below that.

### Determinism of inference

Edge lists are built in record order and sorted by `NodeID`, and no map iteration
order reaches the output. Go randomizes map iteration, and a plan built from a
map walk would differ run to run, making the golden-plan tests in
[Testing](#testing) impossible and a planning regression invisible. The state
table is a map keyed by `resourceID`, but it is only looked up, never ranged over.

---

## Memory planning

The builder computes each transient's live range across the graph and assigns
transients into a pool, aliasing those whose ranges do not overlap.

This is why the tensor layer can run a model without allocating per operation.
It is also why `Graph` exposes its memory requirement before submission: a caller
sizing a KV cache, or deciding how many layers fit, needs the number ahead of
time. Ollama's `Reserve` and `BackendMemory` exist for this and the requirement
is the same here.

Buffers the caller created are never aliased. Only transients the builder owns
participate. The reason is that the builder cannot know a caller buffer's
lifetime: it may be read by another graph, by the host, or by nothing at all
until next frame, and none of that is visible from this graph's declarations.

### Liveness on a DAG: an interference relation, not an interval

A DAG has no single linear order, and that is not a technicality. Take a diamond
where node 1 fans out to nodes 2 and 3, which join at node 4. Suppose transient
`T` is written by 1 and read by 2 and 3, and transient `U` is written by 4. In
record order `T` occupies `[1, 3]` and `U` occupies `[4, ...]`, disjoint, so an
interval-based planner aliases them. That is a bug: node 3 and node 4 are
**unordered**, the backend is free to overlap them, and node 4's write to `U`
lands on top of node 3's read of `T`.

An interval planner is correct only if the executor runs nodes strictly in the
linearization used to compute the intervals, and this design explicitly permits
the executor to overlap independent work. So:

**Two transients may share memory if and only if every node touching one is
ordered, by the inferred DAG, before every node touching the other.**

```go
// reach[i] is the set of nodes reachable from i, transitively, as a bitset.
// Computed once, in reverse record order, which is a reverse topological order:
//     reach[i] = union over successors s of ({s} | reach[s])
func compatible(t, u *transient, reach []bitset) bool {
    return allBefore(t.users, u.users, reach) ||
        allBefore(u.users, t.users, reach)
}

func allBefore(a, b []NodeID, reach []bitset) bool {
    for _, x := range a {
        for _, y := range b {
            if x == y || !reach[x].has(y) {
                return false
            }
        }
    }
    return true
}
```

`x == y` fails the test, and that is the case that matters most: a node reading
`T` and writing `U` in one dispatch, which is every fused elementwise kernel,
keeps `T` and `U` apart.

**Cost.** Reachability is `O(V * E / w)` bit operations for word size `w`, and
`V * V / 8` bytes. For 3000 nodes that is 1.1 MiB of bitsets and a few
milliseconds, at build, once. Compatibility is
`O(|users(T)| * |users(U)|)` bitset probes per pair and `O(n^2)` pairs for `n`
transients; a transient has two or three users and `n` is in the low hundreds, so
this is microseconds.

The alternative to reachability is to force the order: insert an artificial edge
whenever aliasing two transients would otherwise be unsafe, which is what several
frame graph implementations do. It is cheaper to compute and it silently
serializes exactly the independent work the DAG was built to expose. Rejected for
that reason, and the diamond above is why the difference is not theoretical.

### Packing: greedy by size into a single pool

Aliasing is not graph colouring. Colouring assigns transients to a fixed set of
equal slots, and transients have different sizes. The right formulation is
**dynamic storage allocation**: give each transient an offset in one pool such
that transients that interfere do not overlap in bytes. That problem is NP-hard,
so the algorithm is a heuristic and this spec says so rather than implying
optimality.

```
sort transients by size descending, ties broken by first-writer NodeID ascending
for each transient T in that order:
    occupied = []
    for each already-placed U where !compatible(T, U):
        occupied.append([U.offset, U.offset+U.size))
    merge and sort occupied
    T.offset = lowest offset, aligned up to the device's minimum binding
               alignment, whose [offset, offset+T.size) misses every occupied span
poolSize = max over T of (T.offset + T.size)
```

Complexity `O(n^2 log n)` for `n` transients, dominated by the scan of placed
transients per placement. For 200 transients that is tens of thousands of
operations, at build.

Greedy by size descending beats greedy by first use, by live length, and best
fit here because the large transients constrain everything else, so placing them
first avoids the fragmentation a small-first order creates; it is also what
TensorFlow Lite's arena planner uses. The deterministic tie break is not
cosmetic: without it the layout depends on sort stability and the golden-plan
test flaps.

### What this does not optimize

Stated plainly so the ceiling is known:

- **It does not reorder nodes.** Reordering to shorten live ranges is a real win
  in published planners, and it interacts with barrier placement and with
  [005](005-graphics.md)'s rule that draws within a pass keep their order. Not
  attempted at v0.
- **It does not split a transient**, so one live across a long stretch but used
  at two points stays resident throughout, and it does not trade memory for
  recomputation.
- **It does not alias caller buffers**, by the rule above.
- **It does not alias across graphs.** Two graphs alive at once each get a pool.
  This is the cross-graph aliasing open question, and [007](007-tensor-layer.md)
  reports the cost concretely: five prefill buckets at a 200 MiB peak is a
  gigabyte of transients for one session.
- **It ignores locality.** Offsets minimize the pool, not page behaviour.

### What `GraphMemory` means

The three fields of `GraphMemory` are distinct measurements and the gaps between
them are informative, so each is defined exactly:

| Field | Definition |
| --- | --- |
| `UnaliasedBytes` | Sum of every transient's aligned size. What the graph would need with no planning at all. |
| `PeakBytes` | Maximum, over the record-order linearization, of the total size of transients whose record-order interval covers that point. A lower bound achievable only if the executor ran strictly in record order. |
| `TransientBytes` | The pool size the planner actually produced. This is what gets allocated. |

`TransientBytes` minus `PeakBytes` is the price of DAG-safe aliasing plus
fragmentation. The worked example below has a graph where that gap is 4 MiB out
of 16 and is entirely the former. Reporting both is what lets a caller tell a
planner problem from a graph-shape problem.

---

## Barrier insertion

The caller never writes a barrier for correctness. The builder inserts what the
access declarations require, and the backend lowers that to its native
primitive. Explicit barriers exist in the compute model spec for *intra*-kernel
synchronisation, which is a different thing entirely.

Automatic barriers are worth the machinery because manual ones are the single
most common source of nondeterministic GPU bugs, and because a correct manual
barrier still needs different lowering per backend.

### The per-resource state machine

Barrier insertion is a second walk in record order, over the same resources, but
tracking the *current* state of each range rather than pending hazards.

```go
type rangeState struct {
    span                     // off, size for buffers; mip and layer range for textures
    lastWriteStage StageMask // stages that wrote it, empty if clean
    lastWriteMode  AccessMode
    readStages     StageMask // stages that have read since the last write
    readModes      AccessMode
    layout         Layout    // textures only
}
```

For each node, for each access, against each overlapping range state:

| Detected | Emitted | Contents |
| --- | --- | --- |
| RAW | execution barrier plus memory barrier | `src = (lastWriteStage, lastWriteMode)`, `dst = (access.stage, access.mode)`. The availability plus visibility pair. |
| WAW | execution barrier plus memory barrier | Same shape. The earlier write must be made available, or a write-combining cache can land the two out of order. |
| WAR | execution barrier only | No cache operation. A read dirties nothing, so nothing needs flushing; only the ordering matters. Stating this explicitly matters because a memory barrier here is a needless flush and it is the standard mistake. |
| layout mismatch (textures) | image barrier | `oldLayout = state.layout`, `newLayout = layoutFor(access)`. A layout transition is also a memory operation, so it subsumes the RAW or WAW barrier it accompanies. |
| aliasing handover | execution barrier, plus an aliasing barrier where the backend has one | Emitted when a transient's first write reuses bytes a compatible transient last used. See below. |

Layout is derived by this table:

| Access | Stage | Layout |
| --- | --- | --- |
| `ReadSampled` | any shader stage | `ShaderRead` |
| `ReadStorage` or `WriteStorage` | any shader stage | `General` |
| `WriteAttachment` | `StageColourOutput` | `ColourAttachment` |
| `WriteAttachment` | `StageEarlyDepth` or `StageLateDepth` | `DepthAttachment` |
| `ReadAttachment` only, depth | depth stages | `DepthReadOnly` |
| `ReadCopySrc` | `StageTransfer` | `CopySrc` |
| `WriteCopyDst` | `StageTransfer` | `CopyDst` |
| external write with a `Present` final state | end of graph | `Present` |

### Batching, and why it collapses most barriers

All barriers required by one node's accesses are accumulated and issued as **one**
barrier immediately before the node. That is not only an efficiency measure, it
changes the count: a barrier is a queue-wide ordering point, so a barrier emitted
before node `i` also orders every earlier node against every later one for the
stages it names. The state machine exploits this by clearing satisfied pending
hazards when a barrier is emitted, rather than re-emitting per hazard.

The consequence, made concrete in the worked example: eight nodes with nine data
hazards and two aliasing handovers emit six barriers, and two of the six carry
hazards for resources they were not emitted for.

The honest cost: a barrier naming `StageCompute` on both sides is a full compute
pipeline drain, so it stops independent work either side from overlapping even
though the hazard concerns one buffer. Finer primitives exist (Vulkan
range-scoped buffer memory barriers, split barriers via events, D3D12 enhanced
barriers) and none are used at v0. A real performance ceiling, written down
rather than discovered by profiling.

### Aliasing handovers

When transient `U` is placed at bytes transient `T` also occupies, `U`'s first
write must be ordered after `T`'s last use, and some backends need an explicit
aliasing barrier (D3D12 has one by that name; Vulkan needs a barrier whose source
access covers the prior resource's writes). The planner records the handover
pairs it created and the barrier walk emits for them.

These almost never cost an extra barrier: a handover exists only where the
compatibility test held, so every user of `T` is already ordered before every
user of `U`, and a data-flow barrier usually already sits between them. The
worked example has two handovers and emits zero extra barriers for either.

### The render pass constraint

[005](005-graphics.md) forbids barriers inside a render pass, because tile-based
hardware cannot provide one: attachment contents live in tile memory until the
pass ends. The builder honours this structurally rather than by checking for it:

1. A render pass node's access set is the **union** over all its draws. There is
   no per-draw access, so there is no per-draw hazard, so there is nothing to
   emit inside a pass.
2. Every barrier the pass requires is emitted **before** the pass begins.
3. Attachment layout transitions are folded into the pass description itself,
   which is where every backend wants them (Vulkan render pass initial and final
   layouts, Metal load and store actions on the pass descriptor).
4. A pass reading a resource it writes is a build error (V12), not a barrier the
   builder tries to insert. This is why rule 1 does not hide a real hazard.

### Backend mapping, in one line each

Detail lives in [006](006-backends.md) section 4. The shape:

| Backend | Primitive |
| --- | --- |
| Vulkan | `vkCmdPipelineBarrier2` with stage and access masks; image barriers carry the layout transition; attachment transitions ride the render pass. |
| D3D12 | `ResourceBarrier`: transition barriers for state changes, UAV barriers for storage read-after-write, aliasing barriers for handovers. State decay at command list boundaries is why the submission boundary rule below is stated. |
| Metal | Encoder boundaries give ordering between encoders for free; within an encoder, `memoryBarrierWithScope`. Untracked resources need an `MTLFence` between encoders. |
| GLES 3.1 | `glMemoryBarrier` with the bits implied by the destination access. Coarser than every other backend: the bit set names destination access only, so `accel`'s source stage information is discarded here. |
| WebGPU | No explicit barrier exists; the implementation tracks hazards itself. `accel`'s barriers become no-ops, but pass splitting still matters because WebGPU orders at pass granularity. |
| CPU | A barrier is a join over the goroutines running the prior nodes. `go test -race` then reports a missing barrier as a genuine Go data race, per [006](006-backends.md). |

The GLES and WebGPU rows are the reason barriers are computed rather than written
by the caller: a correct hand-written Vulkan barrier carries information that has
no expression in GLES, and a hand-written GLES barrier lacks information Vulkan
needs.

---

## Validation

Build is where every check lives.

This is the tradeoff [`000-decisions.md`](000-decisions.md) decision 1 accepts:
errors arrive at build rather than at the call that caused them. To keep that
diagnosable, a build error must name the node, the binding slot, and the
originating call site. A recorder captures enough source context per node to do
that. An error that says only "type mismatch" is a defect in this design.

### Error format

Errors are collected, not first-only, matching [004](004-kernel-authoring.md)'s
rule for the kernel compiler. A graph with four problems reports four.

```
accel: graph build failed: 3 errors

  model/attn.go:118:0: node 7 "attn.scores" (dispatch, pipeline "softmax_rows"):
      slot 2 "scores": dtype mismatch, slot declares f32 and the bound buffer
      "kv.scores" is f16
  model/attn.go:131:0: node 9 "attn.out" (dispatch, pipeline "gemm_nt"):
      slot 0 "a": buffer "t.qk" is 4194304 bytes and the slot's declared access
      needs 8388608
  render/deferred.go:64:0: node 3 "geometry" (render pass):
      texture "gbuf.normal" is attachment 1 and is also bound for sampling at
      slot 4 of draw 12; a pass may not read a resource it writes (005)
```

Column is `0` where `runtime.Caller` gives no column, which is always at v0. The
field is in the format because the kernel compiler's diagnostics
([004](004-kernel-authoring.md)) do have columns and the two should be parsed by
one tool.

### Error taxonomy

```go
// BuildError is returned by Recorder.Build. It is a collection.
type BuildError struct{ Errs []*NodeError }

func (e *BuildError) Error() string  { /* the format above */ }
func (e *BuildError) Unwrap() []error

// NodeError is one problem with one node.
type NodeError struct {
    Node     NodeID
    Label    string
    Kind     NodeKind
    Slot     int // binding slot, or slotNone
    SlotName string
    Site     callSite
    Cause    error  // one of the sentinels below, matchable with errors.Is
    Detail   string // the human half, carrying the actual numbers
}

func (e *NodeError) Error() string { /* one entry of the format above */ }
func (e *NodeError) Unwrap() error { return e.Cause }

var (
    ErrDTypeMismatch     = errors.New("accel: dtype mismatch")
    ErrKindMismatch      = errors.New("accel: binding kind mismatch")
    ErrAccessMismatch    = errors.New("accel: access mismatch")
    ErrTooSmall          = errors.New("accel: resource too small for declared access")
    ErrUsageMissing      = errors.New("accel: resource lacks a required usage flag")
    ErrLimitExceeded     = errors.New("accel: device limit exceeded")
    ErrCapabilityMissing = errors.New("accel: device lacks a required capability")
    ErrSlotUnbound       = errors.New("accel: binding slot has no resource")
    ErrFeedbackLoop      = errors.New("accel: resource is both read and written by one pass")
    ErrForeignResource   = errors.New("accel: resource belongs to another device")
    ErrClosedResource    = errors.New("accel: resource is closed")
    ErrPoolExhausted     = errors.New("accel: transient pool cannot be allocated")
    ErrGraphCycle        = errors.New("accel: cycle in the inferred graph") // internal assertion
    ErrRebindOverlap     = errors.New("accel: two slots bound to overlapping ranges")
    ErrGraphInFlight     = errors.New("accel: graph already has a submission in flight")
    ErrDeviceLost        = errors.New("accel: device lost")
)
```

`errors.Is(err, accel.ErrDTypeMismatch)` works through `Unwrap() []error`, which
is what a caller writing a fallback path (try f16, fall back to f32) actually
needs. `errors.As` to `*NodeError` gets the node id for a caller that wants to
point at its own structure.

### Every check

| # | Check | Error says | Enforces |
| --- | --- | --- | --- |
| V1 | Every slot in the pipeline's layout has a binding | node, slot index, slot name, "no resource bound" | 002 binding layout |
| V2 | Bound resource kind matches the slot's `BindingKind` | both kinds, slot name | 002 |
| V3 | Bound buffer dtype matches the slot's declared dtype | both dtypes, buffer label | 001 typed buffers |
| V4 | Bound resource access is compatible with the slot's declared `Access` | slot access, resource usage | 002 |
| V5 | Buffer is large enough for the declared range | declared bytes, actual bytes | 001 |
| V6 | Buffer declares the usage the access needs (storage, uniform, indirect, transfer) | the missing usage flag by name | 001, usage is declared up front |
| V7 | Texture declares the usage the access needs and the format supports it | format, usage, access | 001 |
| V8 | Recorded workgroup count is within the per-dimension limit | count, limit, dimension | 002 |
| V9 | An indirect dispatch's `maxCount` is within the same limit | max, limit | 002, this spec |
| V10 | Pipeline workgroup size within max size and max invocations | size, limits | 002 |
| V11 | Shared memory request within the device budget | requested, budget | 002 |
| V12 | No resource is both read and written by one render pass | both uses, attachment index and draw index | 005 feedback loop |
| V13 | Render pipeline attachment formats and count match the pass's | pipeline, node, attachment index | 005 |
| V14 | Render area within every attachment's extent | area, extent, attachment | 005 |
| V15 | Depth clear value within `[0, 1]` | value | 005 |
| V16 | A `Clear` load action carries an explicit value | attachment index | 005, the zero-value depth hazard |
| V17 | Every capability a node needs is present on the device | capability name, device name | 000 decision 6 |
| V18 | Copy extents and offsets are within both resources | extents, sizes | 001 |
| V19 | Every resource belongs to the recorder's device and is open | resource label | 001 lifetime |
| V20 | The planned transient pool fits the device's reported budget | pool size, budget | 001 pools |
| V21 | (at submit, not build) No two slots are bound to overlapping ranges unless both are read-only | both slots, both resources, the overlap | this spec, below |
| V22 | (internal assertion) The inferred edge set is acyclic | node ids on the cycle | builder defect only |
| V23 | No two **statically bound** views at one node overlap unless both are read-only | both slots, both view ranges, the overlap | [001](001-device-resources.md) 6.1, the build-time half of V21 |

**V21 is the one check that cannot happen at build**, and that is a genuine hole
in the validate-once story. Hazards are tracked against `resourceID`, and a
rebindable slot is its own `resourceID` because at record time nobody knows what
will be bound. Bind one buffer to two slots the builder assumed independent and
the inferred edge set is wrong, so the missing barrier is a race. `Rebind` and
`Submit` therefore check pairwise overlap over the bound set, at `O(s^2)` in the
count of **rebindable** slots (single digits in every design seen so far), in
release builds too: a race that reproduces only under load is worth more than the
microsecond.

**V21 covers only the rebindable case.** Two views bound *statically* at record
time are compared at build, where the buffer, offset and length are all known and
the error can carry the recording call site;
[001](001-device-resources.md) §6.1 states that half and this spec states this
one. They are the same rule at two times, and implementing either alone leaves
the other case as a race. The static half is check V23 above; it is
separated from V21 rather than folded into it because the two fire at different
times, carry different diagnostics, and a builder can implement one and believe
it has implemented both.

---

## Submission ordering and fences

Submission is asynchronous and returns a fence. A fence can be waited on, polled,
or used as a dependency for a later submission. Nothing in the API blocks
implicitly.

### Ordering between submissions

| Relationship | Guarantee |
| --- | --- |
| Two submissions to the **same queue** | The second begins no earlier than the first ends, and every write by the first is visible to the second with no caller action. |
| Two submissions to **different queues** | None. They may run in any order or fully overlapped. Ordering comes only from `SubmitAfter`. Which queues exist is reported by `Device.Queues` ([001](001-device-resources.md) §1); at v0 every device reports one, so this row is unexercised. |
| A submission and the **host** | The host sees writes to `Readback` or `Shared` memory only after that submission's fence has signalled and been waited on. |

**Same-queue submissions are fully ordered, and that is a deliberate cost.** The
implementation is a global memory barrier at the head of every submission,
covering the graph's external reads. The alternative, starting submissions in
order but leaving memory hazards to the caller, is what the underlying APIs
provide and is faster, since independent submissions can overlap. It is rejected
because it moves exactly the reasoning this spec exists to remove back into the
caller, and its failure mode is a race that appears under load on one backend.

The cost is a pipeline drain between consecutive submissions on one queue. A
caller wanting overlap has two options inside the model: two queues, where the
device reports two (queue topology is reported per
[001](001-device-resources.md), never assumed), or one graph containing both
halves of the independent work, which the builder overlaps because no edge joins
them.

D3D12 makes the rule load-bearing rather than merely tidy: resource state decays
at command list boundaries for some resource types, so a backend assuming state
carried across submissions would be wrong there specifically. The
head-of-submission barrier makes every backend agree.

### What a fence guarantees

A fence signals when **all** device work of its submission has completed. When
`Wait` returns nil:

- every write the submission made is complete on the device;
- every write to `Readback` or `Shared` memory is visible to the host, including
  any cache invalidation the backend needs (`Wait` performs it, the caller does
  not);
- every resource the submission used may be freed;
- the graph's one-in-flight slot is released.

`Done` reporting true implies the same except host visibility, which `Wait` or
the first read after `Done` establishes. The asymmetry is deliberate: making
`Done` perform an invalidate would make a polling loop expensive.

A fence guarantees **nothing** about partial progress. There is no "node 5 has
finished" observable at any granularity finer than the submission. Adding one
means per-node timestamps or split fences, which is a profiling feature, not a
synchronisation one.

### Timeline underneath, binary on the surface

The public `Fence` is binary: one submission, one signal, no value. Underneath,
each queue keeps a monotonically increasing counter and a fence is a
`(queue, value)` pair, because that is what every backend offers (Vulkan timeline
semaphores, `ID3D12Fence` with a value, `MTLSharedEvent`, a GL sync object, a
counter on CPU).

The timeline stays internal for one reason: a public timeline invites a wait on a
value that will never be signalled, which deadlocks with no diagnosis, and the
binary form cannot express that. The cost is that "wait until three of these five
have finished" needs three specific fences.

`SubmitAfter` lowers to a device-side wait on the given fences' timeline values,
with no host round trip. Cross-queue dependencies are exactly this. Cross-queue
**ownership transfer**, which Vulkan requires when a resource moves between queue
families, is a backend concern and is an open question below.

### Can submission fail

`Submit` returns a `*Fence` and no error, and that signature is kept. Everything
that can go wrong at submit is delivered through the fence:

| Failure | Delivered as |
| --- | --- |
| The graph already has a submission in flight | An already-failed fence wrapping `ErrGraphInFlight`. |
| The V21 overlap check fails | An already-failed fence wrapping `ErrRebindOverlap`. |
| Device lost, at submit or during execution | `ErrDeviceLost` on this fence and on every other outstanding fence for the device. |
| Out of device memory at submit (staging, descriptor pool) | An already-failed fence wrapping the allocator's error. |

An already-failed fence is signalled at construction, so `Wait` returns
immediately, `Done` is true, and `C()` is closed. A caller who ignores the fence
gets no work done and no crash, which is the right failure shape for something
that can only be a programming error or a device loss.

The one-in-flight rule is enforced with an atomic flag on the graph, taken at
`Submit` and released when the fence signals. It is not a mutex: `Submit` must
not block, so the second submitter gets the error rather than the queue.

A single-shot convenience (`Queue.Run`) records a one-use graph, submits it, and
waits. It exists for readability in simple cases and carries the full cost of
building a graph, so it is documented as inappropriate in a hot loop.

---

## Statistics

Two sibling specs promise numbers this one has to produce.
[005](005-graphics.md) says an indirect draw count exceeding its build-time
maximum "is clamped, and the clamp is reported through the node's statistics".
[001](001-device-resources.md) §4.2 defines `CopyStats` so a caller can see a
texture copy's repack cost. Neither named a way to read either back, and a
promise with no retrieval path is a promise that will be quietly dropped.

**The split is plan-time versus run-time, and it is not cosmetic: one is free and
one costs a device round trip.**

### Plan-time facts, free, always available

Whether a texture copy repacks, the row pitch the backend chose, a transient's
pool offset, the inferred edges, and the barrier positions are all decided at
build. They cost nothing to expose because the builder computed them anyway.

```go
// NodeStats reports what the builder decided about one node. Valid as soon as
// Build returns, identical for every submission, and free: these are the plan,
// not a measurement.
type NodeStats struct {
    Node  NodeID
    Kind  NodeKind
    Label string

    Copy *CopyStats // non-nil for copy nodes: see 001 section 4.2

    // Barriers emitted immediately before this node, for the assertions in
    // Testing below and for anyone asking why a graph does not overlap.
    BarriersBefore int
}

func (g *Graph) NodeStats(id NodeID) NodeStats
func (g *Graph) Nodes() []NodeStats
```

This is also what makes 003's own barrier and planner tests assertable on the
plan rather than on results, which several of them already assume.

### Run-time counters, opt-in, because they cost a readback

An indirect dispatch's actual workgroup count, an indirect draw's actual draw
count, and whether either was clamped against its build-time maximum are decided
by the **device**, during execution. Reporting them means the graph writes them
into a buffer and the host reads that buffer back, which is a transfer node, a
barrier, and a `Readback` allocation that a caller who does not want the numbers
should not pay for.

So they are **off by default** and enabled per graph at record time. When
enabled, the builder appends the counters to the graph's own readback block and
`Fence.Stats` reports them once the fence has signalled:

```go
func (r *Recorder) CollectRunStats(bool) // default false

// SubmissionStats is valid only after the submission's fence has signalled.
// Calling it before is an error, not a stale read.
type SubmissionStats struct {
    Indirect []IndirectStats
}

type IndirectStats struct {
    Node    NodeID
    Actual  [3]uint32 // what the device supplied
    Max     [3]uint32 // the recorded build-time maximum
    Clamped bool      // Actual exceeded Max and was reduced
}

func (f *Fence) Stats() (SubmissionStats, error)
```

**A graph with `CollectRunStats` off still clamps**, per the `maxCount` rule
above; what it loses is being told. That is the honest reading of 005's sentence:
the clamp is reported when the caller asked for reporting, and is silent
otherwise, which is why the maximum is a documented caller obligation in release
mode rather than a checked one.

### Queue counters, cumulative

[001](001-device-resources.md) §8.2 says a full staging ring makes `Buffer.Write`
block and that this "is reported through queue statistics". That record lives
here, because the queue owns the staging ring:

```go
// QueueStats are cumulative since device open. They are counters, not a
// profiler: nothing here is per node and nothing here needs a readback.
type QueueStats struct {
    Submissions   int64
    BytesStaged   int64
    StagingWaits  int64 // times a Write blocked waiting for a recycled block
    ImmediateReads int64
    Repacks       int64 // immediate-path texture copies that repacked (001 4.2)
}

func (q *Queue) Stats() QueueStats
```

The immediate transfer path reports here rather than returning a `CopyStats` per
call, because widening `Buffer.Read` and `Texture.Read` to a second return value
would put an observability concern in two signatures every caller touches. A
caller who needs per-copy detail records the copy, which is the path 001 §8.1
already directs them to for anything that is not setup or debugging.

**What none of this is.** There are no per-node timings and no GPU timestamps.
[005](005-graphics.md) defers queries deliberately, and a fence guarantees
nothing about partial progress, so a timing API here would be inventing an
observable the model does not have. That gap is real and is named in the open
questions.

## A worked example

Eight nodes, one genuine diamond, six transients, and aliasing that a
record-order planner gets wrong. This is one attention block, simplified. The
numbers are for a 1024 by 1024 f32 activation: 4 MiB per tensor.

### Recording

```go
r := dev.NewRecorder()

params := r.Slot("params") // rebindable, 256 B uniform
x      := r.Slot("x")      // rebindable, 4 MiB, caller owned
kv     := r.Slot("kv")     // rebindable, caller owned KV slab
y      := r.Slot("y")      // rebindable, 4 MiB, caller owned output

t0 := r.Transient(f32Buf(1 << 20)) // normed        4 MiB
t1 := r.Transient(f32Buf(1 << 20)) // q             4 MiB
t2 := r.Transient(f32Buf(1 << 20)) // k             4 MiB
t3 := r.Transient(f32Buf(1 << 20)) // q after rope  4 MiB
t4 := r.Transient(f32Buf(1 << 19)) // scores        2 MiB
t5 := r.Transient(f32Buf(1 << 20)) // attn out      4 MiB

n0 := r.CopyToBuffer(params, hostParams)                     // host write
n1 := r.Dispatch(pNorm,  bind(x, params, t0),    wg(1024))   // t0 = norm(x)
n2 := r.Dispatch(pGemm,  bind(t0, wQ, t1),       wg(4096))   // t1 = t0 @ Wq
n3 := r.Dispatch(pGemm,  bind(t0, wK, t2),       wg(4096))   // t2 = t0 @ Wk
n4 := r.Dispatch(pRope,  bind(t1, params, t3),   wg(1024))   // t3 = rope(t1)
n5 := r.Dispatch(pScore, bind(t3, t2, kv, t4),   wg(2048))   // t4 = scores
n6 := r.Dispatch(pAttn,  bind(t4, kv, t5),       wg(2048))   // t5 = attend
n7 := r.Dispatch(pAdd,   bind(t5, x, y),         wg(1024))   // y  = x + t5

g, err := r.Build()
```

The diamond is `n1 -> {n2, n3} -> n5`, with `n4` extending one arm.

### Inferred edges

| Edge | Hazard | Resource | Range |
| --- | --- | --- | --- |
| n0 -> n1 | RAW | `params` | whole, 256 B |
| n0 -> n4 | RAW | `params` | whole, 256 B |
| n1 -> n2 | RAW | `t0` | whole |
| n1 -> n3 | RAW | `t0` | whole |
| n2 -> n4 | RAW | `t1` | whole |
| n4 -> n5 | RAW | `t3` | whole |
| n3 -> n5 | RAW | `t2` | whole |
| n5 -> n6 | RAW | `t4` | whole |
| n6 -> n7 | RAW | `t5` | whole |

Nine edges. Note what is **absent**: no edge between `n2` and `n3`, which is the
point, and no edge from `wQ`, `wK`, or `x` on the read side, because nothing in
the graph writes them. `n5` and `n6` both read `kv` and get no edge from that
either (RAR).

`n7` writes `y` and reads `x`, two resources, so no WAR edge. If a caller bound
one buffer to both the `x` and `y` slots, check V21 fires at submit, because the
builder assumed they were independent.

**The sub-range case, as a variant.** Replace `t1` and `t2` with one 8 MiB
transient `qk`, `n2` writing `qk[0 : 4Mi]` and `n3` writing `qk[4Mi : 8Mi]`. The
intervals are disjoint, so **no WAW edge is inferred** and the two GEMMs still
overlap. Under whole-resource comparison there is a WAW edge, a barrier between
them, and the two largest dispatches in the block serialize. The same applies to
the KV cache: `n5` reads `kv[0 : L*row]` and a cache-append node writes
`kv[L*row : (L+1)*row]`, disjoint, no WAR edge, no barrier per layer per token.
But only if the append node can declare that range, which under the per-step
address gap it cannot, so it declares the whole slab and takes the barrier. That
is the gap costing something measurable.

### Live ranges and interference

Users per transient, with the record-order interval alongside for comparison:

| Transient | Size | Users | Record-order interval |
| --- | --- | --- | --- |
| `t0` | 4 MiB | n1 w, n2 r, n3 r | [n1, n3] |
| `t1` | 4 MiB | n2 w, n4 r | [n2, n4] |
| `t2` | 4 MiB | n3 w, n5 r | [n3, n5] |
| `t3` | 4 MiB | n4 w, n5 r | [n4, n5] |
| `t4` | 2 MiB | n5 w, n6 r | [n5, n6] |
| `t5` | 4 MiB | n6 w, n7 r | [n6, n7] |

Compatibility, by the reachability test:

| Pair | Compatible | Why |
| --- | --- | --- |
| t0, t1 | no | `n2` uses both. |
| t0, t2 | no | `n3` uses both. |
| **t0, t3** | **no** | `n3` reads `t0`, `n4` writes `t3`, and **`n3` and `n4` are unordered.** Their record-order intervals are disjoint, so an interval planner aliases them. That is the bug. |
| t0, t4 | yes | Every user of `t0` reaches `n5` and `n6`. |
| t0, t5 | yes | Same. |
| t1, t2 | no | Both live across the diamond. |
| t1, t3 | no | `n4` uses both. |
| t1, t4 | yes | `t1` dies at `n4`, `t4` is born at `n5`, and `n4 -> n5`. |
| t1, t5 | yes | Same shape. |
| t2, t4 | no | `n5` uses both. |
| t2, t5 | yes | `n3 -> n5 -> n6` and `n5 -> n6`. |
| t3, t4 | no | `n5` uses both. |
| t3, t5 | yes | Both users of `t3` reach both users of `t5`. |
| t4, t5 | no | `n6` uses both. |

The `t0, t3` row is the finding. Both are 4 MiB, both look dead-then-born in
record order, and aliasing them corrupts `t0` on any backend that overlaps the
two arms of the diamond, which is every backend doing what the DAG permits. An
interval planner produces a 12 MiB pool that works on the CPU backend and fails
intermittently on Vulkan. That is precisely the class of bug this design claims
to remove, so the planner uses interference, not intervals.

### Packing

Size descending, ties by first-writer node id: `t0`(n1), `t1`(n2), `t2`(n3),
`t3`(n4), `t5`(n6), then `t4` (2 MiB).

| Step | Transient | Interfering placements | Chosen offset |
| --- | --- | --- | --- |
| 1 | t0, 4 MiB | none placed | `[0, 4)` |
| 2 | t1, 4 MiB | t0 `[0,4)` | `[4, 8)` |
| 3 | t2, 4 MiB | t0, t1 | `[8, 12)` |
| 4 | t3, 4 MiB | t0, t1, t2 | `[12, 16)` |
| 5 | t5, 4 MiB | none of the placed (compatible with t0, t1, t2, t3) | `[0, 4)`, aliasing t0 |
| 6 | t4, 2 MiB | t2 `[8,12)`, t3 `[12,16)`, t5 `[0,4)` | `[4, 6)`, aliasing part of t1 |

Offsets in MiB. Result:

| Measure | Bytes | Note |
| --- | --- | --- |
| `UnaliasedBytes` | 22 MiB | 4+4+4+4+2+4. |
| `PeakBytes` | 12 MiB | Record-order interval peak, at `n3`: t0, t1, t2. |
| `TransientBytes` | 16 MiB | What is allocated. |

Aliasing saved 6 MiB of 22, a 27 percent reduction. The 4 MiB gap between
`PeakBytes` and `TransientBytes` is **not** fragmentation: there is none, the
pool is fully packed. It is the price of DAG-safe aliasing, and it is exactly the
`t0`/`t3` pair an interval planner would have merged. Reporting both numbers is
what makes that visible instead of leaving 16 looking like a planner failure.

Is 16 MiB optimal? Here, yes: `{t0, t1, t2, t3}` are pairwise interfering, so no
assignment places four 4 MiB transients in under 16 MiB, and greedy hit the lower
bound. It will not always. Dynamic storage allocation is NP-hard and greedy by
size is a heuristic with no bound proved here.

### Barriers emitted

| Before | Reason | src | dst | Also covers |
| --- | --- | --- | --- | --- |
| n0 | head-of-submission acquire (external reads) | prior submission | transfer, compute | The cross-submission WAW on the transient pool, since last submission's `t5` occupies `t0`'s bytes. |
| n1 | RAW on `params` | transfer, write | compute, uniform read | |
| n2 | RAW on `t0` | compute, storage write | compute, storage read | **`n3`'s read of `t0`**: the barrier is queue-wide and `n3` follows `n2` in the stream. |
| n4 | RAW on `t1` | compute, write | compute, read | **`n5`'s read of `t2`** (written by `n3`, which precedes this barrier), and `n4`'s RAW on `params`. |
| n5 | RAW on `t3`, plus the `t1` to `t4` aliasing handover | compute, write | compute, read and write | |
| n6 | RAW on `t4`, plus the `t0` to `t5` aliasing handover | compute, write | compute, read and write | |
| n7 | RAW on `t5` | compute, write | compute, read | |

Nine data hazards and two aliasing handovers, seven barriers, one of which is
the submission boundary. Two
barriers carry hazards they were not emitted for, and both aliasing handovers
rode on a barrier data flow required anyway. No barrier sits between `n2` and
`n3`, the pair whose overlap matters most: they are the two GEMMs.

There is no end-of-graph barrier. `y` is written last and its consumer is either
the host, in which case `Fence.Wait` performs the visibility operation, or the
next submission, in which case that submission's head barrier does.

### What a submission costs

Per submission, after the first build:

| Backend | Per-submission work |
| --- | --- |
| Vulkan | Up to 4 descriptor writes if slots were rebound, one `vkQueueSubmit` of a pre-recorded primary command buffer. Zero encoding. |
| D3D12 | Descriptor heap writes for rebinds, one `ExecuteCommandLists` of a closed list. Zero encoding. |
| Metal | One fresh `MTLCommandBuffer`, 8 encoded dispatches plus 6 barrier operations from the planned list. No validation, no planning, no allocation. |
| GLES | One crossing to the context thread carrying the whole list, then 8 dispatches and 6 `glMemoryBarrier` calls on that thread. Under the predecessor's per-call marshalling this was 14 channel round trips. |
| CPU | 8 kernel executions plus 6 goroutine joins. |
| WebGPU | One `GPUCommandEncoder`, 8 `dispatchWorkgroups`, barriers are no-ops. |

What **no** backend does per submission: validate, infer edges, compute
reachability, plan memory, compute barriers, or allocate. That is the plan-once
saving [006](006-backends.md) section 4 identifies as backend independent and as
most of the value, and it is why this model is justified even where lowering is
software replay.

---

## Determinism

### Across two submissions of one graph, one backend

**Bit-identical**, given identical inputs, provided the kernels are themselves
deterministic. The graph contributes nothing nondeterministic: the plan, the pool
offsets, the barrier list, and the node order are all fixed at build.

What breaks it, none of which is the graph's doing:

- Atomic float add. Addition is not associative in floating point and the order
  is not defined. [002](002-compute-model.md) makes it a capability, and a kernel
  using it is not bit-reproducible even against itself.
- Reading a transient before writing it within a submission. The pool is aliased,
  so the bytes are whatever the previous submission left there, which differs
  between the first submission and the second. The CPU backend poisons transient
  memory at the start of every submission for exactly this reason, matching
  [002](002-compute-model.md)'s treatment of shared memory.
- Unordered fragment writes to a storage buffer, which
  [`conventions.md`](../docs/conventions.md) documents and this library declines
  to offer as a way to produce a deterministic buffer.
- Subgroup-size-dependent reduction order in a kernel that has one.

### Across backends

**Not bit-identical, and this spec does not claim it.**
[004](004-kernel-authoring.md)'s exactness section and [006](006-backends.md) own
that. Floating point contraction, transcendental implementations, and differing
workgroup sizes all move results within tolerance.

What **is** identical across backends, and is worth testing as such:

| Property | Identical across backends | Why |
| --- | --- | --- |
| The set of nodes | yes | The graph IR is backend independent. |
| The inferred edge set | yes | Inference reads only access declarations. |
| Barrier positions (which node each barrier precedes) | yes | The barrier walk is backend independent; only lowering is not. |
| The interference relation and the aliasing decisions | yes | Reachability over the same DAG. |
| Pool **offsets** and `TransientBytes` | **no** | Minimum binding alignment is device-reported (001), so a device with 256 byte alignment packs differently from one with 16. |
| Numerical results | no, within tolerance | 004, 006. |

The offsets row is the one that surprises people, so it is called out: a
golden-plan test must compare the plan's structure, not its byte offsets, or it
passes on one device and fails on the next.

---

## Open questions

- **Conditional and iterative execution.** A model with early exit, or a loop
  whose trip count depends on device data, cannot be a pure DAG. Options are
  indirect dispatch with a device-written count, sub-graphs invoked by the host
  per iteration, or leaving it out of v0. Not decided.
  [005](005-graphics.md) resolves only the bounded case (a build-time maximum
  count with the device supplying the actual one) and this spec adopts the same
  shape for indirect dispatch. The unbounded case is untouched. A zero-workgroup
  dispatch is the cheapest partial answer, since a node whose device-written
  count is zero costs almost nothing, and it makes "skip this block" expressible
  without leaving the DAG.
- **Cross-graph aliasing.** Two graphs alive at once each plan their own pool.
  Whether they can share is unresolved and matters for memory-constrained
  inference. [007](007-tensor-layer.md) puts a number on it: five prefill buckets
  at 200 MiB is a gigabyte of transients. The obvious shape is a pool object the
  caller creates and several graphs plan into, with a rule that graphs sharing a
  pool are mutually exclusive in flight, generalizing the one-in-flight rule from
  a graph to a pool. Not designed.
- **Whether one graph can have several transient sets.** The dual of the previous
  question and the direct fix for the one-in-flight rule: a submission carries a
  pool index, so N submissions of one graph run against N pools. Same memory cost
  as N graphs, minus N times the build. Deferred.
- **Node reordering for memory.** The planner takes record order as given.
  Reordering independent nodes can shorten live ranges materially, and it
  interacts with barrier batching (reordering to shorten a range can add a
  barrier) and with [005](005-graphics.md)'s draw order rule. Unmeasured.
- **Finer barrier primitives.** Buffer-range memory barriers, split barriers via
  Vulkan events, and D3D12 enhanced barriers all exist and none are used. The
  current design drains the pipeline where a range-scoped barrier would not. This
  is the largest known performance gap in the barrier design.
- **There is no timing observability at all.** The statistics section reports
  what the plan decided and what the device counted, and nothing about how long
  anything took. A fence guarantees nothing about partial progress, so per-node
  timing needs GPU timestamp queries, which [005](005-graphics.md) defers as a
  profiling concern. The consequence is worth stating plainly: at v0 a caller
  asking "which node is slow" has no answer from this library, on a library whose
  reason to exist is throughput. The shape when it arrives is a timestamp pool
  written at node boundaries, opt-in for the same reason the run-time counters
  are, and it should not be designed out by the barrier batching, which currently
  merges the boundaries a timestamp would sit on.
- **Cross-queue ownership transfer.** Vulkan requires a release plus acquire pair
  when a resource moves between queue families; concurrent sharing mode avoids it
  and is slower on some hardware. Neither chosen. It bites only with two queue
  families in use.
- **The per-step address gap**, restated as a question rather than a note: is
  there a fourth kind of variation, a slot whose *offset* varies at submit while
  its resource does not? It would let a KV append declare a tight range and drop
  the barrier the current workaround forces. Against: it is a per-submission
  descriptor update on Vulkan and D3D12, which is what routing through contents
  was avoiding, and it makes the ranges hazards were inferred from
  submission-dependent, which breaks the infer-once property outright. Probably
  no, but the cost of no is now written down.
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
- ~~Rebindable slots versus concurrent submission~~ (raised by
  [005](005-graphics.md)). **Resolved above**: one submission in flight per graph,
  which covers the transient race as well as the rebinding race.

## Testing

### Edge inference

- Two nodes writing disjoint byte ranges of one buffer produce **no** edge. The
  packed QKV shape from the worked example, asserted on the plan, not on results.
- Two nodes writing overlapping ranges produce a WAW edge; a reader after a
  writer produces RAW; a writer after a reader produces WAR. One test per hazard,
  asserting the edge exists and its classification.
- A resource no node writes produces zero edges regardless of reader count, and
  two `AtomicRMW` nodes on one range produce none either, while a plain read after
  an `AtomicRMW` node does.
- A node reading and writing one range hazards against earlier nodes, not itself.
- The inferred edge set is identical on every backend for one recording.

### Memory planning

- **The diamond case**, as its own test, because it is the one an interval
  planner gets wrong: the worked example's `t0` and `t3` must not share bytes,
  asserted on the plan.
- A graph whose transients have genuinely disjoint live ranges reports
  `TransientBytes` below `UnaliasedBytes`, and a graph whose ranges all overlap
  reports them equal.
- The worked example reports exactly 22 MiB unaliased, 12 MiB peak, and 16 MiB
  allocated on a device with 256 byte alignment. Golden numbers, so a planner
  regression is a diff.
- `Graph.Memory().TransientBytes` equals the pool high-water mark actually
  allocated, checked against pool statistics after a submission.
- Aliased transients sharing bytes are never simultaneously live: a debug mode
  writes a per-transient sentinel on first write and checks it on every read, so a
  planner bug is a sentinel mismatch rather than a wrong number.
- Two builds of one recording produce byte-identical plans, repeated enough that
  a map iteration order leak would show.

### Barriers

- A read-after-write chain whose result is wrong if the barrier is missing, run
  enough times to catch a race, on every backend including CPU. Under
  `go test -race` on the CPU backend a missing barrier is a reported data race,
  per [006](006-backends.md).
- The barrier count and positions for the worked example are asserted exactly:
  six plus the submission boundary. A change that adds one is a change someone
  should have to justify.
- A WAR hazard emits an execution barrier and no memory barrier. Asserted on the
  plan, since no result can distinguish them.
- No barrier appears inside a render pass, on a deferred graph with a
  multi-attachment pass, per [005](005-graphics.md).
- An aliasing handover not already covered by a data-flow barrier emits one.
  Constructed rather than found, since the worked example has none.

### Validation

- One test per row of the check table, each asserting the sentinel through
  `errors.Is`, the node id, and the slot index through `errors.As`.
- A graph with four independent problems reports four, not one.
- Every error message contains a file and line from the caller's source, checked
  by matching the test's own file name. This is the check that keeps the
  deferred-validation promise honest, and it is the one most likely to rot.
- V21: binding one buffer to two slots the builder treated as independent fails
  at submit with `ErrRebindOverlap`, and binding it to two read-only slots
  succeeds.
- V23: the same overlap expressed with two *statically* bound views fails at
  build instead, with the recording call site in the message. Both halves are
  tested, because a builder that implements one and not the other passes whichever
  test it has.

### Submission and fences

- A graph submitted twice with identical bindings produces identical results.
- Rebinding an input between submissions changes the result, and rebinding a
  mismatched resource is rejected.
- Two submissions to one queue are ordered: the second reads what the first
  wrote, no caller barrier, under load.
- Submitting a graph with a submission already in flight returns a fence failing
  with `ErrGraphInFlight`, leaving the in-flight submission unaffected.
- `Fence.Wait` returning nil implies host visibility of `Readback` memory with no
  further caller action, on every backend.
- `SubmitAfter` across two queues, where the device has two, orders correctly and
  does not round trip through the host (asserted by timing, not by structure).
- A closed resource used by an in-flight submission is kept alive until the fence
  signals, per [001](001-device-resources.md).

### Determinism

- Two submissions of one graph on one backend are bit-identical, for a kernel set
  with no atomic float add.
- A kernel reading a transient it did not write fails on the CPU backend against
  the poison pattern.
- Plan structure (nodes, edges, barrier positions, aliasing decisions) is
  identical across backends for one recording; pool offsets are compared only
  within a device.

### The whole-plan oracle

The strongest test available, and cheap to build: execute the graph a second time
under a **naive plan**, no aliasing at all and a full barrier between every pair
of nodes in record order, and compare results. Any disagreement is a planner or
barrier bug, localized to the graph builder rather than to a kernel. Run it over
randomly generated graphs (random node counts, random access patterns, random
sub-ranges) as a fuzz target. It catches the interference bug above on the first
seed that produces a diamond.
