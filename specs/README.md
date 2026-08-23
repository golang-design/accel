# accel specs

Design specs, written before implementation. Each is bounded, states its
decisions with the reasoning behind them, and is honest about what it does not
resolve.

These are **internal design documents**: decisions, tradeoffs, and the reasoning
behind them, written for whoever is building or reviewing the thing. Documentation
written for people *using* or *contributing to* accel lives in
[`../docs/`](../docs/).

The decisions everything here is built on live in
[`000-decisions.md`](000-decisions.md). It is normative: a spec that contradicts
it is wrong.

M0 through M3 are complete. A CPU device can be opened, pooled memory allocated
and suballocated, buffers typed and sliced into views, bytes moved to the device
and back, and a kernel written in the Go subset compiled to a generated lowering
that runs and is checked against the source it came from. Graphs of transfers
and flat compute dispatches can be recorded, validated, built once, bound,
submitted, waited on, and replayed with a rebound input: dependency edges are
inferred from declared byte ranges, barriers come from those edges and are
batched, and transients are aliased by reachability and checked against the
conservative plan they replaced. Cooperative kernels are next. Everything past
that still reports `ErrNotImplemented`.
[009](009-sequencing.md) records what has landed and the deviations taken.

**What v0 builds** is stated in [000](000-decisions.md#the-v0-milestone) and is
narrower than this directory: compute only, on the CPU backend and Metal.
[005](005-graphics.md) is a drafted post-v0 parent whose four child designs are
not yet written. [006](006-backends.md) specifies three remaining synchronous
backends plus a deferred asynchronous WebGPU shape; all are unbuilt. A spec being
here means its scope and current decisions are reviewable, not that its code is
next.

## Reading order

Start with [000](000-decisions.md), then [003](003-command-graph.md). Decision 1 of the
design, that the unit of submission is a recordable and replayable command graph
rather than a one-shot encoder, is the choice the rest of layer 1 is shaped
around, and the other specs are hard to evaluate without it.

If you are here to build rather than to review, read
[009](009-sequencing.md) third. It is where the order of work lives, and it is
not the order these files are numbered in.

For implementation, [010](010-kernel-corpus.md) is the exact unquantized kernel
inventory and [011](011-conformance-harness.md) is the shared proof machinery.

## Where the work stands

| | |
| --- | --- |
| Done | M0's cgo-free gate, M1's memory on the CPU backend, M2's kernel compiler in full |
| Next | M3, graph planning and flat compute/transfer submission |
| Blocked on nothing | M2's inputs are 002, 004, and 011, all of which are drafted or in progress |
| Retired | 009's compiler-scope risk, by M2's direct flat E2E |

[009](009-sequencing.md) has the milestone list, what done means for each, and
the deviations taken so far. It is the file to read before picking anything up,
because the order things can be demonstrated in is not the order these files are
numbered in.

## Decisions

| Spec | Status | Covers |
| --- | --- | --- |
| [000-decisions.md](000-decisions.md) | Locked | The two-layer split, the graph submission model, cgo-free, the CPU oracle, the compute model, kernels in Go, queryable capabilities |

## Layer 1: the device

| Spec | Status | Covers |
| --- | --- | --- |
| [001-device-resources.md](001-device-resources.md) | Implemented | Devices, pooled memory with explicit memory kinds, buffers, views, textures, transfers, lifetime |
| [002-compute-model.md](002-compute-model.md) | Drafted | Workgroups, shared memory, barriers, atomics, subgroups, the dtype set, capabilities |
| [003-command-graph.md](003-command-graph.md) | Drafted | Recording, immutability, validation, memory planning, computed barriers, submission and fences |
| [004-kernel-authoring.md](004-kernel-authoring.md) | Drafted | The Go subset that is the kernel language, `go/types` checking, lowering to MSL / GLSL / SPIR-V / HLSL / Go |
| [005-graphics.md](005-graphics.md) | Drafted parent | Post-v0 graphics constraints and the four child specs required before implementation |
| [006-backends.md](006-backends.md) | Drafted | The backend contract, the capability matrix, per-backend assessment, graph lowering, the CPU oracle |
| [008-numerics.md](008-numerics.md) | Drafted | Proven exact domains, normative primitive ceilings, derived reductions, and composed error budgets |
| [012-kernel-pipeline.md](012-kernel-pipeline.md) | Implemented | M2 child: the whole compiler pipeline for one straight-line kernel, and why the cut is vertical |
| [013-kernel-subset.md](013-kernel-subset.md) | Implemented | M2 child: control flow, helpers, and the positioned rejection corpus |
| [014-kernel-uniforms.md](014-kernel-uniforms.md) | Implemented | M2 child: std140 codecs, typed uniform binding, and the device-side layout check |
| [015-graph-recording.md](015-graph-recording.md) | Implemented | M3 child: recording, slots, build validation, submission and fences, and the record-order plan that becomes the oracle |
| [016-graph-execution.md](016-graph-execution.md) | Implemented | M3 child: edge inference, sub-range hazards, barrier planning, and the flat dispatch node |
| [017-graph-aliasing.md](017-graph-aliasing.md) | Implemented | M3 child: interference over reachability, greedy packing, and the whole-plan differential fuzz |
| [018-cooperative-lowering.md](018-cooperative-lowering.md) | Implemented | M4 child: the uniformity analysis, the state split, the workgroup scheduler, and both lowerings from one IR |
| [019-cooperative-diagnostics.md](019-cooperative-diagnostics.md) | Implemented | M4 child: shared-memory definition, barrier arrival, and conflicting access, each reported deterministically |
| [020-cooperative-atomics.md](020-cooperative-atomics.md) | In progress | M4 child: atomics, emulated subgroups and their sweeps, capability inference, and `reduce_sum` |

## Layer 2: the tensor

| Spec | Status | Covers |
| --- | --- | --- |
| [007-tensor-layer.md](007-tensor-layer.md) | Drafted | Caller-owned plans and state, concrete ports, unquantized f16/f32 operators, minimal prefill and decode |
| [010-kernel-corpus.md](010-kernel-corpus.md) | In progress | Required unquantized kernels, variants, layouts, deterministic selection, and per-kernel proof obligations |

## Process

| Spec | Status | Covers |
| --- | --- | --- |
| [009-sequencing.md](009-sequencing.md) | In progress | What gets built in what order, what done means per milestone, the work no spec owns, and the risks with what retires each |
| [011-conformance-harness.md](011-conformance-harness.md) | In progress | Profiles, comparisons, oracles, fuzzing, E2E scenarios, diagnostics, and greater-than-90% coverage gates |

This is the one spec here that is expected to change as work lands. Read it
before picking anything up: the `depends_on` graph is a graph of documents, and
the order things can actually be demonstrated in is different.

## Conventions

- **Numbered** so dependency order is visible at a glance. Numbers are stable
  once assigned; a retired spec keeps its number.
- **Frontmatter** carries `title`, `status`, `layer`, and `depends_on`; 000 is the
  intentionally frontmatter-free normative decision record.
- **Status** is one of drafted, in progress, implemented, or superseded. *In
  progress* means some of the spec has shipped, and the spec says which parts.
  A spec does not reach *implemented* while any section it owns is unbuilt, so
  001 stays in progress with textures and device loss outstanding even though
  everything M1 promised is done.
- Every implementation-bearing spec states its **testing strategy**. Genuine
  unresolved decisions stay under **open questions**; resolved questions are
  removed rather than kept as stale history.
- Costs and unknowns are stated rather than glossed. A design that only lists
  advantages has not been evaluated.
