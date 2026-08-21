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

Nothing here is implemented yet. The repository is a design at this stage, plus
an API surface that compiles and does nothing.

## Reading order

Start with [000](000-decisions.md), then [003](003-command-graph.md). Decision 1 of the
design, that the unit of submission is a recordable and replayable command graph
rather than a one-shot encoder, is the choice the rest of layer 1 is shaped
around, and the other specs are hard to evaluate without it.

## Decisions

| Spec | Status | Covers |
| --- | --- | --- |
| [000-decisions.md](000-decisions.md) | Locked | The two-layer split, the graph submission model, cgo-free, the CPU oracle, the compute model, kernels in Go, queryable capabilities |

## Layer 1: the device

| Spec | Status | Covers |
| --- | --- | --- |
| [001-device-resources.md](001-device-resources.md) | Drafted | Devices, pooled memory with explicit memory kinds, buffers, views, textures, transfers, lifetime |
| [002-compute-model.md](002-compute-model.md) | Drafted | Workgroups, shared memory, barriers, atomics, subgroups, the dtype set, capabilities |
| [003-command-graph.md](003-command-graph.md) | Drafted | Recording, immutability, validation, memory planning, computed barriers, submission and fences |
| [004-kernel-authoring.md](004-kernel-authoring.md) | Drafted | The Go subset that is the kernel language, `go/types` checking, lowering to MSL / GLSL / SPIR-V / HLSL / Go |
| [005-graphics.md](005-graphics.md) | Drafted | Render pipelines, render passes as graph nodes, draws, the render-to-compute handoff, surfaces and present |
| [006-backends.md](006-backends.md) | Drafted | The backend contract, the capability matrix, per-backend assessment, graph lowering, the CPU oracle |

## Layer 2: the tensor

| Spec | Status | Covers |
| --- | --- | --- |
| [007-tensor-layer.md](007-tensor-layer.md) | Drafted | Dtypes, shapes, strides and views, the operator set, graph build and execute, memory planning, quantization |

## Conventions

- **Numbered** so dependency order is visible at a glance. Numbers are stable
  once assigned; a retired spec keeps its number.
- **Frontmatter** carries `title`, `status`, `layer`, and `depends_on`.
- **Status** is one of drafted, in progress, implemented, or superseded.
- Every spec ends with **open questions** and a **testing strategy**. A spec with
  no open questions is usually a spec that has not been thought about hard
  enough.
- Costs and unknowns are stated rather than glossed. A design that only lists
  advantages has not been evaluated.
