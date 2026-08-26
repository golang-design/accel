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

**M0 through M7 are complete, and M8 is five of seven.** Compute runs end to end
on the CPU backend and on Metal: pooled memory and typed views; kernels written
in the Go subset, compiled to a generated lowering and to MSL from one IR;
command graphs with inferred edges, computed barriers and reachability-based
transient aliasing; cooperative kernels with shared memory, barriers and
atomics; a portable tiled GEMM; and the tensor layer above it, with quantized
weights, sampling, a paged KV cache, and prefill and decode attention.

Graphics is being built: its five child designs are written, and the CPU backend
now runs a whole frame — vertex and index buffers, attributes, by-value stage
parameters, depth, blending, and a headless surface through acquire and present.
Metal runs the same passes and is compared against the CPU rasterizer pixel by
pixel, and presents to a `CAMetalLayer` the caller owns. What is unbuilt is the
compositor handoff, which needs a display, and multisampling.
[009](009-sequencing.md) records what has landed, in what order, and the
deviations taken.

**What v0 builds** is stated in [000](000-decisions.md#the-v0-milestone) and is
narrower than this directory: compute only, on the CPU backend and Metal.
[005](005-graphics.md) is the parent of graphics; its five child designs are
written — [032](032-stage-abi.md) through [035](035-cpu-rasterizer.md), plus
[041](041-msaa.md) — and both backends render, with the CPU rasterizer as the
oracle Metal is checked against. [006](006-backends.md) specifies three remaining synchronous
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
| Done | M0 through M7, and five of M8's seven items: quantization, sampling, the plan cache, paged KV with batching, and shared transients |
| In progress | The CPU reference rasterizer ([035](035-cpu-rasterizer.md)), the render API that drives it ([033](033-render-api.md)), the stage-ABI compiler work both need ([032](032-stage-abi.md)), and the headless half of surface/present ([034](034-surface-present.md)) |
| Written, unbuilt | [037](037-vulkan-bringup.md) Vulkan, [038](038-spirv-target.md) SPIR-V, [040](040-batch-scheduler.md) the scheduler, [041](041-msaa.md) MSAA |
| Not blocked, unscheduled | Vulkan — see 009's correction; it is verifiable in CI on lavapipe today |

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
| [002-compute-model.md](002-compute-model.md) | In progress | Workgroups, shared memory, barriers, atomics, subgroups, the dtype set, capabilities |
| [031-shared-transients.md](031-shared-transients.md) | Implemented | M8: one transient pool shared by several graphs, and the in-flight rule that makes it safe |
| [032-stage-abi.md](032-stage-abi.md) | In progress | 005's first child: the vertex and fragment signatures, varyings, clip and depth ranges, texel fetch, and what the IR gains |
| [033-render-api.md](033-render-api.md) | In progress | 005's second child: render pipelines, pass nodes, load/store actions, declared access, draws and indirect counts |
| [034-surface-present.md](034-surface-present.md) | In progress | 005's third child: swapchains, the typed present slot, resize, headless surfaces, and where the windowing line is |
| [035-cpu-rasterizer.md](035-cpu-rasterizer.md) | In progress | 005's fourth child: the reference rasterizer, the fill rule, interpolation, and the conformance corpus with its exact-versus-bounded split |
| [037-vulkan-bringup.md](037-vulkan-bringup.md) | Drafted | The cgo-free Vulkan backend through purego: loader, device, memory, descriptors, submission and device loss |
| [038-spirv-target.md](038-spirv-target.md) | Drafted | Emitting SPIR-V from the shared IR, and how a binary target with no source level is verified |
| [041-msaa.md](041-msaa.md) | Drafted | 005's fifth child: sample positions, resolve, and what the CPU oracle can still prove once a sample pattern exists |
| [003-command-graph.md](003-command-graph.md) | In progress | Recording, immutability, validation, memory planning, computed barriers, submission and fences |
| [004-kernel-authoring.md](004-kernel-authoring.md) | In progress | The Go subset that is the kernel language, `go/types` checking, lowering to MSL / GLSL / SPIR-V / HLSL / Go |
| [005-graphics.md](005-graphics.md) | Drafted | The normative parent of [032](032-stage-abi.md) through [035](035-cpu-rasterizer.md) and [041](041-msaa.md): graphics constraints, with its four open questions closed in those children |
| [006-backends.md](006-backends.md) | In progress | The backend contract, the capability matrix, per-backend assessment, graph lowering, the CPU oracle |
| [008-numerics.md](008-numerics.md) | In progress | Proven exact domains, normative primitive ceilings, derived reductions, and composed error budgets |
| [012-kernel-pipeline.md](012-kernel-pipeline.md) | Implemented | M2 child: the whole compiler pipeline for one straight-line kernel, and why the cut is vertical |
| [013-kernel-subset.md](013-kernel-subset.md) | Implemented | M2 child: control flow, helpers, and the positioned rejection corpus |
| [014-kernel-uniforms.md](014-kernel-uniforms.md) | Implemented | M2 child: std140 layout, generated uniform codecs, and `UniformBuffer[T]` |
| [015-graph-recording.md](015-graph-recording.md) | Implemented | M3 child: recording, slots, build validation, submission and fences, and the record-order plan that becomes the oracle |
| [016-graph-execution.md](016-graph-execution.md) | Implemented | M3 child: edge inference, sub-range hazards, barrier planning, and the flat dispatch node |
| [017-graph-aliasing.md](017-graph-aliasing.md) | Implemented | M3 child: interference over reachability, greedy packing, and the whole-plan differential fuzz |
| [018-cooperative-lowering.md](018-cooperative-lowering.md) | Implemented | M4 child: the uniformity analysis, the state split, the workgroup scheduler, and both lowerings from one IR |
| [019-cooperative-diagnostics.md](019-cooperative-diagnostics.md) | Implemented | M4 child: shared-memory definition, barrier arrival, and conflicting access, each reported deterministically |
| [020-cooperative-atomics.md](020-cooperative-atomics.md) | In progress | M4 child: atomics, emulated subgroups and their sweeps, capability inference, and `reduce_sum` |
| [021-metal-bringup.md](021-metal-bringup.md) | Implemented | M6 child: the Objective-C shim and its ownership rule, enumeration, storage modes, a straight-line MSL emitter, and one kernel on the GPU |
| [022-msl-target.md](022-msl-target.md) | Implemented | M6 child: the Metal numeric profile, then threadgroup memory, barriers, atomics, subgroups, helpers and intrinsics in MSL |
| [023-metal-graph.md](023-metal-graph.md) | Implemented | M6 child: multi-node re-encoding, indirect dispatch, completion-handler lifetime, device loss, and the M6 E2E |

## Layer 2: the tensor

| Spec | Status | Covers |
| --- | --- | --- |
| [007-tensor-layer.md](007-tensor-layer.md) | In progress | Caller-owned plans and state, concrete ports, unquantized f16/f32 operators, minimal prefill and decode |
| [024-tensor-bringup.md](024-tensor-bringup.md) | Implemented | M7 child: the builder, shape and dtype inference, lowering to a recorder, plans and bindings, and the elementwise operators on both backends |
| [025-tensor-operators.md](025-tensor-operators.md) | Implemented | M7 child: views and indexing, materialization, `Rows`, `RMSNorm`, `Softmax`, `RoPE`, `MatMul` and `Linear` |
| [026-tensor-decode.md](026-tensor-decode.md) | Implemented | M7 child: persistent state as versions, the KV cache, attention, and the decode step |
| [027-quantization.md](027-quantization.md) | Implemented | M8: the symmetric int8 block representation, its derived error bound, and quantized Rows and GEMM |
| [028-sampling.md](028-sampling.md) | Implemented | M8: argmax, categorical sampling, and top-k/top-p truncation, with the random draw as an input so a token is reproducible — public `tensor` operators, batched, one draw per row |
| [029-plan-cache.md](029-plan-cache.md) | Implemented | M8: prefill buckets, and a plan cache whose key is the six things that make reuse safe |
| [030-paged-kv.md](030-paged-kv.md) | Implemented | M8: a block pool and page tables, so sequences of different lengths share one cache |
| [043-per-row-values.md](043-per-row-values.md) | Implemented | M8: the one line behind five consumer reports — a value every row of a dispatch shares is a scalar, and a value that differs per row is a tensor |
| [044-unbounded-context.md](044-unbounded-context.md) | Implemented | M8: attention over a cache larger than a workgroup, and why the loop is bounded by a binding's extent rather than by the length |
| [047-linear-attention.md](047-linear-attention.md) | In progress | A matrix state per sequence and a scan over 046's segmented extent, so hybrid models whose three-in-four layers are not softmax attention become expressible |
| [046-segmented-extents.md](046-segmented-extents.md) | In progress | A count per row, the offsets it implies, and the ragged query extent that closes accel issue 16. Written as a primitive because issue 18's grouped GEMM is its second caller |
| [045-texture-attachments.md](045-texture-attachments.md) | In progress | M9: attachments become textures, a view names a subresource and may reinterpret its format, and a stage can fetch a texel. Resources, the render API, both backends, V13 and the stage half of texel fetch are built (§8); binding a texture to a pass, mip levels above one, and feedback rejection are owed |
| [042-surface-completion.md](042-surface-completion.md) | In progress | Completing the public surface: what each exported declaration does, refuses, or is still owed |
| [039-sampling-policy.md](039-sampling-policy.md) | In progress | Post-v0: temperature, penalties, the composition order, and a seeded stream that makes a whole sequence reproducible rather than one token. The stream, the penalty kernels, the composed policy and §9's assertions are built, including the cross-backend token differential; four deviations are recorded |
| [040-batch-scheduler.md](040-batch-scheduler.md) | Drafted | Post-v0: slots over 030's pool, one plan at max batch with parked idle slots, admission, eviction, and the drain a membership-size change costs |
| [010-kernel-corpus.md](010-kernel-corpus.md) | In progress | Required unquantized kernels, variants, layouts, deterministic selection, and per-kernel proof obligations |

## Process and project

| Spec | Status | Covers |
| --- | --- | --- |
| [036-documentation.md](036-documentation.md) | In progress | Who each document is for, what builder-voice looks like, the ten-tutorial deck, and the public-surface freeze record that gates it |
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
  002 stays in progress while §5's remaining reduction operators and dtypes, and
  subgroup operations in divergent control flow, are unbuilt — even though
  everything M4 promised of it is done.
- Every implementation-bearing spec states its **testing strategy**. Genuine
  unresolved decisions stay under **open questions**; resolved questions are
  removed rather than kept as stale history.
- Costs and unknowns are stated rather than glossed. A design that only lists
  advantages has not been evaluated.
