---
title: "Sequencing: what gets built, in what order, and what done means"
status: drafted
layer: process
depends_on:
  - 000-decisions.md
  - 010-kernel-corpus.md
  - 011-conformance-harness.md
---

# Sequencing

This spec orders demonstrable increments. A milestone is complete only when its
tests, coverage, documentation, and E2E gate are present; declarations that panic
with `ErrNotImplemented` are scaffolding, not progress against a definition of
done.

The document changes as work lands. Completed milestone outcomes remain recorded
so the assumptions under which later work began are reviewable.

## Cross-milestone quality gates

Every implementation milestone after M0 must satisfy all of these:

- new behavior has a test that fails without it;
- the affected package has greater than 90% statement coverage on the CPU path;
- generated files are fresh under `go generate ./...` once generation exists;
- `go test -race ./...`, `go vet ./...`, `gofmt`, and the cgo-free gate pass;
- every public API added or changed is documented for a builder audience; and
- the milestone's named E2E runs through public APIs, not internal shortcuts.

Backend-specific code may be unreachable on the CPU, so its line coverage is not
used to dilute or fail the CPU package threshold. It is instead covered by its
backend entry-gate integration tests and target compiler acceptance tests.

[`011-conformance-harness.md`](011-conformance-harness.md) owns common test
execution, comparisons, capability forcing, fuzzing, and coverage reporting.
[`010-kernel-corpus.md`](010-kernel-corpus.md) owns the required kernels and
variants. Neither is a final milestone: both grow incrementally when their first
consumer lands.

## Milestones

### M0. The cgo-free build gate — complete 2026-08-21

Shipped: `.github/workflows/ci.yml` builds, vets, and tests with
`CGO_ENABLED=0` on Linux, macOS, and Windows; checks formatting; and rejects
`import "C"` across Go files.

Implements: [`000-decisions.md`](000-decisions.md) decision 2.

Outcome: the non-functional device API surface compiles on all three v0 host
platforms and the cgo-free constraint is mechanically enforced. There were no
runtime tests because all operations remained design-stage stubs. The workflow's
negative cgo check is the acceptance test for this milestone.

### M1. Memory on the CPU backend

Build:

- CPU enumeration/open, device identity, capabilities, and limits;
- general TLSF and linear allocators;
- pools, buffers, typed views, `ViewAs`, lifetimes, and retain sets; and
- synchronous buffer read/write plus recorded host/device copy data ownership.

Implements: 001 §§1–3 and §§5–8, excluding textures.

Harness increment: CPU device discovery, strict/permissive mode selection,
exact byte/dtype comparisons, capability overrides, allocator fuzz plumbing, and
coverage reporting from 011.

Done:

- round trips at every v0 dtype and memory kind;
- view/reinterpretation bit patterns and every alignment/overlap rejection;
- TLSF fragmentation and bounded-operation tests plus allocator fuzzing;
- lifetime/close/retain races under `-race`; and
- E2E: public `OpenCPU(CPUOptions{})` → allocate → queue write → queue read →
  close.

Textures and formats remain deferred until graphics work; no earlier milestone
reads one.

### M2. Minimum compiler and flat direct CPU execution

Build:

- `go/types` package loading and subset validation;
- the typed structured IR required for flat control flow;
- generated registration, source hashes, binding metadata, and std140
  encoder/decoder;
- the generated Go adapter; and
- a **direct flat CPU executor** that invokes that adapter over independent
  invocations without a device Graph, shared memory, barriers, atomics, or
  subgroups.

The direct executor is test infrastructure and compiler bring-up, not a public
submission API. It disappears behind the common harness after M3. Its restricted
kernel descriptor rejects shared parameters and all cooperative intrinsics.

Implements: the CPU/flat subset of 004 and the first flat kernels from 010.

Harness increment: generated-kernel discovery, source-position negative tests,
exact/primitive bounded comparison contexts, generated-source freshness, and a
flat direct-execution adapter.

Done:

- generated flat `Add` accepts slice and uniform-struct parameters and matches an
  independent reference through the direct executor;
- std140 encode/decode round-trips structs containing scalar/vector/array fields;
- one negative test per rejected v0 construct asserts message and position;
- editing a kernel without regenerating fails freshness checks naming it; and
- E2E: source package → generator → registered adapter → direct CPU execution →
  independently checked output.

004 fixes the tool placement, closed structured-IR node set, intrinsic object
table, and internal corpus package boundary. M2 implements those decisions; it
does not reopen them inside the estimate.

### M3. Graph planning and flat compute/transfer submission

Build:

- recorder, slots, copy and **flat compute** dispatch nodes;
- edge inference, reachability interference, greedy transient packing, barrier
  planning, validation, binding/rebinding, queues, fences, and plan statistics;
- CPU lowering for transfers and flat kernels only.

Shared-memory parameters and cooperative kernels remain rejected until M4. This
keeps M3's whole-plan oracle executable without pretending barriers inside a
kernel already work.

Implements: 003 for transfer and flat compute nodes on one backend.

Harness increment: graph-plan golden comparison, naive non-aliasing/full-barrier
oracle mode, graph fuzz generation, binding failure helpers, and public queue E2E.

Done:

- 003's worked graph asserts 22 MiB unaliased, 12 MiB peak, 16 MiB allocated,
  and seven planned barriers;
- the diamond case proves unsafe transients do not alias;
- every applicable validation row has a focused negative test;
- whole-plan fuzz compares optimized execution with the naive plan; and
- E2E: public recorder with upload → flat Add dispatch → readback, retained and
  replayed with a rebound input.

### M4. Cooperative execution model on the CPU

The cooperative lowering is a compiler pass, not a runtime option, so it is its
own milestone rather than a line item under the GEMM. 004 replaces
goroutine-per-invocation with a generated resumable lowering: the structured IR
is split at barriers and subgroup rendezvous into states, each invocation carries
a program counter and its locals, and a workgroup scheduler advances every active
invocation to its next suspension point. That transform, its instrumentation, and
the analyses that decide when it is required are the work here. Splitting it out
keeps a compiler pass from being estimated as part of a kernel.

The transform is bounded by a rule this design already imposes: 002 §3.1 requires
every barrier to sit in uniform control flow, so suspension points form a
sequence rather than an arbitrary graph and no relooper is needed. That is why
this is a milestone and not a project.

Build:

- the resumable cooperative lowering and the flat lowering, both generated from
  one IR, with the flat path selected when no shared memory, barrier, or subgroup
  operation appears;
- shared-memory definition tracking (a shadow initialized bit per element), the
  deterministic barrier-generation check, and deterministic conflicting-access
  reporting;
- atomics, emulated subgroups, and the CPU developer/strict/mimic modes; and
- uniformity and capability inference over the IR.

Implements: 002, the CPU requirements in 006 §5, 008's CPU numeric profile, and
010's `reduce_sum`.

Harness increment: cooperative diagnostics, subgroup sweeps, contraction and
rounding probes, and reduction budgets.

Done:

- CPU arm64 and amd64 numeric probes establish the available exact domain before
  another test relies on it;
- `reduce_sum` matches its higher-precision reference under 008 at lengths that
  are not multiples of the workgroup size;
- a kernel reading shared memory it never wrote fails for **every** stored bit
  pattern, so the test cannot pass because a sentinel happened to compare
  unequal;
- non-uniform arrival, two invocations reaching different barrier IDs, and an
  unordered conflicting access pair are each reported deterministically with
  source position, workgroup, and invocation, on the first offending run rather
  than on an unlucky interleaving;
- the flat and cooperative lowerings agree on every kernel eligible for both; and
- subgroup paths and their fallbacks agree at sizes 1, 4, 32, and 64.

`go test -race` runs over this milestone, and it checks the CPU **runtime**, not
the kernel. Kernel races are found by the instrumentation above, which is why
they are found deterministically.

### M5. The portable tiled GEMM on the CPU

Build: 010's `matmul` tiled variant and the guarded-tail machinery it needs.

Implements: 002 §7 and the tiled family of 010.

Done:

- the tiled f16-storage, f32-accumulate GEMM matches an independently written
  higher-precision reference under 008's per-output dot-product budget, at
  dimensions that are not multiples of any tile dimension;
- removing either of its two barriers fails: the first through conflicting-access
  reporting, the second through definition tracking or the in-band sentinel
  kernel; and
- E2E: public graph submission runs upload → portable tiled GEMM → readback in
  strict mode.

This is the first point at which the portable compute model is proven, and it is
000's second v0 proof obligation.

### M6. Metal

Build:

- Objective-C runtime shim over `purego` and Metal object lifetime management;
- Metal resources, queue, graph lowering by re-encoding per submission, and
  capability/limit queries; and
- MSL target for every v0 kernel variant already required by 010.

Implements: 006 §§2.2 and 4.3, the MSL subset of 004, and Metal profiles in 008
and 011.

Done, in order:

1. load/probe and complete device profile;
2. contraction, rounding, division, and v0 transcendental probes;
3. buffer round trip at every v0 dtype;
4. barrier/shared-memory reduction against the oracle;
5. portable tiled GEMM against the higher-precision reference; and
6. E2E: the same public upload → GEMM → readback scenario as CPU, selected by
   enumerating a Metal `AdapterID` and calling `OpenDevice`.

The numeric probes run before the GEMM. If Metal misses a normative ceiling, the
lowering or supported domain changes; tests are not widened. Completion-handler
lifetime is exercised under repeated early closes and asynchronous completion.

### M7. Tensor decode plus minimal prefill

Prerequisites: 007's v0 API is stable, 010's complete unquantized v0 tensor
kernel list is implemented on CPU and Metal, and 011's operator/model budget and
E2E layers are operational.

Build:

- `Runtime`, `Builder`, `Tensor`, persistent `State`, and caller-owned `Plan`;
- creation/input/weight/state/output ports, concrete shape/dtype inference,
  views, binding validation, lowering, and reported kernel selections;
- contiguous caller-owned KV cache and versioned ScatterRows mutation;
- the exact-shape decode plan; and
- the exact-shape minimal prefill plan needed for parity.

No automatic plan cache, quantization, sampling, production prefill buckets, or
multi-sequence scheduler is in M7.

Done:

- every 007 v0 operator contract has unit and plan-level coverage;
- a two-layer fixed-weight model produces committed higher-precision reference
  logits on CPU and Metal within its composed budget;
- incremental decode for N tokens equals the minimal prefill of the same N
  tokens within the composed Attention/model budget on both backends;
- retained plan replay, state hazards, binding errors, memory reports, selection
  reports, and one-in-flight rejection are covered; and
- E2E: caller allocation of weights/KV/input/output → compile explicit prefill
  and decode plans → prefill → repeated decode → logits readback.

This completes the v0 proof in 000. It is an unquantized correctness milestone,
not a production model-runtime claim.

### M8 and later

Independently scoped later work includes:

- a quantization representation/numerics spec followed by quantized Rows/GEMM;
- production prefill bucketing and, only then, an optional stable-identity plan
  cache;
- textures/formats and graphics;
- Vulkan plus the SPIR-V emitter, then remaining backends;
- sampling primitives and policy integration; and
- paged KV, multi-sequence scheduling, and additional transient sets.

Vulkan is the first backend priority because it gives the CPU oracle a second
vendor/API opinion and pays the cost of the real SPIR-V IR.

## Risks and retirement tests

| Risk | Retired by | Failure response |
| --- | --- | --- |
| Compiler scope is underestimated | M2's direct flat E2E and explicit IR/intrinsic decisions | Split M2; do not hide compiler design in M3/M4. |
| The cooperative resumable transform is larger than one milestone | M4's flat-versus-cooperative agreement and diagnostic gates | Split M4 again; do not fold the remainder into M5's GEMM. |
| MSL cannot meet exact/contraction or primitive ceilings | M6 probes before other Metal numeric tests | Change lowering/domain or reject primitive; never widen from observation. |
| Uniformity analysis rejects correct cooperative code | M4 negative/positive corpus | Specify a CPU-checked assertion intrinsic in a later scoped change. |
| Graph aliasing is unsound | M3 naive-plan fuzz and diamond golden | Block later milestones until fixed. |
| Metal objects outlive autorelease ownership incorrectly | M6 close/completion stress E2E | Fix retain-set ownership before backend acceptance. |
| Tensor state mutation escapes graph hazards | M7 versioned-state negatives and prefill/decode parity | Fix State lowering; never add an untracked in-place escape hatch. |
| CPU oracle has no second opinion | Vulkan after v0 | Keep strict portable mode conservative and state the limitation. |

## Maintenance rule

Once implementation starts, its definition of done is not rewritten to match
what happened. Split a milestone or record a scoped deviation. On completion,
append its date and actual outcome, update the owning specs, and keep this file as
the historical build order rather than a second source of behavioral truth.
