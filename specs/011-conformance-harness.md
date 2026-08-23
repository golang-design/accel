---
title: "Conformance harness: profiles, comparisons, fuzzing, E2E, and coverage"
status: in progress
layer: process
depends_on:
  - 001-device-resources.md
  - 002-compute-model.md
  - 003-command-graph.md
  - 004-kernel-authoring.md
  - 006-backends.md
  - 007-tensor-layer.md
  - 008-numerics.md
  - 010-kernel-corpus.md
---

# Conformance harness

This spec owns the shared machinery assumed by every implementation spec:
device discovery and modes, backend profiles, exact and derived-bound
comparisons, capability-present/absent execution, generated corpus coverage,
fuzzing, plan goldens, E2E scenarios, diagnostics, and coverage gates.

It is test infrastructure, not a public runtime API. It grows from M1 onward and
must not become a second implementation of device or tensor semantics.

## 1. Package layout

```
internal/conformance/
  device/       discovery, profiles, modes, skips            [M1]
  numeq/        008 comparison API and budget traces          [M1]
  cover/        per-package coverage under section 10's rules [M1]
  direct/       flat kernel execution with no device               [M2]
  oracle/       exact/high-precision references and committed corpora
  graphcheck/   naive plan, plan normalization, graph fuzz
  kernelcheck/  generated 010 manifest and per-variant runner
  scenarios/    public-API E2E scenarios
```

A bracketed milestone means the package exists. Everything else is a name
reserved for the milestone that needs it, so that a later increment adds a
directory rather than arguing about where something goes.

`cover/` was not in the first draft of this list and is here because §10 needs a
program to be a checked rule rather than a claim. It ships with `covercheck`, a
command CI runs after `go test -coverprofile`, and it is the only entry here
with an executable: the rest are libraries the tests call.

`direct/` runs a generated flat kernel over a grid with no device, which is what
makes [004](004-kernel-authoring.md)'s fifth testing level possible before
graphs exist at M3. It refuses a cooperative kernel rather than inventing an
order: those invocations rendezvous, so running them in sequence is a different
program rather than a slower one. Whether it survives M3 is
[012](012-kernel-pipeline.md)'s open question.

Allocator fuzzing is in `internal/alloc` beside the allocators rather than here.
§13's M1 line names it because it is an M1 obligation, not because it is harness
machinery: a fuzz target belongs with the thing it fuzzes, and moving it here
would separate a failing seed from the code that has to explain it.

E2E scenarios are in the package whose public API they exercise until there is
more than one such package. `scenarios/` arrives when 007 gives the tensor layer
a second one.

Production packages expose only the introspection already required by their
specs (`Capabilities`, graph plan statistics, plan ports/selections). The harness
may use test-only constructors to force CPU capability profiles and numeric
classes, but GPU behavior is measured through public APIs.

## 2. Device matrix and profiles

```go
// device.Profile, as built at M1. Driver, Arch, and Numeric are named by the
// draft below and are not here yet: the first two have no answer on the CPU
// backend and arrive with Metal at M6, and NumericProfile is 008's and arrives
// with the numeric probes at M4.
type Profile struct {
	Backend      accel.Backend
	DeviceName   string
	Mode         Mode
	Targets      []accel.Backend // the strict target set, when Mode is StrictPortable
	Capabilities accel.Capabilities
	Limits       accel.Limits
}

type Mode int
const (
	StrictPortable Mode = iota
	Permissive
	Mimic
)
```

Two fields the draft did not have. `Targets` is required rather than optional:
"strict" alone does not say what a kernel that builds under it is portable to,
so a profile that cannot name its target set cannot report what a failure means.
`Limits` is here for the same reason `Capabilities` is, since §3's forced
profiles lower both.

Every test receives a profile explicitly. Logs and failures include the complete
identity, mode, capabilities relevant to the case, numeric probe revision, and
kernel source hashes. A result without that context is not actionable.

The CPU backend is mandatory on every platform. Hardware backends are enumerated
and run when the milestone/CI tier requires them. Absence is a skip only when the
job does not promise that backend; a dedicated Metal job with no Metal device is
a failure.

Strict mode enforces the portable intersection, including subnormal policy and
forced capability absence. Permissive mode exposes the host's natural behavior.
Mimic mode takes an explicit target profile and emulates its limits/capability
set. Tests never compare outputs produced under different modes.

## 3. Capability-present and capability-absent cases

Every capability-gated path declares two cases in the manifest:

1. present: requirements are satisfied and the selected implementation runs;
2. absent: CPU strict/mimic mode removes the capability and either selects the
   required portable implementation or returns the specified compile error.

The harness asserts both selection and result. A fallback result without proof
that the fallback was selected is insufficient. Fused Attention is not a device
capability: its present/absent cases add/remove the registered kernel variant
while leaving the device profile unchanged.

Forced profiles may only remove or lower capabilities/limits. They cannot claim
hardware support the CPU implementation does not emulate. Each override is
scoped to one test and restored automatically.

## 4. Comparisons

The `numeq` comparison package implements 008's explicit context API. It has no function
that accepts an arbitrary absolute/relative tolerance.

Required comparison forms:

- exact bits for a proven `(class, domain, profile)`;
- Special categories and canonical conversion bits;
- primitive ULP/absolute ceilings against committed high-precision references;
- sequential/tree reduction bounds from actual K/depth and magnitudes;
- expression/operator composition with a full budget trace; and
- structured integer/shape/plan equality.

Array comparisons report the first failing index plus maximum observed error and
its index. Bounded failures include got/reference bits, absolute/ULP distance,
budget, every contributing budget term, and the source kernel/operator.
Comparisons reject mismatched shapes, modes, unproved exact classes, NaN in a
finite-domain case, and a missing oracle record before examining values.

The static conformance check scans comparison call sites and rejects known ad hoc
helpers or numeric tolerance parameters. It permits literals used as inputs,
mathematically exact expected values, and committed bit-pattern corpora.

## 5. Oracle sources

Integer, indexing, layout, graph, and conversion references are independent Go
implementations using exact arithmetic or explicit bit operations.

Floating references are independent of kernel source:

- reductions and dot products use exact rational or at least 256-bit arithmetic;
- primitive transcendental corpora are generated by a pinned higher-precision
  oracle and commit input/reference bits plus generator/version metadata;
- operator references are direct mathematical implementations that expose
  intermediates needed by 008's composition, not translations of tiled kernels;
  and
- model reference logits and their budget trace are generated from the same
  fixed model definition but a separate scalar implementation.

Generated oracle artifacts have a reproducible command and freshness check.
Ordinary CI consumes committed artifacts and does not download an oracle or
depend on a C library, preserving the cgo-free build.

## 6. Kernel runner

010's generated manifest lists every v0 kernel variant, source hash, layout
classes, requirements, numeric recipe, reference case set, and target artifacts.
The runner fails if any required field or case is absent.

It supports three execution stages:

1. M2 direct flat CPU adapter execution;
2. M3+ public CPU graph execution; and
3. M6+ public execution on every accepted GPU backend.

The same logical case and reference feed every stage. Cooperative variants are
ineligible for stage 1.

Stage 1 additionally runs each flat kernel through **the authored Go function
called directly**, alongside the generated flat lowering, and compares them per
[004](004-kernel-authoring.md)'s fifth testing level. This is the only check that
the generated lowering means what the authored source says: since the authored
function is no longer executed by the backend, a mistake in IR construction would
otherwise appear in the CPU runner and every GPU artifact identically and pass
differential execution. Integer and layout kernels compare bits; f32 kernels
compare under the contraction bound, because the generated lowering emits
explicit rounding points the authored function does not. Target acceptance compiles emitted source through the
actual target compiler/driver before differential execution.

## 7. Graph goldens and whole-plan oracle

Plan goldens normalize device-dependent facts. They compare node kind/order,
declared accesses, inferred edges, barrier positions/types, transient
compatibility, aligned sizes, and relative alias decisions. Raw device addresses
and backend-private handles never enter a golden.

The naive oracle plan preserves the same DAG and bindings but disables transient
aliasing and inserts a full legal barrier between every dependent pair. Random
graph generation varies DAG topology, ranges, views, access modes, transients,
slots, copies, and flat/cooperative dispatches as their milestones become
available. Optimized and naive results are compared under the same numeric
budget; plan structure is compared independently.

Every fuzz failure records seed, normalized graph, profile, bindings, and source
hashes. A minimized reproducer becomes a permanent regression test before the bug
is fixed.

## 8. E2E scenarios

Scenarios use public APIs exclusively and own their resources with deterministic
cleanup. Required milestone scenarios are:

| Milestone | Scenario |
| --- | --- |
| M1 | Open CPU → allocate → write → read → close — **done 2026-08-22** |
| M2 | Kernel source → generator → direct flat adapter → checked output — **done 2026-08-22** |
| M3 | Upload → flat Add graph → readback → rebind and replay |
| M4 | Upload → shared-memory tree reduction graph → readback, with every cooperative diagnostic exercised |
| M5 | Upload → portable tiled GEMM graph → readback in strict mode |
| M6 | M5 scenario selected explicitly on Metal — **built 2026-08-23** for the compute half: `Enumerate` finds the adapter, `OpenDevice` opens it by id, and a recorded graph runs upload → dispatch → readback |
| M7 | Allocate model/KV/IO → compile explicit prefill/decode plans → prefill → repeated decode → logits |

M7 additionally runs fused Attention present and absent, incremental-decode
versus minimal-prefill parity, and a two-layer golden model on CPU and Metal.
Each scenario asserts selected backend/kernel IDs and the derived budget, not
only the final value.

## 9. Determinism, concurrency, and failure injection

Determinism repeats identical submissions and compares bits only for classes and
kernels specified deterministic by 008. Atomic float add is excluded and instead
checked against its bound across multiple runs.

Concurrency cases cover safe device/pool/queue operations, recorder misuse under
the race detector, one-in-flight Graph/Plan rejection, rebind during flight,
close while retained by a submission, and fence completion visibility.

CPU test hooks inject allocator exhaustion, closed resources, unsupported
capabilities, collapsed exact classes, invalid indirect counts, undefined-memory reads,
non-uniform barriers, and device loss. Hooks are unavailable in production builds
and every injected condition has an assertion that the intended path was reached.

## 10. Coverage

The gate is greater than 90% statement coverage for each affected production
package on the CPU path, reported independently rather than as one repository
average.

### 10.1 The two checked exclusions, and nothing else

An exclusion is a **mechanical rule a program applies**, never a list someone
maintains. There are exactly two, and each retires itself:

| Excluded | Rule | Retires when |
| --- | --- | --- |
| Generated registration boilerplate | the declaration is in a generated file and is registration rather than behaviour | never; it is permanent boilerplate |
| A design-stage stub | the function body is exactly `panic(ErrNotImplemented)` | the function is implemented |

Generated *executable* adapters and encoders count, because they are the thing
under test rather than the wiring around it.

The stub rule exists because this repository declares its whole API surface
before implementing it, so the design reads as Go and the compiler checks it
(000 decision 1's shape is only reviewable that way). Counting several hundred
unimplemented declarations against the package that implements the first of
them would report a number about how much is left to build, not about how well
the built part is tested, and a gate nobody can pass is a gate that gets turned
off.

The rule is deliberately narrow and syntactic. A body of exactly
`panic(ErrNotImplemented)` cannot hide a branch, cannot drift into doing
something, and stops matching the moment a real body replaces it. A function
with any other body counts in full, including one that validates its arguments
before panicking, since that validation is behaviour someone should test.

The exclusion is reported, never silent: every run prints how many declarations
each package excluded, so a package whose coverage is high because most of it
does not exist yet is visible as exactly that. A milestone's completion is
judged on the covered number *and* the excluded count going to zero for the
packages it owns.

Branch-oriented manifests supplement statement coverage:

- every validation-table row has a negative test;
- every reported capability has present/absent coverage where applicable;
- every 010 semantic/variant/layout class has a case;
- every error-budget rule and Special category has a case; and
- every public feature appears in an E2E scenario.

CI fails on a missing manifest obligation even when line coverage remains above
90%.

## 11. CI tiers

- Tier 1, every commit: CPU on Linux/macOS/Windows, strict mode, unit/integration
  tests, race where supported, fuzz seed corpus, coverage, formatting, vet,
  generation freshness, and cgo-free checks.
- Tier 2, blocking when its backend milestone lands: Metal on macOS and later
  explicitly provisioned software drivers; full entry gate and relevant E2Es.

  **Built 2026-08-23** as `.github/workflows/ci-metal.yml`, separate from
  `ci.yml` rather than a row in its matrix. The rule below is why, and it cuts
  both ways: Tier 1 runs the darwin device tests too and they must *skip* there,
  because that job promises only the CPU backend and a hosted macOS runner
  without a GPU must not turn it red. What carries the promise into the tests is
  `ACCEL_REQUIRE_METAL`, which Tier 2 sets and Tier 1 does not; with it set,
  every one of those tests fails rather than skipping.

  A developer on a Mac gets the failure without setting anything, since a device
  is there. The environment variable is the *promise*, not the capability.
- Tier 3, non-blocking: scavenged/heavy browser or platform integrations.
- Tier 4, nightly/manual: real multi-vendor hardware and extended fuzz/probe
  corpora.

A backend-specific blocking job must use an OS-provided driver or an explicitly
installed pinned package. A skipped promised backend is a failure.

## 12. Diagnostics and artifacts

CI retains failing normalized graphs, fuzz seeds, generated shader/source,
backend profiles, numeric probe reports, budget traces, and E2E logs. Successful
runs publish compact maximum-error and coverage summaries without treating
observed maxima as normative tolerances.

No golden contains unstable device addresses, timestamps, temporary paths, or
unordered map output. Stable ordering is part of every serializer.

## 13. Implementation sequence

- M1: device runner, exact bytes, profiles, allocator fuzz, coverage. **Done
  2026-08-22.** `device` runs every case under a named profile, `numeq` compares
  exactly and by float encodings, `cover` gates each package independently, and
  the allocator fuzz lives with the allocators. The comparison package has no
  tolerance parameter and will not grow one; §4's derived-bound forms arrive at
  M4 as new functions, each naming its budget.
- M2: compare context, flat runner, generator negatives/freshness.
- M3: normalized plan goldens, naive oracle, graph fuzz, replay E2E.
- M4: numeric probes/budgets, cooperative diagnostics, reduction budgets.
- M5: dot-product budgets and the GEMM E2E.
- M6: target acceptance, Metal profiles, cross-backend entry gate.

  **What that turned out to mean, 2026-08-23.** Target acceptance is the device
  compiler itself: `newLibraryWithSource:` compiles MSL at runtime, so every
  emitted kernel is accepted or rejected by the compiler that will run it, which
  is stronger than the parse this spec could have settled for. The offline
  toolchain is not installed and is not needed.

  The cross-backend entry gate is a **differential over the whole corpus**: all
  29 kernels run on both backends from one generated record, 22 agreeing bit for
  bit and 7 within a ceiling taken from [008](008-numerics.md) §6 for the
  bounded primitive each reaches. A kernel reaching none keeps an exact
  comparison, which is what stops a ceiling spreading. The case table is checked
  against the generated corpus list, so a kernel added and never compared is an
  error rather than a silent gap.

  One thing the harness had to learn: **the oracle is configured to the device
  it checks.** The CPU emulates subgroups at a width a caller chooses, default
  4, while the device executes 32 — so a reduction over 64 elements was two
  different computations rather than a disagreement. Mimic mode exists for this
  (§2), and the differential opens the CPU at the width the device reports.
- M7: operator/model composition, kernel-manifest completeness, tensor E2Es.

The harness is complete for v0 when every M7 manifest obligation is green on CPU
and Metal and no production package included in M1–M7 is at or below 90%
statement coverage.
