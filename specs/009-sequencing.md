---
title: "Sequencing: what gets built, in what order, and what done means"
status: drafted
layer: process
depends_on:
  - 000-decisions.md
---

# Sequencing

Every other spec answers *what*. This one answers *in what order*, and it is the
only spec in this directory that is expected to change as work lands.

It exists because the `depends_on` frontmatter is not the build order and reading
it as one leads nowhere. 002 depends on 001 on paper; in practice 002's semantics
cannot be tested without 004's compiler, 004's oracle cannot run without 001's
pools and 003's dispatch, and 003's own tests need kernels that only 004 can
produce. Written as a graph of documents it looks like a tree. Written as a graph
of things that can be *demonstrated*, it has one root and it is not 001.

---

## 1. The gate nobody drew

**The first testable thing in this project sits behind the kernel compiler.**

The chain, exactly:

1. [006](006-backends.md) §7 makes a tiled GEMM the entry gate for every backend:
   a backend that cannot run it is not finished.
2. [002](002-compute-model.md) §7 writes that GEMM, and it takes a `Dims` uniform,
   because without a uniform the k-loop's trip count is not provably uniform and
   its own barrier rule rejects the loop.
3. [004](004-kernel-authoring.md) makes the uniform's std140 encoder and decoder,
   and the adapter that calls the Go function with unpacked arguments,
   **generated code**.

So the CPU backend cannot run the project's defining kernel until a code
generator exists, and the CPU backend is what every other spec's tests are
written against. That is the single most important fact about the order of work
here, and it was implicit in three specs and stated in none.

It has a corollary that shapes M2 below: the generator needed at that point is
much smaller than the generator described in
[004](004-kernel-authoring.md). One target, no SPIR-V, no capability inference.
The spec's full scope is a destination, not a prerequisite.

---

## 2. The milestones

Each names what it builds, which specs it implements, and **what makes it done**,
where done is a test that exists rather than a judgement.

### M0. The bet, enforced from the first commit

Build: CI tier 1 only. `CGO_ENABLED=0` for every `GOOS`, a grep for `import "C"`
across the module including tests, `gofmt`, `go vet`, and the existing
non-functional surface compiling on all three platforms.

Implements: [`000-decisions.md`](000-decisions.md) decision 2.

Done: the workflow is green and a commit adding `import "C"` anywhere fails it.

Why first, and not later when there is something to test: decision 2 is the
project's reason to exist, and the cheapest moment to make it mechanical is
before any code wants to violate it. It is roughly an hour of work.

### M1. Memory, on the CPU backend

Build: device open and enumeration for the CPU backend, `Limits`, the TLSF
allocator and the linear allocator, pools with kinds and policies, buffers,
views, the lifetime and retain-set machinery, and both transfer paths.

Implements: [001](001-device-resources.md) §§1–8, minus textures.

Done: 001 §11.1 (round trips at every dtype and memory kind), §11.3 (views and
`ViewAs` bit patterns), §11.4 (allocation, including the deliberate fragmentation
scenario and the O(1) guard), and §11.6 (lifetime) all pass. The allocator fuzz
target runs clean.

Note what is deferred inside this milestone: textures and formats wait for M7,
because nothing before the graphics work reads one, and 001 §4 is the longest part
of the spec.

### M2. The kernel compiler, minimum viable

Build: the `go/types` front end, the typed IR, the Go target, the generator and
its `go generate` integration, the std140 encoder and decoder, kernel
registration with the source hash, and the diagnostics discipline (positions,
collected not first-only).

Implements: [004](004-kernel-authoring.md), narrowed: **one target, the CPU's.**
No MSL, no SPIR-V, no capability inference, no uniformity analysis yet.

Done: a kernel with a slice parameter, a uniform struct parameter, and a shared
array parameter is generated and runs on the CPU backend and produces the right
answer; `go generate ./... && git diff --exit-code` fails when a kernel is edited
without regenerating, naming the kernel; and one negative test per rejected
construct asserts both message and position.

**This is the milestone most likely to be underestimated**, and 004 is currently
the thinnest spec in the directory relative to its risk. Before M2 starts, 004
needs the depth its siblings have: the IR's node set, how `accel.Thread` methods
resolve to intrinsics (its own open question 4, which blocks writing a single
kernel), and how accel's internal conformance kernels avoid an import cycle with
the package they import. That is a spec task, not a coding task, and it belongs
before M2 rather than inside it.

### M3. The graph, on the CPU backend

Build: the recorder, slots, edge inference, the reachability interference
relation, the greedy packer, the barrier walk, the validation table, submission,
fences, and the plan-time statistics.

Implements: [003](003-command-graph.md) in full, on one backend.

Done: 003's worked example asserts its exact numbers (22 MiB unaliased, 12 peak,
16 allocated, seven barriers) as goldens; the diamond test proves `t0` and `t3`
are not aliased; one test per row of the validation table; and the whole-plan
oracle fuzz target (every random graph run twice, once under a naive plan with no
aliasing and a full barrier between every pair) finds no disagreement.

The whole-plan oracle is the highest-value test in the project per line of code
and it should be written at the start of M3, not the end. It catches the
interference bug on the first seed that produces a diamond.

### M4. The compute model, and the GEMM

Build: the cooperative and flat execution strategies, barriers with non-uniform
arrival detection, shared memory poison, atomics including the compare-exchange
float add, emulated subgroups, the uniformity analysis on the IR, and the CPU
backend's permissive, strict-portable and mimic modes.

Implements: [002](002-compute-model.md), and the rest of the CPU backend from
[006](006-backends.md) §5.

Done: **002's tiled GEMM runs on the CPU backend, under strict portable mode,
agreeing with an independently written naive reference within
[008](008-numerics.md)'s derived bound, at dimensions that are not multiples of
16.** Removing either barrier fails the suite, the first under `-race` and the
second under the sentinel test. The subgroup sweep at sizes 1, 4, 32 and 64
agrees at every size.

That paragraph is the definition of the project working at all. Everything before
it is scaffolding and everything after it is reach.

### M5. Metal

Build: the Objective-C runtime shim over `purego`, the Metal backend, the MSL
target in the kernel compiler, capability querying, and graph lowering by
re-encoding per submission.

Implements: [006](006-backends.md) §2.2 and §4.3, and the second half of
[004](004-kernel-authoring.md)'s target list.

Done: 006 §7's per-backend entry gate, in its stated order: probe, a capability
report populating every matrix row for the device, a buffer round trip at every
dtype, a barrier-and-shared-memory reduction matching the oracle, and the GEMM.
Plus [008](008-numerics.md) §4.4's contraction probe, which decides how much of
the exact tier survives and should be run before the GEMM rather than after.

Risk concentrated here: object lifetime across completion handlers, which
[`conventions.md`](../docs/conventions.md) records as crashing inside
`objc_msgSend` with a useless stack. 001 §7.1's retain set is the design that
prevents it and M5 is where it is proven.

### M6. The tensor layer, to one token

Build: `Runtime`, `Builder`, `Tensor`, `Plan`, the plan cache, the shape and dtype
inference, layout resolution, kernel selection, and the operator set a decode step
uses.

Implements: [007](007-tensor-layer.md), narrowed to the decode path.

Done: 007's strongest test, **incremental decode equals prefill** — decoding N
tokens one at a time gives the same logits, within bound, as one prefill of the
same N — passes on both backends, plus the small two-layer golden model.

### M7 and later, in no fixed order

Quantization and the quantized GEMM family; textures and formats
([001](001-device-resources.md) §4); Vulkan and the SPIR-V emitter; graphics
([005](005-graphics.md)) and the CPU rasterizer; the remaining backends.

Vulkan is the first of these worth doing, because it is what gives the oracle a
second opinion (see §4) and because it is the backend the SPIR-V argument for
having an IR was made for.

---

## 3. Work no spec owns

Two deliverables are large, are assumed by several specs, and have no design
document. Naming them is most of the fix.

### 3.1 The kernel corpus

[007](007-tensor-layer.md) lists roughly forty operators and marks most of them
primitive, which means each has its own kernel, before multiplying by dtype and by
quantization scheme. [002](002-compute-model.md) writes one kernel.

Under decision 2 there is no cuBLAS and no GGML, so **that corpus is the bulk of
the project's remaining work**, and nothing sizes it, states its layout
conventions, decides which kernels are v0, or says how a kernel is chosen for a
shape. 007's operator table is a requirements list and reads like a design.

A spec is needed before M6 and it should cover at minimum: the v0 kernel list and
what each one's fast and general paths are, the specialization rule (a `Plan` is
compiled for one shape signature, so specialization is pipeline selection at
compile rather than a runtime branch), the naming and file layout that keeps a
corpus of this size navigable, and the per-kernel test obligation from
[008](008-numerics.md).

### 3.2 The conformance harness

[006](006-backends.md) §7 says "there is one suite" parameterized over backends,
[008](008-numerics.md) specifies the comparison API that suite must use, and
[002](002-compute-model.md), [003](003-command-graph.md) and
[001](001-device-resources.md) each list tests assuming a harness exists.

It is a real piece of software: device enumeration and per-device skipping,
capability-gated cases exercised both present and absent, the CPU backend's three
modes, the derived-bound comparisons, golden plan structures compared without
their device-dependent offsets, and the fuzz targets. It should be built
incrementally from M1 and it should never be allowed to accept a hardcoded
tolerance, which is 008 §8's mechanical check.

---

## 4. Risks, ordered by what they would cost

| Risk | Retired by | If it fails |
| --- | --- | --- |
| MSL cannot be stopped from contracting | [008](008-numerics.md) §4.4's probe, in M5 and ideally earlier | The exact tier on the only v0 GPU backend collapses to integers and conversions. The oracle still works; its strongest claim does not. |
| The kernel compiler is larger than 004 suggests | 004 gaining its missing depth before M2 | M2 slips and everything is behind it, since the GEMM is behind M2. |
| The uniformity analysis rejects too much | M4, and 002 §3.3 already names two false-positive families | Kernels are rejected that are correct, and the escape hatch (`AssumeUniform`, checked on the oracle) becomes v0 work rather than deferred. |
| Metal object lifetime across completion handlers | M5, with 001 §7.1's retain set | Crashes with no usable stack, which is how the predecessor experienced it. |
| The CPU oracle has no second opinion | Vulkan, in M7 | A portability rule that the CPU backend enforces wrongly is invisible: nothing at v0 can contradict it. This is the cost of a two-backend v0 and it does not go away before Vulkan. |

The last row is the one to keep in view, because it is not a schedule risk and
cannot be worked around: 006's oracle rule says the CPU backend enforces the
intersection of what every backend allows, and at v0 it is enforcing an
intersection derived from five backends that are not present to disagree.

---

## 5. How this spec is maintained

It is the one file here that moves. A milestone's definition of done is not edited
once work has started against it: if it turns out to be wrong, the milestone is
split or a new one is added, so the record of what was believed at the time
survives.

When a milestone lands, its entry records the date and what actually shipped
against what was specified. Where the two differ, the sibling spec is corrected;
this spec never becomes the place where the difference between the design and the
code is quietly stored.
