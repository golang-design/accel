---
title: "Completing the public API surface"
status: drafted
layer: project
depends_on:
  - 036-documentation.md
  - 009-sequencing.md
---

# Completing the public API surface

[036](036-documentation.md) §4 reviewed the compute surface and froze it. This
spec is the other half of that job: the audit that followed found declarations a
caller can reach and cannot use, operations the library claims and does not
offer, and a graphics surface that grew to 124 declarations without a review.

**What "complete" means here**, and it is deliberately narrow:

> Every operation the library claims a caller can express, they can express.
> Every exported declaration either works on the CPU backend or is refused with
> a reason naming what is missing and where it is specified.

Not "every backend implements everything". [006](006-backends.md) decision 6
already says absence is reported rather than discovered; this spec makes the
*surface* whole and lets the CPU backend be the one place every operation
exists, which is what makes it the oracle every other backend is checked
against.

## 1. Why surface-first, and why CPU-first

A missing declaration and an unimplemented backend fail a caller differently. A
missing declaration is a *design* answer they cannot get — they restructure
their program around the hole, and the restructuring survives the hole being
filled. An unimplemented backend is a capability query and a fallback path,
which is a shape they already have.

So the order is: name the operation, implement it once on the CPU backend, and
let every other backend report absence until it has one. The alternative — hold
a declaration back until two backends have it — is what produced
`tensor.Contiguous`: an operator named in four error messages, one of which
tells a caller to insert it, and which does not exist.

## 2. The inventory

Three verdicts. **Land** means the declaration arrives and works on the CPU
backend. **Refuse** means it stays absent and something names why. **Remove**
means it exists and should not.

### 2.1 Land

| Surface | Owner | Note |
| --- | --- | --- |
| `tensor.Contiguous` | [025](025-tensor-operators.md) §7 | needs a gather kernel; four sites already name it |
| `ErrNoAdapter`, `ErrPolicy` | [006](006-backends.md) | selection failures a caller branches on |
| A workgroup-count helper | [002](002-compute-model.md) | every tutorial computes `(n+63)/64` by hand |
| `Requirements.SharedBytes`, populated | [016](016-graph-execution.md) | V11 is stated and cannot fire |
| Timestamps | [003](003-command-graph.md) §9 | "no timing observability at all" on a throughput library |
| Line and point rasterization | [035](035-cpu-rasterizer.md) §10 | three of five `Topology` values are refused |
| Subgroup shuffles and scans | [020](020-atomics-subgroups.md) | intrinsics a kernel author cannot call |
| Texture attachments | [033](033-render-api.md) | attachments are buffer views "at this milestone" |
| Texel fetch in a stage | [032](032-stage-abi.md) §5 | also unblocks 033's feedback rejection |

### 2.2 Refuse, with the refusal naming the reason

| Surface | Refused because |
| --- | --- |
| `Ballot` on Metal | `simd_ballot` returns `simd_vote`; [022](022-msl-target.md) §5 |
| Uniform-block array members | std140's 16-byte stride needs the index expression rewritten |
| Non-Metal window handles | [034](034-surface-present.md) §6 lists them; no backend has one |
| MSAA | [041](041-msaa.md), unbuilt |
| `MipLevels`, `ArrayLayers` > 1 | the API cannot name a subresource yet |

A refusal is not a gap when it names what is missing and where the design is.
It is a gap when a caller has to discover it.

### 2.3 Decide

Two declarations exist whose *existence* is the open question, and answering by
building the missing half would answer it by default.

| Surface | The question | This spec's answer |
| --- | --- | --- |
| `UniformBuffer[T]` | [001](001-device-resources.md) §10, [014](014-kernel-uniforms.md) §7: should uniform buffers exist at all, given that std140 padding is the only thing they buy? | **Keep and connect.** §3.1 |
| `Device.Queues`, `QueueFor` | provisional in 036 §5.2 until a backend reports more than one queue | **Keep.** No backend reports two; the surface is what a second one would need, and withdrawing it would break every caller that branches on it when one arrives. |

### 3.1 Uniform buffers: keep, and connect

The argument for removing them is that a by-value parameter already carries a
std140 block, so a uniform *buffer* adds a resource kind for no expressive gain.

The argument for keeping them is [033](033-render-api.md) §6's N-object frame,
and it is decisive: a thousand objects with a fixed byte offset each is
precisely what a by-value parameter cannot express, because the value travels
with the node and a thousand values per frame is the cost the offsets exist to
avoid. Removing uniform buffers would remove the only mechanism that case has.

So they stay, and the missing half — a dispatch that reads one — is built.

## 4. The `Kernel` shape change

`Requirements.SharedBytes` is never populated, so validation rule V11 — a
kernel's shared-memory request against the device budget — is stated and cannot
fire. `Kernel.SharedSizes` holds element *counts* with no element size, so the
byte figure cannot be computed from what the record carries.

The fix adds a field the generator emits. [016](016-graph-execution.md) deferred
it because `Kernel`'s shape was an open freeze question and changing it under a
review was the wrong order. The review has happened;
[036](036-documentation.md) §5.3 moved the generated-code ABI to `kernelabi`
and made `Kernel` a record the emitter constructs. Changing it now costs one
`go generate`.

## 5. The graphics review

[036](036-documentation.md) §5 excluded graphics: *"there is no render API in
code"*. There is now, and it is the largest part of the surface. It gets §4's
pass — inconsistent naming, asymmetries, undocumented required steps, anything a
tutorial would apologise for — and the freeze record gains a graphics section
rather than continuing to say graphics is not in it.

**Evidence it is not settled**, and the reason the review comes before any
tutorial teaches it: four API-level corrections on its first day (a uniform
channel removed and redesigned, a store action reaching nothing, a buffer-index
collision, a resize leaking a drawable) and three more on its second (a package
doc claiming graphics was absent, `Recorder.Build` documenting itself with a
signature, and two capability flags reporting the opposite of the truth).

## 6. The documentation guard

[036](036-documentation.md) §3.1 requires tutorial code to live in `Example`
functions so `go test` compiles it and *"a tutorial cannot drift from the API"*.
There are none, the eight tutorials are fenced blocks nothing compiles, and the
drift has already happened: `docs/tutorial/04-graphs.md` calls `g.Rebind`, which
§5.3 replaced.

The guard is built here, because every declaration this spec lands is one more
thing a tutorial can drift from.

## 7. Done

- `tensor.Contiguous` packs an arbitrary strided view, and the four error
  messages that name it are true;
- a dispatch reads a `UniformBuffer` at a recorded offset, and an N-object
  replay rewrites contents without re-recording;
- V11 fires: a kernel requesting more shared memory than the device reports is
  refused at build, naming both numbers;
- a submission reports elapsed device time when asked, and costs nothing when
  not;
- lines and points rasterize, and their rules are stated in
  [035](035-cpu-rasterizer.md) §10 rather than guessed;
- a subgroup shuffle and a scan run on the CPU backend and are refused with a
  named reason on a backend without them;
- a render pass writes a texture attachment, and a stage reads a texture;
- every tutorial's code is an `Example` that `go test` compiles;
- the freeze record covers graphics; and
- no exported declaration is reachable-and-unusable without a refusal naming
  what is missing.

## 8. Open questions

- **Whether the CPU backend should implement everything.** It is the oracle, so
  an operation it cannot perform is one no differential can check. That argues
  for yes. Against: a CPU implementation of a hardware feature can be a
  caricature — a "subgroup" of one lane proves nothing about a real one. The
  line taken here is that the CPU backend implements the *semantics* and the
  conformance corpus states what a caricature cannot prove.
