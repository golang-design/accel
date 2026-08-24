---
title: "Completing the public API surface"
status: in progress
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
| ~~`tensor.Contiguous`~~ **done** | [025](025-tensor-operators.md) §7 | four sites named it; one told a caller to insert it |
| ~~`ErrNoAdapter`, `ErrPolicy`~~ **done** | [006](006-backends.md) | selection failures a caller branches on |
| ~~A workgroup-count helper~~ **done** | [002](002-compute-model.md) | every tutorial computed `(n+63)/64` by hand |
| ~~`Requirements.SharedBytes`~~ **done** | [016](016-graph-execution.md) | V11 was stated and could not fire |
| ~~Timestamps~~ **done** | [003](003-command-graph.md) §9 | whole-submission device time; per-node still needs the planner to stop merging boundaries |
| ~~Subgroup shuffles~~ **done** | [020](020-cooperative-atomics.md) §6.4 | `Broadcast`, `Shuffle`, `ShuffleXor`, `ShuffleUp`, `ShuffleDown`, on both backends |
| ~~Subgroup scans~~ **done** | [020](020-cooperative-atomics.md) §6.5 | `SubgroupInclusiveAddF32` and `SubgroupExclusiveAddF32`, on both backends |
| Line and point rasterization | [035](035-cpu-rasterizer.md) §10 | needs a measurement first — see below |
| Texture attachments | [033](033-render-api.md) | attachments are buffer views "at this milestone" |
| Texel fetch in a stage | [032](032-stage-abi.md) §5 | also unblocks 033's feedback rejection |
| A draw at a recorded uniform offset | [033](033-render-api.md) §4.1 | the half `UniformBuffer[T]` is missing; see §3.1's correction |

Line and point rasterization moved out of this table. Landing it means *stating*
a rule [035](035-cpu-rasterizer.md) §10 deliberately leaves open, and a rule the
CPU states that Metal does not follow makes the differential fail on lines
forever. It needs a measurement of what the hardware already does, not a
decision here.

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
| `UniformBuffer[T]` | [001](001-device-resources.md) §10, [014](014-kernel-uniforms.md) §7: should uniform buffers exist at all, given that std140 padding is the only thing they buy? | **Keep and connect.** §3.1 — the dispatch path is not yet built |
| `Device.Queues`, `QueueFor` | provisional in 036 §5.2 until a backend reports more than one queue | **Keep.** No backend reports two; the surface is what a second one would need, and withdrawing it would break every caller that branches on it when one arrives. |

### 3.1 Uniform buffers: keep, and connect

The argument for removing them is that a by-value parameter already carries a
std140 block, so a uniform *buffer* adds a resource kind for no expressive gain.

The argument for keeping them is [033](033-render-api.md) §4.1's N-object frame,
and it is decisive: a thousand objects with a fixed byte offset each is
precisely what a by-value parameter cannot express, because the value travels
with the node and a thousand values per frame is the cost the offsets exist to
avoid. Removing uniform buffers would remove the only mechanism that case has.

So they stay, and the missing half is built.

**Correction, 2026-08-24: the missing half is a *draw*, not a dispatch.** This
section said "a dispatch that reads one", and the argument above does not
support that. What the N-object frame needs is a **draw parameterised by a
recorded byte offset** into a uniform buffer — 033 §4.1's
$\text{offset}(i) = i \cdot \text{align}(\text{sizeof}(T), \text{minUniformBufferOffsetAlignment})$
— and 033 deviation 1 removed the draw-time uniform channel that would have
carried it, replacing it with per-stage pass state that is one value per pass
rather than one per draw.

A compute dispatch has no equivalent need: its by-value parameter already
carries a std140 block, and nothing has asked for a thousand of them.

So this item is **graphics-side**, it is blocked behind the same review as every
other graphics item, and it is one instance of the defect class this project
found seven of in its own surface: `UniformBuffer[T]` is exported, allocates,
encodes, and hands back a `BufferView` that no draw can be parameterised by.
The type is not wrong; the half that would use it was removed and not replaced.

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

### 5.1 What a consumer's reports changed

Seven arrived from a team building an inference framework while this spec was
being worked. Four were one decision, which [043](043-per-row-values.md) names,
and they took priority over this list: a report from someone building on the
library is evidence about the surface that no audit of it produces.

They also demonstrated the difference between the two kinds of gap this spec
distinguishes. `tensor.Contiguous` was on this list from an audit. The RoPE
scalar was not on any list, because nothing about the *surface* looked wrong —
it looked wrong only to someone batching two sequences.

### 5.2 The review, done — 2026-08-24

It found more than §5 expected, and the finding that matters is not a count.

#### One decision, many symptoms

**A render attachment is a `BufferView`, not a texture.** `render.go` states it
plainly — *"Attachments are buffer views rather than textures at this
milestone… The shape a caller writes does not change when that lands."*

**The last sentence is not true, and correcting it is the point of this
section.** `ColorAttachment.View BufferView` cannot name a mip level, an array
layer, or a format, and [033](033-render-api.md) §3.3's whole feedback-rejection
design turns on comparing subresources. The shape does change.

Eight of the largest findings are that one decision's consequences rather than
independent defects:

| Consequence | Why it follows |
| --- | --- |
| `ColorTargetState.Format` and `DepthStencilState.Format` reach no backend | a `BufferView` carries a dtype, not a format, so there is nothing to lower into. Metal hardcodes `RGBA32Float` per attachment; the CPU backend reinterprets attachment bytes as `[]float32` unconditionally |
| The per-device `FormatInfo` table is built and thrown away | the render path reads two of its twelve fields; `Blendable` is populated and read **nowhere** |
| Check **V13** is unimplementable in this shape | the same category as the withdrawn V23, and unlike V23 it was not marked |
| sRGB never converts | [035](035-cpu-rasterizer.md) §5 says it converts on write and on read; the rasterizer explicitly declines format encoding, and no layer owns it |
| No stage can sample or fetch a texture | so no pass reads a previous pass's output: no deferred shading, no shadow maps, no material textures |
| 033 §7's feedback rejection is blocked | it needs a subresource to compare |
| `LoadKeep` costs a copy in and a copy out **per pass** on Metal | a buffer is not an attachment, so it is staged into one and back |
| Presenting costs a full-screen conversion draw **every frame** | RGBA32Float is not a compositor format and a blit cannot convert |

**So the cost is not a missing feature, it is a frame's cost model.** At 1080p a
Metal frame makes several full-frame round trips through system memory at four
times the natural byte width, and none of that is visible as a failing test.

#### The verdict

**Not ready to build features on.** Two reasons, in order of weight.

*First, the attachment change is not additive.* Everything built on the buffer
attachment is built to be rebuilt: `Format` fields become real, V13 becomes
implementable, sRGB becomes possible, feedback rejection unblocks, the per-pass
blits and the present conversion draw disappear, `Renderable` and `Blendable`
acquire consumers, three `TextureUsage` flags acquire meaning, and the sampler
block acquires a shape. §2.1 already listed texture attachments and texel fetch
as work to land. What the review adds is that they are not two more items on
that list — **they are the item everything else is downstream of.**

*Second, the defect density matches the compute half before its consumer
arrived.* Eight declarations accepted that reach nothing, four implemented-and
unreachable paths, eleven pieces of spec/code drift, and one panic. **Every one
passed every gate.** Four resemble what the inference consumer filed closely
enough to name:

- `Stage.Discards` is declared in the IR *and* in the stage record, guarded in
  the emitter, and **assigned nowhere in the compiler** — three layers of
  machinery for a value that is permanently false. The ninth instance.
- The stencil pipeline is fully implemented and tested in `internal/raster` and
  is reachable by nothing, `DepthStencilState` having no stencil fields. Worse,
  the CPU backend allocates its stencil buffer per pass, so the cross-pass case
  the rasterizer's own test exercises could not work even if the state were
  reachable. This is `AttentionDecodeBatched`'s shape exactly.
- 033 §7's Outstanding table does not list the attachment-format check, so **by
  the spec's own accounting it is done**. It is not, and cannot be.
- 033 §6 rules out push constants, and the shipped by-value channel is Metal's
  `setVertexBytes`, baked at record — the exact mechanism and cost §6 rejected —
  while the mechanism §6 chose instead cannot be bound to a render pass at all.
  §3.1 above then kept `UniformBuffer[T]` on the strength of that argument.

The last one is the pattern that should decide it: a spec argued a design, a
second spec relied on the first's argument, and neither noticed that the render
path implements the option the first rejected and cannot implement the one it
chose. **That is not a defect anyone finds by adding a feature.**

#### What is right, and must not be disturbed

Named because a review that only lists faults invites throwing out the good with
them. Each of these was independently confirmed as correct by looking at a
renderer that did *not* do it and paid for it:

- **one clip-space convention**, with backends folding the remap;
- **`FrontFace` with no backend-default zero value**, so winding is stated;
- **author-once stages**, where the differences between hand-maintained GLSL and
  MSL copies of one shader turn out to be precisely the clip-z and winding
  conventions this library normalizes;
- **a real barrier and aliasing model**, against `WaitIdle` after every dispatch;
- **pass-granularity synchronisation**, store actions at all, and the per-device
  `FormatInfo` design — which is the right shape even though the render path
  does not ask it.

### 5.3 Three forces shaping the surface that are not the abstraction

Ranked by damage. These are design findings rather than defects, and they are
the reason the review asked more than "does every declaration reach something".

**The CPU oracle is setting the public API's ceiling.** Vertex attributes admit
only float32 vector formats, and the stated reason is that normalized integer
formats "convert on fetch, which is a conversion the CPU rasterizer would have
to match bit for bit to stay an oracle". Every one of the four target APIs has
normalized integer vertex formats. The abstraction is narrower than *all* of its
targets for a reason about testability, and §8's open question — whether the CPU
backend should implement everything — has therefore already been answered by
default, in the direction of a smaller API, without anyone recording the
decision. The constraint is right and the conclusion is too strong: an
unorm8 fetch has one obvious correct definition and snorm16 has two that every
API documents. State the conversion the way the fill rule is stated. Reserve the
refusal for what genuinely has no portable definition — lines, points, and
filtered sampling.

**Metal's ABI is refusing callers on devices that are not Metal.** A pipeline
with more than sixteen vertex buffers is refused on *any* device because a
stage's uniforms begin at index sixteen **on Metal**. That is a device limit
wearing a constant's clothes; report it as `Limits.MaxVertexBuffers` and name
the device's number in the error. Decision 6 already says absence is reported
rather than discovered, and a constant in `internal/mslabi` is neither.

**The barrier vocabulary has no graphics in it.** `stage` is a two-bit mask —
transfer and compute — with a comment saying graphics adds the rest. It has not,
so every render pass is classified as transfer by fallthrough. This is inert
today because the CPU backend has no barriers and Metal tracks hazards
automatically, which is exactly why nothing caught it. A Vulkan backend needs
six stages to place a barrier correctly. **Widen the mask before the first
non-Metal backend**, because widening it changes every inferred edge and every
hazard the graph reports, and doing that under a working bring-up means
re-validating the whole barrier corpus at the worst possible moment. The
accesses are already distinguishable from what a pass records — a vertex-buffer
read, an indirect-argument read and an attachment write are three different
things today, all flattened into one bit.

This is also the sharpest irony in the review: the barrier inference model is
this library's biggest advantage over the field, and it is the one part of the
design that graphics did not get.

### 5.4 One thing to write before the feature, not after

A renderer's hardest-won convention lesson is that texture origin is not per
*backend* but per *resource kind*: on GL, render-target readback is bottom-origin
while compute storage output is not — which is why such a bug survives its own
tests, since the pass that reads storage output never sees the flip.

accel has never met this because it has exactly one resource kind on the render
path: the attachment is a buffer. **The day texture attachments land, accel
acquires a second resource kind and this bug becomes available.** The corpus
entry that compares a texture-attachment readback against a storage-buffer write
of the same image is therefore written *before* the feature.

**Written 2026-08-24, and it skips.** Writing it found something the review did
not: **the texture path has no GPU comparison at all.** Every texture test in
the root package runs on the CPU device, and Metal does not lower a texture copy
— it refuses by name, pointing at [021](021-metal-bringup.md). So the one
convention `docs/conventions.md` records as *known* to diverge between backends
is covered by nothing, and has been since textures were added.

The entry is therefore built to self-activate. Its oracle half asserts today
that the CPU backend returns caller order; its Metal half skips with the
refusal's own text as the reason, and starts comparing on the first day that
refusal stops being returned. When it fails it names the row the byte actually
came from and says whether that is a clean vertical flip or a shear, which is
the measurement `docs/conventions.md` says localises this fastest.

A skipped test that says what it is owed beats a comment saying the same thing,
because the skip is attached to the code that will make it run.

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
