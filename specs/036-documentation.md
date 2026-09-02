---
title: "Documentation: who each document is for, and the tutorial deck"
status: in progress
layer: project
depends_on:
  - 000-decisions.md
  - 009-sequencing.md
---

# Documentation

Two problems, one spec. **Voice**: the reader-facing documents are written for
someone building accel rather than someone using it. **Absence**: there is no
tutorial at all, so a newcomer's only path from "what is this" to "I ran
something" is the README's fragments and the package doc.

The first is fixable now. The second is gated, and section 4 says on what.

## 1. Who each document is for

| Document | Reader | They arrive asking |
| --- | --- | --- |
| `README.md` | someone deciding | What is it? Does it fit? What does it cost me? |
| `docs/tutorial/` | someone starting | How do I do one thing, right now? |
| `docs/` (the rest) | someone using | How do I do this specific task? |
| `docs/conventions.md` | someone writing a backend | Where do backends actually disagree? |
| package doc | someone reading pkg.go.dev | What is in this package and is it built? |
| `specs/` | someone changing accel | Why is it shaped this way? What was rejected? |
| `CONTRIBUTING.md` | a volunteer | What would help, and what are the rules? |

The rule that decides every disagreement: **`docs/` is for someone using the
thing, `specs/` is for someone changing it.** Design rationale in a `docs/` page
is misfiled, not merely verbose — and the fix is to move it, not delete it.

## 2. What "builder-oriented" looks like, so it can be spotted

The tells, each of which appears in the current reader-facing set:

- **Spec numbers as the way of saying what is done.** "003's third kind of
  variation", "000's second v0 proof obligation". A user does not know what 003
  is and has no reason to learn.
- **Milestone identifiers.** M0 through M8 are this project's build order. A
  status table organised by them tells a reader what *we* did, in the sequence
  *we* did it, rather than what they can do.
- **Design rationale where usage belongs.** Why an IR exists, why the planner
  uses reachability rather than record order, what the oracle does and does not
  prove. All true, all interesting to a contributor, all noise to someone who
  wants to multiply two arrays.
- **Inward framing.** "We would rather you knew this", "this is our central
  bet", "the predecessor could not do this". The information is often
  user-facing; the framing is a conversation between maintainers.

**What is not a tell, and must survive the rewrite.** Honest limitations are
user-facing and valuable: no CUDA backend, no training, will not beat vendor
libraries, graphics unbuilt. A rewrite that flattens those into marketing is a
worse document than the one it replaced. Change the framing from "our bet" to
"here is what you get and what you do not"; keep every fact.

## 3. The tutorial deck

One aspect each, short, in dependency order. Each teaches **one** concept and
ends with the reader having run something.

**Written 2026-08-24: eight pages, in `docs/tutorial/`.** The deck below was
ten; it shipped as eight, and the two that are absent are absent for a reason
worth recording.

| # | Tutorial | The one concept | Use case |
| --- | --- | --- | --- |
| 1 | Hello, GPU | open a device, dispatch, read back | scaling an array |
| 2 | Writing a kernel | the Go subset and `go generate` | brightening an image |
| 3 | Memory | pools, buffers, views, ordered closing | weights that outlive a frame |
| 4 | Graphs | record once, replay many | a simulation step |
| 5 | Cooperation | shared memory and barriers | a reduction |
| 6 | Values that are not buffers | uniforms, and `SetUniform` | a runtime coefficient |
| 7 | Tensors | shapes and operators | a feed-forward block |
| 8 | Backends | selection, capabilities, testing without a GPU | shipping to machines you do not own |

**A decode step and quantized weights are not written.** §5.6 gates them on
`Contiguous`, the `PlanCache` ownership documentation, `quant`'s naming, and
f16 host uploads. A tutorial for either would spend more space on workarounds
than on the concept, which is the opposite of one aspect at a time. The
package documentation is the current word on both.

**Every page's code is verified, not by `Example` functions.** §3.1 asked for
those, and §4.1's finding is why they do not fit: a kernel-bearing example needs
a generated kernel in its own package, and the only such package in this
repository is `internal/testkernels`, which a reader cannot import. Teaching
from it would show an import that does not work.

So the check is the one the README already uses, and it is stronger than
compilation: every kernel is **extracted from the prose and run through the real
generator**, every host program is extracted and executed from a clean module,
every Go block is parsed, and every `accel.`/`tensor.`/`quant.` symbol named
anywhere in the pages is checked against `go doc`. Twenty-four blocks, zero
unknown symbols.

> **Correction, 2026-08-24.** That check was run **once, by hand**, and this
> paragraph reads as though it repeats. Only the last of the four is automated:
> `docdrift_test.go` checks that every symbol a document names still exists. A
> document can therefore name only real symbols and still be a program that does
> not run, which is exactly what happened — the README's install step omitted
> the generator's tool dependency, so `go generate` failed on a clean consumer
> module and its error pointed at an `internal` package a consumer cannot get.
> Found by extracting the README's program into an empty module and running it,
> which is the check this paragraph describes and nothing performs on a
> schedule. Automating it is worth doing and is not done.

Graphics tutorials are **not** in this deck. [033](033-render-api.md) and
[034](034-surface-present.md) specify a public API that does not exist in code,
and a tutorial for it would be fiction. They arrive with
[035](035-cpu-rasterizer.md)'s implementation.

### 3.1 Every example compiles, because `go test` runs it

Tutorial code lives in `Example` functions, not in fenced blocks a reader
copies and a compiler never sees.

Three reasons, and the third is the one that decides it: `go test` compiles and
runs them, so a tutorial cannot drift from the API; pkg.go.dev renders them, so
a reader on the package page gets them for free; and an `examples/` directory of
`main` packages would score zero under the per-package coverage gate, which
would have to be widened to accommodate documentation — exactly the kind of
exclusion this project refuses elsewhere.

## 4. The gate: tutorials wait on a public-surface review

**A tutorial is the first thing that makes an inconsistent API obvious.** Writing
one against an unreviewed surface means discovering the inconsistency in the
tutorial and then rewriting both, so the review comes first.

The surface today is **266 exported declarations in `accel` and 91 in
`tensor`**, grown across nine milestones without anyone having read it as a
whole. Before tutorial 1 is written, that surface gets one pass looking for:

- naming that is inconsistent between neighbours doing the same job;
- an asymmetry a reader will trip on — a `New` with no `Close`, a getter with no
  setter where both are expected;
- a step a user must perform that nothing tells them to;
- anything a tutorial would have to apologise for or work around; and
- declarations that are internal detail and should not be exported at all.

The output is a written record of what is frozen and what is provisional, so
"the API will change" in the README can become a specific sentence rather than a
blanket disclaimer.

### 4.1 First finding, from writing one example

Writing the README's quick start found one before the review started, which is
the argument for the review in miniature.

**A kernel package cannot be generated for the first time if anything in it
already refers to the generated symbol.** `accel-kernel` type-checks the package
it compiles, so a single-package program with the kernel and its caller together
fails:

```
accel-kernel: accel: ./... did not type-check:
main.go:17:79: undefined: ScaleKernel
```

The symbol does not exist yet, and it cannot exist until the generator runs. The
workaround is to put kernels in their own package, which is good practice
anyway, and the README now says so. But the error tells a first-time user what
failed and not what to do about it, and "the generator cannot run on the code it
is generating for" is a step a user must discover.

Two candidate fixes, neither yet chosen:

| Fix | Cost |
| --- | --- |
| The generator tolerates undefined symbols whose names match kernels it is about to define | it would have to type-check twice, or reason about names before resolution |
| The error names the cause and the fix | cheap, and does not remove the constraint |

This is exactly the class §4 predicts: not a bug, and not visible from inside the
implementation, because everyone who already has generated output never hits it.

**Scope of the freeze.** The compute half only: device, memory, kernels, graphs,
cooperation, and the tensor layer. Those shipped across M1 through M8 and
[000](000-decisions.md)'s two-layer split is locked, so graphics arrives as a
*sibling* at the device layer rather than a modification of it. Waiting for
graphics before documenting compute would leave the library undocumented through
[035](035-cpu-rasterizer.md)'s remaining steps, the stage-ABI compiler work, and
Metal — which is the problem this spec exists to fix, not a solution to it.

## 5. The freeze record — 2026-08-24

§4's output. Five reviewers read the surface by area; every one returned
*freeze-with-fixes*. A symbol is in exactly one tier:

| Tier | Meaning | A tutorial may | Changing it later costs |
| --- | --- | --- | --- |
| **Frozen** | name, shape and behaviour are the contract | teach it plainly | an argument, a deprecation, a release note |
| **Provisional** | correct today, a *named* future change moves it | teach it saying it may move | nothing — that is what provisional buys |
| **Fix before** | wrong now, cheaper to change today than to support | not teach it yet | rewriting every tutorial that used it |

The freeze covers the compute half: `accel`, `tensor`, `quant`, `kmath`.
Graphics is not in it — there is no render API in code, and §5.3 removes the
stage types until there is.

> **Correction, 2026-08-24.** The reason above expired. There is a render API in
> code — pipelines, passes, attachments, draws — and Metal runs it, compared
> against the CPU rasterizer pixel by pixel. Graphics is still outside the
> freeze, and now for a reason that is written down rather than assumed:
> [042](042-surface-completion.md) §5.2 reviewed it and found the attachment
> model must change **non-additively**, so anything built on today's shape is
> built to be rebuilt. [045](045-texture-attachments.md) is that change.
>
> This matters for a release rather than only for tidiness. "Not frozen because
> it does not exist" and "not frozen because it is being replaced" are different
> promises to a caller, and only the second one lets them decide whether to
> wait. §5.2's own rule — *provisional names the event* — applies to the whole
> half: the event is 045.

### 5.1 Frozen

Devices and selection (`OpenCPU`, `OpenDevice`, `OpenBest`, `Policy`, `Backend`,
`Enumerate`, `DeviceInfo`, `Limits`, `Capabilities`, `CPUMode`); pools, buffers
and views (`NewPool`, `Pool`, `Buffer`, `BufferView`,
`PoolStats`, `MemoryKind`, `BufferUsage`, `DType`); graphs (`NewRecorder`,
`Recorder`, `Graph`, `Slot`, `Queue`, `Fence`, `TransientPool`, the error
types); and `tensor`, `quant`, `kmath` in full apart from §5.6's exclusions.

### 5.2 Provisional, with the reason each one moves

| Surface | Moves when |
| --- | --- |
| `Device.Queues`, `Device.QueueFor`, `QueueKind` | a backend reports more than one queue |
| Texture creation and readback | the memory-kind story settles beyond `TextureDescriptor.Kind` |
| `NodeRenderPass` and the node kinds after it | render passes land |
| `Binding`'s uniform and texture fields | uniforms get their own dispatch argument |

"Provisional" is not a hedge. Each names the event, so a caller can decide
whether it affects them.

### 5.3 Fix before the freeze — **done 2026-08-24**

All twelve landed. Three were not the naming questions they looked like:

| Item | What it actually was |
| --- | --- |
| 10, `quant.Error` | it indexed `scales[i/Int8Block]`, a bound only where a dot product's terms are contiguous — true for a row, false for a column. The new signature takes one scale per term and panics when the counts disagree; it caught this repository's own test on the first run. |
| 12, `AdapterID` | two open CPU devices reported equal `Info().ID` and disagreed about what they could run, because the token is a constant while `CPUStrict` resolves different capabilities. [007](007-tensor-layer.md) keys its plan cache on that identity. |
| 7, `Buckets` | the literal `Buckets{512, 128, 256}` is unsorted and `For` searches, so a size-100 request returned 512. Sealed behind `NewBuckets`. |

And item 1 exposed a migration hazard that was not on the list at all: the
generator type-checks the package it compiles, and its own previous output is
part of that package — so a generated file that no longer compiles after an ABI
change makes the package fail to load, which makes the generator refuse, which
is the one command that would have fixed it. Every downstream user of this
release would have hit it. The file is now overlaid with its own package clause
during the load: hidden from the type checker, left on disk, rewritten.



Every one is breaking, and each costs a `go generate` or a `gofmt -r` today
against a deprecation cycle and a tutorial rewrite later. The two that matter
most:

1. **Move the generated-code ABI out of the root package.** About thirty
   symbols exist only so generated code compiles: `KernelArgs`, `KernelFrame`,
   `KernelMask`, `KernelDType` and the rest. They outnumber the names a tutorial
   teaches, in the same pkg.go.dev index. Moving them to
   `golang.design/x/accel/kernelabi` — and making `Kernel` an opaque handle the
   emitter constructs rather than a literal it fills — is one `go generate` for
   every caller, and `accel-kernel -check` already fails CI on a stale file.
   Freezing instead commits the library to `Kernel.MSL`, a Metal source string
   on a backend-neutral type, and to twelve mutable fields on a package-level
   var, forever.
2. **Split `Binding`.** `Binding.Index` means the pipeline's binding layout,
   except when `Uniform` is set, when it means the kernel's by-value list — and
   neither matches the authored parameter position a reader is looking at.

The rest are naming and constructor consistency, listed in the review output.

### 5.4 The sentence this replaces

The README's "the API will change" becomes, once §5.3 lands:

> The compute API is frozen: devices, pools, buffers, views, kernels,
> pipelines, graphs, queues, fences, and the `tensor`, `quant` and `kmath`
> packages. Four surfaces are provisional and §5.2 names what moves each.
> Graphics is not part of this.

### 5.5 What no tutorial may teach yet

Surfaces that work, or half-work, and that the library should not commit to.
The sharp ones: `Device.Queues`/`QueueFor` (one hardcoded queue, and
`QueueFor(QueueTransfer)` silently returns it); `UniformBuffer` (no dispatch
path consumes one); `Binding.Texture` and `Binding.Sampler` (fail at dispatch);
the stage types and the `//accel:vertex`/`//accel:fragment` directives (nothing
runs them and the directives are silently ignored); the `Kernel` record's
fields; and the tensor view family — `Permute`, `Transpose`, `Slice`,
`Broadcast` — whose results reach nothing but elementwise operators while
`Contiguous` does not exist (§4.1's sibling, recorded in
[025](025-tensor-operators.md)).

### 5.6 What each tutorial waits on

The per-tutorial form of §4's gate. **This is the todo list**, and no tutorial
is written before its row is clear.

| # | Tutorial | Must land first |
| --- | --- | --- |
| 1 | Hello, GPU | §5.3 item 2; a workgroup-count helper or an explicit ceiling-division paragraph |
| 2 | Writing a kernel | the three directives and the kernel-package rule documented; `kmath` named in the diagnostic |
| 3 | Memory | pool-constructor and usage/format naming; ~~the transient-`Buffer` guard~~ **done 2026-08-24** |
| 4 | Graphs | copy-node naming; `Bind` naming; `Recorder` and `Transient` doc fixes |
| 5 | Cooperation | subgroup naming; `Requirements.SharedBytes` populated — see [016](016-graph-execution.md)'s correction |
| 6 | Uniforms and scalars | §5.3 item 2; `UniformBuffer` resolved either way |
| 7 | Tensors | `Contiguous`; a port-buffer helper; the f16/f32 split stated |
| 8 | A decode step | `PlanCache` ownership documented; `Attention`'s position cap stated |
| 9 | Quantized weights | `quant` naming and `Error`'s argument; f16 host uploads |
| 10 | Backends and portability | `ErrNoAdapter`/`ErrPolicy`; ~~`Capabilities.Set`~~ **done 2026-08-24**; `CPUMode` docs |

Two rows were discharged while the review was being written, which is the
argument for the ordering in miniature: both were found by asking what a
tutorial would have to work around, and one of them was a crash.

### 5.7 What the review found that was not a naming question

Recorded because §4 predicted "anything a tutorial would have to apologise for"
and got two process crashes instead:

- closing a graph transient's `Buffer` panicked on a nil pool, in a package
  whose own doc promises "the worst outcome is a rejection";
- `Device.NewTexture` and `Queue.ReadTexture` could never be used together,
  because the convenience constructor hard-coded device-local memory.

Both are fixed. The generalizable part: a surface reviewed *as a surface*
surfaces defects that per-declaration review does not, because the defect is in
the join between two declarations that are each individually reasonable.

## 6. Done

- every reader-facing document names its audience in section 1's table and reads
  as if written for it, with no spec number or milestone identifier used as the
  primary way of saying what a reader can do;
- every honest limitation the current documents state is still stated after the
  rewrite, verified by listing them before and comparing;
- design rationale removed from `docs/` is **moved** into `specs/` or the
  package doc rather than deleted, verified by the same before-and-after list;
- the public surface has a review record naming what is frozen and what is
  provisional;
- the ten tutorials exist, each teaching one concept, each ending with the reader
  having run something; and
- every tutorial example is an `Example` function that `go test` runs.

## 7. Open questions

- **Whether `docs/architecture.md` belongs in `docs/` at all.** Its stated
  audience is "someone who wants to understand or contribute", which is
  `specs/`'s reader by section 1's rule. Splitting it — a short user-facing
  "how it fits together" in `docs/`, the rationale into `specs/` — is the
  obvious move and is not yet decided.
- **Whether the README should carry a status table at all.** It is the section
  that rots fastest and the one most tempted toward milestone voice. A
  capability table ("can I do X today") may serve a reader better than a
  build-order table.
- **How much of `docs/conventions.md` is user-facing.** It claims to be useful
  even to people not using accel, which is a real and unusual asset. Whether it
  should be promoted rather than filed under backend-author documentation is
  open.


## The drift guard covered one package — corrected 2026-08-25

§3.1 requires tutorial code to live in `Example` functions so that `go test`
compiles it. That was specified and never built, and `docdrift_test.go` is the
narrower guard that replaced it: every API name the documentation uses must
still exist.

It checked `accel.Foo` and not `tensor.Foo`, which is the wrong half. The
tutorials are mostly about the tensor layer, so the guard was covering the
package the documentation talks about **least**. Widened, and verified by naming
an operator that does not exist and watching it report.

**The finding that generalises is about stale advice rather than missing
sections.** Tutorial 9 told a reader to take the sampling draw from
`rng.Float32()`. Every name in that sentence existed, so no drift guard of this
shape could ever have flagged it — and it is the exact call
[039](039-sampling-policy.md) §2 argues against, because
`float32(rng.Float64())` rounds up to 1.0 about once in 2^24 and the sampler
clamps that rather than reporting it. A reader following the tutorial got
working code with a bug they could not see.

So: **a name-existence guard cannot see advice that is wrong, only advice that
is gone.** The rule that follows is a review obligation rather than a test —
when an operator subsumes a composition the documentation teaches, the
documentation is part of the change, not follow-up work.
