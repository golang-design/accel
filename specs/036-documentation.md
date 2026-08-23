---
title: "Documentation: who each document is for, and the tutorial deck"
status: drafted
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

| # | Tutorial | The one concept | Use case it is framed around |
| --- | --- | --- | --- |
| 1 | Hello, GPU | open a device, dispatch a kernel, read the result | scale an array |
| 2 | Writing a kernel | the Go subset and `go generate` | elementwise image adjust |
| 3 | Memory | pools, buffers, views, lifetimes | holding weights that outlive a frame |
| 4 | Graphs | record once, replay many | a simulation step run every tick |
| 5 | Cooperation | shared memory and barriers | a reduction, and a tiled GEMM |
| 6 | Uniforms and scalars | passing values that are not buffers | a runtime coefficient |
| 7 | Tensors | shapes, operators, one plan | a small MLP |
| 8 | A decode step | KV cache, attention, sampling | one token from a transformer |
| 9 | Quantized weights | int8 with a per-block scale | fitting a model in memory |
| 10 | Backends and portability | the CPU oracle, selecting a device | testing without a GPU |

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

**Scope of the freeze.** The compute half only: device, memory, kernels, graphs,
cooperation, and the tensor layer. Those shipped across M1 through M8 and
[000](000-decisions.md)'s two-layer split is locked, so graphics arrives as a
*sibling* at the device layer rather than a modification of it. Waiting for
graphics before documenting compute would leave the library undocumented through
[035](035-cpu-rasterizer.md)'s remaining steps, the stage-ABI compiler work, and
Metal — which is the problem this spec exists to fix, not a solution to it.

## 5. Done

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

## 6. Open questions

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
