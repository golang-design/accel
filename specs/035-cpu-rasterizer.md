---
title: "The CPU reference rasterizer and its conformance corpus"
status: in progress
layer: device
depends_on:
  - 005-graphics.md
  - 008-numerics.md
  - 032-stage-abi.md
  - 033-render-api.md
  - 034-surface-present.md
---

# The CPU reference rasterizer and its conformance corpus

[005](005-graphics.md)'s fourth child spec, and the one that decides whether the
other three can be trusted. The CPU backend implements the **full** graphics
path, not just compute.

The case is not aesthetic. [`000-decisions.md`](000-decisions.md) decision 3
makes the CPU backend the correctness oracle, and
[`conventions.md`](../docs/conventions.md)'s first testing rule says every
convention in that document is a way for a GPU backend to disagree with it. Half
of those conventions are graphics conventions — clip-space depth range, face
winding, readback origin, depth texture storage. A CPU backend that does not
rasterize leaves exactly the entries that cost the predecessor hours with no
oracle at all, and leaves every graphics test needing a device, which is the
provisioning problem decision 3 exists to end.

**Scope, so it is not mistaken for a performance path.** Triangles, lines, and
points; top-left fill rule; perspective-correct interpolation; depth and stencil;
per-attachment blending; multiple render targets; no multisampling. It is a
reference implementation, it is not expected to be fast, and no optimization work
is planned for it. The predecessor project is a software rasterizer, so the size
of the job is measured rather than guessed.

## 1. The pipeline, stage by stage

```
 vertex fetch      vertex stage       clip          viewport         raster
┌───────────┐     ┌───────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐
│ attribute │────▶│ generated │──▶│ near/far │──▶│ NDC → win │──▶│ top-left │
│ by index  │     │ Go from   │   │ + guard  │   │  y flip   │   │ fill rule│
└───────────┘     │ 032's IR  │   │  band    │   └───────────┘   └────┬─────┘
                  └───────────┘   └──────────┘                       │
                                                                     ▼
 attachment write   blend        stencil      depth test    fragment stage
┌───────────────┐ ┌────────┐  ┌───────────┐ ┌───────────┐ ┌──────────────────┐
│ per target,   │◀│ per    │◀─│ per face  │◀│ compare + │◀│ generated Go,    │
│ write mask    │ │ target │  │ ops/masks │ │ write     │ │ perspective-      │
└───────────────┘ └────────┘  └───────────┘ └───────────┘ │ correct varyings │
                                                          └──────────────────┘
```

Every box left of the fragment stage is fixed function and is this spec's
subject. The two stage boxes are [032](032-stage-abi.md)'s generated Go, called
by this rasterizer — which is what makes this an oracle rather than a second
implementation: the same IR produces the code the GPU runs and the code this
runs, so a disagreement is a lowering difference and not two people's arithmetic.

## 2. The fill rule is a guarantee, not an implementation detail

Without a stated rule, two triangles sharing an edge can double-shade or leave a
gap, and any coverage comparison against the oracle is meaningless.

**accel guarantees the top-left rule.** A pixel centre exactly on an edge is
covered if and only if that edge is a top edge or a left edge of the triangle.
With edge functions

$$
E_i(x, y) \;=\; (y_{i+1} - y_i)(x - x_i) \;-\; (x_{i+1} - x_i)(y - y_i)
$$

a sample at $(x,y)$ is inside when every $E_i$ is positive, or zero on an edge
that the rule admits:

$$
\text{covered} \;\iff\; \forall i:\; E_i > 0 \;\lor\; \big(E_i = 0 \land \text{topLeft}(i)\big)
$$

where an edge is *top* when it is horizontal and above the interior, and *left*
when it descends on the left side — the standard formulation, stated here so the
implementation has one definition to be checked against rather than a habit.

**The honest limit.** On-edge sample tie-breaking is not identical across GPU
vendors even with the rule stated, so shared-edge pixels remain a bounded
comparison and never an exact one. Section 6 says which side of the line every
corpus test sits on.

## 3. Interpolation

Screen-space barycentrics interpolate $1/w$ linearly; every other attribute is
recovered from it:

$$
\frac{1}{w} = \sum_i \lambda_i \frac{1}{w_i}
\qquad\qquad
a(x,y) = \frac{\displaystyle\sum_i \lambda_i \frac{a_i}{w_i}}{\displaystyle\sum_i \lambda_i \frac{1}{w_i}}
$$

Depth is the exception and is interpolated **linearly in window space**, without
the perspective divide, because window depth is already the post-divide value:

$$
z_{\text{window}}(x,y) = \sum_i \lambda_i\, z_{\text{window},i}
$$

Getting that backwards is a classic bug with a deceptive symptom: the image looks
correct and the depth test picks the wrong surface only where two surfaces are
close, which reads as z-fighting rather than as a formula error. A test therefore
compares interpolated depth against the closed form for a known plane, rather
than only comparing which surface won.

Flat-interpolated varyings take the **provoking vertex**, which is the first
vertex of the primitive. Every target backend can be configured to agree, and
"first" rather than "last" is chosen because it is the default on more of them;
what matters is that it is stated, since a differing provoking vertex changes
integer varyings on shared vertices with no other symptom.

## 4. Clipping

Near and far clipping happen against the clip-space planes before the perspective
divide, since a vertex with $w \le 0$ has no meaningful NDC position:

$$
-w \le z \le w \qquad\text{(the } [-1,1] \text{ convention of 032 §2.3)}
$$

Side planes are **not** clipped geometrically. A guard band plus scissoring to
the render area gives the same result with less arithmetic and fewer degenerate
cases, which matters here because every clip-generated vertex needs its varyings
interpolated and every interpolation is a place for the oracle to disagree with
itself.

Near-plane clipping is not optional and is the subject of one of the sharper
corpus tests: `conventions.md` records that a `[-1, 1]` projection on a `[0, 1]`
backend produces **no coverage at all** for near geometry, which reads like a
broken transform rather than a convention mismatch.

## 5. Depth, stencil, blend, and the write mask

Fixed function, in this order, matching every target backend's **late** tests,
which are the baseline everywhere:

1. **Fragment stage.** A discard ends the invocation here and nothing below it
   runs — no attachment write, and no depth write.
2. **Stencil test**, per face, with its compare function, read mask, and
   reference value; then the fail / depth-fail / pass operations.
3. **Depth test**, with its compare function, then the depth write if enabled.
4. **Blend**, per attachment, with separate colour and alpha factors and
   operations.
5. **Write mask**, per attachment, per channel.

The stage comes **first** and that is the whole point of the ordering. Early
depth testing — running steps 2 and 3 before the stage — is an optimization every
backend performs only when it can prove the stage does not change the answer, and
a stage that can discard cannot be proven that. Listing the tests first would
make [032](032-stage-abi.md) §4.2's guarantee unimplementable: a discarding
fragment would already have written depth before it discarded.

$$
C_{\text{dst}}' \;=\; \text{op}\big(F_{\text{src}} \cdot C_{\text{src}},\; F_{\text{dst}} \cdot C_{\text{dst}}\big)
$$

**Early-Z is an observable, not an optimization, and this rasterizer never
performs it.** [032](032-stage-abi.md) §4.2 records `Discards` in the stage
record precisely so [033](033-render-api.md) does not promise an early-Z a
pipeline cannot have; here, the order above is run for every fragment whatever
the stage does, which is what makes the ordering itself checkable. The cost is
speed, which this rasterizer explicitly does not have.

A corpus entry follows directly: a discarding stage over a pre-cleared depth
attachment must leave the depth untouched, which is exactly the assertion that
distinguishes the two orderings.

sRGB attachment formats convert on write and on read, not in the fragment stage.
The stage sees linear values on every backend, which is what
[001](001-device-resources.md)'s format table already asserts.

## 6. What the oracle proves exactly, and what it proves within a bound

This distinction is what makes the CPU rasterizer credible. Conflating the two
produces a suite that either fails constantly or proves nothing, and this
repository has recorded three times that a criterion checked off against a test
that nearly tests it is the failure mode to design against.

**So every corpus test below declares its side.** A test with no declared side is
not a test in this corpus.

| Exact, on every backend | Why it can be exact |
| --- | --- |
| occlusion ordering, when competing depths are separated by more than both pipelines' interpolation intervals | the answer is a discrete surface, and separation removes the tie |
| attachment routing — which output field lands in which attachment | an integer mapping |
| winding and cull behaviour | a sign test |
| clip-range survival — which geometry is clipped | a comparison against a plane, with fixtures away from the plane |
| pixel origin agreement across fragment stage, on-device fetch, and host readback | three integer indexings of the same memory |
| graph structure — which nodes exist, which edges were inferred | not arithmetic at all |
| discard writing nothing | absence, not a value |
| load and store action effects | `LoadKeep` versus `LoadClear` differ by a whole clear value |

| Within a bound from [008](008-numerics.md) | Why not exact |
| --- | --- |
| interpolated varying values, and anything computed from them | interpolation rounding differs between implementations |
| shaded colours | the stage's own arithmetic, whose bound 008 already derives |
| coverage at shared and near-degenerate edges | §2's honest limit |
| depth-bias results | the resolvable depth difference $r$ is defined per format and per backend |

**No portable winner is asserted when two depth intervals overlap.** Depth
testing chooses a discrete surface, so an ordinary numeric tolerance cannot
repair a different winner: the comparison is either separated enough to be exact
or it is not a comparison. Occlusion fixtures therefore use separated depths, and
report the interpolated depth intervals when they fail, so a failure says whether
the separation was the problem.

## 7. The corpus

Each entry names what it catches, because a test whose failure nobody can
interpret gets deleted the first time it goes red.

| Test | Side | What it catches |
| --- | --- | --- |
| **Triangle.** A clip-space triangle to an offscreen target; interior pixels match. | bounded interior, exact coverage away from edges | Everything at once. The predecessor's first milestone. |
| **Occlusion is draw-order independent.** Two overlapping triangles, near and far, drawn in both orders. | exact | A missing or non-functioning depth attachment. The near one must win both times. |
| **Depth interpolation matches the closed form.** A known plane's interpolated depth against its analytic value. | bounded | §3's linear-versus-perspective error, which otherwise reads as z-fighting. |
| **Discard writes no depth.** A stage discarding over a pre-cleared depth attachment. | exact | §5's ordering: with the tests run before the stage, the depth is already written when the discard happens. |
| **MRT attachments are distinct.** Three attachments, a different constant to each. | exact | Aliased attachments, which a single-target test cannot see. |
| **Winding flip empties coverage.** Same geometry, reversed front face, back-face culling on. | exact | The Metal winding divergence, which otherwise presents as a shading bug. |
| **Near-plane survival.** Geometry straddling the near plane keeps its near half. | exact | The `[-1,1]`-on-`[0,1]` symptom: no coverage at all, reading like a broken transform. |
| **Reverse-Z keeps near geometry.** A reversed projection, depth cleared to 0, `Greater` compare. | exact | [032](032-stage-abi.md) §2.5's claim that reverse-Z needs no API change. |
| **Origin agreement, discriminating form.** A target encoding row position; path A a compute texel fetch of `(x,0)` then a buffer read, path B a host texture read of row 0. Assert `A == B` **and** that both hold the top-row value. | exact | The predecessor's actual bug: `conventions.md` records that its compute-path test passed while the texture path was mirrored. Asserting only `A == B` would pass with both mirrored. |
| **Handoff stays on device.** The deferred graph's nodes inspected: no host transfer between geometry and tonemap. | exact, structural | The predecessor's failure — a G-buffer that went to the host and back every frame. Not sufficient alone, which is why the origin test checks values. |
| **Depth readback through a transfer node.** | exact | The macOS private-depth constraint, which the API refuses to offer a direct path for. |
| **Load and store actions are observed.** `LoadKeep` preserves, `LoadClear` does not, `DontCare` asserted only on aliasing. | exact | A backend ignoring load actions, and the aliasing consequence of the missing edge. |
| **Per-object replay.** N objects at recorded uniform offsets, submitted twice with different transforms. | bounded values, exact structure | Re-recording where 003 promises replay; and the `sizeof(T)` stride mistake. |
| **Indirect draw.** A compute pass writes arguments; the result matches the direct draw. With the count capability, a device count above the maximum clamps and reports. | exact | An indirect-read barrier that is merely a read barrier, which hangs rather than answers wrongly. |
| **Feedback validation.** Overlapping subresource rejected naming both views; disjoint mip and disjoint layer accepted; blended depth-tested pass accepted. | exact | A handle comparison, which gets the disjoint half wrong. |
| **Present-slot identity.** A frame from another surface, from an earlier generation, and an ordinary matching texture, all rejected. | exact | A format-only check, which accepts the third. |
| **Headless frame loop.** Several frames, double buffered, with a resize in the middle. | exact | The surface state machine, on every platform with no display. |
| **Determinism.** The same graph twice on one backend, identical images. | exact | 003's guarantee. |

### 7.1 The differential the corpus is built around

Every corpus entry that produces pixels runs on the CPU backend **and** on Metal
from the same recorded graph and the same generated stages, and the two are
compared on the side the table declares. That is the same shape the compute
corpus already has, and it is the reason this rasterizer is worth its cost.

When a comparison fails, `conventions.md`'s diagnostic order applies: coverage
counts and overlap between competing interpretations first, mathematics last.
**Equal pixel counts with roughly half overlap is the flip fingerprint**, and it
identifies an origin bug in one measurement rather than sending anyone to check
their matrices for an afternoon.

## 8. Implementation order

Deliberate, because the pieces have very different risk:

1. **Offscreen triangle, one colour target, no depth.** Proves fetch, the vertex
   stage, clip, viewport, the fill rule, the fragment stage, and the attachment
   write. Smallest thing that is end to end.
2. **Depth and occlusion.** Adds the test and the write, and the second corpus
   entry becomes meaningful.
3. **MRT, blend, write mask, stencil.** Fixed function breadth; each has a corpus
   entry that fails without it.
4. **The graph integration.** Pass nodes, declared access, inferred edges,
   feedback validation. Structural tests, no pixels.
5. **The headless surface.** The frame loop state machine.
6. **Metal**, against the corpus built above, which by then is an oracle rather
   than an aspiration.

Steps 1 to 3 need no GPU and no surface. Step 6 is where
[`conventions.md`](../docs/conventions.md)'s graphics half gets its first
verification, which is the entire reason 005 exists.

### 8.1 What is built — 2026-08-23

Step 1's fixed-function core, in `internal/raster`: the fill rule,
perspective-correct varying interpolation, linear window-depth interpolation,
near/far clipping, the viewport transform with its y flip, winding and culling,
and the scissor. It takes clip positions and a flat float vector of varyings and
calls back per covered sample, so it is testable without the compiler — which is
what let the arithmetic land before the stage ABI does.

Two of its own tests were **too weak on the first pass**, and both were found by
reinstating the bug rather than by review:

| The test as first written | What survived it |
| --- | --- |
| winding and culling, asserted only through cull modes | dropping the vertex swap that normalises a back-facing triangle's winding. Culling is decided *before* the walk, so every culling assertion still passed while a surviving back-facing triangle covered nothing. Fixed by asserting that under `CullNone` it covers exactly what its front-facing counterpart covers. |
| clipped vertices, asserted as "reads above zero" | a clip that copied an endpoint's varyings into the new vertex. Samples are pixel centres strictly inside the triangle, so nothing ever reads the endpoint's exact value. Fixed by computing the clip parameter and checking the minimum against it. |

Both are the same shape as this repository's recurring finding, one level down:
a criterion checked against a test that nearly tests it. The generalizable part
is that **an assertion about a bound must use the bound the arithmetic
predicts**, not a weaker one that happens to hold.

Steps 2 and 3 followed: the per-fragment chain, in this section 5's order, with
depth, stencil, per-attachment blend, the write mask, and MRT. Two notes.

**The ordering was wrong in this spec before it was right in the code.** §5
numbered the depth test and write *before* the fragment stage — early-Z
ordering — while its own prose said the rasterizer never performs early-Z. The
numbered list is what an implementer reads, and under it §4.2 of
[032](032-stage-abi.md) is unimplementable: a discarding fragment has already
written depth by the time it discards. Corrected before the code was written,
and the corpus gained the entry that separates the two orderings.

**Depth and stencil are one target.** Every backend that offers stencil offers
it packed with depth, and the stencil operations depend on the depth test's
outcome — separating them would put the depth-fail case in a type that cannot
see the depth test. That case is a distinct branch and not folded into the fail
case, which is the mistake a naive implementation makes and which shadow-volume
techniques count on.

**Primitive assembly lives in this package**, not in a backend. The walk from a
draw's counts to triangles — index fetch, topology, the instance loop, and the
two offsets — is where strip winding and index arithmetic live, and a backend
that assembled its own would be a second implementation of both.
[`000-decisions.md`](000-decisions.md) decision 3 makes this the thing a backend
is checked against rather than a peer of it, so geometry logic belongs on this
side of the line.

Strip winding alternates: the second triangle of a strip swaps its first two
vertices, so every triangle presents the same facing as the first. Without it,
every other triangle is back-facing, and with culling on the mesh comes out
striped — which reads as a geometry bug rather than a winding one, so a coverage
test with culling *off* cannot see it. The test therefore culls, and a second
test asserts the triple order directly, because "the quad filled" is consistent
with more than one correct-looking assembly.

Lines and points are refused by name, per §10. Approximating a rule this
document leaves open would put an unstated rule in the oracle every backend is
checked against, which is worse than a topology that does not draw.

One implementation choice worth recording because it is not obvious. Edge
functions are evaluated in `float64` even though everything around them is f32.
The fill rule's decision is about a sample landing *exactly* on an edge, which is
what the rule exists to arbitrate; an f32 edge function makes that a coin toss
between two triangles that round differently, and the double-shading or gap it
produces is exactly what §2 says makes a coverage comparison meaningless.

## 9. Done

- every corpus entry in section 7 exists, declares its side, and runs on the CPU
  backend;
- every pixel-producing entry also runs on Metal and is compared on its declared
  side;
- the fill rule is checked against §2's formulation directly, with a fixture of
  two triangles sharing an edge covering each shared pixel exactly once;
- depth interpolation is checked against the closed form, not only against which
  surface won;
- the flat-interpolation provoking vertex is asserted, with a fixture whose three
  vertices carry different integer varyings;
- the origin test asserts both equality and the top-row value, confirmed by
  mirroring both paths and checking the test still fails; and
- `conventions.md`'s graphics entries — clip depth range, winding, readback
  origin, depth storage mode — each name the corpus test that verifies them,
  which is what turns that document from remembered into measured.

## 10. Open questions

- **Line and point rasterization rules.** Triangles have a stated fill rule;
  lines have a diamond-exit rule on some backends and a Bresenham-ish rule on
  others, and points have a size and a centre convention. No corpus entry needs
  them yet, so they are stated as unspecified rather than guessed, and a topology
  other than triangles is refused until they are.
- **Whether the rasterizer should be parallel.** It is a reference, so
  correctness comes first, but the corpus at real resolutions may be slow enough
  to matter for CI time. Tile-parallel is the obvious shape and would need the
  same determinism guarantee the compute scheduler already provides.
- **Conservative rasterization and sample positions.** Both are capabilities
  elsewhere and neither has a portable definition; out of scope until something
  needs one.

## 11. §7.1's fixed-function gap, and what closing it found — 2026-08-29

§7.1 requires **every pixel-producing corpus entry to run on the CPU backend and
on Metal**, compared on its declared side. Four entries ran on the CPU only —
the write mask, multiple render targets, per-attachment blend, and determinism —
and the first of those was wrong on the GPU the whole time.

### 11.1 The colour write mask reached the wrong channels

`accel.ColorWriteMask` numbers its channels **red at bit 0**, as Vulkan and
D3D12 do. `MTLColorWriteMask` numbers them **alpha at bit 0**, red at bit 3. The
Metal backend passed the value through as `int(m)`, so every mask arrived
mirrored:

| the caller asked for | Metal wrote |
| --- | --- |
| red | alpha |
| green | blue |
| blue | green |
| alpha | red |
| **all** | **all** |
| **red and alpha** | **red and alpha** |

The last two rows are why it survived. `WriteAll` is `0xF` under either
numbering, and it is the default and the only mask any test had used; a mask
symmetric about the middle is likewise its own mirror, and one of the five cases
written for this test passes even against the bug. **A partial fixture would
have reported success.**

It is now `metalWriteMask`, written out by name, which is the rule
`metalLoadAction` already states — *"the two enumerations agree today, and a
numeric conversion mistranslates silently the day either gains a value in the
middle"*. Here they never agreed, and the conversion was wrong from the day it
was written.

### 11.2 The other three were correct, and are now compared

MRT and per-attachment blend both already worked on Metal. That is the useful
outcome of running the search rather than the disappointing one: the entries are
now compared rather than assumed, and each was confirmed to discriminate by
reinstating a fault in the backend — swapping the two attachments moves
attachment 0's clear, and applying attachment zero's blend to both moves
attachment 1 by 0.2.

Determinism had no test at all. `TestPlansAreDeterministic` is about plan-cache
identity, not pixels. One built graph is now submitted **twice** — a replay
rather than two builds, because [003](003-command-graph.md)'s guarantee is about
replay and rebuilding would also be re-planning — and compared bit for bit on
both backends. Bit for bit, not within a bound: two runs of one program over one
input have no licence to differ, and a bound would hide exactly what the entry
is for.

### 11.3 What generalizes

**An enumeration crossing an ABI is a mapping, not a number, and the default
value is the one that hides a wrong mapping.** Both of this session's live
defects were found by asking which backends read what a call writes; this one
adds that agreeing on the *common* value is not evidence, because the common
value is what every existing test uses.

§7.1's remaining CPU-only entry is stencil, which no caller can reach at all —
[033](033-render-api.md) §2.1's state is unbuilt in the public API while
`internal/raster` implements the whole of it.

## 12. Three corpus entries, and two skips that could never fire — 2026-08-30

§7's origin agreement, handoff, and depth-readback rows, built. Five of §7's
eighteen entries had no test; these are three of the five.

### 12.1 Origin agreement, and the assertion that earns its place

§7 singles this entry out as the one that caught the predecessor's *actual*
bug — `docs/conventions.md` records that its compute-path test passed while the
texture path was mirrored. What existed here was
`TestRowZeroIsTheTopRow`, which checks the rasterizer's own window mapping:
one path, with nothing to disagree with.

The two paths now differ in **how the data reaches the host**:

| | |
| --- | --- |
| A | the device reads the texture at row 0 through a texel fetch, writes a buffer-backed attachment, and the host reads the *buffer* |
| B | the host reads the *texture* and takes row 0 |

A buffer has rows only by arithmetic, so path A has no origin convention to get
wrong; path B does. Both assertions were confirmed by reinstating a fault, and
**they catch different things**:

- mirroring the host readback breaks `A == B`, because only path B goes through
  it;
- mirroring the *stored image* — which both paths then read identically —
  leaves `A == B` intact and is caught only by the second assertion, that row
  zero holds the **top** row's value. The failure reads
  *"row zero holds row 4's value, so the origin is at the bottom on both
  paths"*, which is the predecessor's bug stated in the test's own words.

§7 asks for the fetch to be a **compute** one. It cannot be:
[032](032-stage-abi.md) §5.1 refuses a texture in a compute kernel by name,
because a dispatch's argument set cannot carry one. The fragment fetch crosses
the same seam — the device's view of the texture's rows against the host's — and
that substitution is recorded here rather than left as a silent divergence from
the corpus table.

### 12.2 The handoff is structural, and the values are there so it is not vacuous

A G-buffer that goes to the host and back every frame produces an identical
picture, so no value comparison can see it. The assertion is on `Graph.Nodes()`:
two nodes, a pass then a dispatch, no copy node of any kind, and a hazard
between them — without the edge the two could run in either order and be right
by luck on a backend that happens to serialize. The values are asserted too,
because "no transfer" is also satisfied by a graph that did nothing.

The attachment is a **slot** rather than a texture for §12.1's reason: a compute
kernel cannot read a texture, so the handoff a caller can build today is a pass
writing buffer-backed bytes and a dispatch reading them.

### 12.3 Depth readback names the refusal it routes around

`Queue.ReadTexture` refuses every depth format, because a depth texture is
device-private on macOS and the API declines to offer a path that works on one
platform. The entry asserts **both** halves: that the direct read is refused,
and that the recorded copy works and lands the window depth the convention
predicts — clip $z = 0.25$ at $w = 1$ is $0.625$, neither of which is the depth
clear, so a buffer the pass never wrote fails rather than matching by accident.

### 12.4 Two skips that could never fire

Both backends lower a texture copy, and Metal has lowered a texture attachment
since [045](045-texture-attachments.md). Two skip guards were still watching for
refusal text that **exists nowhere in the tree**: `skipUnlessLowered` here and
`notLowered` in the texture round-trip. Each was written for the right reason —
so the comparison would start on the first day it could — and each had outlived
it silently.

**A skip that can never fire is a comparison nobody notices is not happening**,
which is one level up from the class §7.1 exists to catch. They are removed
rather than left standing. The guard is worth writing; what is missing is
anything that tells you when it has done its job.
