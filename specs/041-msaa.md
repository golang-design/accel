---
title: "Multisampling: coverage, resolve, and what the oracle can still prove"
status: drafted
layer: device
depends_on:
  - 001-device-resources.md
  - 005-graphics.md
  - 011-conformance-harness.md
  - 032-stage-abi.md
  - 033-render-api.md
  - 035-cpu-rasterizer.md
  - 006-backends.md
  - 008-numerics.md
---

# Multisampling

[005](005-graphics.md)'s fifth child spec, and the one it names as required
before any sample count above 1 is legal. 005 lists seven obligations and defers
an eighth (the ROA interaction) here. This spec discharges all eight.

The API half is small. The load-bearing half is section 9: what the CPU oracle
can still prove once sample positions exist. 005 deferred MSAA on exactly that
ground, and [006](006-backends.md) repeats it as the reason the CPU row says
`no`. A spec that grows the API and shrinks the oracle without saying by how much
has not answered 005; it has changed the subject.

## 1. The eight obligations, and how each is discharged

Refusal is a discharge in this repository — [035](035-cpu-rasterizer.md) §10
refuses line and point rules, [032](032-stage-abi.md) §5 refuses sampling — but
only when it is visibly a decision. So each row says which it is.

| 005's obligation | Settled in | As |
| --- | --- | --- |
| multisampled texture creation | §3 | defined |
| equal sample counts across all pass attachments | §4 | defined, as three separate checks |
| single-sample resolve targets | §4 | defined |
| load/store/resolve combinations | §4 | defined, as an explicit matrix |
| per-sample shading | §7 | **refused**, with the reason and the cost |
| coverage masks | §6 | defined as a static pipeline mask; a stage-written mask **refused** |
| CPU-oracle limits | §9 | defined, and it is the longest section here |
| ROA × MSAA ordering | §8 | **stays exclusive**, with a named enforcement point |

Also refused, each in its section: depth resolve (§5), integer-format resolve
(§5), fetch of a multisampled texture (§7), alpha-to-coverage (§7), sample counts
8 and 16 (§2).

## 2. Sample positions: the deferral, restated honestly, then mandated

005 says the sample pattern "is not standardized across vendors". That is close
enough to true to have carried the deferral, and it is worth stating precisely,
because the precise form changes what accel does about it:

| Backend | What the API says about positions | Verified |
| --- | --- | --- |
| Vulkan | `standardSampleLocations` is a **queryable limit that may be false**; `VK_EXT_sample_locations` programs them | yes |
| Metal | programmable positions exist (`setSamplePositions:count:`); no default is documented as a contract | yes |
| D3D | documents standard positions per sample count | not checked here |
| GLES 3.1 | positions are queryable per sample; none are documented as mandated | not checked here |

The two verified rows are the two the argument needs, and they are enough:
positions are queryable-but-not-guaranteed on one target backend and
undocumented on another. **accel therefore relies on no device's sample positions
being knowable, stable, or equal to its own**, and every claim below is written
so it survives whatever the device does with them. The last two rows are context;
nothing here rests on them.

**accel mandates a pattern for its own rasterizer.** Not because it matches
anyone, but for [035](035-cpu-rasterizer.md) §2's reason one level down: an
unstated rule inside the oracle every backend is checked against is worse than a
stated one. The positions are accel's own, published, on a 1/16 grid in the
pixel's own top-origin space, and the index order is normative because §6's
static mask names bits by it:

| $N$ | $s_0$ | $s_1$ | $s_2$ | $s_3$ |
| --- | --- | --- | --- | --- |
| 1 | (8/16, 8/16) | | | |
| 2 | (4/16, 4/16) | (12/16, 12/16) | | |
| 4 | (6/16, 2/16) | (14/16, 6/16) | (2/16, 10/16) | (10/16, 14/16) |

A sample of pixel $(x, y)$ sits at $(x + u_i,\; y + v_i)$ in window space, and
§6's coverage test is [035](035-cpu-rasterizer.md) §2's fill rule evaluated
there. **The fill rule does not change and is not restated**: the sample set
moves, the rule does not. `internal/raster` already evaluates it at an arbitrary
point and hard-codes that point as `float64(x)+0.5`; this table replaces the
constant, and 035 §8.1's deliberate `float64` edge functions stay untouched.

**Legal sample counts are 1, 2, and 4.** 8 and 16 are refused by name: nothing in
the corpus needs them, and a published table is a promise — two more rows would
be two more rows nobody has checked against a device.

## 3. Sample count on the texture, and on the pipeline

**Sample count is pipeline-creation state**, per 005's and
[033](033-render-api.md) §2.1's fixed column, which both already reserve the row.
It is `VkPipelineMultisampleStateCreateInfo`, Metal's `rasterSampleCount`, and
D3D12's `SampleDesc` inside the PSO. Putting it only on the pass descriptor would
make the pipeline/pass agreement uncheckable at pipeline creation and push the
error to build — which [033](033-render-api.md) §6 forbids.

`TextureDescriptor` gains `SampleCount int`, where 0 and 1 both mean
single-sampled. `validateTexture` gains, in the shape `MipLevels` already has:

- `SampleCount > 1` requires `UsageRenderTarget`, and refuses `UsageCopySrc`,
  `UsageCopyDst`, and any shader-visible usage;
- `MipLevels` and `ArrayLayers` must be 1, so MSAA reopens nothing about
  subresource naming;
- the count must be in the device's set for that **format**, per §3.1.

A multisampled texture is **not host-copyable and not a copy source**, on the
`Depth24PlusStencil8` precedent in [001](001-device-resources.md): its per-sample
layout is device-defined. `Queue.readTexture` refuses it, naming the resolve path
the way depth readback's error names the transfer node. That refusal is not
politeness: on the CPU backend a multisampled texture is just memory, so a naive
`readTexture` **works there and fails on Metal** — the oracle passing while the
device fails, which is the exact inversion decision 3 exists to prevent.

**`textureBytes` gains the factor $N$.** It computes `pitch * height` today and
nothing else. Adding `SampleCount` without touching it under-allocates every
multisampled texture by $N$, and the symptom is not a wrong image: the transient
packer in `alias.go` reports a valid-looking plan while a multisampled attachment
writes over the neighbour it was packed against, so the failure surfaces as a
corrupted unrelated resource in a different node.

### 3.1 The capability has a value, not a bool

`Capabilities.Multisampling bool` cannot express 4×-yes/8×-no, nor per-format
support — and support **is** per format on every backend (Vulkan's per-format
`sampleCounts`, D3D12's multisample quality levels, Metal's
`supportsTextureSampleCount:`, GLES's per-internalformat query). Inferring that a
count is legal because the format is renderable is the same class of mistake
005's closing amendment records for `RGBA32Float`.

So the answer lives on `FormatInfo`, beside `Renderable` and `HostCopyable`, as a
bitmask of supported counts; `Multisampling bool` stays as the coarse "some
format supports some count above 1". `FormatInfo` reports the **intersection** of
the device's set with accel's legal `{1, 2, 4}`, because reporting 8 hands a
caller a number this spec then refuses. Per
[`000-decisions.md`](000-decisions.md) decision 6 an unsupported count is an
error naming the supported set, never a silent downgrade to the nearest legal
one. Sample count is not a hint, which is how most engines spell it.

## 4. Resolve is a store action, not a node

A resolve is a property of a colour attachment's store. It is not a graph node
and not a draw. A resolve node would put a second unit of synchronisation inside
what the hardware treats as one pass, and on a tiler it would force the
multisample buffer out of tile memory — the exact cost MSAA is chosen to avoid.

```go
Color: []accel.ColorAttachment{{
	View:    msaaColor,           // SampleCount 4
	Load:    accel.LoadClear, Clear: accel.ClearColor{},
	Store:   accel.StoreDiscard,  // the multisample buffer is never written out
	Resolve: accel.ResolveAverage,
	ResolveTarget: singleSampled,  // SampleCount 1
}}
```

`Resolve` is an explicit enumerated action whose zero value is `ResolveNone`.
Following [033](033-render-api.md) §3.1's rule for clear values, a
`ResolveTarget` without a resolve action, or a resolve action without a target,
is an **error rather than an inference**.

| `Store` | `Resolve` | Legal | Meaning |
| --- | --- | --- | --- |
| `StoreKeep` | `ResolveNone` | yes | per-sample data kept, nothing resolved |
| `StoreKeep` | `ResolveAverage` | yes | both written out; costs the most bandwidth |
| `StoreDiscard` | `ResolveAverage` | **yes** | resolve in tile memory, never write the samples out |
| `StoreDiscard` | `ResolveNone` | yes | the pass produced nothing readable |
| any | `ResolveAverage` at `SampleCount` 1 | no | error naming the count |

The third row is the case the feature exists for. Rejecting it by analogy with
"discarding an attachment you also read" turns MSAA into a bandwidth regression.

**Validation is three checks, not one loop** — every colour attachment equal to
the pipeline's count, the depth-stencil attachment equal to them, and every
`ResolveTarget` exactly 1. The obvious loop over colour attachments covers the
first and misses the other two, and the missed cases then fail per backend
instead of at build.

**The resolve target is a new row in [033](033-render-api.md) §3.2's declared
access table**, as a write. So the edge to whoever reads it is inferred rather
than written, and it joins the non-overlap set: a resolve target whose view range
overlaps its own multisample source, or any attachment, is a build error naming
both views. Per 033 §3.2 that comparison is on view ranges, not texture handles.

**A present slot may be a resolve target.** A swapchain image is single-sampled
on every backend, so "render multisampled offscreen, resolve into the present
slot" is the **only** MSAA present path, not one of two.
[034](034-surface-present.md) §2's recorded slot usage gains resolve destination
beside render target; `BindPresent` still takes a `Frame`.

## 5. Resolve arithmetic, which is three answers and not one

**Colour: a box average of the pixel's own samples, in linear space.**

$$
C_{\text{resolved}} \;=\; \frac{1}{N}\ \operatorname*{tree\,\Sigma}_{i=0}^{N-1} \operatorname{lin}(C_i)
$$

Three parts of that formula are load-bearing, and each has a predecessor bug
behind it:

- **the pixel's own samples, and no others.** The predecessor's
  `imageutil/resize.go` is a separable linear filter at explicitly "8-bit
  precision", so neighbouring pixels leak in. The result looks smoother, passes
  any eyeball test for "AA is working", and is not a resolve.
- **in linear space.** The predecessor's `passAntialiasing` gamma-corrects first
  and averages after. The symptom is edges uniformly too dark, which reads as an
  anti-aliasing quality difference rather than a colour-space bug. An sRGB
  attachment decodes, averages, and re-encodes — fixed function per
  [001](001-device-resources.md), never in the stage.
- **$\operatorname{tree}\Sigma$**, a pairwise tree, not a running sum. For $N$ a
  power of two and $N$ identical values $v$ the tree sum is exactly $Nv$ and the
  divide is exact, so a uniformly covered pixel resolves **bit-exactly**. A
  running sum computes $3v$ at $N=4$, which needs 26 mantissa bits and rounds.
  §9's first exact row exists only because this is specified.

**Depth: refused.** An averaged depth lies on no surface at all. It stays silent
until something reconstructs world position from depth — 005's own recommended
G-buffer workaround — and then reads as halos or z-fighting far from the cause. A
depth resolve *mode* (sample zero, min, max) is a Vulkan extension rather than
core, a filter on Metal, and absent from GLES 3.1, so there is no portable answer
to pick either. A multisampled depth-stencil attachment is **legal and is the
normal case**: tested per sample, which is what makes coverage mean anything,
usually stored `StoreDiscard`. Only its *resolve* is refused.

**Integer formats: refused.** A `Uint`/`Sint` attachment has no defined average
on most APIs. The error names the format.

**Unorm targets quantize once, at the end.** The samples are decoded to `f32`,
averaged by the formula above, and re-quantized round-to-nearest-even, matching every conversion row in
[002](002-compute-model.md) §6.2 and both target APIs, rather than ties away
from zero. Stating the tie rule is not pedantry: an unstated one is a second
CPU-versus-Metal divergence at exactly the edge pixels §9 already can only
bracket, and it would be invisible in the interior where every sample agrees.

## 6. The rasterizer shades once per pixel per primitive

This is a semantics decision, not an implementation one, and the predecessor got
it wrong under this name: `render/options.go` spells `MSAA int`,
`render/raster.go` allocates the framebuffer at `Width*MSAA x Height*MSAA`, and
every fragment is shaded at every sample. **That is supersampling.** It produces
different and more expensive numbers, and answering 005's oracle question with it
answers a different question.

MSAA's defining property is one shade per pixel per primitive with coverage
evaluated per sample. The per-fragment chain of [035](035-cpu-rasterizer.md) §5
becomes, mechanically:

```
per pixel, per primitive:
  1. coverage: 035 §2's fill rule at each of §2's N sample positions
  2. scissor and render area, tested PER SAMPLE
  3. AND with the pipeline's static SampleMask          ← before anything writes
  4. if the resulting mask is 0: skip the pixel entirely
  5. fragment stage, ONCE, at the pixel centre           ← 035 §5 step 1, intact
  6. accel.Discard() clears the whole mask: no sample writes
  7. per covered sample: stencil → depth test → depth write
  8. per surviving sample: blend → write mask → attachment write
```

Step 3's position is the trap. [035](035-cpu-rasterizer.md) §8.1 records this
corpus getting the analogous order wrong once already, in the spec before the
code: a loop that tests and writes depth per sample and applies the mask after
writes depth for samples the mask killed, and [032](032-stage-abi.md) §4.2's
guarantee breaks again in a new place. Step 2 is the second trap.
`internal/raster/raster.go` bounds coverage by a per-pixel scissor rectangle;
under MSAA an edge pixel whose centre is outside the render area can still have
covered samples inside it, and the symptom is a one-pixel difference at the
render-area boundary that appears only with MSAA on.

**The coverage mask is a static pipeline field**, `SampleMask uint32`, ANDed in
at step 3. Bit $i$ is sample $i$ of §2's table, which is why that order is
normative. It is fixed function and it is an integer AND, so it is exactly
checkable and entirely pattern-independent: a mask of 0 covers nothing, whatever
the device's positions are.

Storage in `internal/raster/pipeline.go` is indexed per pixel today
(`(f.Y*t.W + f.X) * 4`). Per-sample storage is the change; primitive assembly in
`draw.go` is untouched, which is the boundary showing MSAA is a change to
coverage and storage only.

## 7. What the fragment stage does not gain

**Per-sample shading: refused.** A stage that runs per sample, or reads a sample
index or a sample position, has an output that is a function of the device's
sample pattern — so it is untestable against any GPU by §2's table, and in the
CPU rasterizer it is supersampling by another name.

The refusal buys back an invariant. [032](032-stage-abi.md) §9 says stage errors
arrive **at generation, never at pipeline creation**, and a per-sample built-in
breaks it: the generator cannot know the pipeline's sample count, so "this stage
uses `SampleIndex` and the pipeline is single-sampled" would be the first
stage-related check that must land at pipeline creation. With no such built-in
there is no such check, and 032 §9 stands unamended. That is the price of ever
adding per-sample shading, recorded in §15 rather than paid now.

So `accel.Fragment` gains **nothing**: no `SampleIndex`, no `SamplePosition`, no
`SampleMask` in or out. A stage-written coverage mask is also a per-sample write
from a stage, which 032 §4.3 keeps out of the baseline for ordering reasons; §6's
static mask is the fixed-function form of the same idea.

**Fetch of a multisampled texture: refused by name.** `accel.Fetch` has no sample
index and will not gain one, because a per-sample fetch makes the oracle compare
sample positions — consistent with 005 forbidding attachment feedback and with
004's sampler refusal. The consequence, stated plainly: **a resolve is the only
way to see MSAA data**, and it has to be enough for every consumer in the corpus.

**Alpha-to-coverage: refused.** There is no portable mapping from an alpha value
to a bit pattern, which is the same reason [035](035-cpu-rasterizer.md) refuses
line rules.

## 8. ROA and MSAA stay mutually exclusive

005 defers the choice here. The answer is that they stay exclusive, and the
argument is the corpus rather than the difficulty: [006](006-backends.md)'s
matrix has ROA `cap` everywhere and `no` on WebGPU, MSAA `cap` everywhere, so a
portable per-sample ordering rule would be "a mechanism with no case in the
corpus to prove it" — verbatim the argument [033](033-render-api.md) §5 used to
decline pass merging.

**The enforcement point is what matters, and it is not capability absence.**
`internal/cpu/profile.go` already sets `RasterizerOrderedAccess: true`, so the
moment the CPU backend rasterizes MSAA it becomes the first device reporting
both, absence stops expressing the exclusion, and an ROA pipeline at sample count
4 builds and runs — on the oracle, which is the worst place for it to be legal.

So it is a **pipeline-creation error**, naming the stage's ROA requirement and
the sample count together, checked whenever the stage record carries the ROA
requirement and `SampleCount > 1`.

## 9. What the oracle can still prove

This is the section 005 deferred MSAA over. Each claim below survives the sample
pattern being unknown, and each is written in one of
[011](011-conformance-harness.md) §4's forms — exact bits, structured equality,
or a bound derived from [008](008-numerics.md). "Compare the resolved image
within a tolerance" is not writable in this repository: 011's static conformance
check scans comparison call sites and rejects tolerance parameters.

**The comparison target is the resolved single-sample image, always.** Never a
per-sample buffer. A per-sample comparison between the CPU and Metal compares
sample positions, not correctness: it passes on one vendor, fails on another, and
produces a bug report about MSAA that is really a report about geometry.

| Claim | Side | Why it survives an unknown pattern |
| --- | --- | --- |
| a **uniformly covered pixel** resolves bit-exactly to the fragment's value | exact | all $N$ samples carry one value, and §5's tree sum of $N$ identical floats over power-of-two $N$ is exact. Interior pixels are in the exact column on every device, whatever its positions |
| **an axis-aligned edge on a pixel boundary** produces an image that is exact everywhere | exact | every published position is strictly inside $(0,1)^2$, so **for any pattern whatsoever** every pixel is fully covered or fully uncovered. This is the strongest MSAA fixture available and it needs no bracket at all |
| **attachment routing** — which output field lands in which attachment, and which resolve target it reaches | exact | an integer mapping; MSAA adds a second one and no arithmetic |
| **static `SampleMask` semantics** — mask 0 covers nothing; a full mask equals the unmasked render | exact | an integer AND, evaluated before any write |
| **validation outcomes** — unequal sample counts, a multisampled resolve target, a refused format, a refused count | exact | not arithmetic at all |
| a **partially covered pixel's** resolved value lies within the bracket $[\min, \max]$ of the values contributing to **that pixel** and the loaded or cleared value | bounded, pattern-independent | a box average is a convex combination; the endpoints are exact inputs the test already has, so it is interval containment and not a tolerance |
| **some** partially covered pixel is **strictly between** the bracket endpoints | bounded | a resolve that returns one sample, or a depth buffer that keeps one value per pixel, can only ever return an endpoint |
| **which pixels differ from the 1× render** lie within one pixel of a primitive edge | bounded, structural | coverage differs only where a pixel is partially covered, which is geometry and not positions |

And what is **not** provable, stated so nobody writes it:

- the value of any individual sample, on any device but the CPU;
- the resolved value of a specific edge pixel, beyond the bracket above;
- the number of covered samples in a specific edge pixel;
- anything that depends on the device's positions matching accel's.

**A bracket is a weak claim and it must not be asked to catch strong bugs.** It
is closed, so an endpoint satisfies it, and on a two-colour fixture almost
everything satisfies it — including a resolve that leaks a neighbour's samples,
because the neighbour's value is inside the bracket too. So the bracket catches
exactly one thing, a value outside the convex hull of the contributing values,
and every other claim in §10 is carried by an exact fixture or by the
strictly-between row above. This is [035](035-cpu-rasterizer.md) §8.1's rule
applied before the tests exist rather than after they fail to catch a
reinstated bug.

The bracket is also the one form that has to fit 011 §4 without a new comparison:
interval containment whose endpoints are exact inputs, expressible as a
structured comparison. **If `numeq` cannot express it through an existing form,
that is a finding for 011** and gets fixed there rather than papered over with a
tolerance here.

## 10. The corpus

Amending [035](035-cpu-rasterizer.md)'s scope line explicitly: "no multisampling"
becomes "sample counts 1, 2, and 4 with §2's mandated positions". Every entry
declares its side, per 035 §6, and every pixel-producing entry runs on the CPU
backend **and** on Metal from the same recorded graph, per 035 §7.1.

| Test | Side | What it catches |
| --- | --- | --- |
| **Uniform coverage is bit-exact.** A full-screen quad at 4×, and the interior of a large triangle, equal the 1× render bit for bit. | exact | A filtered downscale, and §5's running-sum rounding: both change interior pixels, which a box average of one pixel's own samples cannot. |
| **Axis-aligned edge on a pixel boundary.** Two quads meeting on integer $x$, different colours; the whole 4× image is compared bit for bit against the analytic image. | exact | The load-bearing entry, and the only strong one at an edge. Every position is inside $(0,1)^2$, so every pixel is fully covered either way — a neighbour-leaking resolve blends across the boundary and is caught here, where the bracket cannot see it. |
| **A diagonal edge stays in the hull, and is not always an endpoint.** A triangle over a cleared attachment: every pixel is in $[\min, \max]$ of the fragment value and the clear, **and at least one is strictly between them**. | bounded | Containment alone catches only a value outside the hull. The strictly-between half is what a resolve returning sample 0, the max, or the clear cannot satisfy. |
| **Differing pixels are edge-adjacent, with a varying output.** The fragment output varies across each pixel (a value derived from `Coord().xy`); the 4× and 1× images differ only within one pixel of an edge. | bounded, structural | Supersampling — which changes interior values too, so the difference set stops being edge-local. **The varying output is required**: with a flat colour, supersampling and MSAA agree everywhere and this entry proves nothing. |
| **sRGB resolve.** An sRGB attachment half covered by a known colour over a known clear; the resolved value matches the linear-space average, not the average of encoded bytes. | exact against a computed reference | The predecessor's gamma-then-average bug, whose only symptom is "edges look dark". |
| **`SampleMask` 0 writes nothing**, and a discarding stage at 4× writes no depth for any sample. | exact | §6's step 3 applied after the writes instead of before — the per-sample port of 035's discard entry. |
| **Depth is per sample.** Two triangles crossing at a shallow angle; at least one crossing pixel resolves **strictly between** both colours. | bounded | A rasterizer keeping one depth value per pixel. It yields one colour or the other — a bracket endpoint — so a containment-only assertion passes with the bug in place. |
| **Render area under MSAA.** Geometry crossing the area edge; no sample outside the area is written. | exact | §6 step 2 left per pixel. |
| **Validation.** Unequal colour counts; a depth attachment at a different count; a multisampled resolve target; a resolve at count 1; `ResolveTarget` without an action; a resolve target overlapping its source. | exact | The single-loop validation, which passes four of these six. |
| **A multisampled texture is not readable.** `ReadTexture` and a texture-to-buffer copy both refuse, naming the resolve path. | exact | The CPU-passes/Metal-fails inversion of §3. |
| **ROA at sample count 4 is refused at pipeline creation**, naming both. | exact | Enforcement through capability absence, which stops working the moment the CPU profile reports both. |
| **Capability absent.** A forced CPU profile without multisampling: a 4× pipeline fails with the specified error, and the harness asserts the **selection**, not only the result. | exact | 011 §3's present/absent pair, which a result-only assertion does not satisfy. |
| **Present through a resolve.** A headless frame rendered multisampled and resolved into the present slot. | exact | A slot that accepts a render-target binding but not a resolve destination. |
| **MSAA transient sizing.** A multisampled transient's planned bytes are $N$ times the single-sampled equivalent. | exact, structural | The `textureBytes` trap, whose real symptom is corruption in an unrelated node. |

## 11. Errors, and where each arrives

Per [033](033-render-api.md) §6, nothing arrives at submission.

| Error | Where |
| --- | --- |
| a sample count outside `{1, 2, 4}`, or one the format does not support — naming the supported set | texture and pipeline creation |
| `SampleCount > 1` with a copy or shader-visible usage, or with mips or layers | texture creation |
| an ROA stage requirement with `SampleCount > 1`, naming both | pipeline creation |
| unequal sample counts across colour attachments, or depth against colour, or a pipeline against its pass | graph build, naming the pipeline, node, and attachment index |
| a resolve target above sample count 1; `ResolveTarget` without an action or the reverse; a resolve on a depth or integer format | graph build, naming the attachment |
| a resolve target overlapping its source or any attachment | graph build, naming both views |
| `ReadTexture` or a copy of a multisampled texture, naming the resolve path | at the call, like every other transfer validation |

## 12. What this amends elsewhere

| Where | Change |
| --- | --- |
| [006](006-backends.md) graphics table | the CPU `Multisampling` row becomes `yes (reference)` |
| [006](006-backends.md) prose after that table | it says MSAA is excluded because the pattern is not standardized; §2 replaces that with the mandated pattern and the verified per-backend form |
| [005](005-graphics.md) child table and open questions | both say this child spec does not exist |
| `internal/cpu/profile.go` | `Multisampling: true`, and the comment that currently says it is `no` |
| `internal/cpu/cpu_test.go` | the assertion that multisampling is `cap` on every target and absent on the CPU |
| [035](035-cpu-rasterizer.md) scope | "no multisampling" becomes counts 1, 2, 4 |
| [033](033-render-api.md) §2.1 | the fixed column's "Sample count, fixed to 1" |
| [034](034-surface-present.md) §2 | the present slot's recorded usage gains resolve destination |
| `limits.go` `FormatInfo` | the per-format supported-count set |
| `texturealloc.go` | `validateTexture`, `textureBytes`, `readTexture` |

Flipping 006's row without `profile.go` and `cpu_test.go` leaves a failing
assertion that reads like an unrelated regression, which is why they are one
change and not three.

## 13. Costs

- **The oracle is genuinely weaker where MSAA touches.** Diagonal edge pixels
  move from bounded-by-interpolation to bounded-by-a-hull, which is a wider
  claim, and no per-sample value is comparable at all. §9 is the accounting: the
  interior and the axis-aligned edge stay exact, diagonal edges get a weaker but
  still falsifiable claim.
- **Per-sample shading is refused**, so per-sample derivatives and per-sample
  stochastic effects cannot be written at all.
- **Multisampled transients do not alias.** A `StoreDiscard`-plus-resolve
  attachment is the ideal aliasing candidate — pass-scoped, never read after the
  pass — but backends impose their own alignment on multisampled resources, and a
  right byte count is not a right placement. So they are excluded from the packer
  until a backend confirms the rules: $N$ times a full attachment, held for the
  graph's life.
- **Memory and bandwidth.** $N\times$ the attachment bytes; on a non-tiler the
  multisample buffer is real traffic, and `StoreKeep` plus a resolve pays twice.
- **A published position table is a promise.** Changing it later changes every
  bounded edge comparison in the corpus.

## 14. Done

- a `SampleCount` of 3, of 8, or of a value the format does not support is
  refused at creation naming the supported set — no silent downgrade — as is one
  with a copy or shader usage, a mip, or a layer;
- `ReadTexture` of a multisampled texture names the resolve path on the CPU
  backend as well as on Metal;
- a multisampled transient's planned bytes are exactly $N$ times its
  single-sampled equivalent, asserted on the plan and not on an image;
- a full-screen quad at 4× resolves bit-identically to the 1× render, and that
  test fails when the resolve sums sequentially instead of as a tree;
- the axis-aligned-boundary fixture matches its analytic image **exactly**, and
  fails when the resolve reads a neighbouring pixel's samples;
- at a diagonal edge every pixel is inside the hull **and at least one is
  strictly between** the endpoints, with no tolerance parameter at any comparison
  call site — the strict half confirmed by reinstating a resolve that returns
  sample 0, which containment alone accepts;
- with a `Coord()`-varying fragment output, the 4× and 1× images differ only
  within one pixel of an edge, confirmed to fail when the rasterizer shades per
  sample instead of per pixel;
- an sRGB resolve matches the linear-space average and not the encoded-space one,
  and an unorm resolve matches §5's stated tie rule;
- `SampleMask` 0 writes no attachment and no depth, and a discarding stage at 4×
  writes no depth for any sample;
- each of §10's six validation cases is refused at build with its named view,
  node, or attachment index;
- an ROA pipeline at sample count 4 is refused at **pipeline creation**,
  confirmed on a CPU profile that reports both capabilities;
- a forced profile without multisampling refuses a 4× pipeline, and the harness
  asserts which implementation was selected; and
- a headless frame renders multisampled and resolves into the present slot.

## 15. Open questions

- **Per-sample shading**, refused in §7. Adding it means adding a sample-index
  built-in, which means the first stage check that cannot happen at generation:
  [032](032-stage-abi.md) §9's "errors, all at generation" would gain an
  exception and [033](033-render-api.md) §2.2 a row. The price is known; nothing
  in the corpus needs it yet.
- **Sample counts 8 and 16.** Two more rows of positions, and a device that needs
  them.
- **Whether multisampled transients can alias**, which needs a backend's
  placement and alignment rules for multisampled resources, not an argument.
- **Depth resolve**, if a consumer ever needs resolved depth rather than
  reconstructing position from a resolved G-buffer. It would arrive as a mode,
  never as an average, and only with a portable definition for all target
  backends.
- **Where this sits in [009](009-sequencing.md)'s order**, relative to
  [035](035-cpu-rasterizer.md) §8's six implementation steps. Landing before step
  6 means Metal validates MSAA and single-sample at once; landing after means
  refactoring the rasterizer for per-sample storage on an already-green corpus.

## Note on scope

This spec is long, and the seam is its own: the API half — sample positions,
texture and pipeline state, resolve, rasterizer semantics — lands in `limits.go`,
`texturealloc.go` and `internal/raster/`, while the oracle half — what can still
be proved once sample positions exist, and the corpus that proves it — lands in
`internal/cpu/profile.go` and the differential. [005](005-graphics.md)'s
obligation list draws the same line, six API obligations against "CPU-oracle
limits" as its own row.

Splitting it into `041-msaa` and a sibling oracle spec is available and is not
taken here, because the oracle limits are the reason the API half is shaped as
it is and separating them invites implementing the API against a weaker check
than this document requires. If the split happens, it happens with that argument
answered, not for length.
