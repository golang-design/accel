---
title: "Render pipelines, passes, attachments, and draws"
status: in progress
layer: device
depends_on:
  - 001-device-resources.md
  - 003-command-graph.md
  - 005-graphics.md
  - 032-stage-abi.md
---

# Render pipelines, passes, attachments, and draws

[005](005-graphics.md)'s second child spec: the public surface a renderer calls.
[032](032-stage-abi.md) fixed what a stage looks like; this spec fixes what
consumes one — the pipeline object, the pass node, and the draws recorded into
it — and what the graph builder validates.

Everything here is recorded, never executed at the call. A render pass is a node
in [003](003-command-graph.md)'s graph like every other node, and the reason is
[`000-decisions.md`](000-decisions.md) decision 1: every backend records a render
pass into a command buffer anyway, so recording costs graphics nothing.

## 1. The one structural decision, restated because everything follows from it

**One render pass is one node. A draw is not a node.**

```
   graph                          inside one pass node
┌──────────┐
│ upload   │                  ┌──────────────────────────────┐
└────┬─────┘                  │ load actions                 │
     │                        │  draw 0 ──▶ draw 1 ──▶ draw 2│  recorded order,
┌────▼──────────┐  ═══════▶   │  no barriers between them    │  never reordered
│ geometry pass │             │ store actions                │
└────┬──────────┘             └──────────────────────────────┘
     │  edge inferred from declared access
┌────▼──────────┐
│ lighting pass │  compute dispatch, reads the attachments
└───────────────┘
```

The pass is the unit at which synchronization is expressible: Vulkan cannot
barrier inside a render pass in the general case, tile-based hardware physically
cannot — attachment contents live in tile memory until the pass ends — and
Metal's encoder has the same shape. Draw granularity would promise an ordering
the hardware cannot provide.

Two caller-visible rules follow. Draws inside a pass execute in recorded order
and the builder never reorders them, because blending is order dependent. And
the builder inserts no barriers inside a pass, because per-pixel ordering between
draws is the ROP's job.

## 2. The pipeline

```go
p, err := dev.NewRenderPipeline(accel.RenderPipelineDescriptor{
	Vertex:   &shaders.GeometryVS,
	Fragment: &shaders.GeometryFS,
	VertexBuffers: []accel.VertexBufferLayout{{
		Stride: 32, StepMode: accel.StepVertex,
		Attributes: []accel.VertexAttribute{
			{Location: 0, Format: accel.AttrFloat32x3, Offset: 0},
			{Location: 1, Format: accel.AttrFloat32x3, Offset: 12},
			{Location: 2, Format: accel.AttrFloat32x2, Offset: 24},
		},
	}},
	Primitive: accel.PrimitiveState{
		Topology: accel.TriangleList,
		FrontFace: accel.CounterClockwise, // no unset state
		Cull: accel.CullBack,
		Fill: accel.FillSolid,
	},
	DepthStencil: &accel.DepthStencilState{
		Format: accel.Depth32Float,
		DepthTest: true, DepthWrite: true, DepthCompare: accel.Less,
	},
	Targets: []accel.ColorTargetState{
		{Format: accel.RGBA8Unorm, WriteMask: accel.WriteAll},
		{Format: accel.RGBA16Float, WriteMask: accel.WriteAll},
	},
	Label: "geometry",
})
```

A pipeline is compiled once, outside the frame loop, and referenced by nodes.
Creating one is expensive on every backend — it compiles shaders and specializes
fixed-function state — which is the reason it is an object rather than a
parameter block.

### 2.1 Fixed at creation, and why

| Fixed at pipeline creation | Recorded per pass or per draw, or bound |
| --- | --- |
| Vertex input layout and attributes | Vertex and index buffer bindings |
| Vertex and fragment stage code | Bind group contents, through 003's rebinding |
| Topology, winding, cull, fill, depth bias | Buffer contents |
| Depth test, compare, write; stencil ops and masks | Stencil reference value |
| Blend state and write mask, per attachment | Blend constant |
| Colour attachment formats and count, depth format | Viewport and scissor, per pass |
| Sample count, fixed to 1 | Draw counts, first index, instance count |

The split is not taste. D3D12 compiles a pipeline state object, Vulkan a
`VkPipeline`, Metal a `MTLRenderPipelineState`, and each takes attachment
formats, blend, and vertex layout as compile-time inputs. The dynamic column is
close to the intersection of what all four can change without recompiling.
Exposing more would mean either a backend that silently recompiles mid-frame — a
notorious stutter source — or a backend that fails at a call the API said was
legal.

**Front-face winding has no unset state.** Metal's default disagrees with GL's,
and getting it backwards keeps back faces instead of front faces: the silhouette
stays right while every per-pixel attribute comes from the wrong surface, so it
reads as a shading bug. A zero value that meant "backend default" would make that
the easiest thing to write.

### 2.2 What the stage record validates

Pipeline creation checks the descriptor against [032](032-stage-abi.md)'s
generated stage record, and each check exists because its absence is silent:

| Check | What goes wrong without it |
| --- | --- |
| `len(Targets)` equals the fragment stage's output field count | A field lands in no attachment, or an attachment holds undefined contents. |
| each target's format admits the field's type | An `f32` field written to a `u32` target reinterprets bits. |
| declared attribute formats match the vertex stage's parameter types | The fetch converts wrongly, and the geometry is subtly deformed rather than absent. |
| attribute `Location`s are dense and match the stage's attribute order | An attribute silently reads another's data. |
| the varyings slot count fits the device limit | 032 §3.2. |
| `len(Targets)` fits the device's attachment limit | Decision 6: absence is reported, not discovered. |
| `DepthStencil` present iff the pass will have a depth attachment | Checked again at build, where the pass is known. |

A pipeline with a stencil state but no stencil aspect in its depth format is an
error at creation, naming both.

### 2.3 Depth bias

Constant, slope, and clamp, present because shadow mapping is unusable without
it and because emulating it in the vertex stage is wrong for sloped geometry:

$$
z' = z + \text{slope} \cdot m + \text{constant} \cdot r,
\qquad m = \max\!\left(\left|\frac{\partial z}{\partial x}\right|, \left|\frac{\partial z}{\partial y}\right|\right)
$$

where $r$ is the smallest resolvable depth difference for the attachment's
format. The clamp bounds the total. [035](035-cpu-rasterizer.md) implements this
formula, and the comparison against a GPU is within a bound rather than exact,
because $r$ is defined per format and per backend.

## 3. The pass node

```go
pass := rec.RenderPass(accel.RenderPassDescriptor{
	Color: []accel.ColorAttachment{
		{View: albedo, Load: accel.LoadClear, Clear: accel.ClearColor{}, Store: accel.StoreKeep},
		{View: normal, Load: accel.LoadClear, Clear: accel.ClearColor{}, Store: accel.StoreKeep},
	},
	Depth: &accel.DepthAttachment{
		View: depth,
		Load: accel.LoadClear, Clear: accel.ClearDepth{Depth: 1.0},
		Store: accel.StoreDiscard,
	},
	Area: accel.Rect{W: 1920, H: 1080},
	Label: "geometry",
})
pass.SetPipeline(p)
pass.SetVertexBuffer(0, verts)
pass.SetIndexBuffer(idx, accel.Index32)
pass.SetBindings(bindings)
pass.DrawIndexed(accel.DrawIndexed{IndexCount: n, InstanceCount: 1})
```

`rec.RenderPass` returns a recorder for one pass, and the pass becomes one node.
Calls on it record state and draws; nothing executes.

### 3.1 Load and store actions

| Load | Meaning |
| --- | --- |
| `LoadClear` | Clear to a stated value. |
| `LoadKeep` | Preserve existing contents. |
| `LoadDontCare` | Contents are undefined at pass start. |

| Store | Meaning |
| --- | --- |
| `StoreKeep` | Contents are readable after the pass. |
| `StoreDiscard` | Contents are undefined after the pass. |

These are not decoration. On a tiler, `LoadClear` costs nothing while a
full-screen clear draw costs a full write of tile memory, and `StoreDiscard` on a
depth attachment nothing reads afterwards saves writing the whole depth buffer
out every frame.

`LoadDontCare` is load-bearing for the graph. An attachment loaded `DontCare`
carries no data dependency on whatever wrote it last, so the read-after-write
edge disappears. The write-after-write edge does not: an earlier node's writes to
the same memory still have to be ordered before this pass's, or they land
afterwards. And where the attachment is a builder-owned transient, its live range
starts at the pass, so it can alias unrelated transients — caller-created
textures are never aliased, per 003.

**A clear value has no zero-value special case.** Clearing depth to 0 by leaving
a field unset is a real hazard; the predecessor defaulted zero to 1.0 for exactly
this reason, which is a default that is wrong for reverse-Z. So `LoadClear`
carries an explicit value, `Clear` without `LoadClear` is an error rather than
ignored, and the builder rejects a depth clear outside `[0, 1]`.

### 3.2 Declared access, inferred edges

A pass node declares, and the builder infers edges from, exactly this:

| Declared | As |
| --- | --- |
| each colour attachment | write; also a read when `LoadKeep` |
| the depth-stencil attachment | write plus read when depth writes are on; read only when the pipeline tests without writing |
| every vertex, index, uniform and storage buffer bound by any draw in the pass | read |
| every texture fetched by any draw | read |
| any indirect argument buffer | indirect read, its own access kind |
| the render area | validated against every attachment's extent |

From that the builder infers edges and computes barriers exactly as it does for a
dispatch — which is the point of 003 declaring access rather than order. The
geometry-pass-to-lighting-pass edge in section 1's diagram has no caller
involvement, and the barrier that makes a render target readable by compute is
computed, not written.

The indirect argument buffer is a **distinct access kind** and not an ordinary
read. The barrier from compute-write to indirect-read is its own pipeline stage
on Vulkan and D3D12, and getting it wrong is a classic hang rather than a wrong
answer.

### 3.3 Feedback: the rule, and the two things it is not

**Forbidden:** an overlapping texture subresource that is both an attachment and
shader-visible in the same pass. Sampling or fetching the mip/layer range
currently bound as a colour or depth attachment is a build error naming both
views, regardless of load action.

**Legal, and not to be mistaken for feedback:**

- Fixed-function attachment read-modify-write. `LoadKeep`, blending, depth test,
  stencil test and their writes are ordered by the ROP. Describing a blended pass
  as "reading a resource it writes" would reject every blended or depth-tested
  pass ever written.
- Disjoint subresources. A different mip or a different array layer is a
  different subresource and is legal where the backend can bind those views
  independently.

So pass validation compares **view ranges**, not texture handles. Comparing
handles is insufficient in one direction (it rejects the legal disjoint case) and
sufficient in the other only by accident.

A depth attachment with writes disabled, used only by the fixed-function test, is
a read-only attachment and is legal everywhere. Binding that same depth
subresource for fetching in the fragment stage is still forbidden feedback.

### 3.4 Viewport, scissor, and area

The render area is per pass and validated at build against every attachment
extent. Viewport and scissor are per pass and default to the area. They are not
per draw: a per-draw viewport exists on some backends and not others, and the
uses that want it — split-screen, atlas rendering — are expressible as separate
passes with no loss.

## 4. Draws

| Call | Arguments |
| --- | --- |
| `Draw` | vertex count, instance count, first vertex, first instance |
| `DrawIndexed` | index count, instance count, first index, vertex offset, first instance |
| `DrawIndirect` | an argument buffer view |
| `DrawIndexedIndirect` | an argument buffer view |
| `DrawIndirectCount` | an argument buffer, a count buffer, and a build-time maximum |

Instancing is not a separate call; it is the instance count, which is why the
non-instanced case is the instanced case with a count of one. A caller never
picks between two entry points for the same drawing.

### 4.1 Per-object data without re-recording

A scene of a thousand objects replayed every frame with new transforms is the
case that decides whether this design works under 003's immutability.

Object $i$'s draw is recorded with a **fixed byte offset** into a uniform buffer:

$$
\text{offset}(i) \;=\; i \cdot \text{align}(\text{sizeof}(T),\ \text{minUniformBufferOffsetAlignment})
$$

The offsets are structure, so they are baked into the graph. The transforms are
contents, so they are rewritten every frame through 003's first kind of
variation. Nothing is re-recorded and the graph replays.

The alignment is a device limit, not `sizeof(T)`, and it is why the stride is
computed rather than assumed: a 68-byte transform on a device with 256-byte
uniform alignment strides by 256, and a caller who wrote `i * 68` gets garbage
for every object but the first.

**There are no push constants.** Vulkan push constants, D3D12 root constants and
Metal `setVertexBytes` put per-draw values into the command stream itself, so
changing one is a re-record — precisely the cost decision 1 exists to avoid.
Offering them would make the convenient path the one that defeats the model. The
recorded-offset mechanism has a native expression everywhere: dynamic uniform
buffer offsets in Vulkan, buffer offsets in Metal, root descriptor offsets in
D3D12.

**The honest cost: the object count is baked in.** A scene that gains an object
needs either a rebuilt graph or a graph recorded for a fixed maximum with absent
objects issuing zero-instance draws. Zero-instance draws are cheap but not free.
This is 003's "cache graphs keyed by shape" applied to scenes, and a renderer
with genuinely churning object counts will feel it.

### 4.2 Indirect draws and the count problem

Indirect **arguments** fit the model with no tension: graph structure is
unchanged and only numbers vary, which is 003's third kind of variation.

005 proposed an amendment to [003](003-command-graph.md) here — its third kind of
variation read "dispatch counts", and someone reading 003 alone would not find
draw counts there. **That amendment is already accepted in 003**, which now reads
"dispatch and draw counts" and records where the widening came from. Nothing is
outstanding; it is noted because this spec is where the draws that need it live.

Indirect **draw count**, where the device decides how many draws happen, is
genuinely in tension with immutability. The resolution matches what the compute
path already does for indirect dispatch: a node records a **build-time maximum**
and reads the actual count from a device-written buffer, clamped to that maximum.

$$
n_{\text{drawn}} \;=\; \min\big(n_{\text{device}},\ n_{\max}\big)
$$

The clamp is unconditional and costs no readback, exactly as the indirect
dispatch clamp does: it is performed on the device. A count exceeding the maximum
is clamped rather than silently truncating, and the clamp is reported through
003's run-time statistics — opt-in there, because reading a device-written count
back costs a transfer and a readback allocation. A graph that did not ask still
clamps, silently, which makes the maximum a caller obligation in release mode.

This is a capability, not a guarantee. D3D12 `ExecuteIndirect` and Vulkan
`vkCmdDrawIndirectCount` provide it directly, Metal has indirect command buffers,
GLES 3.1 has single indirect draws and no count buffer. Absence is reported per
decision 6, and a caller without it falls back to a fixed draw count with
zero-instance draws for the unused slots.

## 5. Decisions this spec closes

**Vertex input layout stays a descriptor** — 005's second open question. The
compiler knows the attribute *types*, and inferring the layout from the vertex
stage signature would remove one of two declarations. But byte offsets, strides,
and step modes are properties of the **buffer**, not of the kernel: the same
stage reads interleaved and planar buffers, and the same buffer feeds stages that
read a subset of its attributes. Inferring them would either guess a packing or
need annotations that are a descriptor by another name. What this spec does take
from the compiler is **validation**: section 2.2 checks the declared formats
against the stage's parameter types, so the second declaration cannot disagree
with the first without an error.

**Pass merging is not attempted** — 005's third open question. Vulkan subpasses
and Metal imageblocks let a deferred renderer keep its G-buffer in tile memory,
and the builder has the information to detect the pattern. It is not attempted at
this stage for a specific reason rather than a general one: 005's own
compute-consumer handoff, which is the shape this design targets, **merges on no
backend** — a compute pass reading an attachment forces it out of tile memory
everywhere. So the first implementation would be a mechanism with no case in the
corpus to prove it. Revisited when a pass-to-pass consumer exists, and left as a
possible automatic merge rather than a hint, because the builder's information is
better than a caller's.

## 6. Errors, and where each arrives

| Error | Where |
| --- | --- |
| descriptor disagrees with the stage record | pipeline creation |
| attachment count or varying slots over a device limit | pipeline creation |
| stencil state without a stencil aspect | pipeline creation |
| a pipeline used in a pass whose attachment formats or count differ | graph build, naming the pipeline, the node, and the attachment index |
| a clear value outside `[0, 1]` for depth | graph build |
| `Clear` set without `LoadClear` | graph build |
| render area outside an attachment extent | graph build |
| feedback: overlapping subresource as attachment and shader-visible | graph build, naming both views |
| a draw with no pipeline set | graph build |
| ~~a vertex buffer bound at a slot the pipeline does not declare~~ | **withdrawn 2026-08-24**; see below |

**The withdrawn row.** It was never implemented, and building it broke an
existing test that was right. Binding a buffer a draw does not fetch is
legitimate rather than a mistake: a caller may bind for the widest pipeline in a
pass and draw with a narrower one, and each draw copies the pass state standing
at the time, so the narrow draw carries a binding its layout does not name.
Refusing that would break a working pattern.

The defect the row was reaching for is real and was elsewhere. `RenderPass`
declared a read on every *bound* buffer while the lowering fetches only the
slots the layout names, so an unfetched binding still reached the node — an edge
against whatever wrote it, for a fetch that does not happen. That is fixed where
it is caused: a draw declares the reads it fetches. A binding nobody fetches now
costs nothing rather than an edge, and needs no refusal.

Recorded rather than deleted because the row is the second validation rule in
this project to be withdrawn for the same reason as check V23 — a rule that
sounds obviously right, forbids something a caller legitimately does, and was
never tested against a caller who does it.

Nothing in that table arrives at submission. That is 003's obligation restated:
an error that says only "type mismatch" is a defect in this design, and every one
of these names the node, the slot, and the recording call's source position.

## 7. Done

**Built as of 2026-08-24**: the pipeline object with its creation-time refusals,
the vertex input layout of §2 with §2.2's validation against the stage record,
by-value stage parameters as pass state, the pass node with declared access,
load and store actions, draws in recorded order, and execution on the CPU
backend — an offscreen triangle renders and its interior pixels match
([035](035-cpu-rasterizer.md) §8 step 1), attributes are fetched per vertex and
per instance and interpolated, `LoadClear` and `LoadKeep` differ, and a depth
attachment is cleared, tested against and written through.

Since then: the load-action edges are asserted on the graph, blended draw order
is checked with two overlapping draws whose result depends on it, blend state is
exposed and fixed at pipeline creation, indexed draws with `BaseVertex` match the
direct draw they replace, indirect draws clamp a device-written count to the
recorded maximum in every build mode, and **Metal runs the whole surface** —
compared against the CPU rasterizer pixel by pixel over seven cases.

**Outstanding**, and why this spec stays *in progress*:

| Row | Why it is not done |
| --- | --- |
| transient attachment aliasing under `DontCare` | the edges are asserted and so is the aliasing; what is untested is `DontCare` specifically *widening* it beyond what `Clear` already allows, and the two declare the same access so it may not |
| feedback rejection for an overlapping subresource | **blocked**, not merely unwritten: a stage cannot read a texture at all until [032](032-stage-abi.md) §5's texel fetch exists, so there is no way to construct feedback |
| the N-object frame at recorded uniform offsets (§4.1) | see deviation 1; the by-value channel answers the one-uniform case |
| a windowed surface reaching a screen | the `CAMetalLayer` drawable path is built ([034](034-surface-present.md) §8.1); the compositor handoff needs a display |

- a pipeline whose target count differs from its fragment stage's output field
  count is refused at creation, naming both numbers;
- a pipeline used in a pass with mismatched attachment formats is refused at
  build, naming the pipeline, node and attachment index;
- draws execute in recorded order, checked with two overlapping blended draws
  whose result depends on the order;
- an attachment loaded `DontCare` produces **no** read-after-write edge from its
  previous writer, and still produces the write-after-write edge — both asserted
  on the graph rather than on an image;
- a transient attachment loaded `DontCare` aliases an unrelated transient, which
  is the aliasing consequence of the edge above;
- `LoadKeep` preserves prior contents and `LoadClear` does not;
- feedback is rejected for an overlapping subresource and **accepted** for a
  disjoint mip and a disjoint array layer, which is the half a handle comparison
  gets wrong;
- a blended, depth-tested pass is accepted, confirming attachment
  read-modify-write is not treated as feedback;
- a graph of N objects at recorded uniform offsets, submitted twice with
  different transform contents, produces two different and individually correct
  frames with no re-recording, with the stride derived from the device's uniform
  alignment rather than from `sizeof(T)`;
- an indirect draw matches the equivalent direct draw, and on a device with the
  count capability a device-written count above the build-time maximum is clamped
  and reported.

### 7.1 Where the oracle stops: undefined contents

The CPU rasterizer is the oracle every other backend is checked against, and
there is exactly one region where it cannot be: **an attachment's undefined
contents.**

`LoadDontCare` says prior contents are undefined at the start of a pass, and
`StoreDiscard` says they are undefined at the end. Both backends honour that and
they honour it differently, because the shapes differ:

| | CPU | Metal |
| --- | --- | --- |
| `LoadDontCare`, untouched pixel | the framebuffer aliases the caller's buffer, so prior contents remain | a fresh texture, so whatever that memory held |
| `StoreDiscard` | nothing to skip; the pass already wrote the caller's buffer | the blit back is skipped, so the buffer is untouched |

Neither is wrong. The rule is therefore stated rather than tested away:

> **What a pass writes is defined and every backend must agree on it. What a
> pass does not write is undefined, and a caller who reads it has a bug the API
> cannot catch.**

The differential asserts the first half and deliberately asserts that the second
half *does* differ. That second assertion is the one worth having: a backend
that quietly began preserving undefined contents would be making a promise this
spec does not, and the next backend would inherit it.

## 8. Deviations

### Deviation 1: the draw-time uniform channel was removed

**What the spec required.** Section 6 makes a uniform buffer at a recorded byte
offset the mechanism a draw parameterises through, with the stride derived from
`minUniformBufferOffsetAlignment`. The spec never described a by-value uniform
on a draw.

**What was built instead.** `RenderPass.Draw` was implemented with a
`uniforms ...UniformValue` variadic, which lowered by appending values in the
order the caller wrote them and ignoring `UniformValue.Index`. Two uniforms
passed out of order bound to each other's parameters, and a subset shifted every
parameter after the omitted one.

**Why it was removed rather than fixed.** The placement bug is the symptom. Two
stages are compiled independently, so each indexes its own uniform space from
zero:

$$
\text{vertex } u[0] \;\ne\; \text{fragment } u[0]
$$

One slice cannot serve both, and the fix is either a per-stage pair of slices or
a pipeline-wide index space the generator cannot assign — both of which widen a
public surface that had just been frozen, for a channel the spec does not
describe.

**What holds now.** `kernel.Stage` records `Uniforms []StageUniform`, and graph
build refuses a draw whose stage declares a by-value parameter, or whose vertex
stage reads an attribute:

```
render pass "uniformed" draw 0: GeometryVS declares the by-value parameter "xf",
and no render path supplies one yet (specs/033-render-api.md deviation 1)
```

Refused rather than passed an empty slice, because the generated adapter would
then index past its end and the diagnostic would come from a backend.

**Closed 2026-08-24.** Both channels are built, and the by-value one took the
shape section 9 left open: **pass state, one call per stage**.

```go
pass.SetVertexUniform(0, Transform{Scale: 0.5})
pass.SetFragmentUniform(0, Tint{Colour: ...})
```

Two calls rather than one because the index spaces are per stage — the
inequality above is why a single indexed call would have to guess which stage a
value was for. Pass state rather than a `Draw` field because it matches
`SetPipeline` and `SetVertexBuffer`, and a value shared by several draws is
written once; a draw captures what is set when it is recorded, so a later call
does not reach back.

Section 6's recorded-offset mechanism is still the answer for a thousand-object
frame and is still unbuilt. It is a different problem: rewriting a thousand Go
values per frame costs what the offsets exist to avoid.

Build checks each stage against its record — every declared parameter has a
value, no value is set for a stage that declares none, and the type matches by
name. The vertex layout of section 2 is built with it, so a stage now reads both
its attributes and its parameters.

## 9. Open questions

- **Whether section 6's recorded uniform offsets are still needed.** The by-value
  channel answers the one-uniform case. A thousand-object frame is a different
  problem — rewriting a thousand Go values per frame costs what the offsets exist
  to avoid — so the mechanism is expected, but the first renderer to need it
  should say what shape it wants.

- **Whether a resize can avoid a graph rebuild.** Carried from 005 and owned by
  [034](034-surface-present.md), because the answer depends on whether attachment
  extents stay build-validated.
- **Bindless resource access.** A geometry pass over many materials wants to index
  a table of textures from the fragment stage. Vulkan descriptor indexing, D3D12
  bindless and Metal argument buffers all provide it; GLES 3.1 has nothing
  comparable, and the CPU backend would need a matching model. A capability, if it
  arrives at all.
- **Per-draw scissor.** Section 3.4 excludes it. If a real caller needs
  atlas-style rendering in one pass rather than several, this is where it would
  go, and it would be capability-gated.
