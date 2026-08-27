---
title: "Graphics: render pipelines, passes, draws, and present"
status: in progress
layer: device
depends_on:
  - 001-device-resources.md
  - 003-command-graph.md
  - 004-kernel-authoring.md
---

# Graphics

The rasterization half of layer 1: how a render pipeline is described, how a
render pass becomes a node in the graph of [003](003-command-graph.md), how draws
are issued, how a rasterized G-buffer is handed to a compute pass without leaving
the device, and how a frame reaches a screen.

**Status: in progress — corrected 2026-08-27.** This spec said `drafted`, which
means nothing it specifies has shipped, and said it "does not freeze a public
graphics API before any implementation has tested that shape". Both were wrong:
the API is public and frozen — `render.go` (1103 lines), `surface.go` (637),
`vertexlayout.go`, `textureview.go` — both backends render and present, and they
are compared pixel by pixel. What the sentence described was the intent at the
time of writing, and it was never revised when the children landed.

[`000-decisions.md`](000-decisions.md)'s v0 milestone still says compute only,
and it is stale in the same way; see [STATUS.md](STATUS.md).

This spec fixes the architectural boundaries and the cross-component invariants.
Implementation was gated on tightly scoped child specs — **five, not the four
this table listed**, since [041](041-msaa.md) was added:

| | Child | Covers |
| --- | --- | --- |
| 1 | [032](032-stage-abi.md) | the vertex/fragment stage ABI and kernel lowering |
| 2 | [033](033-render-api.md) | render pipeline, pass, attachment, and draw APIs |
| 3 | [034](034-surface-present.md) | surfaces, acquisition, resize, and present |
| 4 | [035](035-cpu-rasterizer.md) | the CPU reference rasterizer and its conformance corpus |
| 5 | [041](041-msaa.md) | sample positions, resolve, and what the oracle can prove once a pattern exists |

**One thing this parent specifies has no owning child**, and it is the one the
worked example below is organised around: the render-to-compute handoff where
compute reads the attachments as textures with no copy. A texture in a compute
kernel is refused by name — `Texture2D` and `Fetch` reach vertex and fragment
stages only, and `Binding.Texture` is refused on every dispatch path — and
[032](032-stage-abi.md) §5.1 explicitly declines to own it. So variant 2, the
texture-to-buffer transfer node, is the only handoff that records today.

Each child may refine names and descriptor layout while preserving this parent's
decisions, and each does: render passes are graph nodes, attachment formats are
pipeline state, resource hazards are inferred, presentation remains external to
the graph, and the CPU path is the oracle.
[004](004-kernel-authoring.md) no longer defers the vertex and fragment
directives; 032 defines them.

**This spec's four open questions are closed**, each in the direction this
document itself argued, and each in the child that owns the consequence:
reverse-Z in [032](032-stage-abi.md) §2.5, vertex layout derivation in
[033](033-render-api.md) §5, pass merging in [033](033-render-api.md) §5, and
resize-versus-rebuild in [034](034-surface-present.md) §4.1. The former "Open
questions" section below is retained only for the entries that stay open.

The design target is a deferred renderer: a geometry pass writing several colour
attachments plus depth, then compute doing the shading. That target is chosen
because it is the case the predecessor project could not complete, and because it
exercises every part of this spec at once.

## What this spec inherits

- Textures, formats, pools, and transfers come from
  [001](001-device-resources.md). Texture-to-buffer and buffer-to-texture
  transfers are first class there, and this spec depends on that.
- Shader stages are authored in Go, per [`000-decisions.md`](000-decisions.md)
  decision 5, and compute passes follow [002](002-compute-model.md).
- Nothing here executes when it is called. A render pass is recorded, like every
  other node. [`000-decisions.md`](000-decisions.md) decision 1 already argues
  this costs graphics nothing: every backend records a render pass into a
  command buffer anyway, and GL, which has no command buffer at all, records and
  replays on its context thread regardless (see
  [`conventions.md`](../docs/conventions.md)).

## Render pipeline

A render pipeline is a compiled object. Creating one is expensive on every
backend (it compiles shaders and specialises fixed-function state), so it is
created once, outside the frame loop, and referenced by nodes.

### Vertex input

A pipeline declares its vertex input as a set of **buffer layouts**, each with a
stride and a step mode (per vertex or per instance), and within each a set of
**attributes** with a format, a byte offset, and a shader location.

Step mode being part of the layout rather than the draw is what makes instancing
free: an instance transform buffer is just a layout stepping per instance.

A pipeline may declare **no vertex buffers at all**, in which case the vertex
stage runs off its vertex index alone. This is not a curiosity: it is how a
full-screen pass is drawn (three vertices, positions computed from the index),
and the deferred renderer in this spec uses it for the resolve.

Attribute formats are a distinct enumeration from buffer dtypes, for the same
reason texture formats are in 001: a `unorm8x4` vertex colour and a `u8`
buffer element have the same width and different meanings.

### Shader stages

Two stages, vertex and fragment, both authored in Go. The kernel language and
its directives are [004](004-kernel-authoring.md)'s subject, which defers the
graphics stages and reserves `//accel:vertex` and `//accel:fragment` for them.
What this spec fixes is the shape those stages must have, since the pipeline is
what consumes them.

- The vertex stage returns a clip-space position plus a struct of **varyings**.
- The fragment stage takes the interpolated varyings and returns a struct whose
  fields map, in order, onto the pipeline's colour attachments. One struct field
  per attachment is how MRT is expressed, and it is what the predecessor proved
  on Metal.

Integer varyings must be flat-interpolated. This is not an accel choice, it is
true on every backend, and the kernel compiler rejects a perspective-interpolated
integer varying with an error naming the restriction rather than letting the
driver do it.

**The baseline fragment stage writes attachments, not storage resources.**
[`conventions.md`](../docs/conventions.md) records why: ordinary fragment buffer
writes are unordered with respect to other fragments covering the same pixel, so
a G-buffer written that way holds a nondeterministic winner wherever geometry
overlaps while the depth buffer looks correct. Render targets do not have this
problem because fixed-function depth/stencil/blend processing orders the
attachment operation.

Rasterizer-ordered access (ROA) is a later, capability-gated extension to the
stage ABI, not a loophole in the baseline. A pipeline using it must say so in its
generated stage requirements, every written storage binding is declared
read-write, and graph build requires the device capability. ROA orders covered
fragment shader storage accesses in primitive order; it does **not** make an
attachment simultaneously sampleable, does not turn an unordered storage write
into an attachment write, and is limited to sample count 1 until an MSAA child
spec defines per-sample ordering. The CPU rasterizer must emulate the same order
before it can report ROA support.

### Rasterizer state

- **Front-face winding**, stated explicitly, meaning the same thing on every
  backend. Metal's default disagrees with GL's, and getting it backwards keeps
  back faces instead of front faces: the silhouette stays right while every
  per-pixel attribute comes from the wrong surface, so it reads as a shading bug.
  See [`conventions.md`](../docs/conventions.md). There is no "backend default";
  the field has no unset state.
- **Cull mode**: none, front, or back.
- **Fill mode**: filled or wireframe, where wireframe is a capability (Metal has
  it, GLES 3.1 does not).
- **Depth bias**: constant, slope, and clamp. Present because shadow mapping is
  unusable without it and because emulating it in the vertex stage is wrong for
  sloped geometry.
- **Primitive topology**: triangle list, triangle strip, line list, line strip,
  point list.

### Depth and stencil state

Depth test enable, compare function, and depth write enable, as three separate
fields, because read-only depth (test on, write off) is a real and useful
configuration: it is how a second pass shades exactly the surfaces the geometry
pass kept.

Stencil is specified now rather than added later: compare function, read and
write masks, and the fail/depth-fail/pass operations, per face. It is fixed
function on all four target backends, and adding it later changes the shape of
the pipeline descriptor, which is a breaking change for every caller. The
stencil **reference value** is the one piece that is dynamic; see the fixed
versus dynamic table.

A pipeline declares the **format** of its depth or depth-stencil attachment, or
none. Depth formats carry backend constraints colour formats do not, including
macOS requiring depth textures to be device-private; 001 makes the backend
enforce that rather than the caller discovering it.

### Blend and colour attachments

Each colour attachment carries its own format, write mask, and blend state
(separate colour and alpha factors and operations, or blending disabled).
Per attachment rather than per pipeline, because a G-buffer's albedo target and
its accumulation target want different answers in the same pass.

The **blend constant** is dynamic; everything else about blending is compiled in.

Attachment count is capped by a device-reported limit. Requesting more attachments
than the device supports is a pipeline creation error naming the limit, per
[`000-decisions.md`](000-decisions.md) decision 6.

### Fixed at creation, and what is not

The second column is **not** 003's list of what varies between submissions. Most
of it is recorded values, which is to say graph structure, fixed once the graph is
built. Only bind group contents and buffer contents vary between submissions of a
built graph. The per-object section below makes the same distinction concretely.

| Fixed at pipeline creation | Recorded per pass or per draw, or bound |
| --- | --- |
| Vertex input layout and attributes | Vertex and index buffer bindings |
| Vertex and fragment stage code | Bind group contents (through 003's rebinding) |
| Topology class, winding, cull, fill, depth bias | Buffer contents |
| Depth test, compare, write, stencil ops and masks | Stencil reference value |
| Blend state and write mask, per attachment | Blend constant |
| Colour attachment formats and count, depth format | Viewport and scissor (per pass) |
| Sample count (fixed to 1 until the MSAA child spec) | Draw counts, first index, instance count |

The split is not a taste decision, it is what backends require. D3D12 compiles a
pipeline state object, Vulkan compiles a `VkPipeline`, Metal compiles a
`MTLRenderPipelineState`, and each of them takes attachment formats, blend, and
vertex layout as compile-time inputs. The dynamic column is close to the
intersection of what all four can change without recompiling. Exposing more as
dynamic would mean either a backend that silently recompiles mid-frame (a
notorious stutter source) or a backend that fails at a call the API said was
legal.

Attachment formats being compiled in has a consequence worth stating: a pipeline
is valid only in a pass whose attachments match its declared formats and count.
That agreement is checked at graph build, and the error names the pipeline, the
node, and the attachment index.

The drafted baseline has a sample count field so the eventual pipeline key has a
place for it, but the only legal value is 1. The MSAA child spec must define
multisampled texture creation, equal sample-count validation across all pass
attachments, single-sample resolve targets, load/store/resolve combinations,
per-sample shading, coverage masks, and CPU-oracle limits before any value above
1 is accepted. ROA and MSAA remain mutually exclusive until that spec defines a
portable ordering rule.

## Render passes as graph nodes

**One render pass is one node.** All the draws inside it are recorded into that
node. A draw is not a node.

This is the single most consequential structural decision in the spec, and the
reason is that the render pass, not the draw, is the unit at which
synchronisation is expressible. Vulkan cannot barrier inside a render pass in the
general case, tile-based hardware physically cannot (the attachment contents live
in tile memory until the pass ends), and Metal's encoder has the same shape. Draw
granularity in the graph would promise an ordering the hardware cannot provide.
Pass granularity promises exactly what it can.

Two rules follow, and both are caller-visible:

1. **Draws inside a pass execute in recorded order** and the builder never
   reorders them. Blending is order dependent, and so is any caller reasoning
   about overdraw.
2. **The builder inserts no barriers inside a pass.** Per-pixel ordering between
   draws is the ROP's job, which is exactly the guarantee
   [`conventions.md`](../docs/conventions.md) says fragment buffer writes lack.

### Load and store actions

Each attachment declares what happens at the start and the end of the pass.

| Load | Meaning |
| --- | --- |
| `Clear` | Clear to a stated value. |
| `Load` | Preserve existing contents. |
| `DontCare` | Contents are undefined at pass start. |

| Store | Meaning |
| --- | --- |
| `Store` | Contents are readable after the pass. |
| `DontCare` | Contents are undefined after the pass. |

These are not decoration. On a tiler, `Clear` costs nothing while a full-screen
clear draw costs a full write of tile memory, and `DontCare` on a depth
attachment that nothing reads after the pass saves writing the whole depth buffer
out to memory every frame. A depth attachment used only for occlusion within its
own pass should be `Clear` then `DontCare`, and stating it is the caller's, not
the backend's guess.

`DontCare` is also load-bearing for the graph: an attachment loaded `DontCare`
carries no data dependency on whatever wrote it last, so the read-after-write edge
disappears. The write-after-write edge does not: an earlier node's writes to the
same memory still have to be ordered before this pass's, or they land afterwards.
And where the attachment is a builder-owned transient, its live range starts at
the pass rather than earlier, so it can alias unrelated transients.
Caller-created textures are never aliased, per 003.

Clear values come from the attachment's format. Clearing depth to 0 by leaving a
field zero-valued is a real hazard (the predecessor defaulted zero to 1.0 for
exactly this reason), so a `Clear` load action carries an explicit value with no
zero-value special case, and the graph builder rejects a depth clear outside the
stored depth range.

That range is `[0, 1]`, and it is a different thing from the clip convention.
`[-1, 1]` is clip space and NDC. Every target backend applies the viewport depth
transform to a **stored window depth in `[0, 1]`**, which is why
[`conventions.md`](../docs/conventions.md)'s fragment-side recovery
`position.z * 2 - 1` type-checks on GL and Metal alike. Depth clear values and
depth compares operate in `[0, 1]`; vertex kernels emit `[-1, 1]`.

### Declared access, inferred edges

A render pass node declares:

- each colour attachment, with its load and store action, so an attachment loaded
  `Load` is a read plus a write while one loaded `Clear` or `DontCare` is a write
  only;
- the depth-stencil attachment, which is a read plus a write when depth writes are
  on and a read only when the pipeline tests without writing;
- the union of every resource bound by every draw in the pass, as reads: vertex
  buffers, index buffers, uniform and storage buffers, sampled textures, and any
  indirect argument buffer;
- its render area, which must be within every attachment's extent.

From that the builder infers edges and computes barriers exactly as it does for a
dispatch, which is the point of 003 declaring access rather than order. A geometry
pass writing a G-buffer and a compute pass reading it produce an edge with no
caller involvement, and the barrier that makes the render target readable by
compute is computed, not written.

**Shader feedback and attachment access are different categories.** The fixed
function operations implied by `Load`, blending, depth test, stencil test, and
their writes are legal attachment read-modify-write operations; describing them
as a pass reading a resource it writes would incorrectly reject every blended or
depth-tested pass.

The forbidden case is an overlapping texture subresource that is both an
attachment and shader-visible in the same pass. Sampling or storage-accessing the
mip/layer range currently bound as a colour or depth attachment is a build error
naming both views, regardless of load action and regardless of ROA. Disjoint
mips or array layers are different subresources and are legal when the backend
can bind those views independently. The child render-API spec must make view
ranges part of pass validation; comparing only texture handles is insufficient.

A depth attachment with writes disabled, used only by the fixed-function depth
test, is a read-only attachment and is legal everywhere. Binding that same depth
subresource for sampling in the fragment stage is still forbidden feedback.
ROA applies only to shader storage accesses and does not relax either attachment
rule. Within the ROA storage set, read/write overlap is intentional and ordered;
without ROA, fragment storage writes are absent from the baseline API.

## Draw commands

Recorded into a pass node:

- **Draw**: vertex count, instance count, first vertex, first instance.
- **DrawIndexed**: index count, instance count, first index, vertex offset, first
  instance. The index buffer and its index type (16 or 32 bit) are bound per
  draw.
- **Indirect** forms of both, reading their arguments from a device-written
  buffer.

Instancing is not a separate call, it is the instance count, which is why the
non-instanced case is the instanced case with a count of one. A caller never
picks between two entry points for the same drawing.

### Per-object data without re-recording

A scene of a thousand objects, replayed every frame with new transforms, is the
case that decides whether this design actually works under 003's immutability. It
works like this:

Object `i`'s draw is recorded with a **fixed byte offset** `i * stride` into a
uniform buffer. The offsets are structure, so they are baked into the graph. The
transforms are contents, so they are rewritten every frame through 003's first
kind of variation. Nothing is re-recorded, and the graph replays.

This is why **there are no push constants in the initial graphics API**. Push
constants (Vulkan push constants, D3D12 root constants, Metal `setVertexBytes`)
put per-draw values into the command stream itself, which makes changing them a
re-record. That is precisely the cost decision 1 exists to avoid, and offering
them would make the convenient path the one that defeats the model. The
recorded-offset mechanism has
a native expression everywhere (dynamic uniform buffer offsets in Vulkan, buffer
offsets in Metal, root descriptor offsets in D3D12).

The honest cost: **the object count is baked in**. A scene that gains an object
needs either a rebuilt graph or a graph recorded for a fixed maximum, with absent
objects issuing zero-instance draws. Zero-instance draws are cheap but not free.
This is 003's "cache graphs keyed by shape" applied to scenes, and a renderer with
genuinely churning object counts will feel it.

### Indirect draws, and the count problem

Indirect **arguments** (vertex count, instance count, offsets) written by a
compute pass fit the model with no tension at all. The graph structure is
unchanged, only numbers vary, and that is 003's third kind of variation, which
already covers dispatch counts. **This spec proposes an amendment to 003**:
its third kind of variation reads "dispatch counts" and should read "dispatch and
draw counts". Someone reading 003 alone will not find draw counts there.
The argument buffer is declared as an indirect read, which is its own access kind
(the barrier from compute-write to indirect-read is a distinct stage on Vulkan and
D3D12 and getting it wrong is a classic hang).

Indirect **draw count**, where the device decides how many draws happen, is
genuinely in tension with immutability, since the number of commands is no longer
a property of the graph. The resolution: a multi-draw-indirect node records a
**build-time maximum draw count** and reads the actual count from a device-written
buffer, clamped to that maximum. The graph's structure stays bounded and
plannable, validation still has numbers to check, and the device still decides.
A count exceeding the maximum is clamped rather than silently truncating, and the
clamp is reported through [003](003-command-graph.md)'s run-time statistics.
Reporting is opt-in there, because reading a device-written count back costs a
transfer and a `Readback` allocation: a graph that did not ask still clamps, and
does so silently, which makes the maximum a caller obligation in release mode.

This is a capability, not a guarantee. D3D12 `ExecuteIndirect` and Vulkan
`vkCmdDrawIndirectCount` provide it directly, Metal has indirect command buffers,
and GLES 3.1 has single indirect draws but no count buffer. Absence is reported,
per decision 6, and a caller without the capability falls back to a fixed draw
count with zero-instance draws for the unused slots.

003 lists conditional and iterative execution as an open question. This spec does
not resolve it. It resolves only the bounded case, which is the one GPU-driven
culling actually needs.

## The render-to-compute handoff

This is the worked example, and it is the case the predecessor could not do: it
had no on-device texture-to-buffer transfer, so a G-buffer went out to the host
and back every frame.

The graph, in full:

```mermaid
flowchart TD
    UP["<b>upload transforms</b><br/>transfer node: Upload pool to Device buffer"]
    GEO["<b>geometry pass</b> (render pass node)<br/>0 albedo RGBA8Unorm, Clear to Store<br/>1 normal RGBA16Float, Clear to Store<br/>2 worldpos RGBA32Float, Clear to Store<br/>D depth Depth32Float, Clear to DontCare<br/>N draws, each at its recorded uniform offset"]
    LIT["<b>lighting pass</b> (compute dispatch, 002)<br/>reads albedo, normal, worldpos as textures<br/>reads lights as a storage buffer<br/>writes hdr, a storage texture"]
    TONE["<b>tonemap pass</b><br/>reads hdr<br/>writes the swapchain image in its graph slot"]
    UP --> GEO
    GEO -- "edge inferred from declared access:<br/>attachment write, then texture read" --> LIT
    LIT --> TONE
```

Nothing in that graph touches the host between the upload and the present. The
edges are inferred from the declared access, the barriers that make attachments
readable by compute are computed, and the depth attachment's `DontCare` store
tells the builder nothing downstream needs it, so on a tiler it never leaves tile
memory.

Two variants of the handoff exist, and both are supported:

1. **Compute reads the attachments as textures.** The normal path, and the one
   the diagram shows. No copy at all.
2. **A transfer node copies an attachment into a buffer.** For a consumer that
   wants linear memory, which includes anything in layer 2. This is 001's
   first-class texture-to-buffer transfer, recorded as a node, participating in
   dependency tracking like everything else. It is a device-to-device copy: still
   no host round trip.

Variant 2 is also the **only** way to read a depth attachment back on macOS,
where depth textures must be device-private. A private texture cannot be mapped
by the host but can be copied on device, so depth readback is spelled as a
transfer node plus a buffer read, never as a direct texture readback. The API
does not offer the direct form for depth, so the macOS constraint cannot be
discovered by hitting it.

### One pixel origin, in all three places

A caller names a pixel of a render target in three different places, and they
must agree:

1. the fragment stage's pixel coordinate,
2. a compute kernel's texel index or sample coordinate on that texture, entirely
   on device, with no host involved,
3. host readback of that texture.

```mermaid
flowchart LR
    F["1. fragment stage<br/>writes at pixel (x, y)"]
    T[("render target texture")]
    C["2. compute kernel<br/>reads texel (x, y), on device"]
    H["3. host readback<br/>byte (y*w + x) * bpp"]
    F --> T
    T --> C
    T --> H
```

**Guarantee: row 0 is the top row in all three.** The correction is the backend's
responsibility and its mechanism is the backend's choice (transforming the
coordinate in emitted shader code, or storing top-origin through a flipped
viewport and skipping the readback flip). This spec fixes the observable, not the
implementation, exactly as [`conventions.md`](../docs/conventions.md) does.

Point 2 is the one this spec has to add. `conventions.md` guarantees the readback
path, and records that a compute kernel reading a G-buffer from a storage buffer
never sees the flip while the same data read from a texture does. The handoff
above reads a render target from a compute kernel on device, which is precisely
the combination neither guarantee covered, and it is where the predecessor's bug
would reappear in a document that claims to have learned from it.

## Surfaces and present

A **surface** is a swapchain: a rotation of presentable textures, acquired one per
frame, rendered into, and presented.

### The acquired image is a typed present slot

A graph cannot name a swapchain texture at record time, because which texture the
frame gets is decided at acquire time. An ordinary attachment slot is not enough:
`BindingAttachment` plus a format cannot prove that the eventual texture is
presentable, belongs to the right surface, or is from the surface generation for
which the graph was built. The surface child spec therefore adds a dedicated
present-slot constructor:

Frame loop:

```go
// Recorded once, when the frame graph is built:
swap := rec.PresentSlot(surface, "swapchain")

// Per frame, for the life of the window:
frame, err := surface.Acquire(timeout) // texture plus an acquire fence
graph.BindPresent(swap, frame)
fence := queue.SubmitAfter(graph, frame.Acquired)
surface.Present(frame, fence)
```

Internally the slot records the device, surface identity, surface generation,
format, extent, render-target usage, and the fact that its final state is
`Present`. `BindPresent` accepts a `Frame`, not a naked texture, and verifies the
same surface and generation. This makes the present transition representable to
the graph planner and prevents an ordinary texture with the same format from
being mistaken for a swapchain image. Resize increments the generation, so old
graphs and frames fail binding with an explicit stale-surface error.

This is [003](003-command-graph.md)'s second kind of variation with a stronger
slot type. The graph is built once and replayed until resize or surface
reconfiguration changes that type-level contract.

`Acquire` can block (the swapchain is full, the compositor has not released an
image) so it takes a timeout and can report expiry, rather than blocking forever
inside a call the API described as non-blocking.

### Present in the graph and in fences

Present is **not** a graph node. It is a queue operation taking the submission
fence it must follow.

The reason is that present is not work on the device, it is a handoff to a
compositor whose completion is not the device's to signal. Acquisition and
presentation are paired to one external `Frame`, while a graph describes only
device work. Keeping present outside preserves that boundary; the ordering that
does matter (rendering finishes before the image is shown) is expressed by the
submission fence.

What the graph does contain is the write to the bound swapchain image. The
builder knows that image is presentable, so it inserts the transition to
present-ready as the pass's store, which is the part that genuinely is device
work.

### Headless surfaces

A surface with no window rotates ordinary offscreen textures and "presents" by
making the frame's pixels available for readback. The frame loop code is
identical. This is the predecessor's design and it earned its keep: it is what
lets the entire frame path, acquire, render, present, rotate, resize, run in CI
on every platform with no display and no compositor.

### Resize

`Resize` reallocates the swapchain. It also invalidates any transient sized to
the old extent, and attachment extents are validated at build, so **a resize
means rebuilding the graphs that render into the surface**. Stated plainly
because it is a cost: it is cheap per resize event and would be unacceptable per
frame.

A backend may also report the swapchain as out of date on acquire (the window
changed underneath it). That is an explicit error value, not a silent
reallocation, because the caller has to rebuild graphs either way and hiding it
would leave stale graphs pointing at freed textures.

### Windowing is out of scope, and where the line is

**accel does not create windows.** The caller supplies a native handle and accel
owns everything from the swapchain inward.

| Caller owns | accel owns |
| --- | --- |
| The OS window and its lifetime | The swapchain and its textures |
| The event loop, input, DPI changes | `EGLSurface`, `CAMetalLayer` drawables, DXGI swapchain |
| Choosing a window visual (accel reports the constraint) | Present, acquire, resize |

Window creation is an operating system and event loop concern with no relation to
GPU work: input, focus, DPI, menu bars, and main-thread affinity. Absorbing it
would drag a windowing library, its own cross-platform test matrix, and an
opinion about event loops into a library whose subject is device work, and it
would do so in a codebase that cannot use cgo, where every windowing backend is
hand-written syscall or `purego` binding.

The boundary needs traffic in both directions, and the predecessor proved this
concretely: its `WindowVisualID` reports the native visual an X11 window must be
created with for the EGL config to be compatible. So accel reports its
constraints (required visual or pixel format, supported present modes, supported
extents) and accepts a platform-tagged native handle: `Display*` plus window XID
on X11, `wl_display` plus `wl_surface` on Wayland, `HWND` on Windows,
`CAMetalLayer*` or `NSView*` on macOS.

**Caller obligation on macOS**: a `CAMetalLayer` must be created and resized on
the main thread. accel accepts a layer the caller created and attached, and
documents the requirement. Given an `NSView*` it will create the layer, and then
the call itself must be made on the main thread, and says so.

**The honest cost**: accel cannot ship a runnable windowed example without either
a second package or a test-only window shim. That is not hypothetical, it is
exactly why the predecessor needed an `app/` package. The initial graphics answer
is a test-only shim, small and unexported, sufficient for the present tests
described below and explicitly not a windowing library.

**The known gap to close.** The predecessor implemented on-screen present for
X11/EGL and Win32/ANGLE, and never implemented Metal's `CAMetalLayer` drawable
path, so present worked on GL and not on Metal. That is a warning, not a detail:
Metal is the backend where present differs most from the others (the drawable is
owned by the layer, `nextDrawable` can block or return nothing, and the
completion-handler lifetime rule in
[`conventions.md`](../docs/conventions.md) applies directly). The first
presentation milestone must prove the Metal drawable path or present is not
portable, and it should be proven early rather than left as the last brick again.

## Convention guarantees

Restating the ones this spec depends on, in one place, with the detail and the
reasoning in [`conventions.md`](../docs/conventions.md):

- **Clip space.** One convention, presented to kernels: `z` in `[-1, 1]`. The
  backend remaps for Metal, Vulkan, and D3D12, whose native NDC range is
  `[0, 1]`. A caller never adjusts a projection matrix for the backend, and never
  sees near geometry vanish for the reason `conventions.md` describes. This is a
  statement about clip space and NDC only: the depth attachment stores `[0, 1]`
  on every backend, and clears and compares are in that range.
- **Winding.** Explicit in the pipeline descriptor, meaning the same thing
  everywhere, with no unset state.
- **Pixel origin.** Row 0 is the top row in the fragment stage, in an on-device
  compute read, and in host readback. See the handoff section above, which
  extends the readback guarantee to the on-device case.
- **Depth textures are device-private** where the backend requires it, so depth
  readback goes through a transfer node.
- **Bytes per pixel comes from the format**, always. Attachment size validation,
  transfer size computation, and readback buffer sizing all derive it, and none
  of them assumes 4.

## Out of scope for the initial graphics implementation

Each of these is excluded for a reason, not by omission.

- **Ray tracing.** The abstraction is not obvious (acceleration structure build
  and update, shader binding tables, an entirely separate pipeline type), the
  hardware support is a strict subset of target devices, and there is no CPU
  oracle short of writing a path tracer. Excluding it costs nothing structurally:
  it is an additional pipeline type, not a change to this one.
- **Mesh and task shaders.** They replace the vertex input and primitive assembly
  this spec specifies, so they are a second pipeline model rather than an
  extension. Neither GL nor the CPU backend has any equivalent, so nothing could
  be verified against the oracle.
- **Tessellation.** Metal implements it through a compute pre-pass with a
  genuinely different authoring model, GLES 3.1 lacks it, and the workloads that
  want it are better served by compute today.
- **Geometry shaders.** Metal does not have them at all, they are slow where they
  do exist, and every use has a compute or instancing formulation.
- **Multisampling.** Deliberately deferred rather than designed out. The baseline
  accepts sample count 1 only and does not reserve a partially specified resolve
  action. A child spec must add resolve attachments and actions together with the
  validation and per-sample semantics listed above. The reason is the CPU oracle:
  an MSAA reference rasterizer has to match a sample pattern that is not
  standardized across vendors, so the oracle would be weaker for the paths MSAA
  touches while the API surface grew. ROA remains sample-count-1-only until then.
- **Queries** (occlusion, pipeline statistics, timestamps). Useful, and a
  profiling concern rather than a rendering one.
- **Sparse and virtual texture memory**, consistent with 001.

## Open questions

Four questions that stood here are closed, each in the child spec that owns the
consequence, and each in the direction this document argued: reverse-Z needs no
API change ([032](032-stage-abi.md) §2.5), the vertex layout stays a descriptor
whose formats the compiler now validates ([033](033-render-api.md) §5), pass
merging is not attempted while the only handoff in the corpus merges on no
backend ([033](033-render-api.md) §5), and a resize rebuilds because attachment
extents are what build-time size validation is made of
([034](034-surface-present.md) §4.1). They are removed rather than kept, per
[`README.md`](README.md)'s rule.

What stays open:

- **Bindless resource access.** A geometry pass over many materials wants to index
  a table of textures from the fragment stage. Vulkan descriptor indexing, D3D12
  bindless, and Metal argument buffers all provide it; GLES 3.1 has nothing
  comparable, and the CPU backend would need a matching model. A capability, if it
  arrives at all. Carried in [033](033-render-api.md) §8.
- **Multisampling**, which is out of scope below rather than undecided, but whose
  child spec does not exist. The obligations it must discharge are listed in the
  render-pipeline section.
- Each child spec carries its own, and they are the ones a reader should check
  first: [032](032-stage-abi.md) §11, [033](033-render-api.md) §8,
  [034](034-surface-present.md) §9, [035](035-cpu-rasterizer.md) §10.

## Testing

### The CPU backend rasterizes

**Decision: yes.** The CPU backend implements the full graphics path, not just
compute.

The case is not aesthetic. [`000-decisions.md`](000-decisions.md) decision 3
makes the CPU backend the correctness oracle, and
[`conventions.md`](../docs/conventions.md) testing rule 1 says every convention in
that document is a way for a GPU backend to disagree with it. Half of those
conventions are graphics conventions: clip-space depth range, face winding,
readback origin, depth texture storage. A CPU backend that does not rasterize
leaves exactly the entries that cost the predecessor hours of debugging with no
oracle at all, and leaves every graphics test needing a device, which is the
provisioning problem decision 3 exists to end.

The cost is real and known rather than speculative: a conformant reference
rasterizer means clipping, a fill rule, perspective-correct interpolation, depth
and stencil test, per-attachment blend, and MRT. The predecessor project is a
software rasterizer, so the size of the job is measured, not guessed.

Scope, stated so it is not mistaken for a performance path: the CPU rasterizer is
a **reference** implementation. Triangles, lines, and points; top-left fill rule;
perspective-correct interpolation; depth and stencil; blending; multiple render
targets; no multisampling. It is not expected to be fast and no optimisation work
is planned for it.

**The fill rule is an accel guarantee**, not an implementation detail. Without a
stated rule, two triangles sharing an edge can double-shade or leave a gap, and
any coverage comparison against the oracle is meaningless. accel guarantees
top-left. Honest limit: on-edge sample tie-breaking is not identical across GPU
vendors even with the rule stated, so shared-edge pixels remain a tolerance
comparison.

### What the oracle proves exactly, and what within tolerance

This distinction is what makes the CPU rasterizer credible, and conflating the
two would produce a test suite that either fails constantly or proves nothing.

**Exact**, on every backend and on the non-degenerate domains stated by each
test:

- occlusion ordering when competing depths are separated by more than both
  pipelines' allowed interpolation/rounding intervals,
- attachment routing (which fragment output lands in which attachment),
- winding and cull behaviour,
- clip-range survival (which geometry is clipped),
- pixel origin agreement across the three places named above,
- graph structure (which nodes exist, which edges were inferred).

**Within a stated tolerance**:

- interpolated attribute values and shaded colours, since interpolation rounding
  differs between implementations,
- coverage at shared and near-degenerate edges, per the fill-rule limit above.

No portable winner is asserted when two depth intervals overlap: depth testing
chooses a discrete surface, so an ordinary numeric tolerance cannot repair a
different winner. Occlusion fixtures deliberately use separated depths and
report the interpolated depth intervals when they fail.

### Tests

- **Triangle.** A clip-space triangle renders to an offscreen target and the
  interior pixels match. The smallest end-to-end proof, and the predecessor's
  first milestone.
- **Occlusion is draw-order independent.** Two overlapping triangles, one near and
  one far, drawn in both orders into a colour plus depth target: the near one wins
  the overlap both times. This fails without a working depth attachment, which is
  what makes it worth writing.
- **MRT attachments are distinct.** A fragment stage writing a different constant
  to each of three attachments, read back separately, each holding its own value.
  Catches aliased attachments, which a single-target test cannot.
- **Winding flip empties coverage.** The same geometry with the front face
  reversed and back-face culling on produces no coverage. This is the test that
  catches the Metal winding divergence, which otherwise presents as a shading bug.
- **Near-plane survival.** Geometry straddling the near plane keeps its near half
  on every backend. This is the exact symptom `conventions.md` describes: a
  `[-1, 1]` matrix on a `[0, 1]` backend produces no coverage at all for near
  geometry, and it reads like a broken transform.
- **Origin agreement, the discriminating form.** Render a G-buffer whose written
  value encodes row position (top half distinct from bottom half). Path A: a
  compute kernel reads texel `(x, 0)`, writes it to a storage buffer, and the host
  reads the buffer. Path B: the host reads the texture at row 0. Assert `A == B`
  and assert both hold the top-row value. On GL and Metal exactly one of these
  needs a correction, and `conventions.md` testing rule 2 records that the
  predecessor's compute-path test passed while the texture path was mirrored.
- **Handoff stays on device.** The deferred graph above is built and its nodes
  inspected: no host transfer node exists between the geometry pass and the
  tonemap. Structural, and it is the regression test for the predecessor's actual
  failure. It is not sufficient on its own, which is why the origin test above
  checks values: a handoff with zero host transfers that reads the wrong rows is
  still wrong.
- **Depth readback on a private-depth backend.** Reading depth through a transfer
  node produces the expected values on macOS, where the direct path is impossible.
- **Load and store actions are observed.** `Load` preserves prior contents,
  `Clear` does not, and `DontCare` is not asserted about (it is undefined, so the
  test asserts the graph aliased the memory, not what is in it).
- **Per-object replay.** A graph of N objects at recorded uniform offsets,
  submitted twice with different transform contents, produces two different and
  individually correct frames with no re-recording.
- **Indirect draw.** A compute pass writes draw arguments, the draw consumes them,
  and the result matches the equivalent direct draw. On a device with the count
  capability, a device-written count below the build-time maximum draws that many,
  and a count above it clamps and reports.
- **Feedback validation.** Sampling the same mip/layer range used as an
  attachment is rejected and names both views; disjoint subresources are
  accepted. Attachment `Load`, blending, depth, and stencil read-modify-write are
  accepted. A capability-present ROA storage test is ordered; the same stage is
  rejected without ROA, and ROA never permits attachment sampling feedback.
- **Present-slot identity.** A frame from another surface, an earlier resize
  generation, or an ordinary render-target texture with the same format is
  rejected by `BindPresent`; a frame acquired from the matching surface binds and
  reaches the present state.
- **Headless surface frame loop.** Acquire, render, present, and rotate for
  several frames, with double buffering and a resize in the middle, verified by
  readback. Runs everywhere with no display, on every backend.
- **On-screen present per platform.** X11/EGL and Win32/ANGLE are verifiable in CI
  headlessly, which the predecessor demonstrated. Metal's drawable path needs a
  real display session, so it is an honest, separately tracked state:
  "headless render verified" and "windowed present verified on a display" are not
  the same claim and are not reported as one.
- **Determinism.** The same graph submitted twice on the same backend produces
  identical images, per 003.

When a comparison does fail, the diagnostic order from
[`conventions.md`](../docs/conventions.md) applies: coverage counts and overlap
between competing interpretations first, mathematics last. Equal pixel counts with
roughly half overlap is the flip fingerprint, and it identifies an origin bug in
one measurement.

## Amendment: the worked G-buffer is not portable as written

[001](001-device-resources.md) caught that this spec's deferred example uses
`RGBA32Float` as a colour attachment, and colour-renderable 32-bit float is a
capability on GLES 3.1 rather than a guarantee (it needs an extension there). So
the example as written is not guaranteed to run on the GL backend, which is the
CI oracle for graphics.

The example stands, because it illustrates the handoff and not a portable
G-buffer layout, but it needs the caveat and the workaround beside it:
reconstruct world position from depth rather than storing it, which drops the
widest attachment and is what a production deferred renderer does anyway for
bandwidth reasons.

The general rule this is an instance of: an attachment format has to be checked
against `Device.FormatInfo` rather than assumed, and a spec example that assumes
one is a portability claim it has not earned.
