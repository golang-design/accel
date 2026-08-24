# How accel is put together

What accel can do today, and why it is shaped the way it is.

If you only want to get something running, the [README](../README.md) is shorter
and has the code, and the [tutorials](tutorial/) take it one idea at a time. If you want the formal decision record — what was tried, what
was rejected, and why — that is [`specs/`](../specs/).

Start with [what you can run today](#what-you-can-run-today) if that is your
question; [what will bite you](#what-will-bite-you) is the three things about
memory that look like bugs the first time you meet them.

## The problem

Using a GPU from Go usually means one of two things. You bind to a C library and
accept cgo: slow builds, no cross-compilation, a toolchain on every machine that
builds your code. Or you write against one vendor's API and accept that your code
runs on one platform.

`accel` tries for a third option: one Go API, several backends underneath, and no
cgo anywhere. That last constraint is the interesting one, and it shapes
everything else.

## Two layers

```
   your inference code            your renderer
          |                             |
   +------v---------+                   |
   |  tensor layer  |                   |
   |  shapes, ops,  |                   |
   |  graphs        |                   |
   +------+---------+                   |
          |                             |
   +------v-----------------------------v------+
   |             device layer                  |
   |  buffers, kernels, command graphs         |
   +------+------+------+------+------+--------+
          |      |      |      |      |
        CPU   Metal  Vulkan  D3D12   GL
        ^^^^^^^^^^^  \____________________/
           built        designed, not yet built
```

Two of these five backends work today: the CPU backend and Metal. The other
three are designed in [`006`](../specs/006-backends.md), which keeps adding one a
device-layer job rather than a project, and none is scheduled for a first
release. Until a second GPU backend exists, the CPU backend is the only thing
that can catch a portability bug on hardware you do not have.

WebGPU is a sixth, later backend shape. Browser promises cannot be waited on by
the synchronous surface without deadlocking the event loop, so it is deferred
with an explicit asynchronous API rather than squeezed into this diagram.

The **device layer** is the foundation. It deals in buffers, textures,
workgroups, and barriers. If you are writing a renderer or a simulation, this is
your API.

The **tensor layer** sits on top and deals in dtypes, shapes, and operators. If
you are running a model, this is your API, and you should never need to think
about a bind group.

The boundary is strict: the tensor layer contains no backend-specific code at
all. Everything it does, it does by asking the device layer. That is what makes
adding a backend a contained job rather than a project.

## The idea that shapes everything: graphs, not command streams

Most GPU APIs work like this: begin a pass, set a pipeline, dispatch, end,
submit. Each call goes out immediately, and once submitted it is forgotten.

`accel` does something different. You **record** work into a `Graph`, and the
graph is a thing you keep:

```go
rec := dev.NewRecorder()
rec.Dispatch(pipeline, bindings, accel.WorkgroupCount{X: n})
g, err := rec.Build()

// submit it as many times as you like
for range tokens {
    g.Rebind(newInputs)
    dev.Queue().Submit(g).Wait()
}
```

`Build` is where the work happens. It validates everything, plans which
intermediate buffers can share memory, works out where the barriers go, and hands
the backend something it can replay cheaply.

### Why this matters

Consider running a language model. One transformer layer is roughly a hundred
operations, and a model has dozens of layers. Under an immediate-submission API,
generating each token means re-issuing thousands of commands, asking for every
intermediate allocation again, and giving the driver no chance to see that two
adjacent operations could have been one.

Recording changes that. You pay the analysis cost once, and every token after
that is a replay.

If you want to see this reasoning validated independently, look at
[ollama's `ml` package](https://github.com/ollama/ollama/blob/main/ml/backend.go).
Its `Context` has `Forward` to record, `Compute` to execute, and `Reserve` to
pre-plan memory. It arrived at the same shape from the same pressure.

### Does this hurt graphics?

No. A render pass is already recorded into a command
buffer on every backend, so nothing is lost. Vulkan has secondary command
buffers, D3D12 has bundles, Metal has indirect command buffers: the hardware
APIs already want you to record and replay.

Worth being plain about the state of it, though: **graphics is designed and not
built.** [`../specs/005-graphics.md`](../specs/005-graphics.md) is the drafted
parent design. It records the constraints already known and names four child
specs still required before implementation: the stage ABI, render API,
surfaces/present, and CPU reference rasterizer. v0 implements none of it.

### What it costs

Errors move. Under an immediate API, a bad call fails at the call. Under this
one, it fails at `Build`, possibly far from where you wrote it. That is a real
usability cost, and the design's answer is that a build error must name the node,
the binding slot, and the source position of the call that recorded it. If you
ever see an accel error that says only "type mismatch", that is a bug worth
filing.

## No cgo, anywhere

Every backend reaches its driver through
[purego](https://github.com/ebitengine/purego) or raw syscalls. There is no
`import "C"` in this module and there never will be; CI greps for it.

What you get: `CGO_ENABLED=0` binaries that still use the GPU, cross-compilation
that works, builds that are fast, and no toolchain to install.

What you give up, and this is not a small thing: the entire C machine-learning
ecosystem. No cuBLAS, no cuDNN, no GGML. Every kernel has to be written here, in
Go, and compiled to each backend's shading language. That is a lot of work and it
will not beat vendor libraries on raw throughput for a long time, possibly ever.

So the trade is explicit. If cross-compilation and build simplicity are worth
more to you than peak throughput, accel is a reasonable choice. If they are not,
existing cgo bindings will be faster today and are the better tool.

## Kernels are written in Go

You do not write MSL, or GLSL, or HLSL. You write a restricted subset of Go, and
it compiles to whichever of those the target device speaks.

The subtle part is that the same authored source becomes both targets through one
typed IR. The CPU backend runs generated Go instrumented with explicit rounding,
shared-memory initialization tracking, and barrier checks; the GPU runs emitted
shader code. When their results disagree, one lowering is wrong, and you have a
reproducible case rather than a mystery.

It is worth being precise about what that catches, because it is easy to
overclaim. Sharing the source and typed IR proves the *lowerings agree*: the compiler, the
bindings, the dispatch, the backend's conventions. It says nothing about whether
the kernel computes the right thing, since a wrong formula is wrong identically
everywhere and every parity check still passes. Checking the mathematics needs a
second implementation written independently, and accel's tests carry both.

## The CPU backend is not a fallback

It is a first-class backend, always available, and it does more work than you
would expect:

- It is the **cross-backend oracle**. Every GPU backend is verified against it,
  which catches lowering bugs (see the caveat above about what it does not catch).
- It makes the tensor layer **developable before any GPU backend is finished**.
- It runs in `go test ./...` on any machine, with no GPU and nothing to
  provision. That matters more than it sounds: getting GPU coverage in CI
  otherwise means Mesa llvmpipe, lavapipe, WARP, and hunting for ANGLE inside an
  installed browser, and that last one has already broken once when a CI image
  changed.
- It **catches bugs real hardware hides**. A kernel that reads shared memory
  before writing it gets a poison pattern rather than convenient zeroes. A
  barrier that some invocations reach and others do not is detected and reported,
  where a real GPU would quietly appear to work until it did not.

## Backends disagree, and we wrote it down

Every GPU backend has its own conventions, and where they differ, the differences
are vicious: they usually look like a mathematics bug rather than a convention
mismatch, so you go and check your matrices for an afternoon.

[`conventions.md`](conventions.md) is a table of every such divergence we know
of, what accel guarantees on top of it, and where the correction belongs. Some
highlights:

- Metal clips depth to `[0, 1]` while GL uses `[-1, 1]`, so a GL-convention
  projection silently drops **all** your near geometry. You do not get a distorted
  image, you get nothing.
- Reading a render target back gives you bottom-origin rows on both GL *and*
  Metal, despite Metal's texture origin being top-left. Compute buffers are not
  flipped. So a test that only covers the compute path passes while the texture
  path is mirrored.
- Metal's default face winding is the opposite of GL's, and getting it wrong
  keeps back faces. The silhouette still looks right, so it reads as a shading
  bug.

If you contribute a backend, that file is the contract.

## Three things the graph does that are worth knowing

The reasoning for each is in the spec it names.

**Edges come from declared access, not from record order.** A graph infers its
own dependency edges from what each node says it touches, comparing byte ranges
rather than whole resources — so two nodes writing disjoint halves of one buffer
are not serialized. Barriers come from those edges and are batched, because a
barrier is queue-wide: [spec 003's worked
graph](../specs/003-command-graph.md) has nine hazards and emits seven barriers,
and the test asserts their positions rather than only their count.

**Transient aliasing is reachability, not intervals.** Transients the builder
owns share memory when every node touching one is ordered before every node
touching the other. The difference from an interval-based planner is not
theoretical: intervals alias two transients on opposite arms of a diamond and
corrupt one of them on any backend that runs the arms at once. See
[017](../specs/017-graph-aliasing.md).

**The conservative plan is kept as an oracle.** The plan that ran before edges
were inferred is not deleted. Every random graph is built twice, once optimized
and once naively, and the results compared. It found three real bugs in minutes.
`Recorder.BuildNaive` exposes it for bisecting a suspected planning bug.

## Why cooperative kernels run on a scheduler

A kernel that needs its invocations to cooperate — shared memory, a barrier, a
reduction across a subgroup — is compiled to a resumable form and run by a
scheduler that advances every invocation to its next suspension point before
releasing the epoch.

The point of doing that on a CPU is not speed. It is that the schedule is
deterministic, so a kernel reading shared memory nothing wrote, or whose
invocations reach different barriers, is *reported* with a line number on the
first offending run — rather than producing a plausible number on one machine
and a different one elsewhere. See
[018](../specs/018-cooperative-lowering.md) and
[019](../specs/019-cooperative-diagnostics.md).

## What you can run today

Compute works end to end on two backends, the CPU backend and Metal: pooled
memory, buffers and typed views; kernels written in a subset of Go; recorded
command graphs with inferred barriers and transient aliasing; cooperative
kernels with shared memory and barriers; a portable tiled GEMM; and the tensor
layer above it, with quantized weights, sampling, a paged KV cache, and prefill
and decode attention. Subgroup shuffles and scans are specified and not built.

Graphics runs a frame. A render pipeline, a pass node with load and store
actions, vertex and index buffers, per-vertex and per-instance attributes,
by-value stage parameters, depth testing and blending all go through the graph
to the CPU reference rasterizer, and a headless surface rotates images through
acquire, render and present.

Metal runs the same passes, and the two are compared pixel by pixel — which
makes the CPU rasterizer an oracle for graphics the way it already was for
compute, rather than the only implementation.

There is a windowed path on Metal: you create the window, hand accel the
`CAMetalLayer`, and it owns the swapchain inward. accel does not create windows,
and the reason is a boundary rather than laziness — window creation is input,
focus, DPI and event loops, none of which is GPU work, and absorbing it would
drag a windowing library and an opinion about event loops into a library about
device work.

The frame loop is the same whether it presents to a screen or to a buffer you
read back, which is what the headless surface was built to make true.
Multisampling is specified and unbuilt, and so are the Vulkan, D3D12, OpenGL and
WebGPU backends.

The row-by-row breakdown is the [status table in the
README](../README.md#what-works-today). The order the work was done in, and the
deviations taken, are in
[`../specs/009-sequencing.md`](../specs/009-sequencing.md).

## What will bite you

Three properties of the memory model that are not bugs and will look like bugs
the first time you hit them.

**A pool is exactly one device allocation, and it never grows.** No backend can
resize one in place, and growing by reallocating would invalidate every address
already handed out — a device address is baked into descriptor sets and recorded
commands by the time anything runs. So size a pool for your peak. The one thing
that does grow is the implicit pool behind `Device.NewBuffer`, which grows the
only way a device allocation can: by adding another one.

**Nothing compacts, so fragmentation is permanent for a pool's life.** That is
what a non-compacting allocator is, rather than something to be fixed later.
`PoolStats` reports `LargestFree` beside `Free` so you can see an allocation
failure coming instead of hearing about it afterwards. The mitigation is
separating pools by lifetime class, which is why a pool takes a policy and a
label rather than accel trying to guess.

**Closing is ordered, not recursive.** A pool with live buffers refuses to
close, and so does a device with live pools. Both report and free nothing, and
the children keep working. Close children first. Closing a child out from under
a caller who still holds it would turn their bug into a silent success and make
the next use undefined instead of reported.

## Where to go next

- [`conventions.md`](conventions.md), the backend divergence table. Genuinely
  useful even if you never use accel.
- [`../specs/`](../specs/), the internal design specs, if you want the full
  reasoning and the open questions.
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md), if you want to help.
- [`../specs/009-sequencing.md`](../specs/009-sequencing.md), if you want to know
  what is being built next and what would count as done.
