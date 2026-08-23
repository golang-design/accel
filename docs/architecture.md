# How accel is put together

A tour of the design, written for someone who wants to understand or contribute
to it. If you are looking for the formal decision record instead, that lives in
[`specs/`](../specs/).

The compute half is built, on the CPU backend and on Metal: memory, command
graphs, cooperative kernels, the tensor layer, and the inference pieces above
it. The graphics half is designed and unbuilt, and so are the remaining
backends. Everything below describes what was built and why, so that when you
read the code it makes sense, and [what is built](#what-is-built-so-far) says
which parts you can run today, milestone by milestone.

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

Two of those five synchronous backends are what v0 builds. The others are
designed in [`../specs/006-backends.md`](../specs/006-backends.md) so that adding
one stays a device-layer job, and none is on the path to a first release. This is
also why the CPU backend below matters more than it looks: at v0 it is not one
oracle among several, it is the only thing standing between a kernel and a
portability bug that no available device can produce.

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

No, and that surprised us too. A render pass is already recorded into a command
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

This is the project's central bet. If cross-compilation and build simplicity are
worth more to you than peak throughput, it is a good bet. If they are not,
existing bindings are genuinely the better choice, and we would rather you knew
that early.

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

These moved here from the README, where they were answering a question a user
had not asked. They are the parts of the design a contributor most often needs
explained, and the reasoning for each is in the spec it names.

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

## What is built so far

| Milestone | State |
| --- | --- |
| M0, the cgo-free build gate | done |
| M1, memory on the CPU backend | done |
| M2, the minimum kernel compiler and flat CPU execution | done |
| M3, graph planning and flat submission | done, split into [015](../specs/015-graph-recording.md), [016](../specs/016-graph-execution.md), and [017](../specs/017-graph-aliasing.md) |
| M4, cooperative execution on the CPU | done, split into [018](../specs/018-cooperative-lowering.md), [019](../specs/019-cooperative-diagnostics.md), and [020](../specs/020-cooperative-atomics.md); subgroup shuffles and scans deferred |
| M5, the portable tiled GEMM | done: 000's second v0 proof obligation |
| M6, Metal | done, split into [021](../specs/021-metal-bringup.md), [022](../specs/022-msl-target.md), and [023](../specs/023-metal-graph.md); the encoder-barrier measurement and indirect command buffers stay behind [006](../specs/006-backends.md) §4.3's measurement |
| M7, tensor decode and prefill | done, split into [024](../specs/024-tensor-bringup.md), [025](../specs/025-tensor-operators.md), and [026](../specs/026-tensor-decode.md); this completes 000's v0 proof |
| M8, independently scoped work | five of seven: [027](../specs/027-quantization.md) quantization, [028](../specs/028-sampling.md) sampling, [029](../specs/029-plan-cache.md) prefill buckets and the plan cache, [030](../specs/030-paged-kv.md) paged KV and batching, [031](../specs/031-shared-transients.md) shared transients. Graphics is gated by 000; Vulkan is blocked on this machine |

M1 built the bottom of the device layer: enumeration and device open, the
capability and limit profiles, pooled memory with a two-level segregated fit
allocator, buffers and typed views, explicit lifetimes, and host-to-device
transfers, all on the CPU backend and all reachable through the public API.

Three things about it are worth knowing before reading the code.

**A pool is exactly one device allocation, and it never grows.** No backend can
resize one in place, and growing by reallocating and copying would invalidate
every address already handed out, since a device address is baked into
descriptor sets and recorded commands by the time anything runs. So the choice
was never between fixed and growable; it was between fixed and lying. The one
thing that does grow is the implicit pool behind `Device.NewBuffer`, which grows
the only way a device allocation can: by adding another one.

**Nothing compacts, and fragmentation is permanent for a pool's life.** That is
what a non-compacting allocator is rather than a bug to be fixed, so
`PoolStats` reports `LargestFree` beside `Free` to let a caller see the failure
coming instead of hearing about it afterwards. The mitigation is separating
pools by lifetime class, which is why a pool takes a policy and a label instead
of accel trying to guess.

**Closing is ordered, not recursive.** A pool with live buffers refuses to
close, and so does a device with live pools; both report and free nothing, and
the children keep working. Closing a child out from under a caller who still
holds it turns their bug into a silent success and makes the next use undefined
instead of reported.

M2's first child built the compiler pipeline end to end for one kernel, and two
things about it decide how the rest of the compiler reads.

**The authored function is not what runs.** The CPU backend executes a generated
lowering built from the same typed IR every GPU artifact comes from, with an
explicit rounding point at each arithmetic operation. That is what makes the CPU
backend an oracle rather than a second implementation: two implementations of
the same maths disagree in ways nobody can attribute, and one IR lowered twice
disagrees only where the hardware does. The cost is a bug class where a mistake
in IR construction is wrong identically everywhere, so the authored function is
still run, by a test that compares the two.

**Nothing resolves by name.** Intrinsics are matched on the identity `go/types`
resolved, including the receiver's type. The predecessor keyed its builtin table
by bare name, so a user function called `Dot` lowered to the GPU builtin: nothing
errors, and the kernel computes something else.

The milestone list, what done means for each, and the deviations taken so far
are in [`../specs/009-sequencing.md`](../specs/009-sequencing.md).

## Where to go next

- [`conventions.md`](conventions.md), the backend divergence table. Genuinely
  useful even if you never use accel.
- [`../specs/`](../specs/), the internal design specs, if you want the full
  reasoning and the open questions.
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md), if you want to help.
- [`../specs/009-sequencing.md`](../specs/009-sequencing.md), if you want to know
  what is being built next and what would count as done.
