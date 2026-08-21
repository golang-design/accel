# How accel is put together

A tour of the design, written for someone who wants to understand or contribute
to it. If you are looking for the formal decision record instead, that lives in
[`specs/`](../specs/).

Nothing here is implemented yet. This describes what is being built and why, so
that when you read the code it makes sense.

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
```

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

The subtle part is that the *same source* also runs as ordinary Go on the CPU
backend. When a GPU result disagrees with the CPU result, one of them is wrong,
and you have a reproducible case rather than a mystery.

It is worth being precise about what that catches, because it is easy to
overclaim. Sharing the source proves the *lowering* is right: the compiler, the
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

## Where to go next

- [`conventions.md`](conventions.md), the backend divergence table. Genuinely
  useful even if you never use accel.
- [`../specs/`](../specs/), the internal design specs, if you want the full
  reasoning and the open questions.
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md), if you want to help.
